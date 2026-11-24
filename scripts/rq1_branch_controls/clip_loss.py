"""Numerical implementation used by the reported experiment."""
from Losses.loss import loss as OriginalLoss
from Models.gmmformerV2.model_components import clip_nce
from torch import nn
from Losses.loss import query_diverse_loss


class VariantLoss(nn.Module):
    get_clip_triplet_loss=OriginalLoss.get_clip_triplet_loss

    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.first='clip1' if cfg['variant']=='DualClip' else 'clip'
        self.clip_nce_criterion=clip_nce(reduction='mean')
        if cfg['variant']=='DualClip':self.second_nce_criterion=clip_nce(reduction='mean')
        self.qdl=query_diverse_loss(cfg)
        self.last={}

    def forward(self,out,batch):
        labels=batch['text_labels'];groups=out['label_dict']
        terms={}
        terms[self.first+'_nce']=self.cfg['loss_factor'][0]*self.clip_nce_criterion(labels,groups,out['raw_scores'][self.first])
        terms[self.first+'_triplet']=self.get_clip_triplet_loss(out['scores'][self.first],labels)
        if self.cfg['variant']=='DualClip':
            terms['clip2_nce']=self.cfg['loss_factor'][1]*self.second_nce_criterion(labels,groups,out['raw_scores']['clip2'])
            terms['clip2_triplet']=self.get_clip_triplet_loss(out['scores']['clip2'],labels)
        terms['qdl']=self.cfg['loss_factor'][2]*self.qdl(out['query'],groups)
        terms['ot']=self.cfg['loss_factor'][3]*out['ot']
        total=sum(terms.values())
        self.last={k:float(v.detach()) for k,v in terms.items()}
        return total
