"""Numerical implementation used by the reported experiment."""
from Losses.loss import loss as OriginalLoss
from Models.gmmformerV2.model_components import clip_nce
from torch import nn
from Losses.loss import query_diverse_loss


class VariantLoss(nn.Module):
    get_clip_triplet_loss = OriginalLoss.get_clip_triplet_loss

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nce = nn.ModuleList([clip_nce(reduction='mean') for _ in range(cfg['num_views'])])
        self.qdl = query_diverse_loss(cfg)
        self.last = {}

    def forward(self, out, batch):
        labels, groups, terms = batch['text_labels'], out['label_dict'], {}
        for i, criterion in enumerate(self.nce):
            name = f'clip{i+1}'
            factor = self.cfg['loss_factor'][0 if i == 0 else 1]
            terms[name+'_nce'] = factor*criterion(labels, groups, out['raw_scores'][name])
            terms[name+'_triplet'] = self.get_clip_triplet_loss(out['scores'][name], labels)
        terms['qdl'] = self.cfg['loss_factor'][2]*self.qdl(out['query'], groups)
        terms['ot'] = self.cfg['loss_factor'][3]*out['ot']
        total = sum(terms.values())
        self.last = {key: float(value.detach()) for key, value in terms.items()}
        return total
