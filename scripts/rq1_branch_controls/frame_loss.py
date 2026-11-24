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
        self.first = clip_nce(reduction='mean'); self.second = clip_nce(reduction='mean')
        self.qdl = query_diverse_loss(cfg); self.last = {}

    def forward(self, out, batch):
        a, b = self.cfg['video_branch_names']; groups = out['label_dict']; labels = batch['text_labels']
        terms = {}
        terms[a+'_nce'] = self.cfg['loss_factor'][0]*self.first(labels, groups, out['raw_scores'][a])
        terms[a+'_triplet'] = self.get_clip_triplet_loss(out['scores'][a], labels)
        terms[b+'_nce'] = self.cfg['loss_factor'][1]*self.second(labels, groups, out['raw_scores'][b])
        terms[b+'_triplet'] = self.get_clip_triplet_loss(out['scores'][b], labels)
        terms['qdl'] = self.cfg['loss_factor'][2]*self.qdl(out['query'], groups)
        terms['ot'] = self.cfg['loss_factor'][3]*out['ot']
        self.last = {key: float(value.detach()) for key, value in terms.items()}
        return sum(terms.values())
