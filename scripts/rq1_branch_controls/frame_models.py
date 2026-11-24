"""Numerical implementation used by the reported experiment."""
from torch.nn import functional as F
from Models.gmmformerV2.model import GMMFormerV2_Net as Original
import torch


def pool_valid_windows(encoded, mask):
    length = encoded.shape[1]
    assert length == 128 and mask.shape == encoded.shape[:2]
    # The exact original average_to_fixed_length rule, including the final
    # clamp to length-1 (the final window is [124:127], not [124:128]).
    indices = torch.minimum(torch.round(torch.arange(33)/32*length).long(), torch.tensor(length-1)).tolist()
    values, valid = [], []
    for start, end in zip(indices[:-1], indices[1:]):
        a, b = int(start), int(end)
        if a == b:
            b = a+1
        weights = mask[:, a:b]
        count = weights.sum(dim=1)
        values.append((encoded[:, a:b]*weights.unsqueeze(-1)).sum(dim=1)/count.clamp_min(1).unsqueeze(-1))
        valid.append(count > 0)
    return torch.stack(values, dim=1), torch.stack(valid, dim=1)


def masked_scores(query, context, valid, unnormalized=False):
    if not unnormalized:
        query, context = F.normalize(query, dim=-1), F.normalize(context, dim=-1)
    scores = torch.matmul(context, query.t()).permute(2, 1, 0)
    scores = scores.masked_fill(~valid.t().unsqueeze(0), -torch.inf)
    return scores.max(dim=1)[0]


class FrameNet(Original):
    branch_names = ('clip', 'frame')

    def encode_context(self, clips, frames, mask=None):
        if not self.sparse_frame_enabled:
            encoded, clip = Original.encode_context(self, clips, frames, mask)
            if not self.fine_pool_enabled:
                return encoded, clip
            padded_mask = F.pad(mask, (0, 128-mask.shape[1]))
            return pool_valid_windows(encoded, padded_mask), clip
        assert frames.shape[1] <= 32
        clip = self.encode_input(clips, None, self.clip_input_proj, self.clip_encoder, self.clip_pos_embed, self.weight_token)
        fix = 32-frames.shape[1]
        if fix:
            frames = torch.cat([frames, 0.*frames.mean(dim=1, keepdim=True).repeat(1, fix, 1)], dim=1)
            mask = torch.cat([mask, 0.*mask.mean(dim=1, keepdim=True).repeat(1, fix)], dim=1)
        encoded = self.encode_input(frames, mask, self.frame_input_proj, self.frame_encoder_1, self.frame_pos_embed, self.weight_token)
        encoded = torch.where(mask.unsqueeze(-1).repeat(1, 1, encoded.shape[-1]) == 1., encoded, 0.*encoded)
        return encoded, clip

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                               video_proposal_feat=None, encoded_frame_feat=None, return_query_feats=False):
        if not self.fine_pool_enabled:
            return Original.get_pred_from_raw_query(self, query_feat, query_mask, query_labels,
                        video_proposal_feat, encoded_frame_feat, return_query_feats)
        query = self.encode_query(query_feat, query_mask)
        frame, valid = encoded_frame_feat
        clip_scores = self.get_clip_scale_scores(query, video_proposal_feat)
        frame_scores = masked_scores(query, frame, valid)
        if return_query_feats:
            return (clip_scores, self.get_unnormalized_clip_scale_scores(query, video_proposal_feat),
                    frame_scores, masked_scores(query, frame, valid, unnormalized=True))
        return clip_scores, frame_scores

    def forward(self, batch):
        out = Original.forward(self, batch)
        return dict(scores=dict(clip=out[0], frame=out[3]), raw_scores=dict(clip=out[1], frame=out[4]),
                    label_dict=out[2], query=out[5], ot=out[6])

    def fuse(self, scores):
        return self.fusion_weights[0]*scores['clip']+self.fusion_weights[1]*scores['frame']


def valid_length_pool(encoded,mask):
    valid=mask.bool();lengths=valid.sum(1)
    assert bool((lengths>0).all())
    boundaries=torch.round(torch.arange(33,device=encoded.device).float().unsqueeze(0)*lengths.unsqueeze(1)/32).long()
    raw_start,end=boundaries[:,:-1],boundaries[:,1:]
    nonempty=raw_start<end
    start=torch.minimum(raw_start,lengths.unsqueeze(1)-1)
    end=torch.where(nonempty,end,start+1)
    ordinal=valid.long().cumsum(1)-1
    membership=(ordinal.unsqueeze(1)>=start.unsqueeze(2))&(ordinal.unsqueeze(1)<end.unsqueeze(2))&valid.unsqueeze(1)
    counts=membership.sum(2)
    assert bool((counts>0).all())
    weights=membership.to(encoded.dtype)/counts.unsqueeze(-1)
    clean=encoded.masked_fill(~valid.unsqueeze(-1),0)
    pooled=torch.bmm(weights,clean)
    return pooled,torch.ones_like(counts,dtype=torch.bool)


class RevisedFrameNet(FrameNet):
    def encode_context(self,clips,frames,mask=None):
        if self.fine_pool_enabled:
            encoded,clip=Original.encode_context(self,clips,frames,mask)
            padded=torch.nn.functional.pad(mask,(0,128-mask.shape[1]))
            return valid_length_pool(encoded,padded),clip
        assert self.sparse_frame_enabled and frames.shape[1]==32 and mask is None
        clip=self.encode_input(clips,None,self.clip_input_proj,self.clip_encoder,self.clip_pos_embed,self.weight_token)
        # Exactly 32 real/repeated sampled frames: no padding, no attention mask.
        frame=self.encode_input(frames,None,self.frame_input_proj,self.frame_encoder_1,self.frame_pos_embed,self.weight_token)
        return frame,clip
