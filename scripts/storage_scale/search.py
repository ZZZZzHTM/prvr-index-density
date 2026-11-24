"""Numerical implementation used by the reported experiment."""
from torch.nn import functional as F
from dataclasses import asdict
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class Term:
    video: str
    query: str
    weight: float


@dataclass(frozen=True)
class Spec:
    method: str
    positions: dict[str, int]
    terms: tuple[Term, ...]
    mean_fusion: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        return cls(obj['method'], obj['positions'], tuple(Term(**x) for x in obj['terms']),
                   obj.get('mean_fusion', False))


def score_block(queries, values, lengths, spec):
    """Inputs are normalized before storage. Storage errors are NOT renormalized away."""
    scores = []
    for term in spec.terms:
        context, query = values[term.video], queries[term.query]
        # Exactly the original matmul orientation; temporary is only [block, P, Q].
        position_scores = torch.matmul(context, query.t())
        valid = torch.arange(context.shape[1], device=context.device)[None, :] < lengths[term.video][:, None]
        position_scores.masked_fill_(~valid.unsqueeze(-1), -torch.inf)
        branch_scores = position_scores.max(dim=1).values
        if spec.method == 'B0' and term.video == 'frame':
            # Original GMMFormerV2 includes zero padding in the 128-position max.
            # Use the original valid length, so compact valid-only storage agrees.
            branch_scores = torch.where(lengths[term.video][:, None] < 128,
                                        branch_scores.clamp_min(0), branch_scores)
        scores.append(branch_scores.t())
        del position_scores
    if spec.mean_fusion:
        # Original Lite implementation: sum then divide, not weighted pre-sums.
        return sum(scores)/len(scores)
    result = scores[0]*spec.terms[0].weight
    for term, value in zip(spec.terms[1:], scores[1:]):
        result = result + value*term.weight
    return result


def select_topk(scores, indices, k):
    """Descending score, ties by original video index ascending, independent of blocks."""
    if indices.ndim == 1:
        indices = indices.expand(scores.shape[0], -1)
    by_index = torch.argsort(indices, dim=1, stable=True)
    indices, scores = indices.gather(1, by_index), scores.gather(1, by_index)
    order = torch.argsort(-scores, dim=1, stable=True)[:, :k]
    return scores.gather(1, order), indices.gather(1, order)


def spec_for(config,fusion):
    return Spec('B0' if config['variant']=='uncompressed' else config['variant'],
                dict(clip=config['clip'],frame=config['frame']),
                (Term('clip','query',fusion[0]),Term('frame','query',fusion[1])))


@torch.inference_mode()
def encode(models,features,mask,normalize=False):
    outputs=[]
    for model in models:
        q=model.encode_query(features,mask).reshape(len(features),-1)
        outputs.append({'query':F.normalize(q.float(),dim=-1) if normalize else q})
    return outputs


@torch.inference_mode()
def ensemble_search(indexes,queries,block_size=1024,k=100,device='cuda:0'):
    assert indexes and len(indexes)==len(queries) and len({v.count for v in indexes})==1
    best_scores=best_indices=None
    for start in range(0,indexes[0].count,block_size):
        end=min(start+block_size,indexes[0].count);total=None
        for index,q in zip(indexes,queries):
            values,lengths=index.read(start,end,device)
            fused=score_block(q,values,lengths,index.spec)
            total=fused if total is None else total+fused
            del values,lengths,fused
        # Match analysis_math.simple_mean: left-to-right fp32 sum, then fp32 reciprocal.
        if len(indexes)>1:total=total*float(np.float32(1./len(indexes)))
        scores,indices=select_topk(total,torch.arange(start,end,device=device),k)
        if best_scores is not None:
            scores,indices=select_topk(torch.cat((best_scores,scores),dim=1),torch.cat((best_indices,indices),dim=1),k)
        best_scores,best_indices=scores,indices
        del total,scores,indices
    return best_scores,best_indices
