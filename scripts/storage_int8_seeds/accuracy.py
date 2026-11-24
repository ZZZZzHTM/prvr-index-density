"""Numerical implementation used by the reported experiment."""
import numpy as np
DTYPES = {'fp32': np.float32, 'fp16': np.float16, 'int8': np.int8}
from torch.nn import functional as F
import torch


def normalize(value):
    value = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite embeddings')
    # Match the original evaluator's normalization on the encoder device.
    return F.normalize(value, dim=-1).cpu()


def pack(value, mode):
    """Input has already been L2 normalized. One fp32 scale per vector for int8."""
    if mode not in DTYPES:
        raise ValueError(mode)
    if mode != 'int8':
        return value.numpy().astype(DTYPES[mode]), None
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127., torch.ones_like(maximum))
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.numpy(), scale.squeeze(-1).numpy()


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


def metrics(rank):
    r={f'R@{k}':float(100*np.count_nonzero(rank<=k)/len(rank)) for k in [1,5,10,100]};r['SumR']=sum(r.values());return r


def hits(r):return (np.asarray(r)[:,None]<=np.array([1,5,10,100])).sum(1)


def estimate(point,draws):
    lo,hi=np.quantile(draws,[.025,.975],method='linear')
    return dict(point=float(point),CI95=[float(lo),float(hi)],interpretation='detectable positive' if lo>0 else 'detectable negative' if hi<0 else 'no detectable difference',valid_bootstrap=2000)
