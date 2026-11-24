"""Numerical implementation used by the reported experiment."""
from torch.nn import functional as F
import numpy as np
import time
import torch


def metrics(ranks):
    out = {f'R{k}': 100. * int(np.count_nonzero(ranks <= k)) / len(ranks)
           for k in (1, 5, 10, 100)}
    return dict(out, SumR=sum(out.values()))


def score_new_index(query, bank, counts, output):
    """Original399 normalization/product/max loops, 128-video/50-query blocks."""
    started = time.perf_counter()
    nq, nv = len(query), len(bank)
    norm_q = torch.cat([F.normalize(torch.from_numpy(query[i:i+50].copy()), dim=-1)
                        for i in range(0, nq, 50)])
    temporary = output.with_name(output.name + '.pending')
    matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(nq, nv))
    with torch.inference_mode():
        for vi in range(0, nv, 128):
            ve = min(vi + 128, nv)
            width = int(counts[vi:ve].max())
            context = torch.zeros((ve - vi, width, 384), dtype=torch.float32)
            mask = torch.zeros((ve - vi, width), dtype=torch.bool)
            for local, video in enumerate(range(vi, ve)):
                count = int(counts[video])
                context[local, :count] = torch.from_numpy(bank[video, :count].copy())
                mask[local, :count] = True
            normalized = F.normalize(context, dim=-1)
            for qi in range(0, nq, 50):
                qe = min(qi + 50, nq)
                raw = torch.matmul(normalized, norm_q[qi:qe].t()).permute(2, 1, 0)
                matrix[qi:qe, vi:ve] = raw.masked_fill(~mask.t().unsqueeze(0), -torch.inf).max(1).values.numpy()
    matrix.flush()
    del matrix
    temporary.replace(output)
    return time.perf_counter() - started


def rank_fusion(clip, frame, truth, rank_function):
    assert clip.shape == frame.shape == (len(truth), clip.shape[1])
    assert clip.dtype == frame.dtype == np.float32
    ranks = np.empty(len(truth), np.int32)
    for start in range(0, len(truth), 128):
        stop = min(start + 128, len(truth))
        values = .7 * np.asarray(clip[start:stop]) + .3 * np.asarray(frame[start:stop])
        assert np.isfinite(values).all()
        ranks[start:stop] = rank_function(values, truth[start:stop], torch.device('cpu'))[0]
    return ranks


def rank_scores(scores, positives, device):
    ranks = np.empty(len(scores), dtype=np.int32)
    top10 = np.empty((len(scores), min(10, scores.shape[1])), dtype=np.int64)
    # Match official per-row torch.argsort of the negated score, stable=False.
    for i, positive in enumerate(positives):
        order = torch.argsort(-torch.as_tensor(scores[i], device=device)).cpu().numpy()
        ranks[i] = int(np.flatnonzero(order == positive)[0]) + 1
        top10[i] = order[:top10.shape[1]]
    return ranks, top10
