"""Numerical implementation used by the reported experiment."""
import numpy as np
import torch


def metric(ranks):
    counts = {f'R@{k}': int((ranks <= k).sum()) for k in (1, 5, 10, 100)}
    values = {k: 100*v/len(ranks) for k, v in counts.items()}
    values['SumR'] = sum(values.values())
    return values, counts


def rank_margin(scores, truth, original_rank):
    ranks = np.empty(len(truth), np.int32)
    margins = np.empty(len(truth), np.float64)
    ties, boundaries = 0, {f'K{k}': 0 for k in (1, 5, 10, 100)}
    for begin in range(0, len(truth), 128):
        end = min(begin+128, len(truth))
        a = np.asarray(scores[begin:end])
        gt = truth[begin:end]
        rr, _ = original_rank(a, gt, torch.device('cpu'))
        ranks[begin:end] = rr
        pos = a[np.arange(len(a)), gt]
        first = 1 + (a > pos[:, None]).sum(1)
        equal = (a == pos[:, None]).sum(1)
        assert np.all((first <= rr) & (rr <= first+equal-1))
        ties += int((equal > 1).sum())
        for k in (1, 5, 10, 100):
            boundaries[f'K{k}'] += int(((first <= k) & (first+equal-1 > k)).sum())
        neg = a.copy()
        neg[np.arange(len(a)), gt] = -np.inf
        margins[begin:end] = pos.astype(np.float64) - neg.max(1).astype(np.float64)
    return ranks, margins, dict(positive_tie_queries=ties, ties_crossing_K=boundaries,
                                independent_score_rank_bounds_pass=True)


def rank_scores(scores, positives, device):
    ranks = np.empty(len(scores), dtype=np.int32)
    top10 = np.empty((len(scores), min(10, scores.shape[1])), dtype=np.int64)
    # Match official per-row torch.argsort of the negated score, stable=False.
    for i, positive in enumerate(positives):
        order = torch.argsort(-torch.as_tensor(scores[i], device=device)).cpu().numpy()
        ranks[i] = int(np.flatnonzero(order == positive)[0]) + 1
        top10[i] = order[:top10.shape[1]]
    return ranks, top10
