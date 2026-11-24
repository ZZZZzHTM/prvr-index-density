"""Numerical implementation used by the reported experiment."""
import numpy as np
COMPRESSED = ('frameWard8', 'frameWard8_int8')
FIELDS = ('R1', 'R5', 'R10', 'R100', 'SumR')
K = np.array([1, 5, 10, 100])


def levels(dataset):
    return ('native', 'n1', 'n2', 'n4', 'full') if dataset == 'tvr' else ('native', 'full')


def metrics(rank):
    recalls = 100. * (rank[:, None] <= K).mean(0)
    return dict(zip(FIELDS, [*map(float, recalls), float(recalls.sum())]))


def hit_delta(after, before):
    return (after[:, None] <= K).astype(np.int64) - (before[:, None] <= K).astype(np.int64)


def estimate(delta, interval, nq, nv):
    recall_points = 100. * delta.mean(0)
    point = float(100. * delta.sum(1).mean())
    interval = list(map(float, interval))
    return dict(point=point, ci=interval, ci_low=interval[0], ci_high=interval[1],
                n_query=nq, n_video=nv,
                **{'delta_' + field: float(value) for field, value in zip(FIELDS[:4], recall_points)},
                ci_excludes_zero=bool(interval[0] > 0 or interval[1] < 0))


def paired_columns(ranksets, dataset, selected=None):
    deltas, keys, columns = {}, [], []
    for level in levels(dataset):
        before = ranksets[(level, 'uncompressed')]
        if selected is not None:
            before = before[selected]
        for config in COMPRESSED:
            after = ranksets[(level, config)]
            if selected is not None:
                after = after[selected]
            deltas[(level, config)] = hit_delta(after, before)
    for level in levels(dataset):
        for config in COMPRESSED:
            for label, delta in [('delta', deltas[(level, config)]),
                                 ('D', deltas[(level, config)] - deltas[('native', config)])]:
                keys.append((level, config, label))
                columns.append(delta.sum(1))
    return deltas, keys, np.column_stack(columns)
