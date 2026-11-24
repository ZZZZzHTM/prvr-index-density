"""Numerical implementation used by the reported experiment."""
from collections import Counter
from torch.nn import functional as F
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.spatial.distance import squareform
import time
import torch


def representatives(counts, nominal):
    counts = np.asarray(counts, dtype=np.int64)
    return dict(nominal=int(nominal), minimum=int(counts.min()), maximum=int(counts.max()),
                mean=float(counts.mean()), median=float(np.median(counts)),
                distribution={str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
                fewer_than_nominal_videos=int(np.count_nonzero(counts < nominal)),
                fewer_than_nominal_fraction=float(np.mean(counts < nominal)),
                videos=int(len(counts)), total_vectors=int(counts.sum()))


def normalized_query(query):
    return F.normalize(torch.from_numpy(np.array(query, dtype=np.float32, copy=True)), dim=-1)


def branch_scores(vectors, counts, query, path):
    """Original normalized dot products/max, excluding scratch padding only.

    Uncompressed contexts have all 32 positions valid. For reduced contexts,
    all actual cluster means participate, including any genuinely zero mean;
    only the storage positions beyond that video's representative count are
    excluded. There is no zero floor or frame-branch mask in this model.
    """
    started = time.perf_counter()
    n_video, slots, dimension = vectors.shape
    assert vectors.dtype == np.float32 and dimension == 384
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.shape == (n_video,) and np.all((counts >= 1) & (counts <= slots))
    target = path.with_name(path.name + '.tmp')
    scores = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32,
                                       shape=(len(query), n_video))
    with torch.inference_mode():
        for video_start in range(0, n_video, 256):
            video_stop = min(video_start + 256, n_video)
            raw = np.array(vectors[video_start:video_stop], dtype=np.float32, copy=True)
            assert np.isfinite(raw).all()
            context = F.normalize(torch.from_numpy(raw), dim=-1)
            valid = torch.arange(slots)[None, :] < torch.from_numpy(counts[video_start:video_stop])[:, None]
            for query_start in range(0, len(query), 128):
                query_stop = min(query_start + 128, len(query))
                # Same [video,representative,query] product as native scorer.
                sim = torch.matmul(context, query[query_start:query_stop].t())
                sim.masked_fill_(~valid[:, :, None], -torch.inf)
                values = sim.max(dim=1).values.t().numpy()
                assert np.isfinite(values).all()
                scores[query_start:query_stop, video_start:video_stop] = values
    scores.flush()
    del scores
    target.replace(path)
    return time.perf_counter() - started


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def ward_all(raw,budgets=(8,16,32)):
    x=np.asarray(raw,dtype=np.float64);u=unit(x)
    distance=np.clip(1.-u@u.T,0.,2.);np.fill_diagonal(distance,0.)
    distance=(distance+distance.T)*.5
    tree=linkage(squareform(distance,checks=False),method='ward',optimal_ordering=False) if len(x)>1 else None
    result={}
    for k in budgets:
        if len(x)<=k:result[k]=x.astype(np.float32);continue
        labels=fcluster(tree,t=k,criterion='maxclust')
        groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        groups.sort(key=lambda a:int(a[0]))
        result[k]=np.stack([x[g].mean(0) for g in groups]).astype(np.float32)
        assert 1<=len(result[k])<=k
    return result
