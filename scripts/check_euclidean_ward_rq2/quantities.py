"""Numerical implementation used by the reported experiment."""
import numpy as np
K = np.array([1, 5, 10, 100])
from scipy.linalg import svdvals
import torch


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def spectrum(raw,temporal=True):
    x=unit(raw);s=svdvals(x,check_finite=False)
    if s.sum()==0:
        erank=0.;pc90=pc95=0
    else:
        p=s/s.sum();p=p[p>0];erank=float(np.exp(-(p*np.log(p)).sum()))
        energy=np.cumsum(s*s)/np.sum(s*s)
        pc90=int(np.searchsorted(energy,.9)+1);pc95=int(np.searchsorted(energy,.95)+1)
    adjacent=float(np.einsum('ij,ij->i',x[:-1],x[1:]).mean()) if len(x)>1 and temporal else float('nan')
    return dict(effective_rank=erank,pc90=pc90,pc95=pc95,adjacent_cosine=adjacent)


def rank_scores(scores,truth):
    scores=np.asarray(scores);truth=np.asarray(truth)
    positive=scores[np.arange(len(truth)),truth]
    rank=(scores>positive[:,None]).sum(1)+1
    tied=np.flatnonzero((scores==positive[:,None]).sum(1)>1)
    if len(tied):
        order=torch.argsort(-torch.from_numpy(np.ascontiguousarray(scores[tied])),dim=1,stable=False).numpy()
        rank[tied]=np.argmax(order==truth[tied,None],axis=1)+1
    return rank.astype(np.int32),len(tied)


def hits(ranks):
    return np.count_nonzero(np.asarray(ranks)[:, None] <= K, axis=1)


def contribution(ranks):
    baseline = hits(ranks['original'])
    a = np.column_stack([hits(ranks[k])-baseline for k in ['F', 'O', 'C']])
    return np.column_stack([a, a[:, 0]-a[:, 1]-a[:, 2]])


def losses(data):
    truth = data['truth']; nv = len(data['video_ids'])
    count = np.bincount(truth, minlength=nv)
    out = {}
    for name, before, after in [('fusion', 'original', 'F'), ('frame_only', 'original_frame', 'F_frame')]:
        delta = data['ranks'][after].astype(float)-data['ranks'][before]
        out[name] = np.divide(np.bincount(truth, weights=delta, minlength=nv), count,
                              out=np.full(nv, np.nan), where=count > 0)
    return out


def euclidean_centers(raw, k=8):
    from scipy.cluster.hierarchy import linkage, fcluster
    x=np.asarray(raw,dtype=np.float64)
    if len(x)<=k:
        return x.astype(np.float32)
    tree=linkage(unit(x),method='ward',metric='euclidean',optimal_ordering=False)
    labels=fcluster(tree,t=k,criterion='maxclust')
    groups=[np.flatnonzero(labels==v) for v in np.unique(labels)]
    groups.sort(key=lambda group:int(group[0]))
    return np.stack([x[group].mean(0) for group in groups]).astype(np.float32)
