"""Numerical implementation used by the reported experiment."""
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.spatial.distance import squareform


def geometry(raw,k,distance):
    if distance=='cosine':
        x=np.asarray(raw,dtype=np.float64)
        if len(x)<=k:
            return x.astype(np.float32), np.arange(len(x))
        u=unit(x)
        distances=np.clip(1.-u@u.T,0.,2.)
        np.fill_diagonal(distances,0.)
        distances=(distances+distances.T)*.5
        tree=linkage(squareform(distances,checks=False),method='ward',optimal_ordering=False)
        labels=fcluster(tree,t=k,criterion='maxclust')
        groups=sorted((np.flatnonzero(labels==v) for v in np.unique(labels)),key=lambda g:int(g[0]))
        return np.stack([x[g].mean(0) for g in groups]).astype(np.float32), labels
    x=np.asarray(raw,dtype=np.float64);u=unit(x)
    tree=linkage(u,method='ward',metric='euclidean',optimal_ordering=False) if len(x)>1 else None
    if len(x)<=k:return x.astype(np.float32),np.arange(len(x))
    labels=fcluster(tree,t=k,criterion='maxclust');groups=[np.flatnonzero(labels==v) for v in np.unique(labels)]
    groups.sort(key=lambda a:int(a[0]));means=np.stack([x[g].mean(0) for g in groups]).astype(np.float32)
    return means,labels


def adjusted_rand(a,b):
    _,ia=np.unique(a,return_inverse=True);_,ib=np.unique(b,return_inverse=True)
    ca=np.bincount(ia);cb=np.bincount(ib);n=len(a)
    if n<=1:return 1.
    table=np.bincount(ia*(ib.max()+1)+ib,minlength=len(ca)*len(cb)).reshape(len(ca),len(cb))
    choose=lambda values:np.sum(values*(values-1)//2,dtype=np.int64)
    agreement=float(choose(table));row=float(choose(ca));column=float(choose(cb));pairs=n*(n-1)/2
    expected=row*column/pairs;maximum=.5*(row+column)
    return float((agreement-expected)/(maximum-expected)) if maximum!=expected else 1.


def representative_stats(counts,lengths,k):
    unique,number=np.unique(counts,return_counts=True)
    return dict(mean=float(counts.mean()),median=float(np.median(counts)),fraction_less_than_k=float(np.mean(counts<k)),
        fraction_L_less_than_k=float(np.mean(lengths<k)),fraction_fewer_despite_L_greater_k=float(np.mean((lengths>k)&(counts<k))),
        distribution={str(n):int(m) for n,m in zip(unique,number)},n_video=len(counts))


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
