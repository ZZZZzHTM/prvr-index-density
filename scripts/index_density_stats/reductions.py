"""Numerical implementation used by the reported experiment."""
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.spatial.distance import squareform
import torch


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def adjacent_all(raw,budgets=(8,16,32)):
    x=np.asarray(raw,dtype=np.float64);sums=x.copy();counts=np.ones(len(x),dtype=np.int64)
    targets={min(len(x),k) for k in budgets};snapshots={}
    while True:
        length=len(sums)
        if length in targets:snapshots[length]=(sums/counts[:,None]).astype(np.float32)
        if length<=min(targets):break
        means=sums/counts[:,None];u=unit(means);cos=np.einsum('ij,ij->i',u[:-1],u[1:])
        pos=int(np.argmax(cos))
        sums[pos]+=sums[pos+1];counts[pos]+=counts[pos+1]
        sums=np.delete(sums,pos+1,axis=0);counts=np.delete(counts,pos+1)
    return {k:snapshots[min(len(x),k)] for k in budgets}


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


def farthest(raw,k,seeds=None):
    x=np.asarray(raw,dtype=np.float64)
    if len(x)<k:return x.astype(np.float32)
    u=unit(x);selected=[]
    if seeds is None:
        mean=x.mean(0,keepdims=True);first=int(np.argmax(1.-(u@unit(mean).T)[:,0]))
        selected=[first];representatives=[x[first].copy()];maxcos=u@u[first]
        count=k-1
    else:
        representatives=[v.copy() for v in np.asarray(seeds,dtype=np.float64)]
        maxcos=(u@unit(seeds).T).max(1);count=k-len(representatives)
    for _ in range(count):
        distance=1.-maxcos
        if selected:distance[np.array(selected,dtype=np.int64)]=-np.inf
        index=int(np.argmax(distance));selected.append(index);representatives.append(x[index].copy())
        maxcos=np.maximum(maxcos,u@u[index])
    assert len(selected)==len(set(selected)) and len(representatives)==k
    return np.stack(representatives).astype(np.float32)


def distances(x,centers,xnorm):
    return np.maximum(xnorm[:,None]+np.einsum('ij,ij->i',centers,centers)[None,:]-2.*(x@centers.T),0.)


def fit(encoded,config):
    original=np.asarray(encoded,dtype=np.float64)
    assert original.ndim==2 and len(original)>config['clusters'] and np.isfinite(original).all()
    offset=original.mean(0);x=original-offset
    n,d=x.shape;k=config['clusters'];xnorm=np.einsum('ij,ij->i',x,x)
    threshold=float(np.var(x,axis=0).mean())*config['tol']
    rng=np.random.default_rng(config['seed']);best=None;restarts=[]
    for restart in range(config['n_init']):
        chosen=[int(rng.integers(n))];center=[x[chosen[0]].copy()]
        nearest=((x-center[0])**2).sum(1)
        while len(center)<k:
            total=float(nearest.sum())
            if total>0:
                candidate=int(rng.choice(n,p=nearest/total))
            else:
                candidate=next(i for i in range(n) if i not in chosen)
            assert candidate not in chosen or nearest[candidate]>0
            chosen.append(candidate);center.append(x[candidate].copy())
            nearest=np.minimum(nearest,((x-center[-1])**2).sum(1))
        centers=np.stack(center);initial_inertia=float(nearest.sum())
        previous=None;relocations=0;reason='max_iter'
        for iteration in range(config['max_iter']):
            squared=distances(x,centers,xnorm);labels=np.argmin(squared,axis=1)
            counts=np.bincount(labels,minlength=k)
            residual=squared[np.arange(n),labels]
            for empty in np.flatnonzero(counts==0):
                eligible=counts[labels]>1
                point=int(np.argmax(np.where(eligible,residual,-np.inf)))
                assert eligible[point]
                old=int(labels[point]);counts[old]-=1;labels[point]=empty;counts[empty]+=1
                residual[point]=-np.inf;relocations+=1
            assert np.all(counts>0)
            updated=np.stack([x[labels==j].mean(0) for j in range(k)])
            shift=float(((updated-centers)**2).sum())
            same=previous is not None and np.array_equal(labels,previous)
            centers=updated
            if same:reason='labels_unchanged';break
            if shift<=threshold:reason='tolerance';break
            previous=labels.copy()
        closest=np.argmin(distances(x,centers,xnorm),axis=1)
        inertia=float(((x-centers[closest])**2).sum())
        assert inertia<=initial_inertia+1e-9*max(1.,initial_inertia)
        record=dict(restart=restart,initial_SSE=initial_inertia,final_SSE=inertia,
                    n_iter=iteration+1,stop_reason=reason,empty_relocations=relocations)
        restarts.append(record)
        if best is None or inertia<best[0]:best=(inertia,centers.copy(),record.copy())
    centers=(best[1]+offset).astype(np.float32)
    assert centers.shape==(k,d) and np.isfinite(centers).all()
    return centers,dict(mode='kmeans',L=n,seed=config['seed'],SSE=best[0],selected_restart=best[2]['restart'],
        n_iter=best[2]['n_iter'],stop_reason=best[2]['stop_reason'],
        distinct_output_centers=int(np.unique(centers,axis=0).shape[0]),restarts=restarts)


def valid_length_pool(encoded,mask,k=32):
    valid=mask.bool();lengths=valid.sum(1)
    assert bool((lengths>0).all())
    boundaries=torch.round(torch.arange(k+1,device=encoded.device).float().unsqueeze(0)*lengths.unsqueeze(1)/k).long()
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


def uniform_pool(raw, k):
    values = torch.as_tensor(raw)
    if len(values) < k:
        return values.clone().numpy()
    mask = torch.ones((1, len(values)), dtype=torch.bool)
    pooled, _ = valid_length_pool(values.unsqueeze(0), mask, k)
    return pooled[0].numpy()
