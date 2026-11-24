"""Numerical implementation used by the reported experiment."""
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.spatial.distance import squareform


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def _rank_plan(values):
    order=np.argsort(values,kind='stable');ordered=values[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    inverse=np.empty(len(order),dtype=np.int64)
    inverse[order]=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(order)]))
    return order,starts,inverse


def weighted_centered_ranks(counts,plan):
    order,starts,inverse=plan
    weights=np.add.reduceat(counts[:,order],starts,axis=1)
    ranks=np.cumsum(weights,axis=1)-(weights-1.)/2.
    return ranks[:,inverse]-(counts.sum(1,keepdims=True)+1.)/2.


def tree(raw):
    x=np.asarray(raw,dtype=np.float64);u=unit(x)
    if len(x)==1:return np.empty((0,4),dtype=float)
    distance=np.clip(1.-u@u.T,0.,2.);np.fill_diagonal(distance,0.)
    distance=(distance+distance.T)*.5
    z=linkage(squareform(distance,checks=False),method='ward',optimal_ordering=False)
    assert np.all(np.diff(z[:,2])>=0),'R2 count search requires the monotonic Ward tree'
    return z


def labels_for(z,length,k=None,tau=None):
    cap=min(length,32)
    if tau is None:
        assert k is not None
        if length<=k:return np.arange(length)
        labels=fcluster(z,t=int(k),criterion='maxclust')
    else:
        if length==1:return np.zeros(1,dtype=int)
        labels=fcluster(z,t=float(tau),criterion='distance')
        if len(np.unique(labels))>cap:labels=fcluster(z,t=cap,criterion='maxclust')
    groups=sorted([np.flatnonzero(labels==a) for a in np.unique(labels)],key=lambda g:int(g[0]))
    canonical=np.empty(length,dtype=np.int16)
    for i,g in enumerate(groups):canonical[g]=i
    return canonical


def centers(raw,labels):
    x=np.asarray(raw,dtype=np.float64)
    return np.stack([x[labels==i].mean(0) for i in range(int(labels.max())+1)]).astype(np.float32)


def quotas(feature,lengths,parameter):
    return np.clip(np.rint(parameter*feature),1,np.minimum(lengths,32)).astype(np.int64)


def proportional(feature,lengths,budget):
    positive=feature>0
    upper=max(1.,float(np.max(np.minimum(lengths[positive],32)/feature[positive])*2)) if positive.any() else 1.
    low=0.;limit=int(budget*len(lengths))
    maximum=quotas(feature,lengths,upper)
    if int(maximum.sum())<=limit:
        return maximum,dict(parameter=upper,iterations=0,reason='saturated maximum does not exceed budget')
    iterations=0
    for i in range(80):
        mid=(low+upper)/2
        if mid==low or mid==upper:break
        if int(quotas(feature,lengths,mid).sum())<=limit:low=mid
        else:upper=mid
        iterations=i+1
    result=quotas(feature,lengths,low)
    assert int(result.sum())<=limit
    return result,dict(parameter=low,iterations=iterations,reason='largest feasible lower endpoint',
        bracket=[low,upper],rounding='nearest, ties to even',nominal_mean=float(result.mean()))


def global_threshold(trees,lengths,budget):
    cap_counts=np.array([len(np.unique(labels_for(z,int(l),k=min(l,32)))) for z,l in zip(trees,lengths)])
    heights=np.full((len(lengths),max(1,int(lengths.max())-1)),np.inf)
    for i,z in enumerate(trees):heights[i,:len(z)]=z[:,2]
    finite=heights[np.isfinite(heights)]
    def counts(tau):return np.minimum(lengths-(heights<=tau).sum(1),cap_counts)
    if not len(finite):return np.ones(len(lengths),dtype=int),dict(parameter=0.,iterations=0,reason='all videos have one frame')
    low=float(np.nextafter(finite.min(),-np.inf));high=float(finite.max());limit=int(budget*len(lengths))
    if int(counts(low).sum())<=limit:
        return counts(low),dict(parameter=low,iterations=0,reason='capped maximum does not exceed budget')
    iterations=0
    for i in range(80):
        mid=(low+high)/2
        if mid==low or mid==high:break
        if int(counts(mid).sum())<=limit:high=mid
        else:low=mid
        iterations=i+1
    result=counts(high)
    assert int(result.sum())<=limit
    return result,dict(parameter=high,iterations=iterations,reason='smallest feasible upper threshold',bracket=[low,high])


def allocations(erank,lengths,trees):
    output={};parameters={}
    for b in (4,8,16):
        output[f'B{b}_R0']=np.minimum(lengths,b)
        output[f'B{b}_R1'],parameters[f'B{b}_R1']=proportional(erank,lengths,b)
        output[f'B{b}_R3'],parameters[f'B{b}_R3']=proportional(lengths.astype(float),lengths,b)
        output[f'B{b}_R2'],parameters[f'B{b}_R2']=global_threshold(trees,lengths,b)
    return output,parameters


def correlations(counts,rx,ry,rz):
    vz=(counts*rz*rz).sum(1)
    bx=np.divide((counts*rx*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    by=np.divide((counts*ry*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    xx=rx-bx[:,None]*rz;yy=ry-by[:,None]*rz
    denominator=np.sqrt((counts*xx*xx).sum(1)*(counts*yy*yy).sum(1))
    return np.divide((counts*xx*yy).sum(1),denominator,out=np.full(len(counts),np.nan),where=denominator>0)


def partial_spearman(x,y,z,repetitions=2000):
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z);x=x[valid];y=y[valid];z=z[valid];n=len(x)
    if n<4 or np.unique(x).size<2 or np.unique(y).size<2:
        return dict(n=n,rho=None,CI95=None,reason='insufficient finite or varying data')
    plans=[_rank_plan(a) for a in (x,y,z)]
    ones=np.ones((1,n));rr=[weighted_centered_ranks(ones,p) for p in plans]
    point=correlations(ones,*rr)[0];rng=np.random.default_rng(20261023);output=[]
    for start in range(0,repetitions,64):
        b=min(64,repetitions-start);idx=rng.integers(0,n,size=(b,n))
        counts=np.bincount((idx+np.arange(b)[:,None]*n).ravel(),minlength=b*n).reshape(b,n).astype(float)
        rr=[weighted_centered_ranks(counts,p) for p in plans]
        output.extend(correlations(counts,*rr).tolist())
    a=np.asarray(output);a=a[np.isfinite(a)]
    return dict(n=n,rho=float(point) if np.isfinite(point) else None,
        CI95=np.quantile(a,[.025,.975],method='linear').tolist() if len(a) else None,
        valid_bootstrap=len(a),undefined_bootstrap=repetitions-len(a),
        control='average tied rank of L; intercept included by centered ranks',
        control_constant=np.unique(z).size==1)
