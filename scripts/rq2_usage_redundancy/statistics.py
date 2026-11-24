"""Numerical implementation used by the reported experiment."""
from torch.nn import functional as F
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
import time
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
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


def ward_geometry(raw,k=8):
    x=np.asarray(raw,dtype=np.float64);u=unit(x);length=len(x)
    if length<=k:
        groups=[np.array([i]) for i in range(length)];tree=None
    else:
        distance=np.clip(1.-u@u.T,0.,2.);np.fill_diagonal(distance,0.)
        distance=(distance+distance.T)*.5
        tree=linkage(squareform(distance,checks=False),method='ward',optimal_ordering=False)
        labels=fcluster(tree,t=k,criterion='maxclust')
        groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        groups.sort(key=lambda a:int(a[0]))
    centers=np.stack([x[g].mean(0) for g in groups]).astype(np.float32)
    anchors=np.zeros(length,dtype=bool)
    for g,center in zip(groups,centers):
        similarities=u[g]@unit(center[None,:])[0]
        anchors[g[similarities==similarities.max()]]=True
    merges=length-len(groups)
    last=float(tree[merges-1,2]) if merges else float('nan')
    mean=float(tree[:merges,2].mean()) if merges else float('nan')
    return centers,anchors,dict(ward8_clusters=len(groups),ward8_merge_distance=last,ward8_mean_merge_distance=mean)


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


def bootstrap_correlations(table,fields,target,repetitions=2000,seed=20261023):
    y=np.asarray(table[target],dtype=np.float64);out={};cohorts={}
    for field in fields:
        x=np.asarray(table[field],dtype=np.float64);valid=np.isfinite(x)&np.isfinite(y);n=int(valid.sum())
        if n<3 or np.unique(x[valid]).size<2 or np.unique(y[valid]).size<2:
            out[field]=dict(n=n,rho=None,CI95=None,reason='fewer than 3 finite videos or constant variable');continue
        key=valid.tobytes();cohorts.setdefault(key,dict(valid=valid,fields=[]))['fields'].append(field)
        out[field]=dict(n=n,rho=float(spearmanr(x[valid],y[valid]).statistic))
    for cohort in cohorts.values():
        valid=cohort['valid'];yy=y[valid];n=len(yy);fields0=cohort['fields']
        plans={field:_rank_plan(np.asarray(table[field],dtype=np.float64)[valid]) for field in fields0}
        py=_rank_plan(yy);rng=np.random.default_rng(seed);draws={field:[] for field in fields0}
        for start in range(0,repetitions,64):
            size=min(64,repetitions-start);indices=rng.integers(0,n,size=(size,n))
            counts=np.bincount((indices+np.arange(size)[:,None]*n).ravel(),minlength=size*n).reshape(size,n).astype(np.float64)
            ry=weighted_centered_ranks(counts,py);vy=np.sum(counts*ry*ry,axis=1)
            for field in fields0:
                rx=weighted_centered_ranks(counts,plans[field]);vx=np.sum(counts*rx*rx,axis=1)
                covariance=np.sum(counts*rx*ry,axis=1);denominator=np.sqrt(vx*vy)
                values=np.divide(covariance,denominator,out=np.full(size,np.nan),where=denominator>0)
                draws[field].extend(values.tolist())
        for field in fields0:
            a=np.asarray(draws[field]);a=a[np.isfinite(a)]
            out[field].update(CI95=np.quantile(a,[.025,.975],method='linear').tolist() if len(a) else None,
                valid_bootstrap_replicates=len(a),undefined_bootstrap_replicates=repetitions-len(a))
    return out


def compute_usage(case,frames,clips,mask,queries,centers,counts,truth,folder):
    nq,nv=len(queries),len(frames)
    q=torch.cat([F.normalize(torch.from_numpy(queries[i:i+50].copy()),dim=-1) for i in range(0,nq,50)])
    argmax=np.lib.format.open_memmap(folder/'argmax.npy',mode='w+',dtype=np.uint8,shape=(nv,nq))
    any_max=np.zeros((nv,128),dtype=bool);ties=np.zeros(nv,dtype=np.int64)
    positive_max=np.empty(nq,dtype=np.float32)
    own=case['cache_mode']=='encode_missing'
    scores={name:np.lib.format.open_memmap(case[key],mode='w+',dtype=np.float32,shape=(nq,nv))
        for name,key in [('clip','clip_scores'),('frame','frame_scores'),('ward','ward_scores')]} if own else {}
    start=time.perf_counter();probe={}
    for vi in range(0,nv,16):
        ve=min(vi+16,nv);valid=torch.from_numpy(np.array(mask[vi:ve],copy=True))
        fn=F.normalize(torch.from_numpy(frames[vi:ve].copy()),dim=-1)
        wn=F.normalize(torch.from_numpy(centers[vi:ve].copy()),dim=-1)
        wm=torch.arange(8)[None,:]<torch.from_numpy(counts[vi:ve].copy())[:,None]
        cn=F.normalize(torch.from_numpy(clips[vi:ve].copy()),dim=-1) if own else None
        for qi in range(0,nq,512):
            qe=min(qi+512,nq);qt=q[qi:qe].t()
            sim=torch.matmul(fn,qt).masked_fill(~valid[:,:,None],-torch.inf)
            values,indices=sim.max(1)
            argmax[vi:ve,qi:qe]=indices.numpy().astype(np.uint8)
            tied=sim==values[:,None,:]
            any_max[vi:ve]|=tied.any(2).numpy();ties[vi:ve]+=(tied.sum(1)>1).sum(1).numpy()
            local=np.flatnonzero((truth[qi:qe]>=vi)&(truth[qi:qe]<ve))
            if len(local):positive_max[qi+local]=values[torch.from_numpy(truth[qi+local]-vi),torch.from_numpy(local)].numpy()
            if own:
                raw=values.clone();short=valid.sum(1)<128
                raw[short]=torch.maximum(raw[short],torch.tensor(0.))
                ward=torch.matmul(wn,qt).masked_fill(~wm[:,:,None],-torch.inf).max(1).values
                scores['clip'][qi:qe,vi:ve]=torch.matmul(cn,qt).max(1).values.t().numpy()
                scores['frame'][qi:qe,vi:ve]=raw.t().numpy();scores['ward'][qi:qe,vi:ve]=ward.t().numpy()
            elif vi==qi==0:
                # Check cached scoring identity without re-running an encoder.
                raw=values.clone();short=valid.sum(1)<128;raw[short]=torch.maximum(raw[short],torch.tensor(0.))
                ward=torch.matmul(wn,qt).masked_fill(~wm[:,:,None],-torch.inf).max(1).values
                probe=dict(original_frame_max_abs_error=float(np.max(np.abs(raw.t().numpy()-array(case['frame_scores'])[:qe,:ve]))),
                    ward_frame_max_abs_error=float(np.max(np.abs(ward.t().numpy()-array(case['ward_scores'])[:qe,:ve]))))
                assert max(probe.values())<2e-5,(case['key'],probe)

    argmax.flush()
    for score in scores.values():score.flush()
    return argmax,any_max,ties,positive_max,probe


def array(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)
