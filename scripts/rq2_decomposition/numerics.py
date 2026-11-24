"""Numerical implementation used by the reported experiment."""
import numpy as np
KS=np.array([1,5,10,100])
REPS=2000
SEED=20261023
import hashlib
import torch


def hits(ranks):
    return np.count_nonzero(np.asarray(ranks)[:,None]<=KS,axis=1).astype(np.int64)


def metrics(ranks):
    ranks=np.asarray(ranks)
    values={f'R@{k}':float(100*np.count_nonzero(ranks<=k)/len(ranks)) for k in KS}
    values['SumR']=sum(values.values())
    return values


def rank_scores(scores,truth):
    scores=np.asarray(scores);truth=np.asarray(truth)
    positive=scores[np.arange(len(truth)),truth]
    rank=(scores>positive[:,None]).sum(1)+1
    tied=np.flatnonzero((scores==positive[:,None]).sum(1)>1)
    if len(tied):
        order=torch.argsort(-torch.from_numpy(np.ascontiguousarray(scores[tied])),dim=1,stable=False).numpy()
        rank[tied]=np.argmax(order==truth[tied,None],axis=1)+1
    return rank.astype(np.int32),len(tied)


def score_blocks(case,loaded,hybrid=False):
    nq,nv=len(loaded['qids']),len(loaded['vids']);truth=loaded['truth'];scores=loaded['scores']
    arrays={name:np.empty(nq,np.int32) for name in (('original','F','O','C') if hybrid else ('original','F','original_frame','F_frame'))}
    delta=np.empty(nq,np.float64);tie_counts={key:0 for key in arrays}
    for start in range(0,nq,128):
        end=min(nq,start+128);gt=truth[start:end];ix=np.arange(end-start)
        clip=np.asarray(scores['clip'][start:end]);frame=np.asarray(scores['frame'][start:end]);ward=np.asarray(scores['ward'][start:end])
        base=np.float32(.7)*clip+np.float32(.3)*frame
        full=np.float32(.7)*clip+np.float32(.3)*ward
        assert np.isfinite(base).all() and np.isfinite(full).all()
        oldpos=base[ix,gt].copy();newpos=full[ix,gt].copy();delta[start:end]=newpos.astype(float)-oldpos.astype(float)
        if hybrid:
            one=base.copy();one[ix,gt]=newpos
            competitors=full.copy();competitors[ix,gt]=oldpos
            matrices=dict(original=base,F=full,O=one,C=competitors)
        else:
            matrices=dict(original=base,F=full,original_frame=frame,F_frame=ward)
        for key,matrix in matrices.items():
            ranks,ties=rank_scores(matrix,gt);arrays[key][start:end]=ranks;tie_counts[key]+=ties
    return arrays,delta,tie_counts


def bootstrap_values(values,videos,score_change=None,reps=REPS):
    """values are per-query additive contributions; means multiplied by100.

    Score quantiles use exact repeated-query order statistics, not weighted
    interpolation between distinct score values. All columns share draws.
    """
    values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    labels,inv=np.unique(videos,return_inverse=True);nv=len(labels);n=len(inv)
    counts=np.bincount(inv,minlength=nv);totals=np.zeros((nv,values.shape[1]));np.add.at(totals,inv,values)
    rng=np.random.default_rng(SEED);draws=[];qdraws=[];first=None
    if score_change is not None:
        score_change=np.asarray(score_change,dtype=float);order=np.argsort(score_change,kind='stable')
        sorted_scores=score_change[order];sorted_inv=inv[order]
    for start in range(0,reps,32):
        nr=min(32,reps-start);ix=rng.integers(0,nv,size=(nr,nv))
        weight=np.bincount((ix+np.arange(nr)[:,None]*nv).ravel(),minlength=nr*nv).reshape(nr,nv)
        den=weight@counts;draws.append(100*(weight@totals)/den[:,None])
        if score_change is not None:
            cum=np.cumsum(weight[:,sorted_inv],axis=1)
            targets=(den[:,None]-1)*np.array([.1,.5,.9])[None,:]
            lo=np.floor(targets).astype(int);hi=np.ceil(targets).astype(int);frac=targets-lo
            vals=np.empty((nr,3))
            for j in range(3):
                a=np.argmax(cum>lo[:,j,None],axis=1);b=np.argmax(cum>hi[:,j,None],axis=1)
                vals[:,j]=sorted_scores[a]*(1-frac[:,j])+sorted_scores[b]*frac[:,j]
            qdraws.append(vals)
        if first is None:first=ix[0]
    draws=np.concatenate(draws)
    expanded=np.concatenate([np.flatnonzero(inv==i) for i in first])
    assert np.max(np.abs(draws[0]-100*values[expanded].mean(0)))<1e-10
    qvalues=np.concatenate(qdraws) if qdraws else None
    if qvalues is not None:
        assert np.max(np.abs(qvalues[0]-np.quantile(score_change[expanded],[.1,.5,.9],method='linear')))<1e-12
    return draws,qvalues,dict(n_query=n,n_video=nv,seed=SEED,replicates=reps,
                             video_order='lexicographic video ID',same_draws_all_columns=True,
                             draws_sha256=hashlib.sha256(draws.astype('<f8').tobytes()).hexdigest())


def estimate(point,draws,meta):
    ci=np.quantile(draws,[.025,.975],method='linear')
    return dict(point=float(point),ci_low=float(ci[0]),ci_high=float(ci[1]),
                n_query=meta['n_query'],n_video=meta['n_video'])


def fit_and_bootstrap(X,Y,truth,reps=2000):
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    labels,inv=np.unique(truth,return_inverse=True);nv=len(labels);p=X.shape[1];ny=Y.shape[1]
    XX=np.zeros((nv,p,p));XY=np.zeros((nv,p,ny))
    np.add.at(XX,inv,np.einsum('ni,nj->nij',X,X))
    np.add.at(XY,inv,np.einsum('ni,nj->nij',X,Y))
    point=np.linalg.lstsq(X,Y,rcond=None)[0]
    normal=np.linalg.solve(XX.sum(0),XY.sum(0));point_error=float(np.max(np.abs(point-normal)))
    assert point_error<1e-9
    rng=np.random.default_rng(SEED);all_beta=[];first_error=None;fallback=0
    for start in range(0,reps,32):
        size=min(32,reps-start);ix=rng.integers(0,nv,size=(size,nv))
        weights=np.bincount((ix+np.arange(size)[:,None]*nv).ravel(),minlength=size*nv).reshape(size,nv)
        aa=(weights@XX.reshape(nv,-1)).reshape(size,p,p)
        bb=(weights@XY.reshape(nv,-1)).reshape(size,p,ny)
        try:
            beta=np.linalg.solve(aa,bb)
        except np.linalg.LinAlgError:
            beta=[]
            for k in range(size):
                try:
                    beta.append(np.linalg.solve(aa[k],bb[k]))
                except np.linalg.LinAlgError:
                    sqrt=np.sqrt(weights[k,inv]);beta.append(np.linalg.lstsq(X*sqrt[:,None],Y*sqrt[:,None],rcond=None)[0]);fallback+=1
            beta=np.stack(beta)
        if first_error is None:
            sqrt=np.sqrt(weights[0,inv]);literal=np.linalg.lstsq(X*sqrt[:,None],Y*sqrt[:,None],rcond=None)[0]
            first_error=float(np.max(np.abs(beta[0]-literal)));assert first_error<1e-9
        all_beta.append(beta)
    draws=np.concatenate(all_beta)
    return point,draws,dict(point_sufficient_statistics_error=point_error,first_weighted_lstsq_error=first_error,
                           fallback_replicates=fallback,rank=int(np.linalg.matrix_rank(X)),
                           condition_number=float(np.linalg.cond(X)),bootstrap_replicates=reps,
                           video_order='ascending source id_order video index',
                           bootstrap_coefficients_sha256=hashlib.sha256(draws.astype('<f8').tobytes()).hexdigest())
