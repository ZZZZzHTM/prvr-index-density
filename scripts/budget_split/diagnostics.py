"""Numerical implementation used by the reported experiment."""
import os
import json
import hashlib
from pathlib import Path
import numpy as np
RUNTIME=Path(os.environ.get('PRVR_WORK','work'))
def write(path, value):
    Path(path).write_text(json.dumps(value,allow_nan=False),encoding='utf8')

from scipy.stats import spearmanr
import warnings


def finite(value):return float(value) if np.isfinite(value) else None


def rho(x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');return finite(spearmanr(x,y).statistic)


def distribution(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)];q=np.quantile(a,[.05,.25,.5,.75,.95],method='linear')
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std(ddof=0)),std_ddof=0,P5=float(q[0]),Q25=float(q[1]),median=float(q[2]),Q75=float(q[3]),P95=float(q[4]),IQR=float(q[3]-q[1]))


def centered_intervals(er,target):
    x=np.sort(er,kind='stable');m=float(np.median(x));distance=np.abs(x-m);order=np.argsort(distance,kind='stable');sorted_d=distance[order]
    ends=np.r_[np.flatnonzero(sorted_d[1:]!=sorted_d[:-1]),len(x)-1]
    left=np.minimum.accumulate(order)[ends];right=np.maximum.accumulate(order)[ends]+1;n=ends+1
    assert np.array_equal(right-left,n)
    def quantile(p):
        at=left+p*(n-1);lo=np.floor(at).astype(int);hi=np.ceil(at).astype(int);f=at-lo
        return x[lo]*(1-f)+x[hi]*f
    iqrs=quantile(.75)-quantile(.25);errors=np.abs(iqrs-target)
    return dict(center=m,widths=sorted_d[ends],left=left,right=right,n=n,iqrs=iqrs,errors=errors)


def matched(ds,seed,data,target,mode):
    er=data['er'];loss=data['loss'];plan=centered_intervals(er,target);error=plan['errors']
    valid=np.flatnonzero(error<=1e-6) if mode=='strict' else np.flatnonzero(error<=float(error.min())+1e-12)
    nearest=np.flatnonzero(error<=float(error.min())+1e-12);best_nearest=int(max(nearest,key=lambda i:(int(plan['n'][i]),-float(plan['widths'][i]))))
    out=dict(dataset=ds,seed=seed,mode=mode,target_TVR_IQR=target,original_IQR=distribution(er)['IQR'],center=plan['center'],
             candidate_intervals=len(plan['widths']),maximum_reachable_IQR=float(plan['iqrs'].max()),minimum_absolute_IQR_error=float(error.min()),
             closest_reachable_IQR=float(plan['iqrs'][best_nearest]),closest_n_video=int(plan['n'][best_nearest]),tolerance=1e-6)
    if not len(valid):
        out.update(status='UNAVAILABLE',reason='No centered contiguous empirical subset attains the target IQR within 1e-6; no approximate subset substituted.',rho=None,bootstrap=None)
        return out
    best=int(max(valid,key=lambda i:(int(plan['n'][i]),-float(plan['widths'][i]))));h=float(plan['widths'][best]);selected=np.abs(er-plan['center'])<=h
    actual=distribution(er[selected])['IQR'];assert abs(actual-float(plan['iqrs'][best]))<1e-10
    finite_mask=selected&np.isfinite(er)&np.isfinite(loss);x,y=er[finite_mask],loss[finite_mask];n=len(x)
    if n<3 or np.unique(x).size<2 or np.unique(y).size<2:
        out.update(status='UNAVAILABLE',reason='Matched subset has fewer than three finite videos or a constant variable.',n_video=n,rho=None,bootstrap=None);return out
    rng=np.random.default_rng(20261120);draw_indices=rng.integers(0,n,size=(200,n));draws=[rho(x[a],y[a]) for a in draw_indices]
    clean=np.array([a for a in draws if a is not None]);ci=np.quantile(clean,[.025,.975],method='linear') if len(clean) else [None,None]
    out.update(status='COMPLETE',interval=[plan['center']-h,plan['center']+h],half_width=h,n_video=n,below_200=n<200,
        actual_IQR=actual,absolute_IQR_error=abs(actual-target),relative_IQR_error=abs(actual-target)/target if target else None,
        rho=rho(x,y),selected_video_ids=data['video_ids'][finite_mask].tolist(),
        bootstrap=dict(replicates=200,seed=20261120,unit='video',median=float(np.median(clean)) if len(clean) else None,
                       CI95=[finite(a) if a is not None else None for a in ci],valid_replicates=len(clean),
                       undefined_replicates=200-len(clean),draw_indices_sha256=hashlib.sha256(draw_indices.astype('<i8').tobytes()).hexdigest()))
    folder=RUNTIME/'diagnostics'/'range'/ds/str(seed);folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'selected_mask.npy',finite_mask);np.save(folder/'bootstrap_indices.npy',draw_indices)
    write(folder/'rho_draws.json',dict(rho=draws));return out


def fold(ds,seed,data):
    query_count=np.bincount(data['truth'],minlength=len(data['video_ids']));valid=np.flatnonzero(query_count>=2)
    valid=sorted(valid,key=lambda i:str(data['video_ids'][i]));parts=[]
    for i in valid:
        index=np.flatnonzero(data['truth']==i);parts.append(index[np.argsort(data['query_ids'][index],kind='stable')])
    delta=data['ranks']['F'].astype(float)-data['ranks']['original'];rng=np.random.default_rng(20261120);raw=[];corrected=[];digest=hashlib.sha256()
    for rep in range(200):
        a=[];b=[]
        for indices in parts:
            order=rng.permutation(indices);cut=len(order)//2
            digest.update(data['query_ids'][order].astype('U').tobytes());digest.update(np.asarray([cut],dtype='<i8').tobytes())
            a.append(float(delta[order[:cut]].mean()));b.append(float(delta[order[cut:]].mean()))
        value=rho(a,b);raw.append(value)
        corrected.append(None if value is None or value==-1 else 2*value/(1+value))
    clean=np.array([x for x in corrected if x is not None]);raw_clean=np.array([x for x in raw if x is not None])
    result=dict(dataset=ds,seed=seed,status='COMPLETE',repetitions=200,seed_rng=20261120,n_video=len(valid),
        mean_related_queries_per_video=float(query_count[valid].mean()),raw_spearman_median=float(np.median(raw_clean)) if len(raw_clean) else None,
        Spearman_Brown_median=float(np.median(clean)) if len(clean) else None,valid_repetitions=len(clean),undefined_repetitions=200-len(clean),
        split_assignment_sha256=digest.hexdigest(),negative_values_clipped=False,odd_query_rule='floor(n/2) versus remaining ceil(n/2)')
    folder=RUNTIME/'diagnostics'/'fold'/ds/str(seed);folder.mkdir(parents=True,exist_ok=True)
    write(folder/'draws.json',dict(raw=raw,Spearman_Brown=corrected));write(folder/'SUMMARY.json',result);return result
