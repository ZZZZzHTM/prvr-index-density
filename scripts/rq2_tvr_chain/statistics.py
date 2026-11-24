"""Numerical implementation used by the reported experiment."""
DATASETS=['tvr','charades','activitynet']
SEEDS=[9527,20261024,20261025]
import hashlib
import numpy as np
from scipy.stats import spearmanr
import warnings


def estimate(point,draws,**fields):
    a=np.asarray(draws);valid=np.isfinite(a);finite=a[valid]
    ci=np.quantile(finite,[.025,.975],method='linear') if len(finite) else [None,None]
    label='unavailable' if ci[0] is None else 'detectable positive' if ci[0]>0 else 'detectable negative' if ci[1]<0 else 'no detectable difference'
    return dict(point=float(point) if point is not None and np.isfinite(point) else None,
                CI95=[float(x) if x is not None else None for x in ci],interpretation=label,
                valid_bootstrap=int(valid.sum()),undefined_bootstrap=int((~valid).sum()),**fields)


def correlation(x,y,weights,fn,**fields):
    n=len(x)
    if n<3 or np.unique(x).size<2 or np.unique(y).size<2:
        return dict(point=None,CI95=None,interpretation='unavailable',n_video=n,reason='fewer than three finite videos or a constant variable',**fields),np.full(2000,np.nan)
    px,py=fn['_rank_plan'](x),fn['_rank_plan'](y);draws=np.empty(2000)
    for st in range(0,2000,32):
        w=weights[st:st+32].astype(float);rx=fn['weighted_centered_ranks'](w,px);ry=fn['weighted_centered_ranks'](w,py)
        den=np.sqrt((w*rx*rx).sum(1)*(w*ry*ry).sum(1))
        draws[st:st+len(w)]=np.divide((w*rx*ry).sum(1),den,out=np.full(len(w),np.nan),where=den>0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');point=float(spearmanr(x,y).statistic)
        repeat=np.repeat(np.arange(n),weights[0]);first=float(spearmanr(x[repeat],y[repeat]).statistic)
    assert abs(first-draws[0])<1e-10 or (not np.isfinite(first) and not np.isfinite(draws[0]))
    return estimate(point,draws,n_video=n,recompute_tied_ranks_each_draw=True,first_expanded_video_draw_verified=True,**fields),draws


def gaps(s,weights,labels,fn,data):
    gt=data['truth'];owner=np.asarray(data['video_ids'])[gt];inv=np.searchsorted(labels,owner)
    assert np.array_equal(labels[inv],owner)
    bins=np.searchsorted(s['group_cuts'],data['er'],side='left');baseline=fn['hits'](data['ranks']['original'])
    values=np.column_stack([fn['hits'](data['ranks'][term])-baseline for term in ['O','C']]);group={};samples={};n=len(labels)
    for q in [0,3]:
        qm=bins[gt]==q;count=np.bincount(inv[qm],minlength=n);total=np.zeros((n,2));np.add.at(total,inv[qm],values[qm])
        den=weights@count;draws=np.divide(100*(weights@total),den[:,None],out=np.full((2000,2),np.nan),where=den[:,None]>0)
        point=100*values[qm].mean(0);group[f'Q{q+1}']=dict(n_query=int(qm.sum()),n_video=int(np.unique(gt[qm]).size),
            O=estimate(point[0],draws[:,0]),C=estimate(point[1],draws[:,1]))
        samples[f'Q{q+1}']=draws
    gap_draws=samples['Q1']-samples['Q4'];out={term:estimate(group['Q1'][term]['point']-group['Q4'][term]['point'],gap_draws[:,j]) for j,term in enumerate(['O','C'])}
    first=np.concatenate([np.flatnonzero(inv==i) for i,cnt in enumerate(weights[0]) for _ in range(int(cnt))])
    direct=[]
    for q in [0,3]:direct.append(100*values[first[bins[gt[first]]==q]].mean(0))
    assert np.max(np.abs(gap_draws[0]-(direct[0]-direct[1])))<1e-10
    return dict(Q1_minus_Q4=out,groups=group,bootstrap=dict(replicates=2000,seed=20261023,
        scheme='whole positive-video resampling, conditional query means within fixed groups',
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),same_draws_O_C=True,first_expanded_query_draw_verified=True)),gap_draws


def decisions(rows):
    ring_checks={};selected=None
    for ring in ['A','B','C']:
        by={ds:[next(r for r in rows if r['dataset']==ds and r['seed']==seed)['correlations'][ring]['point'] for seed in SEEDS] for ds in DATASETS}
        available=all(v is not None for values in by.values() for v in values)
        tvr=available and all(v<=.15 for v in by['tvr']);others=available and all(v>=.30 for ds in ['charades','activitynet'] for v in by[ds])
        qualifies=bool(tvr and others);ring_checks[ring]=dict(points=by,all_TVR_le_015=bool(tvr),all_other_six_ge_030=bool(others),qualifies=qualifies,all_required_defined=available)
        if qualifies and selected is None:selected=ring
    available=all(x['all_required_defined'] for x in ring_checks.values());sentence=None
    ranges={ring:{ds:[float(min(x['points'][ds])),float(max(x['points'][ds]))] if all(v is not None for v in x['points'][ds]) else None for ds in DATASETS} for ring,x in ring_checks.items()}
    if selected:
        rr=ranges[selected];f=lambda ds:f"{rr[ds][0]:.3f}–{rr[ds][1]:.3f}"
        tail=f"(ρ = {f('tvr')}, against {f('charades')} on \\Charades{{}} and {f('activitynet')} on \\ActivityNet{{}})."
        if selected=='A':sentence="On \\TVR{}, effective rank barely tracks how far summarization moves a video's frame outputs "+tail
        elif selected=='B':sentence="On \\TVR{}, summarization moves the outputs of less redundant videos further, but their relevant queries' frame scores do not fall more "+tail
        else:sentence="On \\TVR{}, own score drops follow redundancy but barely change the relevant video's own rank "+tail
    positive=all(r['gaps']['Q1_minus_Q4']['O']['CI95'][0] is not None and r['gaps']['Q1_minus_Q4']['O']['CI95'][0]>0 for r in rows if r['dataset'] in ['charades','activitynet'])
    tvr_none=all(r['gaps']['Q1_minus_Q4']['O']['CI95'][0] is not None and r['gaps']['Q1_minus_Q4']['O']['CI95'][0]<=0<=r['gaps']['Q1_minus_Q4']['O']['CI95'][1] for r in rows if r['dataset']=='tvr')
    gap_ranges={ds:[min(r['gaps']['Q1_minus_Q4']['O']['point'] for r in rows if r['dataset']==ds),max(r['gaps']['Q1_minus_Q4']['O']['point'] for r in rows if r['dataset']==ds)] for ds in DATASETS}
    return dict(chain=dict(selected_breakpoint=selected,decision='break at '+selected if selected else 'no qualifying link' if available else 'required statistic undefined',
        checks=ring_checks,three_seed_ranges=ranges,paper_sentence=sentence,add_paper_sentence=selected is not None,
        otherwise='report all three links in supplementary results',mechanism_explanation=None),
        O_gap=dict(all_Charades_ActivityNet_six_detectable_positive=positive,all_TVR_three_no_detectable_difference=tvr_none,
            criterion_met=bool(positive and tvr_none),three_seed_point_ranges=gap_ranges,
            paper_sentence='the gap is detectable on \\Charades{} and \\ActivityNet{} in every seed but not on \\TVR{}' if positive and tvr_none else None,
            otherwise='report the point-estimate range'),paper_modified=False,no_additional_analysis=True)
