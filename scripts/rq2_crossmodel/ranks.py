"""Numerical implementation used by the reported experiment."""
import math
import numpy as np
import torch


def native_model(s):return s['model'].removesuffix('_author')


def main_cell(s):return s['dataset'] in ['charades','activitynet'] and native_model(s) not in ['MS-SL','SDM']


def display_model(s):return {'DreamPRVR_ours':'DreamPRVR retrained','DreamPRVR':'DreamPRVR released'}.get(native_model(s),native_model(s)+' released')


def decision(rows):
    primary=[x for x in rows if x['primary_decision_cell']];n=len(primary);k=sum(x['decision']=='replicated' for x in primary);reverse=sum(x['decision']=='reverse' for x in primary)
    high=math.ceil(.75*n);low=math.ceil(.375*n);reproduced=[x for x in primary if x['decision']=='replicated']
    offset=bool(reproduced) and all(x['own_quartiles']['groups']['Q1']['decomposition']['O']['ci_high']<0 and x['own_quartiles']['groups']['Q1']['decomposition']['C']['ci_low']>0 for x in reproduced)
    remaining=[x for x in primary if x['decision']!='replicated'];same_sign=sum(x['own_quartiles']['groups']['Q1']['delta_SumR']['point']>0 and x['own_quartiles']['groups']['Q4']['delta_SumR']['point']<0 for x in remaining)
    if reverse or k<low:
        branch='limit_to_GMM';sentence=f'For the other methods, effective rank correlates positively with compression loss, but opposite quartile shifts are detectable in only {k} of {n} combinations.'
    elif k>=high:
        branch='beyond_GMM';sentence=f'The same opposite shifts appear beyond \\GMM{{}}: in {k} of {n} further checkpoint--dataset combinations from \\Dream{{}}, \\HLFormer{{}}, and \\Holmes{{}}, the most redundant quartile detectably gains and the least redundant detectably loses.'
        if offset:sentence=sentence.rstrip('.')+', with the same offsetting own and competitor effects.'
    else:branch='partial';sentence=f'the shifts are detectable in {k} of {n} further combinations'
    return dict(n_available_primary=n,reproduced=k,reversed=reverse,uncertain=n-k-reverse,upper_threshold=high,lower_threshold=low,
                branch=branch,paper_sentence=sentence,all_reproduced_Q1_O_negative_C_positive=offset,
                other_cells_same_point_sign=same_sign,other_cells_total=len(remaining),add_same_sign_clause=bool(remaining) and same_sign==len(remaining),paper_modified=False)


def rank_scores(scores,truth):
    scores=np.asarray(scores);truth=np.asarray(truth)
    positive=scores[np.arange(len(truth)),truth]
    rank=(scores>positive[:,None]).sum(1)+1
    tied=np.flatnonzero((scores==positive[:,None]).sum(1)>1)
    if len(tied):
        order=torch.argsort(-torch.from_numpy(np.ascontiguousarray(scores[tied])),dim=1,stable=False).numpy()
        rank[tied]=np.argmax(order==truth[tied,None],axis=1)+1
    return rank.astype(np.int32),len(tied)


def correlations(counts,rx,ry,rz):
    vz=(counts*rz*rz).sum(1)
    bx=np.divide((counts*rx*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    by=np.divide((counts*ry*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    xx=rx-bx[:,None]*rz;yy=ry-by[:,None]*rz
    denominator=np.sqrt((counts*xx*xx).sum(1)*(counts*yy*yy).sum(1))
    return np.divide((counts*xx*yy).sum(1),denominator,out=np.full(len(counts),np.nan),where=denominator>0)


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


def ranks(s, d, clip, frame, ward):
    assert clip.shape==frame.shape==ward.shape==(len(d['truth']),len(d['video_ids']))
    out={k:np.empty(len(d['truth']),np.int32) for k in ['original','F','O','C','original_frame','F_frame']}
    # Native ranker for every method: exactly the same tie order as Fig3.
    for start in range(0,len(d['truth']),128):

        stop=min(start+128,len(d['truth']));gt=d['truth'][start:stop];ix=np.arange(len(gt))
        cs=np.asarray(clip[start:stop]);fs=np.asarray(frame[start:stop]);ws=np.asarray(ward[start:stop])
        base=s['weights'][0]*cs+s['weights'][1]*fs;compressed=s['weights'][0]*cs+s['weights'][1]*ws
        own=base.copy();own[ix,gt]=compressed[ix,gt];competitor=compressed.copy();competitor[ix,gt]=base[ix,gt]
        for key,matrix in zip(out,[base,compressed,own,competitor,fs,ws]):
            assert matrix.dtype==np.float32 and np.isfinite(matrix).all()
            order=np.argsort(-matrix,axis=1) if native_model(s)=='MS-SL' else torch.argsort(-torch.from_numpy(matrix),dim=1,stable=False).numpy()
            out[key][start:stop]=np.argmax(order==gt[:,None],axis=1)+1
    return out
