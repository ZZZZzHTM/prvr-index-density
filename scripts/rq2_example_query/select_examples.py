"""Numerical implementation used by the reported experiment."""
import numpy as np
import torch


def choose(cache,role):
    original=cache['original'];full=cache['F'];quartile=cache['quartile']
    if role=='main':
        mask=(quartile==1)&(original>=2)&(original<=100)&(full<original)
        change=original-full
    else:
        mask=(quartile==4)&(original>=1)&(original<=100)&(full>original)
        change=full-original
    candidates=np.flatnonzero(mask)
    if not len(candidates):return None,dict(candidate_count=0,reason='No candidate under the fixed rule'),mask
    median=float(np.median(change[candidates]))
    exact=candidates[change[candidates]==median]
    selection=dict(candidate_count=len(candidates),median_position_change=median,
                   exact_median_count=len(exact),median_definition='usual sample median; no nearest/lower-median fallback',
                   tie_break=['smallest original positive rank','lexicographically smallest original query_id'])
    if not len(exact):
        selection['reason']='No integer movement equals the sample median; fixed rule has no solution'
        return None,selection,mask
    chosen=min(exact,key=lambda i:(int(original[i]),str(cache['query_ids'][i])))
    selection.update(selected_position_change=int(change[chosen]),
                     smallest_original_rank_among_median=int(original[chosen]),
                     count_tied_at_smallest_original_rank=int(np.count_nonzero(original[exact]==original[chosen])))
    return int(chosen),selection,mask


def proportion(mask,candidates,description):
    numerator=int(np.count_nonzero(mask&candidates));denominator=int(np.count_nonzero(candidates))
    return dict(count=numerator,n_query=denominator,fraction=numerator/denominator if denominator else None,
                percentage=100*numerator/denominator if denominator else None,definition=description)


def typicality(cache,role,candidates):
    original=cache['original'];full=cache['F'];decreased=cache['score_change']<0
    direction=full<original if role=='main' else full>original
    key='own_score_decreased_and_rank_improved' if role=='main' else 'own_score_decreased_and_rank_worsened'
    return dict(n_query=int(candidates.sum()),n_video=len(np.unique(cache['positive_videos'][candidates])),
                denominator='complete candidate pool before median and tie breaking',
                **{key:proportion(decreased&direction,candidates,
                                  'F positive fused score < original; F rank < original' if role=='main' else
                                  'F positive fused score < original; F rank > original')},
                O_worse_C_better=proportion((cache['O']>original)&(cache['C']<original),candidates,
                                           'O rank > original AND C rank < original; strict, excludes ties'))


def describe(cache,ids,videos,texts,scores,sources,role,index,selection):
    qid=str(cache['query_ids'][index]);positive=str(cache['positive_videos'][index])
    qi=ids['query_ids'].index(qid);gt=ids['video_ids'].index(positive)
    clip=np.asarray(scores['clip'][qi]);frame=np.asarray(scores['frame'][qi]);ward=np.asarray(scores['ward'][qi])
    original=np.float32(.7)*clip+np.float32(.3)*frame
    full=np.float32(.7)*clip+np.float32(.3)*ward
    order=torch.argsort(-torch.from_numpy(np.stack([original,full])),dim=1,stable=False).numpy()
    positions=np.empty_like(order)
    positions[np.arange(2)[:,None],order]=np.arange(len(original))[None,:]+1
    oldrank,newrank=int(positions[0,gt]),int(positions[1,gt])
    validation=dict(query_id=qid,original=dict(recomputed=oldrank,cached=int(cache['original'][index])),
                    F=dict(recomputed=newrank,cached=int(cache['F'][index])),
                    passed=oldrank==int(cache['original'][index]) and newrank==int(cache['F'][index]))
    gain=(positions[0]<oldrank)&(positions[1]>newrank)
    loss=(positions[0]>oldrank)&(positions[1]<newrank)
    crossing=gain if role=='main' else loss
    opposite=loss if role=='main' else gain
    indices=sorted(np.flatnonzero(crossing),key=lambda v:(-float(original[v]),ids['video_ids'][v]))
    cuts=np.asarray(sources['effective_rank_cuts'])
    by_video={};distribution={f'Q{i}':0 for i in range(1,5)}
    for v in indices:
        video=ids['video_ids'][v];stats=videos[video];er=float(stats['frame_effective_rank'])
        group='Q'+str(int(np.searchsorted(cuts,er,side='left'))+1)
        distribution[group]+=1
        by_video[video]=dict(video_id=video,effective_rank=er,quartile=group,
                            fused_scores=dict(original=float(original[v]),F=float(full[v]),
                                              change=float(np.float64(full[v])-np.float64(original[v]))),
                            candidate_ranks=dict(original=int(positions[0,v]),F=int(positions[1,v])))
    positive_stats=videos[positive]
    example=dict(dataset='activitynet',model='GMMFormerV2 our run',seed=9527,
                 query=dict(id=qid,text=texts[qid],cache_index_zero_based=index,score_matrix_index_zero_based=qi),
                 positive_video=dict(id=positive,effective_rank=float(positive_stats['frame_effective_rank']),
                                     L=int(positive_stats['L']),quartile='Q'+str(int(cache['quartile'][index]))),
                 ranks={name:int(cache[name][index]) for name in ('original','F','O','C')},
                 positive_fused_scores=dict(original=float(original[gt]),F=float(full[gt]),
                                            change=float(np.float64(full[gt])-np.float64(original[gt]))),
                 crossing_competitors=dict(direction='overtaken_by_positive' if role=='main' else 'overtook_positive',
                                           total=len(indices),quartile_counts=distribution,
                                           ordered_video_ids=[ids['video_ids'][v] for v in indices],by_video=by_video,
                                           top5=[by_video[ids['video_ids'][v]] for v in indices[:5]],
                                           order='original fused score descending; video_id lexical tie break'),
                 opposite_direction_crossing_count=int(opposite.sum()),selection=selection,sources=sources)
    return example,validation
