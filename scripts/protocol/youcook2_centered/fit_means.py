"""Numerical implementation used by the reported experiment."""
import numpy as np


def fit(modules,path,cfg):
    local=dict(cfg,val_caption_file=str(path),test_caption_file=str(path))
    _,ls=modules['data'].make_loaders(local,splits=('val',),workers=8)
    sums={k:np.zeros(1024,np.float64) for k in ('video','query')};counts=dict(video=0,query=0);vids=[];qids=[]
    for clip,frame,mask,indices,ids in ls['val'][0]:
        x=frame.numpy()[mask.numpy().astype(bool)].astype(np.float64)
        sums['video']+=x.sum(0);counts['video']+=len(x);vids.extend(ids)
    for text,mask,indices,ids in ls['val'][1]:
        x=text.numpy()[mask.numpy().astype(bool)].astype(np.float64)
        sums['query']+=x.sum(0);counts['query']+=len(x);qids.extend(ids)
    return {k:(sums[k]/counts[k]).astype(np.float32) for k in sums},counts,vids,qids
