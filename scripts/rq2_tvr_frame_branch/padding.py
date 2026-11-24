import numpy as np
import torch
from torch.nn import functional as F


def valid_only(frame, mask, old, query, truth):
    length=mask.sum(1);short=length<frame.shape[1]
    assert np.array_equal(mask,np.arange(frame.shape[1])[None,:]<length[:,None])
    assert np.all(frame[~mask]==0),'Padding is not exactly zero: need a different audit'
    nq,nv=old.shape;q=F.normalize(torch.from_numpy(np.asarray(query).copy()),dim=-1)
    corrected=np.array(old, dtype=np.float32, copy=True)
    totals=dict(eligible_pairs=int(nq*short.sum()),all_pairs=int(nq*nv),positive_queries=nq,
     eligible_positive_queries=int(short[truth].sum()),candidate_zero_score_pairs=0,padding_strict_wins=0,
     padding_tied_with_valid_max=0,positive_padding_strict_wins=0,positive_padding_ties=0,unexpected_positive_valid_max=0)
    diag=np.full(nq,np.nan);probe=0.
    for v in np.flatnonzero(short):
     qi=np.flatnonzero(old[:,v]==0);totals['candidate_zero_score_pairs']+=len(qi)
     raw=F.normalize(torch.from_numpy(np.asarray(frame[v,mask[v]]).copy()),dim=-1)
     # Positive cached scores prove that a zero padded vector cannot win.
     probeqi=np.flatnonzero(old[:,v]>0)[:2]
     if len(probeqi):
      native=torch.matmul(raw,q[probeqi].t()).max(0).values.numpy();probe=max(probe,float(np.max(np.abs(native-old[probeqi,v]))))
     for a in range(0,len(qi),512):
      ix=qi[a:a+512];values=torch.matmul(raw,q[ix].t()).max(0).values.numpy();corrected[ix,v]=values
      strict=values<0;tie=values==0;positive=truth[ix]==v
      totals['padding_strict_wins']+=int(strict.sum());totals['padding_tied_with_valid_max']+=int(tie.sum())
      totals['positive_padding_strict_wins']+=int((strict&positive).sum());totals['positive_padding_ties']+=int((tie&positive).sum())
      totals['unexpected_positive_valid_max']+=int((values>1e-6).sum())
      diag[ix[positive]]=values[positive]

    assert probe<2e-5 and totals['unexpected_positive_valid_max']==0,(probe,totals)

    return corrected, totals, diag, probe
