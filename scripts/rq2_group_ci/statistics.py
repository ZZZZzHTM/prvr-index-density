"""Numerical implementation used by the reported experiment."""
import numpy as np
K=np.array([1,5,10,100])


def display2(value):
    label=f'{value:.2f}'
    return '0.00' if label=='-0.00' else label


def point(before,after):
    return float(100.*(np.count_nonzero(after[:,None]<=K)-np.count_nonzero(before[:,None]<=K))/len(before))


def draws(before,after,videos,repetitions=2000):
    labels,inverse=np.unique(videos,return_inverse=True);nv=len(labels)
    count=np.bincount(inverse,minlength=nv).astype(float)
    hits=np.stack([(before[:,None]<=K),(after[:,None]<=K)],axis=1).reshape(len(before),8)
    totals=np.zeros((nv,8),float);np.add.at(totals,inverse,hits)
    rng=np.random.default_rng(20261023);parts=[];first_draw=None
    for start in range(0,repetitions,64):
        n=min(64,repetitions-start);indices=rng.integers(0,nv,size=(n,nv))
        weights=np.bincount((indices+np.arange(n)[:,None]*nv).ravel(),minlength=n*nv).reshape(n,nv)
        denominator=weights@count;assert np.all(denominator>0)
        recall=(weights@totals).reshape(n,2,4)*100./denominator[:,None,None]
        parts.append(recall[:,1].sum(1)-recall[:,0].sum(1))
        if first_draw is None:first_draw=indices[0]
    values=np.concatenate(parts)
    # Literal first resample keeps all paired queries of every drawn video.
    expanded=np.concatenate([np.flatnonzero(inverse==i) for i in first_draw])
    assert abs(values[0]-point(before[expanded],after[expanded]))<1e-10
    return values
