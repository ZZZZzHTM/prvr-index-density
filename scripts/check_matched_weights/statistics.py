"""Numerical implementation used by the reported experiment."""
import numpy as np
K=np.array([1,5,10,100])
from scipy.stats import t


def metrics(rank):
 values=[100.*int(np.count_nonzero(rank<=k))/len(rank) for k in K]
 return dict(zip(('R@1','R@5','R@10','R@100','SumR'),[*values,sum(values)]))


def welch(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);diff=float(a.mean()-b.mean())
 va=float(a.var(ddof=1)/len(a));vb=float(b.var(ddof=1)/len(b));se=np.sqrt(va+vb)
 if se==0:return dict(diff=diff,welch_ci=[diff,diff],df=None,standard_error=0.)
 df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1));margin=float(t.ppf(.975,df)*se)
 return dict(diff=diff,welch_ci=[diff-margin,diff+margin],df=df,standard_error=float(se))


def interpretation(ci):return 'detectable positive' if ci[0]>0 else 'detectable negative' if ci[1]<0 else 'no detectable difference'
