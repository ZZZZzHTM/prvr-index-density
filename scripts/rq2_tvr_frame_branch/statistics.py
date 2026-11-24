"""Numerical implementation used by the reported experiment."""
import numpy as np


class Moments:
 def __init__(self):self.n=0;self.total=None;self.sq=None
 def add(self,x):
  x=np.asarray(x,dtype=np.float64);s=x.sum(0);q=np.einsum('ij,ij->j',x,x)
  self.total=s if self.total is None else self.total+s;self.sq=q if self.sq is None else self.sq+q;self.n+=len(x)
 def mean(self):return self.total/self.n
 def finish(self,original_top=None):
  energy=self.sq/self.n;total=float(energy.sum());order=np.argsort(-energy,kind='stable')[:5]
  row=dict(vectors=self.n,dimensions=len(energy),mean_squared_norm=total,dimension_mean=self.mean().tolist(),dimension_energy=energy.tolist(),
   top5=[dict(dimension=int(i),energy_fraction=float(energy[i]/total)) for i in order],
   top5_cumulative_energy_fraction=float(energy[order].sum()/total),common_mean_energy_fraction=float(np.sum(self.mean()**2)/total))
  if original_top is not None:row['original_top5_dimensions']=[dict(dimension=int(i),energy_fraction=float(energy[i]/total)) for i in original_top]
  return row


class Cosines:
 def __init__(self):self.n=0;self.sum=None;self.sum_norm2=0.;self.token_sum=None;self.within_sum_norm2=0.;self.tokens=0;self.within_tokens2=0
 def add(self,instances):
  for x in instances:
   x=np.asarray(x,dtype=np.float64);mean=x.mean(0);mean/=max(float(np.linalg.norm(mean)),1e-12)
   self.sum=mean.copy() if self.sum is None else self.sum+mean;self.sum_norm2+=float(mean@mean);self.n+=1
   u=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);s=u.sum(0)
   self.token_sum=s.copy() if self.token_sum is None else self.token_sum+s
   self.within_sum_norm2+=float(s@s);self.tokens+=len(x);self.within_tokens2+=len(x)**2
 def finish(self):
  return dict(instances=self.n,instance_unordered_pairs=self.n*(self.n-1)//2,
   instance_mean_cosine=float((self.sum@self.sum-self.sum_norm2)/(self.n*(self.n-1))),
   cross_instance_vector_mean_cosine=float((self.token_sum@self.token_sum-self.within_sum_norm2)/(self.tokens**2-self.within_tokens2)),
   cross_instance_vector_unordered_pairs=(self.tokens**2-self.within_tokens2)//2)


def center(x,mu,renorm):
 x=np.asarray(x,dtype=np.float32)-mu
 if renorm:x=x/(np.linalg.norm(x,axis=1,keepdims=True)+np.float32(1e-5))
 return x


def windows(raw_n,duration):
 if raw_n<=128:
  left=np.arange(raw_n);right=left+1
 else:
  cuts=np.round(np.arange(129)/128.*raw_n).astype(np.int32);cuts=np.minimum(cuts,raw_n-1)
  left=cuts[:-1];right=np.where(cuts[1:]>left,cuts[1:],left+1)
 return left*duration/raw_n,right*duration/raw_n
