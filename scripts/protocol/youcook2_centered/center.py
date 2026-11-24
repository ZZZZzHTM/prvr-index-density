"""Numerical implementation used by the reported experiment."""
import torch


def normalize(x,mu,mask=None):
    out=x-mu;out=out/(torch.linalg.vector_norm(out,dim=-1,keepdim=True)+1e-5)
    return out if mask is None else out*mask.unsqueeze(-1)
