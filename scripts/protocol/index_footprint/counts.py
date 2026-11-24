"""Numerical implementation used by the reported experiment."""
import numpy as np


def vector_counts(case,variant,nv):
    if variant=='uncompressed':return np.full(nv,160,dtype=np.int64)
    frame=np.load(case['compression']['frame_ward8']['counts']['file'])
    if variant=='frameWard8':return 32+frame
    return frame+np.load(case['compression']['clip_ward8']['counts']['file'])
