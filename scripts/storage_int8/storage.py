"""Numerical implementation used by the reported experiment."""
import numpy as np
DTYPES = {'fp32': np.float32, 'fp16': np.float16, 'int8': np.int8}
from torch.nn import functional as F
import torch


def normalize(value):
    value = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite embeddings')
    # Match the original evaluator's normalization on the encoder device.
    return F.normalize(value, dim=-1).cpu()


def pack(value, mode):
    """Input has already been L2 normalized. One fp32 scale per vector for int8."""
    if mode not in DTYPES:
        raise ValueError(mode)
    if mode != 'int8':
        return value.numpy().astype(DTYPES[mode]), None
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127., torch.ones_like(maximum))
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.numpy(), scale.squeeze(-1).numpy()
