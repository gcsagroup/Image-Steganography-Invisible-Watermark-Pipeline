"""Legacy-compatible orthonormal 2D DCT transforms."""

from __future__ import annotations

import numpy as np
import torch
from scipy.fftpack import dct, idct


def dct2d(tensor: torch.Tensor) -> torch.Tensor:
    array = tensor.detach().float().cpu().numpy()
    transformed = dct(dct(array, axis=1, norm="ortho"), axis=0, norm="ortho")
    return torch.from_numpy(np.asarray(transformed, dtype=np.float32)).to(tensor.device)


def idct2d(tensor: torch.Tensor) -> torch.Tensor:
    array = tensor.detach().float().cpu().numpy()
    transformed = idct(idct(array, axis=1, norm="ortho"), axis=0, norm="ortho")
    return torch.from_numpy(np.asarray(transformed, dtype=np.float32)).to(tensor.device)
