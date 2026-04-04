from __future__ import annotations

from typing import Tuple

import torch


class TorchOps:
    """PyTorch implementation of BackendOps."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def randn(self, shape: Tuple[int, ...], seed: int):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(self.device)

    def qr(self, matrix):
        return torch.linalg.qr(matrix)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, a):
        return a.mT

    def norm(self, x, dim: int, keepdim: bool = False):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

    def abs(self, x):
        return torch.abs(x)

    def argmin(self, x, dim: int):
        return torch.argmin(x, dim=dim)

    def index_select(self, table, indices):
        return table[indices.long()]

    def unsqueeze(self, x, dim: int):
        return x.unsqueeze(dim)

    def concat(self, arrays: list, dim: int):
        return torch.cat(arrays, dim=dim)

    def zeros(self, shape: Tuple[int, ...]):
        return torch.zeros(*shape, device=self.device)

    def from_numpy(self, arr):
        return torch.from_numpy(arr).to(self.device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def float32(self, x):
        return x.float()

    def float16(self, x):
        return x.half()

    def uint8(self, x):
        return x.to(torch.uint8)

    def int64(self, x):
        return x.long()

    def sign(self, x):
        return torch.sign(x)
