from __future__ import annotations

from typing import Tuple

import mlx.core as mx


class MLXOps:
    """MLX implementation of BackendOps."""

    def randn(self, shape: Tuple[int, ...], seed: int):
        key = mx.random.key(seed)
        return mx.random.normal(shape, key=key, dtype=mx.float32)

    def qr(self, matrix):
        return mx.linalg.qr(matrix, stream=mx.cpu)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, a):
        axes = list(range(a.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return mx.transpose(a, axes)

    def norm(self, x, dim: int, keepdim: bool = False):
        return mx.linalg.norm(x, axis=dim, keepdims=keepdim)

    def abs(self, x):
        return mx.abs(x)

    def argmin(self, x, dim: int):
        return mx.argmin(x, axis=dim)

    def index_select(self, table, indices):
        return table[indices]

    def unsqueeze(self, x, dim: int):
        return mx.expand_dims(x, axis=dim)

    def concat(self, arrays: list, dim: int):
        return mx.concatenate(arrays, axis=dim)

    def zeros(self, shape: Tuple[int, ...]):
        return mx.zeros(shape)

    def from_numpy(self, arr):
        return mx.array(arr)

    def to_numpy(self, x):
        import numpy as np

        return np.array(x)

    def float32(self, x):
        return x.astype(mx.float32)

    def float16(self, x):
        return x.astype(mx.float16)

    def uint8(self, x):
        return x.astype(mx.uint8)

    def int64(self, x):
        return x.astype(mx.int64)

    def int8(self, x):
        return x.astype(mx.int8)

    def sign(self, x):
        return mx.sign(x)
