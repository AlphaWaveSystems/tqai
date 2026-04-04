from __future__ import annotations

import numpy as np
import numpy.testing as npt


def test_randn_seeded(ops):
    a = ops.randn((4, 4), seed=123)
    b = ops.randn((4, 4), seed=123)
    npt.assert_array_equal(ops.to_numpy(a), ops.to_numpy(b))


def test_randn_different_seeds(ops):
    a = ops.randn((4, 4), seed=1)
    b = ops.randn((4, 4), seed=2)
    assert not np.allclose(ops.to_numpy(a), ops.to_numpy(b))


def test_qr_orthogonality(ops):
    G = ops.randn((8, 8), seed=42)
    Q, R = ops.qr(G)
    Q_np = ops.to_numpy(Q)
    identity = Q_np @ Q_np.T
    npt.assert_allclose(identity, np.eye(8), atol=1e-5)


def test_matmul(ops):
    a = ops.randn((3, 4), seed=1)
    b = ops.randn((4, 5), seed=2)
    c = ops.matmul(a, b)
    c_np = ops.to_numpy(c)
    expected = ops.to_numpy(a) @ ops.to_numpy(b)
    npt.assert_allclose(c_np, expected, atol=5e-3)


def test_transpose(ops):
    a = ops.randn((3, 5), seed=1)
    t = ops.transpose(a)
    assert ops.to_numpy(t).shape == (5, 3)


def test_norm(ops):
    a = ops.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    n = ops.norm(a, dim=0)
    npt.assert_allclose(ops.to_numpy(n), 5.0, atol=1e-5)


def test_argmin(ops):
    a = ops.from_numpy(np.array([[3.0, 1.0, 2.0]], dtype=np.float32))
    idx = ops.argmin(a, dim=-1)
    assert ops.to_numpy(idx).item() == 1


def test_index_select(ops):
    table = ops.from_numpy(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    indices = ops.from_numpy(np.array([2, 0, 3], dtype=np.int64))
    result = ops.index_select(table, indices)
    npt.assert_array_equal(ops.to_numpy(result), [30.0, 10.0, 40.0])


def test_from_to_numpy_roundtrip(ops):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = ops.from_numpy(arr)
    back = ops.to_numpy(tensor)
    npt.assert_array_equal(arr, back)


def test_type_casts(ops):
    a = ops.from_numpy(np.array([1.5, 2.5], dtype=np.float32))
    f16 = ops.float16(a)
    f32 = ops.float32(f16)
    npt.assert_allclose(ops.to_numpy(f32), [1.5, 2.5], atol=1e-2)
