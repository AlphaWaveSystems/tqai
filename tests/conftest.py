from __future__ import annotations

import pytest


def _available_backends():
    backends = []
    try:
        import torch  # noqa: F401
        backends.append("torch")
    except ImportError:
        pass
    try:
        import mlx.core  # noqa: F401
        backends.append("mlx")
    except ImportError:
        pass
    return backends


BACKENDS = _available_backends()


@pytest.fixture(params=BACKENDS)
def backend_name(request):
    return request.param


@pytest.fixture
def ops(backend_name):
    from tqai.backend import get_backend
    return get_backend(backend_name)


@pytest.fixture(params=[64, 128])
def head_dim(request):
    return request.param


@pytest.fixture(params=[2, 3, 4])
def bits(request):
    return request.param
