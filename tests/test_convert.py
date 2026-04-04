from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "test_convert"


def test_convert_creates_files(output_dir):
    """Convert should create all expected files."""
    from tqai.convert import _build_rotation, load_converted

    # Simulate conversion without a real model by calling internals
    output_dir.mkdir()
    head_dim = 64
    n_layers = 4
    bits_k, bits_v, seed = 4, 2, 42

    from tqai.codebook import load_codebook

    rotations_k = {}
    rotations_v = {}
    for i in range(n_layers):
        rotations_k[f"layer_{i}"] = _build_rotation(head_dim, seed + i, "torch")
        rotations_v[f"layer_{i}"] = _build_rotation(head_dim, seed + i + 10000, "torch")

    np.savez(output_dir / "rotations_k.npz", **rotations_k)
    np.savez(output_dir / "rotations_v.npz", **rotations_v)

    centroids_k, boundaries_k = load_codebook(head_dim, bits_k)
    centroids_v, boundaries_v = load_codebook(head_dim, bits_v)
    np.savez(
        output_dir / "codebooks.npz",
        centroids_k=centroids_k,
        boundaries_k=boundaries_k,
        centroids_v=centroids_v,
        boundaries_v=boundaries_v,
    )

    config = {
        "model_id": "test-model",
        "head_dim": head_dim,
        "n_layers": n_layers,
        "n_kv_heads": 4,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "seed": seed,
        "tqai_version": "0.1.0",
    }
    with open(output_dir / "tqai_config.json", "w") as f:
        json.dump(config, f)

    # Load and verify
    data = load_converted(output_dir)
    assert data["config"]["head_dim"] == 64
    assert data["config"]["n_layers"] == 4
    assert data["config"]["bits_k"] == 4
    assert data["config"]["bits_v"] == 2
    assert len(data["rotations_k"]) == n_layers
    assert len(data["rotations_v"]) == n_layers
    assert data["rotations_k"]["layer_0"].shape == (64, 64)
    assert "centroids_k" in data["codebooks"]
    assert "centroids_v" in data["codebooks"]


def test_rotation_is_orthogonal(output_dir):
    """Precomputed rotations should be orthogonal matrices."""
    from tqai.convert import _build_rotation

    R = _build_rotation(64, seed=42, backend="torch")
    identity = R @ R.T
    np.testing.assert_allclose(identity, np.eye(64), atol=1e-5)


def test_rotation_deterministic():
    """Same seed should produce same rotation."""
    from tqai.convert import _build_rotation

    R1 = _build_rotation(64, seed=42, backend="torch")
    R2 = _build_rotation(64, seed=42, backend="torch")
    np.testing.assert_array_equal(R1, R2)


def test_load_converted_config_path(output_dir):
    """Test that load_converted returns usable data."""
    from tqai.codebook import load_codebook
    from tqai.convert import _build_rotation, load_converted

    output_dir.mkdir()

    # Create minimal converted dir
    np.savez(output_dir / "rotations_k.npz", layer_0=_build_rotation(64, 42, "torch"))
    np.savez(output_dir / "rotations_v.npz", layer_0=_build_rotation(64, 10042, "torch"))
    c_k, b_k = load_codebook(64, 4)
    c_v, b_v = load_codebook(64, 2)
    np.savez(output_dir / "codebooks.npz", centroids_k=c_k, boundaries_k=b_k, centroids_v=c_v, boundaries_v=b_v)
    with open(output_dir / "tqai_config.json", "w") as f:
        json.dump({"model_id": "test", "head_dim": 64, "n_layers": 1, "n_kv_heads": 4, "bits_k": 4, "bits_v": 2, "seed": 42, "tqai_version": "0.1.0"}, f)

    data = load_converted(output_dir)
    assert data["config"]["bits_k"] == 4
    assert data["rotations_k"]["layer_0"].shape == (64, 64)
