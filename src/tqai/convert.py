"""Offline model conversion: precompute rotation matrices and save config.

Precomputing avoids QR decomposition at runtime and bundles everything
needed for a specific model into a single directory.

Output format::

    output_dir/
    ├── tqai_config.json     # Config + model metadata
    ├── rotations_k.npz      # Key rotation matrices per layer
    ├── rotations_v.npz      # Value rotation matrices per layer
    └── codebooks.npz        # Codebook centroids for bits_k and bits_v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tqai.codebook import load_codebook


def detect_model_info(model_id: str, backend: str) -> dict:
    """Load a model and extract head_dim, n_layers, n_kv_heads."""
    if backend == "mlx":
        import mlx_lm

        model, _ = mlx_lm.load(model_id)
        args = model.args
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            head_dim = args.hidden_size // args.num_attention_heads
        n_layers = args.num_hidden_layers
        n_kv_heads = getattr(args, "num_key_value_heads", args.num_attention_heads)
        return {"head_dim": head_dim, "n_layers": n_layers, "n_kv_heads": n_kv_heads}
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        n_layers = config.num_hidden_layers
        n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        return {"head_dim": head_dim, "n_layers": n_layers, "n_kv_heads": n_kv_heads}


def _build_rotation(head_dim: int, seed: int, backend: str) -> np.ndarray:
    """Build a single rotation matrix and return as numpy."""
    from tqai.backend import get_backend

    ops = get_backend(backend)
    G = ops.randn((head_dim, head_dim), seed=seed)
    Q, R = ops.qr(G)
    diag_R = ops.to_numpy(R)
    diag_sign = np.sign(np.diag(diag_R)).astype(np.float32)
    diag_sign[diag_sign == 0] = 1.0
    Q_np = ops.to_numpy(Q)
    return (Q_np * diag_sign[np.newaxis, :]).astype(np.float32)


def convert_model(
    model_id: str,
    output_dir: str | Path,
    bits_k: int = 4,
    bits_v: int = 2,
    seed: int = 42,
    backend: str | None = None,
) -> Path:
    """Precompute and save everything needed for TurboQuant inference.

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Directory to write output files.
        bits_k: Bits per key coordinate.
        bits_v: Bits per value coordinate.
        seed: RNG seed for rotation matrices.
        backend: ``'torch'``, ``'mlx'``, or ``None`` (auto).

    Returns:
        Path to the output directory.
    """
    from tqai.backend import detect_backend

    backend = backend or detect_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detecting model architecture from {model_id}...")
    model_info = detect_model_info(model_id, backend)
    head_dim = model_info["head_dim"]
    n_layers = model_info["n_layers"]
    n_kv_heads = model_info["n_kv_heads"]
    print(f"  head_dim={head_dim}, n_layers={n_layers}, n_kv_heads={n_kv_heads}")

    # Precompute rotation matrices for all layers
    print(f"Precomputing {n_layers * 2} rotation matrices ({head_dim}x{head_dim})...")
    rotations_k = {}
    rotations_v = {}
    for layer_idx in range(n_layers):
        k_seed = seed + layer_idx
        v_seed = seed + layer_idx + 10000
        rotations_k[f"layer_{layer_idx}"] = _build_rotation(head_dim, k_seed, backend)
        rotations_v[f"layer_{layer_idx}"] = _build_rotation(head_dim, v_seed, backend)

    np.savez(output_dir / "rotations_k.npz", **rotations_k)
    np.savez(output_dir / "rotations_v.npz", **rotations_v)

    # Copy codebooks
    print(f"Saving codebooks (K{bits_k}, V{bits_v})...")
    centroids_k, boundaries_k = load_codebook(head_dim, bits_k)
    centroids_v, boundaries_v = load_codebook(head_dim, bits_v)
    np.savez(
        output_dir / "codebooks.npz",
        centroids_k=centroids_k,
        boundaries_k=boundaries_k,
        centroids_v=centroids_v,
        boundaries_v=boundaries_v,
    )

    # Save config
    config = {
        "model_id": model_id,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "seed": seed,
        "tqai_version": "0.1.0",
    }
    with open(output_dir / "tqai_config.json", "w") as f:
        json.dump(config, f, indent=2)

    size_kb = sum(p.stat().st_size for p in output_dir.iterdir()) / 1024
    print(f"\nSaved to {output_dir} ({size_kb:.0f} KB)")
    print("  tqai_config.json")
    print(f"  rotations_k.npz ({n_layers} matrices)")
    print(f"  rotations_v.npz ({n_layers} matrices)")
    print("  codebooks.npz")

    return output_dir


def load_converted(config_path: str | Path) -> dict:
    """Load a pre-converted tqai config directory.

    Returns a dict with keys: config, rotations_k, rotations_v, codebooks.
    """
    config_path = Path(config_path)

    with open(config_path / "tqai_config.json") as f:
        config = json.load(f)

    rotations_k = dict(np.load(config_path / "rotations_k.npz"))
    rotations_v = dict(np.load(config_path / "rotations_v.npz"))
    codebooks = dict(np.load(config_path / "codebooks.npz"))

    return {
        "config": config,
        "rotations_k": rotations_k,
        "rotations_v": rotations_v,
        "codebooks": codebooks,
    }
