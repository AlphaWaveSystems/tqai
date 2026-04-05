"""Offline rotation baking — fuse PolarQuant rotation into model weights.

Eliminates the per-token matrix multiply (K @ R^T) from the quantization
hot path by baking the rotation matrices R_k and R_v into the model's
Q/K/V/O projection weights once at export time.

Weight transformations per layer (GQA-aware):
  W_K_baked = R_k @ W_K    (keys emerge pre-rotated)
  W_V_baked = R_v @ W_V    (values emerge pre-rotated)
  W_Q_baked = R_k @ W_Q    (Q pre-rotated; Q@K score unchanged: QR^T·RK^T=QK^T)
  W_O_baked = W_O @ R_v    (un-rotates attention output before residual add)

After baking, load the saved model and call ``tqai.patch()`` — the
``tqai_bake_config.json`` in the output directory is auto-detected and
sets ``pre_rotated=True``, skipping rotation at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _build_rotation_np(head_dim: int, seed: int) -> np.ndarray:
    """Build Haar-distributed orthogonal matrix, return as float32 numpy array.

    Must produce the same matrix as PolarQuantizer._build_rotation_matrix().
    """
    from tqai.backend import get_backend

    ops = get_backend("torch")
    G = ops.randn((head_dim, head_dim), seed=seed)
    Q, R = ops.qr(G)
    diag_sign = np.sign(np.diag(ops.to_numpy(R))).astype(np.float32)
    diag_sign[diag_sign == 0] = 1.0
    Q_np = ops.to_numpy(Q)
    return (Q_np * diag_sign[np.newaxis, :]).astype(np.float32)


def _apply_rotation_to_proj(weight: Any, R: np.ndarray, n_heads: int, head_dim: int) -> Any:
    """Apply R to the output (head) dimension of a Q/K/V projection weight.

    weight shape: [n_heads * head_dim, in_features]
    R shape: [head_dim, head_dim]

    Returns W_baked = (R @ W reshaped per-head).reshape_back
    """
    import torch

    w_np = weight.detach().float().cpu().numpy()
    in_features = w_np.shape[1]
    # reshape to [n_heads, head_dim, in_features]
    w_shaped = w_np.reshape(n_heads, head_dim, in_features)
    # R @ w_shaped: [head_dim, head_dim] @ [n_heads, head_dim, in_features]
    # broadcast: for each head h: R @ w_shaped[h]  → [head_dim, in_features]
    w_baked = np.einsum("ij,hjk->hik", R, w_shaped)
    w_baked = w_baked.reshape(n_heads * head_dim, in_features).astype(np.float32)
    return torch.from_numpy(w_baked).to(dtype=weight.dtype, device=weight.device)


def _apply_rotation_to_out_proj(weight: Any, R: np.ndarray, n_heads: int, head_dim: int) -> Any:
    """Apply R to the input (head) dimension of the O projection weight.

    weight shape: [out_features, n_heads * head_dim]
    W_O_baked = W_O @ R (per-head slice of input dimension)

    Derivation: attn_out_rotated = attn_out @ R^T (each head V was rotated by R^T).
    To recover: W_O_baked[:, h*d:(h+1)*d] = W_O[:, h*d:(h+1)*d] @ R
    """
    import torch

    w_np = weight.detach().float().cpu().numpy()
    out_features = w_np.shape[0]
    # reshape to [out_features, n_heads, head_dim]
    w_shaped = w_np.reshape(out_features, n_heads, head_dim)
    # w_shaped @ R.T: [..., head_dim] @ [head_dim, head_dim]
    # einsum "ohd,ed->ohe" computes W_O_h @ R.T (R.T[d,e] = R[e,d])
    w_baked = np.einsum("ohd,ed->ohe", w_shaped, R)
    w_baked = w_baked.reshape(out_features, n_heads * head_dim).astype(np.float32)
    return torch.from_numpy(w_baked).to(dtype=weight.dtype, device=weight.device)


def bake_rotation_into_model(
    model,
    bits_k: int = 4,
    bits_v: int = 2,
    seed: int = 42,
) -> None:
    """Bake rotation matrices into Q/K/V/O projection weights in-place.

    Modifies model weights directly — call this before saving the model.
    The seed convention matches the KV cache quantizer:
      K rotation seed for layer i: seed + i
      V rotation seed for layer i: seed + i + 10000

    Args:
        model: HuggingFace model (must have transformer layer structure).
        bits_k: Key bits (stored in bake config for auto-detection).
        bits_v: Value bits (stored in bake config for auto-detection).
        seed: RNG seed matching the tqai patch seed (default 42).
    """
    from tqai.module_utils import is_attention, iter_transformer_layers

    layers = list(iter_transformer_layers(model))
    if not layers:
        raise ValueError("No transformer layers found. Is this a supported architecture?")

    print(f"Baking rotation matrices into {len(layers)} transformer layers...")
    baked_count = 0

    for layer_idx, (layer_name, layer) in enumerate(layers):
        R_k = _build_rotation_np(
            head_dim=_get_head_dim(layer),
            seed=seed + layer_idx,
        )
        R_v = _build_rotation_np(
            head_dim=_get_head_dim(layer),
            seed=seed + layer_idx + 10000,
        )

        for _name, module in layer.named_modules():
            if not is_attention(module):
                continue

            head_dim = _get_head_dim_from_proj(module)
            if head_dim is None:
                continue

            n_kv_heads = _count_kv_heads(module, head_dim)
            n_q_heads = _count_q_heads(module, head_dim)

            # Bake K and Q with R_k
            if hasattr(module, "k_proj"):
                module.k_proj.weight.data = _apply_rotation_to_proj(
                    module.k_proj.weight, R_k, n_kv_heads, head_dim
                )
            if hasattr(module, "q_proj"):
                module.q_proj.weight.data = _apply_rotation_to_proj(
                    module.q_proj.weight, R_k, n_q_heads, head_dim
                )

            # Bake V with R_v
            if hasattr(module, "v_proj"):
                module.v_proj.weight.data = _apply_rotation_to_proj(
                    module.v_proj.weight, R_v, n_kv_heads, head_dim
                )

            # Bake O projection: W_O @ R_v (un-rotates before residual add)
            if hasattr(module, "o_proj"):
                module.o_proj.weight.data = _apply_rotation_to_out_proj(
                    module.o_proj.weight, R_v, n_q_heads, head_dim
                )

            baked_count += 1
            break  # one attention module per layer

    print(f"Baked {baked_count} attention blocks.")


def save_baked_model(
    model,
    tokenizer,
    output_dir: str | Path,
    base_model_id: str,
    bits_k: int = 4,
    bits_v: int = 2,
    seed: int = 42,
) -> Path:
    """Bake rotations into model weights and save as HuggingFace safetensors.

    Args:
        model: HuggingFace model to bake.
        tokenizer: Matching tokenizer (saved alongside model).
        output_dir: Directory to write the baked model.
        base_model_id: Original model ID (recorded in bake config).
        bits_k: Key quantization bits (default 4).
        bits_v: Value quantization bits (default 2).
        seed: RNG seed (default 42).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baking model: {base_model_id}")
    bake_rotation_into_model(model, bits_k=bits_k, bits_v=bits_v, seed=seed)

    print(f"Saving to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    bake_config = {
        "tqai_baked": True,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "seed": seed,
        "base_model": base_model_id,
    }
    (output_dir / "tqai_bake_config.json").write_text(json.dumps(bake_config, indent=2))
    print("Wrote tqai_bake_config.json")

    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_head_dim(layer) -> int:
    """Get head_dim from a transformer layer (via config or projection weight)."""
    parent = getattr(layer, "_modules", {})
    for _name, mod in parent.items():
        if hasattr(mod, "head_dim"):
            return mod.head_dim

    # Fall back to weight inspection
    for _name, mod in layer.named_modules():
        if hasattr(mod, "k_proj") and hasattr(mod.k_proj, "weight"):
            out = mod.k_proj.weight.shape[0]
            n_kv = _count_kv_heads(mod, None)
            if n_kv and out % n_kv == 0:
                return out // n_kv
        if hasattr(mod, "head_dim"):
            return mod.head_dim

    # Last resort: scan for q_proj or k_proj weight shape
    for _name, mod in layer.named_modules():
        if hasattr(mod, "q_proj") and hasattr(mod.q_proj, "weight"):
            # head_dim is typically 64 or 128; find from head count via config
            w = mod.q_proj.weight
            # Try to get from parent config
            config = getattr(layer, "config", None)
            if config is not None:
                hd = getattr(config, "head_dim", None)
                if hd:
                    return hd
                nh = getattr(config, "num_attention_heads", None)
                hs = getattr(config, "hidden_size", None)
                if nh and hs:
                    return hs // nh
            return w.shape[0]  # fallback — may be wrong for GQA

    raise ValueError(f"Cannot determine head_dim for layer {layer}")


def _get_head_dim_from_proj(module) -> int | None:
    """Get head_dim from an attention module's projection weights."""
    if hasattr(module, "head_dim"):
        return module.head_dim
    # Check config stored on module
    config = getattr(module, "config", None)
    if config is not None:
        hd = getattr(config, "head_dim", None)
        if hd:
            return hd

    # Infer from k_proj and num_key_value_heads
    if hasattr(module, "k_proj") and hasattr(module.k_proj, "weight"):
        n_kv = getattr(module, "num_key_value_heads", None)
        if n_kv:
            return module.k_proj.weight.shape[0] // n_kv
        n_heads = getattr(module, "num_heads", None)
        if n_heads:
            return module.k_proj.weight.shape[0] // n_heads

    if hasattr(module, "q_proj") and hasattr(module.q_proj, "weight"):
        n_heads = getattr(module, "num_heads", None)
        if n_heads:
            return module.q_proj.weight.shape[0] // n_heads

    return None


def _count_kv_heads(module, head_dim: int | None) -> int:
    """Count number of KV heads."""
    n_kv = getattr(module, "num_key_value_heads", None)
    if n_kv:
        return n_kv
    n = getattr(module, "num_heads", None)
    if n:
        return n
    if head_dim and hasattr(module, "k_proj"):
        return module.k_proj.weight.shape[0] // head_dim
    return 1


def _count_q_heads(module, head_dim: int | None) -> int:
    """Count number of Q heads."""
    n = getattr(module, "num_heads", None)
    if n:
        return n
    if head_dim and hasattr(module, "q_proj"):
        return module.q_proj.weight.shape[0] // head_dim
    return 1
