"""CSA + HCA block-pool compression strategy.

Wraps :mod:`tqai.csa_hca` in the :class:`CompressionStrategy` protocol so it
can be selected via ``pipeline={"strategy": "csa_hca", ...}``.

Compression flow:
    1. Pool the entry's KV tensor along the sequence axis into CSA blocks
       (mild, e.g. 4x) and HCA blocks (aggressive, e.g. 128x).
    2. Quantize both block sets through the provided quantizer.
    3. Stash both compressed views plus enough metadata to reconstruct.

Decompression reverses the quantization on each view and broadcasts each
centroid back over its source positions.  This is intentionally lossy: the
strategy's job is to provide a compressed representation suitable for fused
attention, not a token-perfect reconstruction.

References:
    - DeepSeek-V4 paper (April 2026), interleaved CSA/HCA design.
    - Math primitives: :mod:`tqai.csa_hca`.
"""

from __future__ import annotations

from typing import Any

import torch

from tqai.csa_hca import block_pool
from tqai.pipeline.base import ScoredEntry


class CSAHCAStrategy:
    """CSA + HCA block-pool compression.

    Implements the :class:`~tqai.pipeline.base.CompressionStrategy` protocol.

    Args:
        csa_block_size: Sequence-axis pool size for the CSA view (mild).
        hca_block_size: Sequence-axis pool size for the HCA view (aggressive).
            Must be ``>= csa_block_size``.
        broadcast_on_decompress: If True (default), each block centroid is
            replicated back over its source positions on decompress so the
            output shape matches the original sequence length.  If False,
            the compressed (block-resolution) tensors are returned unchanged.
    """

    name = "csa_hca"

    def __init__(
        self,
        csa_block_size: int = 4,
        hca_block_size: int = 128,
        broadcast_on_decompress: bool = True,
    ):
        if csa_block_size < 1:
            raise ValueError(f"csa_block_size must be >= 1, got {csa_block_size}")
        if hca_block_size < csa_block_size:
            raise ValueError(
                f"hca_block_size ({hca_block_size}) must be >= csa_block_size "
                f"({csa_block_size})"
            )
        self._csa_m = csa_block_size
        self._hca_m = hca_block_size
        self._broadcast = broadcast_on_decompress

    def compress(
        self,
        entry: Any,
        quantizer: Any,
        prev_state: dict | None = None,
    ) -> tuple[Any, dict]:
        state = dict(prev_state) if prev_state else {}

        if isinstance(entry, list) and entry and isinstance(entry[0], ScoredEntry):
            data = entry[0].data
        else:
            data = entry

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"CSAHCAStrategy expects a torch.Tensor entry; got {type(data).__name__}"
            )
        if data.ndim < 2:
            raise ValueError(
                f"Entry must have at least 2 dims (..., S, D); got shape {tuple(data.shape)}"
            )

        S = data.shape[-2]
        csa_blocks = block_pool(data, self._csa_m)  # (..., N_csa, D)
        hca_blocks = block_pool(data, self._hca_m)  # (..., N_hca, D)

        csa_q = quantizer.quantize(csa_blocks)
        hca_q = quantizer.quantize(hca_blocks)

        stats = state.get("csa_hca_stats", {"calls": 0, "total_seq_len": 0})
        stats["calls"] += 1
        stats["total_seq_len"] += S
        state["csa_hca_stats"] = stats
        state["last_seq_len"] = S
        state["last_n_csa"] = int(csa_blocks.shape[-2])
        state["last_n_hca"] = int(hca_blocks.shape[-2])

        compressed = (
            "csa_hca",
            {
                "csa": csa_q,
                "hca": hca_q,
                "seq_len": S,
                "csa_block_size": self._csa_m,
                "hca_block_size": self._hca_m,
                "shape": tuple(data.shape),
                "dtype": data.dtype,
            },
        )
        return compressed, state

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any:
        tag, meta = compressed
        if tag != "csa_hca":
            raise ValueError(f"Unknown compressed tag: {tag!r}")

        csa_recon = _dequantize(quantizer, meta["csa"])  # (..., N_csa, D)
        hca_recon = _dequantize(quantizer, meta["hca"])  # (..., N_hca, D)

        if not self._broadcast:
            return {"csa": csa_recon, "hca": hca_recon, "meta": meta}

        # Broadcast block-resolution back to per-token resolution.  Use the
        # CSA view (less lossy) as the primary reconstruction.
        S = meta["seq_len"]
        m_csa = meta["csa_block_size"]
        return _broadcast_blocks(csa_recon, m_csa, S)


def _dequantize(quantizer: Any, packed: tuple) -> torch.Tensor:
    """Dispatch to ``quantizer.dequantize`` accepting both 2- and 3-tuple forms."""
    if isinstance(packed, tuple) and len(packed) >= 2:
        indices, norms = packed[0], packed[1]
        qjl = packed[2] if len(packed) > 2 else None
        return quantizer.dequantize(indices, norms, qjl)
    raise ValueError(f"Unexpected quantized payload type: {type(packed)}")


def _broadcast_blocks(blocks: torch.Tensor, block_size: int, target_len: int) -> torch.Tensor:
    """Replicate each block centroid over its source positions.

    Args:
        blocks: ``(..., N, D)`` block centroids.
        block_size: Pool size used to produce ``blocks``.
        target_len: Original sequence length to broadcast back to.

    Returns:
        ``(..., target_len, D)`` reconstruction.
    """
    N = blocks.shape[-2]
    expected_full = N * block_size
    # Trailing partial block (if any) replicates to fill the remainder.
    repeated = blocks.repeat_interleave(block_size, dim=-2)  # (..., N * block_size, D)
    if repeated.shape[-2] >= target_len:
        return repeated[..., :target_len, :]
    # ``repeated`` is shorter than target only when block_pool produced fewer
    # blocks than expected (rounding); pad with the last block.
    deficit = target_len - repeated.shape[-2]
    last = blocks[..., -1:, :].expand(*blocks.shape[:-2], deficit, blocks.shape[-1])
    return torch.cat([repeated, last], dim=-2)
