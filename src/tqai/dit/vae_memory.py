"""VAE memory optimization for video diffusion pipelines.

The VAE decode step in WAN 2.2 / LTX-2 decodes all frames at once,
causing memory spikes to 100GB+ on a 480p 81-frame video.  This module
enables tiled and sliced decoding to keep peak memory under control.

Usage::

    from tqai.dit.vae_memory import optimize_vae_memory

    optimize_vae_memory(pipeline)  # enables tiling + slicing
    video = pipeline("prompt", num_frames=81)
"""

from __future__ import annotations

from typing import Any


def optimize_vae_memory(
    pipeline: Any,
    enable_tiling: bool = True,
    enable_slicing: bool = True,
    tile_height: int = 128,
    tile_width: int = 128,
    tile_stride_height: int = 96,
    tile_stride_width: int = 96,
) -> None:
    """Enable memory-efficient VAE decoding on a diffusers pipeline.

    Without this, the VAE decode of an 81-frame 480p video allocates
    ~100GB+ of intermediates.  With tiling + slicing, peak VAE memory
    drops to ~8-15GB.

    Args:
        pipeline: A diffusers pipeline with a ``.vae`` attribute.
        enable_tiling: Split spatial dimensions into overlapping tiles.
        enable_slicing: Process batch dimension in slices.
        tile_height: Minimum tile height in pixel space (default 128).
        tile_width: Minimum tile width in pixel space (default 128).
        tile_stride_height: Tile stride (overlap = tile - stride).
        tile_stride_width: Tile stride for width.
    """
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return

    if enable_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling(
            tile_sample_min_height=tile_height,
            tile_sample_min_width=tile_width,
            tile_sample_stride_height=tile_stride_height,
            tile_sample_stride_width=tile_stride_width,
        )

    if enable_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()


def estimate_vae_memory(
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    channels: int = 16,
    spatial_ratio: int = 8,
    temporal_ratio: int = 4,
) -> dict:
    """Estimate VAE decode peak memory for a given video shape.

    Returns dict with memory estimates in GB for both tiled and non-tiled.
    """
    # Latent shape
    lat_t = (num_frames - 1) // temporal_ratio + 1
    lat_h = height // spatial_ratio
    lat_w = width // spatial_ratio

    # Non-tiled: full intermediate at each decoder stage
    # Stage 1: [1, 512, lat_t, lat_h, lat_w] float32
    stage1 = 512 * lat_t * lat_h * lat_w * 4
    # Stage 2: [1, 256, lat_t*2, lat_h*2, lat_w*2]
    stage2 = 256 * lat_t * 2 * lat_h * 2 * lat_w * 2 * 4
    # Stage 3: [1, 128, num_frames, lat_h*4, lat_w*4]
    stage3 = 128 * num_frames * (lat_h * 4) * (lat_w * 4) * 4
    # Output: [1, 3, num_frames, height, width]
    output = 3 * num_frames * height * width * 4

    non_tiled_gb = (stage1 + stage2 + stage3 + output) / 1e9

    # Tiled: only process one tile at a time
    tile_h, tile_w = 128, 128
    tile_lat_h = tile_h // spatial_ratio
    tile_lat_w = tile_w // spatial_ratio
    tile_stage1 = 512 * lat_t * tile_lat_h * tile_lat_w * 4
    tile_stage2 = 256 * lat_t * 2 * tile_lat_h * 2 * tile_lat_w * 2 * 4
    tile_stage3 = 128 * num_frames * (tile_lat_h * 4) * (tile_lat_w * 4) * 4
    tiled_gb = (tile_stage1 + tile_stage2 + tile_stage3 + output) / 1e9

    return {
        "non_tiled_peak_gb": round(non_tiled_gb, 1),
        "tiled_peak_gb": round(tiled_gb, 1),
        "savings_gb": round(non_tiled_gb - tiled_gb, 1),
        "savings_pct": round((1 - tiled_gb / non_tiled_gb) * 100, 0),
        "output_gb": round(output / 1e9, 2),
    }
