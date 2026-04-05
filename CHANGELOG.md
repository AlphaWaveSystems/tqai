# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-04-05

### Added

- **Incremental cache buffer** — O(1) per-token dequantization instead of O(n) full reconstruction. Default strategy (`cache_strategy="auto"`). Inspired by KIVI (arXiv:2402.02750).
- **Residual cache strategy** — KIVI-style: last `residual_window` tokens kept uncompressed, older tokens compressed into incremental buffer. Zero quantization error on recent tokens. (`cache_strategy="residual"`)
- **Fused Metal GPU kernels** — Quantize and dequantize fused into single GPU dispatches via `mx.fast.metal_kernel()` (MLX >= 0.16). Eliminates Python dispatch overhead for the codebook search step.
- **CMA-ES codebook solver** — Evolutionary optimization of codebook centroids using Covariance Matrix Adaptation (arXiv:1710.05311). ~0.5% MSE improvement over Lloyd-Max. Requires `cma>=3.0` (optional `codegen` dependency).
- **Fuzzy C-means codebook solver** — Soft assignment with temperature annealing (arXiv:1908.05033). Smoother optimization landscape, avoids local optima.
- **Attention-aware codebook objective** — Optimizes codebooks to preserve softmax attention scores rather than raw MSE (inspired by APTQ, arXiv:2402.14866).
- **Codebook solver dispatcher** — Unified `solve_codebook()` entry point supporting `lloyd_max`, `cmaes`, `fuzzy`, or `auto` selection.
- **Gemma 4 support** — Updated model detection to handle composite models (Gemma 4's `language_model` wrapper).
- Metal kernel tests (26 tests) and cache strategy tests

### Changed

- `TurboQuantConfig` adds `cache_strategy` and `residual_window` fields
- `patch()` accepts `cache_strategy` and `residual_window` parameters
- `MLXOps` adds `quantize_fused()` and `dequantize_fused()` methods
- `PolarQuantizer` auto-dispatches to Metal kernels when available
- Codebook registry runtime fallback uses solver dispatcher
- Test suite expanded to 293 tests (up from 264)
- Version bumped to 0.3.1

### Benchmark results (MLX, Apple Silicon)

**Throughput improvement (v0.2 → v0.3.1):**

| Model | v0.2 kv-only | v0.3.1 kv-only | Speedup |
|-------|-------------|----------------|---------|
| Qwen2.5-0.5B bf16 | 33 tok/s (10%) | 118 tok/s (36%) | **3.6x** |
| Qwen2.5-3B bf16 | 21 tok/s (30%) | 66 tok/s (93%) | **3.1x** |
| Llama-3.1-8B Q4 | 31 tok/s (30%) | 85 tok/s (84%) | **2.7x** |
| Qwen2.5-7B Q8 | 26 tok/s (42%) | 59 tok/s (93%) | **2.3x** |
| Qwen2.5-14B Q4 | 20 tok/s (35%) | 50 tok/s (89%) | **2.5x** |
| Gemma 4 E4B Q4 | — | 47 tok/s (41%) | new |

Perplexity: Δppl = 0.00 across all models and configs (unchanged from v0.2).

## [0.2.0] - 2026-04-05

### Added

- Forward-pass activation compression: `compress_hidden` and `compress_ffn` hooks for PyTorch models
- 6-bit and 8-bit Lloyd-Max codebooks for all standard head dimensions (64, 96, 128, 256)
- Extended compression configs: `kv-only`, `kv+hidden8`, `kv+hidden6`, `kv+ffn8`, `all8`, `all6`, `aggressive`
- `tqai benchmark --all-configs` flag to sweep all compression configs in one run
- `--config` flag on `tqai run` and `tqai compare` for named compression presets

### Changed

- Default codebook set now includes 6-bit and 8-bit entries (previously 2/3/4-bit only)
- Test suite expanded to 264 tests (up from 179)

### Benchmark results (MLX, Apple Silicon)

- Δppl = 0.00 across all 5 models × 8 configs (40 runs)
- Qwen2.5-7B-Q8: 37% throughput retention (best — compute-heavy model absorbs Python overhead)
- Qwen2.5-14B-Q4 `aggressive`: 38% retention — adding hidden/FFN compression is free at 14B+
- Throughput overhead is fully Python-level; Metal kernel planned for v0.3 to close the gap

## [0.1.0] - 2026-04-04

### Added

- Core PolarQuantizer implementing TurboQuant Stage 1 (random rotation + Lloyd-Max scalar quantization)
- Multi-backend support: PyTorch (CPU/CUDA) and MLX (Apple Silicon)
- HuggingFace Transformers integration via `DynamicCache` subclass
- mlx-lm integration via `KVCache` replacement
- Precomputed Lloyd-Max codebooks for head dimensions 64, 96, 128, 256 at 2/3/4 bits
- Asymmetric K/V bit allocation (e.g., K4/V2 for 80% memory savings)
- Sink token support (keep first N tokens uncompressed)
- CLI tool with commands: `info`, `benchmark`, `run`, `compare`, `convert`
- Offline model conversion (`tqai convert`) for faster startup
- Comprehensive test suite (179 tests) covering accuracy, edge cases, and cross-backend consistency
- Verified quality-neutral on 8B+ parameter models (Llama 3.1 8B, Qwen 14B)
