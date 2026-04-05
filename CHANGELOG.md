# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
