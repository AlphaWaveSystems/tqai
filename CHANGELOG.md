# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
