# tqai

[![PyPI version](https://img.shields.io/pypi/v/tqai)](https://pypi.org/project/tqai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/tqai)](https://pypi.org/project/tqai/)

TurboQuant KV cache compression for local LLM inference.

Compresses the KV cache to ~3 bits per channel with **80%+ memory savings** and zero perplexity change on 8B+ models. Supports both PyTorch (CPU/CUDA) and MLX (Apple Silicon).

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

---

## Installation

```bash
# PyPI
pip install tqai

# With PyTorch backend
pip install tqai[torch]

# With MLX backend (Apple Silicon)
pip install tqai[mlx]

# Global CLI install (no venv management)
pipx install tqai
pipx inject tqai mlx mlx-lm   # add MLX backend
pipx inject tqai torch         # or PyTorch backend
```

---

## Quick Start

### HuggingFace Transformers (PyTorch)

```python
import tqai
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# One line to enable KV cache compression
cache = tqai.patch(model, bits_k=4, bits_v=2)

inputs = tokenizer("Explain quantum entanglement:", return_tensors="pt")
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### MLX (Apple Silicon)

```python
import tqai
import mlx_lm

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-8bit")

# One line to enable KV cache compression
tqai.patch(model, bits_k=4, bits_v=2, backend="mlx")

response = mlx_lm.generate(model, tokenizer, prompt="Explain quantum entanglement:", max_tokens=200)
print(response)

# Restore original behaviour when done
tqai.unpatch(model)
```

---

## Benchmark Results

All results measured on Apple Silicon (MLX). Full data in [`benchmarks/results/`](benchmarks/results/).

### Perplexity — zero change across every model and config

| Model | Baseline PPL | + tqai K4/V2 | + tqai K3/V2 | Δppl |
|-------|-------------|--------------|--------------|------|
| Qwen2.5-0.5B bf16 | 4.34 | 4.34 | 4.34 | **0.00** |
| Qwen2.5-3B bf16 | 2.49 | 2.49 | 2.49 | **0.00** |
| Llama-3.1-8B Q4 | 2.95 | 2.95 | 2.95 | **0.00** |
| Qwen2.5-7B Q8 | 2.40 | 2.40 | 2.40 | **0.00** |
| Qwen2.5-14B Q4 | 2.22 | 2.22 | 2.22 | **0.00** |
| Gemma 4 E4B Q4 | 126.94 | 126.94 | — | **0.00** |

Δppl = 0.00 across **all models and compression configs** tested (6 models, 12 quantization variants).

### Throughput (MLX, v0.3.1 — fused Metal kernels + incremental cache)

| Model | Baseline | kv-only | Retention | vs v0.2 |
|-------|---------|---------|-----------|---------|
| Qwen2.5-0.5B bf16 | 326 tok/s | **118 tok/s** | **36%** | was 10% |
| Qwen2.5-3B bf16 | 71 tok/s | **66 tok/s** | **93%** | was 26% |
| Llama-3.1-8B Q4 | 102 tok/s | **85 tok/s** | **84%** | was 22% |
| **Qwen2.5-7B Q8** | 63 tok/s | **59 tok/s** | **93%** | was 38% |
| Qwen2.5-14B Q4 | 56 tok/s | **50 tok/s** | **89%** | was 25% |
| Gemma 4 E4B Q4 | 113 tok/s | **47 tok/s** | **41%** | new |

v0.3.1 eliminated the O(n²) per-token reconstruction overhead via an incremental dequantized buffer ([KIVI](https://arxiv.org/abs/2402.02750)-inspired). Fused Metal kernels (`mx.fast.metal_kernel`) handle quantize/dequantize in single GPU dispatches. Larger models (3B+) now retain **84–93%** of baseline throughput.

---

## Compression Configs

### KV Cache

| Config | Avg Bits | Memory Saved | Recommended For |
|--------|----------|--------------|-----------------|
| `bits_k=4, bits_v=2` | 3.0 | **80%** | Production — best quality/compression balance |
| `bits_k=3, bits_v=2` | 2.5 | **84%** | Extended context windows |
| `bits_k=4, bits_v=3` | 3.5 | **78%** | Quality-sensitive applications |

### Named Configs (CLI)

| Config | KV | Hidden | FFN | Use Case |
|--------|----|--------|-----|----------|
| `kv-only` | K4/V2 | — | — | KV memory savings only |
| `kv+hidden8` | K4/V2 | 8-bit | — | KV + hidden state compression |
| `kv+hidden6` | K4/V2 | 6-bit | — | More aggressive hidden compression |
| `kv+ffn8` | K4/V2 | — | 8-bit | KV + FFN activation compression |
| `all8` | K4/V2 | 8-bit | 8-bit | Full compression at 8-bit |
| `all6` | K4/V2 | 6-bit | 6-bit | Full compression at 6-bit |
| `aggressive` | K3/V2 | 6-bit | 6-bit | Maximum compression |

---

## How It Works

tqai implements PolarQuant — the core of TurboQuant Stage 1 — via three steps applied to each KV vector at generation time:

1. **Random orthogonal rotation** — Rotates KV vectors by a fixed Haar-distributed matrix to spread information uniformly across all coordinates
2. **Lloyd-Max scalar quantization** — Quantizes each coordinate independently using precomputed optimal codebooks derived from the known post-rotation distribution
3. **Norm preservation** — Stores the vector norm separately in FP16 for lossless magnitude reconstruction

No training, calibration, or model-specific tuning required. Fully data-oblivious — the same codebooks work for any model.

### Cache Strategies (v0.3.1)

tqai supports three cache reconstruction strategies to balance speed and quality:

| Strategy | Per-token cost | Quality | Use case |
|----------|---------------|---------|----------|
| `incremental` (default) | O(1) | Same as full | Production — 2–3x faster than v0.2 |
| `residual` | O(1) | Better (recent tokens exact) | Quality-sensitive, long context |
| `full` | O(n) | Baseline | Debugging, compatibility |

```python
# Use residual strategy — last 128 tokens kept uncompressed (KIVI-style)
tqai.patch(model, bits_k=4, bits_v=2, cache_strategy="residual", residual_window=128)
```

### Codebook Solvers (v0.3.1)

Beyond the default Lloyd-Max solver, tqai offers evolutionary and fuzzy codebook optimizers for build-time codebook generation:

- **CMA-ES** ([arXiv:1710.05311](https://arxiv.org/abs/1710.05311)) — Evolutionary refinement of Lloyd-Max codebooks, ~0.5% MSE improvement
- **Fuzzy C-means** ([arXiv:1908.05033](https://arxiv.org/abs/1908.05033)) — Soft assignment with temperature annealing
- **Attention-aware objective** ([arXiv:2402.14866](https://arxiv.org/abs/2402.14866)) — Optimizes codebooks to preserve softmax attention scores rather than raw MSE

### QJL Stage 2 (opt-in)

tqai optionally implements QJL (Johnson-Lindenstrauss residual sketch), which corrects the systematic inner-product bias left by Stage 1:

```python
cache = tqai.patch(model, bits_k=4, bits_v=2, use_qjl=True, qjl_sketch_size=64)
```

QJL trades bias reduction for added variance. For softmax-based attention, variance typically dominates — this is why QJL is **off by default**. Enable it for very low bit-widths, non-softmax attention, or research use.

---

## CLI

```bash
# Show environment and library info
tqai info

# Quantization accuracy benchmark
tqai benchmark
tqai benchmark --bits-k 3 --bits-v 2 --head-dim 128

# Generate text with compression
tqai run "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --backend torch
tqai run "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit --config aggressive

# Run with QJL Stage 2
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --use-qjl

# Compare baseline vs compressed side by side
tqai compare "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit

# Pre-convert a model for faster startup
tqai convert --model mlx-community/Qwen2.5-7B-Instruct-8bit --output ./qwen7b-tqai/

# Baseline (no compression)
tqai run "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit --no-tqai
```

---

## Advanced Options

```python
cache = tqai.patch(
    model,
    bits_k=4,              # Bits per key coordinate (2–8)
    bits_v=2,              # Bits per value coordinate (2–8)
    sink_tokens=4,         # Keep first N tokens uncompressed (attention sinks)
    backend="torch",       # Force backend: "torch" or "mlx"
    device="cuda",         # PyTorch device (ignored for MLX)
    use_qjl=False,         # Enable QJL Stage 2 residual correction (research)
    qjl_sketch_size=64,    # JL sketch dimension (tradeoff: quality vs memory)
    cache_strategy="auto", # "auto" (incremental), "residual", or "full"
    residual_window=128,   # Recent tokens kept uncompressed (residual strategy)
)
```

---

## Running Tests

```bash
# Install dev dependencies
pip install tqai[dev]

# Unit + accuracy tests (~293 tests, <40s)
pytest tests/ --ignore=tests/test_e2e_models.py --ignore=tests/test_e2e_large_models.py

# End-to-end with real models (requires model downloads)
pytest tests/test_e2e_models.py -v -s

# Large model E2E (7B–14B, requires ~20GB disk)
pytest tests/test_e2e_large_models.py -v -s
```

---

## Project Structure

```
src/tqai/
├── __init__.py          # patch(), unpatch(), TurboQuantConfig
├── config.py            # Configuration dataclass
├── quantizer.py         # PolarQuantizer (core algorithm + QJL Stage 2)
├── kernels/             # Fused Metal GPU kernels (quantize + dequantize)
├── hooks.py             # Forward-pass activation compression hooks
├── module_utils.py      # Transformer layer inspection utilities
├── backend/             # PyTorch + MLX abstraction layer
├── codebook/            # Codebook solvers (Lloyd-Max, CMA-ES, fuzzy) + precomputed data
└── cache/               # HuggingFace DynamicCache + mlx-lm KVCache integrations

benchmarks/
├── benchmark_forward.py # KV + activation compression throughput benchmark
├── benchmark_metal.py   # Metal kernel vs Python path microbenchmark
├── eval_perplexity.py   # Perplexity evaluation helper
└── results/             # Benchmark JSON results + FINDINGS.md
```

---

## Paper

This library implements the TurboQuant algorithm from Google Research:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
> ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

Related work:
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Random rotation + polar coordinate quantization (the core of tqai)
- [QJL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) (AAAI 2025) — Quantized Johnson-Lindenstrauss residual correction (available in tqai as `use_qjl=True`)
- [KIVI](https://arxiv.org/abs/2402.02750) (ICML 2024) — Residual buffer strategy for KV cache compression
- [KVQuant](https://arxiv.org/abs/2401.18079) (NeurIPS 2024) — Fused dequant-attention kernel design
- [APTQ](https://arxiv.org/abs/2402.14866) (2024) — Attention-aware post-training quantization (attention-aware codebook objective)
- [DSQ](https://arxiv.org/abs/1908.05033) (2019) — Differentiable soft quantization (fuzzy codebook solver)
- [IDE-LBG](https://arxiv.org/abs/1710.05311) (2017) — Evolutionary codebook optimization (CMA-ES solver)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits require a DCO sign-off (`git commit -s`).

## License

MIT
