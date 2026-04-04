# tqai

[![PyPI version](https://img.shields.io/pypi/v/tqai)](https://pypi.org/project/tqai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/tqai)](https://pypi.org/project/tqai/)

TurboQuant KV cache compression for local LLM inference.

Compresses the KV cache to ~3 bits per channel with **80%+ memory savings** and near-zero quality loss on 8B+ models. Supports both PyTorch (CPU/CUDA) and MLX (Apple Silicon).

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

## Installation

```bash
# Homebrew (macOS)
brew install alphawavesystems/tap/tqai

# PyPI
pip install tqai

# With PyTorch backend
pip install tqai[torch]

# With MLX backend (Apple Silicon)
pip install tqai[mlx]
```

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

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.1-8B-Instruct-4bit")

# One line to enable KV cache compression
tqai.patch(model, bits_k=4, bits_v=2, backend="mlx")

response = mlx_lm.generate(model, tokenizer, prompt="Explain quantum entanglement:", max_tokens=200)
print(response)

# Restore original behaviour when done
tqai.unpatch(model)
```

## Compression Configs

| Config | Avg Bits | Memory Saved | Recommended For |
|--------|----------|--------------|-----------------|
| `bits_k=4, bits_v=2` | 3.0 | **80%** | Production (best quality/compression balance) |
| `bits_k=3, bits_v=2` | 2.5 | **84%** | Extended context windows |
| `bits_k=4, bits_v=3` | 3.5 | **78%** | Quality-sensitive applications |

## How It Works

tqai implements Stage 1 of TurboQuant (PolarQuant):

1. **Random orthogonal rotation** — Rotates KV vectors by a fixed Haar-distributed matrix to spread information across all coordinates
2. **Lloyd-Max scalar quantization** — Quantizes each coordinate independently using precomputed optimal codebooks
3. **Norm preservation** — Stores vector norms separately in FP16

No training, calibration, or model-specific tuning required. The same codebooks work for any model.

## Quality Results

Tested on Apple Silicon with various model sizes:

| Model | Baseline | + tqai K4/V2 | + tqai K3/V2 |
|-------|----------|--------------|--------------|
| Qwen 0.5B | Good | Degraded | Poor |
| Qwen 3B bf16 | Excellent | Good | Degraded |
| Llama 8B Q4 | Excellent | **Excellent** | **Excellent** |
| Qwen 14B Q4 | Excellent | **Excellent** | **Excellent** |

Quality is near-identical to baseline on 8B+ parameter models.

## CLI

tqai includes a command-line tool for quick testing without writing code:

```bash
# Show environment and library info
tqai info

# Run quantization accuracy benchmark
tqai benchmark
tqai benchmark --bits-k 3 --bits-v 2 --head-dim 128

# Generate text with TurboQuant compression
tqai run "Explain gravity" --model mlx-community/Llama-3.1-8B-Instruct-4bit
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --backend torch

# Compare baseline vs compressed output side by side
tqai compare "Explain gravity" --model mlx-community/Llama-3.1-8B-Instruct-4bit

# Pre-convert a model for faster startup
tqai convert --model mlx-community/Llama-3.1-8B-Instruct-4bit --output ./llama-tqai/
tqai run "Explain gravity" --model mlx-community/Llama-3.1-8B-Instruct-4bit --tqai-config ./llama-tqai/

# Run without compression (baseline)
tqai run "Explain gravity" --model mlx-community/Llama-3.1-8B-Instruct-4bit --no-tqai
```

## Advanced Options

```python
cache = tqai.patch(
    model,
    bits_k=4,           # Bits per key coordinate (2, 3, or 4)
    bits_v=2,           # Bits per value coordinate (2, 3, or 4)
    sink_tokens=4,      # Keep first N tokens uncompressed (attention sinks)
    backend="torch",    # Force backend: "torch" or "mlx"
    device="cuda",      # PyTorch device (ignored for MLX)
)
```

## Running Tests

```bash
# Install dev dependencies
pip install tqai[dev]

# Unit + accuracy tests (175 tests, <1s)
pytest tests/ --ignore=tests/test_e2e_models.py --ignore=tests/test_e2e_large_models.py

# End-to-end with real models (requires model downloads)
pytest tests/test_e2e_models.py -v -s
```

## Project Structure

```
src/tqai/
├── __init__.py          # patch(), unpatch(), TurboQuantConfig
├── config.py            # Configuration dataclass
├── quantizer.py         # PolarQuantizer (core algorithm)
├── backend/             # PyTorch + MLX abstraction
├── codebook/            # Lloyd-Max codebooks (precomputed)
└── cache/               # HuggingFace + mlx-lm integrations
```

## Paper

This library implements the TurboQuant algorithm from Google Research:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
> ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

Related work:
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Stage 1 polar coordinate quantization
- [QJL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) (AAAI 2025) — Quantized Johnson-Lindenstrauss transform

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits require a DCO sign-off (`git commit -s`).

## License

MIT
