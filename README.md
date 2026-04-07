# tqai

[![PyPI version](https://img.shields.io/pypi/v/tqai)](https://pypi.org/project/tqai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaWaveSystems/tqai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/tqai)](https://pypi.org/project/tqai/)

TurboQuant KV cache compression for local LLM inference + a composable
pipeline for experimenting with the latest compression papers locally.

Compresses the KV cache to ~3 bits per channel with **byte-identical
output to baseline** on 8B+ models (Δppl = 0.00 across Qwen, Gemma,
Llama). Supports both PyTorch (CPU/CUDA) and MLX (Apple Silicon). v0.4
adds a plugin-based pipeline so any paper — QuantSparse, DiTFastAttn,
BSA, Sheaf, Fisher — can be added as a single file without touching
core code, plus drop-in support for diffusers video pipelines
(WAN 2.2, LTX-2).

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

> **Honest framing on the 80% number.** The K4/V2 quantization produces
> a compressed-storage form that is ~5× smaller than the uncompressed
> bf16 KV tensors. This reduction is real for **wire format / serialized
> KV / future fused dequant-attention kernels**, but **does not reduce
> peak runtime memory on Apple Silicon today** because the v0.3.1
> incremental cache strategy keeps a persistent dequantized buffer at
> full input precision (the design choice that enables O(1) per-token
> decode). Measured peak memory on Qwen 2.5-7B Q8 at 4K–128K context:
> tqai is within ±0.2 GB of baseline at every length. Full benchmark
> and explanation in [`reports/kv_memory_finding.md`](reports/kv_memory_finding.md).
> The actual unlock for runtime memory savings is a fused dequant-attention
> Metal kernel ([KVQuant](https://arxiv.org/abs/2401.18079) approach),
> which is on the v0.5 roadmap.

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

### v0.4 — Pipeline strategy quality (synthetic NMSE, mean across 7 model profiles)

| Config | NMSE | vs Baseline | Use case |
|--------|-----:|------------:|----------|
| baseline | 0.009290 | — | reference |
| palm+tiered | 0.009301 | 0.0% | Adaptive bit allocation |
| **palm+delta** | **0.003759** | **−59.5%** | Inter-step temporal redundancy |
| **snr+delta2** | **0.003746** | **−59.7%** | Best for diffusion (cosine schedule) |
| sheaf+delta2 | 0.003763 | −59.5% | Spatial-temporal token graphs |
| palm+window | 0.018907 | +103.5% | Trades quality for speed |
| skip_layers | 0.009297 | 0.0% | Layer protection (arXiv:2504.10317) |

Delta strategies cut reconstruction distortion in half on synthetic streaming data — validates QuantSparse's [arXiv:2509.23681](https://arxiv.org/abs/2509.23681) second-order residual insight.

### v0.4 — Video generation (WAN 2.2 5B + LTX-2, MLX, 33 frames @ 480p, 15 steps)

| Model | Config | Time | Peak RSS | CFG share | Note |
|-------|-------:|-----:|---------:|----------:|------|
| WAN 2.2 5B | baseline | 172.8s | 34.6 GB | 0% | reference |
| WAN 2.2 5B | tqai_full | 170.5s | 32.2 GB | 50% | CFG hooks fire correctly |
| LTX-2 | baseline | 88.3s | 73.9 GB | 0% | needs MPS float64 fix to run |
| LTX-2 | tqai_full | 90.9s | 74.4 GB | 100% | batched CFG fully shared |

**The actual local win:** without `optimize_vae_memory()`, the WAN 2.2 VAE decoder spikes >100 GB on an 81-frame video and OOMs on a 128 GB Mac. With it, peak stays at 32 GB and 20-second videos fit comfortably. See [`reports/where_tqai_shines.md`](reports/where_tqai_shines.md) for the full picture.

### v0.4 — Long context (chunked attention, MLX)

| Model | Context | Baseline prefill | Chunked prefill | Slowdown | Output match |
|-------|--------:|----------------:|----------------:|---------:|:------------:|
| Qwen 3B bf16 | 8K | 1.1s | 3.3s | 3.0× | ✓ |
| Qwen 3B bf16 | 16K | 2.6s | 12.0s | 4.6× | ✓ |
| Qwen 7B Q8 | 16K | 5.8s | 27.5s | 4.7× | ✓ |
| Qwen 7B Q8 | 32K | 17.4s | 68.3s | 3.9× | ✓ |

**Honest negative result:** Output is bit-identical to baseline (math is correct), but `mx.fast.scaled_dot_product_attention` is a fused Metal kernel that already implements blocked attention internally. A Python-loop chunked path forces a kernel launch + sync per chunk and loses decisively. **Don't enable `chunk_attention=True` on MLX.** The implementation is shipped for CUDA users without FlashAttention, where the win is real.

---

## Compression Configs

### KV Cache

The "compressed-storage saved" column is the size reduction of the
serialized KV cache form (compressed indices vs bf16 tensors). It is
**not** a peak runtime memory reduction — see the honest framing
note at the top of this README and [`reports/kv_memory_finding.md`](reports/kv_memory_finding.md).

| Config | Avg Bits | Compressed-Storage Saved | Recommended For |
|--------|----------|--------------------------|-----------------|
| `bits_k=4, bits_v=2` | 3.0 | **80%** | Production — best quality/storage balance, byte-identical output |
| `bits_k=3, bits_v=2` | 2.5 | **84%** | Smallest serialized cache, still byte-identical on tested models |
| `bits_k=4, bits_v=3` | 3.5 | **78%** | Quality-sensitive applications |

### Named Configs (CLI)

| Config | KV | Hidden | FFN | Use Case |
|--------|----|--------|-----|----------|
| `kv-only` | K4/V2 | — | — | KV cache compression (storage form, quality-preserving) |
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

## v0.4 — Composable Pipeline Architecture

> **Contributors welcome — bring your paper.** The whole point of the v0.4
> restructuring is to make integrating new compression research a one-file
> change. The core (`quantizer.py`, `cache/hf.py`, `cache/mlx.py`,
> `kernels/`) is **frozen and stable** — every recent paper we tracked
> (TurboQuant, QuantSparse, DiTFastAttn, BSA, Sheaf, Copresheaf, Fisher,
> SparseDiT) ships as a plugin in `scorers/`, `strategies/`, `monitors/`,
> or `adapters/` without touching the proven core. If you're publishing or
> reproducing a KV/attention compression paper, the friction to ship it
> here is a single file plus one line of registration. We actively
> encourage PRs that add new papers — see the Paper Playground below for
> the existing references and the "Adding a new paper" example for the
> exact 5-minute workflow.

v0.4 introduces a plugin-based pipeline so adding a new compression paper
costs **one file** instead of refactoring `quantizer.py`/`hf.py`/`mlx.py`.
The default path (`pipeline=None`) is byte-identical to v0.3.1 — zero
overhead, all 293 pre-v0.4 tests pass unchanged.

```
tqai.patch(model, pipeline={"scorer": "...", "strategy": "...", "monitor": "..."})
   │
   ├─ Scorer       — per-token importance (palm, snr, fisher, fisher_static, sheaf, bsa)
   ├─ Strategy     — how to compress  (tiered, delta, delta2, window, cfg_sharing)
   ├─ Monitor      — runtime adjustment (stability, lyapunov)
   ├─ Adapter      — model family      (llm, dit, wan)
   └─ Optimization — GA policy search over pipeline configs
```

### Adding a new paper takes one file

```python
# src/tqai/scorers/my_paper.py
from tqai.pipeline.base import ScoredEntry

class MyPaperScorer:
    name = "my_paper"

    def score(self, x, layer_idx, step=None, context=None):
        # ... compute importance score ...
        return [ScoredEntry(data=x, score=0.7, tier=2, metadata={})]

    def reset(self): ...
```

```python
# src/tqai/scorers/__init__.py — one line
from tqai.scorers.my_paper import MyPaperScorer
register_scorer("my_paper", MyPaperScorer)
```

```bash
tqai plugins  # Verify it's discoverable
tqai run "prompt" -m Qwen/Qwen2.5-7B --scorer my_paper --strategy delta
```

---

## Paper Playground

Each paper from the recent KV/attention compression literature ships as a
single plugin you can mix and match. The implementations are intentionally
self-contained reference points — use them as is, fork them, or use the
framework to drop in your own.

| Paper | Plugin | Module | Implements |
|-------|--------|--------|------------|
| **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Zandieh et al., ICLR 2026) | `palm` scorer | `scorers/palm.py` | EMA novelty/surprise scoring |
| **Min-SNR Weighting** ([arXiv:2303.09556](https://arxiv.org/abs/2303.09556), Hang et al., ICCV 2023) | `snr` scorer | `scorers/snr.py` | Cosine + linear diffusion schedule scoring |
| **APTQ** ([arXiv:2402.14866](https://arxiv.org/abs/2402.14866), Guan et al., DAC 2024) + **Fisher Information** ([arXiv:1906.08589](https://arxiv.org/abs/1906.08589), Frantar et al.) | `fisher` (runtime proxy, broken) + **`fisher_static`** (offline calibration, works) | `scorers/fisher.py`, `scorers/fisher_static.py`, `optimization/fisher_calibration.py` | Two implementations: the runtime `fisher` uses `mean(x²)` proxy and over-allocates bits (don't use); the offline `calibrate_fisher()` does real forward+backward passes on a calibration set, computes per-layer K/V Fisher diagonals, saves to JSON; the `fisher_static` scorer loads the JSON and serves precomputed scores at runtime (constant-time lookup, no per-call overhead) |
| **Sheaf Attention** ([arXiv:2601.21207](https://arxiv.org/abs/2601.21207), AAAI 2026) | `sheaf` scorer | `scorers/sheaf.py` | Discrete Laplacian harmonicity classifier |
| **BSA — Bidirectional Sparse Attention** ([arXiv:2509.01085](https://arxiv.org/abs/2509.01085)) | `bsa` scorer | `scorers/bsa.py` | Block centroid saliency (KV side only) |
| **TurboQuant** tiered allocation | `tiered` strategy | `strategies/tiered.py` | Dual-quantizer routing by score |
| **DiTFastAttn** step sharing ([arXiv:2406.08552](https://arxiv.org/abs/2406.08552), Yuan et al., NeurIPS 2024) | `delta` strategy | `strategies/delta.py` | First-order inter-step Δ |
| **QuantSparse** ([arXiv:2509.23681](https://arxiv.org/abs/2509.23681)) | `delta2` strategy | `strategies/delta2.py` | Second-order Δ² with order-2 → 1 → full fallback |
| **DiTFastAttn** WA-RS | `window` strategy | `strategies/window.py` | Similarity-based attention output cache |
| **DiTFastAttn** CFG sharing + **Classifier-Free Guidance** ([arXiv:2207.12598](https://arxiv.org/abs/2207.12598), Ho & Salimans) | `cfg_sharing` strategy | `strategies/cfg_sharing.py` | Split-pass (WAN) + batched (LTX) modes |
| **Copresheaf Neural Networks** ([arXiv:2505.21251](https://arxiv.org/abs/2505.21251)) | registry extension | `codebook/registry.py` | head_type discriminator (per-head codebooks) |
| **Attention Analysis in VDiTs** ([arXiv:2504.10317](https://arxiv.org/abs/2504.10317)) | `skip_layers` config | `pipeline/runner.py` | Per-layer compression bypass for non-sparse layers |
| Finite-time Lyapunov exponents (general dynamical systems) | `lyapunov` monitor | `monitors/lyapunov.py` | Local divergence rate estimator |
| Attention entropy tracking | `stability` monitor | `monitors/stability.py` | EMA entropy-shift detection |

### Mixing plugins

```python
import tqai

# LLM with novelty scoring + temporal delta
cache = tqai.patch(
    model,
    bits_k=4, bits_v=2,
    pipeline={
        "scorer": "palm",
        "strategy": "delta",
        "scorer_kwargs": {"alpha": 0.5, "ema_decay": 0.95},
    },
)

# Diffusion with SNR-driven second-order delta
cache = tqai.patch(
    model,
    pipeline={
        "scorer": "snr",
        "strategy": "delta2",
        "monitor": "stability",
        "scorer_kwargs": {"schedule": "cosine"},
        "skip_layers": [0, 1, 28, 29],  # protect non-sparse layers
    },
)
```

### GA policy search (offline)

```python
from tqai.optimization import GASearch

def evaluate(genome):
    # Return -loss for the candidate pipeline configuration
    return -run_evaluation(genome.decode(scorers, strategies, monitors))

best = GASearch(population_size=20, generations=10, objective=evaluate, seed=42).run()
print(best.decode(scorers, strategies, monitors))
```

### Honest results

| Plugin / config | Verdict on Apple Silicon |
|-----------------|--------------------------|
| `bits_k=4, bits_v=2` (v0.3.1 default) | **Ship it** — byte-identical to baseline on 6 models |
| `optimize_vae_memory()` for WAN/LTX | **Ship it** — eliminates 100 GB+ VAE spike |
| `patch_mps_compatibility()` for LTX-2 | **Ship it** — pure unblocker (float64 → float32 RoPE) |
| `palm` + `delta` / `snr` + `delta2` | **Ship it** — −60% NMSE on synthetic streaming data |
| Pipeline framework (`pipeline=None` default) | **Ship it** — zero overhead, +ergonomics |
| `cfg_sharing` strategy | **Functional, no speedup** — Python overhead == compute saved |
| `chunk_attention=True` (MLX) | **Don't use** — fused Metal kernel wins; 3-5× slower |
| `fisher` scorer (runtime, squared activation proxy) | **Don't use** — over-allocates bits (NMSE 12× worse than baseline) |
| `fisher_static` scorer + `tqai calibrate` (v0.4) | **Ship it** — proper offline gradient-based Fisher calibration, constant-time runtime lookup |

Full benchmark write-up: [`reports/where_tqai_shines.md`](reports/where_tqai_shines.md).
Paper coverage matrix: [`reports/paper_coverage_report.md`](reports/paper_coverage_report.md).
Standalone HTML: [`reports/benchmark_report.html`](reports/benchmark_report.html).

---

## Diffusers Video Pipelines (WAN 2.2, LTX-2)

```python
import tqai
from diffusers import WanPipeline
from tqai.dit import optimize_vae_memory, patch_mps_compatibility

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ...)
pipe.to("mps")

# MUST DO: prevents VAE decoder OOM at long videos
optimize_vae_memory(pipe)

# MUST DO for LTX-2: fixes float64 RoPE crash on MPS
patch_mps_compatibility(pipe)

video = pipe("A cat surfing", num_frames=81)
```

| Utility | What it fixes |
|---------|---------------|
| `optimize_vae_memory(pipe)` | WAN VAE 100 GB+ spike → 8 GB peak (88% saved) |
| `patch_mps_compatibility(pipe)` | LTX-2 RoPE float64 crash on Apple MPS |
| `patch_cfg_sharing(pipe)` | Caches conditional attention output for unconditional pass (50%–100% share rate). **Note:** functional but no MPS speedup yet — see honest results above. |

---

## Offline Fisher Information Calibration (v0.4)

The runtime `fisher` scorer uses `mean(x²)` as a proxy for the Fisher
Information diagonal. This proxy over-allocates bits and routes
everything to the high-bit tier (NMSE 12× worse than baseline in our
synthetic benchmark). The fix is **offline gradient-based calibration**:
run a small calibration set through forward+backward passes once,
collect real per-layer K/V Fisher diagonals, save to JSON, then load
the JSON at runtime via the `fisher_static` scorer.

```bash
# Step 1: calibrate (PyTorch only, ~seconds for small models)
tqai calibrate --model Qwen/Qwen2.5-3B-Instruct --output qwen-3b-fisher.json
```

```python
# Step 2: use the calibration at inference time
import tqai
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

cache = tqai.patch(
    model,
    bits_k=4, bits_v=2,
    pipeline={
        "scorer": "fisher_static",
        "scorer_kwargs": {"calibration_path": "qwen-3b-fisher.json"},
        "strategy": "tiered",
    },
)
```

The static scorer is a constant-time dictionary lookup at runtime — no
gradient computation, no proxy, no per-call overhead. The calibration
JSON encodes per-layer importance derived from real `∂L/∂θ` measurements.

For programmatic use:

```python
from tqai.optimization import calibrate_fisher

cal = calibrate_fisher(
    model=model,
    tokenizer=tokenizer,
    prompts=["...", "...", "..."],   # 8-32 prompts is enough
    output_path="my-fisher.json",
)
print(f"Calibrated {cal.num_layers} layers from {cal.num_samples} samples")
```

---

## CLI

```bash
# Show environment and library info
tqai info

# v0.4 — list available pipeline plugins
tqai plugins

# Quantization accuracy benchmark
tqai benchmark
tqai benchmark --bits-k 3 --bits-v 2 --head-dim 128

# Generate text with compression
tqai run "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --backend torch

# v0.4 — generate with a pipeline plugin
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --scorer palm --strategy delta

# Run with QJL Stage 2
tqai run "Explain gravity" --model Qwen/Qwen2.5-3B-Instruct --use-qjl

# Compare baseline vs compressed side by side
tqai compare "Explain gravity" --model mlx-community/Qwen2.5-7B-Instruct-8bit

# Pre-convert a model for faster startup
tqai convert --model mlx-community/Qwen2.5-7B-Instruct-8bit --output ./qwen7b-tqai/

# v0.4 — offline gradient-based Fisher Information calibration
# Runs forward+backward passes on a small calibration set, saves per-layer
# Fisher diagonals to JSON. Use with the `fisher_static` scorer at runtime.
tqai calibrate --model Qwen/Qwen2.5-3B-Instruct --output qwen-3b-fisher.json
tqai calibrate --model Qwen/Qwen2.5-3B-Instruct --output qwen-3b-fisher.json --num-samples 32

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

# Unit + accuracy tests (~447 tests with v0.4, <90s)
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
├── __init__.py            # patch(), unpatch(), TurboQuantConfig
├── config.py              # Configuration dataclass
├── _patch.py              # Backend router (HF / mlx-lm / DiT)
├── quantizer.py           # PolarQuantizer (core algorithm + QJL Stage 2)
├── attention.py           # Chunked SDPA via online softmax (MLX, v0.4)
├── kernels/               # Fused Metal GPU kernels
├── hooks.py               # Forward-pass activation compression (PyTorch + MLX)
├── module_utils.py        # Transformer + DiT layer inspection
├── backend/               # PyTorch + MLX abstraction layer
├── codebook/              # Codebook solvers + per-head-type registry (copresheaf)
├── cache/                 # HuggingFace DynamicCache + mlx-lm KVCache integrations
│
│ # ─── v0.4 composable pipeline ──────────────────────────
├── pipeline/              # Protocol + registry + runner + builder
│   ├── base.py            # Scorer / Strategy / Monitor / ModelAdapter protocols
│   ├── registry.py        # Name-based plugin registration
│   ├── runner.py          # CompressionPipeline (scorer → strategy → quantizer → monitor)
│   └── __init__.py        # build_pipeline()
├── scorers/               # palm, snr, fisher, fisher_static, sheaf, bsa
├── strategies/            # tiered, delta, delta2, window, cfg_sharing
├── monitors/              # stability, lyapunov
├── adapters/              # llm, dit, wan
├── optimization/          # GA policy search (genome + ga_policy) + Fisher calibration
└── dit/                   # CFG sharing patch + VAE memory optimization + MPS fixes

benchmarks/
├── benchmark_forward.py        # Full KV + activation throughput (v0.3.1 + v0.4 configs)
├── benchmark_pipeline.py       # v0.4 — synthetic NMSE for every scorer × strategy
├── benchmark_long_context.py   # v0.4 — chunked attention at 4K/8K/16K/32K context
├── benchmark_video.py          # v0.4 — WAN 2.2 + LTX-2 video generation
├── eval_perplexity.py          # Perplexity evaluation helper
└── results/                    # Benchmark JSON results + actual MP4 outputs

reports/
├── where_tqai_shines.md        # Honest assessment: where v0.4 wins / loses on Apple Silicon
├── paper_coverage_report.md    # Per-paper implementation status
└── benchmark_report.html       # Standalone HTML with charts

paper/
├── tqai_paper.md               # Research paper (markdown source)
└── tqai_paper.tex              # Research paper (LaTeX)
```

---

## Paper

### Core algorithm

This library implements the TurboQuant algorithm from Google Research:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
> ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### KV cache compression family

- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Random rotation + polar coordinate quantization (the core of tqai)
- [QJL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) (AAAI 2025) — Quantized Johnson-Lindenstrauss residual correction (available in tqai as `use_qjl=True`)
- [KIVI](https://arxiv.org/abs/2402.02750) (ICML 2024) — Residual buffer strategy for KV cache compression (`cache_strategy="residual"`)
- [KVQuant](https://arxiv.org/abs/2401.18079) (NeurIPS 2024) — Fused dequant-attention kernel design
- [APTQ](https://arxiv.org/abs/2402.14866) (DAC 2024) — Attention-aware post-training quantization (attention-aware codebook objective + Fisher proxy)

### Diffusion / video compression (v0.4 paper playground)

- [DiTFastAttn](https://arxiv.org/abs/2406.08552) (NeurIPS 2024) — Window attention with residual sharing, step-wise sharing, CFG sharing (`window`, `delta`, `cfg_sharing`)
- [QuantSparse](https://arxiv.org/abs/2509.23681) (2025) — Second-order residual quantization for Wan2.1 KV cache (`delta2`)
- [BSA — Bidirectional Sparse Attention](https://arxiv.org/abs/2509.01085) (2025) — Query + KV joint sparsification (`bsa` scorer; KV side only — Q-side needs custom kernel)
- [Min-SNR Weighting](https://arxiv.org/abs/2303.09556) (Hang et al., ICCV 2023) — Cosine SNR schedule for diffusion training (`snr` scorer)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022) — The CFG protocol that `cfg_sharing` accelerates
- [Attention Analysis in Video DiTs](https://arxiv.org/abs/2504.10317) (2025) — Identifies non-sparse layers that must not be compressed (`skip_layers`)
- [SparseDiT](https://arxiv.org/abs/2412.06028) (NeurIPS 2025) — Tri-segment layer allocation (validates per-layer policy via `tiered` + `skip_layers`)

### Topological / category-theoretic foundations

- [Sheaf Attention & Local Consistency](https://arxiv.org/abs/2601.21207) (AAAI 2026) — Cellular sheaf framework, harmonic substructure (`sheaf` scorer)
- [Copresheaf Neural Networks](https://arxiv.org/abs/2505.21251) (2025) — Per-cell stalks, justifies per-head-type codebooks (`codebook/registry.py`)

### Chunked / memory-efficient attention (v0.4 cherry-pick)

- [Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al., NeurIPS 2022) — Tile-wise attention with online softmax
- Milakov & Gimelshein, *"Online normalizer calculation for softmax"* (NVIDIA tech report, 2018) — The online softmax recurrence used in `attention.py`

### Codebook solvers (build-time)

- [DSQ](https://arxiv.org/abs/1908.05033) (2019) — Differentiable soft quantization (fuzzy codebook solver)
- [IDE-LBG](https://arxiv.org/abs/1710.05311) (2017) — Evolutionary codebook optimization (CMA-ES solver)

### Theoretical context (informs design, not directly implemented)

- *Fisher Information for Neural Network Compression* ([arXiv:1906.08589](https://arxiv.org/abs/1906.08589)) — Theoretical basis for the `fisher` scorer (we ship a squared-activation proxy; true gradient-based Fisher is offline-only)
- Information geometry / Fisher-Rao geodesics — Justifies the cosine schedule in `snr` scorer as the geodesically-optimal SNR allocation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits require a DCO sign-off (`git commit -s`).

**Paper integrations are explicitly welcome.** If you're working on a KV
cache, attention, or diffusion compression paper, the v0.4 architecture
is designed so you can land a reference implementation as a single file
without touching `quantizer.py`, `cache/`, or any other core module.
The fast path:

1. Pick the right slot — `scorers/` (per-token importance), `strategies/`
   (how to compress), `monitors/` (runtime adaptation), or `adapters/`
   (a new model family).
2. Implement the matching protocol from `pipeline/base.py` (3-4 methods).
3. Register your plugin with one line in the directory's `__init__.py`.
4. Add tests in `tests/` and a row to the Paper Playground table above.
5. Open a PR — the core stays untouched, so review is fast.

The point of the freeze on the core is reproducibility: the
`PolarQuantizer` math, the Lloyd-Max codebooks, and the cache strategies
have been benchmarked across 6+ models with `Δppl=0.00`. New ideas
should compose with that baseline, not replace it.

## License

MIT
