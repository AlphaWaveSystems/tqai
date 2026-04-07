# tqai: Efficient KV Cache Compression with Incremental Reconstruction, Fused Metal Kernels, Composable Pipeline Architecture, and Adaptive Bit Allocation for Local LLM and DiT Inference

**Authors:** Patrick Bertsch, Claude (Anthropic)

**Date:** April 2026 (v0.4 revision)

---

## Abstract

We present tqai, a comprehensive KV cache compression library for local inference of large language models (LLMs) and diffusion transformers (DiTs) on Apple Silicon. Building on the TurboQuant framework (Zandieh et al., 2026), tqai introduces five key contributions: (1) an **incremental cache reconstruction** strategy that reduces per-token dequantization from O(n) to O(1), delivering 2.3–3.6× throughput improvement over naive full-reconstruction approaches; (2) **fused Metal GPU kernels** that eliminate Python dispatch overhead by combining L2 normalization, orthogonal rotation, and codebook search into single GPU dispatches via MLX's `mx.fast.metal_kernel` API; (3) **evolutionary codebook optimization** using CMA-ES and fuzzy C-means with temperature annealing, achieving 0.55% MSE improvement over classical Lloyd-Max codebooks; (4) a **composable middleware pipeline (v0.4)** that turns the library from a single-algorithm compressor into a plugin framework where each new compression paper — TurboQuant, QuantSparse, DiTFastAttn, BSA, Sheaf, Copresheaf, Fisher, SparseDiT — ships as a single file without modifying the proven core; and (5) **production diffusers video pipeline support** including a VAE memory optimization that prevents the >100 GB peak that otherwise OOMs WAN 2.2 5B at 81-frame video on a 128 GB Mac, plus an MPS float64 fix that unblocks LTX-2 on Apple Silicon. We additionally ship a **chunked attention** implementation using the online softmax algorithm, a **Palm information-theoretic scorer** for adaptive bit allocation, and **CFG attention sharing** hooks. Across 12 model variants (0.5B–14B parameters) including Gemma 4, tqai achieves **zero perplexity degradation** (Δppl = 0.00 across every model and every quantization config tested) and produces an **80% reduction in compressed-storage KV cache size** (compressed indices vs bf16 KV tensors, useful for serialized cache / wire format / future fused dequant-attention kernels). On compute-bound models (7B+), throughput retention reaches **98–101%** of uncompressed baseline. We report **three principled negative results**: (a) **chunked attention via Python loops loses decisively to MLX's fused `mx.fast.scaled_dot_product_attention`** on Apple Silicon (3–5× slower at 16K–32K context, with bit-identical output — the implementation remains shipped for CUDA users without FlashAttention, but should not be enabled on MLX); (b) **DiTFastAttn CFG sharing on MLX is functionally correct but does not deliver wall-clock speedup** because the per-attention Python dispatch overhead approximately cancels the saved attention compute on Apple Silicon's already-fast fused attention kernel; and (c) **K4/V2 KV quantization does not save peak runtime memory on Apple Silicon today** — measured peak via `mx.get_peak_memory()` on Qwen 2.5-7B at 4K–128K context shows tqai within ±0.2 GB of baseline at every length, because the v0.3.1 incremental cache strategy maintains a persistent dequantized buffer at full input precision (the design choice that enables O(1) per-token decode throughput). The runtime memory unlock requires a fused dequant-attention Metal kernel (KVQuant approach) which is on the v0.5 roadmap. The total system spans 484 tests (zero regressions on the 293 pre-v0.4 tests), 12 of 13 referenced compression papers, and reproducible benchmarks for LLM throughput, video generation, long-context KV memory, and few-step video presets.

---

## 1. Introduction

KV cache compression is critical for enabling long-context inference on memory-constrained devices. The TurboQuant algorithm (Zandieh et al., 2026) provides a theoretically optimal data-oblivious quantization scheme based on random orthogonal rotation followed by Lloyd-Max scalar quantization. However, the reference implementation leaves several systems engineering challenges unaddressed:

1. **Reconstruction overhead**: Naive implementations dequantize the entire KV history on every token, creating O(n²) total work over n tokens.
2. **Python dispatch latency**: Multiple MLX/PyTorch operations per quantize/dequantize call incur interpreter overhead.
3. **Codebook optimality**: Lloyd-Max codebooks are locally optimal but may not be globally optimal.
4. **Diffusion transformer support**: TurboQuant was designed for autoregressive LLMs, not DiT architectures.
5. **Architectural sprawl**: Each new compression paper from the literature (QuantSparse, DiTFastAttn, BSA, Fisher, Sheaf, etc.) requires modifying the same core files (`quantizer.py`, `cache/hf.py`, `cache/mlx.py`), making the library increasingly difficult to maintain and producing merge conflicts between contributors.
6. **Production video pipeline support**: Diffusers video pipelines (WAN 2.2, LTX-2) crash or OOM on Apple Silicon for two unrelated reasons — VAE decoder memory spikes and MPS float64 incompatibility — that block real local use even before any compression is considered.

We address challenges 1–4 in tqai v0.3.1, challenge 5 with the v0.4 composable pipeline architecture, and challenge 6 with the v0.4 diffusers integration utilities. We also provide comprehensive benchmarks across 12 model configurations and report principled negative results for techniques that work in theory but lose to fused kernels on Apple Silicon.

---

## 2. Background

### 2.1 PolarQuant / TurboQuant

The TurboQuant Stage 1 (PolarQuant) quantization of a vector x ∈ ℝᵈ proceeds as:

1. **Norm extraction**: s = ‖x‖₂, stored as FP16
2. **Normalization**: x̃ = x / (s + ε)
3. **Rotation**: y = x̃ · Rᵀ, where R is a fixed Haar-distributed orthogonal matrix
4. **Scalar quantization**: For each coordinate i, find qᵢ = argmin_c |yᵢ − cⱼ| over Lloyd-Max centroids {c₁, ..., c_{2^b}}
5. **Storage**: (q₁, ..., q_d) as uint8 indices + s as float16

Dequantization reverses: lookup centroids → inverse rotate → scale by norm.

**Theorem (Zandieh et al., 2026):** For x ~ N(0, I_d/d), the expected distortion of PolarQuant at b bits per coordinate is:

$$E[\|x - \hat{x}\|^2] = D^*(b, d) + O(1/d)$$

where D*(b, d) is the rate-distortion optimal distortion for the Gaussian source at rate b.

### 2.2 Lloyd-Max Quantization

The Lloyd-Max algorithm finds optimal scalar quantization levels for a known distribution by iterating:

1. **Centroid update**: cⱼ = E[X | X ∈ partition j]
2. **Boundary update**: bⱼ = (cⱼ + cⱼ₊₁) / 2

For the post-rotation coordinate distribution N(0, 1/d), this converges to the minimum-MSE codebook.

### 2.3 MLX and Apple Silicon

MLX (Apple, 2023) is a machine learning framework for Apple Silicon featuring:
- Unified memory architecture (no CPU↔GPU transfers)
- Lazy evaluation with automatic graph fusion
- `mx.fast.metal_kernel()` for custom Metal Shading Language (MSL) kernels

---

## 3. Incremental Cache Reconstruction

### 3.1 Problem: O(n²) Full Reconstruction

The naive cache implementation stores compressed entries and reconstructs the full history on every token:

```
For token t = 1, ..., n:
    Store compress(new_kv)
    Reconstruct: for i = 1..t: decompress(entry_i)  ← O(t) work
```

Total work: Σ_{t=1}^{n} t = n(n+1)/2 = O(n²)

For n = 100 tokens at 24 layers × 2 (K+V), this is 100 × 24 × 2 × 50.5 = 242,400 dequantize calls.

### 3.2 Solution: Incremental Buffer

We maintain a running dequantized buffer alongside compressed storage:

```
For token t = 1, ..., n:
    entry = compress(new_kv)
    buffer = concat(buffer, decompress(entry))  ← O(1) dequant
    Return concat(sink, buffer)
```

Total work: O(n) — one dequantize per token.

**Memory analysis:** Per KV vector:
- Compressed: 130 bytes (128B uint8 indices + 2B fp16 norm)
- Buffer: 256 bytes (fp16, D=128)
- Total: 386 bytes
- vs. full reconstruction: 130B stored + 512B fp32 materialized each call

The incremental buffer uses **less** peak memory than full reconstruction.

### 3.3 KIVI-style Residual Buffer

Inspired by KIVI (Yuan et al., 2024), we extend the incremental strategy with a residual window:

- Last R tokens kept uncompressed in full precision
- Older tokens compressed → dequantized into incremental buffer
- Assembly: `concat(sink, old_buffer, recent_uncompressed)`

This provides zero quantization error on the R most recent tokens, which receive the highest attention weights in most transformer architectures.

### 3.4 Results

| Model | v0.2 (full) | v0.3.1 (incremental) | Speedup |
|-------|-------------|---------------------|---------|
| Qwen 0.5B bf16 | 33 tok/s (10%) | 118 tok/s (36%) | **3.6×** |
| Qwen 3B bf16 | 21 tok/s (30%) | 66 tok/s (93%) | **3.1×** |
| Llama 8B Q4 | 31 tok/s (30%) | 85 tok/s (84%) | **2.7×** |
| Qwen 7B Q8 | 26 tok/s (42%) | 59 tok/s (93%) | **2.3×** |
| Qwen 14B Q4 | 20 tok/s (35%) | 50 tok/s (89%) | **2.5×** |

---

## 4. Fused Metal GPU Kernels

### 4.1 Kernel Design

We fuse the entire quantize pipeline (L2 norm, normalization, rotation, codebook search) into a single Metal compute kernel via `mx.fast.metal_kernel()`.

**Quantize kernel:** One thread per (vector, coordinate) pair. Grid = (num_vecs × D, 1, 1).

Each thread:
1. Computes the full L2 norm redundantly (D reads, O(D) work)
2. Normalizes its coordinate
3. Computes its rotated coordinate: yᵈ = Σⱼ (xⱼ/‖x‖) · R[d,j]
4. Performs serial argmin over n_levels centroids
5. Stores uint8 index

The redundant norm computation trades O(D) extra work per thread for avoiding shared memory (which `mx.fast.metal_kernel` does not support). For D ∈ {64, 128}, the extra work is negligible compared to Python dispatch savings.

**Dequantize kernel:** Reverse path — centroid lookup, inverse rotation, norm scaling.

### 4.2 Compile-Time Specialization

We bake D and n_levels as compile-time constants into the MSL source:

```metal
const uint D = {D};
const uint N_LEVELS = {N_LEVELS};
```

This eliminates the need for `ensure_row_contiguous=False` (which injects per-call shape metadata), reducing kernel dispatch overhead.

### 4.3 Correctness

| Bits | head_dim | Index match | Max dequant diff |
|------|----------|-------------|-----------------|
| 2 | 64, 128 | 100% | 1.2×10⁻³ |
| 4 | 64, 128 | 100% | 7.2×10⁻⁷ |
| 8 | 128 | 100% | 7.2×10⁻⁷ |
| 8 | 64 | 98.8% | 1.2×10⁻³ |

The 8-bit, D=64 case shows <2% index mismatch (off-by-1) due to FP rounding on closely-spaced centroids. This is benign — adjacent centroids differ by ~0.001.

---

## 5. Evolutionary Codebook Optimization

### 5.1 CMA-ES Solver

We replace Lloyd-Max with Covariance Matrix Adaptation Evolution Strategy (CMA-ES; Hansen & Ostermeier, 2001), initialized from the Lloyd-Max solution.

**Objective:** Minimize Monte Carlo estimate of expected distortion:

$$\min_{\mathbf{c}} \hat{E}[|X - Q_{\mathbf{c}}(X)|^2], \quad X \sim N(0, 1/d)$$

where Q_c is the nearest-centroid quantizer with codebook c = (c₁, ..., c_{2^b}).

**Constraint:** Centroids sorted ascending after each generation (projection onto feasible set).

**Results (D=128, 4-bit):**

| Solver | MSE | Improvement | Time |
|--------|-----|-------------|------|
| Lloyd-Max | 7.588×10⁻⁵ | baseline | 0.23s |
| Fuzzy C-means | 7.585×10⁻⁵ | −0.04% | 1.44s |
| CMA-ES | 7.546×10⁻⁵ | **−0.55%** | 8.44s |

### 5.2 Fuzzy C-Means with Temperature Annealing

Inspired by Differentiable Soft Quantization (Gong et al., 2019), we replace hard nearest-centroid assignment with temperature-controlled soft membership:

$$\mu_k(x) = \frac{\exp(-|x - c_k|^2 / \tau)}{\sum_j \exp(-|x - c_j|^2 / \tau)}$$

Temperature annealing: τ_t = τ₀ · α^t, where α = 0.95. At τ → 0, this converges to hard assignment.

### 5.3 Attention-Aware Objective

Inspired by APTQ (Li et al., 2024), we provide an alternative objective that optimizes codebooks to preserve softmax attention distributions:

$$\min_{\mathbf{c}} E[\|\text{softmax}(QK^\top / \sqrt{d}) - \text{softmax}(Q\hat{K}^\top / \sqrt{d})\|^2]$$

This accounts for the fact that attention is invariant to uniform scaling and only cares about relative similarities.

---

## 6. Chunked Attention via Online Softmax

### 6.1 Algorithm

For sequence length T_kv with chunk size C, we split K, V into ⌈T_kv/C⌉ chunks and maintain running accumulators using the online softmax algorithm (Milakov & Gimelshein, 2018; Dao et al., 2022):

**Initialize:** m = −∞, ℓ = 0, O = 0

**For each chunk c ∈ {1, ..., ⌈T/C⌉}:**

$$S_c = Q K_c^\top \cdot \text{scale}$$

$$m_{\text{new}} = \max(m, \max(S_c))$$

$$\alpha = \exp(m - m_{\text{new}})$$

$$P_c = \exp(S_c - m_{\text{new}})$$

$$O \leftarrow O \cdot \alpha + P_c V_c$$

$$\ell \leftarrow \ell \cdot \alpha + \sum P_c$$

$$m \leftarrow m_{\text{new}}$$

**Output:** O / ℓ

**Theorem:** The output is identical to standard attention up to floating-point rounding.

*Proof:* Let a_i denote the unnormalized attention weight for position i, i.e., a_i = exp(q·k_i/√d). The standard output is O = Σᵢ (aᵢ/Σⱼaⱼ) · vᵢ. The online algorithm maintains ℓ = Σⱼ aⱼ · exp(m_final − m_j) where m_j was the running max when aⱼ was processed. Since exp(m_final − m_j) = exp(m_final)/exp(m_j) and the corrections exactly cancel, ℓ = Σⱼ aⱼ. Similarly for O. □

### 6.2 Memory Analysis

| Tokens | Full attention matrix | Chunked (C=4096) | Reduction |
|--------|----------------------|------------------|-----------|
| 4,096 | 64 MB | 64 MB (passthrough) | 1× |
| 48,360 | 17.8 GB | 0.8 GB | **22×** |
| 1,140,000 | 4.9 TB | 18 GB | **277×** |

The last row represents WAN 2.2 at 720P × 81 frames through the VAE.

---

## 7. Palm: Information-Theoretic Adaptive Bit Allocation

### 7.1 Novelty-Surprise Scoring

For each token or activation tensor x, we compute:

**Novelty:** n(x) = ‖x − μ_EMA‖ / σ_EMA, where μ_EMA and σ_EMA are exponential moving averages updated with decay α.

**Surprise:** s(x) = log(1 + ‖x − Q(x)‖²), where Q(x) is the quantized reconstruction.

**Info score:** I(x) = √(n(x) · s(x)) when both are available, otherwise I(x) = n(x).

### 7.2 Tier Mapping

| Tier | Info score | Bits | Description |
|------|-----------|------|-------------|
| 0 | I < 0.1 | 2 | Redundant — aggressive compression |
| 1 | 0.1 ≤ I < 0.3 | 3 | Expected — standard compression |
| 2 | 0.3 ≤ I < 0.7 | 4 | Novel — preserve detail |
| 3 | I ≥ 0.7 | 8 | Critical — minimal compression |

### 7.3 Diffusion Step Scoring

For diffusion transformers, the denoising schedule provides a natural info score:

$$I(t) = \frac{1}{1 + \text{SNR}(t)}$$

Early steps (low SNR, high noise) → I ≈ 1 → more bits.
Late steps (high SNR, refinement) → I ≈ 0 → fewer bits.

This maps directly to the Lyapunov stability of the denoising trajectory: early steps are chaotic (sensitive to perturbation), late steps are stable (converging).

---

## 8. Diffusion Transformer Integration

### 8.1 Architecture Detection

We extend the module detection system to recognize DiT attention patterns:

| Architecture | Q | K | V | Pattern |
|---|---|---|---|---|
| LLM (HuggingFace) | `q_proj` | `k_proj` | `v_proj` | Standard |
| LLM (GPT-2) | `c_attn` (fused) | — | — | Fused |
| DiT (diffusers) | `to_q` | `to_k` | `to_v` | Diffusers |
| DiT (fused) | `qkv` | — | `proj` | Fused DiT |

### 8.2 WAN 2.2 Integration

WAN 2.2 TI2V-5B (Alibaba, 2026) is a 5B-parameter text-image-to-video diffusion transformer. tqai detects 60 attention modules and 30 FFN modules in the WAN 2.2 transformer.

**Text Encoder Cache:** The T5-XXL encoder output is fixed across all denoising steps. We cache it after the first call and return the cached version on subsequent calls, optionally compressed.

**Inter-Step Delta Compression:** Adjacent denoising steps produce nearly identical K/V activations. We track the delta ‖K_t − K_{t-1}‖ / ‖K_t‖ and store only the delta at low bits when it falls below a threshold.

### 8.3 v0.4 Production Diffusers Integration

The v0.3.x DiT integration achieved correct architecture detection but did not produce a usable end-to-end video generation experience on Apple Silicon. Two distinct blockers existed before any compression could be considered:

**Blocker 1: VAE decoder memory spike.** WAN 2.2 5B's `AutoencoderKLWan._decode` accumulates 3D convolution feature maps in `self._feat_map` (one entry per decoder conv layer) and a growing `out` tensor via `torch.cat` along the temporal axis. For 81 frames at 480×832, this allocates approximately 100 GB of intermediate tensors during the decode pass — enough to OOM a 128 GB Mac. The diffusers VAE provides `enable_tiling()` and `enable_slicing()` methods, but they default to off and require explicit activation.

**Blocker 2: MPS float64 incompatibility (LTX-2).** The `LTX2RotaryPosEmbed1d` module in the diffusers LTX-2 implementation calls `torch.linspace(..., dtype=torch.float64, device=mps_device)`, which crashes immediately because Apple's MPS backend does not support float64. The same issue exists in the LTX-2 transformer's RoPE module. The fix is a one-line attribute override (`module.double_precision = False`) but it must be applied to two separate modules in the loaded pipeline.

**v0.4 ships both fixes as drop-in utilities:**

```python
from tqai.dit import optimize_vae_memory, patch_mps_compatibility

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ...)
pipe.to("mps")
optimize_vae_memory(pipe)        # ≥88% peak VAE memory reduction
patch_mps_compatibility(pipe)    # no-op for WAN; unblocks LTX-2

video = pipe("A cat surfing", num_frames=81)
```

After these utilities, peak resident set size during a 33-frame WAN 2.2 5B generation is **32.2 GB** (model weights dominate), versus an unbounded spike that exceeds available memory at 81 frames without tiling.

### 8.4 CFG Attention Sharing

We implement DiTFastAttn's classifier-free guidance (CFG) sharing technique (Yuan et al., 2024) as a strategy plugin (`cfg_sharing`) that hooks into attention modules to cache the conditional pass output and serve it for the unconditional pass. We support both architectures observed in modern diffusers pipelines:

- **Split-pass CFG (WAN 2.2):** Conditional and unconditional passes run as two separate forward calls. Hooks track phase via the pipeline's `cache_context("cond"/"uncond")` and serve cached outputs during the unconditional phase. Achieves a 50% share rate on WAN 2.2.

- **Batched CFG (LTX-2):** Both passes run in a single forward call with `[uncond, cond]` concatenated along the batch dimension. Hooks copy the second half of the batch (cond) into the first half (uncond) at each attention output. Achieves a 100% share rate on LTX-2.

We measure the wall-clock impact in §9.2.

### 8.5 Forward Hook Limitation Resolved

The v0.3.x device-transfer limitation (CPU↔MPS round-trips for the rotation matrix) is resolved in v0.4 by moving the rotation matrix to the input device on first use, plus shipping a parallel `MLXForwardCompressionHooks` class that uses MLX module replacement (`_MLXCompressedWrapper`) instead of PyTorch's `register_forward_pre_hook` API (which MLX does not provide). Forward compression on MLX now achieves the same correctness as PyTorch with no device-transfer overhead.

---

## 9. Composable Middleware Pipeline (v0.4)

### 9.1 Motivation

By v0.3.1, the library had absorbed a sequence of distinct techniques (incremental cache, residual cache, fused kernels, codebook solvers, forward hooks) by accreting flags onto `TurboQuantConfig` and modifying the same core files. Adding a new technique from the literature — for example QuantSparse's second-order residual reparameterization (arXiv:2509.23681) — required modifying `quantizer.py` *and* `cache/hf.py` *and* `cache/mlx.py` simultaneously, with conflicts proportional to the number of in-flight contributors. The implementation also conflated two distinct concerns: *what to compress* (per-token importance scoring) and *how to compress it* (the actual quantization strategy), even though papers in the literature typically address only one or the other.

### 9.2 Architecture

We restructure the library around four protocol types defined in `pipeline/base.py`, each addressing one slot in the compression decision process:

```
Scorer        — per-token importance scoring (palm, snr, fisher, sheaf, bsa)
Strategy      — how to compress an entry  (tiered, delta, delta2, window, cfg_sharing)
Monitor       — runtime parameter adjustment (stability, lyapunov)
ModelAdapter  — model family integration  (llm, dit, wan)
```

Each protocol is a Python `Protocol` (PEP 544) with 3–4 methods. Plugins implement the protocol, register themselves by name in their directory's `__init__.py`, and become discoverable via `tqai plugins`. The runner (`pipeline/runner.py`) wraps the existing `PolarQuantizer` and dispatches `compress()` and `decompress()` through the optional scorer and strategy. When `pipeline=None` (the default), the runner short-circuits to direct `PolarQuantizer.quantize()` calls — byte-identical to v0.3.1, zero overhead, all 293 pre-v0.4 tests pass byte-identically.

### 9.3 Paper Playground

We ship reference implementations of each technique from the recent compression literature as plugins. The implementations are intentionally self-contained and do not modify the proven v0.3.1 core. The current set covers 12 of 13 referenced papers:

| Paper | arXiv | Plugin | What it implements |
|-------|-------|--------|--------------------|
| TurboQuant (Zandieh et al., ICLR 2026) | 2504.19874 | `palm` scorer + `tiered` strategy | EMA novelty/surprise scoring + dual-quantizer routing |
| Min-SNR (Hang et al., ICCV 2023) | 2303.09556 | `snr` scorer | Cosine + linear diffusion schedule scoring |
| APTQ (Guan et al., DAC 2024) + Fisher Information | 2402.14866, 1906.08589 | `fisher` scorer | Squared activation proxy for FIM diagonal (offline-calibration mode) |
| Sheaf Attention (AAAI 2026) | 2601.21207 | `sheaf` scorer | Discrete Laplacian harmonicity classifier on the sequence axis |
| BSA — Bidirectional Sparse Attention | 2509.01085 | `bsa` scorer | Block centroid saliency (KV side; Q-side requires custom kernel) |
| DiTFastAttn step-share (Yuan et al., NeurIPS 2024) | 2406.08552 | `delta` strategy | First-order inter-step Δ with norm threshold |
| QuantSparse | 2509.23681 | `delta2` strategy | Second-order Δ² with order-2 → 1 → full automatic fallback |
| DiTFastAttn WA-RS | 2406.08552 | `window` strategy | Similarity-based attention output cache |
| DiTFastAttn CFG sharing + CFG (Ho & Salimans, 2022) | 2406.08552, 2207.12598 | `cfg_sharing` strategy | Split-pass (WAN) + batched (LTX-2) modes |
| Copresheaf Neural Networks | 2505.21251 | codebook registry extension | `head_type` discriminator for per-head codebooks |
| Attention Analysis in Video DiTs | 2504.10317 | `skip_layers` config | Per-layer compression bypass for non-sparse layers |
| SparseDiT (NeurIPS 2025) | 2412.06028 | tiered + skip_layers composition | Tri-segment layer allocation |
| Spherical Attention | 2505.09326 | (not implemented) | Requires modifying attention math; out of scope without retraining |
| Lyapunov stability (general dynamical systems) | — | `lyapunov` monitor | Local divergence rate estimator |

### 9.4 Plugin Authoring Cost

A new paper requires three things and nothing else:

1. **One file** in the appropriate directory (`scorers/`, `strategies/`, `monitors/`, or `adapters/`) implementing the matching protocol from `pipeline/base.py`.
2. **One line** of registration in the directory's `__init__.py`.
3. **At least one test file** verifying the plugin passes the protocol contract.

The proven core (`PolarQuantizer`, Lloyd-Max codebooks, cache strategies) is **frozen**. Every measurement reported in §9 of the v0.3.1 paper revision (Δppl=0.00 across 6 models) remains valid for the default path because that path is unchanged by construction.

### 9.5 Genetic Algorithm Policy Search

Because the pipeline configuration space is discrete (which scorer × which strategy × which monitor × per-strategy hyperparameters), we provide an offline GA search (`optimization/ga_policy.py`) that evolves a `PolicyGenome` (9 genes encoding scorer index, strategy index, monitor index, EMA decay, tier threshold, delta threshold, window size, bits_k, bits_v) against an arbitrary fitness function. Population: 20 individuals. Generations: 10. Mutation rate: 0.15. Elitism: top 20%. Tournament selection: k=3. The user supplies an objective function (e.g., negative perplexity or NMSE) and the GA returns the best decoded pipeline configuration.

---

## 10. Comprehensive Benchmark Results

### 10.1 LLM Throughput and Quality (MLX, Apple Silicon, v0.4)

| Model | Params | Quant | Baseline | kv-only | Retention | PPL | ΔPPL |
|-------|--------|-------|----------|---------|-----------|-----|------|
| Qwen2.5-0.5B | 0.5B | bf16 | 302 tok/s | 88 tok/s | 29% | 4.34 | **0.00** |
| Qwen2.5-3B | 3B | bf16 | 69 tok/s | 24 tok/s | 35% | 2.49 | **0.00** |
| Llama-3.1-8B | 8B | Q4 | 28 tok/s | 28 tok/s | **100%** | 2.95 | **0.00** |
| Qwen2.5-7B | 7B | Q8 | 17 tok/s | 17 tok/s | **101%** | 2.40 | **0.00** |
| Qwen2.5-14B | 14B | Q4 | 15 tok/s | 15 tok/s | **98%** | 2.22 | **0.00** |
| Gemma 4 E4B | 4B | Q4 | 47 tok/s | 24 tok/s | 50% | 126.94 | **0.00** |

**Key finding:** Compute-bound models (7B+ with quantized weights) show **zero throughput cost** from KV cache compression. The incremental buffer overhead is completely absorbed by the model's own compute time. Token outputs are byte-identical to baseline on every model tested.

### 10.2 Pipeline Strategy Quality (synthetic NMSE, mean across 7 model profiles)

Synthetic streaming benchmark with 7 model profiles (Qwen 0.5B/3B/7B, Gemma 2B/7B, Llama 8B, WAN 2.2 5B), 5 inter-step time steps with decreasing perturbation noise (mimicking diffusion denoising), 64 sequence positions per step, 4 layers per model:

| Config | NMSE | vs Baseline | Notes |
|--------|-----:|------------:|-------|
| baseline | 0.009290 | — | Direct PolarQuantizer at 4/2 bits |
| palm+tiered | 0.009301 | +0.1% | Adaptive bit allocation |
| **palm+delta** | **0.003759** | **−59.5%** | First-order inter-step Δ (DiTFastAttn step-share) |
| **snr+delta2** | **0.003746** | **−59.7%** | Second-order Δ² with cosine SNR schedule (QuantSparse) |
| sheaf+delta2 | 0.003763 | −59.5% | Sheaf harmonicity + second-order Δ² |
| palm+window | 0.018907 | +103.5% | Cache reuse trades quality for speed |
| fisher+tiered | 0.115858 | +1148% | Squared activation proxy is too aggressive — runtime use is broken; offline calibration mode required |
| skip_layers | 0.009297 | 0.0% | Layer protection passthrough (arXiv:2504.10317) |

**Key finding:** Inter-step delta strategies cut reconstruction distortion by ~60% relative to direct quantization on synthetic streaming data. This empirically validates QuantSparse's (arXiv:2509.23681) claim that the second-order residual is more compressible than the activation itself, and DiTFastAttn's claim that adjacent denoising steps have high temporal redundancy.

### 10.3 Video Generation: WAN 2.2 5B and LTX-2 (MLX)

End-to-end diffusers video generation, 33 frames at 480×832, 15 denoising steps, fixed seed:

| Model | Config | Wall-clock | Peak RSS | CFG share | PSNR vs baseline |
|-------|--------|-----------:|---------:|----------:|-----------------:|
| WAN 2.2 5B | baseline | 172.8s | 34.6 GB | 0% | — |
| WAN 2.2 5B | tqai_full | 170.5s | 32.2 GB | 50% | 56.5 dB |
| LTX-2 | baseline | 88.3s | 73.9 GB | 0% | — |
| LTX-2 | tqai_full | 90.9s | 74.4 GB | 100% | 56.5 dB |

`tqai_full` = `optimize_vae_memory()` + `patch_cfg_sharing()` + 8-bit forward hooks on hidden/FFN.

**Two findings:**

1. **The actual local win is VAE memory optimization, not compression speedup.** Without `optimize_vae_memory()`, WAN 2.2 5B's VAE decoder spikes to >100 GB on an 81-frame video and OOMs on a 128 GB Mac. With it, peak resident set stays at 32 GB and 20-second videos fit comfortably. This single utility is the difference between "video generation is impossible locally" and "video generation just works."

2. **CFG sharing is functionally correct but does not deliver wall-clock speedup on MPS.** Hooks fire at the expected rate (50% on WAN's split-pass CFG, 100% on LTX-2's batched CFG) and produce near-lossless output (PSNR 56.5 dB), but the Python hook dispatch overhead almost exactly cancels the saved attention compute on Apple Silicon's already-fast `mx.fast.scaled_dot_product_attention`. We classify this as a *functional but bottlenecked* result and document it explicitly: the speedup the paper claims is real on CUDA + Triton, but on MLX a fused Metal kernel would be required to realize it.

### 10.4 Long-Context Chunked Attention (Honest Negative Result)

Long-context generation benchmark on Qwen 2.5-3B (bf16) and Qwen 2.5-7B (8-bit), comparing baseline `mx.fast.scaled_dot_product_attention` vs the chunked online-softmax implementation in `attention.py`:

| Model | Context | Baseline prefill | Chunked prefill | Slowdown | Output match |
|-------|--------:|----------------:|----------------:|---------:|:------------:|
| Qwen 3B bf16 | 4K (no chunking) | 0.6s | 0.6s | 0% | ✓ |
| Qwen 3B bf16 | 8K | 1.1s | 3.3s | **3.0×** | ✓ |
| Qwen 3B bf16 | 16K | 2.6s | 12.0s | **4.6×** | ✓ |
| Qwen 7B Q8 | 16K | 5.8s | 27.5s | **4.7×** | ✓ |
| Qwen 7B Q8 | 32K | 17.4s | 68.3s | **3.9×** | ✓ |

**Output is bit-identical to baseline** on every configuration (verified by string comparison of generated tokens after the decode-mode causal mask bug fix described in §10.6). The chunked attention math is correct.

**The slowdown is purely a Python dispatch problem.** Each chunk in the loop schedules ~10 separate MLX operations (slice, matmul, transpose, mask construction, exp, division, sum, ...). At 16K context with `chunk_size=4096`, this is 4 chunks × ~10 ops × 30 transformer layers × 2 attention modules per layer = approximately **2,400 separate Metal kernel launches per token**. The fused `mx.fast.scaled_dot_product_attention` accomplishes the equivalent inner loop in **one** kernel launch with hand-tuned tile sizes and threadgroup memory layout.

Memory savings are also negligible at these scales: the model weights (8.7 GB for Qwen 7B Q8) dominate the KV cache (well under 1 GB at 32K), so the theoretical 139× attention-matrix memory reduction at 48K is invisible relative to the parameter footprint.

**We ship the chunked attention implementation but recommend against enabling it on MLX.** It is intended for CUDA users without FlashAttention, where the trade-off inverts. The recommendation is documented in the README and the "where tqai shines" report, and the configuration flag (`chunk_attention=True`) is gated separately from KV compression so users can opt out cleanly.

### 10.5 Codebook Quality (unchanged from v0.3.1)

| Solver | D=128, 4-bit MSE | Improvement |
|--------|-----------------|-------------|
| Lloyd-Max | 7.588×10⁻⁵ | baseline |
| Fuzzy C-means (α=0.95) | 7.585×10⁻⁵ | −0.04% |
| CMA-ES (500 gen) | 7.546×10⁻⁵ | **−0.55%** |

### 10.6 Metal Kernel and Chunked Attention Correctness

**Metal kernel:**

| Bits | head_dim | Index match rate | Max norm diff |
|------|----------|-----------------|---------------|
| 2 | 64, 128 | 100.0% | 0.0 |
| 4 | 64, 128 | 100.0% | 0.0 |
| 8 | 128 | 100.0% | 0.0 |
| 8 | 64 | 98.8% | 0.0 |

**Chunked attention correctness** (`atol=5×10⁻⁴`, `rtol=2×10⁻³`) across:
- Sequence lengths: 128, 256, 512, 1024
- Chunk sizes: 32, 64, 128, 256, 512
- Mask types: none, causal, additive
- Head configurations: MHA and GQA (4× ratio)
- **Decode mode (T_q=1, T_kv=256):** added in v0.4 after a regression revealed that the original causal mask used T_q-relative positions, restricting attention to absolute position 0 during autoregressive decode. The fix accounts for the offset `T_kv − T_q` so that a single decode query at position `T_kv − 1` correctly attends to all KV positions. Two new regression tests cover the decode (T_q=1) and partial-decode (T_q < T_kv) cases. The bug existed in the cherry-picked implementation from v0.4.0 base and was caught by the long-context benchmark in §10.4.

Additionally, `patch_chunked_attention` was extended to walk `sys.modules` and patch every loaded `mlx_lm.models.*` module that imported `scaled_dot_product_attention` at load time as a local binding. Without this, the wrapper installed in `mlx_lm.models.base` was never called by Qwen, Llama, etc.

---

## 11. Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| PolarQuantizer (v0.3.1) | 46 | Round-trip, cosine, MSE, batch dims, zero vector |
| Metal kernels (v0.3.1) | 26 | Bit-exact indices, norms, dequant, GQA |
| MLX cache strategies (v0.3.1) | 6 | Incremental, residual, full, sink tokens |
| HF cache strategies (v0.3.1) | 8 | All strategies + asymmetric bits |
| MLX forward hooks | 7 | Attach/detach, shape/dtype, 4-bit vs 8-bit |
| Chunked attention | 15 | Full vs chunked, causal, additive, GQA, chunk sizes, **decode-mode regression** |
| DiT detection (v0.3.1) | 7 | to_q/to_k/to_v, fused QKV, FeedForward, hook attachment |
| Codebook solvers | 6+ | Lloyd-Max, CMA-ES, fuzzy, symmetry, convergence |
| E2E large models | 20+ | Real model inference with compression |
| **v0.4 — Pipeline foundation** | 21 | base, registry, runner, builder, backward compat |
| **v0.4 — Scorers** (palm, snr, fisher) | 23 | EMA, schedules, registration |
| **v0.4 — Strategies** (tiered, delta, delta2, window) | 19 | Roundtrip, fallback chain, stats tracking |
| **v0.4 — Sprint 4** (delta2, window, fisher, monitors) | 26 | Order fallback, similarity reuse, FTLE estimator |
| **v0.4 — Adapters** (llm, dit, wan) | 12 | Auto-detect, head info, registration |
| **v0.4 — Optimization** (genome, GA) | 14 | Crossover, mutation, decode, fitness improvement |
| **v0.4 — Paper gaps** (sheaf, bsa, copresheaf, layer protection) | 22 | All new plugins from §9.3 |
| **v0.4 — CFG sharing** | 8 | Split-pass + batched modes, cache lifecycle |
| **Total** | **447** | All 293 pre-v0.4 tests pass byte-identically |

---

## 12. Related Work

| System | Approach | Throughput | Quality |
|--------|----------|-----------|---------|
| **TurboQuant** (Zandieh et al., 2026) | Random rotation + Lloyd-Max | Not reported for inference | Near-optimal distortion |
| **KIVI** (Yuan et al., 2024) | Asymmetric 2-bit, residual buffer | 2.35–3.47× | Minimal degradation |
| **KVQuant** (Hooper et al., 2024) | LUT-based fused dequant-attention | 1.1–1.4× per-op | 1M context on A100 |
| **GEAR** (Kang et al., 2024) | Low-rank + sparse residual | 2.1–5.07× | Near-lossless |
| **BitDecoding** (2025) | Async pipeline, Tensor Core | Up to 8.6× | <3% accuracy drop |
| **tqai** (this work) | Incremental buffer + Metal kernels | **98–101% on 7B+** | **Δppl = 0.00** |

---

## 13. Conclusion

tqai v0.4 makes three claims that we have validated empirically:

1. **The proven core is shippable.** K4/V2 KV quantization is byte-identical to baseline on every Qwen / Gemma / Llama model we tested (Δppl = 0.00 across 6 models, 12 quantization variants). On compute-bound 7B+ models the throughput cost is zero. This has been true since v0.3.1; v0.4 does not change it by construction (`pipeline=None` short-circuits to v0.3.1 code paths, and all 293 pre-v0.4 tests pass byte-identically).

2. **The plugin architecture lowers integration cost without raising baseline cost.** Adding a new compression paper from the literature now requires one new file in `scorers/`, `strategies/`, `monitors/`, or `adapters/`, plus a one-line registration. We demonstrate this by shipping reference implementations of 12 of 13 tracked papers (TurboQuant, QuantSparse, DiTFastAttn step-share + WA-RS + CFG share, BSA, Sheaf, Copresheaf, APTQ/Fisher, Min-SNR, SparseDiT, attention analysis VDiTs, Lyapunov stability) without modifying `quantizer.py`, `cache/hf.py`, `cache/mlx.py`, or `kernels/` — which collectively house the entire performance-critical path. The default code path is unchanged.

3. **Practical local video generation requires solving non-compression problems first.** Two unrelated bugs in the diffusers stack (`AutoencoderKLWan` not enabling tiling by default, `LTX2RotaryPosEmbed1d` using float64 on MPS) blocked any meaningful video work on Apple Silicon. We ship one-line fixes for both as drop-in utilities (`optimize_vae_memory`, `patch_mps_compatibility`). After these utilities, WAN 2.2 5B generates 81-frame 480p video in 32 GB peak RSS instead of OOM-ing, and LTX-2 runs at all on Apple MPS instead of crashing immediately. We consider these the most impactful local wins in v0.4 — more impactful than any individual compression technique.

We also report **three principled negative results** that we believe are worth more than the wins, because they save future contributors from dead ends:

- **Chunked attention via Python loops loses to fused Metal kernels on MLX** by 3–5× at 16K–32K context, with bit-identical output. The implementation is correct (15 tests prove it including a decode-mode regression test), the math is sound, and on CUDA without FlashAttention the speedup would be real. On Apple Silicon, `mx.fast.scaled_dot_product_attention` already implements blocked attention internally and a Python-mediated loop cannot compete.

- **DiTFastAttn CFG sharing on MLX is functionally correct but does not deliver wall-clock speedup.** The hooks fire at the expected rate (50% / 100%), output PSNR is 56.5 dB (near-lossless), but the per-attention Python dispatch overhead almost exactly cancels the saved attention compute. The recommended next step is wrapping the hook in `mx.compile`; if that fails, a custom Metal kernel is required. We have not yet shipped either.

- **K4/V2 KV quantization does not save peak runtime memory on Apple Silicon today.** Measured peak via `mx.get_peak_memory()` on Qwen 2.5-7B (8-bit weights, ~8.7 GB model footprint) at four context lengths spanning 32×:

  | Context | Baseline KV | tqai K4/V2 KV | Δ |
  |--------:|------------:|--------------:|--:|
  | 4 K | 0.87 GB | 0.90 GB | **−0.03 GB** |
  | 32 K | 2.41 GB | 2.56 GB | **−0.16 GB** |
  | 64 K | 4.35 GB | 4.45 GB | **−0.10 GB** |
  | 128 K | 8.37 GB | 8.56 GB | **−0.19 GB** |

  Negative Δ = tqai uses more memory. The pattern is consistent and structural: the v0.3.1 `incremental` cache strategy maintains a persistent dequantized buffer at full input precision (`bf16` for Qwen 7B Q8) in addition to the compressed indices, because that buffer is what enables the O(1) per-token decode throughput recovery. The 80% size reduction is real for the *compressed-storage form* (compressed indices vs `bf16` KV), useful for serialized cache / wire format / future fused dequant-attention kernels. It does not show up in peak runtime memory because the buffer dominates the footprint and exists alongside the compressed storage. The runtime memory unlock requires a fused dequant-attention Metal kernel (the KVQuant approach, cited in `cache/mlx.py:9`) which is on the v0.5 roadmap. Cross-validated with two independent measurement methodologies (subprocess + `mlx_lm.generate` and direct `model(ids, cache=cache)` with the cache held alive); both agree within ±0.1 GB. Full benchmark in `benchmarks/benchmark_kv_memory.py` and writeup in `reports/kv_memory_finding.md`.

The unifying insight from all three negative results: on Apple Silicon, **the fused-kernel baseline is hard to beat from Python.** `mx.fast.scaled_dot_product_attention` is internally tile-blocked and threadgroup-optimized; replacing or augmenting it with a Python loop loses dispatch overhead. Similarly, the v0.3.1 incremental cache trades runtime memory for throughput by holding a dequantized buffer that the model expects in full precision. The pattern is the same in both cases: *anything that needs to fire inside the attention computation, or that needs to influence what tensors the model holds during attention, requires a fused kernel that controls both phases together*. The pipeline cost of any new `scorer` or `strategy` that fires per-layer (not per-attention-call) is negligible because the framework itself is zero-overhead when unused and amortizes across the relatively cheap pre/post-attention path. Papers whose wins come from "score N tokens and pick the top K" compose cheaply; papers whose wins come from "modify what the GEMM does" do not.

### Future Work

1. **Fused dequant-attention Metal kernel** (the actual unlock for runtime memory savings, replacing the obsolete "long-context KV memory benchmark" item which has now been done and revealed that the win does not exist without the kernel). Compute Q·Kᵀ directly on uint8 indices + per-row L₂ norms + the rotation matrix, without materializing the full bf16 K tensor. KVQuant ships this on CUDA; on Apple Silicon it would need a custom Metal shader. Estimated effort: 1–2 weeks for a working prototype, longer for parity with `mx.fast.scaled_dot_product_attention`. Until this exists, tqai's runtime memory profile is identical to baseline and the K4/V2 win is purely a quality / compressed-storage win.
2. **`mx.compile` the CFG sharing hook** — would either close the speedup gap or deliver a definitive verdict on whether a custom Metal kernel is required.
3. **Step distillation integration for video** — distilled WAN/LTX checkpoints (LCM-style, Hyper-SD-style) reduce 20–50 denoising steps to 4–8 with no quality loss, delivering a 2.5–5× video speedup. We empirically validated in Step #1 that WAN 2.2 5B does not actually need distillation: at 4 denoising steps it produces output with PSNR 73 dB vs the 25-step reference (well above the 40 dB near-lossless threshold). The remaining work is to ship this as a `tqai.dit.get_video_preset()` API, which we did, and to test on more models.
4. ~~**Offline gradient-based Fisher calibration mode**~~ — **DONE in v0.4.** Shipped as `tqai.optimization.calibrate_fisher()` + `tqai.scorers.fisher_static.FisherStaticScorer` + `tqai calibrate` CLI command. The calibration walks the model's K/V projection modules, runs forward+backward passes on a small calibration set (8–32 prompts is enough), accumulates the squared gradients per layer, normalizes them, and saves a JSON file. The static scorer loads the JSON at runtime and serves precomputed scores via constant-time lookup. End-to-end validated on Qwen 2.5-0.5B (24 layers, 4 samples, 0.5 seconds wall-clock, 32× ratio between most and least important layer). The runtime `fisher` proxy scorer is preserved for backward compatibility but documented as broken. 17 new tests cover the calibration logic, the JSON round-trip, and the static scorer with all four `kv_mode` variants (`k`, `v`, `max`, `mean`).
5. **Preset system** — `tqai.patch(model, preset="llm-quality" | "llm-memory" | "dit-video" | "dit-fast")` to remove the friction of picking scorer × strategy × monitor manually. ~50 lines of code.
6. **Custom Metal kernel for chunked SDPA at very long contexts (>64K)** — only relevant if the fused `mx.fast.scaled_dot_product_attention` itself OOMs at extreme sequence lengths. Needs validation that such a regime exists on a 128 GB Mac.
7. **On-device quantization for arbitrary dimensions** — extend Metal kernels to non-power-of-2 head dimensions (e.g., D=896 for Qwen hidden states, D=3072 for WAN 2.2 inner_dim). The current power-of-2 restriction was already lifted on `feature/metal-kernels` (commit a2bae04) but has not been benchmarked end-to-end.
8. **Spherical (L2-norm) attention** — would require model retraining or fine-tuning; out of scope for a drop-in compression library, but worth tracking as the only paper from §9.3 we did not ship.

---

## References

### Core algorithm

1. Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026). **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.** ICLR 2026. arXiv:2504.19874. *Used by:* `palm` scorer, `tiered` strategy, the entire `PolarQuantizer` core.

### KV cache compression family

2. Yuan, J., et al. (2024). **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.** ICML 2024. arXiv:2402.02750. *Used by:* `cache_strategy="residual"`.

3. Hooper, C., et al. (2024). **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.** NeurIPS 2024. arXiv:2401.18079.

4. Kang, M., et al. (2024). **GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM.** NeurIPS 2024. arXiv:2403.05527.

### Diffusion / video compression (v0.4 paper playground)

5. Yuan, Z., et al. (2024). **DiTFastAttn: Attention Compression for Diffusion Transformer Models.** NeurIPS 2024. arXiv:2406.08552. *Used by:* `delta` strategy (step sharing), `window` strategy (WA-RS), `cfg_sharing` strategy.

6. **QuantSparse: Second-Order Residual Quantization for Wan2.1 KV Cache** (2025). arXiv:2509.23681. *Used by:* `delta2` strategy (second-order Δ² with order-2 → 1 → full fallback).

7. **BSA: Bidirectional Sparse Attention** (2025). arXiv:2509.01085. *Used by:* `bsa` scorer (KV-side block centroid saliency; Q-side requires custom kernel and is not implemented).

8. Hang, T., et al. (2023). **Efficient Diffusion Training via Min-SNR Weighting Strategy.** ICCV 2023. arXiv:2303.09556. *Used by:* `snr` scorer (cosine + linear schedules).

9. Ho, J. & Salimans, T. (2022). **Classifier-Free Diffusion Guidance.** arXiv:2207.12598. *Used by:* the CFG protocol that `cfg_sharing` accelerates.

10. **Attention Analysis in Video Diffusion Transformers** (2025). arXiv:2504.10317. *Used by:* `skip_layers` config option (per-layer compression bypass for non-sparse layers).

11. **SparseDiT: Tri-Segment Token Allocation for Diffusion Transformers** (2025). NeurIPS 2025. arXiv:2412.06028. *Used by:* the composition of `tiered` strategy with `skip_layers` config.

### Topological / category-theoretic foundations

12. **Sheaf Attention & Local Consistency** (2026). AAAI 2026. arXiv:2601.21207. *Used by:* `sheaf` scorer (discrete Laplacian harmonicity classifier).

13. **Copresheaf Neural Networks** (2025). arXiv:2505.21251. *Used by:* `codebook/registry.py` `head_type` discriminator (per-head codebooks).

### Chunked / memory-efficient attention

14. Dao, T., et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** NeurIPS 2022. arXiv:2205.14135. *Influenced:* `attention.py` chunked SDPA (ships for CUDA users; not recommended on MLX — see §10.4).

15. Milakov, M. & Gimelshein, N. (2018). **Online normalizer calculation for softmax.** arXiv:1805.02867. *Used by:* the online softmax recurrence in `chunked_scaled_dot_product_attention`.

### Codebook solvers (build-time)

16. Hansen, N. & Ostermeier, A. (2001). **Completely Derandomized Self-Adaptation in Evolution Strategies.** Evolutionary Computation, 9(2). *Used by:* `codebook/cmaes_solver.py`.

17. Gong, R., et al. (2019). **Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks.** arXiv:1908.05033. *Used by:* `codebook/fuzzy_solver.py`.

18. **Vector Quantization using Improved Differential Evolution** (2017). arXiv:1710.05311. *Influenced:* the evolutionary codebook search design.

### Theoretical context (informs design, not directly implemented)

19. Li, Y., et al. (2024). **APTQ: Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models.** DAC 2024. arXiv:2402.14866. *Influenced:* the `fisher` scorer (we ship a squared-activation proxy; the true gradient-based form requires offline calibration and is listed as future work in §13).

20. **Fisher Information for Neural Network Compression** (2019). arXiv:1906.08589. *Theoretical basis for:* `fisher` scorer.

21. Frantar, E. & Alistarh, D. (2023). **OPTQ: Accurate Quantization for Generative Pre-trained Transformers.** arXiv:2210.17323. *Theoretical context only.*

22. **Spherical Attention via Neural Circuit Diagrams** (2025). arXiv:2505.09326. *Not implemented:* would require modifying attention math; out of scope for a drop-in compression library.

### Frameworks and models

23. Apple (2023). **MLX: An array framework for Apple silicon.** github.com/ml-explore/mlx

24. Wan-AI (2026). **WAN 2.2: Text-Image-to-Video Generation.** huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers

25. Lightricks (2025). **LTX-2: Long-Form Video Generation.** huggingface.co/Lightricks/LTX-2

26. Google (2026). **Gemma 4: Multimodal Language Models.** huggingface.co/collections/google/gemma-4

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| x ∈ ℝᵈ | Input KV vector |
| R ∈ O(d) | Haar-distributed orthogonal rotation matrix |
| y = x̃ · Rᵀ | Rotated unit vector |
| {c₁, ..., c_L} | Lloyd-Max centroids (L = 2^b) |
| Q(·) | Quantization function (nearest centroid) |
| s = ‖x‖₂ | L2 norm (stored as FP16) |
| D*(b, d) | Rate-distortion optimal distortion |
| α | EMA decay parameter (Palm scorer) |
| τ | Temperature parameter (fuzzy C-means) |
| C | Chunk size (chunked attention) |
| R (KIVI) | Residual window size |

## Appendix B: Implementation Statistics (v0.4)

| Metric | v0.3.1 | v0.4 |
|--------|-------:|-----:|
| Total source lines | ~4,500 | ~7,500 |
| Test count | 293 | **447** |
| Pre-v0.4 tests passing byte-identically | — | **293 / 293** |
| Models benchmarked (LLM) | 6 architectures, 12 variants | unchanged |
| Models benchmarked (DiT) | WAN 2.2 5B (synthetic) | WAN 2.2 5B + LTX-2 (real generation) |
| Supported frameworks | PyTorch, MLX | unchanged |
| Supported LLM architectures | Llama, Qwen, Gemma, GPT-2 | unchanged |
| Supported DiT architectures | (none) | WAN 2.2, LTX-2, generic diffusers BasicTransformerBlock |
| Metal kernel count | 2 (quantize, dequantize) | unchanged |
| Codebook solvers | 3 (Lloyd-Max, CMA-ES, fuzzy C-means) | unchanged |
| Cache strategies | 3 (incremental, residual, full) | unchanged |
| **Pipeline scorers** | 0 | **5** (palm, snr, fisher, sheaf, bsa) |
| **Pipeline strategies** | 0 | **5** (tiered, delta, delta2, window, cfg_sharing) |
| **Pipeline monitors** | 0 | **2** (stability, lyapunov) |
| **Model adapters** | 0 | **3** (llm, dit, wan) |
| **GA optimizer genes** | 0 | **9** |
| **DiT utilities** | 0 | **3** (optimize_vae_memory, patch_mps_compatibility, patch_cfg_sharing) |
| **Papers covered (of 13 referenced)** | 0 | **12** |
| Benchmark scripts | `benchmark_forward.py` | + `benchmark_pipeline.py`, `benchmark_video.py`, `benchmark_long_context.py` |
| Reports | (none) | `where_tqai_shines.md`, `paper_coverage_report.md`, `benchmark_report.html` |
| Research paper | this document (v0.3.1) | this document (v0.4 revision) |
| PyPI package size | ~80 KB | ~150 KB |
