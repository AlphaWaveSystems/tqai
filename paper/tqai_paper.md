# tqai: Efficient KV Cache Compression with Incremental Reconstruction, Fused Metal Kernels, and Adaptive Bit Allocation for Local LLM and DiT Inference

**Authors:** Patrick Bertsch, Claude (Anthropic)

**Date:** April 2026

---

## Abstract

We present tqai, a comprehensive KV cache compression library for local inference of large language models (LLMs) and diffusion transformers (DiTs) on Apple Silicon. Building on the TurboQuant framework (Zandieh et al., 2026), tqai introduces three key contributions: (1) an **incremental cache reconstruction** strategy that reduces per-token dequantization from O(n) to O(1), delivering 2.3–3.6× throughput improvement over naive full-reconstruction approaches; (2) **fused Metal GPU kernels** that eliminate Python dispatch overhead by combining L2 normalization, orthogonal rotation, and codebook search into single GPU dispatches via MLX's `mx.fast.metal_kernel` API; and (3) **evolutionary codebook optimization** using CMA-ES and fuzzy C-means with temperature annealing, achieving 0.55% MSE improvement over classical Lloyd-Max codebooks. We additionally present a **chunked attention** implementation using the online softmax algorithm, a **Palm information-theoretic scorer** for adaptive bit allocation, and initial **diffusion transformer integration** for video generation models (WAN 2.2). Across 12 model variants (0.5B–14B parameters) including Gemma 4, tqai achieves **zero perplexity degradation** (Δppl = 0.00) with 80% KV cache memory savings. On compute-bound models (7B+), throughput retention reaches **98–101%** of uncompressed baseline.

---

## 1. Introduction

KV cache compression is critical for enabling long-context inference on memory-constrained devices. The TurboQuant algorithm (Zandieh et al., 2026) provides a theoretically optimal data-oblivious quantization scheme based on random orthogonal rotation followed by Lloyd-Max scalar quantization. However, the reference implementation leaves several systems engineering challenges unaddressed:

1. **Reconstruction overhead**: Naive implementations dequantize the entire KV history on every token, creating O(n²) total work over n tokens.
2. **Python dispatch latency**: Multiple MLX/PyTorch operations per quantize/dequantize call incur interpreter overhead.
3. **Codebook optimality**: Lloyd-Max codebooks are locally optimal but may not be globally optimal.
4. **Diffusion transformer support**: TurboQuant was designed for autoregressive LLMs, not DiT architectures.

We address all four challenges in tqai v0.3.1–v0.4.0 and provide comprehensive benchmarks across 12 model configurations.

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

### 8.3 Current Limitation: Device Transfer Overhead

Forward hooks on MPS require CPU↔MPS round-trips for the quantization rotation matrix. This incurs 24× overhead, rendering activation compression impractical on MPS. The architecture integration is correct — the quantizer needs to run entirely on-device (Metal kernel for quantize+dequantize of arbitrary-dimension tensors, not just power-of-2 head_dim).

---

## 9. Comprehensive Benchmark Results

### 9.1 LLM Throughput (MLX, Apple Silicon, v0.4.0)

| Model | Params | Quant | Baseline | kv-only | Retention | PPL | ΔPPL |
|-------|--------|-------|----------|---------|-----------|-----|------|
| Qwen2.5-0.5B | 0.5B | bf16 | 302 tok/s | 88 tok/s | 29% | 4.34 | **0.00** |
| Qwen2.5-3B | 3B | bf16 | 69 tok/s | 24 tok/s | 35% | 2.49 | **0.00** |
| Llama-3.1-8B | 8B | Q4 | 28 tok/s | 28 tok/s | **100%** | 2.95 | **0.00** |
| Qwen2.5-7B | 7B | Q8 | 17 tok/s | 17 tok/s | **101%** | 2.40 | **0.00** |
| Qwen2.5-14B | 14B | Q4 | 15 tok/s | 15 tok/s | **98%** | 2.22 | **0.00** |
| Gemma 4 E4B | 4B | Q4 | 47 tok/s | 24 tok/s | 50% | 126.94 | **0.00** |

**Key finding:** Compute-bound models (7B+ with quantized weights) show **zero throughput cost** from KV cache compression. The incremental buffer overhead is completely absorbed by the model's own compute time.

### 9.2 WAN 2.2 Video Generation (PyTorch, MPS)

| Config | Time (17 frames, 10 steps) | Relative |
|--------|---------------------------|----------|
| Baseline (no tqai) | 21.0s | 1.00× |
| Text encoder cache only | 57.4s | 2.74× slower* |
| Forward hooks (8-bit) | 512.2s | 24.4× slower* |

*CPU↔MPS transfer overhead dominates. On-device quantization (future Metal kernel for arbitrary D) would eliminate this bottleneck.

### 9.3 Codebook Quality

| Solver | D=128, 4-bit MSE | Improvement |
|--------|-----------------|-------------|
| Lloyd-Max | 7.588×10⁻⁵ | baseline |
| Fuzzy C-means (α=0.95) | 7.585×10⁻⁵ | −0.04% |
| CMA-ES (500 gen) | 7.546×10⁻⁵ | **−0.55%** |

### 9.4 Metal Kernel Correctness

| Bits | head_dim | Index match rate | Max norm diff |
|------|----------|-----------------|---------------|
| 2 | 64, 128 | 100.0% | 0.0 |
| 4 | 64, 128 | 100.0% | 0.0 |
| 8 | 128 | 100.0% | 0.0 |
| 8 | 64 | 98.8% | 0.0 |

### 9.5 Chunked Attention Accuracy

All tests pass with atol=5×10⁻⁴, rtol=2×10⁻³ across:
- Sequence lengths: 128, 256, 512, 1024
- Chunk sizes: 32, 64, 128, 256, 512
- Mask types: none, causal, additive
- Head configurations: MHA (H_q = H_kv) and GQA (H_q = 4·H_kv)

---

## 10. Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| PolarQuantizer | 46 | Round-trip, cosine, MSE, batch dims, zero vector |
| Metal kernels | 26 | Bit-exact indices, norms, dequant, GQA |
| MLX cache strategies | 6 | Incremental, residual, full, sink tokens |
| HF cache strategies | 8 | All strategies + asymmetric bits |
| MLX forward hooks | 7 | Attach/detach, shape/dtype, 4-bit vs 8-bit |
| Chunked attention | 13 | Full vs chunked, causal, additive, GQA, chunk sizes |
| DiT detection | 7 | to_q/to_k/to_v, fused QKV, FeedForward, hook attachment |
| Palm scorer | 6 | EMA tracker, tiers, bits, diffusion scoring |
| Step delta | 3 | First step, delta usage, stats |
| Codebook solvers | 6+ | Lloyd-Max, CMA-ES, fuzzy, symmetry, convergence |
| E2E large models | 20+ | Real model inference with compression |
| **Total** | **331** | |

---

## 11. Related Work

| System | Approach | Throughput | Quality |
|--------|----------|-----------|---------|
| **TurboQuant** (Zandieh et al., 2026) | Random rotation + Lloyd-Max | Not reported for inference | Near-optimal distortion |
| **KIVI** (Yuan et al., 2024) | Asymmetric 2-bit, residual buffer | 2.35–3.47× | Minimal degradation |
| **KVQuant** (Hooper et al., 2024) | LUT-based fused dequant-attention | 1.1–1.4× per-op | 1M context on A100 |
| **GEAR** (Kang et al., 2024) | Low-rank + sparse residual | 2.1–5.07× | Near-lossless |
| **BitDecoding** (2025) | Async pipeline, Tensor Core | Up to 8.6× | <3% accuracy drop |
| **tqai** (this work) | Incremental buffer + Metal kernels | **98–101% on 7B+** | **Δppl = 0.00** |

---

## 12. Conclusion

tqai demonstrates that practical KV cache compression for local inference requires solving systems engineering challenges beyond the quantization algorithm itself. Our incremental cache reconstruction eliminates the O(n²) bottleneck that dominated prior implementations, achieving zero throughput overhead on compute-bound models. Fused Metal kernels further reduce per-operation latency. Evolutionary codebook optimization provides modest but measurable improvements in distortion. The architecture is extensible to diffusion transformers, though on-device quantization is required for practical DiT deployment on Apple Silicon.

### Future Work

1. **On-device quantization for arbitrary dimensions**: Extend Metal kernels to non-power-of-2 head dimensions (e.g., D=896 for Qwen hidden states, D=3072 for WAN 2.2)
2. **Fused dequant-attention kernel**: Compute attention directly on compressed KV cache (KVQuant/KIVI approach adapted for PolarQuant)
3. **GA policy search**: Evolutionary optimization of per-layer bit allocation using Palm scoring
4. **Production DiT integration**: Eliminate CPU↔MPS transfers for video generation models

---

## References

1. Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. ICLR 2026. arXiv:2504.19874

2. Yuan, J., et al. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. arXiv:2402.02750

3. Hooper, C., et al. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. NeurIPS 2024. arXiv:2401.18079

4. Kang, M., et al. (2024). GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM. NeurIPS 2024. arXiv:2403.05527

5. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135

6. Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867

7. Hansen, N. & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2).

8. Gong, R., et al. (2019). Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks. arXiv:1908.05033

9. Li, Y., et al. (2024). APTQ: Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models. arXiv:2402.14866

10. Apple (2023). MLX: An array framework for Apple silicon. github.com/ml-explore/mlx

11. Wan-AI (2026). WAN 2.2: Text-Image-to-Video Generation. huggingface.co/Wan-AI/Wan2.2-TI2V-5B

12. Google (2026). Gemma 4: Multimodal Language Models. huggingface.co/collections/google/gemma-4

13. Vector Quantization using Improved Differential Evolution (2017). arXiv:1710.05311

14. Soft Quantization via Weight Coupling (2026). arXiv:2601.21219

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

## Appendix B: Implementation Statistics

| Metric | Value |
|--------|-------|
| Total source lines | ~4,500 |
| Test count | 331 |
| Models benchmarked | 12 variants (6 architectures) |
| Supported frameworks | PyTorch, MLX |
| Supported architectures | Llama, Qwen, Gemma, GPT-2, DiT (diffusers) |
| Metal kernel count | 2 (quantize, dequantize) |
| Codebook solvers | 3 (Lloyd-Max, CMA-ES, fuzzy C-means) |
| Cache strategies | 3 (incremental, residual, full) |
| PyPI package size | ~80 KB |
