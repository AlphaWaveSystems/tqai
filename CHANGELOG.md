# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-04-22

### Added — Batched multi-head Metal kernels + end-to-end validation

- **`metal_batched_score_keys` / `metal_batched_aggregate_values`** —
  batched Metal kernels that process all KV heads × GQA repeats in a
  single dispatch.  Replaces the v0.5 Python head loop (2×n_q_heads
  dispatches) with exactly 2 total dispatches per decode step.

- **`batched_fused_polar_decode_v2` / `batched_fused_rotor_decode_v2`** —
  high-level batched decode functions for B=1, T_q=1 hot path.

- **`compute_fused_attention`** now auto-selects the v0.6 batched path
  for B=1, T_q=1 decode.  Falls back to the per-head loop for prefill
  (T_q > 1) or multi-batch.

- **Compressed strategy wired into `tqai.patch()`**: calling
  `tqai.patch(model, cache_strategy="compressed")` now automatically
  patches SDPA for fused decode.  `tqai.unpatch()` cleans up correctly.

- **`benchmarks/eval_e2e.py`** — end-to-end evaluation script measuring
  generation quality and autoregressive perplexity across strategies.

- **Benchmark Section 3** expanded with 5 model geometries (Llama-3-8B,
  Llama-3-70B, Mistral-7B, Qwen2-7B) and context lengths up to 131K.
  **Section 3b** added: memory capacity analysis across device budgets.

### Changed

- **Default bits changed to 8/8** (was 4/2). End-to-end evaluation on
  Qwen2.5-3B showed that 4-bit quantization causes 2.5× PPL degradation
  due to error compounding across layers.  At 8-bit the quantization is
  near-lossless (PPL +0.0%) while still achieving 2× KV cache capacity
  via the norm-direction decomposition (uint8 indices + fp16 norms vs
  fp16 full vectors).

  Bit-width quality ladder (Qwen2.5-3B-Instruct-4bit, 256 tokens):
  ```
  8-bit: PPL 3.33 (+0.0%)  — lossless
  6-bit: PPL 3.36 (+0.9%)  — near-lossless
  4-bit: PPL 8.32 (+150%)  — degraded
  2-bit: PPL 952   (286×)  — unusable
  ```

## [0.5.0] - 2026-04-22

### Added — Fused dequant-attention and bit-packing

- **`packing.py`** — `pack(indices, bits)` / `unpack(packed, bits, shape)`:
  bit-pack Lloyd-Max indices at their actual bit-width instead of wasting
  8 bits per index in a uint8 container.

  Compression vs uint8 storage:
  - 2-bit: 4× (4 indices per byte)
  - 4-bit: 2× (2 indices per byte — standard case)
  - 3-bit / 6-bit: bit-stream packing (non-byte-aligned)

  All functions are pure NumPy, backend-agnostic, and designed for
  serialization and DRAM storage.  The hot-path (GPU quantize/dequantize)
  still operates on uint8 in GPU memory.

- **Fused dequant-attention kernels** (`kernels/__init__.py`):
  `metal_score_keys` and `metal_aggregate_values` — two new Metal MSL
  kernels that perform attention over compressed KV caches without
  ever materializing float32 K or V buffers.

  Mathematical basis:
  ```
  score[k] = norm_k * dot(R @ q, centroids[k_indices[k]])
  output    = R.T @ sum_k(weights[k] * norm_k * centroids[v_indices[k]])
  ```
  Prerotating `q` once (O(D²) or O(D) for RotorQuant) makes each key
  score a simple gather-then-dot with no per-key rotation.

  DRAM bandwidth at 4-bit vs float16 KV: **~4× reduction**
  (uint8 + float16 norms vs float16 full vectors).

- **`attention_fused.py`** — high-level fused attention API:
  - `fused_polar_decode_step(q, k_idx, k_norms, v_idx, v_norms, R, centroids, scale)`
    — single-head decode over PolarQuant-compressed cache.
  - `fused_rotor_decode_step(...)` — same for RotorQuant; rotation cost O(D).
  - `batched_fused_polar_decode(queries, ...)` — multi-head / GQA decode.

- **Tests**: 139 packing tests + 14 fused attention tests (total: 760 passing).

### Skipped — entropy coding

Measured empirical entropy of Lloyd-Max indices across all bit-widths
and head dimensions (both PolarQuant and RotorQuant):

| bits | H (bits) | bound | ratio | EC ceiling |
|------|----------|-------|-------|------------|
| 2    | 1.915    | 2     | 95.8% | 4.2%       |
| 3    | 2.828    | 3     | 94.3% | 5.7%       |
| 4    | 3.766    | 4     | 94.1% | 5.9%       |

Lloyd-Max at 94% entropy efficiency leaves only 4–6% on the table.
Entropy coding (ANS, Huffman) is not worth the complexity — skipped.
LZ/dictionary compression gains nothing because the rotation deliberately
decorrelates indices, producing near-i.i.d. uniform data.

## [0.4.1] - 2026-04-21

### Added — RotorQuantizer: Clifford rotor block-diagonal KV compression

- **`quantizer_rotor.py`** — `RotorQuantizer`, a drop-in replacement for
  `PolarQuantizer` that replaces the dense d×d Haar rotation with
  block-diagonal 3×3 quaternion rotations (Clifford rotors from Cl(3,0)).
  Identical Lloyd-Max codebooks and norm-preservation logic; identical
  reconstruction quality (CosSim, NMSE) at all tested bit widths and model
  profiles.

  Key properties:
  - O(d) rotation cost vs O(d²) for PolarQuantizer — 44× fewer rotation
    parameters (128 vs 16,384 at d=128)
  - Supports any `head_dim` including non-multiples of 3 (remainder
    dimensions pass through unrotated)
  - Same `quantize(x) → (indices, norms)` / `dequantize(indices, norms) → x`
    API as `PolarQuantizer`
  - Fallback to numpy einsum on PyTorch backend

- **Fused Metal kernels** (`kernels/__init__.py`) — `metal_rotor_quantize`
  and `metal_rotor_dequantize`: two new MSL kernels that fuse L2-norm +
  block-diagonal rotor rotation + centroid argmin into single GPU dispatches.
  Auto-activated on MLX ≥ 0.16 + Metal. Benchmarked on M5 Max:

  | d | PolarQuant (Metal) | RotorQuant (Metal) | Speedup |
  |---|---|---|---|
  | 64 | 0.64 ms | 0.34 ms | 1.9× |
  | 128 | 0.67 ms | 0.22 ms | **3.0×** |
  | 256 | 1.54 ms | 0.33 ms | **4.7×** |

  Speedup grows with d because the rotation cost is O(d) vs O(d²) — the
  d=256 advantage (4.7×) demonstrates the asymptotic benefit clearly.

- **RotorQuant pipeline configs** (`benchmarks/benchmark_pipeline.py`) —
  five new configs (`rotorquant+bare`, `rotorquant+tiered`, `rotorquant+delta`,
  `rotorquant+delta2`, `rotorquant+window`) benchmarked against all 7 model
  profiles. Mean NMSE and CosSim are statistically indistinguishable from
  their PolarQuant equivalents across 7 model profiles × 5 steps.

- **Tests** — 122 new tests across two files:
  - `tests/test_rotor_quantizer.py` — shape/dtype contracts, cosine
    similarity, MSE-vs-bits monotonicity, determinism, seed isolation, zero
    vectors, batch dims, remainder dimensions (d=65, d=127), quaternion
    orthogonality invariant, Metal flag detection on MLX vs PyTorch
  - `tests/test_metal_rotor_kernels.py` — Metal vs Python index/norm/
    reconstruction parity, round-trip cosine, batch shapes, zero vector,
    Metal fallback via monkeypatch, high-level API consistency

  Total test count: 577 (up from 455).

Reference: Pope, J.D. (2026). "RotorQuant: Clifford Algebra Vector
Quantization for LLM KV Cache Compression." https://www.scrya.com/rotorquant/

## [0.4.0] - 2026-04-06

The v0.4 cycle turns tqai from a single-algorithm KV cache compressor
into a **composable middleware framework** plus a **paper playground**
for the recent KV / attention compression literature. The proven v0.3.1
core (PolarQuantizer, Lloyd-Max codebooks, cache strategies — all with
Δppl=0.00 across 6 models) is **frozen**; every new technique ships as
a single plugin file under `scorers/`, `strategies/`, `monitors/`, or
`adapters/` without touching core code. Default path (`pipeline=None`)
is byte-identical to v0.3.1: zero overhead, all 293 pre-v0.4 tests pass
unchanged.

This release also corrects a major framing error in the v0.3.1 docs:
**K4/V2 KV quantization does NOT save peak runtime memory on Apple
Silicon today.** The 80% number is real for compressed-storage size
(serialized cache, wire format, foundation for a future fused
dequant-attention kernel), but the v0.3.1 incremental cache strategy
holds a persistent dequantized buffer at full input precision in
addition to the compressed indices, so peak memory is unchanged. The
quality preservation (Δppl=0.00, byte-identical output) is unaffected.
Full benchmark and explanation in `reports/kv_memory_finding.md`.

### Added — Composable middleware pipeline (PR #2)

- **`pipeline/`** — `Scorer` / `CompressionStrategy` / `Monitor` /
  `ModelAdapter` protocols + name-based registry + `CompressionPipeline`
  runner. The default path (`pipeline=None`) short-circuits to direct
  `PolarQuantizer.quantize()` calls.
- **`scorers/`** — 5 plugins:
  - `palm` — TurboQuant EMA novelty scoring
  - `snr` — Min-SNR diffusion schedule (cosine + linear)
  - `fisher` — runtime squared-activation proxy (documented as broken; see `fisher_static` in PR #5)
  - `sheaf` — discrete Laplacian harmonicity classifier (Sheaf Attention, AAAI 2026)
  - `bsa` — block centroid saliency (BSA, arXiv:2509.01085, KV side)
- **`strategies/`** — 5 plugins:
  - `tiered` — dual-quantizer routing by score (now actually uses `quantizer_low`, fixed in PR #2)
  - `delta` — first-order inter-step Δ (DiTFastAttn step sharing)
  - `delta2` — second-order Δ² with order-2 → 1 → full automatic fallback (QuantSparse, arXiv:2509.23681)
  - `window` — similarity-based attention output cache (DiTFastAttn WA-RS)
  - `cfg_sharing` — split-pass (WAN) and batched (LTX-2) CFG sharing modes (DiTFastAttn)
- **`monitors/`** — `stability` (entropy-shift detection) + `lyapunov` (FTLE divergence)
- **`adapters/`** — `llm` / `dit` / `wan` model family adapters
- **`optimization/`** — GA policy search (`PolicyGenome` + `GASearch`) over the discrete pipeline configuration space
- **`dit/`** — production diffusers integration:
  - `optimize_vae_memory(pipe)` — enables VAE tiling/slicing, prevents the 100 GB+ peak that otherwise OOMs WAN 2.2 5B at 81-frame video on a 128 GB Mac
  - `patch_mps_compatibility(pipe)` — fixes the LTX-2 RoPE float64 crash on Apple MPS
  - `patch_cfg_sharing(pipe)` — installs CFG sharing hooks
- **Chunked attention** — `attention.py` chunked SDPA via online softmax. **Documented as honest negative result on MLX** — fused `mx.fast.scaled_dot_product_attention` is 3-5× faster at 16K-32K context. Ships for CUDA users without FlashAttention.
- **MLX forward hooks** — `MLXForwardCompressionHooks` via module replacement (MLX has no `register_forward_pre_hook`). Parity with PyTorch path on Apple Silicon.
- **Per-head-type codebook registry** — `codebook/registry.py` accepts a `head_type` discriminator (copresheaf-inspired). No specialized codebooks shipped yet but the slot is in place.
- **Skip-layers config** — `pipeline.skip_layers` lets users protect specific attention layers from any pipeline middleware (per arXiv:2504.10317 finding that some layers must not be compressed).
- **CFG sharing strategy** — `strategies/cfg_sharing.py` for diffusers video pipelines.
- **Research paper update** — `paper/tqai_paper.md` and `.tex` rewritten to reflect v0.4: composable pipeline section, paper playground section, production diffusers integration, three principled negative results, references reorganized into 7 categories.

### Added — Few-step video presets (PR #3)

- **`tqai.dit.get_video_preset(pipe, mode)`** — returns a verified
  `VideoPreset(num_inference_steps, guidance_scale)` for `WanPipeline`
  and `LTX2Pipeline`. Four modes per pipeline:
  - `quality`: model card default (25 steps for WAN, 30 for LTX-2)
  - `balanced`: half the baseline steps
  - `fast`: quarter (8 steps)
  - `draft`: minimum viable (4 steps)
- **`benchmarks/benchmark_video_steps.py`** — sweeps all 4 presets and
  measures PSNR vs the quality reference.
- **Empirical result on WAN 2.2 5B (33 frames, 480x832):** at 4 denoising
  steps the output has PSNR 73.1 dB vs the 25-step reference — well above
  the 40 dB near-lossless threshold. **Modern flow-matching video models
  do not need distillation to handle few-step inference**; their
  underlying ODE solvers degrade gracefully on their own. 1.66× wall-clock
  speedup, perceptually identical output.

### Added — KV memory benchmark + honest correction (PR #4)

- **`benchmarks/benchmark_kv_memory.py`** — subprocess-isolated benchmark
  that uses MLX-native memory APIs (`mx.get_active_memory()` /
  `mx.get_peak_memory()` / `mx.reset_peak_memory()`) instead of
  `resource.getrusage().ru_maxrss` (which is the lifetime maximum and
  doesn't reset between configurations). Each (context, config) pair
  runs in its own Python subprocess so MLX memory state is fresh.
- **`reports/kv_memory_finding.md`** — full writeup of the finding,
  cross-validated with two independent measurement methodologies.
- **Documentation corrections**: README, paper (md + tex), and
  `where_tqai_shines.md` all updated to reframe the "80% memory savings"
  claim as "80% reduction in compressed-storage KV cache size", with an
  explicit note that runtime peak memory is unchanged on Apple Silicon
  and a pointer to the writeup.

### Added — Offline Fisher Information calibration (PR #5)

- **`tqai.optimization.calibrate_fisher()`** — runs forward+backward
  passes on a small calibration set (8-32 prompts), accumulates squared
  gradients per attention layer's K and V projection weights, averages
  across samples, and saves a `FisherCalibration` JSON. The actual
  Fisher Information diagonal, not the runtime activation proxy.
- **`tqai.scorers.fisher_static.FisherStaticScorer`** — loads the
  calibration JSON at construction, normalizes the chosen Fisher
  values to `[0, 1]`, serves them via constant-time lookup at scoring
  time. Four `kv_mode` options (`k`, `v`, `max`, `mean`). Registered
  as `"fisher_static"` in the scorer registry.
- **`tqai calibrate` CLI** — `tqai calibrate --model X --output X-fisher.json`
  with 16 built-in default prompts and `--prompts-file` for custom sets.
- **End-to-end validated** on real Qwen 2.5-0.5B-Instruct: 24 layers,
  4 samples, 0.5 second wall-clock, 32× ratio between most and least
  important layer's K Fisher.

### Added — Reports

- `reports/where_tqai_shines.md` — honest assessment of where v0.4
  delivers and where it doesn't on Apple Silicon
- `reports/paper_coverage_report.md` — per-paper implementation status
- `reports/benchmark_report.html` — standalone HTML with charts
- `reports/kv_memory_finding.md` — Step #4 KV memory writeup

### Changed

- `tqai.patch()` accepts `pipeline: dict | None`, `chunk_attention: bool`,
  `attention_chunk_size: int`, `kv_compression: bool` parameters
- `TurboQuantConfig` adds `pipeline`, `chunk_attention`,
  `attention_chunk_size`, `kv_compression` fields
- `cache/hf.py` and `cache/mlx.py` add a pipeline dispatch path in the
  incremental and residual strategies (zero overhead when `pipeline=None`)
- `cli.py` adds `tqai plugins`, `tqai calibrate`, and `--scorer` /
  `--strategy` flags on `tqai run`
- `module_utils.py` extended for DiT module detection (BasicTransformerBlock)
- `hooks.py` adds device-handling fix for the rotation matrix on MPS
- `attention.py` chunked SDPA: fixed decode-mode causal mask bug (was
  using T_q-relative positions; now correctly accounts for `T_kv - T_q`
  offset so a single decode query at position `T_kv-1` attends to all
  KV positions). Plus a regression test.
- `attention.py` `patch_chunked_attention()` now walks `sys.modules` and
  patches every loaded `mlx_lm.models.*` module that has a local
  binding to `scaled_dot_product_attention`. Without this the wrapper
  was never called by Qwen / Llama / etc.
- README headline reframed from "80%+ memory savings" to "byte-identical
  output to baseline" with an honest framing callout
- Test suite expanded from 293 to **484 tests**

### Honest negative results (documented, not bugs)

- **Chunked attention via Python loops** loses to MLX's fused
  `mx.fast.scaled_dot_product_attention` by 3-5× at 16K-32K context.
  Output is bit-identical (15 tests prove it including a decode-mode
  regression test). Ships for CUDA users without FlashAttention.
  **Don't enable on MLX.**
- **DiTFastAttn CFG sharing on MLX** is functionally correct (50% /
  100% share rates) and produces near-lossless output (PSNR 56.5 dB)
  but does not deliver wall-clock speedup because the per-attention
  Python dispatch overhead approximately cancels the saved attention
  compute on Apple Silicon's already-fast fused attention kernel.
- **K4/V2 KV quantization does not save peak runtime memory on Apple
  Silicon today.** Within ±0.2 GB of baseline at 4K-128K context.
  The runtime memory unlock requires a fused dequant-attention Metal
  kernel (KVQuant approach), which is on the v0.5 roadmap.
- **Runtime `fisher` scorer (squared-activation proxy)** over-allocates
  bits (NMSE 12× worse than baseline). Use the new offline
  `fisher_static` scorer + `tqai calibrate` workflow instead.

### Roadmap (v0.5 and beyond)

1. **Fused dequant-attention Metal kernel** — the actual unlock for
   runtime memory savings on Apple Silicon. ~1-2 weeks of Metal shader
   work. v0.5 target.
2. **Real-model benchmark of `fisher_static` vs `palm+delta`** — does
   the gradient-based per-layer importance signal actually beat Palm's
   runtime EMA novelty signal in practice?
3. **Pre-shipped calibration JSONs** for popular models (Qwen, Gemma,
   Llama) so users don't have to run `tqai calibrate` themselves.
4. **Preset system for LLM patching** needs redesign now that we know
   K4/V2 doesn't save runtime memory.
5. **`mx.compile` the CFG sharing hooks** — would either close the
   speedup gap or deliver a definitive verdict on whether a custom
   Metal kernel is required.
6. **On-device quantization for arbitrary head dimensions** — extend
   Metal kernels to non-power-of-2 head_dim (e.g., D=896 for Qwen
   hidden states, D=3072 for WAN 2.2 inner_dim).

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
