# Step #4 Finding: K4/V2 Does NOT Save Peak Runtime Memory on Apple Silicon

**Generated:** 2026-04-06
**Branch:** `feature/long-context-kv-memory`
**Benchmark:** `benchmarks/benchmark_kv_memory.py`
**Result file:** `benchmarks/results/kv_memory.json`

## TL;DR

I went into Step #4 expecting to validate the headline v0.3.1 win
("K4/V2 saves 80% of KV cache memory") at the scale where it matters
locally — 100K+ context. Instead, the benchmark reveals that **K4/V2
saves zero peak runtime memory on Apple Silicon**, and actually uses
**0.1–0.2 GB more** than baseline at every context length we tested.

This is a major correction to how tqai's headline claim should be
framed. The K4/V2 win is real but it is a **storage/wire/quality** win,
not a **peak runtime memory** win.

## The numbers

Qwen 2.5-7B-Instruct-8bit on Apple Silicon, measured with
`mx.get_peak_memory()` inside an isolated subprocess per (context,
config) pair, default `cache_strategy="incremental"`:

| Context | Model load | Baseline prefill peak | tqai K4/V2 prefill peak | KV growth (baseline) | KV growth (tqai) | Savings |
|--------:|-----------:|----------------------:|------------------------:|---------------------:|-----------------:|--------:|
| 4 K | 8.10 GB | 8.97 GB | 9.00 GB | 0.87 GB | 0.90 GB | **−0.03 GB** |
| 32 K | 8.10 GB | 10.50 GB | 10.66 GB | 2.41 GB | 2.56 GB | **−0.16 GB** |
| 64 K | 8.10 GB | 12.44 GB | 12.55 GB | 4.35 GB | 4.45 GB | **−0.10 GB** |
| 128 K | 8.10 GB | 16.46 GB | 16.65 GB | 8.37 GB | 8.56 GB | **−0.19 GB** |

**Negative savings = tqai uses more memory than baseline.**

The pattern is consistent across four context lengths spanning 32×.
This is not noise — it's a structural property of the v0.3.1
incremental cache strategy.

## Why this happens

Read `src/tqai/cache/mlx.py`. The `incremental` cache strategy (the
default since v0.3.1) maintains a **persistent dequantized buffer**
at full input precision (`bf16` for Qwen 7B Q8):

```python
# Lines 75-77
# Incremental dequantized buffer (strategies: incremental, residual)
self._k_buffer: Any | None = None  # [1, H, seq, D] in input dtype
self._v_buffer: Any | None = None

# Lines 178-180 (in _update_incremental):
self._k_buffer = k_dequant   # full-precision tensor stored permanently
self._v_buffer = v_dequant
```

So when a new prefill batch arrives:

1. The original input KV tensors arrive (~size N)
2. tqai quantizes them to compressed indices (~size N/5)
3. tqai immediately dequantizes back to bf16 and **stores the result
   in `_k_buffer` / `_v_buffer`** (~size N)
4. The compressed indices are kept in the layer's quantizer state
   (~size N/5)
5. Plus rotation matrices and codebook lookup tables (~tens of MB
   per layer)

At the peak of step 3, **both the dequantized buffer (N) AND the
compressed indices (N/5) coexist in memory**, plus the constant
overhead. Total: ~1.2N + constant overhead. Baseline is just N
(the original input KV that the model holds anyway).

The compressed storage exists *in addition to* the dequantized buffer,
not *instead of* it. This is by design — it's how v0.3.1 got the O(1)
per-token decode throughput recovery (decode reads from the persistent
dequantized buffer and only dequants the new token, never the full
history).

## The trade-off was always there

This isn't a bug. It's the explicit v0.3.1 design choice: **trade
runtime memory for throughput**. The KIVI paper (which we cite in
`cache/mlx.py:8`) makes the same trade-off. The cache strategy
constants are explicit:

- **incremental** — keeps dequantized buffer permanently, O(1) decode,
  same memory as baseline + overhead. **Default.** Throughput-optimal.
- **residual** — last R tokens uncompressed, older tokens compressed
  → dequantized into incremental buffer. Same peak memory.
- **full** — stores ONLY compressed indices, dequantizes the entire
  history on every cache read. O(n²) per token but the compressed
  storage is the only persistent state. **Still doesn't save peak
  memory** because the per-call dequantized result has the same size
  as the baseline KV tensors and exists transiently during attention
  compute.

**No tqai cache strategy saves peak runtime memory** because attention
always needs the full-precision K/V at compute time, and the peak
occurs during attention compute.

## What the v0.3.1 K4/V2 win actually IS

Three things, in order of value:

1. **Quality preservation** (still valid). K4/V2 produces byte-identical
   token outputs to baseline on every Qwen / Gemma / Llama we tested in
   `benchmark_forward.py`. Δppl = 0.00. This is the most important
   property and it is unaffected by the memory finding. The pipeline
   exists, the math is correct, and the round-trip distortion is below
   the model's effective precision.

2. **Compressed-storage size** (still valid). If you serialize the cache
   to disk (e.g., for prompt caching or KV cache offloading), the
   compressed indices are 5× smaller than the bf16 KV tensors. tqai's
   `convert` command exploits this for offline cache pre-conversion.

3. **Foundation for a future fused dequant-attention kernel** (potential).
   The KVQuant paper (cited in `cache/mlx.py:9`) shows that on CUDA you
   can compute attention directly on compressed K/V without ever
   materializing the full bf16 tensors. tqai's compressed storage format
   is the prerequisite for that kernel — but the kernel itself does not
   exist yet on Apple Silicon (it would need a custom Metal shader).

## What the v0.3.1 K4/V2 win is NOT

**It is not a peak runtime memory win.** The README, the paper, and
several reports state or imply that K4/V2 saves 80% of KV cache memory
at runtime. This is **wrong** in the way most users would interpret it.
The 80% figure is the size ratio of the compressed indices vs the
uncompressed bf16 KV tensors, which is true for the *compressed-storage
form*. But the runtime memory measured by `mx.get_peak_memory()` shows
no savings — and in fact a slight regression — because the dequantized
buffer dominates the runtime footprint.

## Implications for the project

### Documentation updates needed

The "saves 80% memory" framing should be corrected in three places:

1. **README** — the headline currently says "Compresses the KV cache to
   ~3 bits per channel with 80%+ memory savings". This should be
   reframed as "compressed storage" with a clear note that runtime
   peak memory is not reduced.

2. **`paper/tqai_paper.md` and `.tex`** — the abstract claims "80% KV
   cache memory savings" without qualification. Should be: "80%
   reduction in compressed-storage size, with runtime peak memory
   unchanged". The conclusion section should add this as a third
   honest negative result alongside chunked attention and CFG sharing.

3. **`reports/where_tqai_shines.md`** — the top line of the TL;DR
   currently says "KV cache quantization (4/2 bit): Quality-neutral on
   Qwen/Gemma/Llama → Ship it — main win". Should be split into a
   quality verdict (still ship) and a memory verdict (no peak runtime
   savings; unblock with future fused kernel).

### Roadmap updates

Items that this finding adds to the roadmap:

- **Fused dequant-attention Metal kernel.** The actual unlock for
  runtime memory savings. Would need to compute Q×K^T on uint8 indices
  + per-row L2 norms + the rotation matrix, without materializing the
  full bf16 K tensor. This is a research project (probably 1-2 weeks
  for a working prototype, longer for parity with `mx.fast.scaled_dot_product_attention`).
  Until this exists, tqai's runtime memory profile is identical to
  baseline.

- **Better measurement on the existing benchmark suite.** The
  `benchmark_forward.py` script reports `peak_mb` from `_rss_mb()`
  which is process RSS — we should switch to `mx.get_peak_memory()`
  for the MLX backend to get accurate per-config measurements with
  the same fix as `benchmark_kv_memory.py`. Otherwise we may be
  reporting misleading numbers.

### What still works

To be clear, the rest of v0.4 is unaffected:

- ✅ Pipeline architecture (zero-overhead default path)
- ✅ Paper playground plugins (synthetic NMSE results stand)
- ✅ Step #1 video presets (1.66× speedup, real wall-clock)
- ✅ VAE memory optimization (the actual local video memory win)
- ✅ MPS float64 fix for LTX-2
- ✅ Quality preservation of K4/V2 (Δppl = 0.00)

The v0.3.1 K4/V2 quantization is still a valid technique. Its headline
just needs to be corrected from "memory savings" to "lossless
compressed-storage form + foundation for future fused kernels".

## What I'd do next

1. **Update the docs and paper to reflect this finding.** ~1 hour.
   The honest framing is more useful than the wrong framing, even
   though it's less marketable.

2. **Add the fused dequant-attention Metal kernel to the roadmap as a
   high-priority item.** This is now the actual unlock for the headline
   memory claim. Without it, tqai users on Apple Silicon don't get any
   memory reduction from KV quantization. With it, they would get the
   full ~5× reduction the paper promised.

3. **Skip Step #5 (preset system) for now.** That was the smallest item
   on the roadmap and depends on having concrete recommendations to
   present. Now that we know "K4/V2 doesn't save memory at runtime",
   the preset table needs to be redesigned around what tqai *does*
   actually do (quality preservation, compressed storage for offline
   workflows, video pipelines).
