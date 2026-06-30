# lmcache-turbo-quant-serde

**tqai KV-cache compression as a drop-in LMCache v1 serde plugin.**

Compresses the KV tensors written to LMCache's L2 storage (disk, remote, DRAM
off-device) using tqai's `PolarQuantizer` — the same algorithm that achieves
Δppl = 0.00 on 8B+ models with ≥74% size reduction. Plugs into LMCache's
existing `register_serde_factory` extension point; no LMCache source changes
required.

---

## Quick start

```bash
pip install lmcache-turbo-quant-serde          # includes tqai automatically
pip install lmcache-turbo-quant-serde[lmcache] # + LMCache v1 engine
```

```python
import lmcache_turbo_quant_serde

# Register once at startup — before LMCache initialises its engine
lmcache_turbo_quant_serde.register()
```

Then set the serde type in your LMCache config:

```yaml
# lmcache_config.yaml
chunk_size: 256
local_device: "cpu"
remote_serde:
  type: tqai
  kwargs:
    head_dim: 128   # must match your model's KV head dimension
    bits: 4         # 2 | 3 | 4 (recommend 4 for Δppl = 0.00)
    seed: 42
    use_qjl: false
```

Or programmatically:

```python
from lmcache.v1.distributed.serde import SerdeConfig, create_serde_processor

proc = create_serde_processor(
    SerdeConfig(type="tqai", kwargs={"head_dim": 128, "bits": 4})
)
```

---

## How it fits into LMCache

LMCache sits between the vLLM/HF attention layer and external storage. Every
time a KV block is evicted from GPU memory to an L2 backend, LMCache calls the
configured **serde** to compress the tensor before writing and decompress it on
recall.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  vLLM / HF model                                                 │
 │  attention layer  →  GPU KV cache (fp16 / bf16)                  │
 └────────────────────────────┬─────────────────────────────────────┘
                              │ evict block
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  LMCache StorageManager                                          │
 │                                                                  │
 │  serialize(src: MemoryObj, dst: MemoryObj)   ◄─── TqaiSerializer│
 │      │  compressed bytes written to dst.tensor                   │
 │      ▼                                                           │
 │  L2 backend  (local disk / Redis / NIXL / remote DRAM)          │
 │      │  raw bytes read back                                      │
 │      ▼                                                           │
 │  deserialize(src: MemoryObj, dst: MemoryObj) ◄─TqaiDeserializer │
 │      │  reconstructed bf16 tensor in dst.tensor                  │
 │      ▼                                                           │
 │  GPU KV cache refilled                                           │
 └──────────────────────────────────────────────────────────────────┘
```

### What the serializer does (per block)

1. **Reshape** `[2, L, T, H×D]` → `[2, L, T, H, head_dim]` so each head
   vector is the last axis (required by `PolarQuantizer`).
2. **Rotate** each head vector by the fixed Haar-distributed `head_dim×head_dim`
   orthogonal matrix, spreading information uniformly across coordinates.
3. **Quantize** each coordinate against the pre-computed Lloyd-Max codebook
   for N(0, 1/d), producing `uint8` indices and `float16` per-vector norms.
4. **Pack** the indices at `bits`-per-value using the vectorised bit-packer
   (2/4 bit: byte-aligned; 3/6 bit: LCM-block NumPy reduction).
5. **Encode** everything into a self-describing wire frame: magic `TQ01` +
   header (bits, seed, shape, dtype) + packed indices + norms.

The deserializer reverses steps 5 → 1 using only data from the wire frame —
**no configuration is needed at decode time**.

### Wire format

```
offset  size   field
──────  ─────  ──────────────────────────────────────────
0       4      magic  b"TQ01"
4       1      version  (0x01)
5       1      bits per index
6       1      use_qjl flag
7       2      qjl_sketch_size (uint16 LE)
9       4      seed  (int32 LE)
13      4      ndim  (number of shape dimensions)
17      4×ndim shape  (int32 LE each)
17+4n   var    dtype string (null-terminated)
var     4      packed_len  (uint32 LE)
var     packed_len  packed indices (uint8)
var     4      norms_len  (uint32 LE)
var     norms_len   norms  (float16 LE)
[var    4      sketch_len + sketch data + residual_norms]  ← if use_qjl
```

The format is intentionally self-describing so compressed blobs can be
inspected or decoded by any language without a schema file.

---

## Configuration reference

| kwarg | Default | Notes |
|---|---|---|
| `head_dim` | `128` | Must match your model's per-head hidden size (LLaMA-3: 128, Mistral-7B: 128, Gemma-2-9B: 256) |
| `bits` | `4` | Bits per coordinate. 4 → Δppl=0.00 on all tested models; 3 → Δppl≈0.00; 2 → slight quality loss |
| `seed` | `42` | RNG seed for the Haar rotation matrix. Must be the same on serializer and deserializer |
| `use_qjl` | `false` | Enable QJL Stage 2 residual sketch (reduces inner-product bias; adds ~10% size) |
| `qjl_sketch_size` | `64` | JL sketch dimension (only used when `use_qjl=true`) |
| `max_workers` | `1` | Thread pool size passed to `AsyncSerdeProcessor` |

### Choosing `bits`

| bits | Storage vs fp16 | Cosine similarity | Δppl (LLaMA-3 8B) | Recommended for |
|------|----------------|-------------------|--------------------|-----------------|
| 4 | 25.8% | 0.9954 | 0.00 | Production — best quality/storage balance |
| 3 | 19.5% | 0.9831 | ≈ 0.00 | Aggressive storage saving |
| 2 | 13.3% | 0.9405 | < 0.05 | Bandwidth-constrained edge transfer |

---

## Benchmark results

CPU baseline (Apple M-series, bfloat16, LLaMA-3 8B):

| Tokens | Bits | Size vs fp16 | Cosine ↑ | Serialize | Deserialize |
|--------|------|-------------|----------|-----------|-------------|
| 64 | 4 | 25.8% | 0.9954 | 21 ms | 4 ms |
| 64 | 3 | 19.5% | 0.9831 | 21 ms | 10 ms |
| 64 | 2 | 13.3% | 0.9405 | 12 ms | 4 ms |
| 256 | 4 | 25.8% | 0.9954 | 81 ms | 15 ms |
| 256 | 3 | 19.5% | 0.9831 | 81 ms | 39 ms |
| 256 | 2 | 13.3% | 0.9406 | 48 ms | 15 ms |
| 1024 | 4 | 25.8% | 0.9954 | 475 ms | 64 ms |
| 1024 | 3 | 19.5% | 0.9831 | 323 ms | 156 ms |
| 1024 | 2 | 13.3% | 0.9406 | 194 ms | 62 ms |

> GPU numbers would be significantly faster. The serialize path is currently
> CPU-bound (the quantize loop runs on PyTorch CPU even when the KV tensor
> lives on GPU — see Roadmap below).

### Why this matters for LMCache deployments

LMCache's primary use case is **P/D disaggregation** (prefill on one node,
decode on another) and **long-context reuse**. Both scenarios transfer large KV
blocks over the network or to disk. At 4-bit:

- **74% smaller writes** to the L2 backend
- **74% less network bandwidth** on P/D disaggregation transfers
- **74% more KV blocks fit in a fixed DRAM/disk budget**

For a LLaMA-3 70B inference cluster with 128-token KV blocks at bf16, each
block is `2 × 80 × 128 × 8192 × 2 bytes = 335 MB`. At 4-bit, the same block
compresses to `87 MB` — fitting 3.8× more blocks in the same Redis budget or
reducing cross-node transfer time proportionally.

---

## Standalone usage (no LMCache)

The package also exports a simpler `to_bytes` / `from_bytes` codec that works
without LMCache installed:

```python
from lmcache_turbo_quant_serde import TurboQuantSerializer, TurboQuantDeserializer
import torch

ser = TurboQuantSerializer(bits=4, head_dim=128)
des = TurboQuantDeserializer()

# kv shape: [2, num_layers, num_tokens, num_heads * head_dim]
kv = torch.randn(2, 32, 64, 1024, dtype=torch.bfloat16)

compressed = ser.to_bytes(kv)       # → bytes  (25.8% of original size)
recovered  = des.from_bytes(compressed)  # → torch.Tensor, same shape + dtype
```

This is useful for custom serialization pipelines, storage backends, or
testing without a full LMCache installation.

---

## Running tests

```bash
cd integrations/lmcache
pip install -e ".[dev]"
pytest tests/ -v
```

With LMCache installed, the full integration suite (47 tests) runs including:
- Wire format round-trips across all bit widths
- Standalone codec quality and compression ratio checks
- LMCache v1 `Serializer` / `Deserializer` ABC conformance
- `register()` → `create_serde_processor()` end-to-end flow
- E2E roundtrip at 4 model configs (LLaMA-3 8B, Mistral 7B variants)

Without LMCache, the standalone suite (21 wire + codec tests) still runs.

```bash
python benchmarks/bench_serde.py   # prints Markdown table
```

---

## Roadmap / known limitations

### Current limitations

| Area | Status | Detail |
|---|---|---|
| **CPU-only quantize path** | Active limitation | `PolarQuantizer` runs on CPU even when the KV tensor is on GPU. Requires a CPU→GPU copy before each deserialize and GPU→CPU before each serialize. This is the dominant cost at large sequence lengths. |
| **3-bit deserialize** | Slower than 4-bit | The `_unpack_bitstream` path (3/6 bit) is vectorised but still ~2.5× slower than the byte-aligned 4-bit path. Not a bottleneck when the network RTT dominates, but noticeable on local storage. |
| **No chunked streaming** | Missing | The full KV tensor is compressed in one shot. For very long contexts (>4K tokens), a chunked/streaming encode would reduce peak CPU memory. |
| **No async GPU pipeline** | Missing | Serialize and deserialize block the calling thread. `AsyncSerdeProcessor` wraps them in a thread pool, but the GIL means only one compression runs at a time. |

### Planned improvements

**P1 — CUDA/Metal quantize kernel**
Move the rotation matrix multiply and Lloyd-Max lookup into a CUDA kernel (or
Metal for Apple). This eliminates the CPU↔GPU sync on every block eviction and
should bring serialize latency from ~80ms to ~5ms for the 256-token case.
LMCache's own `turboquant` serde demonstrates this is viable — it uses a CUDA
extension for the quantize step.

**P2 — Async, non-blocking pipeline**
Overlap quantization with the network/disk write so the GPU is not stalled
waiting for the serde to finish. Requires decoupling the `MemoryObj` lifetime
from the serde call and using a proper producer/consumer queue.

**P3 — 3-bit vectorised unpack**
The `_unpack_bitstream` path still runs an inner loop per block during
deserialization. A fully vectorised version (using the same
`(blocks << byte_shifts).sum(axis=1)` pattern as pack) should close the
3-bit/4-bit gap on the decode side.

**P4 — QJL + ultra-low bit (2-bit)**
At 2-bit, inner-product bias from the Haar rotation becomes noticeable.
Enabling `use_qjl=True` corrects this via a Johnson-Lindenstrauss residual
sketch, but the sketch overhead currently exceeds the space saved at 2-bit.
Tuning `qjl_sketch_size` and the packing of the int8 sketch should bring 2-bit
closer to 3-bit quality at comparable size.

**P5 — Per-layer bit allocation**
Different transformer layers compress at different fidelity. Early and late
layers tolerate 2-bit; middle layers need 3–4 bit. A static per-layer config
(e.g. from tqai's offline `fisher_static` calibration) stored in the LMCache
engine config could reduce average bits-per-element from 4 to ~3 with no
perceptible quality change.

**P6 — PyPI release + LMCache optional extra**
Publish to PyPI and add `lmcache-turbo-quant-serde` as an optional extra in
`lmcache`'s own `pyproject.toml`:
```toml
[project.optional-dependencies]
tqai = ["lmcache-turbo-quant-serde>=0.1"]
```
So users can `pip install lmcache[tqai]` and call `register()` without
installing the integration separately.

---

## Package layout

```
integrations/lmcache/
├── pyproject.toml
├── src/lmcache_turbo_quant_serde/
│   ├── __init__.py      # exports + register()
│   ├── _wire.py         # self-describing binary frame (no torch/tqai imports)
│   ├── _codec.py        # standalone TurboQuantSerializer / TurboQuantDeserializer
│   ├── _v1_codec.py     # LMCache v1 TqaiSerializer / TqaiDeserializer
│   └── _register.py     # register_serde_factory("tqai", ...)
├── tests/
│   ├── test_wire.py
│   ├── test_codec.py
│   ├── test_integration.py
│   └── test_e2e.py
└── benchmarks/
    └── bench_serde.py
```
