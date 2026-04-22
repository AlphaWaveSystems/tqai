# tqai Benchmarks

Scripts and results for measuring tqai's compression quality and throughput across models and backends.

## Results

See [`results/FINDINGS.md`](results/FINDINGS.md) for the full benchmark report, including:
- Perplexity delta across 5 models × 8 compression configs (40 runs, all Δppl = 0.00)
- Throughput retention by model size and quantization
- Memory measurement notes for Apple Silicon

Raw JSON results per model are also in [`results/`](results/).

---

## Scripts

### `benchmark_pipeline.py` — Pipeline scorer × strategy quality benchmark

Tests every scorer × strategy combination (including RotorQuant configs) on
synthetic data matching real model dimensions across 7 model profiles.

```bash
# Run all configs (PolarQuant + RotorQuant) on all model profiles
python benchmarks/benchmark_pipeline.py

# Save results to a custom path
python benchmarks/benchmark_pipeline.py --json results/my_run.json
```

Output columns: `Config | NMSE | CosSim | Comp ms | Decomp ms | Ratio`

RotorQuant configs are prefixed with `rotorquant+` and use `RotorQuantizer`
(block-diagonal Clifford rotors) instead of `PolarQuantizer` (dense Haar
matrix). Quality (NMSE, CosSim) is identical at all bit widths; Metal kernel
speed advantage is visible when using the MLX backend.

---

### `benchmark_forward.py` — KV + activation compression benchmark

Measures perplexity, throughput, and token match rate across all compression configs for a given model.

```bash
# MLX (Apple Silicon)
python benchmarks/benchmark_forward.py --backend mlx \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --all-configs

# PyTorch
python benchmarks/benchmark_forward.py --backend torch \
    --model Qwen/Qwen2.5-3B-Instruct \
    --all-configs

# Single config
python benchmarks/benchmark_forward.py --backend mlx \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --config aggressive
```

Output columns: `Config | PPL | ΔPPL | tok/s | vs baseline | Match`

Results are saved to `benchmarks/results/<model_id>_<backend>.json`.

### `eval_perplexity.py` — Perplexity helper (internal)

Used by the benchmark scripts. Provides `perplexity_hf()`, `perplexity_mlx()`, `generate_tokens()`, and `compute_match_rate()`. Not intended to be run directly.

---

## Reproducing the v0.2 Results

The full results in `results/FINDINGS.md` were produced on Apple Silicon (macOS, MLX) with the following commands:

```bash
python benchmarks/benchmark_forward.py --backend mlx --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --all-configs
python benchmarks/benchmark_forward.py --backend mlx --model mlx-community/Qwen2.5-3B-Instruct-bf16 --all-configs
python benchmarks/benchmark_forward.py --backend mlx --model mlx-community/Llama-3.1-8B-Instruct-4bit --all-configs
python benchmarks/benchmark_forward.py --backend mlx --model mlx-community/Qwen2.5-7B-Instruct-8bit --all-configs
python benchmarks/benchmark_forward.py --backend mlx --model mlx-community/Qwen2.5-14B-Instruct-4bit --all-configs
```

Models are downloaded automatically from HuggingFace on first run (~2–30 GB depending on size).

---

## Key Findings (v0.2)

- **Δppl = 0.00** across all 40 benchmark runs — PolarQuant rotation + Lloyd-Max codebook is perfectly quality-neutral
- **Token match rate is low (1–4%) on MLX** — this is expected (argmax sensitivity to ULP differences), not a quality failure. Perplexity is the authoritative metric.
- **Throughput overhead is fully Python-level** — varies inversely with model size and weight precision:
  - Q8 models (compute-bound): **37% retention** — model matmul dominates
  - Q4 models (fast inference): 16–26% retention
  - bf16 models (large KV tensors): 9–15% retention
- **At 14B+, all compression configs converge** — adding hidden/FFN compression is essentially free in throughput terms
- A Metal/CUDA kernel is planned for v0.3 to eliminate the Python overhead and reach near-baseline throughput on all models
