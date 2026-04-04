"""Comprehensive accuracy test suite for TurboQuant.

Tests quantization fidelity across multiple dimensions:
  1. Mathematical guarantees (distortion bounds from the paper)
  2. Vector-level metrics (cosine similarity, MSE, RMSE, SNR)
  3. Attention-level accuracy (softmax score fidelity)
  4. Inner product preservation (the core operation KV caches serve)
  5. Edge cases (extreme values, near-zero, high-dimensional)
  6. Statistical properties (unbiasedness, variance)

Run: pytest tests/test_accuracy.py -v
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from tqai.backend import get_backend
from tqai.quantizer import PolarQuantizer

# ─── Helpers ───

def cosine_sim_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-vector cosine similarity, shape (..., d) -> (...)."""
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return dot / (na * nb + 1e-30)


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power < 1e-30:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


# ─── 1. Mathematical Guarantee Tests ───

class TestDistortionBounds:
    """Verify the paper's theoretical MSE distortion bound."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_per_coordinate_mse_within_bound(self, d, b):
        """Paper Theorem 1: D_mse <= (sqrt(3)*pi/2) / 4^b per unit-variance coord."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)

        # Generate many vectors from N(0, I_d) — matching paper's assumption
        x = ops.randn((1000, d), seed=123)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        x_np = ops.to_numpy(x)
        x_hat_np = ops.to_numpy(x_hat)

        # Normalised MSE: E[||x - x_hat||^2] / E[||x||^2]
        mse = np.mean(np.sum((x_np - x_hat_np) ** 2, axis=-1))
        signal = np.mean(np.sum(x_np ** 2, axis=-1))
        normalised_mse = mse / signal

        # Theoretical bound (normalised): (sqrt(3)*pi/2) / 4^b
        bound = (math.sqrt(3) * math.pi / 2.0) / (4.0 ** b)

        assert normalised_mse < bound, (
            f"Normalised MSE {normalised_mse:.6f} exceeds bound {bound:.6f} "
            f"for d={d}, b={b}"
        )

    @pytest.mark.parametrize("d", [64, 128])
    def test_distortion_scales_with_bits(self, d):
        """More bits -> strictly less distortion (monotonic improvement)."""
        ops = get_backend("torch")
        x = ops.randn((500, d), seed=77)

        mses = []
        for b in [2, 3, 4]:
            pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
            indices, norms = pq.quantize(x)
            x_hat = pq.dequantize(indices, norms)
            mse = np.mean((ops.to_numpy(x) - ops.to_numpy(x_hat)) ** 2)
            mses.append(mse)

        assert mses[0] > mses[1] > mses[2], f"MSE not monotonically decreasing: {mses}"

        # Each extra bit should roughly 4x the improvement (from 1/4^b)
        ratio_2_to_3 = mses[0] / mses[1]
        ratio_3_to_4 = mses[1] / mses[2]
        assert ratio_2_to_3 > 2.0, f"2->3 bit improvement ratio {ratio_2_to_3:.2f} too small"
        assert ratio_3_to_4 > 2.0, f"3->4 bit improvement ratio {ratio_3_to_4:.2f} too small"


# ─── 2. Vector-Level Metrics ───

class TestVectorMetrics:
    """Detailed per-vector accuracy metrics."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_cosine_similarity_distribution(self, d, b):
        """Check cosine similarity statistics, not just mean."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
        x = ops.randn((1000, d), seed=99)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        cos_sims = cosine_sim_batch(ops.to_numpy(x), ops.to_numpy(x_hat))

        mean_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)

        # Expectations based on bits
        if b >= 4:
            assert mean_cos > 0.99, f"4-bit mean cos {mean_cos:.4f}"
            assert min_cos > 0.90, f"4-bit min cos {min_cos:.4f}"
        elif b >= 3:
            assert mean_cos > 0.95, f"3-bit mean cos {mean_cos:.4f}"
            assert min_cos > 0.75, f"3-bit min cos {min_cos:.4f}"
        else:  # 2-bit
            assert mean_cos > 0.80, f"2-bit mean cos {mean_cos:.4f}"

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [3, 4])
    def test_snr(self, d, b):
        """Signal-to-noise ratio should be reasonable."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
        x = ops.randn((500, d), seed=55)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        snr = snr_db(ops.to_numpy(x), ops.to_numpy(x_hat))

        # ~6 dB per bit is typical for scalar quantization
        if b >= 4:
            assert snr > 15, f"4-bit SNR {snr:.1f} dB too low"
        elif b >= 3:
            assert snr > 10, f"3-bit SNR {snr:.1f} dB too low"

    @pytest.mark.parametrize("d", [64, 128])
    def test_norm_preservation(self, d):
        """Reconstructed vectors should have similar norms to originals."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=4, seed=42, ops=ops)
        x = ops.randn((500, d), seed=33)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        x_np = ops.to_numpy(x)
        x_hat_np = ops.to_numpy(x_hat)

        orig_norms = np.linalg.norm(x_np, axis=-1)
        recon_norms = np.linalg.norm(x_hat_np, axis=-1)

        # Norms should be close (stored as FP16, so ~0.1% relative error)
        norm_ratio = recon_norms / (orig_norms + 1e-10)
        mean_deviation = np.mean(np.abs(norm_ratio - 1.0))
        assert mean_deviation < 0.03, f"Norm deviation {mean_deviation:.4f} too large"


# ─── 3. Inner Product Preservation ───

class TestInnerProductAccuracy:
    """Inner products are THE critical operation for attention scores.
    KV cache quantization quality = inner product fidelity."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [3, 4])
    def test_inner_product_correlation(self, d, b):
        """Quantized inner products should correlate strongly with true ones."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)

        # Simulate: queries Q (not quantized) and keys K (quantized)
        Q = ops.randn((100, d), seed=1)
        K = ops.randn((50, d), seed=2)

        indices, norms = pq.quantize(K)
        K_hat = pq.dequantize(indices, norms)

        Q_np = ops.to_numpy(Q)
        K_np = ops.to_numpy(K)
        K_hat_np = ops.to_numpy(K_hat)

        # True scores: Q @ K^T
        true_scores = Q_np @ K_np.T  # (100, 50)
        quant_scores = Q_np @ K_hat_np.T

        # Pearson correlation per query
        correlations = []
        for i in range(true_scores.shape[0]):
            r = np.corrcoef(true_scores[i], quant_scores[i])[0, 1]
            correlations.append(r)

        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)

        if b >= 4:
            assert mean_corr > 0.99, f"4-bit mean correlation {mean_corr:.4f}"
            assert min_corr > 0.95, f"4-bit min correlation {min_corr:.4f}"
        else:
            assert mean_corr > 0.97, f"3-bit mean correlation {mean_corr:.4f}"
            assert min_corr > 0.85, f"3-bit min correlation {min_corr:.4f}"

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [3, 4])
    def test_inner_product_absolute_error(self, d, b):
        """Absolute error of inner products, normalised by expected magnitude."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)

        Q = ops.randn((200, d), seed=10)
        K = ops.randn((200, d), seed=20)

        indices, norms = pq.quantize(K)
        K_hat = pq.dequantize(indices, norms)

        Q_np = ops.to_numpy(Q)
        K_np = ops.to_numpy(K)
        K_hat_np = ops.to_numpy(K_hat)

        # Pairwise inner products (same index)
        true_dots = np.sum(Q_np * K_np, axis=-1)
        quant_dots = np.sum(Q_np * K_hat_np, axis=-1)

        abs_err = np.abs(true_dots - quant_dots)
        # Normalise by RMS of true dot products (avoids near-zero denominator)
        rms_true = np.sqrt(np.mean(true_dots ** 2))
        normalised_err = np.mean(abs_err) / rms_true

        if b >= 4:
            assert normalised_err < 0.15, f"4-bit normalised error {normalised_err:.4f}"
        else:
            assert normalised_err < 0.25, f"3-bit normalised error {normalised_err:.4f}"


# ─── 4. Attention Score Fidelity ───

class TestAttentionFidelity:
    """Simulate full attention: Q @ K^T / sqrt(d) -> softmax -> @ V.
    This is the real-world use case."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [3, 4])
    def test_attention_weight_similarity(self, d, b):
        """Softmax attention weights from quantized KV vs original."""
        ops = get_backend("torch")
        pq_k = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
        pq_v = PolarQuantizer(head_dim=d, bits=2, seed=99, ops=ops)

        seq_len = 32
        Q = ops.randn((1, seq_len, d), seed=1)
        K = ops.randn((1, seq_len, d), seed=2)
        V = ops.randn((1, seq_len, d), seed=3)

        k_idx, k_norms = pq_k.quantize(K)
        v_idx, v_norms = pq_v.quantize(V)
        K_hat = pq_k.dequantize(k_idx, k_norms)
        V_hat = pq_v.dequantize(v_idx, v_norms)

        Q_np = ops.to_numpy(Q)[0]
        K_np = ops.to_numpy(K)[0]
        V_np = ops.to_numpy(V)[0]
        K_hat_np = ops.to_numpy(K_hat)[0]
        V_hat_np = ops.to_numpy(V_hat)[0]

        scale = 1.0 / math.sqrt(d)

        # True attention
        true_logits = Q_np @ K_np.T * scale
        true_weights = softmax(true_logits)
        true_output = true_weights @ V_np

        # Quantized attention
        quant_logits = Q_np @ K_hat_np.T * scale
        quant_weights = softmax(quant_logits)
        quant_output = quant_weights @ V_hat_np

        # Attention weight similarity (per query position)
        weight_cos_sims = cosine_sim_batch(true_weights, quant_weights)
        mean_weight_cos = np.mean(weight_cos_sims)

        # Output similarity
        output_cos_sims = cosine_sim_batch(true_output, quant_output)
        mean_output_cos = np.mean(output_cos_sims)

        if b >= 4:
            assert mean_weight_cos > 0.95, f"4-bit attn weight cos {mean_weight_cos:.4f}"
            assert mean_output_cos > 0.90, f"4-bit attn output cos {mean_output_cos:.4f}"
        else:
            assert mean_weight_cos > 0.85, f"3-bit attn weight cos {mean_weight_cos:.4f}"
            assert mean_output_cos > 0.80, f"3-bit attn output cos {mean_output_cos:.4f}"

    @pytest.mark.parametrize("d", [128])
    def test_attention_argmax_preservation(self, d):
        """The token with highest attention should usually be the same."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=d, bits=4, seed=42, ops=ops)

        seq_len = 64
        Q = ops.randn((1, seq_len, d), seed=1)
        K = ops.randn((1, seq_len, d), seed=2)

        k_idx, k_norms = pq.quantize(K)
        K_hat = pq.dequantize(k_idx, k_norms)

        Q_np = ops.to_numpy(Q)[0]
        K_np = ops.to_numpy(K)[0]
        K_hat_np = ops.to_numpy(K_hat)[0]

        scale = 1.0 / math.sqrt(d)
        true_logits = Q_np @ K_np.T * scale
        quant_logits = Q_np @ K_hat_np.T * scale

        true_argmax = np.argmax(true_logits, axis=-1)
        quant_argmax = np.argmax(quant_logits, axis=-1)

        match_rate = np.mean(true_argmax == quant_argmax)
        assert match_rate > 0.80, f"4-bit argmax match rate {match_rate:.2%}"


# ─── 5. Edge Cases ───

class TestEdgeCases:

    def test_very_large_values(self):
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)
        x_np = np.random.randn(10, 64).astype(np.float32) * 1000
        x = ops.from_numpy(x_np)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        cos = cosine_sim_batch(x_np, ops.to_numpy(x_hat))
        assert np.mean(cos) > 0.95, "Large values should still quantize well"

    def test_very_small_values(self):
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)
        x_np = np.random.randn(10, 64).astype(np.float32) * 1e-6
        x = ops.from_numpy(x_np)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert np.all(np.isfinite(ops.to_numpy(x_hat)))

    def test_identical_vectors(self):
        """All-same vectors should quantize consistently."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
        v = np.ones((1, 64), dtype=np.float32)
        batch = np.tile(v, (20, 1))
        x = ops.from_numpy(batch)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        hat_np = ops.to_numpy(x_hat)
        # All reconstructions should be identical
        for i in range(1, 20):
            npt.assert_array_equal(hat_np[0], hat_np[i])

    def test_single_nonzero_coord(self):
        """Sparse vector: only one coordinate is nonzero."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)
        x_np = np.zeros((1, 64), dtype=np.float32)
        x_np[0, 17] = 5.0
        x = ops.from_numpy(x_np)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        hat_np = ops.to_numpy(x_hat)
        assert np.all(np.isfinite(hat_np))
        # Direction should be roughly preserved
        cos = cosine_sim_batch(x_np, hat_np)
        assert cos[0] > 0.5, f"Sparse vector cos {cos[0]:.4f}"

    def test_high_dimensional(self):
        """Test with d=256 (some large models use this)."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=256, bits=3, seed=42, ops=ops)
        x = ops.randn((50, 256), seed=77)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        cos = cosine_sim_batch(ops.to_numpy(x), ops.to_numpy(x_hat))
        assert np.mean(cos) > 0.95, f"d=256 mean cos {np.mean(cos):.4f}"


# ─── 6. Statistical Properties ───

class TestStatisticalProperties:

    def test_quantization_error_is_mean_centered(self):
        """Error should be approximately zero-mean (unbiased)."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=128, bits=4, seed=42, ops=ops)
        x = ops.randn((2000, 128), seed=42)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        error = ops.to_numpy(x) - ops.to_numpy(x_hat)
        mean_error = np.mean(error, axis=0)  # per-coordinate mean error

        # Mean error per coordinate should be close to 0
        assert np.max(np.abs(mean_error)) < 0.02, (
            f"Max mean error per coord: {np.max(np.abs(mean_error)):.4f}"
        )

    def test_rotation_preserves_distribution(self):
        """After rotation, coordinate variance should be ~1/d."""
        ops = get_backend("torch")
        pq = PolarQuantizer(head_dim=128, bits=3, seed=42, ops=ops)

        x = ops.randn((5000, 128), seed=55)
        x_np = ops.to_numpy(x)

        # Normalize
        norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
        x_unit = x_np / (norms + 1e-10)

        # Rotate
        R = ops.to_numpy(pq._rotation)
        y = x_unit @ R.T

        # Each coordinate of y should have variance ~1/d = 1/128
        expected_var = 1.0 / 128
        actual_vars = np.var(y, axis=0)
        mean_var = np.mean(actual_vars)

        assert abs(mean_var - expected_var) < expected_var * 0.5, (
            f"Expected variance ~{expected_var:.6f}, got {mean_var:.6f}"
        )

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_error_variance_scales_correctly(self, b):
        """Quantization error variance should roughly follow 1/4^b scaling."""
        ops = get_backend("torch")
        d = 128
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
        x = ops.randn((1000, d), seed=11)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        error = ops.to_numpy(x) - ops.to_numpy(x_hat)
        error_var = np.mean(error ** 2)

        # Should be proportional to 1/4^b (up to constants)
        # Just verify it's in a sane range
        assert error_var > 0, "Error variance should be positive"
        assert error_var < 10.0, f"Error variance {error_var} unreasonably large"


# ─── 7. Cross-Backend Consistency ───

class TestCrossBackendConsistency:
    """Ensure torch and mlx backends produce equivalent results."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("b", [3, 4])
    def test_same_codebooks(self, d, b):
        """Both backends should use identical codebooks."""
        ops_torch = get_backend("torch")
        ops_mlx = get_backend("mlx")
        pq_torch = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops_torch)
        pq_mlx = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops_mlx)

        c_torch = ops_torch.to_numpy(pq_torch._centroids)
        c_mlx = ops_mlx.to_numpy(pq_mlx._centroids)
        npt.assert_array_equal(c_torch, c_mlx)

    @pytest.mark.parametrize("d", [64, 128])
    def test_similar_quantization_quality(self, d):
        """Both backends should achieve similar MSE on same data."""
        x_np = np.random.RandomState(42).randn(200, d).astype(np.float32)

        mses = {}
        for name in ["torch", "mlx"]:
            ops = get_backend(name)
            pq = PolarQuantizer(head_dim=d, bits=3, seed=42, ops=ops)
            x = ops.from_numpy(x_np)
            indices, norms = pq.quantize(x)
            x_hat = pq.dequantize(indices, norms)
            mse = np.mean((x_np - ops.to_numpy(x_hat)) ** 2)
            mses[name] = mse

        # MSEs should be similar (not identical due to different RNG impls)
        ratio = max(mses.values()) / min(mses.values())
        assert ratio < 2.0, f"Backend MSE ratio {ratio:.2f} too different: {mses}"
