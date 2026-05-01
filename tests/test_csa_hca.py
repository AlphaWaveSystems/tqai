"""Tests for CSA + HCA block-pool attention compression.

Covers:
    * Math primitives (block_pool, csa_score, csa_top_k, gather_blocks)
    * End-to-end csa_hca_attention: hand-calculated values + agreement with
      full SDPA in the limit (csa_block_size=1, csa_top_k=S, hca_block_size=S)
    * CSAHCAStrategy compress/decompress + registry integration
"""

from __future__ import annotations

import math

import pytest
import torch

from tqai.backend import get_backend
from tqai.csa_hca import (
    block_pool,
    csa_hca_attention,
    csa_score,
    csa_top_k,
    gather_blocks,
)
from tqai.pipeline.base import ScoredEntry
from tqai.quantizer import PolarQuantizer


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


class TestBlockPool:
    def test_simple_mean_pool(self):
        # 8 tokens × 4 dims, block_size=2 → 4 blocks of size 2
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 4)
        blocks = block_pool(x, block_size=2)
        assert blocks.shape == (1, 1, 4, 4)
        expected = torch.eye(4).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(blocks, expected)

    def test_hca_pool(self):
        # Same input, block_size=4 → 2 blocks
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0).unsqueeze(0)
        blocks = block_pool(x, block_size=4)
        assert blocks.shape == (1, 1, 2, 4)
        expected = torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
        ).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(blocks, expected)

    def test_partial_remainder(self):
        # 7 tokens, block_size=3 → 2 full blocks + 1 partial
        x = torch.arange(7 * 2, dtype=torch.float32).reshape(1, 1, 7, 2)
        blocks = block_pool(x, block_size=3)
        assert blocks.shape == (1, 1, 3, 2)
        # First block: rows 0,1,2 → mean of [0,1],[2,3],[4,5] = [2, 3]
        torch.testing.assert_close(blocks[0, 0, 0], torch.tensor([2.0, 3.0]))
        # Second block: rows 3,4,5 → mean of [6,7],[8,9],[10,11] = [8, 9]
        torch.testing.assert_close(blocks[0, 0, 1], torch.tensor([8.0, 9.0]))
        # Trailing block: row 6 only → [12, 13]
        torch.testing.assert_close(blocks[0, 0, 2], torch.tensor([12.0, 13.0]))

    def test_block_size_larger_than_seq(self):
        # block_size > S → single trailing block = mean of all tokens
        x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        blocks = block_pool(x, block_size=10)
        assert blocks.shape == (1, 1, 1, 2)
        torch.testing.assert_close(blocks[0, 0, 0], torch.tensor([2.0, 3.0]))

    def test_block_size_one_is_identity(self):
        x = torch.randn(2, 3, 5, 7)
        blocks = block_pool(x, block_size=1)
        torch.testing.assert_close(blocks, x)

    def test_invalid_block_size(self):
        x = torch.randn(1, 1, 4, 4)
        with pytest.raises(ValueError, match="block_size"):
            block_pool(x, block_size=0)


class TestCSAScore:
    def test_hand_calculated(self):
        # Query [0, 1, 0, 0] against 4 one-hot blocks: only block 1 hits.
        q = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])  # (1, 1, 1, 4)
        k_blocks = torch.eye(4).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
        scores = csa_score(q, k_blocks, scale=1.0)  # bypass scale
        torch.testing.assert_close(
            scores.squeeze(), torch.tensor([0.0, 1.0, 0.0, 0.0])
        )

    def test_default_scale(self):
        q = torch.ones(1, 1, 1, 4)
        k_blocks = torch.ones(1, 1, 1, 4)
        scores = csa_score(q, k_blocks)
        # dot = 4, scale = 1/sqrt(4) = 0.5 → 2.0
        torch.testing.assert_close(scores.squeeze(), torch.tensor(2.0))

    def test_gqa_repeat(self):
        # H_q = 4, H_kv = 2, repeats = 2
        q = torch.randn(1, 4, 1, 8)
        k_blocks = torch.randn(1, 2, 3, 8)
        scores = csa_score(q, k_blocks, scale=1.0)
        assert scores.shape == (1, 4, 1, 3)
        # Heads 0,1 share kv head 0; heads 2,3 share kv head 1
        manual_h0 = (q[0, 0, 0] * k_blocks[0, 0]).sum(dim=-1)
        manual_h1 = (q[0, 1, 0] * k_blocks[0, 0]).sum(dim=-1)
        torch.testing.assert_close(scores[0, 0, 0], manual_h0)
        torch.testing.assert_close(scores[0, 1, 0], manual_h1)

    def test_invalid_gqa(self):
        q = torch.randn(1, 3, 1, 4)
        k_blocks = torch.randn(1, 2, 1, 4)
        with pytest.raises(ValueError, match="not divisible"):
            csa_score(q, k_blocks)


class TestCSATopK:
    def test_basic(self):
        scores = torch.tensor([[[[0.1, 0.9, 0.5, 0.3]]]])  # (1, 1, 1, 4)
        idx = csa_top_k(scores, k=2)
        # topk returns descending: indices of 0.9 and 0.5
        assert idx.shape == (1, 1, 1, 2)
        idx_set = set(idx.flatten().tolist())
        assert idx_set == {1, 2}

    def test_k_clamped_to_n(self):
        scores = torch.tensor([[[[0.1, 0.9]]]])
        idx = csa_top_k(scores, k=10)
        assert idx.shape == (1, 1, 1, 2)

    def test_invalid_k(self):
        scores = torch.tensor([[[[0.1, 0.9]]]])
        with pytest.raises(ValueError, match="k must be"):
            csa_top_k(scores, k=0)


class TestGatherBlocks:
    def test_basic(self):
        blocks = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]]
        )  # (1, 1, 4, 2)
        idx = torch.tensor([[[[0, 2]]]])  # pick blocks 0 and 2
        out = gather_blocks(blocks, idx)
        assert out.shape == (1, 1, 1, 2, 2)
        torch.testing.assert_close(
            out.squeeze(0).squeeze(0).squeeze(0),
            torch.tensor([[1.0, 0.0], [2.0, 2.0]]),
        )

    def test_gqa_gather(self):
        blocks = torch.randn(1, 2, 5, 8)  # H_kv=2
        idx = torch.tensor(
            [[[[0, 1]], [[2, 3]], [[0, 4]], [[1, 2]]]]
        )  # (1, H_q=4, T_q=1, k=2)
        out = gather_blocks(blocks, idx)
        assert out.shape == (1, 4, 1, 2, 8)
        # Head 0 → kv head 0, gather indices [0, 1]
        torch.testing.assert_close(out[0, 0, 0, 0], blocks[0, 0, 0])
        torch.testing.assert_close(out[0, 0, 0, 1], blocks[0, 0, 1])
        # Head 2 → kv head 1, gather indices [0, 4]
        torch.testing.assert_close(out[0, 2, 0, 0], blocks[0, 1, 0])
        torch.testing.assert_close(out[0, 2, 0, 1], blocks[0, 1, 4])


# ---------------------------------------------------------------------------
# End-to-end attention
# ---------------------------------------------------------------------------


class TestCSAHCAAttention:
    def _make_kv(self):
        # Same setup as the hand-calculation: 8 one-hot tokens.
        K = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 4)
        V = K.clone()
        return K, V

    def test_query_selects_correct_block(self):
        K, V = self._make_kv()
        q = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])  # (1, 1, 1, 4)

        out, info = csa_hca_attention(
            q, K, V,
            csa_block_size=2,
            hca_block_size=4,
            csa_top_k_blocks=1,
        )
        assert out.shape == (1, 1, 1, 4)
        assert info["n_csa_blocks"] == 4
        assert info["n_hca_blocks"] == 2
        assert info["k_eff"] == 1

        # The query is closest to CSA block 1 = [0,1,0,0] (value).
        # HCA blocks are [0.5,0.5,0,0] and [0,0,0.5,0.5].
        # Joint softmax over [csa_sel_score, hca_score_0, hca_score_1] with scale=0.5:
        #   csa_sel = q · [0,1,0,0] * 0.5 = 0.5
        #   hca_0   = q · [0.5,0.5,0,0] * 0.5 = 0.25
        #   hca_1   = q · [0,0,0.5,0.5] * 0.5 = 0.0
        scale = 0.5
        s = torch.tensor([1.0 * scale, 0.5 * scale, 0.0])
        w = torch.softmax(s, dim=-1)
        expected = (
            w[0] * torch.tensor([0.0, 1.0, 0.0, 0.0])
            + w[1] * torch.tensor([0.5, 0.5, 0.0, 0.0])
            + w[2] * torch.tensor([0.0, 0.0, 0.5, 0.5])
        )
        torch.testing.assert_close(out.squeeze(), expected)

    def test_degenerate_equals_full_attention(self):
        # csa_block_size=1, csa_top_k=S means CSA selects every single token.
        # hca_block_size>=S means a single HCA centroid (the global mean).
        # Result must equal SDPA over the original tokens but with the global
        # mean appended as one extra key/value — we can directly construct the
        # baseline.
        torch.manual_seed(0)
        B, H, S, D = 1, 2, 6, 8
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        q = torch.randn(B, H, 1, D)

        out, info = csa_hca_attention(
            q, K, V,
            csa_block_size=1,
            hca_block_size=S,  # single global block
            csa_top_k_blocks=S,
        )
        assert info["n_csa_blocks"] == S
        assert info["n_hca_blocks"] == 1
        assert info["k_eff"] == S

        # Manual: keys = [original S tokens] ++ [global mean]; values likewise.
        global_k = K.mean(dim=-2, keepdim=True)
        global_v = V.mean(dim=-2, keepdim=True)
        K_aug = torch.cat([K, global_k], dim=-2)  # (B, H, S+1, D)
        V_aug = torch.cat([V, global_v], dim=-2)
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, K_aug.transpose(-1, -2)) * scale
        weights = torch.softmax(scores, dim=-1)
        expected = torch.matmul(weights, V_aug)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    def test_gqa_compatibility(self):
        torch.manual_seed(1)
        B, H_q, H_kv, S, D = 1, 4, 2, 16, 8
        K = torch.randn(B, H_kv, S, D)
        V = torch.randn(B, H_kv, S, D)
        q = torch.randn(B, H_q, 1, D)
        out, info = csa_hca_attention(
            q, K, V,
            csa_block_size=2,
            hca_block_size=8,
            csa_top_k_blocks=2,
        )
        assert out.shape == (B, H_q, 1, D)
        assert info["n_csa_blocks"] == 8
        assert info["n_hca_blocks"] == 2

    def test_invalid_block_sizes(self):
        K = torch.randn(1, 1, 4, 4)
        V = torch.randn(1, 1, 4, 4)
        q = torch.randn(1, 1, 1, 4)
        with pytest.raises(ValueError, match="hca_block_size"):
            csa_hca_attention(q, K, V, csa_block_size=4, hca_block_size=2, csa_top_k_blocks=1)


# ---------------------------------------------------------------------------
# Strategy wrapper + registry
# ---------------------------------------------------------------------------


@pytest.fixture
def quantizer():
    ops = get_backend("torch")
    return PolarQuantizer(head_dim=64, bits=8, seed=42, ops=ops)


class TestCSAHCAStrategy:
    def _make_strategy(self, **kwargs):
        from tqai.strategies.csa_hca import CSAHCAStrategy
        return CSAHCAStrategy(**kwargs)

    def test_registration(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import get_strategy
        s = get_strategy("csa_hca")
        assert s.name == "csa_hca"

    def test_compress_returns_two_views(self, quantizer):
        strat = self._make_strategy(csa_block_size=2, hca_block_size=4)
        x = torch.randn(1, 4, 16, 64)
        compressed, state = strat.compress(x, quantizer)
        tag, meta = compressed
        assert tag == "csa_hca"
        assert "csa" in meta and "hca" in meta
        assert meta["seq_len"] == 16
        assert state["last_n_csa"] == 8
        assert state["last_n_hca"] == 4

    def test_compress_with_scored_entry(self, quantizer):
        strat = self._make_strategy(csa_block_size=2, hca_block_size=4)
        x = torch.randn(1, 4, 8, 64)
        entry = [ScoredEntry(data=x, score=0.7, tier=2, metadata={})]
        compressed, _ = strat.compress(entry, quantizer)
        assert compressed[0] == "csa_hca"

    def test_decompress_broadcast_recovers_shape(self, quantizer):
        strat = self._make_strategy(csa_block_size=2, hca_block_size=4)
        x = torch.randn(1, 4, 16, 64)
        compressed, state = strat.compress(x, quantizer)
        recon = strat.decompress(compressed, quantizer, state)
        assert recon.shape == x.shape

    def test_decompress_no_broadcast_returns_blocks(self, quantizer):
        strat = self._make_strategy(
            csa_block_size=2, hca_block_size=4, broadcast_on_decompress=False
        )
        x = torch.randn(1, 4, 16, 64)
        compressed, _ = strat.compress(x, quantizer)
        recon = strat.decompress(compressed, quantizer)
        assert "csa" in recon and "hca" in recon
        assert recon["csa"].shape == (1, 4, 8, 64)
        assert recon["hca"].shape == (1, 4, 4, 64)

    def test_constant_input_recovers_exactly_under_broadcast(self, quantizer):
        # When all tokens in a block are identical, mean-pooling is lossless
        # and broadcast reconstruction is exact (modulo quantization noise).
        x = torch.zeros(1, 1, 8, 64)
        x[0, 0, :4, :] = 1.0  # first 4 tokens all-ones
        x[0, 0, 4:, :] = -1.0  # last 4 tokens all -ones
        strat = self._make_strategy(csa_block_size=4, hca_block_size=8)
        compressed, _ = strat.compress(x, quantizer)
        recon = strat.decompress(compressed, quantizer)
        # 8-bit quantizer is byte-identical for unit-norm directions; small
        # tolerance is fine for our per-channel data.
        torch.testing.assert_close(recon, x, rtol=5e-2, atol=5e-2)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            self._make_strategy(csa_block_size=8, hca_block_size=4)
        with pytest.raises(ValueError):
            self._make_strategy(csa_block_size=0)

    def test_non_tensor_input_rejected(self, quantizer):
        strat = self._make_strategy()
        with pytest.raises(TypeError):
            strat.compress([1.0, 2.0], quantizer)

    def test_stats_accumulate(self, quantizer):
        strat = self._make_strategy(csa_block_size=2, hca_block_size=4)
        x = torch.randn(1, 1, 8, 64)
        state = {}
        for _ in range(3):
            _, state = strat.compress(x, quantizer, state)
        assert state["csa_hca_stats"]["calls"] == 3
        assert state["csa_hca_stats"]["total_seq_len"] == 24
