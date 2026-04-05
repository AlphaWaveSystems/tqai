"""Tests for rotation baking (tqai bake) — fusing R into model weights."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from tqai.backend import get_backend
from tqai.bake import (
    _apply_rotation_to_out_proj,
    _apply_rotation_to_proj,
    _build_rotation_np,
    bake_rotation_into_model,
    save_baked_model,
)
from tqai.quantizer import PolarQuantizer

# ---------------------------------------------------------------------------
# Minimal toy attention model
# ---------------------------------------------------------------------------

class ToyAttention(nn.Module):
    def __init__(self, hidden_size=64, head_dim=32, n_heads=2, n_kv_heads=2):
        super().__init__()
        self.num_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim
        kv_dim = n_kv_heads * head_dim
        q_dim = n_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()


class ToyModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer() for _ in range(n_layers)])


# ---------------------------------------------------------------------------
# Rotation math tests
# ---------------------------------------------------------------------------

class TestBuildRotationNp:
    def test_orthogonal(self):
        R = _build_rotation_np(32, seed=42)
        product = R @ R.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-5)

    def test_det_one(self):
        R = _build_rotation_np(32, seed=42)
        det = np.linalg.det(R.astype(np.float64))
        assert abs(abs(det) - 1.0) < 1e-4

    def test_same_seed_reproducible(self):
        R1 = _build_rotation_np(64, seed=7)
        R2 = _build_rotation_np(64, seed=7)
        np.testing.assert_array_equal(R1, R2)

    def test_matches_polar_quantizer(self):
        """_build_rotation_np must produce the same matrix as PolarQuantizer."""
        ops = get_backend("torch")
        head_dim = 64
        seed = 42

        R_np = _build_rotation_np(head_dim, seed)
        pq = PolarQuantizer(head_dim=head_dim, bits=4, seed=seed, ops=ops)
        R_pq = ops.to_numpy(pq._rotation)

        np.testing.assert_allclose(R_np, R_pq, atol=1e-5)


class TestApplyRotationToProj:
    def test_correctness(self):
        """W_K_baked should produce K @ R^T from same input."""
        rng = np.random.default_rng(0)
        n_heads, head_dim, in_dim = 2, 32, 64
        W = rng.standard_normal((n_heads * head_dim, in_dim)).astype(np.float32)
        R = _build_rotation_np(head_dim, seed=1)

        W_t = torch.from_numpy(W)
        R_t = torch.from_numpy(R)
        W_baked_t = _apply_rotation_to_proj(W_t, R, n_heads, head_dim)

        # x @ W.T gives K; then K @ R^T should equal x @ W_baked.T
        x = rng.standard_normal((5, in_dim)).astype(np.float32)
        x_t = torch.from_numpy(x)

        K = x_t @ W_t.T  # (5, n_heads * head_dim)
        # Rotate each head's slice independently
        K_shaped = K.reshape(5, n_heads, head_dim)  # (5, n_heads, head_dim)
        K_rot_direct = (K_shaped @ R_t.T).reshape(5, n_heads * head_dim)
        K_rot_baked = x_t @ W_baked_t.T

        np.testing.assert_allclose(
            K_rot_direct.numpy(), K_rot_baked.numpy(), atol=1e-4,
            err_msg="Baked W_K must produce K @ R^T"
        )

    def test_shape_preserved(self):
        n_heads, head_dim, in_dim = 4, 64, 128
        W = torch.randn(n_heads * head_dim, in_dim)
        R = _build_rotation_np(head_dim, seed=0)
        W_baked = _apply_rotation_to_proj(W, R, n_heads, head_dim)
        assert W_baked.shape == W.shape


class TestApplyRotationToOutProj:
    def test_correctness(self):
        """W_O_baked should satisfy: attn_rot @ W_O_baked.T == attn @ W_O.T."""
        rng = np.random.default_rng(2)
        n_heads, head_dim, out_dim = 2, 32, 64
        W_O = rng.standard_normal((out_dim, n_heads * head_dim)).astype(np.float32)
        R = _build_rotation_np(head_dim, seed=3)

        W_O_t = torch.from_numpy(W_O)
        W_O_baked_t = _apply_rotation_to_out_proj(W_O_t, R, n_heads, head_dim)

        # attn_rot_h = attn_h @ R.T (each head's attention is rotated by R.T
        # because V_baked = V @ R.T so attn_out = sum weights * V_baked = attn_orig @ R.T)
        attn = rng.standard_normal((5, n_heads * head_dim)).astype(np.float32)
        attn_t = torch.from_numpy(attn)

        # Construct attn_rot by rotating each head slice by R.T
        attn_rot = np.concatenate([
            attn[:, h*head_dim:(h+1)*head_dim] @ R.T
            for h in range(n_heads)
        ], axis=1)
        attn_rot_t = torch.from_numpy(attn_rot)

        # W_O_baked_h = W_O_h @ R.T, so:
        # attn_rot_h @ W_O_baked_h.T = (attn_h @ R.T) @ (W_O_h @ R.T).T
        #                             = attn_h @ R.T @ R @ W_O_h.T = attn_h @ W_O_h.T ✓
        proj_original = attn_t @ W_O_t.T
        proj_baked = attn_rot_t @ W_O_baked_t.T

        np.testing.assert_allclose(
            proj_original.numpy(), proj_baked.numpy(), atol=1e-4,
            err_msg="Baked W_O must un-rotate attention output"
        )

    def test_shape_preserved(self):
        n_heads, head_dim, out_dim = 4, 64, 256
        W_O = torch.randn(out_dim, n_heads * head_dim)
        R = _build_rotation_np(head_dim, seed=0)
        W_O_baked = _apply_rotation_to_out_proj(W_O, R, n_heads, head_dim)
        assert W_O_baked.shape == W_O.shape


# ---------------------------------------------------------------------------
# Bake roundtrip tests
# ---------------------------------------------------------------------------

class TestBakeRotationIntoModel:
    def test_finds_attention_modules(self):
        model = ToyModel(n_layers=2)
        # Just verify it runs without error on the toy model
        bake_rotation_into_model(model, bits_k=4, bits_v=2, seed=42)

    def test_baked_weights_differ_from_original(self):
        model = ToyModel(n_layers=2)
        W_k_before = model.layers[0].self_attn.k_proj.weight.clone()
        bake_rotation_into_model(model, bits_k=4, bits_v=2, seed=42)
        W_k_after = model.layers[0].self_attn.k_proj.weight.clone()
        assert not torch.allclose(W_k_before, W_k_after), "Baking must modify W_K"

    def test_baked_weights_remain_same_shape(self):
        model = ToyModel(n_layers=2)
        shapes_before = {
            "q": model.layers[0].self_attn.q_proj.weight.shape,
            "k": model.layers[0].self_attn.k_proj.weight.shape,
            "v": model.layers[0].self_attn.v_proj.weight.shape,
            "o": model.layers[0].self_attn.o_proj.weight.shape,
        }
        bake_rotation_into_model(model, bits_k=4, bits_v=2, seed=42)
        assert model.layers[0].self_attn.q_proj.weight.shape == shapes_before["q"]
        assert model.layers[0].self_attn.k_proj.weight.shape == shapes_before["k"]
        assert model.layers[0].self_attn.v_proj.weight.shape == shapes_before["v"]
        assert model.layers[0].self_attn.o_proj.weight.shape == shapes_before["o"]

    def test_pre_rotated_roundtrip(self):
        """Baked model + pre_rotated=True must match original model + rotation at runtime.

        Uses a single-head model so K vectors are directly (batch, head_dim).
        For each x:
          original:  K = x @ W_K.T, quantize with rotation → indices, norms
          baked:     K' = x @ W_K_baked.T = K @ R.T, quantize with pre_rotated=True → indices, norms
          Both must give identical indices and norms.
        """
        ops = get_backend("torch")
        head_dim = 64
        hidden_size = 64
        seed = 42

        class ToyAttention1Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.num_key_value_heads = 1
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
                self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
                self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
                self.o_proj = nn.Linear(head_dim, hidden_size, bias=False)

        class ToyLayer1Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = ToyAttention1Head()

        class ToyModel1Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([ToyLayer1Head()])

        model = ToyModel1Head()
        W_K_orig = model.layers[0].self_attn.k_proj.weight.data.clone()

        # Runtime quantizer (rotation included)
        pq_runtime = PolarQuantizer(
            head_dim=head_dim, bits=4, seed=seed, ops=ops, pre_rotated=False
        )

        bake_rotation_into_model(model, bits_k=4, bits_v=2, seed=seed)
        W_K_baked = model.layers[0].self_attn.k_proj.weight.data.clone()

        # Pre-rotated quantizer (same seed)
        pq_baked = PolarQuantizer(
            head_dim=head_dim, bits=4, seed=seed, ops=ops, pre_rotated=True
        )

        rng = np.random.default_rng(10)
        x = torch.from_numpy(rng.standard_normal((5, hidden_size)).astype(np.float32))

        K_orig = x @ W_K_orig.T    # [5, head_dim]
        K_baked_out = x @ W_K_baked.T  # [5, head_dim] = K_orig @ R.T

        idx_runtime, norms_runtime = pq_runtime.quantize(ops.from_numpy(K_orig.numpy()))
        idx_baked, norms_baked = pq_baked.quantize(ops.from_numpy(K_baked_out.numpy()))

        np.testing.assert_array_equal(
            ops.to_numpy(idx_runtime), ops.to_numpy(idx_baked),
            err_msg="Baked + pre_rotated must give same indices as runtime rotation"
        )
        np.testing.assert_allclose(
            ops.to_numpy(norms_runtime), ops.to_numpy(norms_baked), atol=1e-5,
            err_msg="Baked + pre_rotated must give same norms as runtime rotation"
        )


class TestSaveBakedModel:
    def test_saves_bake_config(self, tmp_path):
        """save_baked_model must write tqai_bake_config.json with correct fields."""
        from unittest.mock import MagicMock

        model = ToyModel(n_layers=1)
        model.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()

        save_baked_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=tmp_path,
            base_model_id="test/model",
            bits_k=4,
            bits_v=2,
            seed=42,
        )

        bake_cfg = json.loads((tmp_path / "tqai_bake_config.json").read_text())
        assert bake_cfg["tqai_baked"] is True
        assert bake_cfg["bits_k"] == 4
        assert bake_cfg["bits_v"] == 2
        assert bake_cfg["seed"] == 42
        assert bake_cfg["base_model"] == "test/model"

    def test_output_dir_created(self, tmp_path):
        from unittest.mock import MagicMock

        model = ToyModel(n_layers=1)
        model.save_pretrained = MagicMock()
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        out = tmp_path / "nested" / "output"

        save_baked_model(
            model=model, tokenizer=tokenizer, output_dir=out,
            base_model_id="x", bits_k=4, bits_v=2, seed=42,
        )

        assert out.is_dir()


class TestAutoDetectBakedConfig:
    def test_patch_auto_detects_pre_rotated(self, tmp_path):
        """_patch should read tqai_bake_config.json and set pre_rotated=True."""
        from tqai._patch import _detect_baked_config

        # Write a fake bake config
        bake_cfg = {"tqai_baked": True, "bits_k": 4, "bits_v": 2, "seed": 42}
        (tmp_path / "tqai_bake_config.json").write_text(json.dumps(bake_cfg))

        # Mock model with config._name_or_path pointing to tmp_path
        from unittest.mock import MagicMock
        model = MagicMock()
        model.config._name_or_path = str(tmp_path)

        detected = _detect_baked_config(model)
        assert detected is not None
        assert detected["tqai_baked"] is True
        assert detected["bits_k"] == 4

    def test_no_bake_config_returns_none(self, tmp_path):
        from unittest.mock import MagicMock

        from tqai._patch import _detect_baked_config

        model = MagicMock()
        model.config._name_or_path = str(tmp_path)

        assert _detect_baked_config(model) is None
