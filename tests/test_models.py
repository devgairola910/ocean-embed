"""Unit tests for model architectures, positional encodings, and forward pass shapes."""

import pytest
import torch
from src.models.mae_encoder import OceanMAEEncoder, PatchEmbedding2D, Sinusoidal2DPositionalEmbedding
from src.models.depth_decoder import DepthConditionedDecoder, OceanEmbedFullModel
from src.models.baseline_direct import DirectRegressionBaseline, ClimatologyBaseline


def test_patch_embedding_and_2d_sinusoid():
    H, W = 32, 64
    patch_size = 8
    in_channels = 35
    embed_dim = 64

    patch_proj = PatchEmbedding2D(in_channels, embed_dim, patch_size=patch_size, img_size=(H, W))
    assert patch_proj.grid_h == 4
    assert patch_proj.grid_w == 8
    assert patch_proj.num_patches == 32

    x = torch.randn(2, in_channels, H, W)
    tokens = patch_proj(x)
    assert tokens.shape == (2, 32, embed_dim)

    pos_embed = Sinusoidal2DPositionalEmbedding(embed_dim, (4, 8))
    pe = pos_embed()
    assert pe.shape == (1, 32, embed_dim)


def test_mae_masking_and_pretraining_loss():
    H, W = 32, 64
    in_channels = 35
    embed_dim = 64

    model = OceanMAEEncoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        patch_size=8,
        img_size=(H, W),
        mask_ratio=0.75,
        decoder_embed_dim=32,
        decoder_depth=2,
        decoder_num_heads=2
    )

    # Input: (B=2, T_lag=5, C=7, H=32, W=64)
    x = torch.randn(2, 5, 7, H, W)
    loss, pred_patches, mask = model.forward_pretrain_loss(x)

    assert loss.ndim == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert mask.shape == (2, 32)
    # Mask ratio should be approximately 75%
    masked_count = mask.sum().item()
    assert 40 <= masked_count <= 56  # 32*2*0.75 = 48 expected


def test_depth_decoder_and_full_model_forward():
    H, W = 32, 64
    full_model = OceanEmbedFullModel(
        in_channels=35,
        embed_dim=64,
        encoder_depth=2,
        decoder_layers=2,
        num_heads=4,
        patch_size=8,
        img_size=(H, W)
    )

    x = torch.randn(2, 5, 7, H, W)
    pred_depth, pred_thermo = full_model(x)

    # Shape contracts
    # pred_depth: (B=2, D=15, H=32, W=64)
    assert pred_depth.shape == (2, 15, H, W)
    # pred_thermo: (B=2, H=32, W=64)
    assert pred_thermo.shape == (2, H, W)
    assert not torch.isnan(pred_depth).any()
    assert not torch.isnan(pred_thermo).any()


def test_direct_regression_and_climatology_baselines():
    H, W = 32, 64
    x = torch.randn(2, 5, 7, H, W)

    direct_model = DirectRegressionBaseline(in_channels=35, num_depths=15, hidden_dim=32)
    p_dep, p_th = direct_model(x)
    assert p_dep.shape == (2, 15, H, W)
    assert p_th.shape == (2, H, W)

    clim_model = ClimatologyBaseline()
    c_dep, c_th = clim_model(x)
    assert c_dep.shape == (2, 15, H, W)
    assert torch.all(c_dep == 0.0)
