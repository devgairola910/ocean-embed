"""Supervised Depth-Conditioned Decoder and Full OceanEmbed Model.

Follows architecture.md & design.md:
- Stage C: Cross-attention over spatial surface embeddings conditioned on 15 learned depth query tokens
- Predicts 15-depth temperature anomalies: (B, 15, H, W)
- Predicts auxiliary thermocline depth map: (B, H, W)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mae_encoder import OceanMAEEncoder, TransformerBlock
from src.data.grid import STANDARD_DEPTH_LEVELS


class DepthCrossAttentionBlock(nn.Module):
    """Transformer block with multi-head cross-attention over spatial patch embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """Initialize cross-attention block."""
        super().__init__()
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention (queries attending to spatial context) + self-attention + MLP.

        Args:
            queries: Tensor of shape (B, N_queries, embed_dim)
            context: Tensor of shape (B, N_patches, embed_dim)

        Returns:
            Updated queries tensor of shape (B, N_queries, embed_dim)
        """
        # Cross-Attention: Depth tokens query spatial surface tokens
        q_norm = self.norm_query(queries)
        c_norm = self.norm_context(context)
        attn_cross, _ = self.cross_attn(query=q_norm, key=c_norm, value=c_norm)
        queries = queries + attn_cross

        # Self-Attention among depth queries
        s_norm = self.norm_self(queries)
        attn_self, _ = self.self_attn(query=s_norm, key=s_norm, value=s_norm)
        queries = queries + attn_self

        # Feed-Forward MLP
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class DepthConditionedDecoder(nn.Module):
    """Decodes spatial surface embeddings into 15-level subsurface temperature anomaly volumes."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_depths: int = 15,
        depth_levels: Optional[List[float]] = None,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        img_size: Tuple[int, int] = (100, 240),
        patch_size: int = 16
    ) -> None:
        """Initialize DepthConditionedDecoder.

        Args:
            embed_dim: Latent representation dimension
            num_depths: Number of discrete depth levels (15)
            depth_levels: List of depth values in meters
            num_layers: Number of cross-attention decoder layers
            num_heads: Attention heads
            mlp_ratio: Feed-forward expansion ratio
            img_size: Target spatial dimensions (H, W)
            patch_size: Spatial patch size
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_depths = num_depths
        self.depth_levels = depth_levels or STANDARD_DEPTH_LEVELS
        self.img_size = img_size
        self.patch_size = patch_size

        pad_h = (patch_size - img_size[0] % patch_size) % patch_size
        pad_w = (patch_size - img_size[1] % patch_size) % patch_size
        self.grid_h = (img_size[0] + pad_h) // patch_size
        self.grid_w = (img_size[1] + pad_w) // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # 1. Learned Depth Query Tokens (1 per depth level)
        self.depth_queries = nn.Parameter(torch.randn(1, num_depths, embed_dim) * 0.02)

        # 2. Continuous Depth Embedding projection (combines physical depth in meters with learned token)
        self.depth_phys_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        depth_tensor = torch.tensor(self.depth_levels, dtype=torch.float32).unsqueeze(-1)  # (15, 1)
        self.register_buffer("depth_meters", depth_tensor, persistent=False)

        # 3. Spatial projection head (per patch to spatial grid)
        # Maps decoded token representation -> spatial pixel map
        self.spatial_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size)
        )

        # 4. Cross-Attention Decoder Blocks
        self.decoder_layers = nn.ModuleList([
            DepthCrossAttentionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 5. Auxiliary Thermocline Depth Prediction Head
        self.thermo_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            nn.Upsample(size=img_size, mode="bilinear", align_corners=False),
            nn.ReLU()  # Thermocline depth in meters is non-negative
        )

        # Pointwise 2D fusion refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(num_depths, num_depths, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_depths, num_depths, kernel_size=1)
        )

    def forward(self, spatial_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode spatial surface features into 15 depth levels and thermocline depth map.

        Args:
            spatial_latents: Latent patch tokens from encoder of shape (B, N_patches, embed_dim)

        Returns:
            Tuple of:
              - pred_depth_anomalies: (B, 15, H, W) temperature anomaly map in °C
              - pred_thermocline: (B, H, W) thermocline depth in meters
        """
        B = spatial_latents.shape[0]
        H, W = self.img_size
        P = self.patch_size
        gh, gw = self.grid_h, self.grid_w

        # Combine learned depth queries with physical depth encoding
        phys_emb = self.depth_phys_proj(self.depth_meters).unsqueeze(0)  # (1, 15, embed_dim)
        queries = (self.depth_queries + phys_emb).repeat(B, 1, 1)        # (B, 15, embed_dim)

        # Pass through cross-attention layers
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_latents)
        queries = self.norm(queries)  # (B, 15, embed_dim)

        # Project interaction of depth tokens with each spatial patch
        # (B, 15, 1, D) * (B, 1, N, D) -> (B, 15, N, D)
        combined_feat = queries.unsqueeze(2) * spatial_latents.unsqueeze(1)  # (B, 15, N, D)
        # Project each patch to (P * P) spatial pixels
        patch_pixels = self.spatial_proj(combined_feat)  # (B, 15, N, P*P)

        # Fold patches back into 2D spatial image: (B, 15, gh, gw, P, P) -> (B, 15, H, W)
        patch_pixels = patch_pixels.view(B, self.num_depths, gh, gw, P, P)
        pred_map = patch_pixels.permute(0, 1, 2, 4, 3, 5).reshape(B, self.num_depths, gh * P, gw * P)

        # Trim padding if necessary to match exact (H, W)
        pred_depth_anomalies = self.refine_conv(pred_map[:, :, :H, :W])

        # Auxiliary Thermocline Depth Prediction Head
        # Reshape spatial latents back to (B, embed_dim, gh, gw)
        spatial_2d = spatial_latents.transpose(1, 2).view(B, self.embed_dim, gh, gw)
        pred_thermocline = self.thermo_head(spatial_2d).squeeze(1)[:, :H, :W]  # (B, H, W)

        return pred_depth_anomalies, pred_thermocline


class OceanEmbedFullModel(nn.Module):
    """End-to-End OceanEmbed Model uniting ViT MAE Encoder and Depth-Conditioned Decoder."""

    def __init__(
        self,
        in_channels: int = 35,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
        img_size: Tuple[int, int] = (100, 240),
        freeze_encoder: bool = False
    ) -> None:
        """Initialize Full OceanEmbed Model."""
        super().__init__()
        self.encoder = OceanMAEEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            patch_size=patch_size,
            img_size=img_size
        )
        self.depth_decoder = DepthConditionedDecoder(
            embed_dim=embed_dim,
            num_depths=15,
            num_layers=decoder_layers,
            num_heads=num_heads,
            img_size=img_size,
            patch_size=patch_size
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from 7-channel lagged surface inputs to 15 subsurface depths.

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W)

        Returns:
            Tuple of:
              - pred_depth_anomalies: (B, 15, H, W) in standardized temperature anomaly space
              - pred_thermocline: (B, H, W) in meters
        """
        spatial_latents = self.encoder.extract_features(x)
        pred_depth, pred_thermo = self.depth_decoder(spatial_latents)
        return pred_depth, pred_thermo
