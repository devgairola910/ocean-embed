"""Self-Supervised Masked Autoencoder (MAE) with 2D Sinusoidal Position Embeddings for Ocean Surface Fields.

Follows architecture.md & design.md:
- Pretrains on multi-channel lagged satellite surface fields without subsurface labels
- 75% patch masking ratio
- 2D Sinusoidal positional embeddings respecting spatial lat/lon semantics
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sinusoidal2DPositionalEmbedding(nn.Module):
    """Generates 2D sinusoidal positional encodings for grid patches."""

    def __init__(self, embed_dim: int, grid_size: Tuple[int, int], temperature: float = 10000.0) -> None:
        """Initialize 2D sinusoidal embedding.

        Args:
            embed_dim: Total embedding dimension (must be divisible by 4).
            grid_size: Tuple of (grid_h, grid_w) number of patches.
            temperature: Positional frequency temperature.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size
        self.temperature = temperature

        pe = self._build_2d_sinusoid()
        self.register_buffer("pe", pe, persistent=False)  # (1, num_patches, embed_dim)

    def _build_2d_sinusoid(self) -> torch.Tensor:
        """Construct static 2D sinusoidal position table."""
        d_half = self.embed_dim // 2
        d_quarter = d_half // 2

        # Create coordinate meshes
        y = torch.arange(self.grid_h, dtype=torch.float32)
        x = torch.arange(self.grid_w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        grid_y = grid_y.flatten()  # (N_patches,)
        grid_x = grid_x.flatten()  # (N_patches,)

        omega = torch.arange(d_quarter, dtype=torch.float32) / d_quarter
        omega = 1.0 / (self.temperature ** omega)  # (d_quarter,)

        out_y = torch.einsum("m,d->md", grid_y, omega)
        out_x = torch.einsum("m,d->md", grid_x, omega)

        pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # (N_patches, d_half)
        pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # (N_patches, d_half)

        pe = torch.cat([pe_y, pe_x], dim=1)  # (N_patches, embed_dim)
        return pe.unsqueeze(0)  # (1, N_patches, embed_dim)

    def forward(self) -> torch.Tensor:
        """Return position embedding tensor."""
        return self.pe


class PatchEmbedding2D(nn.Module):
    """Splits (B, in_channels, H, W) into non-overlapping patches and projects to D_model."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 16,
        img_size: Tuple[int, int] = (100, 240)
    ) -> None:
        """Initialize patch projection layer.

        Args:
            in_channels: Total input channels (T_lag * C = 5 * 7 = 35).
            embed_dim: Target embedding dimension D_model.
            patch_size: Square patch size P in grid cells.
            img_size: Spatial dimensions (H, W).
        """
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        pad_h = (patch_size - img_size[0] % patch_size) % patch_size
        pad_w = (patch_size - img_size[1] % patch_size) % patch_size
        self.grid_h = (img_size[0] + pad_h) // patch_size
        self.grid_w = (img_size[1] + pad_w) // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image tensor to patch token sequence.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        # Pad spatial dimensions if not perfectly divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # Proj: (B, embed_dim, grid_h, grid_w) -> flatten to (B, embed_dim, N) -> transpose to (B, N, embed_dim)
        out = self.proj(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block with multi-head self-attention and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """Initialize transformer block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feed-forward transformation."""
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class OceanMAEEncoder(nn.Module):
    """Masked Autoencoder (MAE) for self-supervised pretraining on surface satellite fields."""

    def __init__(
        self,
        in_channels: int = 35,       # T_lag * C = 5 * 7
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        img_size: Tuple[int, int] = (100, 240),
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 3,
        decoder_num_heads: int = 4
    ) -> None:
        """Initialize OceanMAE model.

        Args:
            in_channels: 35 (5 days lagged stack of 7 surface channels)
            embed_dim: Encoder latent dimension
            depth: Number of encoder Transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            patch_size: Spatial patch size in degrees/grid cells
            img_size: Input spatial resolution (H, W)
            mask_ratio: Fraction of patches to mask out during pretraining (default: 0.75)
            decoder_embed_dim: Lightweight pretraining decoder dimension
            decoder_depth: Pretraining decoder depth
            decoder_num_heads: Pretraining decoder attention heads
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_ratio = mask_ratio

        # 1. Patch Embedding & 2D Position Encoding
        self.patch_embed = PatchEmbedding2D(in_channels, embed_dim, patch_size, img_size)
        self.pos_embed = Sinusoidal2DPositionalEmbedding(
            embed_dim,
            (self.patch_embed.grid_h, self.patch_embed.grid_w)
        )

        # 2. Encoder Transformer Blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 3. Pretraining Reconstruction Decoder (Lightweight)
        self.decoder_embed_dim = decoder_embed_dim
        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = Sinusoidal2DPositionalEmbedding(
            decoder_embed_dim,
            (self.patch_embed.grid_h, self.patch_embed.grid_w)
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head: projects decoder token back to patch pixel values
        self.patch_pixels = patch_size * patch_size * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_pixels)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random patch masking (MAE mechanism).

        Args:
            x: Patch token tensor of shape (B, N, D)
            mask_ratio: Fraction to mask (e.g. 0.75)

        Returns:
            Tuple of:
              - x_visible: (B, N_visible, D)
              - mask: Binary tensor (B, N) where 1 is masked and 0 is visible
              - ids_restore: Index tensor to restore original sequence order (B, N)
        """
        B, N, D = x.shape
        len_keep = int(N * (1.0 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass input through patch projection, position embedding, masking, and encoder blocks.

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W) or (B, in_channels, H, W)
            mask_ratio: If provided, apply masking. If None or 0.0, process all patches.

        Returns:
            Tuple of (latent_tokens, mask, ids_restore)
        """
        # Flatten lag and channel dimensions: (B, T_lag, C, H, W) -> (B, T_lag * C, H, W)
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)

        # Patchify and add 2D position embeddings
        tokens = self.patch_embed(x)  # (B, N, D)
        pe = self.pos_embed().to(tokens.device)
        tokens = tokens + pe

        if mask_ratio is not None and mask_ratio > 0.0:
            tokens_vis, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        else:
            tokens_vis = tokens
            mask = torch.zeros(tokens.shape[0], tokens.shape[1], device=tokens.device)
            ids_restore = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).repeat(tokens.shape[0], 1)

        # Encoder transformer blocks
        for blk in self.encoder_blocks:
            tokens_vis = blk(tokens_vis)
        latent = self.encoder_norm(tokens_vis)

        return latent, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Lightweight decoder reconstructing original patch values for pretraining loss.

        Args:
            latent: Visible patch tokens from encoder (B, N_visible, D_enc)
            ids_restore: Indices to re-assemble full sequence (B, N)

        Returns:
            Reconstructed patch values (B, N, patch_pixels)
        """
        B = latent.shape[0]
        N = ids_restore.shape[1]

        # Project encoder dimension to decoder dimension
        dec_tokens = self.enc_to_dec(latent)  # (B, N_vis, D_dec)

        # Append mask tokens for all masked positions
        num_mask = N - dec_tokens.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        full_tokens = torch.cat([dec_tokens, mask_tokens], dim=1)

        # Unshuffle to original spatial order
        full_tokens = torch.gather(
            full_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed_dim)
        )

        # Add decoder 2D position embedding
        pe_dec = self.decoder_pos_embed().to(full_tokens.device)
        full_tokens = full_tokens + pe_dec

        # Pass through decoder transformer blocks
        for blk in self.decoder_blocks:
            full_tokens = blk(full_tokens)
        full_tokens = self.decoder_norm(full_tokens)

        # Project to patch pixels
        pred_pixels = self.decoder_pred(full_tokens)  # (B, N, patch_pixels)
        return pred_pixels

    def forward_pretrain_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute self-supervised pretraining MSE loss over masked patches.

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W)

        Returns:
            Tuple of:
              - loss: Scalar reconstruction MSE on masked patches
              - pred_patches: (B, N, patch_pixels)
              - mask: (B, N)
        """
        # Patchify target for ground-truth comparison
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B, T * C, H, W)
        else:
            x_flat = x

        # Extract ground truth target patches (B, N, patch_pixels)
        P = self.patch_size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x_flat = F.pad(x_flat, (0, pad_w, 0, pad_h), mode="replicate")

        gh = x_flat.shape[2] // P
        gw = x_flat.shape[3] // P

        # Reshape to patches
        # (B, C_in, gh, P, gw, P) -> (B, gh, gw, C_in, P, P) -> (B, N, P*P*C_in)
        target_patches = x_flat.view(B, -1, gh, P, gw, P).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, -1)

        # Forward encoder with masking
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=self.mask_ratio)
        # Forward decoder
        pred_patches = self.forward_decoder(latent, ids_restore)

        # Loss only on masked patches (MAE standard)
        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)  # (B, N) mean error per patch
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss, pred_patches, mask

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch embeddings without masking (for Stage C Depth Decoder).

        Args:
            x: Input tensor of shape (B, T_lag, C, H, W)

        Returns:
            Spatial patch embeddings of shape (B, N_patches, D_model)
        """
        latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)
        return latent
