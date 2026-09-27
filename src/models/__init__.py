"""OceanEmbed model architectures: ViT MAE Encoder, Depth-Conditioned Decoder, and Physics Losses."""

from src.models.mae_encoder import OceanMAEEncoder, PatchEmbedding2D, Sinusoidal2DPositionalEmbedding
from src.models.depth_decoder import DepthConditionedDecoder, OceanEmbedFullModel
from src.models.physics_loss import OceanPhysicsLoss, compute_stability_penalty
from src.models.baseline_direct import DirectRegressionBaseline, ClimatologyBaseline

__all__ = [
    "OceanMAEEncoder",
    "PatchEmbedding2D",
    "Sinusoidal2DPositionalEmbedding",
    "DepthConditionedDecoder",
    "OceanEmbedFullModel",
    "OceanPhysicsLoss",
    "compute_stability_penalty",
    "DirectRegressionBaseline",
    "ClimatologyBaseline"
]
