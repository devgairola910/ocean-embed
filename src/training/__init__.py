"""Training loops and optimization pipelines for OceanEmbed."""

from src.training.pretrain_mae import pretrain_mae_epoch, run_pretraining
from src.training.train_decoder import train_decoder_epoch, evaluate_decoder, run_supervised_training

__all__ = [
    "pretrain_mae_epoch",
    "run_pretraining",
    "train_decoder_epoch",
    "evaluate_decoder",
    "run_supervised_training"
]
