"""Stage C: Supervised Depth Decoder Fine-Tuning with Physics-Aware Constraints.

Follows architecture.md §3 & design.md §4-5:
- Trains depth decoder to predict 15-level temperature anomalies + thermocline depth
- Regularized by differentiable physical stability penalty (OceanPhysicsLoss)
- Evaluates on held-out validation years (no temporal leakage)
"""

from typing import Dict, Optional, Tuple
import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.models.depth_decoder import OceanEmbedFullModel
from src.models.physics_loss import OceanPhysicsLoss, verify_profile_monotonicity


def train_decoder_epoch(
    model: OceanEmbedFullModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: OceanPhysicsLoss,
    device: torch.device
) -> Dict[str, float]:
    """Train Depth Decoder for one epoch.

    Args:
        model: OceanEmbedFullModel
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: OceanPhysicsLoss instance
        device: Target compute device

    Returns:
        Dict of average loss components across epoch.
    """
    model.train()
    metrics_sum = {"loss_total": 0.0, "loss_temp": 0.0, "loss_thermo": 0.0, "loss_stability": 0.0}
    num_batches = 0

    for batch in dataloader:
        surface_inputs = batch["surface_input"].to(device)
        target_depth = batch["target_depth"].to(device)
        target_thermo = batch["target_thermocline"].to(device)
        land_mask = batch["land_mask"].to(device)

        optimizer.zero_grad()
        pred_depth, pred_thermo = model(surface_inputs)

        losses = criterion(pred_depth, target_depth, pred_thermo, target_thermo, land_mask)
        losses["loss_total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in metrics_sum:
            metrics_sum[k] += losses[k].item()
        num_batches += 1

    return {k: v / max(1, num_batches) for k, v in metrics_sum.items()}


def evaluate_decoder(
    model: OceanEmbedFullModel,
    dataloader: DataLoader,
    criterion: OceanPhysicsLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on held-out validation dataset.

    Returns:
        Dict with validation losses and physical stability percentage.
    """
    model.eval()
    metrics_sum = {"loss_total": 0.0, "loss_temp": 0.0, "loss_thermo": 0.0, "loss_stability": 0.0}
    num_batches = 0
    all_pred_profiles = []

    with torch.no_grad():
        for batch in dataloader:
            surface_inputs = batch["surface_input"].to(device)
            target_depth = batch["target_depth"].to(device)
            target_thermo = batch["target_thermocline"].to(device)
            land_mask = batch["land_mask"].to(device)

            pred_depth, pred_thermo = model(surface_inputs)
            losses = criterion(pred_depth, target_depth, pred_thermo, target_thermo, land_mask)

            for k in metrics_sum:
                metrics_sum[k] += losses[k].item()
            num_batches += 1

            # Extract random sample profiles over ocean to check physical stability
            # (B, 15, H, W) -> sample ocean points
            mask_np = land_mask.cpu().numpy()
            pred_np = pred_depth.cpu().numpy()
            for b in range(pred_np.shape[0]):
                ocean_idx = np.where(mask_np[b] if mask_np.ndim == 3 else mask_np)
                if len(ocean_idx[0]) > 0:
                    chosen = np.random.choice(len(ocean_idx[0]), size=min(10, len(ocean_idx[0])), replace=False)
                    for c in chosen:
                        i_idx, j_idx = ocean_idx[0][c], ocean_idx[1][c]
                        all_pred_profiles.append(pred_np[b, :, i_idx, j_idx])

    results = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    if all_pred_profiles:
        prof_arr = np.array(all_pred_profiles)
        results["stability_pct"] = verify_profile_monotonicity(prof_arr)
    else:
        results["stability_pct"] = 100.0

    return results


def run_supervised_training(
    model: OceanEmbedFullModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    w_temp: float = 1.0,
    w_thermo: float = 0.2,
    w_stability: float = 0.1,
    device: Optional[torch.device] = None,
    save_dir: str = "checkpoints"
) -> Dict[str, list]:
    """Execute Stage C supervised training and fine-tuning loop.

    Args:
        model: OceanEmbedFullModel instance
        train_loader: Training DataLoader (e.g. 2016-2020)
        val_loader: Validation DataLoader (e.g. 2021)
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        w_temp: Temperature anomaly loss weight
        w_thermo: Thermocline loss weight
        w_stability: Physical stability loss weight
        device: Device
        save_dir: Checkpoint directory

    Returns:
        Training history dict.
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(dev)
    os.makedirs(save_dir, exist_ok=True)

    criterion = OceanPhysicsLoss(w_temp=w_temp, w_thermo=w_thermo, w_stability=w_stability)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_temp_loss": [], "val_stability_pct": []}
    date_str = datetime.now().strftime("%Y%m%d")
    best_val_loss = float("inf")

    print(f"=== Starting Stage C: Depth Decoder Fine-Tuning ({epochs} epochs on {dev}) ===")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_decoder_epoch(model, train_loader, optimizer, criterion, dev)
        val_metrics = evaluate_decoder(model, val_loader, criterion, dev)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss_total"])
        history["val_loss"].append(val_metrics["loss_total"])
        history["val_temp_loss"].append(val_metrics["loss_temp"])
        history["val_stability_pct"].append(val_metrics["stability_pct"])

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_metrics['loss_total']:.4f} (Temp: {train_metrics['loss_temp']:.4f}) | "
            f"Val Loss: {val_metrics['loss_total']:.4f} (Temp: {val_metrics['loss_temp']:.4f}, Stability: {val_metrics['stability_pct']:.1f}%) | "
            f"{elapsed:.2f}s"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_path = os.path.join(save_dir, f"decoder_vit-mae_{date_str}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "history": history
            }, best_path)

    final_path = os.path.join(save_dir, f"decoder_vit-mae_{date_str}_e{epochs}.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "history": history
    }, final_path)

    print(f"Fine-tuning complete in {time.time() - start_time:.2f}s. Saved best model to {best_path}")
    return history
