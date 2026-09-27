"""Stage B: Self-Supervised MAE Pretraining on Satellite Surface Fields.

Follows architecture.md §3 & design.md §5:
- Masks 75% of patches and reconstructs surface fields using lightweight decoder
- Trains on the full satellite surface record without subsurface labels
- Uses AdamW with linear warmup and cosine decay
"""

from typing import Dict, Optional, Tuple
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.models.mae_encoder import OceanMAEEncoder


def pretrain_mae_epoch(
    model: OceanMAEEncoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train OceanMAE for one epoch on unlabelled surface inputs.

    Args:
        model: OceanMAEEncoder model
        dataloader: DataLoader yielding surface input batches
        optimizer: AdamW optimizer
        device: Target compute device

    Returns:
        Average reconstruction loss across the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        surface_inputs = batch["surface_input"].to(device)  # (B, T_lag, C, H, W)
        optimizer.zero_grad()

        loss, _, _ = model.forward_pretrain_loss(surface_inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def run_pretraining(
    model: OceanMAEEncoder,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    device: Optional[torch.device] = None,
    save_dir: str = "checkpoints"
) -> Dict[str, list]:
    """Execute Stage B self-supervised pretraining loop.

    Args:
        model: OceanMAEEncoder model instance
        train_loader: DataLoader with surface tensors
        epochs: Number of training epochs
        lr: Peak learning rate
        weight_decay: Weight decay
        device: Device or auto-detected
        save_dir: Checkpoint directory

    Returns:
        History dict with loss trajectory.
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(dev)
    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"loss": []}
    date_str = datetime.now().strftime("%Y%m%d")

    print(f"=== Starting Stage B: MAE Pretraining ({epochs} epochs on {dev}) ===")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        loss = pretrain_mae_epoch(model, train_loader, optimizer, dev)
        scheduler.step()
        history["loss"].append(loss)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Pretrain Masked MSE: {loss:.5f} - LR: {scheduler.get_last_lr()[0]:.2e} - {elapsed:.2f}s")

    # Save final pretraining checkpoint according to rules.md naming convention
    ckpt_path = os.path.join(save_dir, f"pretrain_vit-mae_{date_str}_e{epochs}.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, ckpt_path)
    print(f"Pretraining complete in {time.time() - start_time:.2f}s. Saved checkpoint to {ckpt_path}")

    return history
