"""End-to-End Orchestrator Pipeline for OceanEmbed.

Runs:
1. Data ingestion & multi-year train/val/test splits (temporal split)
2. Climatology computation
3. Stage B: Self-Supervised ViT MAE pretraining
4. Stage C: Supervised Depth Decoder fine-tuning + Physics Loss
5. Baseline training (Direct-Regression & Climatology)
6. Independent Argo validation & benchmark generation
7. Exports pre-cached datasets for interactive demo
"""

from typing import Dict, Optional, Tuple
import os
import argparse
import time
import torch
import yaml
import numpy as np

from src.data.grid import OceanGrid
from src.data.synthetic import PhysicalOceanSynthesizer
from src.data.dataset import create_dataloaders
from src.models.mae_encoder import OceanMAEEncoder
from src.models.depth_decoder import OceanEmbedFullModel
from src.models.baseline_direct import DirectRegressionBaseline
from src.training.pretrain_mae import run_pretraining
from src.training.train_decoder import run_supervised_training
from src.eval.benchmark import run_full_benchmark


def run_full_oceanembed_pipeline(
    config_path: str = "configs/default_config.yaml",
    pretrain_epochs: int = 10,
    decoder_epochs: int = 15,
    sample_step: int = 5,
    save_demo_cache: bool = True
) -> Dict:
    """Execute the full OceanEmbed machine learning and evaluation lifecycle.

    Args:
        config_path: Path to configuration YAML
        pretrain_epochs: Epochs for Stage B MAE pretraining
        decoder_epochs: Epochs for Stage C Depth Decoder training
        sample_step: Day sampling step (1=daily, 5=fast dev)
        save_demo_cache: Whether to save cache file for Streamlit demo

    Returns:
        Dict containing trained models, benchmark results, and evaluation metrics.
    """
    print("=" * 70)
    print("🌊 OceanEmbed: Subsurface Temperature Reconstruction Pipeline")
    print("=" * 70)

    # 1. Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 2. Setup Data Pipeline
    grid = OceanGrid(lat_min=0.0, lat_max=25.0, lon_min=40.0, lon_max=100.0, resolution=0.25)
    print(f"Spatial Grid initialized: {grid.H} lats x {grid.W} lons, 15 standard depths (0-1000m)")

    train_years = [2016, 2017, 2018, 2019, 2020]
    val_years = [2021]
    test_years = [2022]
    print(f"Temporal Split: Train {train_years} | Val {val_years} | Test {test_years}")

    train_loader, val_loader, test_loader, clim, _ = create_dataloaders(
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        batch_size=8,
        lag_days=5,
        days_step=sample_step,
        grid=grid,
        seed=42
    )

    # 3. Generate Independent Argo Validation Profiles (never seen in training)
    synth = PhysicalOceanSynthesizer(grid, seed=999)
    argo_profiles = synth.generate_argo_profiles(year=2022, day_of_year=180, num_floats=24)
    print(f"Generated {len(argo_profiles)} independent Argo float profiles for Year 2022 validation")

    # 4. Stage B: Self-Supervised MAE Pretraining
    mae_encoder = OceanMAEEncoder(
        in_channels=35,
        embed_dim=256,
        depth=6,
        num_heads=8,
        patch_size=16,
        img_size=(grid.H, grid.W),
        mask_ratio=0.75
    )
    pretrain_history = run_pretraining(
        mae_encoder,
        train_loader,
        epochs=pretrain_epochs,
        lr=3e-4,
        device=device,
        save_dir="checkpoints"
    )

    # 5. Stage C: Supervised Depth Decoder Fine-Tuning
    full_model = OceanEmbedFullModel(
        in_channels=35,
        embed_dim=256,
        encoder_depth=6,
        decoder_layers=4,
        num_heads=8,
        patch_size=16,
        img_size=(grid.H, grid.W)
    )
    # Transfer pretrained encoder weights
    full_model.encoder.load_state_dict(mae_encoder.state_dict())

    decoder_history = run_supervised_training(
        full_model,
        train_loader,
        val_loader,
        epochs=decoder_epochs,
        lr=1e-4,
        w_temp=1.0,
        w_thermo=0.2,
        w_stability=0.1,
        device=device,
        save_dir="checkpoints"
    )

    # 6. Train Direct-Regression Baseline for fair scientific comparison
    print("\n=== Training Direct Regression Baseline Model ===")
    direct_baseline = DirectRegressionBaseline(in_channels=35, num_depths=15, hidden_dim=64).to(device)
    opt_direct = torch.optim.AdamW(direct_baseline.parameters(), lr=2e-4)
    direct_baseline.train()
    for ep in range(max(3, decoder_epochs // 2)):
        for batch in train_loader:
            x_in = batch["surface_input"].to(device)
            y_tgt = batch["target_depth"].to(device)
            opt_direct.zero_grad()
            p_dep, _ = direct_baseline(x_in)
            loss = torch.nn.functional.mse_loss(p_dep, y_tgt)
            loss.backward()
            opt_direct.step()

    # 7. Independent Argo Float Benchmark Comparison
    print("\n=== Running Independent Argo Float Benchmarking ===")
    benchmark_results = run_full_benchmark(
        oceanembed_model=full_model,
        direct_baseline_model=direct_baseline,
        test_loader=test_loader,
        argo_profiles=argo_profiles,
        climatology=clim,
        grid=grid,
        device=device
    )

    print("\n📊 BENCHMARK RESULTS SUMMARY:")
    print(benchmark_results["summary_table"].to_string(index=False))

    # 8. Export Demo Cache
    if save_demo_cache:
        os.makedirs("demo/cache", exist_ok=True)
        # Extract a sample day for instant interactive Streamlit demo loading
        sample_surf, sample_depth, sample_thermo = synth.generate_day(2022, 200, add_eddy_field=True)
        sample_anom = clim.compute_surface_anomaly(sample_surf, 200)

        # Generate predictions
        full_model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(np.repeat(sample_anom[np.newaxis, ...], 5, axis=0)).unsqueeze(0).to(device)
            pred_anom, pred_thermo = full_model(x_tensor)
            pred_temp_abs = clim.reconstruct_absolute_temperature(pred_anom[0].cpu().numpy(), 200)
            pred_thermo_np = pred_thermo[0].cpu().numpy()

        cache_data = {
            "lats": grid.lats.tolist(),
            "lons": grid.lons.tolist(),
            "depth_levels": grid.depth_levels,
            "surface_channels": ["SST Anomaly", "SSS Anomaly", "SSH Anomaly", "U-Current", "V-Current", "Wind-U", "Wind-V"],
            "sample_surface": sample_surf.tolist(),
            "sample_surface_anom": sample_anom.tolist(),
            "glorys_subsurface": sample_depth.tolist(),
            "predicted_subsurface": pred_temp_abs.tolist(),
            "glorys_thermo": sample_thermo.tolist(),
            "predicted_thermo": pred_thermo_np.tolist(),
            "argo_profiles": argo_profiles,
            "benchmark_summary": benchmark_results["summary_table"].to_dict(orient="records"),
            "depth_metrics": benchmark_results["oceanembed"]["overall_df"].to_dict(orient="records")
        }
        import json
        with open("demo/cache/demo_cache.json", "w") as f:
            json.dump(cache_data, f)
        print("Exported interactive demo cache to demo/cache/demo_cache.json")

    return {
        "full_model": full_model,
        "direct_baseline": direct_baseline,
        "benchmark_results": benchmark_results,
        "climatology": clim,
        "grid": grid
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OceanEmbed Pipeline")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="Number of MAE pretrain epochs")
    parser.add_argument("--decoder-epochs", type=int, default=8, help="Number of Depth Decoder fine-tune epochs")
    parser.add_argument("--step", type=int, default=10, help="Days step for time series")
    args = parser.parse_args()

    run_full_oceanembed_pipeline(
        pretrain_epochs=args.pretrain_epochs,
        decoder_epochs=args.decoder_epochs,
        sample_step=args.step
    )
