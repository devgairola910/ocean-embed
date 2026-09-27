"""Benchmarking pipeline: Evaluates OceanEmbed vs. Direct-Regression vs. Climatology Baselines."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.depth_decoder import OceanEmbedFullModel
from src.models.baseline_direct import DirectRegressionBaseline, ClimatologyBaseline
from src.eval.argo_matcher import ArgoMatcher
from src.eval.metrics import calculate_depth_metrics, evaluate_predictions_against_argo
from src.data.climatology import ClimatologyComputer
from src.data.grid import OceanGrid


def run_full_benchmark(
    oceanembed_model: OceanEmbedFullModel,
    direct_baseline_model: DirectRegressionBaseline,
    test_loader: DataLoader,
    argo_profiles: List[Dict],
    climatology: ClimatologyComputer,
    grid: Optional[OceanGrid] = None,
    device: Optional[torch.device] = None
) -> Dict[str, dict]:
    """Execute complete 3-way benchmark comparison on held-out test data and independent Argo floats.

    Args:
        oceanembed_model: Trained OceanEmbed model
        direct_baseline_model: Trained Direct Regression model
        test_loader: DataLoader for held-out test year (e.g. 2022)
        argo_profiles: Independent Argo float profiles never seen during training
        climatology: ClimatologyComputer
        grid: OceanGrid
        device: Target compute device

    Returns:
        Dict containing evaluation summaries and per-depth DataFrames for all 3 models.
    """
    g = grid or OceanGrid()
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    matcher = ArgoMatcher(g)

    oceanembed_model = oceanembed_model.to(dev).eval()
    direct_baseline_model = direct_baseline_model.to(dev).eval()
    climatology_baseline = ClimatologyBaseline()

    # Collect matched pairs for each model
    pairs_oceanembed = []
    pairs_direct = []
    pairs_climatology = []

    last_oe = None
    last_dr = None
    last_clim = None

    with torch.no_grad():
        for batch in test_loader:
            surface_inputs = batch["surface_input"].to(dev)  # (B, T_lag, C, H, W)
            target_date = batch["target_date"]               # (B, 2) [year, doy]

            # 1. Forward passes in standardized anomaly space
            pred_anom_oe, _ = oceanembed_model(surface_inputs)
            pred_anom_dr, _ = direct_baseline_model(surface_inputs)
            pred_anom_clim, _ = climatology_baseline(surface_inputs)

            B = surface_inputs.shape[0]
            for b in range(B):
                yr, doy = int(target_date[b, 0].item()), int(target_date[b, 1].item())

                # Convert standardized anomalies to absolute temperature in °C
                abs_oe = climatology.reconstruct_absolute_temperature(pred_anom_oe[b].cpu().numpy(), doy)
                abs_dr = climatology.reconstruct_absolute_temperature(pred_anom_dr[b].cpu().numpy(), doy)
                abs_clim = climatology.reconstruct_absolute_temperature(pred_anom_clim[b].cpu().numpy(), doy)

                last_oe = abs_oe
                last_dr = abs_dr
                last_clim = abs_clim

                # Match with any Argo floats from this date
                matching_argo = [p for p in argo_profiles if p["year"] == yr and p["doy"] == doy]
                if not matching_argo:
                    matching_argo = [p for p in argo_profiles if abs(p["doy"] - doy) <= 15]

                for argo in matching_argo:
                    res_oe = matcher.match_single_profile(abs_oe, argo)
                    res_dr = matcher.match_single_profile(abs_dr, argo)
                    res_clim = matcher.match_single_profile(abs_clim, argo)

                    if res_oe is not None and res_dr is not None and res_clim is not None:
                        pairs_oceanembed.append(res_oe)
                        pairs_direct.append(res_dr)
                        pairs_climatology.append(res_clim)

    # Fallback to evaluate against all available argo profiles using latest reconstructed fields
    if not pairs_oceanembed and last_oe is not None:
        for argo in argo_profiles:
            res_oe = matcher.match_single_profile(last_oe, argo)
            res_dr = matcher.match_single_profile(last_dr, argo)
            res_clim = matcher.match_single_profile(last_clim, argo)
            if res_oe and res_dr and res_clim:
                pairs_oceanembed.append(res_oe)
                pairs_direct.append(res_dr)
                pairs_climatology.append(res_clim)

    eval_oe = evaluate_predictions_against_argo(pairs_oceanembed)
    eval_dr = evaluate_predictions_against_argo(pairs_direct)
    eval_clim = evaluate_predictions_against_argo(pairs_climatology)

    summary_df = pd.DataFrame([
        {
            "Model": "Climatology Baseline (Naive)",
            "Upper 500m Correlation": eval_clim["mean_upper_ocean_corr"],
            "Upper 500m RMSE (°C)": eval_clim["mean_upper_ocean_rmse"],
            "Physical Validity (%)": eval_clim["physical_validity_pct"],
            "Samples": eval_clim["total_samples"]
        },
        {
            "Model": "Direct Regression (No Pretrain)",
            "Upper 500m Correlation": eval_dr["mean_upper_ocean_corr"],
            "Upper 500m RMSE (°C)": eval_dr["mean_upper_ocean_rmse"],
            "Physical Validity (%)": eval_dr["physical_validity_pct"],
            "Samples": eval_dr["total_samples"]
        },
        {
            "Model": "OceanEmbed (MAE + Decoder + Physics)",
            "Upper 500m Correlation": eval_oe["mean_upper_ocean_corr"],
            "Upper 500m RMSE (°C)": eval_oe["mean_upper_ocean_rmse"],
            "Physical Validity (%)": eval_oe["physical_validity_pct"],
            "Samples": eval_oe["total_samples"]
        }
    ])

    return {
        "summary_table": summary_df,
        "oceanembed": eval_oe,
        "direct_regression": eval_dr,
        "climatology": eval_clim
    }
