"""Evaluation metrics computation for OceanEmbed against independent Argo float observations."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.data.grid import STANDARD_DEPTH_LEVELS
from src.models.physics_loss import verify_profile_monotonicity


def calculate_depth_metrics(
    pred_profiles: np.ndarray,
    actual_profiles: np.ndarray,
    depth_levels: Optional[List[float]] = None
) -> pd.DataFrame:
    """Calculate per-depth-level Pearson correlation, RMSE, bias, and sample count.

    Args:
        pred_profiles: Array of shape (N_samples, 15) in °C
        actual_profiles: Array of shape (N_samples, 15) in °C
        depth_levels: List of 15 depth values in meters

    Returns:
        pandas DataFrame with columns ['depth_m', 'correlation', 'rmse_degC', 'bias_degC', 'sample_count']
    """
    N, D = pred_profiles.shape
    depths = depth_levels or STANDARD_DEPTH_LEVELS

    records = []
    for d in range(D):
        y_true = actual_profiles[:, d]
        y_pred = pred_profiles[:, d]

        # Filter NaNs if any
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        n_valid = int(np.sum(valid))

        if n_valid < 2:
            records.append({
                "depth_m": depths[d],
                "correlation": np.nan,
                "rmse_degC": np.nan,
                "bias_degC": np.nan,
                "sample_count": n_valid
            })
            continue

        yt = y_true[valid]
        yp = y_pred[valid]

        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        bias = float(np.mean(yp - yt))

        # Pearson correlation
        std_t = float(np.std(yt))
        std_p = float(np.std(yp))
        if std_t > 1e-6 and std_p > 1e-6:
            r = float(np.corrcoef(yp, yt)[0, 1])
        else:
            r = 0.0

        records.append({
            "depth_m": depths[d],
            "correlation": round(r, 4),
            "rmse_degC": round(rmse, 4),
            "bias_degC": round(bias, 4),
            "sample_count": n_valid
        })

    return pd.DataFrame(records)


def evaluate_predictions_against_argo(
    matched_pairs: List[Dict[str, Union[np.ndarray, float, str]]]
) -> Dict[str, Union[pd.DataFrame, float, Dict[str, pd.DataFrame]]]:
    """Evaluate matched model-Argo pairs with overall and sub-basin breakdowns.

    Args:
        matched_pairs: List of dicts with 'pred_temperature', 'argo_temperature', 'basin'

    Returns:
        Dict containing:
          - 'overall_df': DataFrame with depth-wise metrics for all samples
          - 'regional_dfs': Dict mapping basin name -> DataFrame
          - 'total_samples': Total number of float profiles
          - 'physical_validity_pct': Percentage of monotonic profiles (target: 100%)
          - 'mean_upper_ocean_corr': Mean correlation across upper 500m (target: >0.7)
          - 'mean_upper_ocean_rmse': Mean RMSE across upper 500m
    """
    if not matched_pairs:
        raise ValueError("Cannot evaluate empty list of matched pairs.")

    all_preds = np.array([p["pred_temperature"] for p in matched_pairs])
    all_actuals = np.array([p["argo_temperature"] for p in matched_pairs])
    basins = [p.get("basin", "all") for p in matched_pairs]

    overall_df = calculate_depth_metrics(all_preds, all_actuals)

    # Sub-regional breakdown (Bay of Bengal vs Arabian Sea)
    regional_dfs = {}
    for basin_name in set(basins):
        idx = [i for i, b in enumerate(basins) if b == basin_name]
        if idx:
            regional_dfs[basin_name] = calculate_depth_metrics(all_preds[idx], all_actuals[idx])

    # Upper 500m summary metrics (depths <= 500m are first 13 levels)
    upper_mask = overall_df["depth_m"] <= 500.0
    mean_upper_corr = float(overall_df.loc[upper_mask, "correlation"].mean())
    mean_upper_rmse = float(overall_df.loc[upper_mask, "rmse_degC"].mean())
    physical_validity = verify_profile_monotonicity(all_preds)

    return {
        "overall_df": overall_df,
        "regional_dfs": regional_dfs,
        "total_samples": len(matched_pairs),
        "physical_validity_pct": round(physical_validity, 2),
        "mean_upper_ocean_corr": round(mean_upper_corr, 4),
        "mean_upper_ocean_rmse": round(mean_upper_rmse, 4)
    }
