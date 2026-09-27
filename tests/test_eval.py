"""Unit tests for Argo matching, depth metrics, and benchmark evaluation."""

import pytest
import numpy as np
import pandas as pd
from src.data.grid import OceanGrid
from src.eval.argo_matcher import ArgoMatcher
from src.eval.metrics import calculate_depth_metrics, evaluate_predictions_against_argo


def test_argo_matcher_nearest_match():
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    matcher = ArgoMatcher(grid)

    pred_3d = np.full((15, grid.H, grid.W), 20.0, dtype=np.float32)
    argo_sample = {
        "float_id": "ARGO_TEST_01",
        "lat": 5.2,
        "lon": 65.1,
        "depths": grid.depth_levels,
        "temperature": [28.0, 27.5, 26.0, 24.0, 20.0, 16.0, 14.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0],
        "basin": "arabian_sea"
    }

    matched = matcher.match_single_profile(pred_3d, argo_sample)
    assert matched is not None
    assert matched["float_id"] == "ARGO_TEST_01"
    assert len(matched["pred_temperature"]) == 15
    assert len(matched["argo_temperature"]) == 15


def test_depth_metrics_calculation():
    # 10 samples across 15 depths
    np.random.seed(42)
    actuals = np.random.normal(15.0, 3.0, size=(10, 15))
    preds = actuals + np.random.normal(0.0, 0.5, size=(10, 15))

    df = calculate_depth_metrics(preds, actuals)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 15
    assert "correlation" in df.columns
    assert "rmse_degC" in df.columns
    assert "bias_degC" in df.columns
    assert "sample_count" in df.columns
    assert df["sample_count"].iloc[0] == 10
    assert df["correlation"].mean() > 0.8  # Strong correlation due to small noise


def test_evaluate_predictions_against_argo():
    matched_pairs = [
        {
            "pred_temperature": np.linspace(28, 4, 15),
            "argo_temperature": np.linspace(28, 4, 15) + np.random.normal(0, 0.2, 15),
            "basin": "bay_of_bengal"
        },
        {
            "pred_temperature": np.linspace(27, 4, 15),
            "argo_temperature": np.linspace(27, 4, 15) + np.random.normal(0, 0.2, 15),
            "basin": "arabian_sea"
        }
    ]

    results = evaluate_predictions_against_argo(matched_pairs)
    assert "overall_df" in results
    assert "regional_dfs" in results
    assert "bay_of_bengal" in results["regional_dfs"]
    assert "arabian_sea" in results["regional_dfs"]
    assert results["total_samples"] == 2
    assert results["physical_validity_pct"] == 100.0
