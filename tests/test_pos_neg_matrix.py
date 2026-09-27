"""Comprehensive Positive and Negative Test Case Matrix for OceanEmbed.

Tests every system component in paired Positive (valid/expected) and Negative (edge-case/malformed/invalid) scenarios:
1. Spatial Grid & Coordinate Bounds
2. Climatology & Anomaly Inversion
3. Lagged Window Dataset & Edge Batches
4. ViT-MAE Masking & Tensor Dimensions
5. Depth-Conditioned Decoder & Query Extremes
6. Physics Stability Penalty & Inversion Gradients
7. Argo Float Matcher & Out-of-Bounds Handling
8. Backend REST API Endpoints (FastAPI)
9. Frontend Data Contract & Schema Compliance
"""

import os
import sys
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Core AI / ML Imports
from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS
from src.data.climatology import ClimatologyComputer
from src.data.synthetic import PhysicalOceanSynthesizer
from src.data.dataset import OceanEmbedDataset, create_dataloaders
from src.data.regrid import OceanDataRegridder
from src.models.mae_encoder import OceanMAEEncoder, PatchEmbedding2D
from src.models.depth_decoder import DepthConditionedDecoder, OceanEmbedFullModel
from src.models.physics_loss import OceanPhysicsLoss, compute_stability_penalty, verify_profile_monotonicity
from src.eval.argo_matcher import ArgoMatcher
from src.eval.metrics import calculate_depth_metrics, evaluate_predictions_against_argo
from demo.cached_data import DemoDataProvider, format_export_json_contract

# Backend API Imports
backend_path = os.path.abspath("oceanembed-backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
from app.main import app
from app.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.db.models import ArgoProfile, IngestionJob
from app.mock.generator import generate_profile_and_surface

SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def seed_test_database():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    # Clean prior records
    db.query(ArgoProfile).delete()
    test_data = [
        ("ARGO_TEST_001", 12.55, 65.05, "2025-05-15"),
        ("ARGO_TEST_002", 12.45, 64.95, "2025-05-15"),
        ("ARGO_TEST_003", 12.60, 65.10, "2025-05-15"),
        ("ARGO_TEST_004", 10.00, 70.00, "2025-05-15"),
        ("ARGO_TEST_005", 15.00, 85.00, "2024-06-01"),
    ]
    for fid, lat, lon, date_str in test_data:
        profile_data, _ = generate_profile_and_surface(lat, lon, date_str)
        argo_obj = ArgoProfile(
            id=fid,
            lat=lat,
            lon=lon,
            date=date_str,
            profile_data=profile_data,
            source="TEST_MOCK_SEED",
        )
        db.add(argo_obj)
    db.commit()
    db.close()


@pytest.fixture
def api_client():
    settings = get_settings()
    settings.MODE = "mock"
    settings.DATABASE_URL = SQLALCHEMY_TEST_DATABASE_URL
    seed_test_database()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# ====================================================================
# 1. SPATIAL GRID & COORDINATE RESOLUTION (Pos/Neg Pair)
# ====================================================================

def test_grid_positive_within_domain():
    """Positive: Valid coordinates in Bay of Bengal and Arabian Sea resolve to ocean pixels."""
    grid = OceanGrid()
    # Bay of Bengal point (15.0°N, 88.0°E)
    lat_i, lon_j = grid.find_nearest_indices(15.0, 88.0)
    assert grid.land_mask[lat_i, lon_j] is np.True_ or grid.land_mask[lat_i, lon_j] == 1
    # Arabian Sea point (14.0°N, 60.0°E)
    lat_i2, lon_j2 = grid.find_nearest_indices(14.0, 60.0)
    assert grid.land_mask[lat_i2, lon_j2] is np.True_ or grid.land_mask[lat_i2, lon_j2] == 1


def test_grid_negative_land_and_extreme_bounds():
    """Negative: Coordinates on Central Indian landmass correctly identified as land; extreme coordinates clamp safely."""
    grid = OceanGrid()
    # Central India land point (20.0°N, 78.0°E)
    lat_i, lon_j = grid.find_nearest_indices(20.0, 78.0)
    assert grid.land_mask[lat_i, lon_j] == 0 or grid.land_mask[lat_i, lon_j] is np.False_

    # Out-of-bounds coordinates clamp to nearest valid boundary index without crashing
    lat_out, lon_out = grid.find_nearest_indices(999.0, -999.0)
    assert 0 <= lat_out < grid.H
    assert 0 <= lon_out < grid.W


# ====================================================================
# 2. CLIMATOLOGY & ANOMALY TRANSFORMATIONS (Pos/Neg Pair)
# ====================================================================

def test_climatology_positive_cyclic_and_leap_day():
    """Positive: Climatology handles all 366 DOYs smoothly and accurately reconstructs anomalies."""
    grid = OceanGrid(lat_min=0.0, lat_max=5.0, lon_min=60.0, lon_max=70.0, resolution=1.0)
    synth = PhysicalOceanSynthesizer(grid, seed=101)

    surf_list, depth_list, doys = [], [], []
    for doy in [1, 100, 200, 300, 366]:
        s, dep, _ = synth.generate_day(2020, doy)
        surf_list.append(s)
        depth_list.append(dep)
        doys.append(doy)

    clim = ClimatologyComputer(num_days=366)
    clim.fit(np.stack(surf_list), np.stack(depth_list), np.array(doys))

    # Test exact reconstruction for DOY 366 (leap year boundary)
    anom = clim.compute_depth_anomaly(depth_list[-1], 366)
    reconstructed = clim.reconstruct_absolute_temperature(anom, 366)
    np.testing.assert_allclose(reconstructed, depth_list[-1], rtol=1e-4, atol=1e-4)


def test_climatology_negative_unfitted_or_out_of_range():
    """Negative: Handling of unfitted climatology and wrapping of arbitrary out-of-range DOY integers."""
    clim = ClimatologyComputer(num_days=366)
    assert clim.is_fitted is False

    # Fit small dummy series
    dummy_surf = np.ones((2, 7, 5, 5), dtype=np.float32)
    dummy_depth = np.ones((2, 15, 5, 5), dtype=np.float32) * 20.0
    clim.fit(dummy_surf, dummy_depth, np.array([1, 10]))

    # Negative: Out-of-range DOY (e.g. 500, -10) wraps cyclically without IndexError
    anom_wrapped = clim.compute_depth_anomaly(dummy_depth[0], doy=500)
    assert anom_wrapped.shape == (15, 5, 5)
    assert not np.isnan(anom_wrapped).any()


# ====================================================================
# 3. LAGGED WINDOW DATASET (Pos/Neg Pair)
# ====================================================================

def test_dataset_positive_lag_stacking():
    """Positive: Sliding window correctly stacks T_lag=5 days in causal sequence."""
    grid = OceanGrid(lat_min=0.0, lat_max=5.0, lon_min=60.0, lon_max=70.0, resolution=1.0)
    synth = PhysicalOceanSynthesizer(grid, seed=42)

    surf_list, depth_list, thermo_list, dates = [], [], [], []
    for d in range(1, 11):
        s, dep, th = synth.generate_day(2022, d)
        surf_list.append(s)
        depth_list.append(dep)
        thermo_list.append(th)
        dates.append((2022, d))

    clim = ClimatologyComputer()
    clim.fit(np.stack(surf_list), np.stack(depth_list), np.array([d for _, d in dates]))

    ds = OceanEmbedDataset(
        np.stack(surf_list), np.stack(depth_list), np.stack(thermo_list), dates, clim, lag_days=5, grid=grid
    )
    # Total samples for 10 days with lag=5 should be 10 - 5 + 1 = 6
    assert len(ds) == 6
    item = ds[0]
    assert item["surface_input"].shape == (5, 7, grid.H, grid.W)
    assert item["target_depth"].shape == (15, grid.H, grid.W)
    assert item["target_date"].tolist() == [2022, 5]  # Target date is the 5th day (last in window)


def test_dataset_negative_insufficient_length():
    """Negative: Dataset raises AssertionError if time series is shorter than lag_days."""
    grid = OceanGrid(lat_min=0.0, lat_max=5.0, lon_min=60.0, lon_max=70.0, resolution=1.0)
    synth = PhysicalOceanSynthesizer(grid)
    s, dep, th = synth.generate_day(2022, 1)

    clim = ClimatologyComputer()
    clim.fit(s[np.newaxis, ...], dep[np.newaxis, ...], np.array([1]))

    with pytest.raises(AssertionError):
        # 2 timesteps cannot satisfy lag_days=5
        OceanEmbedDataset(
            np.stack([s, s]), np.stack([dep, dep]), np.stack([th, th]),
            [(2022, 1), (2022, 2)], clim, lag_days=5, grid=grid
        )


# ====================================================================
# 4. ViT-MAE ENCODER & MASKING (Pos/Neg Pair)
# ====================================================================

def test_mae_encoder_positive_feature_extraction():
    """Positive: Encoder extracts (B, N_patches, D) spatial embeddings without masking during Stage C inference."""
    H, W = 32, 64
    model = OceanMAEEncoder(in_channels=35, embed_dim=64, depth=2, num_heads=4, patch_size=8, img_size=(H, W))

    x = torch.randn(2, 5, 7, H, W)
    latents = model.extract_features(x)
    # 32/8 * 64/8 = 4 * 8 = 32 patches
    assert latents.shape == (2, 32, 64)
    assert not torch.isnan(latents).any()


def test_mae_encoder_negative_extreme_mask_ratios():
    """Negative: Model behaves gracefully with 0.0 mask ratio (no mask) and high 0.90 mask ratio."""
    H, W = 32, 64
    model = OceanMAEEncoder(in_channels=35, embed_dim=64, depth=2, num_heads=4, patch_size=8, img_size=(H, W))
    x = torch.randn(2, 35, H, W)

    # 0.0 mask ratio (100% visible)
    latents_vis, mask_zero, _ = model.forward_encoder(x, mask_ratio=0.0)
    assert latents_vis.shape == (2, 32, 64)
    assert mask_zero.sum().item() == 0

    # 0.90 extreme mask ratio (only 10% visible = 3 patches)
    latents_sparse, mask_high, _ = model.forward_encoder(x, mask_ratio=0.90)
    assert latents_sparse.shape == (2, 3, 64)
    assert mask_high.sum().item() == (32 - 3) * 2


# ====================================================================
# 5. DEPTH DECODER & CONTINUOUS CONDITIONING (Pos/Neg Pair)
# ====================================================================

def test_depth_decoder_positive_15_levels():
    """Positive: Decoder maps spatial surface latents to 15 standard depth levels and thermocline depth map."""
    H, W = 32, 64
    decoder = DepthConditionedDecoder(embed_dim=64, num_depths=15, num_layers=2, num_heads=4, img_size=(H, W), patch_size=8)

    latents = torch.randn(2, 32, 64)
    pred_depth, pred_thermo = decoder(latents)

    assert pred_depth.shape == (2, 15, H, W)
    assert pred_thermo.shape == (2, H, W)
    # Thermocline depth in meters is non-negative
    assert torch.all(pred_thermo >= 0.0)


def test_depth_decoder_negative_custom_depth_configuration():
    """Negative: Decoder supports non-standard custom depth levels (e.g. 5 arbitrary deep layers) without error."""
    H, W = 32, 64
    custom_depths = [100.0, 500.0, 1000.0, 2000.0, 4000.0]
    decoder = DepthConditionedDecoder(
        embed_dim=64, num_depths=5, depth_levels=custom_depths, num_layers=2, num_heads=4, img_size=(H, W), patch_size=8
    )

    latents = torch.randn(2, 32, 64)
    pred_depth, pred_thermo = decoder(latents)
    assert pred_depth.shape == (2, 5, H, W)


# ====================================================================
# 6. PHYSICS LOSS & MONOTONICITY (Pos/Neg Pair)
# ====================================================================

def test_physics_loss_positive_stable_monotonic_profile():
    """Positive: Stable physical profile receives 0.0 stability penalty."""
    # Monotonically decreasing temperature from 29°C at surface to 4°C at 1000m
    stable_temps = torch.linspace(29.0, 4.0, 15).view(1, 15, 1, 1).repeat(2, 1, 8, 8)
    penalty = compute_stability_penalty(stable_temps, allowed_tolerance=0.05)
    assert penalty.item() == 0.0


def test_physics_loss_negative_severe_inversion_penalty_and_gradient():
    """Negative: Gravitationally unstable temperature profile incurs positive penalty with active corrective gradients."""
    # Completely reversed: surface 4°C, bottom 30°C
    inverted_temps = torch.linspace(4.0, 30.0, 15).view(1, 15, 1, 1).repeat(2, 1, 8, 8).requires_grad_(True)
    penalty = compute_stability_penalty(inverted_temps, allowed_tolerance=0.0)

    assert penalty.item() > 0.0
    penalty.backward()
    assert inverted_temps.grad is not None
    # Corrective gradient pushes deep layer colder and surface warmer
    assert inverted_temps.grad[0, -1, 0, 0].item() > 0.0
    assert inverted_temps.grad[0, 0, 0, 0].item() < 0.0


# ====================================================================
# 7. ARGO FLOAT MATCHER & VALIDATION (Pos/Neg Pair)
# ====================================================================

def test_argo_matcher_positive_ocean_point():
    """Positive: In-situ float in open ocean matches nearest grid point and interpolates depths."""
    grid = OceanGrid(lat_min=0.0, lat_max=10.0, lon_min=50.0, lon_max=70.0, resolution=0.5)
    matcher = ArgoMatcher(grid)

    pred_3d = np.ones((15, grid.H, grid.W), dtype=np.float32) * 18.0
    argo = {
        "float_id": "ARGO_6901234",
        "lat": 6.25,
        "lon": 62.25,
        "depths": [0, 50, 100, 500, 1000],
        "temperature": [28.0, 25.0, 20.0, 10.0, 4.5],
        "basin": "arabian_sea"
    }

    res = matcher.match_single_profile(pred_3d, argo)
    assert res is not None
    assert res["float_id"] == "ARGO_6901234"
    assert len(res["pred_temperature"]) == 15
    assert len(res["argo_temperature"]) == 15


def test_argo_matcher_negative_land_location():
    """Negative: Argo float coordinates erroneously placed on land return None safely."""
    grid = OceanGrid()
    matcher = ArgoMatcher(grid)
    pred_3d = np.ones((15, grid.H, grid.W), dtype=np.float32) * 18.0

    # Land coordinate on Indian subcontinent (18.0°N, 78.0°E)
    land_argo = {
        "float_id": "ARGO_LAND_ERR",
        "lat": 18.0,
        "lon": 78.0,
        "depths": grid.depth_levels,
        "temperature": np.ones(15).tolist(),
        "basin": "arabian_sea"
    }
    res = matcher.match_single_profile(pred_3d, land_argo)
    assert res is None


# ====================================================================
# 8. BACKEND REST API ENDPOINTS (Pos/Neg Matrix)
# ====================================================================

def test_api_health_positive(api_client):
    """Positive: /health returns 200 and operational status."""
    res = api_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_predict_positive_ocean_point(api_client):
    """Positive: /api/v1/predict returns 15 depth levels and surface inputs for valid ocean point."""
    res = api_client.get("/api/v1/predict", params={"lat": 12.5, "lon": 65.0, "date": "2025-05-15"})
    assert res.status_code == 200
    data = res.json()
    assert "profile" in data
    assert len(data["profile"]) == 15
    assert "requested_location" in data
    assert data["requested_location"]["lat"] == 12.5
    assert "surface_inputs_used" in data
    # Monotonic thermal stratification check
    assert data["profile"][0]["temperature_c"] > data["profile"][-1]["temperature_c"]


def test_api_predict_negative_land_point_400(api_client):
    """Negative: /api/v1/predict returns 400 Bad Request when query point falls on land."""
    res = api_client.get("/api/v1/predict", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "OUT_OF_DOMAIN"


def test_api_predict_negative_out_of_bounds_400(api_client):
    """Negative: /api/v1/predict returns 400 when coordinates exceed North Indian Ocean domain."""
    res = api_client.get("/api/v1/predict", params={"lat": 35.0, "lon": 80.0, "date": "2025-05-15"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "OUT_OF_DOMAIN"


def test_api_predict_negative_malformed_type_400(api_client):
    """Negative: /api/v1/predict returns 400 INVALID_PARAMETER for unparseable parameter types."""
    res = api_client.get("/api/v1/predict", params={"lat": "not_a_number", "lon": 65.0, "date": "2025-05-15"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] in ["INVALID_PARAMETER", "VALIDATION_ERROR"]


def test_api_grid_positive_and_negative_depth(api_client):
    """Pos/Neg: /api/v1/grid returns 200 for valid depth 0.0m, 400 for invalid depth level."""
    # Positive
    res_pos = api_client.get("/api/v1/grid", params={"date": "2025-05-15", "depth": 0.0})
    assert res_pos.status_code == 200
    assert "temperature_c" in res_pos.json()
    assert len(res_pos.json()["lats"]) > 0

    # Negative: Non-standard depth level
    res_neg = api_client.get("/api/v1/grid", params={"date": "2025-05-15", "depth": 12.3})
    assert res_neg.status_code == 400
    assert res_neg.json()["error"]["code"] == "INVALID_DEPTH"


def test_api_transect_positive_and_negative_bounds(api_client):
    """Pos/Neg: /api/v1/transect returns sampled points for valid ocean line, 400 for land/out-of-domain."""
    # Positive: Arabian Sea transect
    res_pos = api_client.get("/api/v1/transect", params={
        "lat1": 10.0, "lon1": 65.0, "lat2": 15.0, "lon2": 75.0, "date": "2025-05-15", "n_points": 10
    })
    assert res_pos.status_code == 200
    assert len(res_pos.json()["points"]) == 10

    # Negative: Land transect in central India
    res_neg = api_client.get("/api/v1/transect", params={
        "lat1": 22.0, "lon1": 78.0, "lat2": 23.0, "lon2": 79.0, "date": "2025-05-15", "n_points": 10
    })
    assert res_neg.status_code == 400
    assert res_neg.json()["error"]["code"] == "OUT_OF_DOMAIN"


def test_api_products_positive_and_negative(api_client):
    """Pos/Neg: /api/v1/products/ohc returns Ocean Heat Content for valid ocean location, 400 for land."""
    # Positive
    res_pos = api_client.get("/api/v1/products/ohc", params={"lat": 12.5, "lon": 65.0, "date": "2025-05-15"})
    assert res_pos.status_code == 200
    assert res_pos.json()["ohc_value"] > 0
    assert res_pos.json()["unit"] == "GJ/m^2"

    # Negative: Land location
    res_neg = api_client.get("/api/v1/products/ohc", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res_neg.status_code == 400
    assert res_neg.json()["error"]["code"] == "OUT_OF_DOMAIN"


def test_api_argo_positive_and_negative(api_client):
    """Pos/Neg: /api/v1/argo/profiles returns floats nearby from seeded DB, 400 for land location."""
    # Positive
    res_pos = api_client.get("/api/v1/argo/profiles", params={"lat": 12.5, "lon": 65.0, "radius_km": 100.0, "date": "2025-05-15"})
    assert res_pos.status_code == 200
    assert len(res_pos.json()["profiles"]) >= 1

    # Negative: Land location
    res_neg = api_client.get("/api/v1/argo/profiles", params={"lat": 20.0, "lon": 78.0, "date": "2025-05-15"})
    assert res_neg.status_code == 400
    assert res_neg.json()["error"]["code"] == "OUT_OF_DOMAIN"


# ====================================================================
# 9. FRONTEND STREAMLIT SCHEMA (Pos/Neg Pair)
# ====================================================================

def test_frontend_contract_positive_json_structure():
    """Positive: Export JSON contract perfectly satisfies design.md §7.3 schema."""
    contract = format_export_json_contract(
        date_str="2022-07-15",
        lat=15.0,
        lon=88.0,
        pred_profile=np.linspace(28, 4.5, 15),
        glorys_profile=np.linspace(28.2, 4.4, 15),
        argo_profile=np.linspace(28.1, 4.5, 15),
        thermocline_depth=65.2,
        correlation=0.88,
        rmse=0.45,
        bias=0.01
    )
    assert set(contract.keys()) == {"date", "lat", "lon", "predicted_profile", "glorys_profile", "argo_profile", "thermocline_depth_m", "skill"}
    assert set(contract["skill"].keys()) == {"correlation", "rmse", "bias"}


def test_frontend_contract_negative_missing_argo_resilience():
    """Negative: Export contract handles None in comparison float profile gracefully."""
    contract = format_export_json_contract(
        date_str="2022-07-15",
        lat=15.0,
        lon=88.0,
        pred_profile=np.linspace(28, 4.5, 15),
        glorys_profile=np.linspace(28.2, 4.4, 15),
        argo_profile=None,
        thermocline_depth=65.2,
        correlation=0.88,
        rmse=0.45,
        bias=0.01
    )
    assert contract["argo_profile"] is None
    assert contract["skill"]["correlation"] == 0.88
