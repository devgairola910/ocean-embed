"""Unit tests for frontend data provider, transects, and design.md §7.3 JSON interface contract."""

import pytest
import numpy as np
from demo.cached_data import DemoDataProvider, DEMO_SEASONS, format_export_json_contract
from src.data.grid import OceanGrid


def test_frontend_data_provider_seasons():
    """Verify demo data provider generates valid 2D and 3D data across all monsoonal regimes."""
    provider = DemoDataProvider()
    for season_name, s_info in DEMO_SEASONS.items():
        data = provider.generate_season_dataset(s_info["year"], s_info["doy"])
        assert "surface_raw" in data
        assert "glorys_subsurface" in data
        assert "oe_subsurface" in data
        assert "ohc_map" in data
        assert "mhw_anom_100m" in data
        assert "argo_profiles" in data

        # Shape verifications
        assert data["surface_raw"].shape == (7, 101, 241)
        assert data["glorys_subsurface"].shape == (15, 101, 241)
        assert data["oe_subsurface"].shape == (15, 101, 241)
        assert data["ohc_map"].shape == (101, 241)
        assert len(data["argo_profiles"]) >= 10


def test_design_md_json_contract_schema():
    """Verify exact conformity to design.md §7.3 JSON interface contract."""
    pred_prof = np.linspace(28.0, 4.5, 15)
    glorys_prof = np.linspace(28.2, 4.4, 15)
    argo_prof = np.linspace(28.1, 4.5, 15)

    contract = format_export_json_contract(
        date_str="2022-07-15",
        lat=15.5,
        lon=88.25,
        pred_profile=pred_prof,
        glorys_profile=glorys_prof,
        argo_profile=argo_prof,
        thermocline_depth=62.3,
        correlation=0.81,
        rmse=0.64,
        bias=-0.05
    )

    # Check keys required by design.md §7.3
    assert "date" in contract
    assert "lat" in contract
    assert "lon" in contract
    assert "predicted_profile" in contract
    assert "glorys_profile" in contract
    assert "argo_profile" in contract
    assert "thermocline_depth_m" in contract
    assert "skill" in contract
    assert "correlation" in contract["skill"]
    assert "rmse" in contract["skill"]
    assert "bias" in contract["skill"]

    assert len(contract["predicted_profile"]) == 15
    assert len(contract["glorys_profile"]) == 15
    assert len(contract["argo_profile"]) == 15
    assert contract["lat"] == 15.5
    assert contract["lon"] == 88.25
