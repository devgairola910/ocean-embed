"""Unit tests for satellite image ingestion and 15-level reconstruction."""

import numpy as np
from PIL import Image
import pytest

from src.data.image_ingestion import SatelliteImageAnalyzer, STANDARD_DEPTH_LEVELS_15


def test_standard_depth_levels_15():
    """Verify exact conformity to specified standard depth levels."""
    expected = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    assert STANDARD_DEPTH_LEVELS_15 == expected
    assert len(STANDARD_DEPTH_LEVELS_15) == 15


def test_satellite_image_ingestion():
    """Verify loading synthetic satellite raster image to 0.25 deg grid."""
    analyzer = SatelliteImageAnalyzer()

    # Create synthetic satellite RGB image
    img = Image.new("RGB", (300, 200), color=(240, 120, 40))
    result = analyzer.load_satellite_image(img, min_temp_c=22.0, max_temp_c=31.0)

    assert "sst_grid" in result
    assert "rgb_array" in result
    assert "sla_grid" in result
    assert result["sst_grid"].shape == (101, 241)
    assert 22.0 <= result["mean_sst"] <= 31.0


def test_subsurface_reconstruction_from_image():
    """Verify 0-1000m thermal profile generation from image extracted SST."""
    analyzer = SatelliteImageAnalyzer()
    profile_data = analyzer.reconstruct_subsurface_profile(sst=29.5, sla_cm=18.0, sss_psu=33.2)

    assert len(profile_data["predicted_temperatures_c"]) == 15
    assert profile_data["depth_levels_m"] == STANDARD_DEPTH_LEVELS_15
    assert profile_data["predicted_temperatures_c"][0] >= profile_data["predicted_temperatures_c"][-1]
    assert profile_data["physical_stability"] == "100% Monotonically Stable"
    assert 30.0 <= profile_data["thermocline_depth_m"] <= 120.0
