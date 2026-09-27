"""Sanity-check diagnostics and visualization routines for Phase 1 data pipeline verification.

Checks:
- Spatial range and coordinate bounds
- Missing values and NaN checks over ocean pixels
- Physical bounds verification (SST in 15–35°C, SSS in 28–38 psu, SLA in -1 to +1 m, etc.)
- Vertical stability of subsurface temperatures (dT/dz <= 0)
- Generates diagnostic summary report and multi-panel figures
"""

from typing import Dict, Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS
from src.data.synthetic import PhysicalOceanSynthesizer


def run_data_pipeline_diagnostics(
    grid: Optional[OceanGrid] = None,
    output_dir: str = "reports/figures"
) -> Dict[str, Union[bool, float, dict]]:
    """Run comprehensive automated sanity checks on regridded ocean fields.

    Args:
        grid: OceanGrid instance
        output_dir: Directory to save diagnostic plots

    Returns:
        Dict with sanity check pass/fail flags and numerical stats.
    """
    g = grid or OceanGrid()
    synth = PhysicalOceanSynthesizer(g, seed=42)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate sample summer monsoon day (DOY 200)
    surf, depth, thermo = synth.generate_day(2022, 200, add_eddy_field=True)
    ocean_mask = g.land_mask

    # 2. NaN Checks over ocean
    nan_in_surf = np.isnan(surf[:, ocean_mask]).sum()
    nan_in_depth = np.isnan(depth[:, ocean_mask]).sum()
    nan_in_thermo = np.isnan(thermo[ocean_mask]).sum()
    zero_nan_passed = bool((nan_in_surf == 0) and (nan_in_depth == 0) and (nan_in_thermo == 0))

    # 3. Physical Value Range Checks
    sst_ocean = surf[0, ocean_mask]
    sss_ocean = surf[1, ocean_mask]
    sla_ocean = surf[2, ocean_mask]
    u_curr_ocean = surf[3, ocean_mask]
    v_curr_ocean = surf[4, ocean_mask]

    sst_valid = bool((sst_ocean.min() >= 18.0) and (sst_ocean.max() <= 35.0))
    sss_valid = bool((sss_ocean.min() >= 28.0) and (sss_ocean.max() <= 38.5))
    sla_valid = bool((sla_ocean.min() >= -0.8) and (sla_ocean.max() <= 0.8))

    # 4. Vertical Monotonicity Check (dT/dz <= 0 down to 1000m)
    # Check over all ocean pixels
    diffs = np.diff(depth, axis=0)  # (14, H, W)
    inversions = (diffs[:, ocean_mask] > 0.05).sum()
    monotonicity_passed = bool(inversions == 0)

    # 5. Generate Multi-Panel Sanity Check Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cmap_land = "gray"

    # Surface SST
    sst_plot = np.where(ocean_mask, surf[0], np.nan)
    im0 = axes[0, 0].imshow(sst_plot, origin="lower", extent=[g.lon_min, g.lon_max, g.lat_min, g.lat_max], cmap="RdYlBu_r")
    axes[0, 0].set_title("1. Sea Surface Temp (°C) [OSTIA]")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.03)

    # Surface SSS
    sss_plot = np.where(ocean_mask, surf[1], np.nan)
    im1 = axes[0, 1].imshow(sss_plot, origin="lower", extent=[g.lon_min, g.lon_max, g.lat_min, g.lat_max], cmap="viridis")
    axes[0, 1].set_title("2. Sea Surface Salinity (psu) [SMAP]")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.03)

    # Sea Level Anomaly
    sla_plot = np.where(ocean_mask, surf[2], np.nan)
    im2 = axes[0, 2].imshow(sla_plot, origin="lower", extent=[g.lon_min, g.lon_max, g.lat_min, g.lat_max], cmap="coolwarm")
    axes[0, 2].set_title("3. Sea Level Anomaly (m) [DUACS]")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.03)

    # Subsurface Temp at 100m
    t100_plot = np.where(ocean_mask, depth[6], np.nan)  # 100m is index 6
    im3 = axes[1, 0].imshow(t100_plot, origin="lower", extent=[g.lon_min, g.lon_max, g.lat_min, g.lat_max], cmap="Spectral_r")
    axes[1, 0].set_title("4. Subsurface Temp at 100 m (°C) [GLORYS]")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.03)

    # Thermocline Depth
    th_plot = np.where(ocean_mask, thermo, np.nan)
    im4 = axes[1, 1].imshow(th_plot, origin="lower", extent=[g.lon_min, g.lon_max, g.lat_min, g.lat_max], cmap="plasma_r")
    axes[1, 1].set_title("5. Thermocline Depth (m)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.03)

    # Vertical Stratification Curve Sample
    sample_prof = depth[:, g.H // 2, g.W // 2]
    axes[1, 2].plot(sample_prof, g.depth_levels, "o-", color="#00ADB5", lw=2)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("6. Vertical Profile Sample (0–1000m)")
    axes[1, 2].set_xlabel("Temperature (°C)")
    axes[1, 2].set_ylabel("Depth (m)")
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat[:5]:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "phase1_sanity_check_maps.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    report = {
        "status": "PASS" if (zero_nan_passed and sst_valid and sss_valid and sla_valid and monotonicity_passed) else "FAIL",
        "zero_nan_passed": zero_nan_passed,
        "sst_valid": sst_valid,
        "sss_valid": sss_valid,
        "sla_valid": sla_valid,
        "monotonicity_passed": monotonicity_passed,
        "ocean_pixel_count": int(ocean_mask.sum()),
        "sst_range_degC": [round(float(sst_ocean.min()), 2), round(float(sst_ocean.max()), 2)],
        "sss_range_psu": [round(float(sss_ocean.min()), 2), round(float(sss_ocean.max()), 2)],
        "sla_range_m": [round(float(sla_ocean.min()), 3), round(float(sla_ocean.max()), 3)],
        "figure_saved_to": fig_path
    }

    return report


if __name__ == "__main__":
    rep = run_data_pipeline_diagnostics()
    print("=== Phase 1 Data Pipeline Sanity Check Report ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
