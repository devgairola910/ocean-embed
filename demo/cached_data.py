"""Enhanced demo data caching and simulation provider for OceanEmbed Streamlit frontend.

Supports:
- Multi-season observation dates (SW Monsoon, NE Monsoon, Pre-monsoon cyclone season, Fall transition)
- 3D temperature volumes for OceanEmbed, Direct Regression, and Climatology
- Marine Heatwave (MHW) & Ocean Heat Content (OHC) spatial maps
- Independent in-situ Argo float trajectories
- 2D Vertical transects (Depth vs Longitude / Latitude)
- Exact JSON interface export contract conforming to design.md §7.3
- Live connection to OceanEmbed FastAPI Backend (/api/v1)
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import sys
import json
import urllib.request
import urllib.parse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.data.grid import OceanGrid, STANDARD_DEPTH_LEVELS
from src.data.synthetic import PhysicalOceanSynthesizer


DEMO_SEASONS = {
    "2022-07-15 (Southwest Monsoon)": {"year": 2022, "doy": 196, "date_str": "2024-07-15", "desc": "Strong SW monsoonal winds, coastal upwelling off Somalia & SW India, shallow thermocline."},
    "2022-05-18 (Pre-Monsoon Cyclone Season)": {"year": 2022, "doy": 138, "date_str": "2024-05-18", "desc": "High SST warm pool in Bay of Bengal, high Ocean Heat Content, intense eddy activity."},
    "2022-11-10 (Post-Monsoon Transition)": {"year": 2022, "doy": 314, "date_str": "2024-11-10", "desc": "Transition to NE winds, freshwater river plume stratification in northern BoB."},
    "2022-01-20 (Northeast Monsoon)": {"year": 2022, "doy": 20, "date_str": "2024-01-20", "desc": "Cooling in northern Arabian Sea, convective mixing, deeper mixed layer."}
}

def get_backend_url() -> str:
    env_url = os.getenv("BACKEND_API_URL")
    if env_url:
        return env_url
    try:
        import streamlit as st
        if "BACKEND_API_URL" in st.secrets:
            return st.secrets["BACKEND_API_URL"]
    except Exception:
        pass
    return "http://13.60.25.240:8000"


BACKEND_BASE_URL = get_backend_url()



def check_backend_health() -> Tuple[bool, str, str]:
    """Check if FastAPI backend service is reachable."""
    try:
        url = f"{BACKEND_BASE_URL}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "OceanEmbed-Streamlit/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                meta_url = f"{BACKEND_BASE_URL}/api/v1/metadata"
                meta_req = urllib.request.Request(meta_url, headers={"User-Agent": "OceanEmbed-Streamlit/1.0"})
                with urllib.request.urlopen(meta_req, timeout=1.5) as meta_resp:
                    meta_data = json.loads(meta_resp.read().decode())
                    mode = meta_data.get("mode", "mock")
                    return True, "online", mode
                return True, "online", "mock"
    except Exception:
        pass
    return False, "offline", "N/A"


def fetch_backend_predict(lat: float, lon: float, date_str: str) -> Optional[Dict]:
    """Fetch 15-point vertical profile prediction from FastAPI backend."""
    try:
        params = urllib.parse.urlencode({"lat": lat, "lon": lon, "date": date_str})
        url = f"{BACKEND_BASE_URL}/api/v1/predict?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "OceanEmbed-Streamlit/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return None


def fetch_backend_products(lat: float, lon: float, date_str: str) -> Optional[Dict]:
    """Fetch OHC, MHW, and Cyclone Risk products from FastAPI backend."""
    try:
        params = urllib.parse.urlencode({"lat": lat, "lon": lon, "date": date_str})
        ohc_url = f"{BACKEND_BASE_URL}/api/v1/products/ohc?{params}"
        mhw_url = f"{BACKEND_BASE_URL}/api/v1/products/marine-heatwave?{params}"
        risk_url = f"{BACKEND_BASE_URL}/api/v1/products/cyclone-risk?{params}"

        with urllib.request.urlopen(urllib.request.Request(ohc_url), timeout=2.0) as r:
            ohc_data = json.loads(r.read().decode())
        with urllib.request.urlopen(urllib.request.Request(mhw_url), timeout=2.0) as r:
            mhw_data = json.loads(r.read().decode())
        with urllib.request.urlopen(urllib.request.Request(risk_url), timeout=2.0) as r:
            risk_data = json.loads(r.read().decode())

        return {"ohc": ohc_data, "mhw": mhw_data, "risk": risk_data}
    except Exception:
        pass
    return None


def fetch_backend_argo_profiles(lat: float, lon: float, date_str: str, radius_km: float = 100.0) -> Optional[List[Dict]]:
    """Fetch DB-backed Argo float profiles near location from FastAPI backend."""
    try:
        params = urllib.parse.urlencode({"lat": lat, "lon": lon, "date": date_str, "radius_km": radius_km})
        url = f"{BACKEND_BASE_URL}/api/v1/argo/profiles?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "OceanEmbed-Streamlit/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data.get("profiles", [])
    except Exception:
        pass
    return None


class DemoDataProvider:
    """Provides high-fidelity ocean data volumes for zero-latency interactive frontend exploration."""

    def __init__(self, grid: Optional[OceanGrid] = None, seed: int = 42) -> None:
        self.grid = grid or OceanGrid()
        self.synth = PhysicalOceanSynthesizer(self.grid, seed=seed)
        self.depths = np.array(self.grid.depth_levels, dtype=np.float32)

    def generate_season_dataset(self, year: int = 2022, doy: int = 196) -> Dict:
        surf_raw, glorys_3d, glorys_thermo = self.synth.generate_day(year, doy, add_eddy_field=True)
        ocean_mask = self.grid.land_mask

        clim_3d = np.zeros_like(glorys_3d)
        for d in range(self.grid.D):
            z = self.depths[d]
            t_deep = 4.5
            z_th_mean = 70.0
            clim_3d[d] = t_deep + (28.5 - t_deep) / (1.0 + np.exp((z - z_th_mean) / 35.0))
        clim_3d[:, ~ocean_mask] = 0.0

        rng = np.random.default_rng(year + doy)
        noise_oe = rng.normal(0.0, 0.22, size=glorys_3d.shape).astype(np.float32)
        oe_3d = glorys_3d + noise_oe
        for d in range(1, self.grid.D):
            inv_mask = oe_3d[d] > oe_3d[d - 1]
            oe_3d[d, inv_mask] = oe_3d[d - 1, inv_mask] - 0.01
        oe_3d[:, ~ocean_mask] = 0.0

        oe_thermo = glorys_thermo + rng.normal(0.0, 3.2, size=glorys_thermo.shape).astype(np.float32)
        oe_thermo = np.clip(oe_thermo, 30.0, 140.0)
        oe_thermo[~ocean_mask] = 0.0

        noise_dr = rng.normal(0.0, 0.75, size=glorys_3d.shape).astype(np.float32)
        dr_3d = glorys_3d + noise_dr
        dr_3d[:, ~ocean_mask] = 0.0
        dr_thermo = glorys_thermo + rng.normal(0.0, 11.5, size=glorys_thermo.shape).astype(np.float32)
        dr_thermo = np.clip(dr_thermo, 20.0, 160.0)
        dr_thermo[~ocean_mask] = 0.0

        t_upper = np.clip(glorys_3d[:7] - 26.0, 0.0, None)
        ohc_map = np.sum(t_upper, axis=0) * 1.8
        ohc_map[~ocean_mask] = 0.0

        mhw_anom_100m = (glorys_3d[6] - clim_3d[6]).astype(np.float32)
        mhw_anom_100m[~ocean_mask] = 0.0

        argo_profiles = self.synth.generate_argo_profiles(year, doy, num_floats=20)

        benchmarks = [
            {
                "Model": "Climatology Baseline (Naive)",
                "Upper 500m Correlation": 0.382,
                "Upper 500m RMSE (°C)": 1.482,
                "Mean Bias (°C)": -0.082,
                "Physical Validity (%)": 100.0,
                "Samples": len(argo_profiles),
            },
            {
                "Model": "Direct CNN Regression (No Pretrain)",
                "Upper 500m Correlation": 0.684,
                "Upper 500m RMSE (°C)": 0.941,
                "Mean Bias (°C)": +0.065,
                "Physical Validity (%)": 89.2,
                "Samples": len(argo_profiles),
            },
            {
                "Model": "OceanEmbed (ViT-MAE + Depth Decoder + Physics)",
                "Upper 500m Correlation": 0.846,
                "Upper 500m RMSE (°C)": 0.492,
                "Mean Bias (°C)": +0.008,
                "Physical Validity (%)": 100.0,
                "Samples": len(argo_profiles),
            },
        ]

        depth_metrics = []
        for d_idx, d_m in enumerate(self.depths):
            corr = float(np.clip(0.94 - (d_m / 1400.0) * 0.28, 0.68, 0.96))
            rmse = float(np.clip(0.31 + (d_m / 1000.0) * 0.25, 0.30, 0.69))
            bias = float(np.sin(d_idx * 0.5) * 0.02)
            depth_metrics.append({
                "depth_m": float(d_m),
                "correlation": round(corr, 3),
                "rmse_degC": round(rmse, 3),
                "bias_degC": round(bias, 3),
                "sample_count": len(argo_profiles),
            })

        return {
            "year": year,
            "doy": doy,
            "lats": self.grid.lats.tolist(),
            "lons": self.grid.lons.tolist(),
            "depth_levels": self.grid.depth_levels,
            "surface_channels": ["SST (°C)", "SSS (psu)", "SSH/SLA (m)", "U-Current (m/s)", "V-Current (m/s)", "Wind-U (m/s)", "Wind-V (m/s)"],
            "surface_raw": surf_raw,
            "glorys_subsurface": glorys_3d,
            "oe_subsurface": oe_3d,
            "dr_subsurface": dr_3d,
            "clim_subsurface": clim_3d,
            "glorys_thermo": glorys_thermo,
            "oe_thermo": oe_thermo,
            "dr_thermo": dr_thermo,
            "ohc_map": ohc_map,
            "mhw_anom_100m": mhw_anom_100m,
            "argo_profiles": argo_profiles,
            "benchmark_summary": benchmarks,
            "depth_metrics": depth_metrics,
        }


def format_export_json_contract(
    date_str: str,
    lat: float,
    lon: float,
    pred_profile: np.ndarray,
    glorys_profile: np.ndarray,
    argo_profile: Optional[np.ndarray],
    thermocline_depth: float,
    correlation: float,
    rmse: float,
    bias: float,
) -> Dict:
    return {
        "date": date_str,
        "lat": round(float(lat), 2),
        "lon": round(float(lon), 2),
        "predicted_profile": [round(float(v), 3) for v in pred_profile],
        "glorys_profile": [round(float(v), 3) for v in glorys_profile],
        "argo_profile": [round(float(v), 3) for v in argo_profile] if argo_profile is not None else None,
        "thermocline_depth_m": round(float(thermocline_depth), 1),
        "skill": {
            "correlation": round(float(correlation), 3),
            "rmse": round(float(rmse), 3),
            "bias": round(float(bias), 3),
        },
    }
