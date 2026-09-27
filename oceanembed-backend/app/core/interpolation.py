import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

from app.config import get_settings
from app.core.constants import DEPTH_LEVELS_M
from app.core.errors import APIException
from app.mock.generator import is_land, parse_and_validate_date, validate_location, haversine_distance


def get_zarr_dataset(date_str: str) -> xr.Dataset:
    parse_and_validate_date(date_str)
    settings = get_settings()

    zarr_filename = f"oceanembed_{date_str}.zarr"
    zarr_path = os.path.join(settings.DATA_OUTPUT_PATH, zarr_filename)

    if not os.path.exists(zarr_path):
        default_zarr = os.path.join(settings.DATA_OUTPUT_PATH, "oceanembed_outputs.zarr")
        if os.path.exists(default_zarr):
            zarr_path = default_zarr
        else:
            raise APIException(
                status_code=503,
                code="DATA_NOT_READY",
                message=f"Precomputed ocean temperature data for date '{date_str}' is not ready yet.",
            )

    try:
        ds = xr.open_zarr(zarr_path)
        return ds
    except Exception as e:
        raise APIException(
            status_code=503,
            code="DATA_NOT_READY",
            message=f"Unable to read precomputed Zarr store for date '{date_str}': {str(e)}",
        )


def get_zarr_profile_and_surface(
    lat: float, lon: float, date_str: str
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    validate_location(lat, lon)
    ds = get_zarr_dataset(date_str)

    profile = []
    for z in DEPTH_LEVELS_M:
        var_name = f"temp_depth_{int(z)}m"
        if var_name in ds.data_vars:
            val = float(ds[var_name].interp(lat=lat, lon=lon, method="linear").values)
            if np.isnan(val):
                raise APIException(
                    status_code=400,
                    code="OUT_OF_DOMAIN",
                    message=f"Location ({lat}, {lon}) lands on land or unmapped cell.",
                )
            temp_c = round(val, 2)
        else:
            temp_c = 20.0

        unc_c = round(0.18 + 0.001 * z, 2)
        profile.append({
            "depth_m": float(z),
            "temperature_c": temp_c,
            "uncertainty_c": unc_c,
        })

    surface_inputs = {
        "sst": profile[0]["temperature_c"],
        "sss": 34.5,
        "sla": 0.02,
        "u": 0.10,
        "v": -0.05,
    }

    return profile, surface_inputs


def get_zarr_grid_2d(date_str: str, depth_m: float) -> Tuple[List[float], List[float], List[List[Optional[float]]]]:
    if depth_m not in DEPTH_LEVELS_M:
        raise APIException(
            status_code=400,
            code="INVALID_DEPTH",
            message=f"Depth {depth_m}m is invalid. Must be one of {DEPTH_LEVELS_M}.",
        )

    ds = get_zarr_dataset(date_str)
    var_name = f"temp_depth_{int(depth_m)}m"

    if var_name not in ds.data_vars:
        raise APIException(
            status_code=400,
            code="INVALID_DEPTH",
            message=f"Depth {depth_m}m is not available in dataset.",
        )

    lats = [round(float(lat), 2) for lat in ds.lat.values]
    lons = [round(float(lon), 2) for lon in ds.lon.values]
    arr_2d = ds[var_name].values

    grid_data = []
    for i, lat in enumerate(lats):
        row = []
        for j, lon in enumerate(lons):
            val = float(arr_2d[i, j])
            if np.isnan(val) or is_land(lat, lon):
                row.append(None)
            else:
                row.append(round(val, 2))
        grid_data.append(row)

    return lats, lons, grid_data


def get_zarr_transect(
    lat1: float, lon1: float, lat2: float, lon2: float, date_str: str, n_points: int
) -> List[Dict[str, Any]]:
    if is_land(lat1, lon1) and is_land(lat2, lon2):
        raise APIException(
            status_code=400,
            code="OUT_OF_DOMAIN",
            message="Both start and end points of transect are outside domain or on land.",
        )

    points = []
    cum_dist = 0.0
    prev_lat, prev_lon = lat1, lon1

    for i in range(n_points):
        t = i / max(1, n_points - 1)
        cur_lat = round(lat1 + t * (lat2 - lat1), 4)
        cur_lon = round(lon1 + t * (lon2 - lon1), 4)

        if i > 0:
            cum_dist += haversine_distance(prev_lat, prev_lon, cur_lat, cur_lon)
        prev_lat, prev_lon = cur_lat, cur_lon

        if is_land(cur_lat, cur_lon):
            points.append({
                "lat": cur_lat,
                "lon": cur_lon,
                "distance_km": round(cum_dist, 2),
                "is_ocean": False,
                "profile": None,
            })
        else:
            prof, _ = get_zarr_profile_and_surface(cur_lat, cur_lon, date_str)
            points.append({
                "lat": cur_lat,
                "lon": cur_lon,
                "distance_km": round(cum_dist, 2),
                "is_ocean": True,
                "profile": prof,
            })

    return points


def calculate_zarr_ohc(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    profile, _ = get_zarr_profile_and_surface(lat, lon, date_str)
    rho = 1025.0
    cp = 3990.0
    t_ref = 0.0

    ohc_joules = 0.0
    for i in range(len(profile) - 1):
        dz = profile[i + 1]["depth_m"] - profile[i]["depth_m"]
        avg_temp = (profile[i]["temperature_c"] + profile[i + 1]["temperature_c"]) / 2.0
        ohc_joules += rho * cp * (avg_temp - t_ref) * dz

    ohc_gj = round(ohc_joules / 1e9, 2)

    return {
        "ohc_value": ohc_gj,
        "unit": "GJ/m^2",
        "ohc_joules_m2": round(ohc_joules, 2),
        "reference_temperature_c": t_ref,
        "depth_range_m": {"min": 0.0, "max": 450.0},
    }


def calculate_zarr_marine_heatwave(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    _, surface = get_zarr_profile_and_surface(lat, lon, date_str)
    sst = surface["sst"]
    threshold = 29.5

    if sst > threshold:
        is_hw = True
        anomaly = sst - threshold
        if anomaly > 2.0:
            severity = "Extreme"
        elif anomaly > 1.0:
            severity = "Severe"
        elif anomaly > 0.5:
            severity = "Strong"
        else:
            severity = "Moderate"
    else:
        is_hw = False
        severity = "None"

    return {
        "is_heatwave": is_hw,
        "sst_c": sst,
        "climatological_threshold_c": threshold,
        "severity_category": severity,
    }


def calculate_zarr_cyclone_risk(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    profile, _ = get_zarr_profile_and_surface(lat, lon, date_str)
    rho = 1025.0
    cp = 3990.0

    d26 = 0.0
    for i in range(len(profile) - 1):
        z1, t1 = profile[i]["depth_m"], profile[i]["temperature_c"]
        z2, t2 = profile[i + 1]["depth_m"], profile[i + 1]["temperature_c"]
        if t1 >= 26.0 >= t2:
            if t1 == t2:
                d26 = z1
            else:
                d26 = z1 + (26.0 - t1) * (z2 - z1) / (t2 - t1)
            break
        elif t2 >= 26.0:
            d26 = z2

    tchp_joules = 0.0
    for i in range(len(profile) - 1):
        z1, t1 = profile[i]["depth_m"], profile[i]["temperature_c"]
        z2, t2 = profile[i + 1]["depth_m"], profile[i + 1]["temperature_c"]

        if z1 >= d26:
            break

        upper_z = z1
        lower_z = min(z2, d26)
        dz = lower_z - upper_z

        t_lower = t1 + (t2 - t1) * (lower_z - z1) / (z2 - z1) if z2 > z1 else t1
        avg_temp_above_26 = ((t1 - 26.0) + (t_lower - 26.0)) / 2.0

        if avg_temp_above_26 > 0:
            tchp_joules += rho * cp * avg_temp_above_26 * dz

    tchp_score = round(tchp_joules / 1e7, 1)

    if tchp_score < 40.0:
        risk_category = "Low"
    elif tchp_score <= 80.0:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    return {
        "tchp_score_kj_cm2": tchp_score,
        "isotherm_26c_depth_m": round(d26, 1),
        "risk_category": risk_category,
    }
