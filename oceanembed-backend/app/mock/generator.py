import hashlib
import math
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from app.core.constants import (
    GRID_LAT_MIN, GRID_LAT_MAX,
    GRID_LON_MIN, GRID_LON_MAX,
    GRID_RES, DEPTH_LEVELS_M,
    VALID_DATE_START, VALID_DATE_END,
)
from app.core.errors import APIException


def parse_and_validate_date(date_str: str) -> date:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise APIException(
            status_code=400,
            code="INVALID_PARAMETER",
            message=f"Date '{date_str}' is invalid. Expected format YYYY-MM-DD."
        )
    start_d = datetime.strptime(VALID_DATE_START, "%Y-%m-%d").date()
    end_d = datetime.strptime(VALID_DATE_END, "%Y-%m-%d").date()
    if d < start_d or d > end_d:
        raise APIException(
            status_code=400,
            code="OUT_OF_DOMAIN",
            message=f"Date '{date_str}' is outside valid range [{VALID_DATE_START} to {VALID_DATE_END}]."
        )
    return d


def is_land(lat: float, lon: float) -> bool:
    if lat < GRID_LAT_MIN or lat > GRID_LAT_MAX or lon < GRID_LON_MIN or lon > GRID_LON_MAX:
        return True

    # Indian Peninsula Mainland
    if 8.0 <= lat <= 23.0:
        west_coast = 77.5 - (lat - 8.0) * 0.63
        east_coast = 77.5 + (lat - 8.0) * 0.82
        if west_coast <= lon <= east_coast:
            return True

    # Northern India / Pakistan / Bangladesh (lat > 23°N)
    if lat > 23.0 and 65.0 <= lon <= 92.0:
        return True

    # Sri Lanka
    if 5.8 <= lat <= 9.8 and 79.5 <= lon <= 81.8:
        return True

    # Arabian Peninsula & Oman/Iran Coast
    if lat >= 22.0 and lon <= 60.0:
        return True
    if lat >= 12.0 and lon <= 55.0:
        return True

    # Horn of Africa
    if lat <= 12.0 and lon <= 51.0:
        return True

    # SE Asia / Myanmar / Thailand / Malay Peninsula
    if lat >= 10.0 and lon >= 98.0:
        return True
    if lat < 10.0 and lon >= 99.0:
        return True

    return False


def validate_location(lat: float, lon: float):
    if lat < GRID_LAT_MIN or lat > GRID_LAT_MAX or lon < GRID_LON_MIN or lon > GRID_LON_MAX:
        raise APIException(
            status_code=400,
            code="OUT_OF_DOMAIN",
            message=f"Location ({lat}, {lon}) is outside grid domain [{GRID_LAT_MIN}-{GRID_LAT_MAX}°N, {GRID_LON_MIN}-{GRID_LON_MAX}°E]."
        )
    if is_land(lat, lon):
        raise APIException(
            status_code=400,
            code="OUT_OF_DOMAIN",
            message=f"Location ({lat}, {lon}) is on land."
        )


def snap_to_grid(lat: float, lon: float) -> Tuple[float, float]:
    snap_lat = round(round(lat / GRID_RES) * GRID_RES, 4)
    snap_lon = round(round(lon / GRID_RES) * GRID_RES, 4)
    return snap_lat, snap_lon


def deterministic_hash_seeds(lat: float, lon: float, date_str: str) -> List[float]:
    snap_lat, snap_lon = snap_to_grid(lat, lon)
    key = f"{snap_lat:.2f}_{snap_lon:.2f}_{date_str}"
    md5 = hashlib.md5(key.encode()).hexdigest()
    seeds = []
    for i in range(0, 32, 4):
        val = int(md5[i:i+4], 16) / 65535.0
        seeds.append(val)
    return seeds


def generate_profile_and_surface(lat: float, lon: float, date_str: str) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    snap_lat, snap_lon = snap_to_grid(lat, lon)
    seeds = deterministic_hash_seeds(snap_lat, snap_lon, date_str)

    sst = round(27.5 + 2.0 * math.sin(snap_lat * 0.1 + snap_lon * 0.05) + (seeds[0] - 0.5) * 1.5, 2)
    sss = round(34.2 + (seeds[1] - 0.5) * 0.8, 2)
    sla = round((seeds[2] - 0.5) * 0.3, 3)
    u = round((seeds[3] - 0.5) * 0.4, 2)
    v = round((seeds[4] - 0.5) * 0.4, 2)

    surface_inputs = {
        "sst": sst,
        "sss": sss,
        "sla": sla,
        "u": u,
        "v": v,
    }

    t_deep = 7.5 + (seeds[5] - 0.5) * 0.5
    zm = 70.0 + (seeds[6] - 0.5) * 30.0
    scale = 45.0 + (seeds[7] - 0.5) * 10.0

    profile = []
    for z in DEPTH_LEVELS_M:
        t_z = t_deep + (sst - t_deep) / (1.0 + math.exp((z - zm) / scale))
        temp_c = round(t_z, 2)
        unc_c = round(0.18 + 0.001 * z + (seeds[7] - 0.5) * 0.05, 2)
        profile.append({
            "depth_m": float(z),
            "temperature_c": temp_c,
            "uncertainty_c": unc_c
        })

    return profile, surface_inputs


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def generate_grid_2d(date_str: str, depth_m: float) -> Tuple[List[float], List[float], List[List[Optional[float]]]]:
    parse_and_validate_date(date_str)
    if depth_m not in DEPTH_LEVELS_M:
        raise APIException(
            status_code=400,
            code="INVALID_DEPTH",
            message=f"Depth {depth_m}m is invalid. Must be one of {DEPTH_LEVELS_M}."
        )

    lats = [round(lat, 2) for lat in np.arange(GRID_LAT_MIN, GRID_LAT_MAX + 0.001, GRID_RES).tolist()]
    lons = [round(lon, 2) for lon in np.arange(GRID_LON_MIN, GRID_LON_MAX + 0.001, GRID_RES).tolist()]

    depth_idx = DEPTH_LEVELS_M.index(depth_m)

    grid = []
    for lat in lats:
        row = []
        for lon in lons:
            if is_land(lat, lon):
                row.append(None)
            else:
                prof, _ = generate_profile_and_surface(lat, lon, date_str)
                row.append(prof[depth_idx]["temperature_c"])
        grid.append(row)

    return lats, lons, grid


def generate_transect(
    lat1: float, lon1: float, lat2: float, lon2: float, date_str: str, n_points: int
) -> List[Dict[str, Any]]:
    parse_and_validate_date(date_str)
    if is_land(lat1, lon1) and is_land(lat2, lon2):
        raise APIException(
            status_code=400,
            code="OUT_OF_DOMAIN",
            message="Both start and end points of transect are outside domain or on land."
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
                "profile": None
            })
        else:
            prof, _ = generate_profile_and_surface(cur_lat, cur_lon, date_str)
            points.append({
                "lat": cur_lat,
                "lon": cur_lon,
                "distance_km": round(cum_dist, 2),
                "is_ocean": True,
                "profile": prof
            })

    return points


def calculate_ohc(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    parse_and_validate_date(date_str)
    validate_location(lat, lon)
    profile, _ = generate_profile_and_surface(lat, lon, date_str)

    # Formula:
    # OHC = \rho * c_p * \int_{0}^{z_max} (T(z) - T_ref) dz
    # \rho = 1025 kg/m^3 (seawater density)
    # c_p = 3990 J/(kg * K) (specific heat capacity of seawater)
    # T_ref = 0.0 °C (reference temperature for total upper ocean heat content)
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
        "depth_range_m": {"min": 0.0, "max": 450.0}
    }


def calculate_marine_heatwave(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    parse_and_validate_date(date_str)
    validate_location(lat, lon)
    _, surface = generate_profile_and_surface(lat, lon, date_str)

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
        "severity_category": severity
    }


def calculate_cyclone_risk(lat: float, lon: float, date_str: str) -> Dict[str, Any]:
    parse_and_validate_date(date_str)
    validate_location(lat, lon)
    profile, _ = generate_profile_and_surface(lat, lon, date_str)

    # Tropical Cyclone Heat Potential (TCHP) in kJ/cm^2
    # TCHP = \rho * c_p * \int_{0}^{D26} (T(z) - 26.0) dz
    # 1 kJ/cm^2 = 10^7 J/m^2
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
        "risk_category": risk_category
    }


def generate_argo_validation(date_from: str, date_to: str) -> Dict[str, Any]:
    parse_and_validate_date(date_from)
    parse_and_validate_date(date_to)

    by_depth = []
    for z in DEPTH_LEVELS_M:
        rmse = round(0.25 + 0.001 * z, 2)
        corr = round(0.98 - 0.00015 * z, 2)
        by_depth.append({
            "depth_m": float(z),
            "rmse_c": rmse,
            "correlation": corr
        })

    return {
        "date_from": date_from,
        "date_to": date_to,
        "n_matchups": 1450,
        "overall": {"rmse_c": 0.42, "correlation": 0.96},
        "by_depth": by_depth
    }


def generate_argo_profiles(lat: float, lon: float, radius_km: float, date_str: str) -> List[Dict[str, Any]]:
    parse_and_validate_date(date_str)
    validate_location(lat, lon)

    float_ids = ["ARGO_2901482", "ARGO_2901483", "ARGO_2901484"]
    offsets = [(0.15, -0.10), (-0.20, 0.25), (0.05, 0.30)]

    results = []
    for fid, (dlat, dlon) in zip(float_ids, offsets):
        f_lat = round(lat + dlat, 4)
        f_lon = round(lon + dlon, 4)
        if not is_land(f_lat, f_lon):
            dist = round(haversine_distance(lat, lon, f_lat, f_lon), 1)
            if dist <= radius_km:
                prof, _ = generate_profile_and_surface(f_lat, f_lon, date_str)
                results.append({
                    "float_id": fid,
                    "lat": f_lat,
                    "lon": f_lon,
                    "date": date_str,
                    "distance_km": dist,
                    "profile": prof
                })

    return results
