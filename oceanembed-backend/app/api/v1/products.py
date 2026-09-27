from fastapi import APIRouter, Query
from app.config import get_settings
from app.core.interpolation import (
    calculate_zarr_ohc,
    calculate_zarr_marine_heatwave,
    calculate_zarr_cyclone_risk,
)
from app.mock.generator import (
    calculate_ohc,
    calculate_marine_heatwave,
    calculate_cyclone_risk,
)
from app.models.predict import LocationPoint
from app.models.products import (
    OHCProductResponse,
    DepthRange,
    MarineHeatwaveResponse,
    CycloneRiskResponse,
)

router = APIRouter()


@router.get("/products/ohc", response_model=OHCProductResponse, summary="Get Ocean Heat Content (OHC)")
def get_ohc(
    lat: float = Query(..., description="Latitude °N", examples=[12.5]),
    lon: float = Query(..., description="Longitude °E", examples=[65.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
):
    settings = get_settings()
    if settings.MODE == "real":
        data = calculate_zarr_ohc(lat, lon, date)
    else:
        data = calculate_ohc(lat, lon, date)

    return OHCProductResponse(
        location=LocationPoint(lat=lat, lon=lon),
        date=date,
        ohc_value=data["ohc_value"],
        unit=data["unit"],
        ohc_joules_m2=data["ohc_joules_m2"],
        reference_temperature_c=data["reference_temperature_c"],
        depth_range_m=DepthRange(**data["depth_range_m"]),
        mode=settings.MODE,
    )


@router.get("/products/marine-heatwave", response_model=MarineHeatwaveResponse, summary="Get Marine Heatwave (MHW) assessment")
def get_marine_heatwave(
    lat: float = Query(..., description="Latitude °N", examples=[12.5]),
    lon: float = Query(..., description="Longitude °E", examples=[65.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
):
    settings = get_settings()
    if settings.MODE == "real":
        data = calculate_zarr_marine_heatwave(lat, lon, date)
    else:
        data = calculate_marine_heatwave(lat, lon, date)

    return MarineHeatwaveResponse(
        location=LocationPoint(lat=lat, lon=lon),
        date=date,
        is_heatwave=data["is_heatwave"],
        sst_c=data["sst_c"],
        climatological_threshold_c=data["climatological_threshold_c"],
        severity_category=data["severity_category"],
        mode=settings.MODE,
    )


@router.get("/products/cyclone-risk", response_model=CycloneRiskResponse, summary="Get Tropical Cyclone Heat Potential & Risk")
def get_cyclone_risk(
    lat: float = Query(..., description="Latitude °N", examples=[12.5]),
    lon: float = Query(..., description="Longitude °E", examples=[65.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
):
    settings = get_settings()
    if settings.MODE == "real":
        data = calculate_zarr_cyclone_risk(lat, lon, date)
    else:
        data = calculate_cyclone_risk(lat, lon, date)

    return CycloneRiskResponse(
        location=LocationPoint(lat=lat, lon=lon),
        date=date,
        tchp_score_kj_cm2=data["tchp_score_kj_cm2"],
        isotherm_26c_depth_m=data["isotherm_26c_depth_m"],
        risk_category=data["risk_category"],
        mode=settings.MODE,
    )
