from fastapi import APIRouter, Query
from app.config import get_settings
from app.core.interpolation import get_zarr_profile_and_surface
from app.mock.generator import (
    parse_and_validate_date,
    validate_location,
    snap_to_grid,
    generate_profile_and_surface,
)
from app.models.predict import PredictResponse, LocationPoint, ProfilePoint, SurfaceInputs

router = APIRouter()


@router.get("/predict", response_model=PredictResponse, summary="Predict vertical temperature profile")
def predict(
    lat: float = Query(..., description="Latitude in °N (0 to 25)", examples=[12.5]),
    lon: float = Query(..., description="Longitude in °E (45 to 100)", examples=[65.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
):
    settings = get_settings()
    snap_lat, snap_lon = snap_to_grid(lat, lon)

    if settings.MODE == "real":
        profile, surface_inputs = get_zarr_profile_and_surface(lat, lon, date)
    else:
        parse_and_validate_date(date)
        validate_location(lat, lon)
        profile, surface_inputs = generate_profile_and_surface(lat, lon, date)

    return PredictResponse(
        requested_location=LocationPoint(lat=lat, lon=lon),
        nearest_grid_point=LocationPoint(lat=snap_lat, lon=snap_lon),
        date=date,
        profile=[ProfilePoint(**p) for p in profile],
        surface_inputs_used=SurfaceInputs(**surface_inputs),
        mode=settings.MODE,
    )
