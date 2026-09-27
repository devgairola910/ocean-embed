from fastapi import APIRouter, Query
from app.config import get_settings
from app.core.interpolation import get_zarr_transect
from app.mock.generator import generate_transect
from app.models.predict import LocationPoint, ProfilePoint
from app.models.transect import TransectResponse, TransectPoint

router = APIRouter()


@router.get("/transect", response_model=TransectResponse, summary="Sample ocean transect profile along a line")
def get_transect(
    lat1: float = Query(..., description="Start latitude °N", examples=[10.0]),
    lon1: float = Query(..., description="Start longitude °E", examples=[65.0]),
    lat2: float = Query(..., description="End latitude °N", examples=[15.0]),
    lon2: float = Query(..., description="End longitude °E", examples=[75.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
    n_points: int = Query(50, ge=2, le=200, description="Number of sampling points", examples=[50]),
):
    settings = get_settings()
    if settings.MODE == "real":
        raw_points = get_zarr_transect(lat1, lon1, lat2, lon2, date, n_points)
    else:
        raw_points = generate_transect(lat1, lon1, lat2, lon2, date, n_points)

    transect_points = []
    for p in raw_points:
        prof = [ProfilePoint(**item) for item in p["profile"]] if p["profile"] else None
        transect_points.append(TransectPoint(
            lat=p["lat"],
            lon=p["lon"],
            distance_km=p["distance_km"],
            is_ocean=p["is_ocean"],
            profile=prof,
        ))

    return TransectResponse(
        date=date,
        n_points=n_points,
        start=LocationPoint(lat=lat1, lon=lon1),
        end=LocationPoint(lat=lat2, lon=lon2),
        points=transect_points,
        mode=settings.MODE,
    )
