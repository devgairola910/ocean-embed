from fastapi import APIRouter, Query
from app.config import get_settings
from app.core.interpolation import get_zarr_grid_2d
from app.mock.generator import generate_grid_2d
from app.models.grid import GridResponse

router = APIRouter()


@router.get("/grid", response_model=GridResponse, summary="Get 2D temperature grid slice")
def get_grid(
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
    depth: float = Query(..., description="Depth level in meters", examples=[0.0]),
):
    settings = get_settings()
    if settings.MODE == "real":
        lats, lons, grid_data = get_zarr_grid_2d(date, depth)
    else:
        lats, lons, grid_data = generate_grid_2d(date, depth)

    return GridResponse(
        date=date,
        depth_m=depth,
        lats=lats,
        lons=lons,
        temperature_c=grid_data,
        mode=settings.MODE,
    )
