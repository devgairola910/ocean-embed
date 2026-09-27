from fastapi import APIRouter
from app.core.constants import (
    GRID_LAT_MIN, GRID_LAT_MAX,
    GRID_LON_MIN, GRID_LON_MAX,
    GRID_RES, DEPTH_LEVELS_M,
    VALID_DATE_START, VALID_DATE_END,
    MODEL_VERSION,
)
from app.models.metadata import MetadataResponse, DomainBounds, ValidDateRange

router = APIRouter()


@router.get("/metadata", response_model=MetadataResponse, summary="Get domain & model metadata")
def get_metadata():
    return MetadataResponse(
        domain_bounds=DomainBounds(
            lat_min=GRID_LAT_MIN,
            lat_max=GRID_LAT_MAX,
            lon_min=GRID_LON_MIN,
            lon_max=GRID_LON_MAX,
            resolution_deg=GRID_RES,
        ),
        depth_levels_m=DEPTH_LEVELS_M,
        valid_date_range=ValidDateRange(
            start=VALID_DATE_START,
            end=VALID_DATE_END,
        ),
        model_version=MODEL_VERSION,
        mode="mock",
    )
