from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.repositories import argo_repository
from app.models.predict import LocationPoint, ProfilePoint
from app.models.argo import (
    ArgoValidationResponse,
    OverallValidation,
    DepthValidation,
    ArgoProfilesResponse,
    ArgoFloatProfile,
)

router = APIRouter()


@router.get("/argo/validation", response_model=ArgoValidationResponse, summary="Get Argo matchup validation statistics")
def get_argo_validation(
    date_from: str = Query("2024-01-01", description="Start date (YYYY-MM-DD)", examples=["2024-01-01"]),
    date_to: str = Query("2026-06-30", description="End date (YYYY-MM-DD)", examples=["2026-06-30"]),
    db: Session = Depends(get_db),
):
    data = argo_repository.get_argo_validation_stats(db, date_from, date_to)
    return ArgoValidationResponse(
        date_from=data["date_from"],
        date_to=data["date_to"],
        n_matchups=data["n_matchups"],
        overall=OverallValidation(**data["overall"]),
        by_depth=[DepthValidation(**item) for item in data["by_depth"]],
        mode="mock",
    )


@router.get("/argo/profiles", response_model=ArgoProfilesResponse, summary="Get nearby Argo float profiles")
def get_argo_profiles(
    lat: float = Query(..., description="Center latitude °N", examples=[12.5]),
    lon: float = Query(..., description="Center longitude °E", examples=[65.0]),
    radius_km: float = Query(100.0, ge=1.0, le=1000.0, description="Radius in km", examples=[100.0]),
    date: str = Query(..., description="Target date (YYYY-MM-DD)", examples=["2025-05-15"]),
    db: Session = Depends(get_db),
):
    raw_profiles = argo_repository.get_argo_profiles_near(db, lat, lon, radius_km, date)
    float_profiles = []
    for f in raw_profiles:
        float_profiles.append(ArgoFloatProfile(
            float_id=f["float_id"],
            lat=f["lat"],
            lon=f["lon"],
            date=f["date"],
            distance_km=f["distance_km"],
            profile=[ProfilePoint(**item) for item in f["profile"]],
        ))

    return ArgoProfilesResponse(
        requested_location=LocationPoint(lat=lat, lon=lon),
        radius_km=radius_km,
        date=date,
        profiles=float_profiles,
        mode="mock",
    )
