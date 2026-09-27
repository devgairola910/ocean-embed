from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import Float, String, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArgoProfile(Base):
    __tablename__ = "argo_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    lat: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    lon: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    profile_data: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="INCOIS_ARGO_MOCK", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
