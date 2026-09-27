from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.errors import APIException
from app.db.models import IngestionJob
from app.db.session import get_db
from app.ingestion.pipeline import run_ingestion_pipeline
from app.models.admin import IngestStatusResponse, IngestTriggerResponse

router = APIRouter()


def _run_pipeline_task(job_id: str, date_str: str):
    from app.db.session import SessionLocal

    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    try:
        run_ingestion_pipeline(date_str)
        if job:
            job.status = "succeeded"
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        if job:
            job.status = "failed"
            job.finished_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/admin/ingest/trigger", response_model=IngestTriggerResponse, summary="Trigger ingestion pipeline job")
def trigger_ingestion(
    background_tasks: BackgroundTasks,
    date: str = Query("2025-05-15", description="Target date YYYY-MM-DD", examples=["2025-05-15"]),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    job_id = f"JOB_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    job_record = IngestionJob(
        id=job_id,
        status="running",
        started_at=now,
        finished_at=None,
        error_message=None,
    )
    db.add(job_record)
    db.commit()

    background_tasks.add_task(_run_pipeline_task, job_id, date)

    return IngestTriggerResponse(
        job_id=job_id,
        status="running",
        started_at=now.isoformat(),
        mode=settings.MODE,
    )


@router.get("/admin/ingest/status/{job_id}", response_model=IngestStatusResponse, summary="Get ingestion job status")
def get_ingestion_status(
    job_id: str,
    db: Session = Depends(get_db),
):
    settings = get_settings()
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        raise APIException(
            status_code=400,
            code="JOB_NOT_FOUND",
            message=f"Ingestion job '{job_id}' not found.",
        )

    return IngestStatusResponse(
        job_id=job.id,
        status=job.status,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        error_message=job.error_message,
        mode=settings.MODE,
    )
