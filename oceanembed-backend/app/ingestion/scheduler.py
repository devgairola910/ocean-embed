from datetime import datetime, timezone
import uuid

from apscheduler.schedulers.background import BackgroundScheduler
from app.config import get_settings
from app.db.models import IngestionJob
from app.db.session import SessionLocal
from app.ingestion.pipeline import run_ingestion_pipeline

scheduler = BackgroundScheduler()


def execute_scheduled_ingestion():
    settings = get_settings()

    if settings.MODE != "real":
        return

    job_id = f"JOB_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()

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

    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_ingestion_pipeline(today_str)

        job_record.status = "succeeded"
        job_record.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        job_record.status = "failed"
        job_record.finished_at = datetime.now(timezone.utc)
        job_record.error_message = str(e)
        db.commit()
    finally:
        db.close()


def start_scheduler():
    settings = get_settings()
    if settings.MODE == "real":
        interval_hours = getattr(settings, "INGESTION_INTERVAL_HOURS", 24)
        scheduler.add_job(
            execute_scheduled_ingestion,
            "interval",
            hours=interval_hours,
            id="daily_ingestion_job",
            replace_existing=True,
        )
        scheduler.start()
        print(f"APScheduler started: daily ingestion active every {interval_hours}h.")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
