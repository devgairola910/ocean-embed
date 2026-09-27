import os
import sys
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.db.models import ArgoProfile, IngestionJob
from app.mock.generator import is_land, generate_profile_and_surface


def seed_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    existing_count = db.query(ArgoProfile).count()
    if existing_count > 0:
        print(f"Database already seeded with {existing_count} Argo profiles.")
        db.close()
        return

    print("Seeding Argo profiles into database...")

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 6, 30)
    total_days = (end_date - start_date).days

    profiles_to_insert = []
    float_seq = 2901000

    random.seed(42)
    count = 0

    while count < 350:
        lat = round(random.uniform(1.0, 24.0), 2)
        lon = round(random.uniform(46.0, 97.0), 2)

        if is_land(lat, lon):
            continue

        days_offset = random.randint(0, total_days)
        profile_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

        profile_data, _ = generate_profile_and_surface(lat, lon, profile_date)
        float_id = f"ARGO_{float_seq + count}"

        argo_obj = ArgoProfile(
            id=float_id,
            lat=lat,
            lon=lon,
            date=profile_date,
            profile_data=profile_data,
            source="INCOIS_ARGO_MOCK",
        )
        profiles_to_insert.append(argo_obj)
        count += 1

    db.add_all(profiles_to_insert)

    now = datetime.now(timezone.utc)
    job = IngestionJob(
        id="JOB_INIT_001",
        status="succeeded",
        started_at=now - timedelta(hours=1),
        finished_at=now,
        error_message=None,
    )
    db.add(job)

    db.commit()
    print(f"Successfully seeded {len(profiles_to_insert)} Argo profiles and initial IngestionJob.")
    db.close()


if __name__ == "__main__":
    seed_database()
