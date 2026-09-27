import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.db.models import ArgoProfile, IngestionJob
from app.mock.generator import generate_profile_and_surface
from app.main import create_app


SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def seed_test_database():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()

    test_data = [
        ("ARGO_TEST_001", 12.55, 65.05, "2025-05-15"),
        ("ARGO_TEST_002", 12.45, 64.95, "2025-05-15"),
        ("ARGO_TEST_003", 12.60, 65.10, "2025-05-15"),
        ("ARGO_TEST_004", 10.00, 70.00, "2025-05-15"),
        ("ARGO_TEST_005", 15.00, 85.00, "2024-06-01"),
    ]

    for fid, lat, lon, date_str in test_data:
        profile_data, _ = generate_profile_and_surface(lat, lon, date_str)
        argo_obj = ArgoProfile(
            id=fid,
            lat=lat,
            lon=lon,
            date=date_str,
            profile_data=profile_data,
            source="TEST_MOCK_SEED",
        )
        db.add(argo_obj)

    job = IngestionJob(
        id="TEST_JOB_001",
        status="succeeded",
    )
    db.add(job)
    db.commit()
    db.close()


@pytest.fixture
def client():
    settings = get_settings()
    settings.MODE = "mock"
    settings.DATABASE_URL = SQLALCHEMY_TEST_DATABASE_URL

    seed_test_database()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()
