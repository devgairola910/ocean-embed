import pytest
from app.config import get_settings
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.fetch import LocalFileSatelliteDataSource


@pytest.fixture
def real_mode_client(client, tmp_path):
    settings = get_settings()
    settings.MODE = "real"
    settings.DATA_OUTPUT_PATH = str(tmp_path)

    run_ingestion_pipeline(
        date_str="2025-05-15",
        data_source=LocalFileSatelliteDataSource(raw_data_dir=str(tmp_path)),
        output_dir=str(tmp_path),
    )

    yield client

    settings.MODE = "mock"


def test_real_mode_predict(real_mode_client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}
    res = real_mode_client.get("/api/v1/predict", params=params)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    assert len(data["profile"]) == 15


def test_real_mode_grid(real_mode_client):
    params = {"date": "2025-05-15", "depth": 0.0}
    res = real_mode_client.get("/api/v1/grid", params=params)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    assert len(data["lats"]) > 0


def test_real_mode_transect(real_mode_client):
    params = {
        "lat1": 10.0,
        "lon1": 65.0,
        "lat2": 15.0,
        "lon2": 75.0,
        "date": "2025-05-15",
        "n_points": 10,
    }
    res = real_mode_client.get("/api/v1/transect", params=params)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    assert len(data["points"]) == 10


def test_real_mode_products(real_mode_client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2025-05-15"}

    res_ohc = real_mode_client.get("/api/v1/products/ohc", params=params)
    assert res_ohc.status_code == 200
    assert res_ohc.json()["mode"] == "real"

    res_mhw = real_mode_client.get("/api/v1/products/marine-heatwave", params=params)
    assert res_mhw.status_code == 200
    assert res_mhw.json()["mode"] == "real"

    res_risk = real_mode_client.get("/api/v1/products/cyclone-risk", params=params)
    assert res_risk.status_code == 200
    assert res_risk.json()["mode"] == "real"


def test_real_mode_data_not_ready(real_mode_client):
    params = {"lat": 12.5, "lon": 65.0, "date": "2024-02-01"}
    res = real_mode_client.get("/api/v1/predict", params=params)
    assert res.status_code == 503
    data = res.json()
    assert data["error"]["code"] == "DATA_NOT_READY"
