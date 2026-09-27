# OceanEmbed Backend

OceanEmbed is a Smart India Hackathon project (SIH26066, MoES/INCOIS) for reconstructing subsurface ocean temperature profiles from surface satellite data.

This FastAPI service acts as the production backend, delivering high-resolution 3D ocean temperature fields, Argo matchup validations, and derived marine risk products.

---

## 🏗️ System Architecture

```
                                 ┌────────────────────────┐
                                 │ Satellite Data Sources │
                                 │ (Local NetCDF / API)   │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Ingestion Pipeline       │
                               │  (Fetch → Regrid → Mask)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  ONNX Model Inference     │
                               │  (model.onnx / Fallback)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Zarr Store Output        │
                               │  (data/outputs/*.zarr)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
 ┌────────────────┐            ┌───────────────────────────┐            ┌────────────────┐
 │ Frontend App   │ ──(CORS)─► │ FastAPI REST Service      │ ─────────► │ SQLite / DB    │
 │ (Port 3000/8501)│           │ (MODE=mock | MODE=real)   │            │ (Argo & Jobs)  │
 └────────────────┘            └───────────────────────────┘            └────────────────┘
```

---

## 📁 Repository Structure

```
oceanembed-backend/
├── app/
│   ├── main.py              # FastAPI app factory, middleware, CORS, logging & /ready check
│   ├── config.py            # Application settings (pydantic-settings)
│   ├── api/
│   │   └── v1/              # V1 API Endpoints
│   │       ├── router.py    # V1 Router aggregator
│   │       ├── metadata.py  # GET /api/v1/metadata
│   │       ├── predict.py   # GET /api/v1/predict
│   │       ├── grid.py      # GET /api/v1/grid
│   │       ├── transect.py  # GET /api/v1/transect
│   │       ├── argo.py      # GET /api/v1/argo/validation & GET /api/v1/argo/profiles
│   │       ├── products.py  # GET /api/v1/products/ohc, marine-heatwave, cyclone-risk
│   │       └── admin.py     # POST /admin/ingest/trigger & GET /admin/ingest/status/{job_id}
│   ├── core/
│   │   ├── constants.py     # Domain bounds & grid constants (0-25°N, 45-100°E)
│   │   ├── errors.py        # APIException & error definitions
│   │   ├── inference.py     # ONNX Runtime model inference & mock fallback handler
│   │   ├── interpolation.py # Zarr store dataset reader & spatial interpolator
│   │   └── logging.py       # Structured logging setup
│   ├── db/
│   │   ├── base.py          # SQLAlchemy DeclarativeBase
│   │   ├── session.py       # Engine & SessionLocal setup
│   │   ├── models.py        # ArgoProfile & IngestionJob ORM models
│   │   └── repositories/    # Repository pattern for database queries
│   ├── ingestion/           # Real ingestion pipeline (Write-Side)
│   │   ├── fetch.py         # SatelliteDataSource ABC & local file implementation
│   │   ├── regrid.py        # 0.25° grid interpolation, land masking & normalization
│   │   ├── pipeline.py      # Fetch -> Regrid -> Infer -> Zarr store writer
│   │   └── scheduler.py     # APScheduler background runner for MODE=real
│   ├── models/              # Pydantic request/response & admin schemas
│   └── mock/
│       └── generator.py     # Physics-based thermocline & product generators
├── scripts/
│   ├── seed_mock_argo.py    # Database seed script for Argo profiles & IngestionJob
│   └── export_openapi.py    # Export openapi.json contract file
├── data/                    # Data storage (gitignored)
│   ├── raw/                 # Input satellite NetCDF files
│   ├── processed/           # Regridded intermediate data
│   └── outputs/             # Ingested output Zarr stores
├── models_store/            # ML model store directory (gitignored, contains model.onnx)
├── tests/                   # Pytest test suite (28 tests)
├── docker-compose.yml       # Docker Compose setup
├── Dockerfile               # Container build configuration
├── openapi.json             # Exported OpenAPI contract file
├── requirements.txt         # Core dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration & Execution Modes (`MODE`)

Managed in `app/config.py` using `pydantic-settings`:

| Variable | Default | Description |
|---|---|---|
| `MODE` | `mock` | `mock` (physics-based fake generator) or `real` (precomputed Zarr store). |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501,http://localhost:5173` | Allowed CORS origins (comma-separated). |
| `DATABASE_URL` | `sqlite:///./oceanembed.db` | Database connection string. |
| `MODEL_PATH` | `models_store` | Path to ML model directory containing `model.onnx`. |
| `DATA_OUTPUT_PATH` | `data/outputs` | Output directory for Zarr stores. |
| `INGESTION_INTERVAL_HOURS` | `24` | APScheduler interval for background ingestion. |

### How to Run:

#### 1. Mock Mode (Default for Frontend Dev)
```bash
# In .env file: MODE=mock
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Real Mode (Using ONNX Model & Zarr Store)
```bash
# In .env file: MODE=real
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 3. Via Docker Compose
```bash
docker-compose up --build
```

---

## 🧪 Running Tests

To run the complete automated test suite (28 offline unit tests):

```bash
pytest
```

---

## 🔗 Complete API Reference & Working `curl` Examples

### 1. GET `/health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### 2. GET `/ready`
```bash
curl http://localhost:8000/ready
```
```json
{"status": "ok"}
```

### 3. GET `/api/v1/metadata`
```bash
curl http://localhost:8000/api/v1/metadata
```
```json
{
  "domain_bounds": {"lat_min": 0.0, "lat_max": 25.0, "lon_min": 45.0, "lon_max": 100.0, "resolution_deg": 0.25},
  "depth_levels_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450],
  "valid_date_range": {"start": "2024-01-01", "end": "2026-06-30"},
  "model_version": "0.1.0-mock",
  "mode": "mock"
}
```

### 4. GET `/api/v1/predict`
```bash
curl "http://localhost:8000/api/v1/predict?lat=12.5&lon=65.0&date=2025-05-15"
```
```json
{
  "requested_location": {"lat": 12.5, "lon": 65.0},
  "nearest_grid_point": {"lat": 12.5, "lon": 65.0},
  "date": "2025-05-15",
  "profile": [
    {"depth_m": 0.0, "temperature_c": 28.5, "uncertainty_c": 0.18},
    {"depth_m": 450.0, "temperature_c": 8.12, "uncertainty_c": 0.63}
  ],
  "surface_inputs_used": {"sst": 28.5, "sss": 34.2, "sla": 0.02, "u": 0.1, "v": -0.05},
  "mode": "mock"
}
```

### 5. GET `/api/v1/grid`
```bash
curl "http://localhost:8000/api/v1/grid?date=2025-05-15&depth=0.0"
```
```json
{
  "date": "2025-05-15",
  "depth_m": 0.0,
  "lats": [0.0, 0.25, 0.5, "..."],
  "lons": [45.0, 45.25, "..."],
  "temperature_c": [[28.5, null, 27.8]],
  "mode": "mock"
}
```

### 6. GET `/api/v1/transect`
```bash
curl "http://localhost:8000/api/v1/transect?lat1=10.0&lon1=65.0&lat2=15.0&lon2=75.0&date=2025-05-15&n_points=50"
```

### 7. GET `/api/v1/argo/validation`
```bash
curl "http://localhost:8000/api/v1/argo/validation?date_from=2024-01-01&date_to=2026-06-30"
```

### 8. GET `/api/v1/argo/profiles`
```bash
curl "http://localhost:8000/api/v1/argo/profiles?lat=12.5&lon=65.0&radius_km=100&date=2025-05-15"
```

### 9. GET `/api/v1/products/ohc`
```bash
curl "http://localhost:8000/api/v1/products/ohc?lat=12.5&lon=65.0&date=2025-05-15"
```

### 10. GET `/api/v1/products/marine-heatwave`
```bash
curl "http://localhost:8000/api/v1/products/marine-heatwave?lat=12.5&lon=65.0&date=2025-05-15"
```

### 11. GET `/api/v1/products/cyclone-risk`
```bash
curl "http://localhost:8000/api/v1/products/cyclone-risk?lat=12.5&lon=65.0&date=2025-05-15"
```

### 12. POST `/api/v1/admin/ingest/trigger`
```bash
curl -X POST "http://localhost:8000/api/v1/admin/ingest/trigger?date=2025-05-15"
```

---

## 🛠️ Troubleshooting Guide

| Issue | Cause | Fix |
|---|---|---|
| **`503 DATA_NOT_READY`** | Running in `MODE=real` but no precomputed Zarr store exists in `data/outputs/` for the requested date. | Trigger ingestion via `POST /api/v1/admin/ingest/trigger?date=YYYY-MM-DD` or set `MODE=mock` in `.env`. |
| **`503 ONNX model not loaded` on `/ready`** | `MODE=real` is active but `models_store/model.onnx` is missing. | Place the exported `model.onnx` into `models_store/` or switch to `MODE=mock`. |
| **CORS Request Blocked** | Frontend origin URL is missing from `CORS_ORIGINS`. | Add your frontend origin URL to `CORS_ORIGINS` in `.env` (e.g., `CORS_ORIGINS=http://localhost:3000,http://localhost:8501`). |
| **`400 OUT_OF_DOMAIN`** | Requested coordinates are outside bounds (0–25°N, 45–100°E) or on land. | Pass valid ocean coordinates (e.g. lat=12.5, lon=65.0 in Arabian Sea). |
