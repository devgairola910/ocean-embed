from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.v1.router import router as api_v1_router
from app.config import get_settings
from app.core.errors import APIException
from app.core.inference import get_onnx_session
from app.core.logging import setup_logging, logger
from app.db.base import Base
from app.db.session import engine, get_db

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Base.metadata.create_all(bind=engine)

    session = get_onnx_session()
    model_status = "loaded" if session is not None else "not_loaded/mock"
    logger.info(
        f"OceanEmbed API starting up | MODE={settings.MODE} | DB={settings.DATABASE_URL} | ONNX_Model={model_status}"
    )
    yield
    logger.info("OceanEmbed API shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OceanEmbed API",
        description="Subsurface Ocean Temperature Reconstruction Backend (SIH26066)",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        logger.warning(f"API Error [{exc.status_code}] path={request.url.path} code={exc.code} message={exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation Error [400] path={request.url.path} detail={str(exc)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_PARAMETER", "message": str(exc)}},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTP Error [{exc.status_code}] path={request.url.path} detail={str(exc.detail)}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}},
        )

    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": "OceanEmbed API",
            "version": "0.1.0",
            "description": "Subsurface Ocean Temperature Reconstruction Backend",
            "docs": "/docs",
            "health": "/health",
            "api_v1": settings.API_V1_PREFIX,
        }

    @app.get("/health", tags=["Health Check"])
    def health_check():
        return {"status": "ok"}

    @app.get("/ready", tags=["Health Check"])
    def readiness_check(db: Session = Depends(get_db)):
        try:
            db.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"Readiness DB check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "error": "Database unreachable"},
            )

        if settings.MODE == "real":
            session = get_onnx_session()
            if session is None:
                logger.error("Readiness check failed: MODE=real but ONNX model not loaded.")
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "error": "ONNX model not loaded"},
                )

        return {"status": "ok"}

    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
