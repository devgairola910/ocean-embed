from fastapi import APIRouter
from app.api.v1.metadata import router as metadata_router
from app.api.v1.predict import router as predict_router
from app.api.v1.grid import router as grid_router
from app.api.v1.transect import router as transect_router
from app.api.v1.argo import router as argo_router
from app.api.v1.products import router as products_router
from app.api.v1.admin import router as admin_router

router = APIRouter()

router.include_router(metadata_router, tags=["Metadata"])
router.include_router(predict_router, tags=["Prediction"])
router.include_router(grid_router, tags=["Grid"])
router.include_router(transect_router, tags=["Transect"])
router.include_router(argo_router, tags=["Argo Validation & Profiles"])
router.include_router(products_router, tags=["Derived Ocean Products"])
router.include_router(admin_router, tags=["Admin & Pipeline Control"])
