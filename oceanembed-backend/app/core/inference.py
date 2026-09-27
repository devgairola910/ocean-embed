import os
from typing import List, Optional
import numpy as np

from app.config import get_settings
from app.core.constants import DEPTH_LEVELS_M
from app.mock.generator import is_land, generate_profile_and_surface

_onnx_session = None


def get_onnx_session():
    global _onnx_session
    settings = get_settings()
    if settings.MODE == "mock":
        return None

    model_path = os.path.join(settings.MODEL_PATH, "model.onnx")
    if not os.path.exists(model_path):
        return None

    if _onnx_session is None:
        try:
            import onnxruntime as ort
            _onnx_session = ort.InferenceSession(model_path)
        except Exception:
            _onnx_session = None

    return _onnx_session


def predict_subsurface(
    surface_batch: np.ndarray,
    lats: List[float],
    lons: List[float],
    date_str: str,
) -> np.ndarray:
    """Predict 3D subsurface temperature grid of shape (15, len(lats), len(lons))."""
    session = get_onnx_session()
    H, W = len(lats), len(lons)
    num_depths = len(DEPTH_LEVELS_M)

    if session is not None:
        try:
            input_name = session.get_inputs()[0].name
            raw_out = session.run(None, {input_name: surface_batch.astype(np.float32)})[0]
            if raw_out.ndim == 4:
                raw_out = raw_out[0]
            return raw_out
        except Exception:
            pass

    result_grid = np.zeros((num_depths, H, W), dtype=np.float32)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if is_land(lat, lon):
                result_grid[:, i, j] = np.nan
            else:
                prof, _ = generate_profile_and_surface(lat, lon, date_str)
                temps = [p["temperature_c"] for p in prof]
                result_grid[:, i, j] = temps

    return result_grid
