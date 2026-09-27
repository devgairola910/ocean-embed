from typing import List

GRID_LAT_MIN: float = 0.0
GRID_LAT_MAX: float = 25.0
GRID_LON_MIN: float = 45.0
GRID_LON_MAX: float = 100.0
GRID_RES: float = 0.25

DEPTH_LEVELS_M: List[float] = [
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450
]

VALID_DATE_START: str = "2024-01-01"
VALID_DATE_END: str = "2026-06-30"
MODEL_VERSION: str = "0.1.0-mock"
