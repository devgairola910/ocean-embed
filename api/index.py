"""Vercel Serverless Function entrypoint for OceanEmbed FastAPI backend."""

import os
import sys

# Ensure oceanembed-backend is on the Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "oceanembed-backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
