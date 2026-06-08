"""
api/index.py — Entry point cho Vercel Serverless Function.
Chuyển hướng đến app FastAPI từ app.py.
"""

import sys
from pathlib import Path

# Đảm bảo thư mục gốc của project nằm trong sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app import app  # noqa: E402
