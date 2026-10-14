"""Ensures the project root is importable (``from src import ...``) when
pytest is run from the project root, a subfolder, or an IDE test runner."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
