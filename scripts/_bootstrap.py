"""Shared bootstrap helpers for scripts run directly or with ``python -m``."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path() -> Path:
    """Make repository modules importable when a script is launched by path."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT

