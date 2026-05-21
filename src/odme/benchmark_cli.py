"""Development entry point for the repository benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (_PROJECT_ROOT / "benchmarks").exists():
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.cli import app  # noqa: E402

__all__ = ["app"]
