"""Pytest configuration and shared fixtures for AnomalyScope AI tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for test discovery
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
