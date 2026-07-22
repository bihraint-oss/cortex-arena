"""Lux/Kaggle process entry point.

Keep this file at the submission root. All strategy code lives in the tested
``cortex_arena`` package under ``src``.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cortex_arena.protocol import agent_fn, run_stdio  # noqa: E402, F401

if __name__ == "__main__":
    run_stdio()
