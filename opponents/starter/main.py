"""Repository-layout wrapper for the packaged deterministic starter agent."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cortex_arena.starter import agent_fn, run_stdio  # noqa: E402, F401

if __name__ == "__main__":
    run_stdio()
