"""Tiny CLI wrapper for the P05-T016 approved 36-fit pilot boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas_sers.evaluation.p05_pilot import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
