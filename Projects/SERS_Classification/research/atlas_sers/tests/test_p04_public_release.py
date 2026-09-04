from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def test_p04_public_release_when_materialized() -> None:
    if not (PROJECT / "results/p04_deep/release_manifest.json").is_file():
        pytest.skip("P04 public release has not been materialized yet.")
    result = subprocess.run(
        [sys.executable, "scripts/validate_p04_public_release.py"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
