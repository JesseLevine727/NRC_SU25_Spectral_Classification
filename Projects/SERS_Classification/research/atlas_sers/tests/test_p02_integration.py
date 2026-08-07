from __future__ import annotations

from pathlib import Path

from atlas_sers.governance.p02 import execute_p02, validate_latest_p02
from tests.p02_fixtures import install_synthetic_p01, synthetic_manifest

PROJECT = Path(__file__).resolve().parents[1]


def test_synthetic_p02_build_validates_rebuilds_and_verified_skips(tmp_path: Path) -> None:
    private = tmp_path / "private_inputs"
    artifacts = tmp_path / "private_artifacts"
    private.mkdir()
    artifacts.mkdir()
    install_synthetic_p01(artifacts, synthetic_manifest())
    first, run_dir, action = execute_p02(
        project_root=PROJECT,
        private_root=private,
        artifact_root=artifacts,
    )
    assert action == "new"
    assert first["status"] == "pass"
    assert first["fit_invocations"] == 0
    assert first["primary_domains"] == 13
    assert all(first["checks"].values())
    assert (run_dir / "rebuild_evidence.json").is_file()
    repeated, repeated_dir, repeated_action = execute_p02(
        project_root=PROJECT,
        private_root=private,
        artifact_root=artifacts,
    )
    assert repeated_action == "verified_skip"
    assert repeated_dir == run_dir
    assert repeated["run_id"] == first["run_id"]
    validation = validate_latest_p02(artifacts)
    assert validation["status"] == "pass"
    assert all(validation["checks"].values())
