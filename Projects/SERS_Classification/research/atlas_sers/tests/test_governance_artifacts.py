from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from atlas_sers.governance.artifacts import ArtifactStore

NOW = datetime(2026, 8, 5, tzinfo=UTC)


def _store(tmp_path: Path) -> ArtifactStore:
    project = tmp_path / "public" / "atlas_sers"
    inputs = tmp_path / "private_inputs"
    artifacts = tmp_path / "private_artifacts"
    project.mkdir(parents=True)
    inputs.mkdir()
    return ArtifactStore(
        artifact_root=artifacts,
        input_root=inputs,
        project_root=project,
        clock=lambda: NOW,
    )


def test_successful_matching_run_is_verified_and_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = store.begin(run_id="P00-" + "1" * 24, protected_state_sha256="2" * 64)
    assert lease.action == "new"
    assert lease.work_dir is not None
    (lease.work_dir / "result.json").write_text("{}\n")
    store.commit(lease, scientific_status="pass")
    repeated = store.begin(run_id=lease.run_id, protected_state_sha256="2" * 64)
    assert repeated.action == "verified_skip"


def test_corrupt_completed_run_is_quarantined_not_overwritten(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id = "P00-" + "3" * 24
    lease = store.begin(run_id=run_id, protected_state_sha256="4" * 64)
    assert lease.work_dir is not None
    (lease.work_dir / "result.json").write_text("{}\n")
    final = store.commit(lease, scientific_status="pass")
    (final / "result.json").write_text("corrupt\n")
    replacement = store.begin(run_id=run_id, protected_state_sha256="4" * 64)
    assert replacement.action == "new"
    quarantine = list(store.quarantine.glob(f"{run_id}--*"))
    assert len(quarantine) == 1
    assert (quarantine[0] / "QUARANTINE_REASON.json").is_file()


def test_output_root_inside_inputs_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "public"
    inputs = tmp_path / "inputs"
    project.mkdir()
    inputs.mkdir()
    with pytest.raises(ValueError, match="must not overlap"):
        ArtifactStore(
            artifact_root=inputs / "outputs",
            input_root=inputs,
            project_root=project,
        )


def test_output_root_inside_public_project_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "public"
    inputs = tmp_path / "inputs"
    project.mkdir()
    inputs.mkdir()
    with pytest.raises(ValueError, match="outside the public project"):
        ArtifactStore(
            artifact_root=project / "artifacts",
            input_root=inputs,
            project_root=project,
        )
