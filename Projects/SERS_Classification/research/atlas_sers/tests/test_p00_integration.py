from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.governance.p00 import execute_p00

SOURCE_PROJECT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, capture_output=True, check=True)


def _private_fixture(root: Path) -> list[dict[str, object]]:
    (root / "tables").mkdir(parents=True)
    (root / "arrays").mkdir()
    (root / "preprocessing").mkdir()
    table = root / "tables" / "primary.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_uid", "target"])
        writer.writerow(["o1", "a"])
        writer.writerow(["o2", "b"])
    raw = root / "arrays" / "raw.npz"
    np.savez_compressed(raw, axis_cm1=np.arange(3), intensity=np.ones((2, 3)))
    candidates = root / "preprocessing" / "candidates.npz"
    np.savez_compressed(candidates, minimal=np.ones((2, 3)))
    validation = root / "validation.json"
    validation.write_text('{"status":"pass"}\n')
    return [
        {
            "path": "${ATLAS_PRIVATE_ROOT}/tables/primary.csv",
            "sha256": sha256_file(table),
            "expected_rows": 2,
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/arrays/raw.npz",
            "sha256": sha256_file(raw),
            "expected_shape": [2, 3],
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/preprocessing/candidates.npz",
            "sha256": sha256_file(candidates),
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/validation.json",
            "sha256": sha256_file(validation),
            "required_status": "pass",
        },
    ]


def _project_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    project = repository / "research" / "atlas_sers"
    private = tmp_path / "private_inputs"
    artifacts = tmp_path / "private_artifacts"
    shutil.copytree(
        SOURCE_PROJECT,
        project,
        ignore=shutil.ignore_patterns(".pytest_cache", ".ruff_cache", "__pycache__", "*.egg-info"),
    )
    contract_path = project / "plan" / "contracts" / "research_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["authoritative_inputs"] = _private_fixture(private)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    phase_path = project / "plan" / "registries" / "phase_registry.csv"
    planned_phase = (
        "P00,primary,Governance and artifact contract,completed_restart,"
        'validated restart artifacts,"protocol registry; environment lock; hashes; '
        'deviations log",G0,planned'
    )
    completed_phase = (
        "P00,primary,Governance and artifact contract,completed_restart,"
        'validated restart artifacts,"protocol registry; environment lock; hashes; '
        'deviations log",G0,complete'
    )
    phase_path.write_text(phase_path.read_text().replace(planned_phase, completed_phase))
    repository.mkdir(exist_ok=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "ATLAS Test")
    _git(repository, "config", "user.email", "atlas-test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "Synthetic P00 fixture")
    return project, private, artifacts


def test_full_p00_passes_and_second_execution_is_verified_skip(tmp_path: Path) -> None:
    project, private, artifacts = _project_fixture(tmp_path)
    report, final_dir, action = execute_p00(
        project_root=project,
        private_root=private,
        artifact_root=artifacts,
    )
    assert action == "new"
    assert report["status"] == "pass", report["diagnostics"]
    assert report["checks"]["p00_validation_schema_conforms"] is True
    assert report["fit_invocations"] == 0
    assert all(report["checks"].values())
    required = set(report["required_artifacts"])
    assert required <= {path.name for path in final_dir.iterdir()}
    repeated, repeated_dir, repeated_action = execute_p00(
        project_root=project,
        private_root=private,
        artifact_root=artifacts,
    )
    assert repeated_action == "verified_skip"
    assert repeated == report
    assert repeated_dir == final_dir
    assert (artifacts / "p00" / "LATEST.json").is_file()
