from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.governance.inputs import verify_authoritative_inputs


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, capture_output=True, check=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    private = repository / "private_inputs"
    (private / "tables").mkdir(parents=True)
    (private / "arrays").mkdir()
    (private / "preprocessing").mkdir()
    table = private / "tables" / "primary.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_uid", "label"])
        writer.writerow(["o1", "a"])
        writer.writerow(["o2", "b"])
    raw = private / "arrays" / "raw.npz"
    np.savez_compressed(raw, axis_cm1=np.arange(3), intensity=np.ones((2, 3)))
    candidates = private / "preprocessing" / "candidates.npz"
    np.savez_compressed(candidates, minimal=np.ones((2, 3)))
    validation = private / "validation.json"
    validation.write_text(json.dumps({"status": "pass"}))
    declarations = [
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
    return repository, private, {"authoritative_inputs": declarations}


def test_private_input_verification_passes_and_exposes_no_root(tmp_path: Path) -> None:
    repository, private, contract = _fixture(tmp_path)
    report = verify_authoritative_inputs(contract, private_root=private, repository_root=repository)
    assert report["status"] == "pass"
    serialized = json.dumps(report)
    assert str(private) not in serialized
    assert report["checks"]["private_inputs_are_not_git_tracked"] is True


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    repository, private, contract = _fixture(tmp_path)
    (private / "tables" / "primary.csv").write_text("changed\n")
    report = verify_authoritative_inputs(contract, private_root=private, repository_root=repository)
    assert report["status"] == "fail"
    assert report["inputs"][0]["checks"]["sha256_matches"] is False


def test_missing_input_is_explicitly_blocked(tmp_path: Path) -> None:
    repository, private, contract = _fixture(tmp_path)
    (private / "validation.json").unlink()
    report = verify_authoritative_inputs(contract, private_root=private, repository_root=repository)
    assert report["status"] == "blocked"
    assert report["inputs"][-1]["checks"]["exists"] is False


def test_git_tracked_private_input_fails(tmp_path: Path) -> None:
    repository, private, contract = _fixture(tmp_path)
    _git(repository, "add", "private_inputs/tables/primary.csv")
    report = verify_authoritative_inputs(contract, private_root=private, repository_root=repository)
    assert report["status"] == "fail"
    assert report["checks"]["private_inputs_are_not_git_tracked"] is False
