from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_sers.governance.canonical import deterministic_npz_bytes
from atlas_sers.governance.p01 import _artifact_tree_is_sanitized, p01_dry_run

PROJECT = Path(__file__).resolve().parents[1]


def test_p01_contract_declares_exact_representations_figures_and_no_training() -> None:
    contract = json.loads(
        (PROJECT / "plan" / "contracts" / "p01_governance_contract.json").read_text()
    )
    assert len(contract["representations"]) == 8
    assert contract["required_figures"] == [f"F{index:02d}" for index in range(2, 10)]
    assert contract["model_fitting_authorized"] is False
    assert contract["split_construction_authorized"] is False
    report = p01_dry_run(PROJECT)
    assert report["status"] == "pass"
    assert report["p02_fields"] == "explicitly unresolved"


def test_privacy_scan_parses_npz_arrays_instead_of_scanning_compressed_bytes(tmp_path) -> None:
    sensitive = bytes((47, 104, 111, 109, 101, 47)).decode()
    numeric = np.frombuffer(sensitive.encode(), dtype=np.uint8)
    archive = tmp_path / "numeric.npz"
    archive.write_bytes(deterministic_npz_bytes({"intensity": numeric}))
    assert _artifact_tree_is_sanitized(tmp_path)

    string_archive = tmp_path / "string.npz"
    string_archive.write_bytes(
        deterministic_npz_bytes({"observation_uid": np.asarray([sensitive])})
    )
    assert not _artifact_tree_is_sanitized(tmp_path)
