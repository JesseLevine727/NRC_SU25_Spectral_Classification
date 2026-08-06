from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_sers.governance.canonical import deterministic_npz_bytes
from atlas_sers.preprocessing.representations import (
    build_representations,
    representation_invariants,
    representation_registry,
)

PROJECT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads((PROJECT / "plan" / "contracts" / "p01_governance_contract.json").read_text())


def _raw(rows: int = 8) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(400, 1850, dtype=float)
    matrix = np.asarray(
        [
            2 + 0.001 * axis + np.sin(axis / (11 + index)) + 0.1 * np.cos(axis / 7)
            for index in range(rows)
        ]
    )
    return axis, matrix


def test_all_eight_representations_pass_declared_invariants() -> None:
    axis, raw = _raw()
    bundles = build_representations(axis, raw, _contract())
    assert len(bundles) == 8
    assert bundles["R_MIN_400_1849"].intensity.shape == (8, 1450)
    for identifier, bundle in bundles.items():
        expected_features = 1450 if identifier == "R_MIN_400_1849" else 1401
        assert bundle.intensity.shape == (8, expected_features)
        assert representation_invariants(bundle)["status"] == "pass"


def test_flat_rows_are_reason_coded_and_fail_invariant() -> None:
    axis, raw = _raw(rows=2)
    raw[0] = 1.0
    bundles = build_representations(axis, raw, _contract())
    primary = bundles["R_MIN_400_1800"]
    assert primary.valid_rows[0] == np.False_
    assert primary.reason_codes[0] == "nonfinite_or_zero_range"
    assert representation_invariants(primary)["status"] == "fail"


def test_npz_bytes_and_row_order_hashes_are_deterministic_and_sensitive() -> None:
    axis, raw = _raw(rows=3)
    bundle = build_representations(axis, raw, _contract())["R_MIN_400_1800"]
    arrays = {"axis_cm1": bundle.axis, "intensity": bundle.intensity}
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(arrays)
    first = representation_registry(
        {bundle.representation_id: bundle},
        np.asarray(["a", "b", "c"]),
        source_bundle_sha256="1" * 64,
        code_sha256="2" * 64,
        config_sha256="3" * 64,
        run_id="P01-" + "4" * 24,
    )
    second = representation_registry(
        {bundle.representation_id: bundle},
        np.asarray(["c", "b", "a"]),
        source_bundle_sha256="1" * 64,
        code_sha256="2" * 64,
        config_sha256="3" * 64,
        run_id="P01-" + "4" * 24,
    )
    assert first.loc[0, "row_order_sha256"] != second.loc[0, "row_order_sha256"]
