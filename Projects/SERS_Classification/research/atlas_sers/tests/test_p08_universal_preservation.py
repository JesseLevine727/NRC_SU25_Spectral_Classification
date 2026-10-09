"""Compact private-prep tests for p08_universal_preservation.

Synthetic schema-hashed fixtures only. No field data, no fitting, no public
runtime changes, no writes outside pytest tmp_path. Only frozen counts/pins
shrink inside fixtures; validation logic is never monkeypatched.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p08_universal_preservation as p08
from atlas_sers.evaluation.p08_universal_evidence import (
    UniversalEvidenceError,
    _authenticated_file,
    _prepare_roots,
    _read_json_file,
    _stream_sha256,
)

MANIFEST_ROWS = (
    ("u0", "M0", "S0", "I0", "F0", "A0"),
    ("u1", "M0", "S0", "I0", "F0", "A0"),
    ("u2", "M1", "S0", "I0", "F0", "A0"),
    ("u3", "M2", "S1", "I1", "F1", "A1"),
    ("u4", "M2", "S1", "I1", "F1", "A1"),
    ("u5", "M3", "S1", "I1", "F1", "A1"),
    ("u6", "M4", "S0", "I0", "F0", "A2"),
)
CELL_SPECS = (
    ("S0", "I0", "A0", (("M0", ("u0", "u1")), ("M1", ("u2",)))),
    ("S1", "I1", "A1", (("M2", ("u3", "u4")), ("M3", ("u5",)))),
    ("S0", "I0", "A2", (("M4", ("u6",)),)),
)
DOMAIN_MEMBERSHIPS = (
    ("S0", "I0", ("u0", "u1", "u2", "u6")),
    ("S1", "I1", ("u3", "u4", "u5")),
)
SHRUNK = {
    "EXPECTED_PRIMARY_SPECTRA": 7,
    "EXPECTED_PRIMARY_MASTERS": 5,
    "EXPECTED_PRIMARY_INSTRUMENTS": 2,
    "EXPECTED_PRIMARY_DOMAINS": 2,
    "EXPECTED_HELD_DOMAINS": 1,
    "EXPECTED_EXPLORATORY_DOMAINS": 1,
    "EXPECTED_PUBLIC_CELLS": 3,
    "EXPECTED_ELIGIBLE_CELLS": 2,
    "EXPECTED_UNAVAILABLE_CELLS": 1,
    "EXPECTED_PUBLIC_CURVES": 6,
    "EXPECTED_HISTORICAL_RECORDS": 56,
    "EXPECTED_HISTORICAL_REPRESENTATIONS": 8,
    "EXPECTED_PRIMARY_ALIAS_RECORDS": 21,
    "EXPECTED_DOMAIN_ACTION_GROUPS": 6,
    "EXPECTED_FEATURES": 5,
    "AXIS_START": 400,
    "AXIS_STOP": 405,
}


def _catalog() -> dict:
    cells = []
    for station, instrument, analyte, groups in CELL_SPECS:
        master_groups = [
            {"master_id": master, "observation_uids": list(uids)}
            for master, uids in groups
        ]
        masters = len(master_groups)
        cells.append(
            {
                "station": station,
                "instrument": instrument,
                "analyte": analyte,
                "master_groups": master_groups,
                "n_masters": masters,
                "n_spectra": sum(len(group["observation_uids"]) for group in master_groups),
                "public_spectral_aggregate_eligible": masters >= 2,
            }
        )
    return {
        "spectral_figure_cells": cells,
        "domain_memberships": [
            {"station": station, "instrument": instrument, "observation_uids": list(uids)}
            for station, instrument, uids in DOMAIN_MEMBERSHIPS
        ],
        "selected_preservation_rows": {
            action: [row[0] for row in MANIFEST_ROWS] for action in p08.ACTION_ORDER
        },
    }


def _sha_size(path: Path) -> dict:
    return {"sha256": _stream_sha256(path), "size_bytes": path.stat().st_size}


def _build_world(tmp_path, monkeypatch) -> dict:
    for name, value in SHRUNK.items():
        monkeypatch.setattr(p08, name, value)
    package = tmp_path / "package"
    p01 = tmp_path / "p01"
    private = tmp_path / "private"
    p01.mkdir(parents=True)
    private.mkdir(parents=True)

    manifest_path = p01 / p08.P01_PRIMARY_MANIFEST_FILENAME
    manifest_path.write_text(
        "observation_uid,master_sample_id,station,instrument,sensor_family,target_analyte\n"
        + "".join(",".join(row) + "\n" for row in MANIFEST_ROWS),
        encoding="utf-8",
    )

    header = [
        "representation_id",
        "observation_uid",
        "station",
        "instrument",
        "sensor_family",
        *p08.METRIC_COLUMNS,
    ]
    lines = [",".join(header)]
    for rep in [*p08.ACTION_ORDER, *[f"R{index:02d}" for index in range(5)]]:
        for index, (uid, _master, station, instrument, family, _analyte) in enumerate(
            MANIFEST_ROWS
        ):
            values = [
                ""
                if (uid == "u6" and column == "baseline_span")
                else f"{0.01 * (index + position + 1):.6f}"
                for position, column in enumerate(p08.METRIC_COLUMNS)
            ]
            lines.append(",".join([rep, uid, station, instrument, family, *values]))
    metrics_path = p01 / p08.P01_PRESERVATION_METRICS_FILENAME
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_instrument_path = p01 / p08.P01_PRESERVATION_BY_INSTRUMENT_FILENAME
    by_instrument_path.write_text(
        "station,instrument,representation_id,metric,n_spectra\n"
        "S0,I0,R00,baseline_span,4\n",
        encoding="utf-8",
    )

    axis = np.arange(p08.AXIS_START, p08.AXIS_STOP, dtype=np.float32)
    uids = np.array([row[0] for row in MANIFEST_ROWS], dtype="U2")
    intensity = np.full((len(MANIFEST_ROWS), axis.size), 0.25, dtype=np.float32)
    npz_paths = {}
    for action in p08.ACTION_ORDER:
        path = p01 / p08.REPRESENTATIONS_DIRNAME / f"{action}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, axis_cm1=axis, intensity=intensity, observation_uid=uids)
        npz_paths[action] = path

    source_path = package / p08.REPRESENTATIONS_SOURCE_REL
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# frozen source\n", encoding="utf-8")

    artifact_files: dict = {}
    state_files: dict = {}
    for name, path in (
        (p08.P01_PRIMARY_MANIFEST_FILENAME, manifest_path),
        (p08.P01_PRESERVATION_METRICS_FILENAME, metrics_path),
        (p08.P01_PRESERVATION_BY_INSTRUMENT_FILENAME, by_instrument_path),
    ):
        artifact_files[name] = _sha_size(path)
        state_files[name] = _stream_sha256(path)
    for action, path in npz_paths.items():
        key = f"{p08.REPRESENTATIONS_DIRNAME}/{action}.npz"
        artifact_files[key] = _sha_size(path)
        state_files[key] = _stream_sha256(path)

    artifact_path = p01 / p08.P01_ARTIFACT_MANIFEST_FILENAME
    artifact_path.write_text(json.dumps({"files": artifact_files}), encoding="utf-8")
    artifact_sha = _stream_sha256(artifact_path)
    state_files[p08.P01_ARTIFACT_MANIFEST_FILENAME] = artifact_sha

    state_path = p01 / p08.P01_STATE_FILENAME
    state_path.write_text(
        json.dumps(
            {
                "execution_status": "complete",
                "scientific_status": "pass",
                "files": state_files,
            }
        ),
        encoding="utf-8",
    )
    state_sha = _stream_sha256(state_path)

    input_audit = {
        "actions": [
            {"representation_id": action, "file_sha256": _stream_sha256(npz_paths[action])}
            for action in p08.ACTION_ORDER
        ]
    }
    input_audit_path = package / p08.INPUT_SUPPORT_AUDIT_REL
    input_audit_path.parent.mkdir(parents=True, exist_ok=True)
    input_audit_path.write_text(json.dumps(input_audit), encoding="utf-8")
    input_audit_sha = _stream_sha256(input_audit_path)
    monkeypatch.setattr(p08, "INPUT_SUPPORT_AUDIT_SHA256", input_audit_sha)

    catalog = _catalog()
    catalog_path = private / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    canonical = json.dumps(
        catalog, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    catalog_sha = hashlib.sha256(canonical).hexdigest()

    preservation_audit = {
        "input_hashes": {
            "input_audit": input_audit_sha,
            "P01_state": state_sha,
            "P01_artifact_manifest": artifact_sha,
            "primary_manifest": state_files[p08.P01_PRIMARY_MANIFEST_FILENAME],
            "preservation_metrics": state_files[p08.P01_PRESERVATION_METRICS_FILENAME],
            "preservation_by_instrument": state_files[
                p08.P01_PRESERVATION_BY_INSTRUMENT_FILENAME
            ],
            "preservation_source": _stream_sha256(source_path),
        },
        "private_catalog_sha256": catalog_sha,
    }
    preservation_audit_path = package / p08.PRESERVATION_REPORTING_AUDIT_REL
    preservation_audit_path.write_text(json.dumps(preservation_audit), encoding="utf-8")
    monkeypatch.setattr(
        p08,
        "PRESERVATION_REPORTING_AUDIT_SHA256",
        _stream_sha256(preservation_audit_path),
    )

    contexts_path = private / "contexts.csv"
    contexts_path.write_text(
        "context_id,domain,station,held_instrument,outer_repeat,outer_fold,"
        "phase_gate,experiment_id,outer_test_uid_sha256\n"
        "c0,spectral,S0,I0,0,0,held_evaluation,EXP-N00-T3,deadbeef\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        p08, "CONTEXT_PINS", {"contexts": ("contexts.csv", _stream_sha256(contexts_path))}
    )

    return {
        "tmp": tmp_path,
        "package": package,
        "p01": p01,
        "private": private,
        "catalog_path": catalog_path,
    }


@pytest.fixture()
def world(tmp_path, monkeypatch):
    return _build_world(tmp_path, monkeypatch)


def _load(world):
    return p08.load_preservation_inputs(
        p01_root=world["p01"],
        private_root=world["private"],
        catalog_path=world["catalog_path"],
        package_root=world["package"],
        allowed_evidence_roots=[world["p01"], world["private"], world["package"]],
        check=lambda stage: None,
    )


def _semantics(data):
    return p08.build_preservation_semantics(
        manifest=data["manifest"],
        metrics=data["metrics"],
        representations=data["representations"],
        catalog=data["catalog"],
        held_domains=data["held_domains"],
    )


def test_private_helper_roundtrip(world):
    roots = _prepare_roots([world["p01"], world["private"], world["package"]])
    assert roots
    state_path = world["p01"] / p08.P01_STATE_FILENAME
    authenticated = _authenticated_file(state_path, roots, _stream_sha256(state_path))
    assert _read_json_file(authenticated, roots)["execution_status"] == "complete"


def test_loader_roundtrip_schema(world):
    data = _load(world)
    assert data["held_domains"] == [("S0", "I0")]
    assert set(data["representations"]) == set(p08.ACTION_ORDER)
    assert set(data["catalog"]) >= {
        "spectral_figure_cells",
        "domain_memberships",
        "selected_preservation_rows",
    }
    assert data["provenance"]["held_domain_count"] == 1
    for action in p08.ACTION_ORDER:
        arrays = data["representations"][action]
        assert arrays["axis_cm1"].dtype == np.float32
        assert arrays["intensity"].dtype == np.float32
        assert arrays["observation_uid"].dtype.kind == "U"


def test_tampered_file_refused(world):
    metrics_path = world["p01"] / p08.P01_PRESERVATION_METRICS_FILENAME
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "junk\n", encoding="utf-8"
    )
    with pytest.raises(UniversalEvidenceError, match="hash_mismatch"):
        _load(world)


def test_build_semantics_support_and_cells(world):
    result = _semantics(_load(world))
    support = result["support"]
    assert (
        support["public_cells"],
        support["public_cells_eligible"],
        support["public_cells_unavailable"],
    ) == (3, 2, 1)
    assert support["public_curves"] == 6
    assert (support["held_domains"], support["exploratory_domains"]) == (1, 1)
    assert result["spectral"]["action_order"] == list(p08.ACTION_ORDER)
    cells = result["spectral"]["cells"]
    assert len(cells) == 3
    unavailable = [cell for cell in cells if not cell["available"]]
    assert len(unavailable) == 1 and unavailable[0]["reason"]
    assert sum(cell["held_comparison_domain"] for cell in cells) == 2
    for cell in cells:
        if cell["available"]:
            assert set(cell["curves"]) == set(p08.ACTION_ORDER)
            assert len(cell["curves"][p08.ACTION_ORDER[0]]) == p08.EXPECTED_FEATURES
        else:
            assert cell["curves"] == {}


def test_master_equal_is_not_row_mean():
    intensity = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    reps = {action: {"intensity": intensity} for action in p08.ACTION_ORDER}
    record = {
        "master_groups": [
            {"master_id": "M0", "observation_uids": ("a", "b", "c")},
            {"master_id": "M1", "observation_uids": ("d",)},
        ]
    }
    positions = {"a": 0, "b": 1, "c": 2, "d": 3}
    curves = p08._cell_curves(record, reps, positions)
    assert curves[p08.ACTION_ORDER[0]] == pytest.approx([0.5, 0.5])
    assert curves[p08.ACTION_ORDER[0]] != pytest.approx([0.75, 0.25])


def test_all_metrics_saved_with_missing_and_linear_quantiles(world):
    data = _load(world)
    table = _semantics(data)["preservation"]
    assert list(table.columns) == list(p08.PRESERVATION_COLUMNS)
    assert set(table["metric"]) == set(p08.METRIC_COLUMNS)
    assert len(p08.METRIC_COLUMNS) == 11
    assert len(table) == p08.EXPECTED_DOMAIN_ACTION_GROUPS * 11
    row = table[
        (table.station == "S0")
        & (table.instrument == "I0")
        & (table.representation_id == p08.ACTION_ORDER[0])
        & (table.metric == "baseline_span")
    ].iloc[0]
    assert row["finite_count"] + row["undefined_count"] == row["n_spectra"]
    assert row["undefined_count"] == 1
    raw = data["metrics"]
    subset = raw[
        (raw.representation_id == p08.ACTION_ORDER[0])
        & (raw.station == "S0")
        & (raw.instrument == "I0")
    ]
    values = pd.to_numeric(subset["baseline_span"], errors="coerce").dropna().to_numpy(float)
    assert row["median"] == pytest.approx(float(np.median(values)))
    for key, quantile in (("q10", 0.1), ("q90", 0.9)):
        assert row[key] == pytest.approx(
            float(np.quantile(values, quantile, method="linear"))
        )


def _dup_cell_key(catalog, reps):
    catalog["spectral_figure_cells"][1].update(station="S0", instrument="I0", analyte="A0")


def _conflict_membership(catalog, reps):
    catalog["spectral_figure_cells"][0]["master_groups"][0]["observation_uids"][1] = "u3"


def _reverse_uids(catalog, reps):
    arrays = dict(reps[p08.ACTION_ORDER[0]])
    arrays["observation_uid"] = arrays["observation_uid"][::-1].copy()
    reps[p08.ACTION_ORDER[0]] = arrays


def _wrong_axis_values(catalog, reps):
    arrays = dict(reps[p08.ACTION_ORDER[0]])
    arrays["axis_cm1"] = (arrays["axis_cm1"] + np.float32(1)).astype(np.float32)
    reps[p08.ACTION_ORDER[0]] = arrays


def _wrong_axis_dtype(catalog, reps):
    arrays = dict(reps[p08.ACTION_ORDER[0]])
    arrays["axis_cm1"] = arrays["axis_cm1"].astype(np.float64)
    reps[p08.ACTION_ORDER[0]] = arrays


def _bad_intensity(catalog, reps):
    arrays = dict(reps[p08.ACTION_ORDER[0]])
    intensity = arrays["intensity"].copy()
    intensity[0, 0] = 2.0
    arrays["intensity"] = intensity
    reps[p08.ACTION_ORDER[0]] = arrays


@pytest.mark.parametrize(
    "mutate,reason",
    (
        (_dup_cell_key, "catalog_cell_key_duplicate"),
        (_conflict_membership, "catalog_membership_mismatch"),
        (_reverse_uids, "representation_uid_order"),
        (_wrong_axis_values, "representation_axis_invalid"),
        (_wrong_axis_dtype, "representation_axis_dtype_invalid"),
        (_bad_intensity, "representation_intensity_range"),
    ),
)
def test_semantic_refusals(world, mutate, reason):
    data = _load(world)
    catalog = copy.deepcopy(data["catalog"])
    reps = copy.deepcopy(data["representations"])
    mutate(catalog, reps)
    with pytest.raises(p08.PreservationSemanticError) as exc:
        p08.build_preservation_semantics(
            manifest=data["manifest"],
            metrics=data["metrics"],
            representations=reps,
            catalog=catalog,
            held_domains=data["held_domains"],
        )
    assert exc.value.reason_code == reason


def test_public_output_has_no_private_identities_or_paths(world):
    blob = json.dumps(_semantics(_load(world)), default=str)
    for token in ("u0", "u1", "u2", "u3", "u4", "u5", "u6", "M0", "M1", "M2", "M3", "M4"):
        assert token not in blob
    assert str(world["tmp"]) not in blob
