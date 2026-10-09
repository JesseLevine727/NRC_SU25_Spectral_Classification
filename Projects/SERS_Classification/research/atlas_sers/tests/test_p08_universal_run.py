"""Tests for the T323 universal analysis runner and bundle codec.

The tests use only the three imported implementation seams (loader, assembler,
analyzer).  They replace them with tiny fakes and monkeypatch the declared
dimension constants; they never read real scientific data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p08_universal_run as run


def _guard_recorder():
    events = []

    def check(label):
        events.append(label)

    return check, events


def _assert_faithful(restored, original, where="root"):
    if isinstance(original, pd.DataFrame):
        assert isinstance(restored, pd.DataFrame), where
        pd.testing.assert_frame_equal(restored, original, check_dtype=True)
        return
    if isinstance(original, pd.Series):
        assert isinstance(restored, pd.Series), where
        pd.testing.assert_series_equal(restored, original, check_dtype=True)
        return
    if isinstance(original, np.ndarray):
        assert isinstance(restored, np.ndarray), where
        assert restored.dtype == original.dtype, where
        assert restored.shape == original.shape, where
        if original.dtype.kind in ("f", "c"):
            np.testing.assert_array_equal(restored, original, err_msg=where)
        else:
            assert np.array_equal(restored, original), where
        return
    if original is pd.NA:
        assert restored is pd.NA, where
        return
    if original is pd.NaT:
        assert restored is pd.NaT, where
        return
    if isinstance(original, Mapping):
        assert isinstance(restored, Mapping), where
        assert set(restored) == set(original), where
        for key in original:
            _assert_faithful(restored[key], original[key], where + "." + repr(key))
        return
    if isinstance(original, tuple):
        assert isinstance(restored, tuple), where
        assert len(restored) == len(original), where
        for position, (left, right) in enumerate(zip(restored, original, strict=True)):
            _assert_faithful(left, right, where + "." + str(position))
        return
    if isinstance(original, list):
        assert isinstance(restored, list), where
        assert len(restored) == len(original), where
        for position, (left, right) in enumerate(zip(restored, original, strict=True)):
            _assert_faithful(left, right, where + "." + str(position))
        return
    if isinstance(original, (float, np.floating)):
        assert isinstance(restored, (float, np.floating)), where
        if math.isnan(original):
            assert math.isnan(restored), where
        else:
            assert restored == original, where
        return
    assert restored == original, where


def _tiny_constants(monkeypatch):
    monkeypatch.setattr(run, "EXPECTED_REPORT_CELLS", 2)
    monkeypatch.setattr(run, "EXPECTED_DISTINCT_ENDPOINTS", 2)
    monkeypatch.setattr(run, "EXPECTED_GLOBAL_MASTERS", 2)
    monkeypatch.setattr(run, "EXPECTED_GLOBAL_INSTRUMENTS", 2)
    monkeypatch.setattr(run, "EXPECTED_DRAW_COUNT", 3)
    monkeypatch.setattr(run, "EXPECTED_MASTER_SEED", 1)
    monkeypatch.setattr(run, "EXPECTED_INSTRUMENT_SEED", 2)
    monkeypatch.setattr(run, "EXPECTED_HIERARCHY_SEED", 3)


def _fake_evidence():
    return {
        "manifest": pd.DataFrame(),
        "contexts": pd.DataFrame(),
        "roles": pd.DataFrame(),
        "jobs": [],
        "aliases": {},
        "endpoint_frames": {},
        "training_fit_summaries": {},
        "diagnostics": {
            "graph_sha256": "graph",
            "graph_plan_sha256": "plan",
            "min_bridge_sha256": "bridge",
            "report_cells": run.EXPECTED_REPORT_CELLS,
            "distinct_endpoints": run.EXPECTED_DISTINCT_ENDPOINTS,
            "outer_reverification_required": True,
        },
    }


def _fake_assembly():
    coverage = pd.DataFrame(
        {
            "context_id": ["c1", "c2"],
            "policy_id": ["p", "p"],
            "model_id": ["m", "m"],
            "endpoint_job_id": ["j1", "j2"],
            "recipe_id": ["r", "r"],
            "reference_count": [1, 1],
            "unique_endpoint": [True, True],
            "expected_n_spectra": [2, 2],
            "actual_n_spectra": [2, 2],
            "expected_n_masters": [1, 1],
            "actual_n_masters": [1, 1],
            "complete": [True, True],
        }
    )
    endpoint_index = pd.DataFrame(
        {
            "context_id": ["c1", "c2"],
            "policy_id": ["p", "p"],
            "model_id": ["m", "m"],
            "endpoint_job_id": ["j1", "j2"],
            "recipe_id": ["r", "r"],
        }
    )
    return {
        "registered_test_rows": pd.DataFrame(),
        "predictions": pd.DataFrame(),
        "coverage": coverage,
        "global_masters": ["m1", "m2"],
        "global_instruments": ["i1", "i2"],
        "domain_families": {"domain": "family"},
        "endpoint_index": endpoint_index,
        "panels": {"a": {"p": {"e": pd.DataFrame({"x": [1.0]})}}},
    }


def _fake_analysis():
    boundary = {label: True for label in run.BOUNDARY_FLAGS}
    result = {
        "weights": {
            "number_draws": run.EXPECTED_DRAW_COUNT,
            "master_seed": run.EXPECTED_MASTER_SEED,
            "instrument_seed": run.EXPECTED_INSTRUMENT_SEED,
            "hierarchy_seed": run.EXPECTED_HIERARCHY_SEED,
            "global_masters": ["m1", "m2"],
            "global_instruments": ["i1", "i2"],
            "master_weights": np.zeros((run.EXPECTED_DRAW_COUNT, run.EXPECTED_GLOBAL_MASTERS)),
            "instrument_weights": np.zeros(
                (run.EXPECTED_DRAW_COUNT, run.EXPECTED_GLOBAL_INSTRUMENTS)
            ),
            "generated_once": "shared",
        },
        "metrics": {},
        "contrasts": {},
        "contrast_summary": {},
        "hierarchy_support": {},
        "boundary": dict(boundary),
    }
    for label in run.BOUNDARY_FLAGS:
        result[label] = True
    return result


def _patch_pipeline(monkeypatch, loader=None, assembler=None, analyzer=None):
    monkeypatch.setattr(run, "load_evidence", loader or (lambda **kwargs: _fake_evidence()))
    monkeypatch.setattr(run, "assemble_panel", assembler or (lambda **kwargs: _fake_assembly()))
    monkeypatch.setattr(
        run, "analyze_panel", analyzer or (lambda panels, **kwargs: _fake_analysis())
    )


def test_bundle_mixed_roundtrip_preserves_values(tmp_path):
    check, _events = _guard_recorder()
    frame = pd.DataFrame(
        {
            "f": pd.Series([1.5, np.nan, np.inf, -np.inf], dtype="float64"),
            "o": pd.Series([None, np.nan, ("a", 1), "text"], dtype="object"),
            "n": pd.Series([1, pd.NA, 3, 4], dtype="Int64"),
            "s": pd.Series(["x", pd.NA, "z", "w"], dtype="string"),
            "b": pd.Series([True, False, True, False], dtype="bool"),
        }
    )
    frame.index = pd.RangeIndex(4, name="row")
    payload = {
        "evidence_diagnostics": {
            "graph_sha256": "g",
            "tuple": ("a", 1),
            "list": [1, 2],
            "none": None,
            "pdna": pd.NA,
            "nat": pd.NaT,
            "nan": float("nan"),
            "pinf": float("inf"),
            "ninf": float("-inf"),
            "npint": np.int64(7),
            "npfloat": np.float64(2.5),
        },
        "panel": {"frame": frame, "arr": np.array([1.0, np.nan, -np.inf])},
        "analysis": {"weights": {"number_draws": np.int64(3)}},
    }
    info = run.write_bundle(tmp_path / "bundle", payload, check)
    restored = run.read_bundle(
        tmp_path / "bundle", expected_manifest_sha256=info["manifest_sha256"]
    )

    diagnostics = restored["evidence_diagnostics"]
    assert diagnostics["tuple"] == ("a", 1) and isinstance(diagnostics["tuple"], tuple)
    assert diagnostics["list"] == [1, 2] and isinstance(diagnostics["list"], list)
    assert diagnostics["none"] is None
    assert diagnostics["pdna"] is pd.NA
    assert diagnostics["nat"] is pd.NaT
    assert np.isnan(diagnostics["nan"])
    assert diagnostics["pinf"] == float("inf")
    assert diagnostics["ninf"] == float("-inf")
    assert diagnostics["npint"] == 7
    assert diagnostics["npfloat"] == 2.5

    pd.testing.assert_frame_equal(restored["panel"]["frame"], frame, check_dtype=True)
    array = restored["panel"]["arr"]
    assert np.isnan(array[1]) and np.isneginf(array[2])
    assert restored["analysis"]["weights"]["number_draws"] == 3

    for path in (tmp_path / "bundle").rglob("*.npy"):
        assert path.read_bytes().startswith(b"\x93NUMPY")


def test_object_array_and_unsupported_type_rejected(tmp_path):
    check, _events = _guard_recorder()
    with pytest.raises(run._BundleError):
        run.write_bundle(
            tmp_path / "b1", {"a": np.array([{"x": 1}], dtype=object)}, check
        )
    with pytest.raises(run._BundleError):
        run.write_bundle(tmp_path / "b2", {"a": {1, 2}}, check)
    with pytest.raises(run._BundleError):
        run.write_bundle(tmp_path / "b3", {1: "nonstring"}, check)


def test_inventory_tamper_unlisted_and_symlink_rejected(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0, 2.0])}, check)
    target = next(root.rglob("*.npy"))
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])

    root2 = tmp_path / "bundle2"
    info2 = run.write_bundle(root2, {"a": np.array([1.0, 2.0])}, check)
    (root2 / "payload" / "unlisted.bin").write_bytes(b"x")
    with pytest.raises(run._BundleError):
        run.read_bundle(root2, expected_manifest_sha256=info2["manifest_sha256"])

    root3 = tmp_path / "bundle3"
    info3 = run.write_bundle(root3, {"a": np.array([1.0, 2.0])}, check)
    listed = next(root3.rglob("*.npy"))
    outside = tmp_path / "outside.npy"
    outside.write_bytes(listed.read_bytes())
    listed.unlink()
    os.symlink(outside, listed)
    with pytest.raises(run._BundleError):
        run.read_bundle(root3, expected_manifest_sha256=info3["manifest_sha256"])


def test_manifest_path_traversal_rejected(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    run.write_bundle(root, {"a": 1}, check)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../evil.npy"] = {"size": 1, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256=pin)


def test_existing_output_rejected_before_loader(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    loaded = []
    _patch_pipeline(
        monkeypatch,
        loader=lambda **kwargs: loaded.append(True) or _fake_evidence(),
    )
    check, events = _guard_recorder()
    with pytest.raises(run._BundleError):
        run.run_analysis(evidence_kwargs={}, output=output, check=check)
    assert loaded == []
    assert events == [run.CHECK_INITIAL]
    assert not (output / run.ERROR_NAME).exists()


def test_destination_inside_completed_run_rejected_before_loader(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    run_root = tmp_path / "run"
    run_root.mkdir()
    output = run_root / "out"
    monkeypatch.setattr(run, "COMPLETED_RUN_ROOT", run_root)
    loaded = []
    _patch_pipeline(
        monkeypatch,
        loader=lambda **kwargs: loaded.append(True) or _fake_evidence(),
    )
    check, events = _guard_recorder()
    with pytest.raises(run._BundleError):
        run.run_analysis(evidence_kwargs={}, output=output, check=check)
    assert loaded == []
    assert events == [run.CHECK_INITIAL]


def test_destination_under_private_root_sibling_allowed(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    private_root = tmp_path / "private"
    private_root.mkdir()
    run_root = private_root / "inputs"
    run_root.mkdir()
    outputs = private_root / "outputs"
    outputs.mkdir()
    output = outputs / "run1"
    monkeypatch.setattr(run, "COMPLETED_RUN_ROOT", run_root)
    _patch_pipeline(monkeypatch)
    check, _events = _guard_recorder()
    receipt = run.run_analysis(evidence_kwargs={}, output=output, check=check)
    assert receipt["status"] == "success"
    assert output.is_dir()


def test_pipeline_stage_order_and_boundaries(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    order = []

    def loader(**kwargs):
        order.append("load")
        return _fake_evidence()

    def assembler(**kwargs):
        order.append("assemble")
        return _fake_assembly()

    def analyzer(panels, **kwargs):
        order.append("analyze")
        assert kwargs["global_masters"] == ["m1", "m2"]
        assert kwargs["global_instruments"] == ["i1", "i2"]
        assert kwargs["domain_families"] == {"domain": "family"}
        return _fake_analysis()

    _patch_pipeline(monkeypatch, loader, assembler, analyzer)
    output = tmp_path / "out"
    check, events = _guard_recorder()
    receipt = run.run_analysis(evidence_kwargs={}, output=output, check=check)

    assert order == ["load", "assemble", "analyze"]
    assert events[0] == run.CHECK_INITIAL
    assert run.CHECK_STAGE_LOAD in events
    assert run.CHECK_STAGE_ASSEMBLE in events
    assert run.CHECK_STAGE_ANALYZE in events
    assert events.index(run.CHECK_STAGE_LOAD) < events.index(run.CHECK_STAGE_ASSEMBLE)
    assert events.index(run.CHECK_STAGE_ASSEMBLE) < events.index(run.CHECK_STAGE_ANALYZE)
    assert run.CHECK_STAGE_ANALYZE_DONE in events
    assert run.CHECK_BUNDLE_DONE in events
    assert events.index(run.CHECK_STAGE_ANALYZE) < events.index(
        run.CHECK_STAGE_ANALYZE_DONE
    )
    assert events.index(run.CHECK_STAGE_ANALYZE_DONE) < events.index(
        run.CHECK_BUNDLE_DONE
    )
    assert events.index(run.CHECK_BUNDLE_DONE) < events.index(run.CHECK_RECEIPT)
    assert any(event.startswith(run.CHECK_PAYLOAD_PREFIX) for event in events)
    assert run.CHECK_MANIFEST in events
    assert events[-1] == run.CHECK_RECEIPT
    assert (output / run.RECEIPT_NAME).is_file()
    assert (output / run.BUNDLE_DIRNAME / run.MANIFEST_NAME).is_file()
    assert stat.S_IMODE((output / run.RECEIPT_NAME).stat().st_mode) == 0o600
    assert stat.S_IMODE((output / run.BUNDLE_DIRNAME).stat().st_mode) == 0o700
    restored = run.read_bundle(
        output / run.BUNDLE_DIRNAME,
        expected_manifest_sha256=receipt["bundle"]["manifest_sha256"],
    )
    assert set(restored["analysis"]) == set(_fake_analysis())
    for label in run.BOUNDARY_FLAGS:
        assert receipt["boundary_flags"][label] is True
    assert receipt["outer_reverification_required"] is True
    assert receipt["no_automatic_superiority_claim"] is True
    assert receipt["bundle"]["manifest_sha256"]


def test_wrong_draws_refuse_and_no_success_receipt(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    bad = _fake_analysis()
    bad["weights"]["number_draws"] = 999
    _patch_pipeline(monkeypatch, analyzer=lambda panels, **kwargs: bad)
    output = tmp_path / "out"
    check, _events = _guard_recorder()
    with pytest.raises(run._BundleError):
        run.run_analysis(evidence_kwargs={}, output=output, check=check)
    assert not (output / run.RECEIPT_NAME).exists()
    assert (output / run.ERROR_NAME).is_file()


def test_partial_failure_preserves_files(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)

    def boom(panels, **kwargs):
        raise RuntimeError("boom")

    _patch_pipeline(monkeypatch, analyzer=boom)
    output = tmp_path / "out"
    check, _events = _guard_recorder()
    with pytest.raises(RuntimeError):
        run.run_analysis(evidence_kwargs={}, output=output, check=check)
    assert output.is_dir()
    assert (output / run.ERROR_NAME).is_file()
    assert stat.S_IMODE((output / run.ERROR_NAME).stat().st_mode) == 0o600
    assert not (output / run.RECEIPT_NAME).exists()


def test_guard_failure_and_non_callable_guard(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    loaded = []
    _patch_pipeline(
        monkeypatch,
        loader=lambda **kwargs: loaded.append(True) or _fake_evidence(),
    )
    output = tmp_path / "out"

    def failing_check(label):
        raise RuntimeError("guard refused " + label)

    with pytest.raises(RuntimeError):
        run.run_analysis(evidence_kwargs={}, output=output, check=failing_check)
    assert loaded == []
    assert not output.exists()

    with pytest.raises(TypeError):
        run.run_analysis(evidence_kwargs={}, output=output, check=None)


def test_boundary_flag_names_are_exact():
    assert set(run.BOUNDARY_FLAGS) == {
        "conditional_on_saved_fits_observed_support",
        "descriptive_sign_symmetry_not_randomized",
        "no_G4_decision",
        "no_model_or_policy_selection",
    }


def test_write_bundle_refuses_existing_root_preserving_bytes_and_modes(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0, 2.0])}, check)
    manifest_path = root / "manifest.json"
    original = manifest_path.read_bytes()
    manifest_mode = stat.S_IMODE(manifest_path.stat().st_mode)
    payload_modes = {
        path: stat.S_IMODE(path.stat().st_mode) for path in root.rglob("*.npy")
    }

    with pytest.raises(run._BundleError):
        run.write_bundle(root, {"a": np.array([3.0, 4.0])}, check)

    assert manifest_path.read_bytes() == original
    assert stat.S_IMODE(manifest_path.stat().st_mode) == manifest_mode
    for path, mode in payload_modes.items():
        assert stat.S_IMODE(path.stat().st_mode) == mode
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    assert restored["a"].tolist() == [1.0, 2.0]


def test_bundle_directory_and_file_modes_are_private(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0, 2.0])}, check)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "payload").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "manifest.json").stat().st_mode) == 0o600
    for path in root.rglob("*.npy"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(info["files"]) >= 1


def test_write_bundle_refuses_symlink_parent(tmp_path):
    check, _events = _guard_recorder()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(run._BundleError):
        run.write_bundle(link / "bundle", {"a": 1}, check)
    assert not (real / "bundle").exists()


def test_read_bundle_requires_authenticated_pin_and_invokes_check(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0, 2.0])}, check)

    with pytest.raises(TypeError):
        run.read_bundle(root)

    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256="0" * 64)

    read_check, events = _guard_recorder()
    restored = run.read_bundle(
        root,
        expected_manifest_sha256=info["manifest_sha256"],
        check=read_check,
    )
    assert restored["a"].tolist() == [1.0, 2.0]
    assert run.CHECK_MANIFEST in events
    assert any(event.startswith(run.CHECK_PAYLOAD_PREFIX) for event in events)


def test_read_bundle_rejects_symlink_ancestor(tmp_path):
    check, _events = _guard_recorder()
    real = tmp_path / "real"
    real.mkdir()
    info = run.write_bundle(real / "bundle", {"a": np.array([1.0])}, check)
    link = tmp_path / "linked"
    os.symlink(real, link)
    with pytest.raises(run._BundleError):
        run.read_bundle(
            link / "bundle", expected_manifest_sha256=info["manifest_sha256"]
        )


def test_manifest_scalar_and_descriptor_tamper_with_old_pin_refused(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0, 2.0])}, check)
    manifest_path = root / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    manifest = json.loads(original_manifest)
    file_key = next(iter(manifest["files"]))
    manifest["files"][file_key]["size"] = manifest["files"][file_key]["size"] + 1
    manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])

    manifest = json.loads(original_manifest)
    manifest["payload"] = {"t": "none"}
    manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])


def test_object_tuple_column_roundtrip_preserves_one_dimensional_tuples(tmp_path):
    check, _events = _guard_recorder()
    frame = pd.DataFrame({"pair": pd.Series([("a", 1), ("b", 2)], dtype=object)})
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"frame": frame}, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    column = restored["frame"]["pair"]
    assert column.dtype == object
    assert column.tolist() == [("a", 1), ("b", 2)]
    assert all(isinstance(value, tuple) for value in column)


def test_object_index_tuple_and_float_none_nan_roundtrip(tmp_path):
    check, _events = _guard_recorder()
    tuple_values = np.empty(2, dtype=object)
    tuple_values[0] = ("a", 1)
    tuple_values[1] = ("b", 2)
    tuple_index = pd.Index(tuple_values, dtype=object, name="tuple_key")
    root = tmp_path / "bundle"
    info = run.write_bundle(
        root, {"frame": pd.DataFrame({"v": [1, 2]}, index=tuple_index)}, check
    )
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    index = restored["frame"].index
    assert index.dtype == object
    assert index.tolist() == [("a", 1), ("b", 2)]
    assert all(isinstance(value, tuple) for value in index)

    float_index = pd.Index(
        np.array([1.5, None, float("nan")], dtype=object), dtype=object, name="f"
    )
    root2 = tmp_path / "bundle2"
    info2 = run.write_bundle(
        root2, {"frame": pd.DataFrame({"v": [1, 2, 3]}, index=float_index)}, check
    )
    restored2 = run.read_bundle(root2, expected_manifest_sha256=info2["manifest_sha256"])
    index2 = restored2["frame"].index
    assert index2.dtype == object
    assert index2[0] == 1.5
    assert index2[1] is None
    assert np.isnan(index2[2])


def test_full_precision_float_roundtrip(tmp_path):
    check, _events = _guard_recorder()
    values = [0.1 + 0.2, 1.2345678901234567, -9.876543210987654e-12]
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"values": values}, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    assert restored["values"] == values
    assert repr(restored["values"][0]) == repr(values[0])


def _training_job(job_hex, policy_id, model_id, stage):
    return {
        "job_id": "P08JOB-" + job_hex,
        "policy_id": policy_id,
        "model_id": model_id,
        "stage": stage,
        "graph_sha256": job_hex,
        "outer_repeat": 1,
    }


def _valid_training_summaries():
    job_a = _training_job("1" * 64, "PP-U-SG", "D0-M", "source_fit")
    job_b = _training_job("2" * 64, "PP-U-ARPLS", "D3", "final_refit")
    return {
        job_b["job_id"]: {
            "job": job_b,
            "summary": {
                "scores": [0.25, 0.75, 0.1234567890123456],
                "meta": {"count": 3, "nested": {"label": "final_refit"}},
            },
            "summary_sha256": "b" * 64,
            "receipt_sha256": "c" * 64,
        },
        job_a["job_id"]: {
            "job": job_a,
            "summary": {"accuracy": 0.5, "labels": ["a", "b"]},
            "summary_sha256": "d" * 64,
            "receipt_sha256": "e" * 64,
        },
    }


_TRAINING_MALFORMED_CASES = (
    "not_mapping",
    "entry_not_mapping",
    "entry_extra_key",
    "entry_missing_key",
    "job_key_prefix",
    "job_key_digest",
    "job_id_mismatch",
    "job_not_mapping",
    "summary_not_mapping",
    "summary_sha_uppercase",
    "summary_sha_short",
    "receipt_sha_not_hex",
    "policy_classic",
    "policy_min",
    "model_classic",
    "stage_unknown",
)


def _malformed_training_summaries(case):
    job = _training_job("a" * 64, "PP-U-SG", "D0-M", "source_fit")
    entry = {
        "job": job,
        "summary": {"score": 0.5},
        "summary_sha256": "b" * 64,
        "receipt_sha256": "c" * 64,
    }
    if case == "not_mapping":
        return ["not", "a", "mapping"]
    if case == "entry_not_mapping":
        return {job["job_id"]: ["not", "a", "mapping"]}
    if case == "entry_extra_key":
        broken = dict(entry)
        broken["extra"] = 1
        return {job["job_id"]: broken}
    if case == "entry_missing_key":
        broken = dict(entry)
        del broken["receipt_sha256"]
        return {job["job_id"]: broken}
    if case == "job_key_prefix":
        return {"JOB-" + "a" * 64: entry}
    if case == "job_key_digest":
        return {"P08JOB-" + "A" * 64: entry}
    if case == "job_id_mismatch":
        other = _training_job("9" * 64, "PP-U-SG", "D0-M", "source_fit")
        return {
            job["job_id"]: {
                "job": other,
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "job_not_mapping":
        return {
            job["job_id"]: {
                "job": "not a mapping",
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "summary_not_mapping":
        return {
            job["job_id"]: {
                "job": job,
                "summary": [1, 2],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "summary_sha_uppercase":
        return {
            job["job_id"]: {
                "job": job,
                "summary": entry["summary"],
                "summary_sha256": "B" * 64,
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "summary_sha_short":
        return {
            job["job_id"]: {
                "job": job,
                "summary": entry["summary"],
                "summary_sha256": "b" * 63,
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "receipt_sha_not_hex":
        return {
            job["job_id"]: {
                "job": job,
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": "z" * 64,
            }
        }
    if case in {"policy_classic", "policy_min"}:
        policy_id = "C-RBF-SVM" if case == "policy_classic" else "PP-U-MIN"
        bad_job = _training_job("a" * 64, policy_id, "D0-M", "source_fit")
        return {
            bad_job["job_id"]: {
                "job": bad_job,
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "model_classic":
        bad_job = _training_job("a" * 64, "PP-U-SG", "C-RANDOM-FOREST", "source_fit")
        return {
            bad_job["job_id"]: {
                "job": bad_job,
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    if case == "stage_unknown":
        bad_job = _training_job("a" * 64, "PP-U-SG", "D0-M", "classic_fit")
        return {
            bad_job["job_id"]: {
                "job": bad_job,
                "summary": entry["summary"],
                "summary_sha256": entry["summary_sha256"],
                "receipt_sha256": entry["receipt_sha256"],
            }
        }
    raise AssertionError("unknown training case: " + case)


def test_training_evidence_snapshot_persisted_private(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    summaries = _valid_training_summaries()
    expected_jobs = [summaries[key]["job"] for key in sorted(summaries)]
    evidence = _fake_evidence()
    evidence["training_fit_summaries"] = summaries
    evidence["jobs"] = expected_jobs

    def loader(**kwargs):
        return evidence

    _patch_pipeline(monkeypatch, loader)
    output = tmp_path / "out"
    check, _events = _guard_recorder()
    receipt = run.run_analysis(evidence_kwargs={}, output=output, check=check)

    restored = run.read_bundle(
        output / run.BUNDLE_DIRNAME,
        expected_manifest_sha256=receipt["bundle"]["manifest_sha256"],
    )
    snapshot = restored["training_evidence"]
    assert set(snapshot) == {
        "schema_version",
        "jobs",
        "training_fit_summaries",
        "diagnostics",
        "private_only",
        "no_publication",
    }
    assert snapshot["schema_version"] == "nato-sers-p08-training-evidence-snapshot-v1"
    assert snapshot["private_only"] is True
    assert snapshot["no_publication"] is True
    assert snapshot["jobs"] == expected_jobs
    assert snapshot["training_fit_summaries"] == summaries
    assert snapshot["diagnostics"] == evidence["diagnostics"]
    assert set(restored) == {
        "evidence_diagnostics",
        "panel",
        "analysis",
        "training_evidence",
    }
    assert evidence["training_fit_summaries"] == _valid_training_summaries()


@pytest.mark.parametrize("case", _TRAINING_MALFORMED_CASES)
def test_malformed_training_bindings_refused(case, tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    evidence = _fake_evidence()
    evidence["training_fit_summaries"] = _malformed_training_summaries(case)

    def loader(**kwargs):
        return evidence

    _patch_pipeline(monkeypatch, loader)
    check, _events = _guard_recorder()
    with pytest.raises(run._BundleError, match="training_evidence_"):
        run.run_analysis(evidence_kwargs={}, output=tmp_path / "out", check=check)


def test_missing_training_bindings_refused(tmp_path, monkeypatch):
    _tiny_constants(monkeypatch)
    evidence = _fake_evidence()
    del evidence["training_fit_summaries"]

    def loader(**kwargs):
        return evidence

    _patch_pipeline(monkeypatch, loader)
    check, _events = _guard_recorder()
    with pytest.raises(run._BundleError, match="training_evidence_bindings_missing"):
        run.run_analysis(evidence_kwargs={}, output=tmp_path / "out", check=check)


def test_training_snapshot_cannot_hide_missing_graph_fit():
    evidence = _fake_evidence()
    evidence["jobs"] = [_training_job("1" * 64, "PP-U-SG", "D0-M", "source_fit")]
    with pytest.raises(run._BundleError, match="training_evidence_graph_coverage_mismatch"):
        run._snapshot_training_evidence(evidence, evidence["diagnostics"])


def test_training_snapshot_requires_exact_graph_fields_and_typed_keys():
    for field in ("policy_id", "model_id", "stage"):
        summaries = _valid_training_summaries()
        job = next(iter(summaries.values()))["job"]
        alias = "job_stage" if field == "stage" else field.removesuffix("_id")
        job[alias] = job.pop(field)
        with pytest.raises(run._BundleError, match="training_evidence_"):
            run._validate_training_fit_summaries(summaries)
    with pytest.raises(run._BundleError, match="training_evidence_job_key_invalid"):
        run._validate_training_fit_summaries({1: {}, "P08JOB-" + "a" * 64: {}})


def test_real_synthetic_analysis_bundle_roundtrip_preserves_all_keys(
    tmp_path, monkeypatch
):
    from atlas_sers.evaluation import p08_universal_analysis as analysis_mod
    from atlas_sers.evaluation import p08_universal_units as units

    monkeypatch.setattr(analysis_mod, "DRAW_COUNT", 4)

    vocabulary = ("A", "B", "C")
    specs = (
        ("c1", 1, "m1", "A", "o1"),
        ("c1", 1, "m2", "B", "o2"),
        ("c2", 2, "m3", "A", "o3"),
        ("c2", 2, "m4", "B", "o4"),
        ("c3", 3, "m5", "A", "o5"),
        ("c3", 3, "m6", "B", "o6"),
        ("c4", 4, "m7", "A", "o7"),
        ("c4", 4, "m8", "B", "o8"),
    )
    registered = pd.DataFrame(
        [
            {
                "context_id": context_id,
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "outer_repeat": 1,
                "outer_fold": fold,
                "observation_uid": observation,
                "master_sample_id": master,
                "true_label": label,
                "class_vocabulary": vocabulary,
            }
            for context_id, fold, master, label, observation in specs
        ]
    )
    policy_shift = {"PP-U-MIN": 0, "PP-U-SG": 1, "PP-U-ARPLS": 2}
    model_shift = {
        "C-RBF-SVM": 0,
        "C-RANDOM-FOREST": 1,
        "C-EXTRA-TREES": 2,
        "D0-M": 0,
        "P05-SELECTED": 0,
    }
    prediction_rows = []
    for record in registered.itertuples(index=False):
        vocabulary_tuple = tuple(record.class_vocabulary)
        true_index = vocabulary_tuple.index(record.true_label)
        for policy_id in units.POLICIES:
            for model_id in units.MODELS:
                choice = (
                    true_index + policy_shift[policy_id] + model_shift[model_id]
                ) % 3
                probabilities = [0.1, 0.1, 0.1]
                probabilities[choice] = 0.8
                prediction_rows.append(
                    {
                        "context_id": record.context_id,
                        "policy_id": policy_id,
                        "model_id": model_id,
                        "observation_uid": record.observation_uid,
                        "master_sample_id": record.master_sample_id,
                        "instrument": record.instrument,
                        "station": record.station,
                        "true_label": record.true_label,
                        "class_vocabulary": vocabulary_tuple,
                        "probability_0": probabilities[0],
                        "probability_1": probabilities[1],
                        "probability_2": probabilities[2],
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    panels = units.build_units(predictions, registered)

    global_masters = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    result = analysis_mod.analyze_panel(
        panels,
        global_masters=global_masters,
        global_instruments=["i1"],
        domain_families={"d1": "unknown"},
    )
    assert {"metrics", "contrasts", "weights", "registry"} <= set(result)

    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"analysis": result}, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])

    decoded = restored["analysis"]
    _assert_faithful(decoded, result)


def test_string_dtype_variants_roundtrip_with_sentinels(tmp_path):
    check, _events = _guard_recorder()
    frame = pd.DataFrame(
        {
            "explicit": pd.Series(["x", pd.NA, "z"], dtype="string"),
            "inferred": pd.Series(["x", np.nan, "z"]),
        }
    )
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"frame": frame}, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    pd.testing.assert_frame_equal(restored["frame"], frame, check_dtype=True)


def test_index_extension_and_numeric_dtypes_preserved(tmp_path):
    check, _events = _guard_recorder()
    string_index = pd.Index(
        pd.array(["a", pd.NA, "c"], dtype=pd.StringDtype()), name="s"
    )
    int_index = pd.Index(pd.array([1, pd.NA, 3], dtype="Int64"), name="i")
    numeric_index = pd.Index([1.5, 2.5, 3.5], name="n")
    payload = {
        "string": pd.DataFrame({"v": [1, 2, 3]}, index=string_index),
        "int": pd.DataFrame({"v": [1, 2, 3]}, index=int_index),
        "numeric": pd.DataFrame({"v": [1, 2, 3]}, index=numeric_index),
    }
    root = tmp_path / "bundle"
    info = run.write_bundle(root, payload, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    pd.testing.assert_frame_equal(restored["string"], payload["string"], check_dtype=True)
    pd.testing.assert_frame_equal(restored["int"], payload["int"], check_dtype=True)
    pd.testing.assert_frame_equal(restored["numeric"], payload["numeric"], check_dtype=True)


def test_tuple_key_mapping_roundtrip_and_rejections(tmp_path):
    check, _events = _guard_recorder()
    payload = {"contrasts": {("c1", "A"): -1.0, ("c2", "B"): 2.5, "plain": 3}}
    root = tmp_path / "bundle"
    info = run.write_bundle(root, payload, check)
    restored = run.read_bundle(root, expected_manifest_sha256=info["manifest_sha256"])
    contrasts = restored["contrasts"]
    assert set(contrasts) == {("c1", "A"), ("c2", "B"), "plain"}
    assert contrasts[("c1", "A")] == -1.0
    assert contrasts[("c2", "B")] == 2.5
    assert contrasts["plain"] == 3

    with pytest.raises(run._BundleError):
        run.write_bundle(tmp_path / "b1", {("a", 1): 2.0}, check)
    with pytest.raises(run._BundleError):
        run.write_bundle(tmp_path / "b2", {1: "nonstring"}, check)
    with pytest.raises(run._BundleError):
        run.write_bundle(tmp_path / "b3", {("a", ("b",)): 1.0}, check)


def test_manifest_tuple_key_duplicate_rejected(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    run.write_bundle(root, {"a": 1}, check)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key = {"t": "tuple", "items": [{"t": "str", "v": "c1"}, {"t": "str", "v": "A"}]}
    manifest["payload"] = {
        "t": "dict",
        "items": [
            [key, {"t": "float", "v": "1.0"}],
            [key, {"t": "float", "v": "2.0"}],
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(run._BundleError):
        run.read_bundle(root, expected_manifest_sha256=pin)


def test_write_bundle_guard_fails_before_mkdir(tmp_path):
    def failing(label):
        raise RuntimeError("refused " + label)

    root = tmp_path / "bundle"
    with pytest.raises(RuntimeError):
        run.write_bundle(root, {"a": 1}, failing)
    assert not root.exists()


def test_read_bundle_guard_fails_before_filesystem_read(tmp_path):
    check, _events = _guard_recorder()
    root = tmp_path / "bundle"
    info = run.write_bundle(root, {"a": np.array([1.0])}, check)

    def failing(label):
        raise RuntimeError("refused " + label)

    with pytest.raises(RuntimeError):
        run.read_bundle(
            root,
            expected_manifest_sha256=info["manifest_sha256"],
            check=failing,
        )
