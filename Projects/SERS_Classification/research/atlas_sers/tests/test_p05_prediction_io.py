"""Tests for P05 prediction I/O: endpoint indexing and durable persistence."""

from __future__ import annotations

# Optional torch is required by the canonical specification checker.
# ruff: noqa: E402
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_prediction_io as prediction_io
from tests.test_p05_prediction import _make_spec

SPECS = [_make_spec("D0-M"), _make_spec("D3")]
R1, R2 = SPECS[0]["refit_id"], SPECS[1]["refit_id"]
UIDS = ["001", "NA"]


def _code(exc):
    return getattr(exc, "reason_code", None) or str(exc)


def _plan(endpoints=None, specs=None, aliases=None):
    return {
        "endpoints": [{"context_id": "ctx-1"}] if endpoints is None else endpoints,
        "unique_refits": {R1: SPECS[0], R2: SPECS[1]} if specs is None else specs,
        "strategy_aliases": (
            [{"refit_id": R1}, {"refit_id": R2}, {"refit_id": R1}] if aliases is None else aliases
        ),
    }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(prediction_io, "EXPECTED_CONTEXTS", 1)
    monkeypatch.setattr(prediction_io, "EXPECTED_ALIASES", 3)


@pytest.fixture
def unit(tmp_path):
    path = tmp_path / "unit"
    path.mkdir()
    return path


@pytest.fixture
def spec():
    return _make_spec("D0-M")


def test_index_endpoints_canonical(patched):
    indexed = prediction_io.index_endpoints(_plan())
    assert list(indexed) == ["ctx-1"]
    entry = indexed["ctx-1"]
    assert entry["endpoint"] == {"context_id": "ctx-1"}
    assert [s["recipe_id"] for s in entry["specs"]] == ["D0-M", "D3"]


INDEX_CASES = [
    ({"contexts": [], "refits": {}, "aliases": []}, "plan_endpoints_malformed"),
    (_plan(specs={"bad": SPECS[0], R2: SPECS[1]}), "plan_spec_key_mismatch"),
    (
        _plan(aliases=[{"refit_id": R1}, {"refit_id": R2}, {"refit_id": "ghost"}]),
        "plan_alias_unregistered_refit",
    ),
    (_plan(aliases=[{"refit_id": R1}] * 3), "plan_alias_reference_mismatch"),
    (_plan(endpoints=[{"context_id": "ctx-2"}]), "plan_context_set_mismatch"),
]


@pytest.mark.parametrize("plan, code", INDEX_CASES)
def test_index_endpoints_rejects_bad_plan(patched, plan, code):
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        prediction_io.index_endpoints(plan)
    assert _code(info.value) == code


def test_index_endpoints_rejects_duplicate_endpoint(monkeypatch):
    monkeypatch.setattr(prediction_io, "EXPECTED_CONTEXTS", 2)
    monkeypatch.setattr(prediction_io, "EXPECTED_ALIASES", 3)
    plan = _plan(endpoints=[{"context_id": "ctx-1"}, {"context_id": "ctx-1"}])
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        prediction_io.index_endpoints(plan)
    assert _code(info.value) == "plan_context_duplicate"


def _endpoint(context_id="ctx-1", uids=UIDS):
    return {"context_id": context_id, "test_uids": list(uids)}


def _frame(logits=None, probs=None, uids=UIDS):
    logits = np.asarray(
        [[1e300, -1e300, 1.0], [-1e300, 1e300, -1.0]] if logits is None else logits,
        dtype=np.float64,
    )
    probs = np.asarray(
        [[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]] if probs is None else probs, dtype=np.float64
    )
    data = {"observation_uid": list(uids)}
    for i in range(3):
        data[f"logit_{i}"] = logits[:, i]
        data[f"probability_{i}"] = probs[:, i]
    return pd.DataFrame(data)


def _audit(spec, uids=UIDS, **overrides):
    payload = {
        "refit_id": spec["refit_id"],
        "classes": list(spec["classes"]),
        "rows": len(uids),
        "test_uid_set_sha256": core._canon().sha256_value(sorted(uids)),
        "optimizer_steps": 0,
        "elapsed_seconds": 0.0,
        "peak_cuda_bytes": 0,
        "model_state_sha256": "a" * 64,
        "calibration_state_sha256": "b" * 64,
    }
    payload.update(overrides)
    return payload


def _persist(unit, spec, *, endpoint=None, frame=None, audit=None):
    return prediction_io.persist_prediction(
        unit,
        spec,
        _endpoint() if endpoint is None else endpoint,
        _frame() if frame is None else frame,
        _audit(spec) if audit is None else audit,
    )


def _assert_empty(unit):
    assert list(unit.rglob("*")) == []


def test_persist_prediction_roundtrip(unit, spec):
    frame = _frame()
    audit = _audit(spec)
    result = prediction_io.persist_prediction(unit, spec, _endpoint(), frame, audit)
    assert result["row_count"] == 2
    predictions_path = unit / prediction_io.PREDICTIONS_FILENAME
    audit_path = unit / prediction_io.AUDIT_FILENAME
    reloaded = pd.read_csv(
        predictions_path,
        dtype={"observation_uid": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    assert reloaded["observation_uid"].tolist() == UIDS
    for i in range(3):
        for name in (f"logit_{i}", f"probability_{i}"):
            assert np.array_equal(
                reloaded[name].to_numpy(dtype=np.float64), frame[name].to_numpy(dtype=np.float64)
            )
    stored = core._read_json(audit_path, "prediction_audit")
    assert core._canon().canonical_json_bytes(stored) == core._canon().canonical_json_bytes(audit)


def test_persist_prediction_refuses_regular_file(unit, spec):
    (unit / prediction_io.PREDICTIONS_FILENAME).write_text("occupied")
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        _persist(unit, spec)
    assert _code(info.value) == "prediction_output_exists"


def test_persist_prediction_refuses_dangling_symlink(unit, spec):
    (unit / prediction_io.PREDICTIONS_FILENAME).symlink_to(unit / "missing.csv")
    with pytest.raises(core.P05CoreError):
        _persist(unit, spec)


def test_persist_prediction_refuses_existing_audit(unit, spec):
    (unit / prediction_io.AUDIT_FILENAME).write_text("{}")
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        _persist(unit, spec)
    assert _code(info.value) == "prediction_output_exists"
    assert not (unit / prediction_io.PREDICTIONS_FILENAME).exists()


def test_persist_prediction_rejects_unit_symlink(tmp_path, spec):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "unit"
    link.symlink_to(real)
    with pytest.raises(core.P05CoreError):
        _persist(link, spec)


BAD_FRAMES = [
    ("uid", "frame_uids_mismatch"),
    ("column", "frame_columns_mismatch"),
    ("reorder", "frame_columns_mismatch"),
    ("nan", "frame_logits_nonfinite"),
    ("range", "frame_probabilities_range"),
    ("sum", "frame_probabilities_sum"),
]


def _bad_frame(kind):
    frame = _frame()
    if kind == "uid":
        frame.loc[0, "observation_uid"] = "ZZZ"
    elif kind == "column":
        frame = frame.rename(columns={"logit_0": "logit_9"})
    elif kind == "reorder":
        frame = frame[
            [
                "observation_uid",
                "probability_0",
                "logit_0",
                "logit_1",
                "probability_1",
                "logit_2",
                "probability_2",
            ]
        ]
    elif kind == "nan":
        frame.loc[0, "logit_0"] = np.nan
    elif kind == "range":
        frame.loc[0, "probability_0"] = 1.5
    elif kind == "sum":
        frame.loc[0, "probability_0"] = 0.6
    return frame


@pytest.mark.parametrize("kind, code", BAD_FRAMES)
def test_persist_prediction_rejects_bad_frame(unit, spec, kind, code):
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        _persist(unit, spec, frame=_bad_frame(kind))
    assert _code(info.value) == code
    _assert_empty(unit)


BAD_AUDITS = [
    ({"refit_id": "0" * 64}, "audit_refit_id_mismatch"),
    ({"classes": ["A", "B", "D"]}, "audit_classes_mismatch"),
    ({"rows": True}, "audit_rows_mismatch"),
    ({"optimizer_steps": True}, "audit_optimizer_steps_invalid"),
    ({"optimizer_steps": 1}, "audit_optimizer_steps_invalid"),
    ({"elapsed_seconds": -1.0}, "audit_elapsed_invalid"),
    ({"peak_cuda_bytes": prediction_io.MAX_PEAK_CUDA_BYTES + 1}, "audit_peak_cuda_invalid"),
    ({"model_state_sha256": "nothex"}, "audit_digest_invalid"),
]


@pytest.mark.parametrize("overrides, code", BAD_AUDITS)
def test_persist_prediction_rejects_bad_audit(unit, spec, overrides, code):
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        _persist(unit, spec, audit=_audit(spec, **overrides))
    assert _code(info.value) == code
    _assert_empty(unit)


def test_persist_prediction_readback_rejects_corrupt_numeric(unit, spec, monkeypatch):
    real_read_csv = pd.read_csv

    def corrupting_read_csv(*args, **kwargs):
        loaded = real_read_csv(*args, **kwargs)
        loaded.loc[0, "logit_2"] = loaded.loc[0, "logit_2"] + 1.0
        return loaded

    monkeypatch.setattr(pd, "read_csv", corrupting_read_csv)
    with pytest.raises(prediction_io.P05PredictionIOError) as info:
        _persist(unit, spec)
    assert _code(info.value) == "predictions_values_mismatch"
