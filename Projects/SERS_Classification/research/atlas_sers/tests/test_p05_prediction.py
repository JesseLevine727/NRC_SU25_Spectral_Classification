"""CPU-only, no-fit tests for the P05 single-refit prediction helper."""

from __future__ import annotations

# Optional torch must be checked before importing torch-dependent project modules.
# ruff: noqa: E402
import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import classical
from atlas_sers.evaluation import p04_runtime as runtime
from atlas_sers.evaluation import p05_calibration as calibration
from atlas_sers.evaluation import p05_comprehensive_inputs as comprehensive_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_development as development
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_refit as refit
from atlas_sers.evaluation import p05_refit_evidence as evidence
from atlas_sers.evaluation import p05_refit_io as refit_io
from atlas_sers.evaluation import p05_refit_plan as refit_plan
from atlas_sers.evaluation import p05_selection as selection
from atlas_sers.evaluation import p05_smoke as smoke
from atlas_sers.models import acquisition
from tests.test_p05_comprehensive_refits import _build_result, _temperature_calibration

RECIPES = ("D0-M", "D3")
FITTING_UIDS = ["fA", "fB", "fC"]
CLASSES = ["A", "B", "C"]
CALIBRATION_SLOTS = ["calib-1"]
EPOCHS = 30
TEST_UIDS = ["t1", "t2", "t3"]
FUTURE_DEADLINE = 1.0e12


def _make_spec(recipe_id):
    identity = {
        "context_id": "ctx-1",
        "fitting_role_id": "outer_fit",
        "source_uid_set_sha256": core._canon().sha256_value(list(FITTING_UIDS)),
        "recipe_id": recipe_id,
        "seed": sorted(selection.SEEDS)[0],
        "epochs": EPOCHS,
        "calibration_slot_ids": list(CALIBRATION_SLOTS),
        "permit_sha256": comprehensive_inputs.COMPREHENSIVE_PERMIT_SHA256,
    }
    return {
        **identity,
        "refit_id": refit_plan._sha256_canonical(identity),
        "fitting_uids": list(FITTING_UIDS),
        "classes": list(CLASSES),
    }


def _real_state(recipe_id, seed):
    use_projection = bool(smoke.RECIPE_SPECIFICATIONS[recipe_id][2])
    torch.manual_seed(seed)
    model = acquisition.AcquisitionClassifier(
        class_count=refit_io.EXPECTED_CLASS_COUNT, use_projection=use_projection
    )
    return dict(model.state_dict())


def _audit(cal, spec):
    return {
        "calibration_state_sha256": cal.state_sha256,
        "refit_id": spec["refit_id"],
        "context_id": spec["context_id"],
        "recipe_id": spec["recipe_id"],
        "seed": spec["seed"],
        "calibration_slot_ids": list(spec["calibration_slot_ids"]),
        "temperature": cal.temperature,
        "optimizer_success": True,
        "optimizer_objective": cal.optimizer_objective,
    }


@pytest.fixture(params=RECIPES)
def case(tmp_path, monkeypatch, request):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    spec = _make_spec(request.param)
    state = _real_state(request.param, spec["seed"])
    result = dataclasses.replace(
        _build_result(spec),
        terminal_state_dict=state,
        terminal_state_digest=runtime._state_hash(state),
    )
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    refit_io.persist_refit_result(torch, unit_dir, spec, result)
    cal = dataclasses.replace(_temperature_calibration(), temperature=2.3)
    evidence.persist_calibration(unit_dir, cal, _audit(cal, spec), spec)
    values = (
        np.random.default_rng(7)
        .standard_normal((len(TEST_UIDS), prediction.EXPECTED_FEATURES))
        .astype(np.float32)
    )
    return SimpleNamespace(
        tmp_path=tmp_path,
        unit_dir=unit_dir,
        spec=spec,
        canonical=refit_io._check_spec(spec),
        state=state,
        result=result,
        calibration=cal,
        values=values,
        uids=list(TEST_UIDS),
    )


def _predict(case, *, values=None, uids=None, deadline=FUTURE_DEADLINE, unit_dir=None):
    return prediction.predict_refit(
        spec=case.spec,
        unit_dir=case.unit_dir if unit_dir is None else unit_dir,
        values=case.values if values is None else values,
        observation_uids=case.uids if uids is None else uids,
        device="cpu",
        deadline=deadline,
    )


def _code(exc):
    return getattr(exc, "reason_code", None) or str(exc)


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path, payload):
    path.write_bytes(core._canon().canonical_json_bytes(payload))


def _reference(recipe_id, state, calibration_obj, values):
    use_projection = bool(smoke.RECIPE_SPECIFICATIONS[recipe_id][2])
    model = acquisition.AcquisitionClassifier(
        class_count=refit_io.EXPECTED_CLASS_COUNT, use_projection=use_projection
    )
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(values[:, None, :]))
    with torch.no_grad():
        logits = development._predict_logits(model, tensor, torch.device("cpu"))
        probabilities = classical.apply_temperature(logits, calibration_obj)
    return np.asarray(logits, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)


def test_forward_logits_and_probabilities_match_reference(case):
    frame, audit = _predict(case)
    expected_logits, expected_probabilities = _reference(
        case.spec["recipe_id"], case.state, case.calibration, case.values
    )
    assert frame["observation_uid"].tolist() == case.uids
    got_logits = np.column_stack(
        [frame[f"logit_{i}"].to_numpy() for i in range(prediction.EXPECTED_CLASS_COUNT)]
    )
    got_probabilities = np.column_stack(
        [frame[f"probability_{i}"].to_numpy() for i in range(prediction.EXPECTED_CLASS_COUNT)]
    )
    assert np.array_equal(got_logits, expected_logits)
    assert np.array_equal(got_probabilities, expected_probabilities)
    assert audit["classes"] == CLASSES
    assert audit["rows"] == len(case.uids)
    assert audit["optimizer_steps"] == 0


def test_prediction_writes_nothing_and_fits_nothing(case, monkeypatch):
    before = _tree_hash(case.tmp_path)

    def _no_fit(*args, **kwargs):
        raise AssertionError("fitting must not run")

    def _no_optimizer(self, *args, **kwargs):
        raise AssertionError("optimizer must not be constructed")

    monkeypatch.setattr(refit, "train_refit", _no_fit)
    monkeypatch.setattr(calibration, "calibrate_spec", _no_fit)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", _no_optimizer)
    frame, _ = _predict(case)
    assert len(frame) == len(case.uids)
    assert _tree_hash(case.tmp_path) == before


def test_summary_restores_tuple_fields(case, monkeypatch):
    recorded = {}
    real_cls = refit.RefitResult

    class RecordingResult(real_cls):
        def __init__(self, **kwargs):
            recorded.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(refit, "RefitResult", RecordingResult)
    prediction._load_result(case.unit_dir, case.canonical)
    assert isinstance(recorded["classes"], tuple)
    assert isinstance(recorded["source_noise_levels"], tuple)
    assert recorded["classes"] == tuple(CLASSES)
    assert recorded["source_noise_levels"] == (0.0,)


@pytest.mark.parametrize(
    "values, code",
    [
        (np.zeros((3, 1400), dtype=np.float32), "values_shape_malformed"),
        (np.zeros((3, 1401), dtype=np.float64), "values_malformed"),
        (np.full((3, 1401), np.nan, dtype=np.float32), "values_malformed"),
        (np.zeros((0, 1401), dtype=np.float32), "values_empty"),
    ],
)
def test_bad_values_fail_closed(case, values, code):
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case, values=values)
    assert _code(info.value) == code


@pytest.mark.parametrize(
    "uids, code",
    [
        (["t1", "t1", "t2"], "uids_malformed"),
        (["fA", "t2", "t3"], "source_test_uid_overlap"),
    ],
)
def test_bad_uids_fail_closed(case, uids, code):
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case, uids=uids)
    assert _code(info.value) == code


def test_uid_order_is_preserved(case):
    uids = ["z9", "a1", "m5"]
    frame, _ = _predict(case, uids=uids)
    assert frame["observation_uid"].tolist() == uids


@pytest.mark.parametrize(
    "deadline, code",
    [
        ("soon", "deadline_malformed"),
        (-1.0, "deadline_exceeded"),
    ],
)
def test_bad_deadline_fails_before_io(case, deadline, code):
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case, deadline=deadline, unit_dir=case.tmp_path / "absent")
    assert _code(info.value) == code


def test_tampered_terminal_checkpoint_fails_closed(case):
    path = case.unit_dir / "terminal.pt"
    state = dict(torch.load(path, weights_only=True, map_location="cpu")["state_dict"])
    key = next(name for name, value in state.items() if value.is_floating_point())
    state[key] = state[key] + 1.0
    path.unlink()
    core._save_state(torch, state, path)
    with pytest.raises(core.P05CoreError) as info:
        _predict(case)
    assert _code(info.value) == "refit_terminal_digest_mismatch"


def _remove(path):
    path.unlink()


def _symlink(path):
    real = path.with_name(path.name + ".real")
    path.rename(real)
    path.symlink_to(real)


@pytest.mark.parametrize(
    "name, mutate, code",
    [
        ("terminal.pt", _remove, "refit_terminal_checkpoint_missing"),
        ("terminal.pt", _symlink, None),
        ("calibration.json", _remove, "calibration_missing"),
        ("calibration.json", _symlink, None),
    ],
)
def test_missing_or_symlinked_artifacts_fail_closed(case, name, mutate, code):
    mutate(case.unit_dir / name)
    with pytest.raises(core.P05CoreError) as info:
        _predict(case)
    if code is not None:
        assert _code(info.value) == code


def _rewrite_summary(unit_dir, **overrides):
    payload = core._read_json(unit_dir / "summary.json", "summary")
    payload.update(overrides)
    _write_json(unit_dir / "summary.json", payload)


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"epochs": 29}, "refit_epoch_budget_mismatch"),
        ({"classes": ["A", "B", "D"]}, "refit_class_order_mismatch"),
        ({"status": "fail"}, "refit_not_complete"),
        ({"parameter_count": 12345}, "refit_parameter_count_mismatch"),
    ],
)
def test_corrupt_summary_fails_closed(case, overrides, code):
    _rewrite_summary(case.unit_dir, **overrides)
    with pytest.raises(core.P05CoreError) as info:
        _predict(case)
    assert _code(info.value) == code


def _rewrite_calibration(unit_dir, **overrides):
    payload = core._read_json(unit_dir / "calibration.json", "calibration")
    state = dict(payload["state"])
    state.update(overrides)
    kwargs = dict(state)
    kwargs["class_vocabulary"] = tuple(str(value) for value in kwargs["class_vocabulary"])
    rebuilt = calibration.TemperatureCalibration(**kwargs)
    _write_json(
        unit_dir / "calibration.json", {"state": state, "state_sha256": rebuilt.state_sha256}
    )


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"temperature": 0.0}, "calibration_temperature_invalid"),
        ({"optimizer_success": False}, "calibration_failed"),
        ({"class_vocabulary": ["X", "Y", "Z"]}, "calibration_classes_mismatch"),
    ],
)
def test_corrupt_calibration_state_fails_closed(case, overrides, code):
    _rewrite_calibration(case.unit_dir, **overrides)
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case)
    assert _code(info.value) == code


def test_calibration_wrapper_hash_mismatch_fails_closed(case):
    payload = core._read_json(case.unit_dir / "calibration.json", "calibration")
    payload["state_sha256"] = "0" * 64
    _write_json(case.unit_dir / "calibration.json", payload)
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case)
    assert _code(info.value) == "calibration_state_sha_mismatch"


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 2.0},
        {"calibration_state_sha256": "0" * 64},
    ],
)
def test_corrupt_calibration_audit_fails_closed(case, overrides):
    payload = core._read_json(case.unit_dir / "calibration_audit.json", "calibration_audit")
    payload.update(overrides)
    _write_json(case.unit_dir / "calibration_audit.json", payload)
    with pytest.raises(prediction.P05PredictionError) as info:
        _predict(case)
    assert _code(info.value) == "calibration_audit_mismatch"
