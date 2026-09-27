"""Deterministic synthetic tests for the P05 refit I/O boundary.

These tests never touch a real dataset, artifact store or CUDA device. The
frozen container loader and noise frame are stubbed for input tests, and all
check/persistence tests run against synthetic metadata plus (for the single
integration case) one tiny synthetic 30-epoch CPU refit.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p04_runtime as runtime  # noqa: E402
from atlas_sers.evaluation import p05_comprehensive_inputs as comprehensive  # noqa: E402
from atlas_sers.evaluation import p05_core_run as core  # noqa: E402
from atlas_sers.evaluation import p05_refit_plan as plan  # noqa: E402
from atlas_sers.evaluation import p05_selection as selection  # noqa: E402
from atlas_sers.evaluation.p05_refit import RefitResult, train_refit  # noqa: E402
from atlas_sers.evaluation.p05_refit_io import (  # noqa: E402
    P05RefitIOError,
    check_completed_refit,
    persist_refit_result,
    prepare_refit_inputs,
)
from atlas_sers.evaluation.p05_sampling import Observation  # noqa: E402

PERMIT = comprehensive.COMPREHENSIVE_PERMIT_SHA256
SEED = int(selection.SEEDS[0])
CONTEXT_ID = "ctx-1"
FIT_ROLE_ID = "role-fit"
TEST_ROLE_ID = "role-test"
STATION = "station-1"
HELD_INSTRUMENT = "held-inst"
CLASSES = ["A", "B", "C"]
FEATURES = 1401

SOURCE_SPECS = (
    ("src-A-0", "A", "m-A-0", "inst-a"),
    ("src-A-1", "A", "m-A-1", "inst-a"),
    ("src-B-0", "B", "m-B-0", "inst-a"),
    ("src-B-1", "B", "m-B-1", "inst-a"),
    ("src-C-0", "C", "m-C-0", "inst-a"),
    ("src-C-1", "C", "m-C-1", "inst-a"),
)
TEST_SPECS = (
    ("held-A-0", "A", "m-held-A", "held-inst"),
    ("held-B-0", "B", "m-held-B", "held-inst"),
    ("held-C-0", "C", "m-held-C", "held-inst"),
)
SOURCE_UIDS = sorted(item[0] for item in SOURCE_SPECS)


def _manifest_row(uid, label, master, instrument, station=STATION):
    return {
        "observation_uid": uid,
        "master_sample_id": master,
        "station": station,
        "target_analyte": label,
        "instrument": instrument,
        "sensor_family": "sf",
    }


def _role_rows(role_id, role, specs):
    return [
        {
            "role_id": role_id,
            "context_id": CONTEXT_ID,
            "role": role,
            "observation_uid": uid,
            "master_sample_id": master,
            "instrument": instrument,
            "target_analyte": label,
        }
        for uid, label, master, instrument in specs
    ]


def _build_support(source, test, manifest, *, held=HELD_INSTRUMENT, station=STATION):
    return types.SimpleNamespace(
        contexts=[
            {
                "context_id": CONTEXT_ID,
                "station": station,
                "held_instrument": held,
                "selection_mode": "master_cv",
                "phase_gate": "held_evaluation",
            }
        ],
        roles=_role_rows(FIT_ROLE_ID, "outer_fit", source)
        + _role_rows(TEST_ROLE_ID, "outer_test", test),
        manifest=[_manifest_row(*item) for item in manifest],
    )


def _support():
    return _build_support(SOURCE_SPECS, TEST_SPECS, SOURCE_SPECS + TEST_SPECS)


def _bundle(support, tmp_path):
    return {
        "support": support,
        "contract": {
            "population": {"rows": len(support.manifest)},
            "input_pins": {"representation_sha256": "0" * 64, "manifest_sha256": "1" * 64},
        },
        "p01_path": tmp_path,
    }


def _make_spec(
    *,
    fitting_uids=None,
    classes=None,
    recipe="D0-M",
    seed=None,
    epochs=30,
    context_id=CONTEXT_ID,
    fitting_role_id=FIT_ROLE_ID,
    permit=PERMIT,
    calibration=("cal-1", "cal-2"),
    refit_id=None,
):
    fitting_uids = list(SOURCE_UIDS if fitting_uids is None else fitting_uids)
    classes = list(CLASSES if classes is None else classes)
    seed = SEED if seed is None else seed
    identity = {
        "context_id": context_id,
        "fitting_role_id": fitting_role_id,
        "source_uid_set_sha256": core._canon().sha256_value(fitting_uids),
        "recipe_id": recipe,
        "seed": seed,
        "epochs": epochs,
        "calibration_slot_ids": list(calibration),
        "permit_sha256": permit,
    }
    spec = dict(identity)
    spec["fitting_uids"] = list(fitting_uids)
    spec["classes"] = list(classes)
    spec["refit_id"] = plan._sha256_canonical(identity) if refit_id is None else refit_id
    return spec


def _fake_load(calls):
    def load(path, expected_sha256, manifest_uids, expected_rows):
        calls.append(1)
        labels = list(manifest_uids)
        intensity = np.stack(
            [np.full(FEATURES, float(index), dtype=np.float32) for index in range(len(labels))]
        )
        return intensity, labels

    return load


def _fake_noise(calls):
    def frame(path, manifest_sha256, fitting, pandas_module):
        calls.append(list(fitting))
        return "noise-frame"

    return frame


def _prepare_raises(bundle, spec, monkeypatch):
    calls = []
    monkeypatch.setattr(core, "_load_representation", _fake_load(calls))
    monkeypatch.setattr(core, "_noise_frame", _fake_noise([]))
    with pytest.raises(P05RefitIOError):
        prepare_refit_inputs(bundle, spec)
    assert calls == []


# --------------------------------------------------------------------------- #
# prepare_refit_inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reverse_manifest", [False, True])
def test_prepare_exposes_only_source_rows(tmp_path, monkeypatch, reverse_manifest):
    support = _support()
    if reverse_manifest:
        support.manifest.reverse()
    bundle = _bundle(support, tmp_path)
    spec = _make_spec()
    load_calls, noise_calls = [], []
    monkeypatch.setattr(core, "_load_representation", _fake_load(load_calls))
    monkeypatch.setattr(core, "_noise_frame", _fake_noise(noise_calls))
    inputs = prepare_refit_inputs(bundle, spec)
    assert load_calls == [1]
    assert noise_calls == [SOURCE_UIDS]
    assert inputs["noise_metadata"] == "noise-frame"
    assert inputs["role_id"] == FIT_ROLE_ID
    assert inputs["recipe"] == "D0-M"
    assert inputs["seed"] == SEED
    assert inputs["epochs"] == 30
    assert inputs["values"].shape == (len(SOURCE_UIDS), FEATURES)
    full_order = [row["observation_uid"] for row in support.manifest]
    assert inputs["values"][:, 0].tolist() == [float(full_order.index(uid)) for uid in SOURCE_UIDS]
    assert [observation.uid for observation in inputs["observations"]] == SOURCE_UIDS
    assert not ({item[0] for item in TEST_SPECS} & set(SOURCE_UIDS))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("fitting_uids"),
        lambda s: s.update(fitting_uids=[1, 2, 3]),
        lambda s: s.update(fitting_uids=list(reversed(SOURCE_UIDS))),
        lambda s: s.update(fitting_uids=[SOURCE_UIDS[0], SOURCE_UIDS[0]]),
        lambda s: s.pop("classes"),
        lambda s: s.update(classes=[1, 2, 3]),
        lambda s: s.update(classes=["B", "A", "C"]),
        lambda s: s.update(classes=["A", "B"]),
        lambda s: s.update(refit_id="0" * 64),
        lambda s: s.update(recipe_id="D9"),
        lambda s: s.update(epochs=29),
        lambda s: s.update(epochs=201),
        lambda s: s.update(seed=999999),
        lambda s: s.update(permit_sha256="0" * 64),
        lambda s: s.update(calibration_slot_ids=[]),
        lambda s: s.update(calibration_slot_ids=["b", "a"]),
    ],
)
def test_prepare_rejects_spec_before_loader(tmp_path, monkeypatch, mutate):
    bundle = _bundle(_support(), tmp_path)
    spec = _make_spec()
    mutate(spec)
    _prepare_raises(bundle, spec, monkeypatch)


def test_prepare_rejects_wrong_role_before_loader(tmp_path, monkeypatch):
    bundle = _bundle(_support(), tmp_path)
    spec = _make_spec(fitting_role_id="role-other")
    _prepare_raises(bundle, spec, monkeypatch)


def test_prepare_rejects_held_instrument_before_loader(tmp_path, monkeypatch):
    support = _build_support(SOURCE_SPECS, TEST_SPECS, SOURCE_SPECS + TEST_SPECS, held="inst-a")
    bundle = _bundle(support, tmp_path)
    _prepare_raises(bundle, _make_spec(), monkeypatch)


def test_prepare_rejects_test_master_overlap_before_loader(tmp_path, monkeypatch):
    test = (
        ("held-A-0", "A", "m-A-0", "held-inst"),
        ("held-B-0", "B", "m-held-B", "held-inst"),
        ("held-C-0", "C", "m-held-C", "held-inst"),
    )
    manifest = SOURCE_SPECS + test
    support = _build_support(SOURCE_SPECS, test, manifest)
    bundle = _bundle(support, tmp_path)
    _prepare_raises(bundle, _make_spec(), monkeypatch)


def test_prepare_rejects_uid_overlap_before_loader(tmp_path, monkeypatch):
    test = (
        ("src-A-0", "A", "m-A-0", "inst-a"),
        ("held-B-0", "B", "m-held-B", "held-inst"),
        ("held-C-0", "C", "m-held-C", "held-inst"),
    )
    manifest = SOURCE_SPECS + test[1:]
    support = _build_support(SOURCE_SPECS, test, manifest)
    bundle = _bundle(support, tmp_path)
    _prepare_raises(bundle, _make_spec(), monkeypatch)


def test_prepare_rejects_metadata_mismatch_before_loader(tmp_path, monkeypatch):
    source = (("src-A-0", "A", "m-A-0", "inst-B"),) + SOURCE_SPECS[1:]
    support = _build_support(source, TEST_SPECS, SOURCE_SPECS + TEST_SPECS)
    bundle = _bundle(support, tmp_path)
    _prepare_raises(bundle, _make_spec(), monkeypatch)


def test_prepare_rejects_station_mismatch_before_loader(tmp_path, monkeypatch):
    support = _build_support(
        SOURCE_SPECS, TEST_SPECS, SOURCE_SPECS + TEST_SPECS, station="station-2"
    )
    bundle = _bundle(support, tmp_path)
    _prepare_raises(bundle, _make_spec(), monkeypatch)


def test_prepare_rejects_class_set_mismatch_before_loader(tmp_path, monkeypatch):
    source = (
        ("src-A-0", "A", "m-A-0", "inst-a"),
        ("src-A-1", "A", "m-A-1", "inst-a"),
        ("src-B-0", "B", "m-B-0", "inst-a"),
        ("src-B-1", "B", "m-B-1", "inst-a"),
        ("src-D-0", "D", "m-D-0", "inst-a"),
        ("src-D-1", "D", "m-D-1", "inst-a"),
    )
    support = _build_support(source, TEST_SPECS, source + TEST_SPECS)
    bundle = _bundle(support, tmp_path)
    uids = sorted(item[0] for item in source)
    spec = _make_spec(fitting_uids=uids, classes=CLASSES)
    _prepare_raises(bundle, spec, monkeypatch)


def test_prepare_rejects_spec_fitting_uid_mismatch_before_loader(tmp_path, monkeypatch):
    bundle = _bundle(_support(), tmp_path)
    spec = _make_spec(fitting_uids=SOURCE_UIDS[:-1])
    _prepare_raises(bundle, spec, monkeypatch)


# --------------------------------------------------------------------------- #
# Synthetic result fixture for persistence/check tests
# --------------------------------------------------------------------------- #


def _training_fit(seed=12345):
    rng = np.random.default_rng(seed)
    values, observations, metadata = [], [], []
    for uid, label, master, instrument in SOURCE_SPECS:
        vector = rng.normal(size=FEATURES).astype(np.float32)
        low, high = float(vector.min()), float(vector.max())
        vector = ((vector - low) / (high - low)).astype(np.float32)
        values.append(vector)
        observations.append(Observation(uid, master, STATION, label, instrument, "na"))
        metadata.append(
            {"observation_uid": uid, "first_difference_noise_mad": 0.01, "intensity_range": 1.0}
        )
    return np.asarray(values, dtype=np.float32), observations, pd.DataFrame(metadata)


@pytest.fixture(scope="module")
def trained():
    values, observations, metadata = _training_fit()
    result = train_refit(
        values=values,
        observations=observations,
        noise_metadata=metadata,
        role_id=FIT_ROLE_ID,
        recipe="D0-M",
        seed=SEED,
        epochs=30,
        device="cpu",
    )
    assert result.status == "complete"
    assert list(result.classes) == CLASSES
    spec = _make_spec(recipe="D0-M", seed=SEED, epochs=30)
    return result, spec


def _small_state():
    return {"w": torch.zeros(4, dtype=torch.float32), "b": torch.ones(3, dtype=torch.float32)}


def test_train_persist_check_integration(tmp_path, trained):
    result, spec = trained
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    persisted = persist_refit_result(torch, run_dir, spec, result)
    assert persisted["status"] == "complete"
    assert (run_dir / "summary.json").is_file()
    loaded = torch.load(run_dir / "terminal.pt", weights_only=True, map_location="cpu")
    assert set(loaded["state_dict"]) == set(result.terminal_state_dict)
    checked = check_completed_refit(torch, run_dir, spec, result)
    assert checked["status"] == "complete"
    assert checked["terminal_state_digest"] == result.terminal_state_digest


def test_persist_partial_failed_result(tmp_path, trained):
    _, spec = trained
    state = _small_state()
    failed = RefitResult(
        status="fit_failure",
        reason_code="callback_RuntimeError",
        history=[
            {
                "epoch": 1,
                "chemical_ce": 0.5,
                "total_loss": 0.5,
                "supcon_loss": 0.0,
                "paired_loss": 0.0,
                "epoch_optimizer_steps": 4,
                "total_optimizer_steps": 4,
            }
        ],
        epochs=spec["epochs"],
        epochs_completed=1,
        optimizer_steps=4,
        finite_gradient_batches=4,
        classes=tuple(spec["classes"]),
        role_id=spec["fitting_role_id"],
        recipe=spec["recipe_id"],
        seed=spec["seed"],
        terminal_state_dict=state,
        terminal_state_digest=runtime._state_hash(state),
    )
    run_dir = tmp_path / "failed"
    run_dir.mkdir()
    persisted = persist_refit_result(torch, run_dir, spec, failed)
    assert persisted["status"] == "fit_failure"
    assert persisted["reason_code"] == "callback_RuntimeError"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "terminal.pt").is_file()
    saved = json.loads((run_dir / "summary.json").read_text())
    assert saved["reason_code"] == "callback_RuntimeError"


@pytest.mark.parametrize(
    "field,value",
    [
        ("role_id", "other-role"),
        ("recipe", "D1"),
        ("seed", SEED + 1),
        ("epochs", 31),
    ],
)
def test_persist_rejects_result_identity_before_write(tmp_path, trained, field, value):
    result, spec = trained
    run_dir = tmp_path / field
    run_dir.mkdir()
    bad = dataclasses.replace(result, **{field: value})
    with pytest.raises(P05RefitIOError):
        persist_refit_result(torch, run_dir, spec, bad)
    assert list(run_dir.iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("refit_id", "0" * 64),
        ("recipe_id", "D9"),
        ("epochs", 29),
        ("seed", 999999),
    ],
)
def test_persist_rejects_spec_identity_before_write(tmp_path, trained, field, value):
    result, spec = trained
    run_dir = tmp_path / field
    run_dir.mkdir()
    bad = dict(spec)
    bad[field] = value
    with pytest.raises(P05RefitIOError):
        persist_refit_result(torch, run_dir, bad, result)
    assert list(run_dir.iterdir()) == []


def test_persist_refuses_overwrite(tmp_path, trained):
    result, spec = trained
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    persist_refit_result(torch, run_dir, spec, result)
    with pytest.raises(P05RefitIOError):
        persist_refit_result(torch, run_dir, spec, result)


def test_persist_rejects_symlink_run_dir(tmp_path, trained):
    result, spec = trained
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported")
    with pytest.raises(P05RefitIOError):
        persist_refit_result(torch, link, spec, result)


def test_persist_rejects_symlinked_summary(tmp_path, trained):
    result, spec = trained
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    try:
        (run_dir / "summary.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unsupported")
    with pytest.raises(P05RefitIOError):
        persist_refit_result(torch, run_dir, spec, result)
    assert outside.read_text() == "{}"


# --------------------------------------------------------------------------- #
# check_completed_refit
# --------------------------------------------------------------------------- #


def test_check_rejects_tampered_summary(tmp_path, trained):
    result, spec = trained
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    persist_refit_result(torch, run_dir, spec, result)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["status"] = "failed"
    core._atomic_write(run_dir / "summary.json", core._canon().canonical_json_bytes(summary))
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, run_dir, spec, result)


def test_check_rejects_tampered_checkpoint(tmp_path, trained):
    result, spec = trained
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    persist_refit_result(torch, run_dir, spec, result)
    loaded = torch.load(run_dir / "terminal.pt", weights_only=True, map_location="cpu")
    state = dict(loaded["state_dict"])
    key = next(iter(state))
    state[key] = state[key] + 1.0
    core._save_state(torch, state, run_dir / "terminal.pt")
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, run_dir, spec, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_capture_failed", True),
        ("epochs_completed", 1),
        ("optimizer_steps", 1),
        ("finite_gradient_batches", 1),
        ("parameter_count", 1),
    ],
)
def test_check_rejects_bad_counts(tmp_path, trained, field, value):
    result, spec = trained
    bad = dataclasses.replace(result, **{field: value})
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, tmp_path, spec, bad)


def test_check_rejects_class_order(tmp_path, trained):
    result, spec = trained
    bad = dataclasses.replace(result, classes=tuple(reversed(result.classes)))
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, tmp_path, spec, bad)


def test_check_rejects_nonfinite_loss(tmp_path, trained):
    result, spec = trained
    history = [dict(record) for record in result.history]
    history[0]["chemical_ce"] = float("nan")
    bad = dataclasses.replace(result, history=history)
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, tmp_path, spec, bad)


def test_check_rejects_missing_loss(tmp_path, trained):
    result, spec = trained
    history = [dict(record) for record in result.history]
    del history[0]["total_loss"]
    bad = dataclasses.replace(result, history=history)
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, tmp_path, spec, bad)


def test_check_rejects_reason_code(tmp_path, trained):
    result, spec = trained
    bad = dataclasses.replace(result, reason_code="fit_failure")
    with pytest.raises(P05RefitIOError):
        check_completed_refit(torch, tmp_path, spec, bad)
