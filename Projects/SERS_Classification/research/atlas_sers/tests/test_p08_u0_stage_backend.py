"""Synthetic CPU integration tests for the P08 U0 source-stage backend (T110).

The seven ``RuntimePair`` objects are prepared by the sibling U0
runtime-input fixture from invented spectra/metadata, the real public frozen
source specifications and an invented pinned candidate registry.  Each of the
three classical and four neural models is fitted exactly once at module scope
by the real, unmodified inherited P03/P05 kernels.  Every negative case reuses
those results or ``dataclasses.replace`` copies of them and never refits.

The tests exercise the required T110 semantics: one dispatch to the inherited
fit entry point, in-memory artifact preparation without inference, byte-pin
authentication before the numerical predictor, exact score/logits parity
rejection for consistently repinned finite tampering, estimator retention
without refit, wrong-pair refusal, frozen/fresh wrappers, unchanged
``KeyboardInterrupt``/``SystemExit`` propagation and permanent execution
denial.

Non-claims
----------
* Everything is CPU-only and synthetic.  The neural job manifest carries a GPU
  accounting label, but the backend is not admission and this test never claims
  GPU execution or a preprocessing comparison.
* The PP-U-SG labels are inherited placeholders; no SG/arPLS transform is run
  here and no toy label is reported as a scientific result.
"""

from __future__ import annotations

import builtins
import copy
import dataclasses
import io
import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p08_u0_stage_backend as backend  # noqa: E402
from atlas_sers.governance.canonical import sha256_value  # noqa: E402
from tests import test_p08_u0_inputs as metadata_fixture  # noqa: E402
from tests import test_p08_u0_runtime_inputs as runtime_fixture  # noqa: E402

CLASSICAL_MODELS = tuple(runtime_fixture.CLASSICAL_MODELS)
NEURAL_MODELS = tuple(runtime_fixture.NEURAL_RECIPES)
ALL_MODELS = CLASSICAL_MODELS + NEURAL_MODELS

NEURAL_ARTIFACT_NAMES = frozenset(
    {"summary.json", "best.pt", "terminal.pt", "validation_logits.npz"}
)
CLASSICAL_ARTIFACT_NAMES = frozenset({"summary.json", "predictions.csv"})

_EXPECTED_FITS = {"classical": len(CLASSICAL_MODELS), "neural": len(NEURAL_MODELS)}


def _fit_job(pair):
    return json.loads(pair.prepared_pair.fit_job_json)


def _prediction_job(pair):
    return json.loads(pair.prepared_pair.prediction_job_json)


def _pair_policy(pair):
    return getattr(pair.prepared_pair.inputs, "policy_id", None)


def _load_checkpoint_state(raw):
    loaded = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    assert isinstance(loaded, dict) and loaded
    assert all(isinstance(value, torch.Tensor) for value in loaded.values())
    return loaded


def _install_deny_spies(monkeypatch):
    from atlas_sers.evaluation import p03_runtime, p05_development
    from atlas_sers.models import classical

    labels = (
        "open",
        "classical_fit",
        "neural_fit",
        "classical_estimator",
        "cuda_init",
        "cuda_lazy_init",
    )
    counters = {label: 0 for label in labels}

    def deny(label):
        def inner(*args, **kwargs):
            counters[label] += 1
            raise AssertionError(label)

        return inner

    monkeypatch.setattr(builtins, "open", deny("open"), raising=True)
    monkeypatch.setattr(Path, "open", deny("open"), raising=True)
    monkeypatch.setattr(p03_runtime, "run_candidate_fit", deny("classical_fit"), raising=True)
    monkeypatch.setattr(p05_development, "train_development_fit", deny("neural_fit"), raising=True)
    monkeypatch.setattr(
        classical.AuditedClassifier, "fit", deny("classical_estimator"), raising=True
    )
    monkeypatch.setattr(torch.cuda, "init", deny("cuda_init"), raising=True)
    if hasattr(torch.cuda, "_lazy_init"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", deny("cuda_lazy_init"), raising=True)
    return counters


def _guard_numerical_predictors(monkeypatch):
    from atlas_sers.evaluation import p03_runtime, p05_development

    calls = {"aligned": 0, "logits": 0}
    real_aligned = p03_runtime._aligned_scores
    real_logits = p05_development._predict_logits

    def counting_aligned(*args, **kwargs):
        calls["aligned"] += 1
        return real_aligned(*args, **kwargs)

    def counting_logits(*args, **kwargs):
        calls["logits"] += 1
        return real_logits(*args, **kwargs)

    monkeypatch.setattr(p03_runtime, "_aligned_scores", counting_aligned, raising=True)
    monkeypatch.setattr(p05_development, "_predict_logits", counting_logits, raising=True)
    return calls


def _assert_classical_artifacts(pair, result, job, values):
    assert set(values) == set(CLASSICAL_ARTIFACT_NAMES)
    for value in values.values():
        assert type(value) is bytes and value
    summary = json.loads(values["summary.json"].decode("utf-8"))
    assert summary["status"] == "complete"
    assert summary["fit_id"] == job["job_id"]
    assert summary["model_id"] == job["model_id"]
    assert summary["candidate_id"] == job["candidate_id"]
    frame = pd.read_csv(io.BytesIO(values["predictions.csv"]), dtype=str, keep_default_na=False)
    assert list(frame["observation_uid"]) == [
        str(observation.uid) for observation in pair.validation_observations
    ]


def _assert_neural_artifacts(pair, result, job, values):
    assert set(values) == set(NEURAL_ARTIFACT_NAMES)
    for value in values.values():
        assert type(value) is bytes and value
    summary = json.loads(values["summary.json"].decode("utf-8"))
    assert summary["status"] == "complete"
    assert summary["recipe_id"] == job["model_id"]
    assert "seed" in summary and summary["seed"] == result.seed
    _load_checkpoint_state(values["best.pt"])
    _load_checkpoint_state(values["terminal.pt"])
    with np.load(io.BytesIO(values["validation_logits.npz"]), allow_pickle=False) as archive:
        assert "logits" in archive.files
        logits = np.asarray(archive["logits"], dtype=np.float64)
    assert logits.shape[0] == len(pair.validation_observations)
    assert bool(np.isfinite(logits).all())


def _bump_argmax_scores(frame):
    tampered = frame.copy(deep=True)
    scores = []
    for value in tampered["scores"]:
        row = json.loads(value)
        index = int(np.argmax(np.asarray(row, dtype=np.float64)))
        row[index] = float(row[index]) + 0.125
        scores.append(json.dumps(row, separators=(",", ":")))
    tampered["scores"] = scores
    return tampered


@pytest.fixture(scope="module")
def stage():
    from atlas_sers.evaluation import p03_runtime, p05_development

    with pytest.MonkeyPatch.context() as patch:
        ctx, inputs = runtime_fixture._prepare(patch, "master_cv")

        counters = {"classical": 0, "neural": 0}
        real_run = p03_runtime.run_candidate_fit
        real_train = p05_development.train_development_fit

        def counting_run(*args, **kwargs):
            counters["classical"] += 1
            return real_run(*args, **kwargs)

        def counting_train(*args, **kwargs):
            counters["neural"] += 1
            return real_train(*args, **kwargs)

        patch.setattr(p03_runtime, "run_candidate_fit", counting_run)
        patch.setattr(p05_development, "train_development_fit", counting_train)

        selected = {}
        for pair in inputs.pairs:
            model_id = _fit_job(pair).get("model_id")
            if model_id not in ALL_MODELS or model_id in selected:
                continue
            if _pair_policy(pair) != "PP-U-SG":
                continue
            selected[model_id] = pair
        assert set(selected) == set(ALL_MODELS)

        results = {}
        artifacts = {}
        with torch.random.fork_rng(devices=[]):
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            try:
                for model_id in ALL_MODELS:
                    pair = selected[model_id]
                    result = backend.invoke_source_fit(
                        pair, device="cpu", global_deadline=None, on_epoch=None
                    )
                    assert result.status == "complete"
                    results[model_id] = result
                    artifacts[model_id] = backend.prepare_fit_artifacts(pair, result)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)

        yield {
            "ctx": ctx,
            "inputs": inputs,
            "selected": selected,
            "results": results,
            "artifacts": artifacts,
            "counters": counters,
        }

        assert counters == _EXPECTED_FITS


def test_single_dispatch_per_model(stage):
    assert stage["counters"] == _EXPECTED_FITS
    assert set(stage["results"]) == set(ALL_MODELS)
    assert set(stage["artifacts"]) == set(ALL_MODELS)
    for model_id in ALL_MODELS:
        assert stage["results"][model_id].status == "complete"
        assert stage["artifacts"][model_id].status == "succeeded"


def test_success_identity_and_artifact_layout(stage):
    for model_id in ALL_MODELS:
        pair = stage["selected"][model_id]
        result = stage["results"][model_id]
        artifacts = stage["artifacts"][model_id]
        job = _fit_job(pair)
        prediction_job = _prediction_job(pair)
        assert artifacts.result is result
        assert artifacts.fit_job_id == job["job_id"]
        assert artifacts.prediction_job_id == prediction_job["job_id"]
        values = artifacts.artifact_bytes()
        if model_id in NEURAL_MODELS:
            roles = pair.prepared_pair.inputs.source_roles
            assert result.role_id == roles.fitting_role_id
            assert result.recipe == model_id
            assert result.seed == job["seed"]
            assert job["candidate_id"] == "fixed_recipe"
            assert tuple(result.classes) == tuple(roles.classes)
            assert tuple(result.validation_uids) == tuple(
                sorted(str(observation.uid) for observation in pair.validation_observations)
            )
            _assert_neural_artifacts(pair, result, job, values)
        else:
            assert result.fit_id == job["job_id"]
            assert result.model_id == model_id
            assert result.candidate_id == job["candidate_id"]
            assert result.seed == job["seed"]
            assert (
                pair.classical_kwargs()["parameters"] == runtime_fixture.CLASSICAL_PARAMS[model_id]
            )
            _assert_classical_artifacts(pair, result, job, values)


def test_prepare_fit_artifacts_never_infers(stage, monkeypatch):
    from atlas_sers.evaluation import p03_runtime, p05_development

    calls = {"aligned": 0, "logits": 0, "classical": 0, "neural": 0}
    real_aligned = p03_runtime._aligned_scores
    real_logits = p05_development._predict_logits
    real_classical = backend._structural_classical
    real_neural = backend._structural_neural

    def counting_aligned(*args, **kwargs):
        calls["aligned"] += 1
        return real_aligned(*args, **kwargs)

    def counting_logits(*args, **kwargs):
        calls["logits"] += 1
        return real_logits(*args, **kwargs)

    def counting_classical(*args, **kwargs):
        calls["classical"] += 1
        return real_classical(*args, **kwargs)

    def counting_neural(*args, **kwargs):
        calls["neural"] += 1
        return real_neural(*args, **kwargs)

    monkeypatch.setattr(p03_runtime, "_aligned_scores", counting_aligned, raising=True)
    monkeypatch.setattr(p05_development, "_predict_logits", counting_logits, raising=True)
    monkeypatch.setattr(backend, "_structural_classical", counting_classical, raising=True)
    monkeypatch.setattr(backend, "_structural_neural", counting_neural, raising=True)

    try:
        for model_id in ALL_MODELS:
            pair = stage["selected"][model_id]
            result = stage["results"][model_id]
            prepared = backend.prepare_fit_artifacts(pair, result)
            assert prepared.status == "succeeded"
        assert calls["classical"] == _EXPECTED_FITS["classical"]
        assert calls["neural"] == _EXPECTED_FITS["neural"]
    finally:
        assert calls["aligned"] == 0
        assert calls["logits"] == 0


def _neural_bad_history(result):
    return dataclasses.replace(result, history=[])


def _neural_missing_best(result):
    return dataclasses.replace(result, best_state_dict=None)


def _neural_missing_terminal(result):
    return dataclasses.replace(result, terminal_state_dict=None)


def _neural_wrong_seed(result):
    return dataclasses.replace(result, seed=result.seed + 1)


def _neural_wrong_role(result):
    return dataclasses.replace(result, role_id="not-the-bound-role")


def _classical_wrong_candidate(result):
    return dataclasses.replace(result, candidate_id="bogus-candidate")


def _classical_wrong_uid_hash(result):
    return dataclasses.replace(result, fit_uid_sha256="0" * 64)


MALFORMED_COMPLETE = [
    ("neural-history", "D0-M", _neural_bad_history),
    ("neural-missing-best", "D0-M", _neural_missing_best),
    ("neural-missing-terminal", "D1", _neural_missing_terminal),
    ("neural-wrong-seed", "D0-M", _neural_wrong_seed),
    ("neural-wrong-role", "D1", _neural_wrong_role),
    ("classical-wrong-candidate", "C-RANDOM-FOREST", _classical_wrong_candidate),
    ("classical-wrong-uid-hash", "C-EXTRA-TREES", _classical_wrong_uid_hash),
]


@pytest.mark.parametrize("case", MALFORMED_COMPLETE, ids=[item[0] for item in MALFORMED_COMPLETE])
def test_prepare_rejects_malformed_complete_results(stage, case):
    _label, model_id, mutate = case
    pair = stage["selected"][model_id]
    broken = mutate(stage["results"][model_id])
    assert broken.status == "complete"
    with pytest.raises(backend.StageError):
        backend.prepare_fit_artifacts(pair, broken)


@pytest.mark.parametrize("model_id", ["C-EXTRA-TREES", "D3"])
def test_noncomplete_results_preserved_and_refuse_prediction(stage, model_id):
    pair = stage["selected"][model_id]
    failed = dataclasses.replace(
        stage["results"][model_id],
        status="resource_failure",
        reason_code="deadline_exceeded",
    )
    artifacts = backend.prepare_fit_artifacts(pair, failed)
    assert artifacts.status == "failed"
    assert artifacts.result is failed
    assert artifacts.result.status == "resource_failure"
    assert artifacts.result.reason_code == "deadline_exceeded"
    with pytest.raises(backend.StageError) as info:
        backend.verify_source_prediction(
            pair, artifacts, saved_artifact_bytes=artifacts.artifact_bytes(), device="cpu"
        )
    assert info.value.reason_code == "fit_not_succeeded"


def test_failed_neural_result_preserved_without_fabricated_seed(stage):
    model_id = "D0-M"
    pair = stage["selected"][model_id]
    expected_seed = _fit_job(pair)["seed"]
    original = stage["results"][model_id]
    failed = dataclasses.replace(
        original,
        status="resource_failure",
        reason_code="deadline_exceeded",
        seed=None,
        best_state_dict=None,
        terminal_state_dict=None,
        validation_logits=None,
        history=[],
    )
    artifacts = backend.prepare_fit_artifacts(pair, failed)
    assert artifacts.status == "failed"
    assert artifacts.result is failed
    assert artifacts.result.status == "resource_failure"
    assert artifacts.result.reason_code == "deadline_exceeded"
    assert artifacts.result.seed is None
    assert original.status == "complete"
    assert original.seed is not None
    values = artifacts.artifact_bytes()
    assert set(values) == {"summary.json"}
    summary = json.loads(values["summary.json"].decode("utf-8"))
    assert summary.get("seed") is None
    assert summary.get("seed") != expected_seed
    with pytest.raises(backend.StageError) as info:
        backend.verify_source_prediction(pair, artifacts, saved_artifact_bytes=values, device="cpu")
    assert info.value.reason_code == "fit_not_succeeded"


def test_verification_report_is_canonical_and_private(stage):
    uids = set(stage["ctx"]["uids"])
    masters = {row["master_sample_id"] for row in stage["ctx"]["case"]["rows"]}
    families = {value for value in set(stage["ctx"]["sensor_family"].values()) if value}
    for model_id in ALL_MODELS:
        pair = stage["selected"][model_id]
        artifacts = stage["artifacts"][model_id]
        report = backend.verify_source_prediction(
            pair,
            artifacts,
            saved_artifact_bytes=artifacts.artifact_bytes(),
            device="cpu",
        )
        assert report["status"] == "verified"
        assert report["execution_authorized"] is False
        assert report["prediction_parity_verified"] is True
        assert report["kernel"] == ("neural" if model_id in NEURAL_MODELS else "classical")
        assert report["row_count"] == len(pair.validation_observations)
        assert report["class_count"] == 3
        without = {key: value for key, value in report.items() if key != "report_sha256"}
        assert report["report_sha256"] == sha256_value(without)
        rendered = json.dumps(report)
        for uid in uids:
            assert uid not in rendered
        for master in masters:
            assert master not in rendered
        for family in families:
            assert family not in rendered
        assert metadata_fixture.SENTINEL not in rendered
        assert "balanced_accuracy" not in rendered
        assert "negative_log_likelihood" not in rendered
        assert "macro_f1" not in rendered
        for value in report.values():
            assert not isinstance(value, (bytes, bytearray))


@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_verification_touches_no_files_and_no_refit(stage, monkeypatch, model_id):
    pair = stage["selected"][model_id]
    artifacts = stage["artifacts"][model_id]
    counters = _install_deny_spies(monkeypatch)
    before = torch.random.get_rng_state().clone()
    try:
        report = backend.verify_source_prediction(
            pair,
            artifacts,
            saved_artifact_bytes=artifacts.artifact_bytes(),
            device="cpu",
        )
        assert report["status"] == "verified"
    finally:
        assert all(value == 0 for value in counters.values()), counters
        assert torch.equal(torch.random.get_rng_state(), before)


PIN_CASES = [
    ("missing", "D0-M", None),
    ("extra", "C-RBF-SVM", None),
    ("changed", "C-RBF-SVM", "summary.json"),
    ("changed", "C-RBF-SVM", "predictions.csv"),
    ("changed", "D0-M", "summary.json"),
    ("changed", "D0-M", "validation_logits.npz"),
    ("changed", "D0-M", "best.pt"),
    ("changed", "D1", "terminal.pt"),
]


@pytest.mark.parametrize(
    "case", PIN_CASES, ids=[f"{item[0]}-{item[1]}-{item[2]}" for item in PIN_CASES]
)
def test_artifact_byte_mismatch_rejected_before_predictor(stage, monkeypatch, case):
    kind, model_id, member = case
    pair = stage["selected"][model_id]
    artifacts = stage["artifacts"][model_id]
    calls = _guard_numerical_predictors(monkeypatch)
    saved = artifacts.artifact_bytes()
    if kind == "missing":
        name = sorted(saved)[0]
        saved = {key: value for key, value in saved.items() if key != name}
        expected = "artifact_set_mismatch"
    elif kind == "extra":
        saved = {**saved, "unexpected.bin": b"x"}
        expected = "artifact_set_mismatch"
    else:
        data = bytearray(saved[member])
        data[-1] ^= 0xFF
        saved[member] = bytes(data)
        expected = "artifact_hash_mismatch"
    with pytest.raises(backend.StageError) as info:
        backend.verify_source_prediction(pair, artifacts, saved_artifact_bytes=saved, device="cpu")
    assert info.value.reason_code == expected
    assert calls == {"aligned": 0, "logits": 0}


def test_repinned_classical_score_tamper_reaches_score_parity(stage):
    model_id = "C-RBF-SVM"
    pair = stage["selected"][model_id]
    result = stage["results"][model_id]
    tampered = dataclasses.replace(
        result, validation_predictions=_bump_argmax_scores(result.validation_predictions)
    )
    artifacts = backend.prepare_fit_artifacts(pair, tampered)
    assert artifacts.status == "succeeded"
    with pytest.raises(backend.StageError) as info:
        backend.verify_source_prediction(
            pair, artifacts, saved_artifact_bytes=artifacts.artifact_bytes(), device="cpu"
        )
    assert info.value.reason_code == "score_parity_mismatch"


def test_repinned_neural_logits_tamper_reaches_logits_parity(stage):
    model_id = "D0-M"
    pair = stage["selected"][model_id]
    result = stage["results"][model_id]
    altered = np.array(result.validation_logits, dtype=np.float64, copy=True)
    altered[0, 0] += 0.125
    tampered = dataclasses.replace(result, validation_logits=altered)
    artifacts = backend.prepare_fit_artifacts(pair, tampered)
    assert artifacts.status == "succeeded"
    before = torch.random.get_rng_state().clone()
    try:
        with pytest.raises(backend.StageError) as info:
            backend.verify_source_prediction(
                pair, artifacts, saved_artifact_bytes=artifacts.artifact_bytes(), device="cpu"
            )
        assert info.value.reason_code == "logits_parity_mismatch"
    finally:
        assert torch.equal(torch.random.get_rng_state(), before)


def test_classical_missing_estimator_without_refit(stage, monkeypatch):
    model_id = "C-RANDOM-FOREST"
    pair = stage["selected"][model_id]
    artifacts = stage["artifacts"][model_id]
    stripped = dataclasses.replace(
        artifacts, result=dataclasses.replace(artifacts.result, estimator=None)
    )
    counters = _install_deny_spies(monkeypatch)
    try:
        with pytest.raises(backend.StageError) as info:
            backend.verify_source_prediction(
                pair,
                stripped,
                saved_artifact_bytes=stripped.artifact_bytes(),
                device="cpu",
            )
        assert info.value.reason_code == "estimator_missing"
    finally:
        assert all(value == 0 for value in counters.values()), counters


@pytest.mark.parametrize("model_id", ["C-RBF-SVM", "D0-M"])
def test_wrong_pair_identity_refused(stage, model_id):
    pair = stage["selected"][model_id]
    artifacts = stage["artifacts"][model_id]
    original_id = _fit_job(pair)["job_id"]
    other = None
    for candidate in stage["inputs"].pairs:
        job = _fit_job(candidate)
        if job["model_id"] == model_id and job["job_id"] != original_id:
            other = candidate
            break
    assert other is not None
    with pytest.raises(backend.StageError) as info:
        backend.verify_source_prediction(
            other,
            artifacts,
            saved_artifact_bytes=artifacts.artifact_bytes(),
            device="cpu",
        )
    assert info.value.reason_code in {"fit_pair_mismatch", "job_pair_mismatch"}


def test_artifact_wrapper_fresh_immutable_and_private(stage):
    artifacts = stage["artifacts"]["D0-M"]
    first = artifacts.artifact_bytes()
    second = artifacts.artifact_bytes()
    assert first is not second
    assert first == second
    for value in first.values():
        assert type(value) is bytes and value
    first["summary.json"] = b"tampered"
    assert artifacts.artifact_bytes() != first
    with pytest.raises(FrozenInstanceError):
        artifacts.status = "x"
    with pytest.raises(FrozenInstanceError):
        artifacts.result = None
    rendered = repr(artifacts)
    for uid in stage["ctx"]["uids"]:
        assert uid not in rendered
    for row in stage["ctx"]["case"]["rows"]:
        assert row["master_sample_id"] not in rendered
    assert metadata_fixture.SENTINEL not in rendered


def test_caller_result_reachable_after_serialization_error(stage, monkeypatch):
    model_id = "D0-M"
    pair = stage["selected"][model_id]
    result = stage["results"][model_id]

    def boom(*args, **kwargs):
        raise RuntimeError("serialization refused")

    monkeypatch.setattr(torch, "save", boom, raising=True)
    monkeypatch.setattr(np, "savez_compressed", boom, raising=True)
    with pytest.raises(backend.StageError) as info:
        backend.prepare_fit_artifacts(pair, result)
    assert info.value.reason_code == "artifact_preparation_failed"
    assert result.status == "complete"
    assert stage["results"][model_id] is result


_SIGNAL_TYPES = (KeyboardInterrupt, SystemExit)
_SIGNAL_IDS = ["KeyboardInterrupt", "SystemExit"]


@pytest.mark.parametrize("signal_type", _SIGNAL_TYPES, ids=_SIGNAL_IDS)
def test_signal_propagates_from_invoke(stage, monkeypatch, signal_type):
    from atlas_sers.evaluation import p03_runtime

    pair = stage["selected"]["C-RBF-SVM"]
    signal = signal_type("stage-backend")

    def boom(*args, **kwargs):
        raise signal

    monkeypatch.setattr(p03_runtime, "run_candidate_fit", boom, raising=True)
    with pytest.raises(signal_type) as info:
        backend.invoke_source_fit(pair, device="cpu", global_deadline=None, on_epoch=None)
    assert info.value is signal


@pytest.mark.parametrize("signal_type", _SIGNAL_TYPES, ids=_SIGNAL_IDS)
def test_signal_propagates_from_serialization(stage, monkeypatch, signal_type):
    pair = stage["selected"]["D0-M"]
    result = stage["results"]["D0-M"]
    signal = signal_type("stage-backend")

    def boom(*args, **kwargs):
        raise signal

    monkeypatch.setattr(torch, "save", boom, raising=True)
    monkeypatch.setattr(np, "savez_compressed", boom, raising=True)
    with pytest.raises(signal_type) as info:
        backend.prepare_fit_artifacts(pair, result)
    assert info.value is signal


@pytest.mark.parametrize("signal_type", _SIGNAL_TYPES, ids=_SIGNAL_IDS)
def test_signal_propagates_from_verification(stage, monkeypatch, signal_type):
    from atlas_sers.evaluation import p03_runtime

    pair = stage["selected"]["C-RBF-SVM"]
    artifacts = stage["artifacts"]["C-RBF-SVM"]
    signal = signal_type("stage-backend")

    def boom(*args, **kwargs):
        raise signal

    monkeypatch.setattr(p03_runtime, "_aligned_scores", boom, raising=True)
    with pytest.raises(signal_type) as info:
        backend.verify_source_prediction(
            pair,
            artifacts,
            saved_artifact_bytes=artifacts.artifact_bytes(),
            device="cpu",
        )
    assert info.value is signal


def test_invoke_neural_copies_inputs_without_fit(stage, monkeypatch):
    from atlas_sers.evaluation import p05_development

    pair = stage["selected"]["D0-M"]
    originals = dict(pair.neural_kwargs())
    sentinel = object()
    callback = object()
    calls = {"count": 0}
    captured = {}

    def stub_train(*args, **kwargs):
        calls["count"] += 1
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(p05_development, "train_development_fit", stub_train, raising=True)
    result = backend.invoke_source_fit(pair, device="cpu", global_deadline=123.5, on_epoch=callback)
    assert result is sentinel
    assert calls["count"] == 1
    kwargs = captured["kwargs"]
    assert set(kwargs) == set(originals) | {"device", "global_deadline", "on_epoch"}

    for name in ("values", "validation_values"):
        source = originals[name]
        copied = kwargs[name]
        assert isinstance(copied, np.ndarray)
        assert copied.dtype == np.float32
        assert copied.flags["C_CONTIGUOUS"]
        assert copied.flags.writeable
        assert np.array_equal(copied, np.asarray(source, dtype=np.float32))
        assert not np.shares_memory(copied, source)
        snapshot = np.array(source, dtype=np.float32, copy=True)
        copied[...] = np.float32(0.0)
        assert np.array_equal(np.asarray(source, dtype=np.float32), snapshot)

    for name, source in originals.items():
        if name in ("values", "validation_values"):
            continue
        value = kwargs[name]
        if isinstance(source, pd.DataFrame):
            assert isinstance(value, pd.DataFrame)
            pd.testing.assert_frame_equal(value, source)
        elif isinstance(source, np.ndarray):
            assert isinstance(value, np.ndarray)
            assert np.array_equal(value, source)
        elif isinstance(source, (list, tuple)):
            assert isinstance(value, (list, tuple))
            assert len(value) == len(source)
            for left, right in zip(value, source, strict=True):
                if isinstance(right, np.ndarray):
                    assert np.array_equal(left, right)
                else:
                    assert left == right
        else:
            assert value == source

    assert kwargs["device"] == "cpu"
    assert kwargs["global_deadline"] == 123.5
    assert kwargs["on_epoch"] is callback


CORRUPTED_AUDIT_CASES = [
    ("unique-master-hash", "C-RBF-SVM", "master_uid_sha256"),
    ("domain-hash", "C-EXTRA-TREES", "domain_uid_sha256"),
]


@pytest.mark.parametrize(
    "case",
    CORRUPTED_AUDIT_CASES,
    ids=[item[0] for item in CORRUPTED_AUDIT_CASES],
)
def test_classical_corrupted_fit_audit_rejected(stage, case):
    _label, model_id, field = case
    pair = stage["selected"][model_id]
    result = stage["results"][model_id]
    assert stage["artifacts"][model_id].status == "succeeded"
    original_audit = result.estimator.fit_audit
    copied = copy.copy(result.estimator)
    copied.fit_audit = dataclasses.replace(original_audit, **{field: "0" * 64})
    broken = dataclasses.replace(result, estimator=copied)
    with pytest.raises(backend.StageError) as info:
        backend.prepare_fit_artifacts(pair, broken)
    assert info.value.reason_code == "fit_pair_mismatch"
    assert result.estimator.fit_audit is original_audit


@pytest.mark.parametrize("model_id", ["C-RBF-SVM", "D0-M"])
def test_reversed_caller_validation_refused(stage, model_id):
    pair = stage["selected"][model_id]
    result = stage["results"][model_id]
    assert len(pair.validation_observations) >= 2
    reversed_pair = dataclasses.replace(
        pair, validation_observations=tuple(reversed(pair.validation_observations))
    )
    with pytest.raises(backend.StageError) as info:
        backend.prepare_fit_artifacts(reversed_pair, result)
    assert info.value.reason_code == "validation_uid_mismatch"
    assert stage["artifacts"][model_id].status == "succeeded"


def test_require_scientific_execution_always_denies():
    calls = (
        lambda: backend.require_scientific_execution(),
        lambda: backend.require_scientific_execution(True),
        lambda: backend.require_scientific_execution(execution_authorized=True),
        lambda: backend.require_scientific_execution(None, authorized=True, token="forged"),
    )
    for call in calls:
        with pytest.raises(backend.StageError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"
