"""Focused tests for the compact post-source FIT composition backend."""

from __future__ import annotations

import dataclasses
import io
import pickle
import time

import pandas as pd
import pytest
import torch

from atlas_sers.evaluation import p08_plan
from atlas_sers.evaluation import p08_u1_fit_backend as backend

RECIPE = "D1"
NEURAL_EPOCHS = 30
NEURAL_OPTIMIZER_STEPS = backend.OPTIMIZER_STEPS_PER_EPOCH * NEURAL_EPOCHS
HASH64 = "0" * 64
POLICY = "PP-U-SG"


def _fields(
    model_id,
    stage,
    unit_id,
    seed,
    resolution,
    *,
    policy=POLICY,
    validation_uid_sha256=HASH64,
    test_uid_sha256=HASH64,
    fit_uid_sha256=HASH64,
):
    return {
        "policy_id": policy,
        "representation_id": p08_plan.POLICY_REPRESENTATION[policy],
        "array_sha256": HASH64,
        "context_id": "CTX-1",
        "model_id": model_id,
        "model_spec_sha256": HASH64,
        "stage": stage,
        "unit_id": unit_id,
        "seed": seed,
        "candidate_id": "candidate-x",
        "hyperparameter_sha256": HASH64,
        "fit_uid_sha256": fit_uid_sha256,
        "validation_uid_sha256": validation_uid_sha256,
        "test_uid_sha256": test_uid_sha256,
        "resolution": resolution,
        "evidence_status": p08_plan.EVIDENCE_FUTURE,
    }


def _job(**fields):
    dependencies = fields.pop("dependencies", ())
    fields.setdefault("unit_id", "unit-1")
    fields.setdefault("resolution", "native")
    return p08_plan._new_job(_fields(**fields), list(dependencies))


def _metadata(unit):
    return pd.DataFrame(
        [
            {
                "observation_uid": f"{unit}-uid-0",
                "master_sample_id": f"{unit}-master-0",
                "instrument": f"{unit}-inst-0",
                "station": "ST-1",
                "target_analyte": "A",
            },
            {
                "observation_uid": f"{unit}-uid-1",
                "master_sample_id": f"{unit}-master-1",
                "instrument": f"{unit}-inst-1",
                "station": "ST-1",
                "target_analyte": "B",
            },
            {
                "observation_uid": f"{unit}-uid-2",
                "master_sample_id": f"{unit}-master-2",
                "instrument": f"{unit}-inst-2",
                "station": "ST-1",
                "target_analyte": "C",
            },
        ]
    )


def _calibration_job(**overrides):
    fields = {
        "stage": "calibration_model_fit",
        "model_id": "C-RBF-SVM",
        "seed": "deterministic",
    }
    fields.update(overrides)
    return _job(**fields)


def _final_classical_job(**overrides):
    fields = {
        "stage": "final_refit",
        "model_id": "C-RANDOM-FOREST",
        "seed": 20260805,
    }
    fields.update(overrides)
    return _job(**fields)


def _neural_job(**overrides):
    fields = {
        "stage": "final_refit",
        "model_id": "D1",
        "seed": 20260805,
    }
    fields.update(overrides)
    return _job(**fields)


def _calibration_kwargs(job, **overrides):
    base = {
        "dataset": "dataset",
        "fit_id": job["job_id"],
        "model_id": job["model_id"],
        "candidate_id": "cand-1",
        "parameters": {"C": 1.0},
        "seed": job["seed"],
        "fit_uids": ("fit-a",),
        "expected_fit_uid_sha256": HASH64,
        "validation_uids": ("val-a",),
        "class_vocabulary": ("a", "b", "c"),
        "expected_validation_uid_sha256": HASH64,
    }
    base.update(overrides)
    return base


def _final_kwargs(job, **overrides):
    base = {
        "dataset": "dataset",
        "fit_id": job["job_id"],
        "model_id": job["model_id"],
        "candidate_id": "cand-1",
        "parameters": {"C": 1.0},
        "seed": job["seed"],
        "fit_uids": ("fit-a",),
        "expected_fit_uid_sha256": HASH64,
    }
    base.update(overrides)
    return base


@dataclasses.dataclass
class _FitAudit:
    rows: int = 3
    note: str = "audit"


@dataclasses.dataclass
class _TinyEstimator:
    weights: tuple = (1, 2, 3)
    fit_audit: object = None


def _estimator():
    return _TinyEstimator(fit_audit=_FitAudit())


def _candidate_outcome(job, **overrides):
    base = {
        "fit_id": job["job_id"],
        "status": "complete",
        "reason_code": None,
        "model_id": job["model_id"],
        "candidate_id": "cand-1",
        "seed": job["seed"],
        "fit_uid_sha256": HASH64,
        "validation_uid_sha256": HASH64,
        "fit_master_sha256": "m" * 64,
        "elapsed_seconds": 0.5,
        "inference_seconds": None,
        "serialized_model_bytes": None,
        "warnings": [],
        "traceback_digest": None,
        "validation_predictions": pd.DataFrame({"y": [1, 2]}),
        "validation_metrics": {"accuracy": 1.0},
        "estimator": _estimator(),
    }
    base.update(overrides)
    return backend.CandidateFitOutcome(**base)


def _final_outcome(job, **overrides):
    base = {
        "fit_id": job["job_id"],
        "status": "complete",
        "reason_code": None,
        "model_id": job["model_id"],
        "candidate_id": "cand-1",
        "seed": job["seed"],
        "fit_uid_sha256": HASH64,
        "fit_master_sha256": "m" * 64,
        "elapsed_seconds": 0.5,
        "serialized_model_bytes": None,
        "warnings": [],
        "traceback_digest": None,
        "estimator": _estimator(),
        "fit_label_sha256": None,
    }
    base.update(overrides)
    return backend.FinalFitOutcome(**base)


@dataclasses.dataclass
class _Observation:
    target: str
    uid: str
    noise_level: float = 0.0


def _observations(unit="unit-1"):
    metadata = _metadata(unit)
    return [
        _Observation(row.target_analyte.lower(), row.observation_uid, 0.1 * index)
        for index, row in enumerate(metadata.itertuples(index=False))
    ]


def _neural_kwargs(job, **overrides):
    observations = _observations()
    base = {
        "values": torch.zeros((3, 1401), dtype=torch.float32),
        "observations": observations,
        "noise_metadata": pd.DataFrame(
            {
                "observation_uid": [observation.uid for observation in observations],
                "noise_level": [observation.noise_level for observation in observations],
            }
        ),
        "role_id": "P04-role",
        "recipe": RECIPE,
        "seed": job["seed"],
        "epochs": NEURAL_EPOCHS,
        "maximum_fit_seconds": 120.0,
        "maximum_cuda_allocated_bytes": 4 * 2**30,
    }
    base.update(overrides)
    return base


def _history():
    return [{"loss": 0.5} for _ in range(NEURAL_EPOCHS)]


def _refit_result(job, **overrides):
    base = {
        "status": "complete",
        "reason_code": None,
        "history": _history(),
        "epochs": NEURAL_EPOCHS,
        "epochs_completed": NEURAL_EPOCHS,
        "parameter_count": 0,
        "optimizer_steps": NEURAL_OPTIMIZER_STEPS,
        "zero_gradient_batches": 0,
        "initial_state_digest": None,
        "terminal_state_digest": None,
        "initial_backbone_digest": None,
        "terminal_backbone_digest": None,
        "initial_head_digest": None,
        "terminal_head_digest": None,
        "terminal_state_dict": None,
        "state_capture_failed": False,
        "classes": ("a", "b", "c"),
        "source_noise_levels": (0.0, 0.1, 0.2),
        "augmentation_digest": None,
        "sampling_digest": None,
        "pair_digest": None,
        "finite_gradient_batches": 1,
        "nonzero_gradient_elements": 1,
        "supcon_support": {},
        "paired_support": {},
        "role_id": "P04-role",
        "recipe": RECIPE,
        "seed": job["seed"],
        "elapsed_seconds": 0.1,
        "peak_cuda_bytes": 0,
        "traceback_digest": None,
    }
    base.update(overrides)
    return backend.RefitResult(**base)


def _neural_success(job, **overrides):
    state = {"w": torch.ones(2, dtype=torch.float32)}
    base = {
        "recipe": RECIPE,
        "seed": job["seed"],
        "role_id": "P04-role",
        "epochs": NEURAL_EPOCHS,
        "epochs_completed": NEURAL_EPOCHS,
        "history": _history(),
        "optimizer_steps": NEURAL_OPTIMIZER_STEPS,
        "terminal_state_dict": state,
        "terminal_state_digest": backend._state_hash(state),
        "parameter_count": backend._expected_parameters(RECIPE),
    }
    base.update(overrides)
    return _refit_result(job, **base)


class _FakeInputs:
    def __init__(self, *, classical=None, neural=None):
        self.classical = None if classical is None else dict(classical)
        self.neural = None if neural is None else dict(neural)
        self.classical_calls = []
        self.neural_calls = []
        self.held_calls = 0

    def classical_fit_kwargs(self, job, selection):
        self.classical_calls.append((job, selection))
        return dict(self.classical)

    def neural_refit_kwargs(self, job, epochs):
        self.neural_calls.append((job, epochs))
        return dict(self.neural)

    @property
    def held_inputs(self):
        self.held_calls += 1
        raise AssertionError("held inputs must not be accessed during fitting")


def test_calibration_invokes_candidate_fit_once(monkeypatch):
    job = _calibration_job()
    kwargs = _calibration_kwargs(job)
    inputs = _FakeInputs(classical=kwargs)
    received = []

    def spy(**call_kwargs):
        received.append(call_kwargs)
        return _candidate_outcome(job)

    monkeypatch.setattr(backend, "run_candidate_fit", spy)
    bundle = backend.invoke_fit(job, inputs, selection="sel")
    assert received == [kwargs]
    assert inputs.classical_calls == [(job, "sel")]
    assert inputs.held_calls == 0
    assert bundle.status == "complete"
    assert bundle.job["job_id"] == job["job_id"]
    payload = bundle.artifact_bytes()
    assert set(payload) == {"summary.json", "predictions.csv"}
    assert b"validation_metrics" in payload["summary.json"]
    assert b"fit_audit" in payload["summary.json"]


def test_failed_calibration_keeps_status_and_omits_empty_predictions(monkeypatch):
    job = _calibration_job()
    kwargs = _calibration_kwargs(job)
    inputs = _FakeInputs(classical=kwargs)
    failed = _candidate_outcome(
        job,
        status="fit_failure",
        reason_code="boom",
        estimator=None,
        validation_predictions=pd.DataFrame({"y": []}),
        validation_metrics=None,
    )
    monkeypatch.setattr(backend, "run_candidate_fit", lambda **call_kwargs: failed)
    bundle = backend.invoke_fit(job, inputs)
    assert bundle.status == "fit_failure"
    payload = bundle.artifact_bytes()
    assert "summary.json" in payload
    assert "predictions.csv" not in payload


def test_final_classical_pickle_roundtrip(monkeypatch):
    job = _final_classical_job()
    kwargs = _final_kwargs(job)
    inputs = _FakeInputs(classical=kwargs)
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: _final_outcome(job))
    bundle = backend.invoke_fit(job, inputs)
    payload = bundle.artifact_bytes()
    assert set(payload) == {"summary.json", "estimator.pkl"}
    loaded = pickle.loads(payload["estimator.pkl"])
    assert loaded.weights == (1, 2, 3)
    assert loaded.fit_audit == _FitAudit()
    assert b"validation_metrics" not in payload["summary.json"]


def test_neural_final_invokes_train_refit_once(monkeypatch):
    job = _neural_job()
    kwargs = _neural_kwargs(job)
    inputs = _FakeInputs(neural=kwargs)
    result = _neural_success(job)
    received = []

    def spy(**call_kwargs):
        received.append(call_kwargs)
        return result

    monkeypatch.setattr(backend, "train_refit", spy)
    callback = object()
    bundle = backend.invoke_fit(
        job,
        inputs,
        epochs=NEURAL_EPOCHS,
        device="cpu",
        global_deadline=None,
        on_epoch=callback,
    )
    assert len(received) == 1
    assert received[0]["on_epoch"] is callback
    assert received[0]["device"] == "cpu"
    assert received[0]["global_deadline"] is None
    assert received[0]["recipe"] == kwargs["recipe"]
    assert received[0]["epochs"] == NEURAL_EPOCHS
    assert inputs.held_calls == 0
    payload = bundle.artifact_bytes()
    assert set(payload) == {"summary.json", "terminal.pt"}
    assert b"terminal_state_dict" not in payload["summary.json"]
    assert b"fit_job_id" in payload["summary.json"]
    loaded = torch.load(io.BytesIO(payload["terminal.pt"]), weights_only=True)
    assert set(loaded) == {"w"}


def test_failed_neural_retains_partial_terminal_bytes(monkeypatch):
    job = _neural_job()
    kwargs = _neural_kwargs(job)
    inputs = _FakeInputs(neural=kwargs)
    state = {"w": torch.ones(1, dtype=torch.float32)}
    failed = _refit_result(
        job,
        status="fit_failure",
        reason_code="boom",
        epochs=NEURAL_EPOCHS,
        epochs_completed=0,
        history=[],
        optimizer_steps=0,
        terminal_state_dict=state,
        terminal_state_digest=None,
        parameter_count=0,
        traceback_digest="t" * 64,
    )
    monkeypatch.setattr(backend, "train_refit", lambda **call_kwargs: failed)
    bundle = backend.invoke_fit(job, inputs, epochs=NEURAL_EPOCHS)
    assert bundle.status == "fit_failure"
    payload = bundle.artifact_bytes()
    assert "terminal.pt" in payload
    assert payload["terminal.pt"]


def test_calibration_identity_mismatch_rejects(monkeypatch):
    job = _calibration_job()
    kwargs = _calibration_kwargs(job)
    inputs = _FakeInputs(classical=kwargs)
    bad = _candidate_outcome(job, model_id="C-EXTRA-TREES")
    monkeypatch.setattr(backend, "run_candidate_fit", lambda **call_kwargs: bad)
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs)


def test_neural_state_digest_mismatch_rejects(monkeypatch):
    job = _neural_job()
    kwargs = _neural_kwargs(job)
    inputs = _FakeInputs(neural=kwargs)
    state = {"w": torch.ones(2, dtype=torch.float32)}
    bad = _neural_success(
        job, terminal_state_dict=state, terminal_state_digest="deadbeef"
    )
    monkeypatch.setattr(backend, "train_refit", lambda **call_kwargs: bad)
    with pytest.raises(backend.FitBackendError) as caught:
        backend.invoke_fit(job, inputs, epochs=NEURAL_EPOCHS)
    assert caught.value.result is bad
    assert set(caught.value.partial_artifacts) == {"summary.json", "terminal.pt"}


def test_expired_deadline_never_calls_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: calls.append(1))
    job = _final_classical_job(model_id="C-RBF-SVM", seed="deterministic")
    inputs = _FakeInputs(classical=_final_kwargs(job))
    with pytest.raises(TimeoutError):
        backend.invoke_fit(job, inputs, global_deadline=time.perf_counter() - 1.0)
    assert calls == []


def test_after_fit_deadline_sets_stop_flag(monkeypatch):
    job = _final_classical_job(model_id="C-RBF-SVM", seed="deterministic")
    inputs = _FakeInputs(classical=_final_kwargs(job))

    def spy(**call_kwargs):
        time.sleep(0.1)
        return _final_outcome(job)

    monkeypatch.setattr(backend, "run_final_fit", spy)
    deadline = time.perf_counter() + 0.05
    bundle = backend.invoke_fit(job, inputs, global_deadline=deadline)
    assert bundle.deadline_exceeded is True
    assert bundle.status == "complete"
    assert "estimator.pkl" in bundle.artifact_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"stage": "nonsense"},
        {"seed": 123456},
        {"policy_id": "PP-BAD"},
        {"representation_id": "wrong"},
    ],
)
def test_bad_job_rejected_before_invoke(monkeypatch, overrides):
    calls = []
    monkeypatch.setattr(backend, "run_candidate_fit", lambda **call_kwargs: calls.append(1))
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: calls.append(1))
    monkeypatch.setattr(backend, "train_refit", lambda **call_kwargs: calls.append(1))
    original = _final_classical_job()
    fields = {k: v for k, v in original.items() if k not in {"job_id", "dependencies"}}
    fields.update(overrides)
    job = p08_plan._new_job(fields, original["dependencies"])
    inputs = _FakeInputs(
        classical=_final_kwargs(job),
        neural=_neural_kwargs(_neural_job()),
    )
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs, epochs=NEURAL_EPOCHS)
    assert calls == []


def test_bad_job_id_rejected_before_invoke(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: calls.append(1))
    job = _final_classical_job(model_id="C-RBF-SVM", seed="deterministic")
    job["job_id"] = "P08JOB-deadbeef"
    inputs = _FakeInputs(classical=_final_kwargs(job))
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs)
    assert calls == []


def test_calibration_rejects_neural_model(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "train_refit", lambda **call_kwargs: calls.append(1))
    job = _calibration_job(model_id="D1", seed=20260805)
    inputs = _FakeInputs(neural=_neural_kwargs(_neural_job()))
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs, epochs=NEURAL_EPOCHS)
    assert calls == []


def test_artifact_bytes_returns_fresh_dict(monkeypatch):
    job = _final_classical_job(model_id="C-RBF-SVM", seed="deterministic")
    inputs = _FakeInputs(classical=_final_kwargs(job))
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: _final_outcome(job))
    bundle = backend.invoke_fit(job, inputs)
    first = bundle.artifact_bytes()
    first["injected"] = b"x"
    second = bundle.artifact_bytes()
    assert second is not first
    assert "injected" not in second


def test_classical_provider_identity_checked_before_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "run_final_fit", lambda **call_kwargs: calls.append(1))
    job = _final_classical_job(model_id="C-RBF-SVM", seed="deterministic")
    inputs = _FakeInputs(classical=_final_kwargs(job, fit_id="fit-1"))
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs)
    assert calls == []


def test_neural_provider_identity_checked_before_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "train_refit", lambda **call_kwargs: calls.append(1))
    job = _neural_job()
    inputs = _FakeInputs(neural=_neural_kwargs(job, seed=20260817))
    with pytest.raises(backend.FitBackendError):
        backend.invoke_fit(job, inputs, epochs=NEURAL_EPOCHS)
    assert calls == []
