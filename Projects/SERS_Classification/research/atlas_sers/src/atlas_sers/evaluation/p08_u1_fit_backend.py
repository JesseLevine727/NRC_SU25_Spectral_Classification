"""Compact post-source FIT composition for P08 U1.

The backend validates one planned job, invokes exactly one existing fit kernel
once and packages the genuine result with in-memory artifacts. It performs no
filesystem access, no retries, no metric recomputation and no pickle loading.
The retained result is for dependent classical calibration verification only and
is never treated as a capability.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import pickle
import time
from collections.abc import Mapping
from typing import Any

import torch

from atlas_sers.evaluation.p03_runtime import (
    CandidateFitOutcome,
    FinalFitOutcome,
    run_candidate_fit,
    run_final_fit,
)
from atlas_sers.evaluation.p04_runtime import _state_hash
from atlas_sers.evaluation.p05_refit import RefitResult, train_refit
from atlas_sers.evaluation.p05_smoke import RECIPE_SPECIFICATIONS
from atlas_sers.governance.canonical import canonical_json_bytes
from atlas_sers.models.acquisition import BASE_PARAMETERS, PROJECTION_MODEL_PARAMETERS

from . import p08_plan

CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
NEURAL_MODELS = ("D0-M", "D1", "D2", "D3")
ALLOWED_MODELS = CLASSICAL_MODELS + NEURAL_MODELS
CALIBRATION_STAGE = "calibration_model_fit"
FINAL_STAGE = "final_refit"
ALLOWED_STAGES = (CALIBRATION_STAGE, FINAL_STAGE)
ALLOWED_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
DETERMINISTIC_SVM_SEED = "deterministic"
REPLICA_SEEDS = (20260805, 20260817, 20260829)
OPTIMIZER_STEPS_PER_EPOCH = 4
_SUCCESS_STATUSES = frozenset({"complete"})
_MISSING = object()


class FitBackendError(ValueError):
    """Raised when a planned job or a kernel receipt violates the contract."""


@dataclasses.dataclass(frozen=True)
class FitBundle:
    """Immutable receipt for one composed fit invocation."""

    job_json: bytes
    result: Any
    status: str
    artifacts: tuple[tuple[str, bytes], ...]
    deadline_exceeded: bool = False

    @property
    def job(self) -> dict[str, Any]:
        """Return a fresh decoded snapshot of the planned job."""
        return json.loads(self.job_json)

    def artifact_bytes(self) -> dict[str, bytes]:
        """Return a fresh filename -> bytes mapping for this bundle."""
        return dict(self.artifacts)


def _canonical_bytes(value: Any) -> bytes:
    raw = canonical_json_bytes(value)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return bytes(raw)


def _job_fields(job: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in job.items() if name != "job_id"}


def _compute_job_id(job: Mapping[str, Any]) -> str:
    return "P08JOB-" + p08_plan._hash(_job_fields(job))


def _validate_seed(model_id: str, seed: Any) -> None:
    if model_id == "C-RBF-SVM":
        if seed != DETERMINISTIC_SVM_SEED:
            raise FitBackendError("the RBF SVM seed is deterministic")
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FitBackendError("job seed must be an integer")
    if seed not in REPLICA_SEEDS:
        raise FitBackendError(f"seed is not an allowed replica seed: {seed!r}")


def _validate_job(job: Any) -> None:
    if not isinstance(job, Mapping):
        raise FitBackendError("job must be a mapping")
    missing = [name for name in p08_plan.JOB_FIELDS if name not in job]
    if missing:
        raise FitBackendError(f"job is missing required fields: {missing}")
    unknown = sorted(set(job) - set(p08_plan.JOB_FIELDS) - {"job_id"})
    if unknown:
        raise FitBackendError(f"job contains unknown fields: {unknown}")
    if job.get("job_id") != _compute_job_id(job):
        raise FitBackendError("job_id does not match the job field hash")
    policy = job.get("policy_id")
    if policy not in ALLOWED_POLICIES:
        raise FitBackendError(f"policy is not allowed: {policy!r}")
    if job.get("representation_id") != p08_plan.POLICY_REPRESENTATION[policy]:
        raise FitBackendError("policy representation does not match policy")
    stage = job.get("stage")
    if stage not in ALLOWED_STAGES:
        raise FitBackendError(f"stage is not allowed: {stage!r}")
    model_id = job.get("model_id")
    if model_id not in ALLOWED_MODELS:
        raise FitBackendError(f"model_id is not allowed: {model_id!r}")
    if stage == CALIBRATION_STAGE and model_id not in CLASSICAL_MODELS:
        raise FitBackendError("calibration_model_fit only supports classical models")
    _validate_seed(model_id, job.get("seed"))


def _check_deadline_before(global_deadline: Any) -> None:
    if global_deadline is None:
        return
    if isinstance(global_deadline, bool) or not isinstance(global_deadline, (int, float)):
        raise FitBackendError("global_deadline must be a real number")
    if not math.isfinite(global_deadline):
        raise FitBackendError("global_deadline must be finite")
    if time.perf_counter() >= global_deadline:
        raise TimeoutError("global deadline expired before the fit kernel was invoked")


def _deadline_exceeded(global_deadline: Any) -> bool:
    return global_deadline is not None and time.perf_counter() >= global_deadline


def _result_status(result: Any) -> str:
    status = getattr(result, "status", None)
    if not isinstance(status, str) or not status:
        raise FitBackendError("fit kernel returned no terminal status")
    return status


def _is_success(status: str) -> bool:
    return status in _SUCCESS_STATUSES


def _validate_classical_arguments(
    job: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> None:
    if not isinstance(kwargs, Mapping):
        raise FitBackendError("classical fit arguments must be a mapping")
    if kwargs.get("fit_id") != job["job_id"]:
        raise FitBackendError("classical fit_id does not match the job id")
    if kwargs.get("model_id") != job["model_id"]:
        raise FitBackendError("classical model_id does not match the job")
    if kwargs.get("seed") != job["seed"]:
        raise FitBackendError("classical seed does not match the job")


def _validate_neural_arguments(
    job: Mapping[str, Any], kwargs: Mapping[str, Any], epochs: int | None
) -> None:
    if not isinstance(kwargs, Mapping):
        raise FitBackendError("neural refit arguments must be a mapping")
    if kwargs.get("recipe") != job["model_id"]:
        raise FitBackendError("neural refit recipe is not a known recipe")
    if kwargs.get("seed") != job["seed"]:
        raise FitBackendError("neural refit seed does not match the job")
    if type(epochs) is not int or not 30 <= epochs <= 200 or kwargs.get("epochs") != epochs:
        raise FitBackendError("neural refit epochs do not match the request")


def _classical_summary(result: Any, *, with_validation: bool) -> bytes:
    summary: dict[str, Any] = dict(result.status_record())
    if with_validation:
        summary["validation_metrics"] = getattr(result, "validation_metrics", None)
    estimator = getattr(result, "estimator", None)
    audit = getattr(estimator, "fit_audit", None)
    if audit is not None:
        try:
            summary["fit_audit"] = dataclasses.asdict(audit)
        except TypeError:
            summary["fit_audit"] = audit
    return _canonical_bytes(summary)


def _calibration_artifacts(result: Any) -> list[tuple[str, bytes]]:
    items = [("summary.json", _classical_summary(result, with_validation=True))]
    predictions = getattr(result, "validation_predictions", None)
    if predictions is not None and len(predictions) > 0:
        items.append(("predictions.csv", predictions.to_csv(index=False).encode("utf-8")))
    return items


def _final_artifacts(result: Any, *, success: bool) -> list[tuple[str, bytes]]:
    items = [("summary.json", _classical_summary(result, with_validation=False))]
    if success:
        estimator = getattr(result, "estimator", None)
        if estimator is not None:
            payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
            items.append(("estimator.pkl", payload))
    return items


def _neural_summary(result: Any, job: Mapping[str, Any]) -> bytes:
    summary: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        if field.name == "terminal_state_dict":
            continue
        summary[field.name] = getattr(result, field.name)
    summary["fit_job_id"] = job["job_id"]
    return _canonical_bytes(summary)


def _neural_artifacts(result: Any, job: Mapping[str, Any]) -> list[tuple[str, bytes]]:
    items = [("summary.json", _neural_summary(result, job))]
    state = getattr(result, "terminal_state_dict", None)
    if state is not None:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        items.append(("terminal.pt", buffer.getvalue()))
    return items


def _require_calibration_success(result: Any, kwargs: Mapping[str, Any]) -> None:
    if not isinstance(result, CandidateFitOutcome):
        raise FitBackendError("calibration kernel returned an unexpected result type")
    if getattr(result, "estimator", None) is None:
        raise FitBackendError("successful calibration requires an estimator")
    for attribute, key in (
        ("fit_id", "fit_id"),
        ("model_id", "model_id"),
        ("candidate_id", "candidate_id"),
        ("seed", "seed"),
        ("fit_uid_sha256", "expected_fit_uid_sha256"),
        ("validation_uid_sha256", "expected_validation_uid_sha256"),
    ):
        if getattr(result, attribute) != kwargs[key]:
            raise FitBackendError(f"calibration result {attribute} does not match input")
    predictions = getattr(result, "validation_predictions", None)
    if predictions is None or len(predictions) == 0:
        raise FitBackendError("successful calibration requires validation predictions")


def _require_final_success(result: Any, kwargs: Mapping[str, Any]) -> None:
    if not isinstance(result, FinalFitOutcome):
        raise FitBackendError("final kernel returned an unexpected result type")
    if getattr(result, "estimator", None) is None:
        raise FitBackendError("successful final fit requires an estimator")
    for attribute, key in (
        ("fit_id", "fit_id"),
        ("model_id", "model_id"),
        ("candidate_id", "candidate_id"),
        ("seed", "seed"),
        ("fit_uid_sha256", "expected_fit_uid_sha256"),
    ):
        if getattr(result, attribute) != kwargs[key]:
            raise FitBackendError(f"final result {attribute} does not match input")


def _observation_classes(observations: Any) -> tuple[Any, ...]:
    classes = []
    for observation in observations:
        target = getattr(observation, "target", _MISSING)
        if target is _MISSING:
            raise FitBackendError("source observation exposes no target")
        classes.append(target)
    return tuple(sorted(set(classes)))


def _expected_parameters(recipe: str) -> int:
    try:
        specification = RECIPE_SPECIFICATIONS[recipe]
    except KeyError as exc:
        raise FitBackendError(f"unknown recipe: {recipe!r}") from exc
    return PROJECTION_MODEL_PARAMETERS if specification[2] else BASE_PARAMETERS


def _state_is_finite(state: Mapping[str, Any]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in state.values())


def _require_finite(value: Any, *, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FitBackendError(f"non-finite value in neural result at {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, path=f"{path}[{index}]")


def _require_neural_success(result: Any, kwargs: Mapping[str, Any]) -> None:
    if not isinstance(result, RefitResult):
        raise FitBackendError("neural kernel returned an unexpected result type")
    epochs = kwargs["epochs"]
    if result.recipe != kwargs["recipe"]:
        raise FitBackendError("neural result recipe does not match input")
    if result.seed != kwargs["seed"]:
        raise FitBackendError("neural result seed does not match input")
    if result.role_id != kwargs["role_id"]:
        raise FitBackendError("neural result role_id does not match input")
    if result.epochs != epochs:
        raise FitBackendError("neural result epochs do not match input")
    if result.epochs_completed != epochs:
        raise FitBackendError("neural result did not complete every epoch")
    if len(result.history) != epochs:
        raise FitBackendError("neural result history length does not match epochs")
    if result.optimizer_steps != OPTIMIZER_STEPS_PER_EPOCH * epochs:
        raise FitBackendError("neural result optimizer_steps does not match epochs")
    expected_classes = _observation_classes(kwargs["observations"])
    if len(expected_classes) != 3 or result.classes != expected_classes:
        raise FitBackendError("neural result classes do not match source observations")
    if result.state_capture_failed:
        raise FitBackendError("neural result reports a failed state capture")
    state = result.terminal_state_dict
    if not state:
        raise FitBackendError("successful neural fit requires a terminal state")
    if not _state_is_finite(state):
        raise FitBackendError("neural terminal state contains non-finite values")
    if _state_hash(state) != result.terminal_state_digest:
        raise FitBackendError("neural terminal state digest does not match")
    if result.parameter_count != _expected_parameters(kwargs["recipe"]):
        raise FitBackendError("neural result parameter_count does not match recipe")
    for field in dataclasses.fields(result):
        if field.name == "terminal_state_dict":
            continue
        _require_finite(getattr(result, field.name), path=field.name)


def invoke_fit(
    job: Mapping[str, Any],
    inputs: Any,
    *,
    selection: Any = None,
    epochs: int | None = None,
    device: str = "cpu",
    global_deadline: float | None = None,
    on_epoch: Any = None,
) -> FitBundle:
    """Validate one job, invoke exactly one fit kernel once and package results."""
    _validate_job(job)
    _check_deadline_before(global_deadline)
    stage = job["stage"]
    model_id = job["model_id"]

    if stage == CALIBRATION_STAGE:
        kwargs = inputs.classical_fit_kwargs(job, selection)
        _validate_classical_arguments(job, kwargs)
        result = run_candidate_fit(**kwargs)
    elif model_id in CLASSICAL_MODELS:
        kwargs = inputs.classical_fit_kwargs(job, selection)
        _validate_classical_arguments(job, kwargs)
        result = run_final_fit(**kwargs)
    else:
        kwargs = inputs.neural_refit_kwargs(job, epochs)
        _validate_neural_arguments(job, kwargs, epochs)
        result = train_refit(
            **kwargs,
            device=device,
            global_deadline=global_deadline,
            on_epoch=on_epoch,
        )
    artifacts = []
    try:
        status = _result_status(result)
        if stage == CALIBRATION_STAGE:
            artifacts = _calibration_artifacts(result)
            if _is_success(status):
                _require_calibration_success(result, kwargs)
        elif model_id in CLASSICAL_MODELS:
            artifacts = _final_artifacts(result, success=_is_success(status))
            if _is_success(status):
                _require_final_success(result, kwargs)
        else:
            artifacts = _neural_artifacts(result, job)
            if _is_success(status):
                _require_neural_success(result, kwargs)
    except Exception as error:
        # A completed numerical attempt is never erased by a later acceptance error.
        # The dispatcher persists these bytes under FAILED, never as reusable success.
        error.result = result
        error.partial_artifacts = dict(artifacts)
        raise

    return FitBundle(
        job_json=_canonical_bytes(job),
        result=result,
        status=status,
        artifacts=tuple(artifacts),
        deadline_exceeded=_deadline_exceeded(global_deadline),
    )
