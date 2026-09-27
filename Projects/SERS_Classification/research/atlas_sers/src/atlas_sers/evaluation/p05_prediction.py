"""P05 refit prediction: bounded, no-optimizer inference for one completed refit.

Reconstructs one completed refit from disk, re-verifies its summary, terminal
checkpoint and calibration, then emits outer-test logits and probabilities.  It
performs no fitting, augmentation, calibration optimization, noise statistics or
writes.  The caller MUST authenticate the completed-all-refits stage first; this
standalone helper is not execution authority.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_refit_io as refit_io

EXPECTED_FEATURES = 1401
EXPECTED_CLASS_COUNT = refit_io.EXPECTED_CLASS_COUNT


class P05PredictionError(core.P05CoreError):
    """Stable, path-free prediction failure."""


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05PredictionError(code)


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise P05PredictionError("deadline_exceeded")


def _validate_inputs(values: Any, observation_uids: Any, spec: Mapping[str, Any]) -> list[str]:
    _require(isinstance(values, np.ndarray) and values.ndim == 2, "values_shape_malformed")
    _require(int(values.shape[1]) == EXPECTED_FEATURES, "values_shape_malformed")
    _require(values.dtype == np.float32 and bool(np.isfinite(values).all()), "values_malformed")
    count = int(values.shape[0])
    _require(count > 0, "values_empty")
    _require(
        isinstance(observation_uids, Sequence) and not isinstance(observation_uids, (str, bytes)),
        "uids_malformed",
    )
    uids = list(observation_uids)
    _require(
        len(uids) == count
        and all(isinstance(uid, str) and bool(uid) and uid == uid.strip() for uid in uids)
        and len(set(uids)) == count,
        "uids_malformed",
    )
    _require(not (set(uids) & set(spec["fitting_uids"])), "source_test_uid_overlap")
    return uids


def _load_result(unit_dir: Path, spec: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    refit = _module("atlas_sers.evaluation.p05_refit")
    summary = core._read_json(unit_dir / "summary.json", "refit_summary")
    _require(isinstance(summary, Mapping), "refit_summary_malformed")
    _require("terminal_state_dict" not in summary, "refit_summary_terminal_state_present")
    names = {field.name for field in dataclasses.fields(refit.RefitResult)}
    _require(names - {"terminal_state_dict"} <= set(summary), "refit_summary_fields_missing")
    _require(
        set(summary) - names <= set(refit_io.BINDING_FIELDS), "refit_summary_unexpected_fields"
    )
    kwargs = {
        name: summary[name] for name in names if name != "terminal_state_dict" and name in summary
    }
    _require(
        "classes" in kwargs and "source_noise_levels" in kwargs, "refit_summary_fields_missing"
    )
    kwargs["classes"] = tuple(str(value) for value in kwargs["classes"])
    kwargs["source_noise_levels"] = tuple(float(value) for value in kwargs["source_noise_levels"])
    result = refit.RefitResult(**kwargs)
    refit_io.check_completed_refit(torch, unit_dir, spec, result)
    runtime = _module("atlas_sers.evaluation.p04_runtime")
    path = unit_dir / "terminal.pt"
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "terminal_checkpoint_missing")
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    _require(isinstance(payload, Mapping), "terminal_checkpoint_malformed")
    state = dict(payload)
    expected = getattr(result, "terminal_state_digest", None)
    _require(
        isinstance(expected, str) and runtime._state_hash(state) == expected,
        "terminal_digest_mismatch",
    )
    return state, str(expected)


def _load_calibration(unit_dir: Path, spec: Mapping[str, Any]) -> tuple[Any, str]:
    classical = _module("atlas_sers.evaluation.classical")
    path = unit_dir / "calibration.json"
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "calibration_missing")
    wrapper = core._read_json(path, "calibration")
    _require(isinstance(wrapper, Mapping), "calibration_malformed")
    state_sha = wrapper.get("state_sha256")
    _require(isinstance(state_sha, str) and bool(state_sha), "calibration_state_sha_missing")
    raw = wrapper.get("state")
    _require(isinstance(raw, Mapping), "calibration_state_malformed")
    kwargs = dict(raw)
    _require("state_sha256" not in kwargs, "calibration_state_sha_not_property")
    classes = kwargs.get("class_vocabulary")
    _require(
        isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)),
        "calibration_classes_malformed",
    )
    kwargs["class_vocabulary"] = tuple(str(value) for value in classes)
    calibration = classical.TemperatureCalibration(**kwargs)
    _require(calibration.state_sha256 == state_sha, "calibration_state_sha_mismatch")
    temperature = getattr(calibration, "temperature", None)
    _require(
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(float(temperature))
        and float(temperature) > 0.0,
        "calibration_temperature_invalid",
    )
    _require(getattr(calibration, "optimizer_success", None) is True, "calibration_failed")
    objective = calibration.optimizer_objective
    _require(
        isinstance(objective, (int, float))
        and not isinstance(objective, bool)
        and math.isfinite(float(objective)),
        "calibration_objective_invalid",
    )
    _require(
        tuple(map(str, calibration.class_vocabulary)) == tuple(map(str, spec["classes"])),
        "calibration_classes_mismatch",
    )
    audit_path = unit_dir / "calibration_audit.json"
    core._reject_symlink_chain(audit_path)
    _require(audit_path.is_file() and not audit_path.is_symlink(), "calibration_audit_missing")
    audit = core._read_json(audit_path, "calibration_audit")
    _require(isinstance(audit, Mapping), "calibration_audit_malformed")
    _require(
        audit.get("calibration_state_sha256") == state_sha
        and all(
            str(audit.get(name)) == str(spec[name])
            for name in ("refit_id", "context_id", "recipe_id")
        )
        and audit.get("seed") == spec["seed"]
        and list(audit.get("calibration_slot_ids", ())) == list(spec["calibration_slot_ids"])
        and audit.get("temperature") == temperature
        and audit.get("optimizer_success") is True
        and audit.get("optimizer_objective") == objective,
        "calibration_audit_mismatch",
    )
    return calibration, str(state_sha)


def _build_model(spec: Mapping[str, Any], state: Mapping[str, Any], device: torch.device) -> Any:
    smoke = _module("atlas_sers.evaluation.p05_smoke")
    acquisition = _module("atlas_sers.models.acquisition")
    pilot = _module("atlas_sers.evaluation.p05_pilot")
    use_projection = bool(smoke.RECIPE_SPECIFICATIONS[spec["recipe_id"]][2])
    model = acquisition.AcquisitionClassifier(
        class_count=EXPECTED_CLASS_COUNT, use_projection=use_projection
    )
    model.load_state_dict(dict(state), strict=True)
    expected = int(
        acquisition.PROJECTION_MODEL_PARAMETERS if use_projection else acquisition.BASE_PARAMETERS
    )
    _require(
        int(sum(parameter.numel() for parameter in model.parameters())) == expected,
        "parameter_count_mismatch",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    if device.type == "cuda":
        pilot._enforce_cuda_cap(torch, "cuda")
    model.eval()
    return model


def predict_refit(
    *,
    spec: Any,
    unit_dir: Any,
    values: Any,
    observation_uids: Any,
    device: Any,
    deadline: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return calibrated outer-test logits and probabilities for one completed refit."""

    started = time.perf_counter()
    _require(
        isinstance(deadline, (int, float))
        and not isinstance(deadline, bool)
        and math.isfinite(float(deadline)),
        "deadline_malformed",
    )
    deadline = float(deadline)
    _check_deadline(deadline)
    smoke = _module("atlas_sers.evaluation.p05_smoke")
    smoke._configure_determinism()
    canonical = refit_io._check_spec(spec)
    unit_dir = Path(unit_dir)
    core._reject_symlink_chain(unit_dir)
    _require(not unit_dir.is_symlink() and unit_dir.is_dir(), "unit_directory_missing")
    uids = _validate_inputs(values, observation_uids, canonical)
    count = int(values.shape[0])
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        raise P05PredictionError("device_malformed") from None
    _require(torch_device.type in ("cpu", "cuda"), "device_malformed")
    _check_deadline(deadline)
    state, model_sha = _load_result(unit_dir, canonical)
    calibration, calibration_sha = _load_calibration(unit_dir, canonical)
    _check_deadline(deadline)
    model = _build_model(canonical, state, torch_device)
    tensor = torch.from_numpy(np.ascontiguousarray(values[:, None, :]))
    development = _module("atlas_sers.evaluation.p05_development")
    classical = _module("atlas_sers.evaluation.classical")
    _check_deadline(deadline)
    with torch.no_grad():
        logits = development._predict_logits(model, tensor, torch_device)
        probabilities = classical.apply_temperature(logits, calibration)
    _check_deadline(deadline)
    logits_array = np.asarray(logits, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    _require(logits_array.shape == (count, EXPECTED_CLASS_COUNT), "logits_shape_mismatch")
    _require(probability_array.shape == (count, EXPECTED_CLASS_COUNT), "probability_shape_mismatch")
    _require(bool(np.isfinite(logits_array).all()), "logits_nonfinite")
    _require(
        bool(np.isfinite(probability_array).all())
        and bool(((probability_array >= 0) & (probability_array <= 1)).all())
        and bool(np.allclose(probability_array.sum(axis=1), 1.0, rtol=0, atol=1e-12)),
        "probabilities_invalid",
    )
    frame = pd.DataFrame({"observation_uid": list(uids)})
    for index in range(EXPECTED_CLASS_COUNT):
        frame[f"logit_{index}"] = logits_array[:, index]
        frame[f"probability_{index}"] = probability_array[:, index]
    peak = int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0
    _require(0 <= peak <= smoke.DEFAULT_CUDA_BYTES, "prediction_cuda_exceeded")
    if torch_device.type == "cuda":
        _module("atlas_sers.evaluation.p05_pilot")._enforce_cuda_cap(torch, "cuda")
    audit = {
        "refit_id": canonical["refit_id"],
        "classes": list(canonical["classes"]),
        "model_state_sha256": model_sha,
        "calibration_state_sha256": calibration_sha,
        "test_uid_set_sha256": core._canon().sha256_value(sorted(uids)),
        "rows": count,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_cuda_bytes": peak,
        "optimizer_steps": 0,
    }
    _check_deadline(deadline)
    return frame, audit


__all__ = ["P05PredictionError", "predict_refit"]
