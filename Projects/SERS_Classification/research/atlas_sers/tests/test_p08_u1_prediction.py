"""Prediction composition against actual inherited interfaces; no model fitting."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from atlas_sers.evaluation import p03_runtime, p04_runtime, p08_plan
from atlas_sers.evaluation import p08_u1_prediction as pred
from atlas_sers.evaluation.classical import TemperatureCalibration, apply_temperature
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value
from atlas_sers.models.acquisition import AcquisitionClassifier

CLASSES = ("A", "B", "C")
SCORES = np.asarray([[2, 0, -1], [-1, 2, 0], [0, -1, 2]], dtype=np.float64)


def metadata(prefix="v"):
    return pd.DataFrame(
        dict(
            observation_uid=[f"{prefix}{i}" for i in range(3)],
            master_sample_id=[f"00{prefix}{i}" for i in range(3)],
            instrument=["H"] * 3,
            station=["S"] * 3,
            target_analyte=CLASSES,
        )
    )


def job(stage, model="C-RBF-SVM", dependencies=(), **changes):
    fields = dict(
        policy_id="PP-U-SG",
        representation_id="R_SG_400_1800",
        array_sha256="a" * 64,
        context_id="ctx",
        model_id=model,
        model_spec_sha256="b" * 64,
        stage=stage,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed="deterministic" if model == "C-RBF-SVM" else 20260805,
        candidate_id=p08_plan.SOURCE_SELECTION_DEPENDENT,
        hyperparameter_sha256=p08_plan.NOT_APPLICABLE,
        fit_uid_sha256=sha256_value(["f0", "f1", "f2"]),
        validation_uid_sha256=sha256_value(["v0", "v1", "v2"]),
        test_uid_sha256=sha256_value(["h0", "h1", "h2"]),
        resolution=p08_plan.SOURCE_SELECTION_DEPENDENT,
        evidence_status=p08_plan.EVIDENCE_FUTURE,
    )
    fields.update(changes)
    return p08_plan._new_job(fields, list(dependencies))


class Estimator:
    classes_ = np.asarray(CLASSES)

    def scores(self, values):
        assert values.shape == (3, 1401)
        return SCORES.copy()


class HeldInputs:
    def __init__(self):
        self.calls = 0

    def held_inputs(self, record):
        self.calls += 1
        return np.zeros((3, 1401), dtype=np.float32), metadata("h"), CLASSES, ("f0", "f1", "f2")


def calibration_case():
    fit = job("calibration_model_fit")
    requested = job("calibration_validation_prediction", dependencies=[fit["job_id"]])
    frame = p03_runtime._prediction_frame(
        metadata=metadata(), scores=SCORES, class_vocabulary=CLASSES, fit_id=fit["job_id"]
    )
    dataset = p03_runtime.P03Dataset.from_frozen_representation(
        intensity=np.zeros((3, 1401)),
        representation_uids=np.asarray(["v0", "v1", "v2"]),
        metadata=metadata(),
    )
    result = p03_runtime.CandidateFitOutcome(
        fit_id=fit["job_id"],
        status="complete",
        reason_code=None,
        model_id="C-RBF-SVM",
        candidate_id="candidate",
        seed="deterministic",
        fit_uid_sha256=fit["fit_uid_sha256"],
        validation_uid_sha256=fit["validation_uid_sha256"],
        fit_master_sha256="c" * 64,
        elapsed_seconds=0.1,
        inference_seconds=0.01,
        serialized_model_bytes=1,
        warnings=[],
        traceback_digest=None,
        validation_predictions=frame,
        validation_metrics=p03_runtime._master_metrics(frame, SCORES, CLASSES),
        estimator=Estimator(),
    )
    artifacts = dict()
    bundle = SimpleNamespace(
        job=fit, result=result, status="complete", artifact_bytes=lambda: dict(artifacts)
    )

    def refresh():
        artifacts.update(
            {
                "summary.json": canonical_json_bytes(result.status_record()),
                "predictions.csv": result.validation_predictions.to_csv(index=False).encode(),
            }
        )

    refresh()
    kwargs = dict(
        candidate_id="candidate",
        class_vocabulary=CLASSES,
        dataset=dataset,
        validation_uids=("v0", "v1", "v2"),
    )
    return dict(
        job=requested,
        fit_job=fit,
        fit_bundle=bundle,
        fit_kwargs=kwargs,
        saved_artifact_bytes=artifacts,
    ), refresh


def test_calibration_prediction_exact_roundtrip():
    args, _ = calibration_case()
    frame = pred.verify_calibration_prediction(**args)
    assert frame.probabilities.isna().all()
    assert frame.seed.eq("deterministic").all()
    assert frame.master_sample_id.iloc[0] == "00v0"
    assert len(frame) == 3


@pytest.mark.parametrize(
    "column,value",
    [
        ("master_sample_id", "wrong"),
        ("true_label", "B"),
        ("scores", "[0,0,0]"),
        ("observation_uid", "wrong"),
        ("probabilities", "[0.3,0.3,0.4]"),
        ("class_vocabulary", '["B","A","C"]'),
    ],
)
def test_source_prediction_tamper_rejected(column, value):
    args, refresh = calibration_case()
    args["fit_bundle"].result.validation_predictions.loc[0, column] = value
    refresh()
    with pytest.raises((pred.PredictionError, ValueError)):
        pred.verify_calibration_prediction(**args)


def test_persisted_byte_tamper_rejected():
    args, _ = calibration_case()
    args["saved_artifact_bytes"] = args["saved_artifact_bytes"] | {"predictions.csv": b"wrong"}
    with pytest.raises(pred.PredictionError, match="artifact_bytes_mismatch"):
        pred.verify_calibration_prediction(**args)


def test_classical_held_is_uncalibrated():
    fit = job("final_refit")
    requested = job("held_prediction", dependencies=[fit["job_id"]])
    inputs = HeldInputs()
    summary = dict(
        status="complete",
        fit_id=fit["job_id"],
        model_id=fit["model_id"],
        seed=fit["seed"],
        fit_uid_sha256=fit["fit_uid_sha256"],
    )
    frame = pred.predict_held(
        job=requested, fit_job=fit, inputs=inputs, estimator=Estimator(), fit_summary=summary
    )
    assert inputs.calls == 1 and frame.probabilities.isna().all()
    np.testing.assert_array_equal(np.asarray([json.loads(x) for x in frame.scores]), SCORES)


def neural_case():
    fit = job("final_refit", "D1")
    scalar = job(
        "scalar_calibration",
        "D1",
        fit_uid_sha256=p08_plan.NOT_APPLICABLE,
        validation_uid_sha256=p08_plan.NOT_APPLICABLE,
    )
    requested = job("held_prediction", "D1", dependencies=[fit["job_id"], scalar["job_id"]])
    with torch.random.fork_rng():
        state = AcquisitionClassifier(use_projection=True).state_dict()
    summary = dict(
        status="complete",
        fit_job_id=fit["job_id"],
        recipe="D1",
        seed=20260805,
        classes=list(CLASSES),
        terminal_state_digest=p04_runtime._state_hash(state),
    )
    calibration = TemperatureCalibration(
        temperature=2.0,
        class_vocabulary=CLASSES,
        observations=3,
        masters=3,
        fit_observation_uid_sha256="a" * 64,
        fit_master_uid_sha256="b" * 64,
        optimizer_success=True,
        optimizer_objective=0.5,
    )
    return dict(
        job=requested,
        fit_job=fit,
        inputs=HeldInputs(),
        fit_summary=summary,
        terminal_state=state,
        calibration_job=scalar,
        calibration=calibration,
    )


def test_neural_held_uses_inherited_logits_and_per_seed_temperature(monkeypatch):
    args = neural_case()
    calls = []

    def logits(model, tensor, device):
        calls.append((model, tensor.shape, device))
        return SCORES.copy()

    monkeypatch.setattr(pred.p05_development, "_predict_logits", logits)
    state = torch.random.get_rng_state().clone()
    frame = pred.predict_held(**args)
    assert len(calls) == 1 and calls[0][1] == (3, 1, 1401)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert frame.probability_status.eq("cross_fitted_temperature").all()
    actual = np.asarray([json.loads(p) for p in frame.probabilities])
    np.testing.assert_array_equal(actual, apply_temperature(SCORES, args["calibration"]))


@pytest.mark.parametrize(
    "problem", ["temperature", "optimizer", "class_order", "state", "recipe", "seed"]
)
def test_neural_bad_source_evidence_precedes_held_access(problem):
    args = neural_case()
    if problem in ("temperature", "optimizer", "class_order"):
        changes = {
            "temperature": {"temperature": -1},
            "optimizer": {"optimizer_success": False},
            "class_order": {"class_vocabulary": ("B", "A", "C")},
        }[problem]
        args["calibration"] = dataclasses.replace(args["calibration"], **changes)
    else:
        field, value = {
            "state": ("terminal_state_digest", "f" * 64),
            "recipe": ("recipe", "D2"),
            "seed": ("seed", 20260817),
        }[problem]
        args["fit_summary"][field] = value
    with pytest.raises(pred.PredictionError):
        pred.predict_held(**args)
    assert args["inputs"].calls == 0
