"""Compose reviewed U1 stages inside an admitted worker.

This dispatcher does not grant execution: the controller must record admission
before calling execute. It authenticates local dependency artifacts, delegates
to unchanged numerical kernels, retains fit/prediction affinity, and persists
each outcome once. The launcher owns runtime binding and resource enforcement.
"""

from __future__ import annotations

import dataclasses
import io
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from atlas_sers.evaluation import p03_runtime, p08_plan
from atlas_sers.evaluation import p08_u0_stage_backend as source_backend
from atlas_sers.evaluation import p08_u1_calibration as calibration_backend
from atlas_sers.evaluation import p08_u1_fit_backend as fit_backend
from atlas_sers.evaluation import p08_u1_prediction as prediction_backend
from atlas_sers.evaluation import p08_u1_selection as selection_backend
from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.governance.canonical import canonical_json_bytes
from atlas_sers.visualization.p08_live_monitor import EpochMonitor


class UniversalDispatcher:
    def __init__(
        self,
        *,
        jobs,
        source_factory,
        post_inputs,
        artifacts,
        candidate_registry_bytes,
        monitor_root,
        stream,
        device,
        global_deadline,
    ):
        self.jobs = {job["job_id"]: job for job in jobs}
        self.source_factory = source_factory
        self.inputs = post_inputs
        self.artifacts = artifacts
        self.candidates = candidate_registry_bytes
        self.monitor_root = Path(monitor_root)
        self.stream = stream
        self.device = device
        self.deadline = global_deadline
        self.retained = None
        self.outer_jobs = {}
        for job in jobs:
            if job["stage"] == "final_refit":
                self.outer_jobs.setdefault(self._cell(job), job)

    @staticmethod
    def _cell(job):
        return job["policy_id"], job["context_id"], job["model_id"]

    def telemetry(self):
        if torch.device(self.device).type == "cuda":
            return dict(
                allocated_gpu_bytes=int(torch.cuda.memory_allocated(self.device)),
                reserved_gpu_bytes=int(torch.cuda.memory_reserved(self.device)),
            )
        return dict(allocated_gpu_bytes=0, reserved_gpu_bytes=0)

    def _saved(self, job_id):
        receipt, payload = self.artifacts.verify(self.jobs[job_id])
        if receipt["status"] != "complete":
            raise ValueError("Dependency artifact is not complete.")
        return payload

    def _frame(self, job_id):
        raw = self._saved(job_id)["predictions.csv"]
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
        # CSV has no native null or integer-seed types; restore those recorded types.
        if "seed" in frame:
            expected = self.jobs[job_id]["seed"]
            if not frame.seed.eq(str(expected)).all():
                raise ValueError("Persisted prediction seed differs from graph.")
            frame["seed"] = expected
        if "probabilities" in frame:
            frame["probabilities"] = frame.probabilities.replace("", None)
        return frame

    def _source_metadata(self, job):
        return self.inputs.role_metadata(self.outer_jobs[self._cell(job)], "fit")

    def _classes(self, job):
        return tuple(sorted(set(self._source_metadata(job).target_analyte)))

    def _calibration(self, job_id):
        payload = json.loads(self._saved(job_id)["calibration.json"])
        payload["class_vocabulary"] = tuple(payload["class_vocabulary"])
        return TemperatureCalibration(**payload)

    def _selection(self, job):
        if len(job["dependencies"]) != 1:
            raise ValueError("Fitting must have exactly one source-selection dependency.")
        return json.loads(self._saved(job["dependencies"][0])["selection.json"])

    def _monitor(self, job, epoch_budget):
        return EpochMonitor(
            str(self.monitor_root / job["job_id"]),
            job_id=job["job_id"].removeprefix("P08JOB-"),
            model_id=job["model_id"],
            policy_id=job["policy_id"],
            seed=job["seed"],
            stage=job["stage"],
            validation_available=job["stage"] == "source_fit",
            epoch_budget=epoch_budget,
            stream=self.stream,
        )

    def _fit(self, job):
        if self.retained is not None:
            raise ValueError("An unconsumed fit must be verified before another fit.")
        stage = job["stage"]
        neural = job["model_id"] in p08_plan.NEURAL_RECIPES
        selection = self._selection(job) if stage != "source_fit" else None
        epochs = selection["epochs"] if neural and selection is not None else 200
        monitor = self._monitor(job, epochs) if neural else None
        result = None
        try:
            if stage == "source_fit":
                pair = self.source_factory.pair(job["job_id"])
                result = source_backend.invoke_source_fit(
                    pair, device=self.device, global_deadline=self.deadline, on_epoch=monitor
                )
                bundle = source_backend.prepare_fit_artifacts(pair, result)
                self.retained = (job["job_id"], pair, bundle)
                complete = bundle.status == "succeeded"
            else:
                bundle = fit_backend.invoke_fit(
                    job,
                    self.inputs,
                    selection=selection,
                    epochs=epochs if neural else None,
                    device=self.device,
                    global_deadline=self.deadline,
                    on_epoch=monitor,
                )
                result = bundle.result
                complete = bundle.status == "complete"
                if stage == "calibration_model_fit":
                    self.retained = (job["job_id"], selection, bundle)
            if monitor is not None:
                reason = "error"
                if complete:
                    reason = (
                        "fixed_duration"
                        if stage == "final_refit"
                        else ("epoch_limit" if result.epochs_completed == 200 else "patience")
                    )
                monitor.finish("complete" if complete else "failed", stop_reason=reason)
            return bundle.artifact_bytes(), "complete" if complete else "failed"
        except BaseException as error:
            if monitor is not None:
                monitor.close()
            # Keep returned source evidence even if structural acceptance failed.
            if result is not None and stage == "source_fit":
                try:
                    partial = (
                        source_backend._neural_artifacts(pair, result, job)
                        if neural
                        else (source_backend._classical_artifacts(result))
                    )
                    error.partial_artifacts = dict(partial)
                except Exception:
                    pass
            raise

    def _prediction_after_fit(self, job):
        if self.retained is None or self.retained[0] != job["dependencies"][0]:
            raise ValueError("Fit/prediction worker affinity was lost; no refit is allowed.")
        fit_id, context, bundle = self.retained
        saved = self._saved(fit_id)
        if job["stage"] == "source_validation_prediction":
            report = source_backend.verify_source_prediction(
                context, bundle, saved_artifact_bytes=saved, device=self.device
            )
            if job["model_id"] in p08_plan.CLASSICAL_MODELS:
                frame = pd.read_csv(
                    io.BytesIO(saved["predictions.csv"]), dtype=str, keep_default_na=False
                )
                frame["probabilities"] = None
            else:
                with np.load(io.BytesIO(saved["validation_logits.npz"]), allow_pickle=False) as z:
                    uids, classes, logits = (
                        z["uids"].tolist(),
                        tuple(z["classes"].tolist()),
                        z["logits"],
                    )
                metadata = self.inputs.role_metadata(job, "validation")
                metadata = (
                    metadata.set_index("observation_uid", drop=False)
                    .loc[uids]
                    .reset_index(drop=True)
                )
                frame = p03_runtime._prediction_frame(
                    metadata=metadata, scores=logits, class_vocabulary=classes, fit_id=fit_id
                )
            frame["seed"] = job["seed"]
            frame["selection_unit_id"] = job["unit_id"]
        else:
            kwargs = self.inputs.classical_fit_kwargs(self.jobs[fit_id], context)
            frame = prediction_backend.verify_calibration_prediction(
                job=job,
                fit_job=self.jobs[fit_id],
                fit_bundle=bundle,
                fit_kwargs=kwargs,
                saved_artifact_bytes=saved,
            )
            report = dict(status="verified", fit_job_id=fit_id, prediction_job_id=job["job_id"])
        self.retained = None
        return {
            "predictions.csv": frame.to_csv(index=False).encode(),
            "verification.json": canonical_json_bytes(report),
        }

    def _select(self, job):
        dependencies = {}
        for prediction_id in job["dependencies"]:
            prediction_job = self.jobs[prediction_id]
            self._saved(prediction_id)
            fit_job = self.jobs[prediction_job["dependencies"][0]]
            summary = json.loads(self._saved(fit_job["job_id"])["summary.json"])
            dependencies[prediction_id] = dict(
                fit_job=fit_job, prediction_job=prediction_job, summary=summary
            )
        if job["stage"] == "select_hyperparameters":
            selected = selection_backend.select_classical(
                selection_job=job,
                dependencies=dependencies,
                candidate_registry_bytes=self.candidates,
            )
        else:
            selected = selection_backend.select_epochs(selection_job=job, dependencies=dependencies)
        return {"selection.json": canonical_json_bytes(selected)}

    def _alias(self, job):
        selections = [
            self.jobs[j]
            for j in job["dependencies"]
            if self.jobs[j]["stage"] == "select_hyperparameters"
        ]
        if len(selections) != 1:
            raise ValueError("Alias must have one source-selection dependency.")
        selected = json.loads(self._saved(selections[0]["job_id"])["selection.json"])
        matches = [
            j
            for j in job["dependencies"]
            if self.jobs[j]["stage"] == "source_validation_prediction"
            and self.jobs[j]["candidate_id"] == selected["selected_candidate_id"]
        ]
        if len(matches) != 1:
            raise ValueError("Alias must resolve to exactly one selected source prediction.")
        reference = self.jobs[matches[0]]
        for key in (
            "context_id",
            "policy_id",
            "model_id",
            "seed",
            "unit_id",
            "fit_uid_sha256",
            "validation_uid_sha256",
            "test_uid_sha256",
        ):
            if reference[key] != job[key]:
                raise ValueError("Selected alias role differs from registered role.")
        frame = self._frame(matches[0])
        return {
            "predictions.csv": frame.to_csv(index=False).encode(),
            "resolution.json": canonical_json_bytes(
                dict(
                    source_prediction_job_id=matches[0],
                    selection_job_id=selections[0]["job_id"],
                    candidate_id=selected["selected_candidate_id"],
                )
            ),
        }

    def _scalar(self, job):
        dependencies = {
            j: dict(job=self.jobs[j], predictions=self._frame(j)) for j in job["dependencies"]
        }
        calibration, audit = calibration_backend.calibrate(
            job=job,
            dependencies=dependencies,
            classes=self._classes(job),
            source_metadata=self._source_metadata(job),
        )
        return {
            "calibration.json": canonical_json_bytes(dataclasses.asdict(calibration)),
            "audit.json": canonical_json_bytes(audit),
        }

    def _held(self, job):
        fits = [j for j in job["dependencies"] if self.jobs[j]["stage"] == "final_refit"]
        if len(fits) != 1:
            raise ValueError("Held prediction must have exactly one final refit.")
        fit_id = fits[0]
        saved = self._saved(fit_id)  # Authentication precedes any local model deserialization.
        kwargs = dict(
            job=job,
            fit_job=self.jobs[fit_id],
            inputs=self.inputs,
            fit_summary=json.loads(saved["summary.json"]),
            device=self.device,
        )
        if job["model_id"] in p08_plan.CLASSICAL_MODELS:
            kwargs["estimator"] = pickle.loads(saved["estimator.pkl"])
        else:
            scalar = [
                j for j in job["dependencies"] if self.jobs[j]["stage"] == "scalar_calibration"
            ]
            if len(scalar) != 1:
                raise ValueError("Neural held prediction must have one source calibration.")
            kwargs.update(
                terminal_state=torch.load(
                    io.BytesIO(saved["terminal.pt"]), weights_only=True, map_location="cpu"
                ),
                calibration_job=self.jobs[scalar[0]],
                calibration=self._calibration(scalar[0]),
            )
        frame = prediction_backend.predict_held(**kwargs)
        return {"predictions.csv": frame.to_csv(index=False).encode()}

    def _ensemble(self, job):
        dependencies = {}
        for dep in job["dependencies"]:
            dependency = self.jobs[dep]
            entry = dict(job=dependency)
            if dependency["stage"] == "scalar_calibration":
                entry["calibration"] = self._calibration(dep)
            else:
                entry["predictions"] = self._frame(dep)
            dependencies[dep] = entry
        frame = calibration_backend.ensemble(
            job=job, dependencies=dependencies, classes=self._classes(job)
        )
        return {"predictions.csv": frame.to_csv(index=False).encode()}

    def execute(self, job):
        if self.jobs.get(job.get("job_id")) != job:
            raise ValueError("Worker was sent an unregistered or changed job.")
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Cumulative execution deadline reached.")
        try:
            stage = job["stage"]
            status = "complete"
            if stage in ("source_fit", "calibration_model_fit", "final_refit"):
                artifacts, status = self._fit(job)
            elif stage in ("source_validation_prediction", "calibration_validation_prediction"):
                artifacts = self._prediction_after_fit(job)
            elif stage in ("select_hyperparameters", "select_refit_epochs"):
                artifacts = self._select(job)
            elif stage == "calibration_prediction_alias":
                artifacts = self._alias(job)
            elif stage == "scalar_calibration":
                artifacts = self._scalar(job)
            elif stage == "held_prediction":
                artifacts = self._held(job)
            elif stage == "seed_ensemble_prediction":
                artifacts = self._ensemble(job)
            else:
                raise ValueError("Unsupported universal stage.")
        except Exception as error:
            status = "failed"
            artifacts = dict(getattr(error, "partial_artifacts", {}))
            diagnostic = dict(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                job_id=job["job_id"],
            )
            artifacts["error.json"] = canonical_json_bytes(diagnostic)
        receipt = self.artifacts.write(
            job,
            artifacts,
            status=status,
            extra=dict(deadline_exceeded=time.perf_counter() >= self.deadline),
        )
        return dict(job_id=job["job_id"], status=status, receipt=receipt)

    def close(self):
        self.retained = None
        if self.stream is not None:
            self.stream.flush()


def receipt_result_verifier(artifact_store, job, result):
    """Verify saved bytes and the exact receipt returned by an admitted worker."""
    receipt, payload = artifact_store.verify(
        job, expected_receipt_sha256=result["receipt"]["sha256"]
    )
    if receipt != result["receipt"] or receipt["status"] != result["status"]:
        raise ValueError("Worker completion differs from durable receipt.")
    if receipt["status"] == "complete" and "error.json" in payload:
        raise ValueError("Failed scientific operation cannot be marked complete.")
    return receipt
