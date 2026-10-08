"""P08-U1 governed scheduler over the accepted durable ledger (T294).

This module is a single-process scheduler.  It owns no scientific kernel,
spawns no process and touches no artifact bytes.  It sequences jobs that were
already authenticated by the caller across workers the caller already spawned,
while recording lifecycle state in an accepted
:class:`~atlas_sers.evaluation.p08_u1_store.P08U1Store`.

Integration boundary (deliberate and honest):

* The caller creates/reopens the store, imports and seals reuse evidence and
  authenticates graph membership *before* calling :func:`run_controller`.
  This module never creates, reopens, resets or retries the store.
* The caller spawns and owns the worker objects.  A worker only has to expose
  ``worker_id``/``kind``/``submit``/``poll``/``alive``/``terminate``.
* ``verify_result`` is the caller's durable-artifact and scientific-status
  verifier.  No worker receipt is written to the ledger before it returns.
* ``resource_snapshot`` is the caller's live measurement of the process tree.
  This module never fabricates zero/unavailable values and never resets a
  counter; it only forwards the exact mapping to the ledger.
* No model kernel, tensor library, array type or private path is imported here.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time

from .p08_u1_store import (
    ALLOWED_POLICIES,
    KNOWN_MODELS,
    KNOWN_STAGES,
    MAX_CPU_WORKERS,
    MAX_GPU_WORKERS,
    P08U1Error,
    _expected_worker_kind,
)

__all__ = ["ControllerError", "run_controller", "sha256_value"]

MAX_TOTAL_WORKERS = 5
PROGRESS_INTERVAL_SECONDS = 5.0
IDLE_RESOURCE_INTERVAL_SECONDS = 2.0
MAX_POLL_SECONDS = 1.0

# The two estimator-affinity pairs: the fit artifact lives in the worker until
# its single validation prediction has consumed it.
AFFINITY_NEXT = {
    "source_fit": "source_validation_prediction",
    "calibration_model_fit": "calibration_validation_prediction",
}
AFFINITY_PREDICTION_STAGES = (
    "source_validation_prediction",
    "calibration_validation_prediction",
)

TERMINAL_STATUSES = ("complete", "failed", "interrupted")
_WORKER_KINDS = ("CPU", "GPU")
_HEX = frozenset("0123456789abcdef")


class ControllerError(P08U1Error):
    """Structured scheduler refusal carrying a reason and progress summary."""

    def __init__(self, reason_code, *, summary=None, job_id=None, cause=None):
        self.reason_code = reason_code
        self.reason = reason_code
        self.summary = {} if summary is None else summary
        self.job_id = job_id
        self.cause = cause
        super().__init__("controller_failure:" + str(reason_code))

    def as_dict(self):
        return {
            "reason_code": self.reason_code,
            "job_id": self.job_id,
            "summary": dict(self.summary),
        }


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ControllerError("value_not_canonical_json", cause=exc) from exc


def sha256_value(value):
    """Public helper: canonical-JSON sha256 of a payload (before any hash field)."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _is_hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _failure_receipt(job_id, reason_code):
    payload = {"job_id": job_id, "reason_code": reason_code}
    return {"job_id": job_id, "reason_code": reason_code, "sha256": sha256_value(payload)}


def _expected_kind(index, job_id):
    record = index[job_id]
    return _expected_worker_kind(record["stage"], record["model_id"])


def _index_jobs(jobs):
    if not isinstance(jobs, (list, tuple)) or isinstance(jobs, (str, bytes)):
        raise ControllerError("jobs_must_be_sequence")
    index = {}
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ControllerError("job_must_be_mapping")
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id.startswith("P08JOB-"):
            raise ControllerError("job_id_invalid")
        if raw.get("stage") not in KNOWN_STAGES:
            raise ControllerError("job_stage_invalid", job_id=job_id)
        if raw.get("model_id") not in KNOWN_MODELS:
            raise ControllerError("job_model_invalid", job_id=job_id)
        if raw.get("policy_id") not in ALLOWED_POLICIES:
            raise ControllerError("job_policy_invalid", job_id=job_id)
        dependencies = raw.get("dependencies")
        if not isinstance(dependencies, (list, tuple)) or isinstance(dependencies, (str, bytes)):
            raise ControllerError("job_dependencies_invalid", job_id=job_id)
        if any(not isinstance(dep, str) for dep in dependencies):
            raise ControllerError("job_dependency_invalid", job_id=job_id)
        if len(set(dependencies)) != len(dependencies):
            raise ControllerError("job_dependency_duplicate", job_id=job_id)
        if job_id in index:
            raise ControllerError("duplicate_job_id", job_id=job_id)
        index[job_id] = raw
    for job_id, raw in index.items():
        for dep in raw["dependencies"]:
            if dep == job_id:
                raise ControllerError("self_dependency", job_id=job_id)
            if dep not in index:
                raise ControllerError("unknown_dependency", job_id=job_id)
    return index


def _build_affinity(index):
    dependents = {job_id: [] for job_id in index}
    for job_id, raw in index.items():
        for dep in raw["dependencies"]:
            dependents[dep].append(job_id)
    for job_id in dependents:
        dependents[job_id].sort()
    pred_to_fit = {}
    fit_to_pred = {}
    for job_id, raw in index.items():
        expected = AFFINITY_NEXT.get(raw["stage"])
        if expected is None:
            continue
        matches = [dep for dep in dependents[job_id] if index[dep]["stage"] == expected]
        if len(matches) != 1:
            raise ControllerError("affinity_pair_missing_or_ambiguous", job_id=job_id)
        prediction = matches[0]
        if list(index[prediction]["dependencies"]) != [job_id]:
            raise ControllerError(
                "affinity_prediction_dependency_invalid", job_id=prediction
            )
        if prediction in pred_to_fit:
            raise ControllerError("affinity_prediction_ambiguous", job_id=prediction)
        if _expected_kind(index, job_id) != _expected_kind(index, prediction):
            raise ControllerError("affinity_kind_mismatch", job_id=prediction)
        pred_to_fit[prediction] = job_id
        fit_to_pred[job_id] = prediction
    for job_id, raw in index.items():
        if raw["stage"] in AFFINITY_PREDICTION_STAGES and job_id not in pred_to_fit:
            raise ControllerError("affinity_prediction_without_fit", job_id=job_id)
    return dependents, pred_to_fit, fit_to_pred


def _validate_workers(workers, needed_kinds):
    if not isinstance(workers, (list, tuple)) or isinstance(workers, (str, bytes)):
        raise ControllerError("workers_must_be_sequence")
    seen = set()
    cpu = 0
    gpu = 0
    validated = []
    for worker in workers:
        worker_id = getattr(worker, "worker_id", None)
        kind = getattr(worker, "kind", None)
        if not isinstance(worker_id, str) or not worker_id:
            raise ControllerError("worker_id_invalid")
        if worker_id in seen:
            raise ControllerError("worker_id_duplicate")
        seen.add(worker_id)
        if kind not in _WORKER_KINDS:
            raise ControllerError("worker_kind_invalid")
        for method in ("submit", "poll", "alive", "terminate"):
            if not callable(getattr(worker, method, None)):
                raise ControllerError("worker_protocol_invalid")
        if kind == "CPU":
            cpu += 1
        else:
            gpu += 1
        validated.append(worker)
    if cpu > MAX_CPU_WORKERS:
        raise ControllerError("cpu_worker_limit")
    if gpu > MAX_GPU_WORKERS:
        raise ControllerError("gpu_worker_limit")
    if cpu + gpu > MAX_TOTAL_WORKERS:
        raise ControllerError("worker_total_limit")
    if "CPU" in needed_kinds and cpu == 0:
        raise ControllerError("cpu_worker_required")
    if "GPU" in needed_kinds and gpu == 0:
        raise ControllerError("gpu_worker_required")
    return validated


def _clamp_poll(poll_seconds):
    if isinstance(poll_seconds, bool):
        raise ControllerError("poll_seconds_invalid")
    try:
        value = float(poll_seconds)
    except (TypeError, ValueError) as exc:
        raise ControllerError("poll_seconds_invalid", cause=exc) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ControllerError("poll_seconds_invalid")
    return min(value, MAX_POLL_SECONDS)


class _Controller:
    def __init__(
        self,
        *,
        store,
        jobs,
        workers,
        completed,
        resource_snapshot,
        verify_result,
        on_progress,
        poll_seconds,
        clock,
        sleep,
    ):
        if store is None:
            raise ControllerError("store_required")
        if not callable(resource_snapshot):
            raise ControllerError("resource_snapshot_not_callable")
        if not callable(verify_result):
            raise ControllerError("verify_result_not_callable")
        self._store = store
        self._jobs = _index_jobs(jobs)
        self._completed = set(completed)
        if not self._completed <= set(self._jobs):
            raise ControllerError("completed_not_subset_of_jobs")
        (
            self._dependents,
            self._pred_to_fit,
            self._fit_to_pred,
        ) = _build_affinity(self._jobs)
        for fit_id, pred_id in self._fit_to_pred.items():
            if fit_id in self._completed and pred_id not in self._completed:
                raise ControllerError("resumed_fit_without_prediction", job_id=fit_id)
        self._indegree = {}
        for job_id, raw in self._jobs.items():
            if job_id in self._completed:
                self._indegree[job_id] = 0
            else:
                self._indegree[job_id] = sum(
                    1 for dep in raw["dependencies"] if dep not in self._completed
                )
        self._ready = {"CPU": [], "GPU": []}
        for job_id, degree in self._indegree.items():
            if job_id in self._completed or degree != 0:
                continue
            heapq.heappush(self._ready[_expected_kind(self._jobs, job_id)], job_id)
        self._workers = _validate_workers(workers, self._needed_kinds())
        self._resource_snapshot = resource_snapshot
        self._verify_result = verify_result
        self._on_progress = on_progress
        self._clock = clock
        self._sleep = sleep
        self._poll_seconds = _clamp_poll(poll_seconds)
        self._assignment = {}
        self._reserved = {}
        self._started = set()
        self._state = "running"
        self._start_time = clock()
        self._last_progress = self._start_time
        self._last_resource = self._start_time - IDLE_RESOURCE_INTERVAL_SECONDS

    def _kind_of(self, job_id):
        return _expected_kind(self._jobs, job_id)

    def _needed_kinds(self):
        kinds = set()
        for job_id in self._jobs:
            if job_id in self._completed:
                continue
            kinds.add(self._kind_of(job_id))
        return kinds

    def _is_done(self):
        return len(self._completed) >= len(self._jobs)

    def _active_count(self):
        return len(self._assignment)

    def _summary(self, state):
        total = len(self._jobs)
        done = len(self._completed)
        running_stages = set()
        for worker_id in self._assignment:
            running_stages.add(self._jobs[self._assignment[worker_id]]["stage"])
        if len(running_stages) == 1:
            stage_label = next(iter(running_stages))
        elif len(running_stages) > 1:
            stage_label = "mixed"
        else:
            stage_label = "complete" if done >= total else "idle"
        stages = {}
        for job_id, raw in self._jobs.items():
            if job_id in self._completed:
                continue
            stages[raw["stage"]] = stages.get(raw["stage"], 0) + 1
        if stage_label == "idle" and len(stages) == 1:
            stage_label = next(iter(stages))
        return {
            "state": state,
            "complete": done,
            "completed": done,
            "remaining": total - done,
            "running": self._active_count(),
            "stage": stage_label,
            "stages": dict(sorted(stages.items())),
            "elapsed_seconds": self._clock() - self._start_time,
        }

    def _emit_progress(self, state=None):
        if self._on_progress is None:
            return
        self._on_progress(self._summary(state or self._state))

    def _safe_progress_emit(self, state=None):
        if self._on_progress is None:
            return
        try:
            self._on_progress(self._summary(state or self._state))
        except BaseException:
            pass

    def _terminate_all(self):
        for worker in self._workers:
            try:
                worker.terminate()
            except BaseException:
                pass

    def _halt(self, reason_code, cause, job_id=None):
        self._terminate_all()
        for started_id in list(self._started):
            receipt = _failure_receipt(started_id, reason_code)
            try:
                self._store.finish(started_id, "interrupted", receipt)
            except BaseException:
                pass
        self._started.clear()
        self._state = "failed"
        self._safe_progress_emit()
        if isinstance(cause, KeyboardInterrupt):
            raise cause
        raise ControllerError(
            reason_code,
            summary=self._summary("failed"),
            job_id=job_id,
            cause=cause,
        )

    def _emergency_shutdown(self, reason_code):
        self._terminate_all()
        for started_id in list(self._started):
            receipt = _failure_receipt(started_id, reason_code)
            try:
                self._store.finish(started_id, "interrupted", receipt)
            except BaseException:
                pass
        self._started.clear()

    def _capture_resources(self):
        try:
            resources = self._resource_snapshot(self._active_count())
        except BaseException as exc:
            self._halt("resource_snapshot_failed", exc)
        if not isinstance(resources, dict):
            self._halt("resource_snapshot_invalid", None)
        return resources

    def _record_resources(self):
        resources = self._capture_resources()
        try:
            self._store.record_resources(resources)
        except BaseException as exc:
            self._halt("resource_record_failed", exc)
        self._last_resource = self._clock()

    def _launch(self, worker, job_id, now):
        kind = self._kind_of(job_id)
        if worker.kind != kind:
            self._halt("worker_kind_mismatch", None, job_id=job_id)
        resources = self._capture_resources()
        try:
            self._store.record_resources(resources)
        except BaseException as exc:
            self._halt("resource_record_failed", exc, job_id=job_id)
        try:
            self._store.start(job_id, kind, resources)
        except BaseException as exc:
            self._halt("start_failed", exc, job_id=job_id)
        self._started.add(job_id)
        self._assignment[worker.worker_id] = job_id
        try:
            worker.submit(self._jobs[job_id])
        except BaseException as exc:
            self._halt("submit_failed", exc, job_id=job_id)
        self._last_resource = self._clock()

    def _dispatch(self, now):
        dispatched = False
        for worker in self._workers:
            worker_id = worker.worker_id
            if worker_id in self._assignment:
                continue
            try:
                alive = worker.alive()
            except BaseException as exc:
                self._halt("liveness_check_failed", exc)
            if not alive:
                self._halt(
                    "worker_dead_before_assign",
                    None,
                    job_id=self._reserved.get(worker_id),
                )
            reserved = self._reserved.pop(worker_id, None)
            if reserved is not None:
                if reserved in self._completed:
                    continue
                if self._indegree.get(reserved, 0) != 0:
                    self._halt("reserved_job_not_ready", None, job_id=reserved)
                self._launch(worker, reserved, now)
                dispatched = True
                continue
            heap = self._ready[worker.kind]
            while heap:
                candidate = heapq.heappop(heap)
                if candidate in self._completed:
                    continue
                if candidate in self._pred_to_fit:
                    self._halt("pinned_job_in_ready", None, job_id=candidate)
                if self._kind_of(candidate) != worker.kind:
                    self._halt("ready_kind_mismatch", None, job_id=candidate)
                self._launch(worker, candidate, now)
                dispatched = True
                break
        return dispatched

    def _mark_complete(self, job_id, worker_id):
        self._completed.add(job_id)
        for dependent in self._dependents.get(job_id, ()):
            if dependent in self._completed:
                continue
            self._indegree[dependent] -= 1
            if self._indegree[dependent] > 0:
                continue
            if dependent in self._pred_to_fit:
                # The estimator must stay in the worker that produced the fit.
                self._reserved[worker_id] = dependent
            else:
                heapq.heappush(self._ready[self._kind_of(dependent)], dependent)

    def _handle_result(self, worker, job_id, result):
        if not isinstance(result, dict):
            self._halt("result_invalid", None, job_id=job_id)
        if result.get("job_id") != job_id:
            self._halt("result_job_mismatch", None, job_id=job_id)
        status = result.get("status")
        if status not in TERMINAL_STATUSES:
            self._halt("result_status_invalid", None, job_id=job_id)
        if not isinstance(result.get("receipt"), dict):
            self._halt("result_receipt_missing", None, job_id=job_id)
        try:
            validated = self._verify_result(self._jobs[job_id], result)
        except BaseException as exc:
            self._halt("verification_failed", exc, job_id=job_id)
        if not isinstance(validated, dict) or not _is_hex_digest(validated.get("sha256")):
            self._halt("verified_receipt_invalid", None, job_id=job_id)
        try:
            self._store.finish(job_id, status, validated)
        except BaseException as exc:
            self._halt("finish_failed", exc, job_id=job_id)
        self._started.discard(job_id)
        self._assignment.pop(worker.worker_id, None)
        if status == "complete":
            self._mark_complete(job_id, worker.worker_id)
            if validated.get("extra", {}).get("deadline_exceeded"):
                self._halt("operation_crossed_deadline", None, job_id=job_id)
            self._record_resources()
        else:
            self._halt("operation_" + status, None, job_id=job_id)

    def _poll(self, now):
        polled = False
        for worker in self._workers:
            job_id = self._assignment.get(worker.worker_id)
            if job_id is None:
                continue
            try:
                result = worker.poll()
            except BaseException as exc:
                self._halt("poll_failed", exc, job_id=job_id)
            if result is None:
                try:
                    alive = worker.alive()
                except BaseException as exc:
                    self._halt("liveness_check_failed", exc, job_id=job_id)
                if not alive:
                    self._halt("worker_died_while_busy", None, job_id=job_id)
                continue
            polled = True
            self._handle_result(worker, job_id, result)
        return polled

    def run(self):
        try:
            try:
                self._emit_progress()
            except BaseException as exc:
                self._halt("progress_failed", exc)
            while not self._is_done():
                now = self._clock()
                dispatched = self._dispatch(now)
                polled = self._poll(now)
                if self._is_done():
                    break
                if not self._assignment and not self._reserved and not any(self._ready.values()):
                    self._halt("graph_stalled", None)
                now = self._clock()
                if now - self._last_progress >= PROGRESS_INTERVAL_SECONDS:
                    self._last_progress = now
                    try:
                        self._emit_progress()
                    except BaseException as exc:
                        self._halt("progress_failed", exc)
                if not dispatched and not polled:
                    now = self._clock()
                    if now - self._last_resource >= IDLE_RESOURCE_INTERVAL_SECONDS:
                        self._record_resources()
                    self._sleep(self._poll_seconds)
            self._state = "complete"
            try:
                self._emit_progress("complete")
            except BaseException as exc:
                self._halt("progress_failed", exc)
        except KeyboardInterrupt:
            self._emergency_shutdown("keyboard_interrupt")
            raise
        except BaseException:
            self._emergency_shutdown("controller_exception")
            raise
        self._terminate_all()
        return self._summary("complete")


def run_controller(
    *,
    store,
    jobs,
    workers,
    completed,
    resource_snapshot,
    verify_result,
    on_progress=None,
    poll_seconds=0.25,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Sequence an already-authenticated U1 graph over caller-owned workers.

    Returns a completion summary on success and raises :class:`ControllerError`
    on any failure.  The caller remains the owner of the store and closes it.
    """
    try:
        controller = _Controller(
            store=store,
            jobs=jobs,
            workers=workers,
            completed=completed,
            resource_snapshot=resource_snapshot,
            verify_result=verify_result,
            on_progress=on_progress,
            poll_seconds=poll_seconds,
            clock=clock,
            sleep=sleep,
        )
    except BaseException:
        for worker in workers:
            try:
                worker.terminate()
            except BaseException:
                pass
        raise
    return controller.run()
