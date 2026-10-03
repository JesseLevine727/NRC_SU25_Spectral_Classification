"""P08-T117 serial U0 session controller: orchestration half (internal).

This leaf drives an ordered sequence of U0 pair attempts over one accepted
journal and one caller-supplied runtime input set.  It composes the internal
session IO leaf with the inherited serial resource guard and the inherited
stage backend.  It never fits outside the injected backend, never retries,
never repairs, never resets counters and grants no execution authority.

Classical native kernels cannot be hard preempted: observation happens before
and after the call and cooperatively from neural epoch callbacks only.  No peak
RAM during an uninterruptible native call is guaranteed to be observed.
"""

from __future__ import annotations

import json
import time

from . import p08_resources as resource_guard
from . import p08_serial_resources as serial
from . import p08_u0_session_io as session_io
from . import p08_u0_stage_backend as stage_backend
from . import p08_u0_store as store
from .p08_qc_blocks import canonical_sha256
from .p08_u0_runtime_inputs import RuntimeInputs

__all__ = ["SCHEMA_VERSION", "SessionError", "require_scientific_execution"]

SCHEMA_VERSION = "nato-sers-p08-u0-session-report-v1"

_NANOS_PER_SECOND = 10**9
_EXPECTED_PAIRS = 78
_EXPECTED_JOBS = 156

_CPU_MODELS = frozenset({"C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES"})
_GPU_MODELS = frozenset({"D0-M", "D1", "D2", "D3"})

_REASON_CODES = frozenset(
    {
        "invalid_session_input",
        "previous_session_requires_review",
        "invalid_pair_set",
        "invalid_pair_binding",
        "unregistered_pair",
        "pair_already_attempted",
        "session_busy",
        "session_stopped",
        "session_closed",
        "candidate_blocked",
        "stale_measurement",
        "invalid_measurement_window",
        "invalid_resource_input",
        "resource_limit_exceeded",
        "fit_failed",
        "prediction_failed",
        "prediction_parity_failed",
        "artifact_readback_mismatch",
        "pair_binding_mismatch",
        "control_mismatch",
        "invalid_fit_artifacts",
        "session_io_error",
        "storage_io_error",
        "close_refused",
        "internal_error",
        "interrupted",
        "scientific_execution_not_authorized",
    }
)


class SessionError(ValueError):
    """Static, data-free session failure exposing ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_session_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise SessionError(reason_code) from None


def _as_count(value):
    if type(value) is not int or value < 0:
        return 0
    return value


def _categorize_exception(exc):
    code = getattr(exc, "reason_code", None)
    if type(code) is str and code in _REASON_CODES:
        return code
    module = getattr(type(exc), "__module__", "") or ""
    if module.endswith("p08_u0_session_io"):
        return "session_io_error"
    if module.endswith("p08_u0_store"):
        return "storage_io_error"
    if module.endswith("p08_serial_resources") or module.endswith("p08_resources"):
        return "invalid_resource_input"
    if module.endswith("p08_u0_stage_backend"):
        return "invalid_fit_artifacts"
    if isinstance(exc, OSError):
        return "storage_io_error"
    return "internal_error"


def _canonical_bytes(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("internal_error")


def _decode_job(text):
    if type(text) is not str or text == "":
        _fail("invalid_pair_binding")
    try:
        value = json.loads(text)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_pair_binding")
    if type(value) is not dict:
        _fail("invalid_pair_binding")
    return value


def _job_identity(job):
    job_id = job.get("job_id")
    stage = job.get("stage")
    dependencies = job.get("dependencies")
    if type(job_id) is not str or job_id == "":
        _fail("invalid_pair_binding")
    if type(stage) is not str or stage == "":
        _fail("invalid_pair_binding")
    if type(dependencies) not in (list, tuple):
        _fail("invalid_pair_binding")
    for dependency in dependencies:
        if type(dependency) is not str:
            _fail("invalid_pair_binding")
    return job_id, stage, tuple(dependencies)


def _pair_model_id(pair):
    configuration = getattr(pair, "configuration", None)
    if not callable(configuration):
        _fail("invalid_pair_binding")
    try:
        value = configuration()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_pair_binding")
    if type(value) is not dict:
        _fail("invalid_pair_binding")
    model_id = value.get("model_id")
    if type(model_id) is not str or model_id == "":
        _fail("invalid_pair_binding")
    return model_id


def _derive_worker(model_id):
    if model_id in _CPU_MODELS:
        return "cpu"
    if model_id in _GPU_MODELS:
        return "gpu"
    _fail("invalid_pair_binding")


def _artifact_members(fit_artifacts):
    getter = getattr(fit_artifacts, "artifact_bytes", None)
    if not callable(getter):
        _fail("invalid_fit_artifacts")
    try:
        value = getter()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_fit_artifacts")
    if type(value) is not dict:
        _fail("invalid_fit_artifacts")
    members = {}
    for name, payload in value.items():
        if type(name) is not str or name == "" or "/" in name or "\\" in name:
            _fail("invalid_fit_artifacts")
        if type(payload) is not bytes:
            _fail("invalid_fit_artifacts")
        members[name] = payload
    return members


class _SourceSession:
    """Ordered serial U0 pair driver over one accepted journal.

    ``started_monotonic_ns`` predating this session only absorbs outer setup
    time; it grants no permission and is not a second allowance.  Observed
    bytes cover the three roots only while ``launch.json`` is the sole fixed
    control file, and a terminal record written after close is charged by the
    caller.
    """

    def __init__(
        self,
        owner,
        runtime_inputs,
        *,
        artifact_root,
        torch_module,
        started_monotonic_ns=None,
        launch_control_root=None,
        launch_record_sha256=None,
    ):
        if started_monotonic_ns is None:
            self._started_ns = time.monotonic_ns()
        else:
            if type(started_monotonic_ns) is not int or started_monotonic_ns < 0:
                _fail("invalid_session_input")
            if started_monotonic_ns > time.monotonic_ns():
                _fail("invalid_session_input")
            self._started_ns = started_monotonic_ns
        if type(artifact_root) is not str or artifact_root == "":
            _fail("invalid_session_input")
        try:
            is_owner = type(owner) is store.Store
        except Exception:
            is_owner = False
        if not is_owner:
            _fail("invalid_session_input")
        try:
            is_inputs = type(runtime_inputs) is RuntimeInputs
        except Exception:
            is_inputs = False
        if not is_inputs:
            _fail("invalid_session_input")
        try:
            state = owner.snapshot()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_session_input")
        if type(state) is not dict:
            _fail("invalid_session_input")
        manifest = state.get("manifest")
        summary = state.get("summary")
        if type(manifest) is not dict or type(summary) is not dict:
            _fail("invalid_session_input")
        if summary.get("journal_state") != "not_started":
            _fail("previous_session_requires_review")

        self._owner = owner
        self._torch_module = torch_module
        self._pair_infos = {}
        self._fit_ids = set()
        self._pred_ids = set()
        self._validate_pairs(runtime_inputs, manifest)

        self._stopped = False
        self._closed = False
        self._busy = False
        self._active_job_id = None
        self._active_artifacts = {}
        self._attempted_fit_ids = set()
        self._succeeded_fit_ids = set()
        self._completed_pred_ids = set()
        self._failed_fit_ids = set()
        self._failed_pred_ids = set()
        self._last_result = None
        self._last_summary = summary
        self._observation_current = True
        self._prior_closed_wall_ns = _as_count(summary.get("closed_session_wall_ns"))
        self._recorded_wall_ns = self._prior_closed_wall_ns
        self._recorded_artifact_bytes = 0
        self._measured_wall_ns = self._prior_closed_wall_ns
        self._measured_artifact_bytes = 0
        self._finalization_tail_ns = 0
        self._last_durable_ns = self._started_ns

        session = self._call(
            session_io._SessionIO,
            owner,
            artifact_root,
            started_monotonic_ns=self._started_ns,
            launch_control_root=launch_control_root,
            launch_record_sha256=launch_record_sha256,
        )
        self._io = session
        try:
            self._call(session.open_session)
        except BaseException:
            try:
                session.close()
            except BaseException:
                pass
            raise

    # -- input validation -------------------------------------------------

    def _validate_pairs(self, runtime_inputs, manifest):
        pairs = getattr(runtime_inputs, "pairs", None)
        if type(pairs) is not tuple or len(pairs) != _EXPECTED_PAIRS:
            _fail("invalid_pair_set")
        jobs = manifest.get("jobs")
        if type(jobs) not in (list, tuple) or len(jobs) != _EXPECTED_JOBS:
            _fail("invalid_pair_set")
        manifest_by_id = {}
        for job in jobs:
            if type(job) is not dict:
                _fail("invalid_pair_set")
            job_id = job.get("job_id")
            stage = job.get("stage")
            worker = job.get("worker")
            dependencies = job.get("dependencies")
            if type(job_id) is not str or job_id == "":
                _fail("invalid_pair_set")
            if type(stage) is not str or stage == "":
                _fail("invalid_pair_set")
            if worker not in ("cpu", "gpu"):
                _fail("invalid_pair_set")
            if type(dependencies) not in (list, tuple):
                _fail("invalid_pair_set")
            for dependency in dependencies:
                if type(dependency) is not str:
                    _fail("invalid_pair_set")
            if job_id in manifest_by_id:
                _fail("invalid_pair_set")
            manifest_by_id[job_id] = (
                stage,
                worker,
                tuple(sorted(dependencies)),
            )

        decoded = {}
        fit_stages = set()
        pred_stages = set()
        for pair in pairs:
            prepared = getattr(pair, "prepared_pair", None)
            if prepared is None:
                _fail("invalid_pair_binding")
            fit_job = _decode_job(getattr(prepared, "fit_job_json", None))
            pred_job = _decode_job(getattr(prepared, "prediction_job_json", None))
            model_id = _pair_model_id(pair)
            worker = _derive_worker(model_id)
            device = "cuda" if worker == "gpu" else "cpu"
            if fit_job.get("model_id") != model_id:
                _fail("invalid_pair_binding")
            if pred_job.get("model_id") != model_id:
                _fail("invalid_pair_binding")
            fit_id, fit_stage, fit_deps = _job_identity(fit_job)
            pred_id, pred_stage, pred_deps = _job_identity(pred_job)
            if fit_id == pred_id or fit_id in decoded or pred_id in decoded:
                _fail("invalid_pair_binding")
            if fit_deps:
                _fail("invalid_pair_binding")
            if len(pred_deps) != 1 or pred_deps[0] != fit_id:
                _fail("invalid_pair_binding")
            fit_stages.add(fit_stage)
            pred_stages.add(pred_stage)
            decoded[fit_id] = (fit_stage, worker, tuple(sorted(fit_deps)))
            decoded[pred_id] = (pred_stage, worker, tuple(sorted(pred_deps)))
            self._fit_ids.add(fit_id)
            self._pred_ids.add(pred_id)
            self._pair_infos[fit_id] = {
                "pair": pair,
                "fit_job_id": fit_id,
                "prediction_job_id": pred_id,
                "device": device,
            }
        if fit_stages != {"source_fit"}:
            _fail("invalid_pair_binding")
        if pred_stages != {"source_validation_prediction"}:
            _fail("invalid_pair_binding")
        if set(decoded) != set(manifest_by_id):
            _fail("invalid_pair_set")
        for job_id, projection in decoded.items():
            if manifest_by_id[job_id] != projection:
                _fail("invalid_pair_binding")

    # -- guarded calls / observation --------------------------------------

    def _call(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except SessionError:
            raise
        except Exception as exc:
            _fail(_categorize_exception(exc))

    def _state(self):
        state = self._call(self._io.snapshot)
        if type(state) is not dict:
            _fail("invalid_session_input")
        if type(state.get("summary")) is not dict:
            _fail("invalid_session_input")
        if type(state.get("manifest")) is not dict:
            _fail("invalid_session_input")
        return state

    def _elapsed(self):
        value = self._call(self._io.elapsed_ns)
        if type(value) is not int or value < 0:
            _fail("invalid_measurement_window")
        return value

    def _observed_bytes(self):
        value = self._call(self._io.observed_bytes)
        if type(value) is not int or value < 0:
            _fail("invalid_measurement_window")
        return value

    def _measure_now(self):
        wall = self._prior_closed_wall_ns + self._elapsed()
        artifacts = self._observed_bytes()
        if wall > self._measured_wall_ns:
            self._measured_wall_ns = wall
        if artifacts > self._measured_artifact_bytes:
            self._measured_artifact_bytes = artifacts
        return wall, artifacts

    def _usage(self, summary):
        return {
            "model_fit_attempts": _as_count(summary.get("model_fit_attempts")),
            "scalar_calibration_attempts": 0,
            "active_wall_ns": max(
                self._recorded_wall_ns, self._measured_wall_ns
            ),
            "new_artifact_bytes": max(
                _as_count(summary.get("new_artifact_bytes")),
                self._measured_artifact_bytes,
                self._recorded_artifact_bytes,
            ),
        }

    def _absorb_summary(self, summary):
        active_wall = _as_count(summary.get("active_wall_ns"))
        if active_wall > self._recorded_wall_ns:
            self._recorded_wall_ns = active_wall
        summary_bytes = _as_count(summary.get("new_artifact_bytes"))
        if summary_bytes > self._recorded_artifact_bytes:
            self._recorded_artifact_bytes = summary_bytes

    def _update_highwater(self, summary):
        self._absorb_summary(summary)

    def _refresh_summary(self):
        try:
            state = self._io.snapshot()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._observation_current = False
            return
        if type(state) is dict and type(state.get("summary")) is dict:
            self._last_summary = state["summary"]
            self._observation_current = True
            self._update_highwater(state["summary"])
        else:
            self._observation_current = False

    def _observe(self):
        self._call(self._io.progress)
        started, finished, resources, cuda = self._call(
            serial._sample, self._io.artifact_root, self._torch_module, 1
        )
        state = self._state()
        summary = state["summary"]
        self._last_summary = summary
        self._observation_current = True
        resources = dict(resources)
        resources["active_cpu_workers"] = _as_count(summary.get("active_cpu_workers"))
        resources["active_gpu_workers"] = _as_count(summary.get("active_gpu_workers"))
        self._measure_now()
        record = self._call(
            resource_guard.evaluate_resource_snapshot,
            "U0",
            self._usage(summary),
            resources,
        )
        if type(record) is not dict or record.get("within_proposed_limits") is not True:
            _fail("resource_limit_exceeded")
        now = time.monotonic_ns()
        if type(started) is not int or type(finished) is not int:
            _fail("invalid_measurement_window")
        if started < 0 or finished < 0 or not (started <= finished <= now):
            _fail("invalid_measurement_window")
        if now - started > serial.MAXIMUM_AGE_NS:
            _fail("stale_measurement")
        limits = self._call(resource_guard.proposed_limits, "U0")
        ceiling = limits.get("cuda_allocated_bytes")
        peak = cuda.get("peak") if type(cuda) is dict else None
        if type(peak) is not int or type(ceiling) is not int or peak > ceiling:
            _fail("resource_limit_exceeded")
        self._update_highwater(summary)
        return record

    # -- admission --------------------------------------------------------

    def _admit(self, job_id):
        self._active_artifacts = {}
        self._call(self._io.progress, next_job_id=job_id)
        state = self._state()
        summary = state["summary"]
        self._last_summary = summary
        self._update_highwater(summary)
        report = self._call(
            serial.check_serial_u0_candidate,
            state["manifest"],
            state["events"],
            job_id=job_id,
            expected_head_sha256=summary.get("head_sha256"),
            output_directory=self._io.artifact_root,
            torch_module=self._torch_module,
            model_threads=1,
        )
        if type(report) is not dict:
            _fail("candidate_blocked")
        if report.get("proposed_serial_candidate_admissible") is not True:
            _fail("candidate_blocked")
        measurement_started = report.get("measurement_started_monotonic_ns")
        now = time.monotonic_ns()
        if type(measurement_started) is not int:
            _fail("invalid_measurement_window")
        if measurement_started < 0 or now < measurement_started:
            _fail("invalid_measurement_window")
        if now - measurement_started > serial.MAXIMUM_AGE_NS:
            _fail("stale_measurement")
        resources = report.get("resources")
        if type(resources) is not dict:
            _fail("invalid_resource_input")
        self._call(
            self._io.start,
            job_id,
            resources,
            measurement_started_ns=measurement_started,
        )
        self._active_job_id = job_id

    # -- pair execution ---------------------------------------------------

    def run_pair(self, fit_job_id):
        if self._closed:
            _fail("session_closed")
        if self._stopped:
            _fail("session_stopped")
        if self._busy:
            _fail("session_busy")
        if fit_job_id not in self._pair_infos:
            _fail("unregistered_pair")
        if fit_job_id in self._attempted_fit_ids:
            _fail("pair_already_attempted")
        self._attempted_fit_ids.add(fit_job_id)
        self._busy = True
        try:
            self._run_pair_inner(fit_job_id)
        except BaseException as exc:
            self._stopped = True
            try:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    self._best_effort_failure("interrupted", "interrupted")
                elif isinstance(exc, SessionError):
                    self._best_effort_failure(exc.reason_code, "failed")
                else:
                    self._best_effort_failure("internal_error", "failed")
            except BaseException:
                pass
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SessionError):
                raise
            raise SessionError("session_stopped") from None
        finally:
            self._busy = False
            self._active_job_id = None
        return self.report()

    def _run_pair_inner(self, fit_job_id):
        info = self._pair_infos[fit_job_id]
        pair = info["pair"]
        prediction_job_id = info["prediction_job_id"]
        device = info["device"]

        self._admit(fit_job_id)
        record = self._observe()
        deadline = time.perf_counter() + (
            record.get("remaining_active_wall_ns", 0) / _NANOS_PER_SECOND
        )

        def on_epoch(*args, **kwargs):
            self._observe()

        result = self._call(
            stage_backend.invoke_source_fit,
            pair,
            device=device,
            global_deadline=deadline,
            on_epoch=on_epoch,
        )
        self._last_result = result
        self._observe()
        fit_artifacts = self._call(stage_backend.prepare_fit_artifacts, pair, result)
        status = getattr(fit_artifacts, "status", None)
        if getattr(fit_artifacts, "fit_job_id", None) != fit_job_id:
            _fail("pair_binding_mismatch")
        if getattr(fit_artifacts, "prediction_job_id", None) != prediction_job_id:
            _fail("pair_binding_mismatch")
        if status not in ("succeeded", "failed"):
            _fail("invalid_fit_artifacts")
        self._observe()
        members = self._call(_artifact_members, fit_artifacts)

        fit_prefixed = {}
        for name in sorted(members):
            self._observe()
            prefixed_name = fit_job_id + "-" + name
            payload = members[name]
            self._call(self._io.write, prefixed_name, payload)
            readback = self._call(self._io.read, prefixed_name)
            if readback != payload:
                _fail("artifact_readback_mismatch")
            self._observe()
            self._active_artifacts[prefixed_name] = payload
            fit_prefixed[prefixed_name] = payload
        self._call(self._io.finish, fit_job_id, status, dict(self._active_artifacts))
        self._active_job_id = None
        self._refresh_summary()
        if status != "succeeded":
            self._failed_fit_ids.add(fit_job_id)
            _fail("fit_failed")
        self._succeeded_fit_ids.add(fit_job_id)

        self._admit(prediction_job_id)
        self._active_artifacts.update(fit_prefixed)
        saved = {}
        for name in sorted(members):
            self._observe()
            saved[name] = self._call(self._io.read, fit_job_id + "-" + name)
            self._observe()
        verification = self._call(
            stage_backend.verify_source_prediction,
            pair,
            fit_artifacts,
            saved_artifact_bytes=saved,
            device=device,
        )
        if type(verification) is not dict:
            _fail("prediction_parity_failed")
        if verification.get("prediction_parity_verified") is not True:
            _fail("prediction_parity_failed")
        self._observe()
        report_name = prediction_job_id + "-verification.json"
        report_value = dict(verification)
        try:
            report_value["report_sha256"] = canonical_sha256(report_value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("internal_error")
        report_bytes = _canonical_bytes(report_value)
        self._call(self._io.write, report_name, report_bytes)
        readback = self._call(self._io.read, report_name)
        if readback != report_bytes:
            _fail("artifact_readback_mismatch")
        self._active_artifacts[report_name] = report_bytes
        combined = dict(fit_prefixed)
        combined[report_name] = report_bytes
        self._call(self._io.finish, prediction_job_id, "succeeded", combined)
        self._active_job_id = None
        self._completed_pred_ids.add(prediction_job_id)
        self._refresh_summary()
        self._observe()
        fit_artifacts = None

    # -- best-effort failure logging --------------------------------------

    def _best_effort_failure(self, reason_code, status):
        job_id = self._active_job_id
        if job_id is None:
            return
        self._active_job_id = None
        if reason_code not in _REASON_CODES:
            reason_code = "internal_error"
        diagnostics_name = job_id + "-failure.json"
        diagnostics = _canonical_bytes(
            {"reason_code": reason_code, "status": status}
        )
        artifacts = dict(self._active_artifacts)
        try:
            self._io.write(diagnostics_name, diagnostics)
            artifacts[diagnostics_name] = diagnostics
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        persisted = False
        try:
            self._io.finish(job_id, status, artifacts)
            persisted = True
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            persisted = False
        if persisted:
            if job_id in self._fit_ids:
                self._failed_fit_ids.add(job_id)
            elif job_id in self._pred_ids:
                self._failed_pred_ids.add(job_id)

    # -- close / report ---------------------------------------------------

    def _summary_id_set(self, summary, *keys):
        result = set()
        if type(summary) is not dict:
            return result
        for key in keys:
            values = summary.get(key)
            if type(values) in (list, tuple, set, frozenset):
                for value in values:
                    if type(value) is str:
                        result.add(value)
        return result

    def _is_clean_to_close(self):
        if self._active_job_id is not None or self._busy:
            return False
        try:
            state = self._io.snapshot()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return False
        if type(state) is not dict:
            return False
        summary = state.get("summary")
        if type(summary) is not dict:
            return False
        self._last_summary = summary
        if summary.get("in_flight_job_ids"):
            return False
        succeeded = self._summary_id_set(summary, "succeeded_job_ids")
        terminal = self._summary_id_set(
            summary,
            "succeeded_job_ids",
            "failed_job_ids",
            "interrupted_job_ids",
            "terminal_job_ids",
        )
        for fit_job_id in succeeded:
            info = self._pair_infos.get(fit_job_id)
            if info is None:
                continue
            if info["prediction_job_id"] not in terminal:
                return False
        return True

    def _release_io(self):
        try:
            self._io.close()
        except BaseException:
            pass

    def close(self):
        if self._closed:
            return self.report()
        try:
            clean = self._is_clean_to_close()
        except BaseException:
            self._stopped = True
            self._observation_current = False
            self._release_io()
            raise
        if not clean:
            self._stopped = True
            self._observation_current = False
            self._release_io()
            _fail("close_refused")
        try:
            new_summary = self._call(self._io.close_session)
        except BaseException:
            self._stopped = True
            self._observation_current = False
            self._release_io()
            raise
        self._closed = True
        if type(new_summary) is dict:
            self._last_summary = new_summary
            self._absorb_summary(new_summary)
        self._observation_current = True
        observed = None
        pending_exc = None
        try:
            observed = self._observed_bytes()
        except (KeyboardInterrupt, SystemExit) as exc:
            self._observation_current = False
            pending_exc = exc
        except Exception:
            self._observation_current = False
        if observed is not None and observed > self._measured_artifact_bytes:
            self._measured_artifact_bytes = observed
        try:
            self._release_io()
        finally:
            finish = time.monotonic_ns()
        measured_wall = (finish - self._started_ns) + self._prior_closed_wall_ns
        self._measured_wall_ns = measured_wall
        tail = measured_wall - self._recorded_wall_ns
        self._finalization_tail_ns = tail
        if tail < 0:
            self._observation_current = False
            if pending_exc is not None:
                raise pending_exc
            raise SessionError("invalid_measurement_window")
        if pending_exc is not None:
            raise pending_exc
        return self.report()

    def _completed_pair_count(self):
        succeeded = self._summary_id_set(
            self._last_summary, "succeeded_job_ids"
        )
        completed = 0
        for fit_job_id, info in self._pair_infos.items():
            if fit_job_id not in succeeded:
                continue
            if info["prediction_job_id"] in succeeded:
                completed += 1
        return completed

    def _incomplete(self):
        return self._completed_pair_count() != len(self._pair_infos)

    def report(self):
        summary = self._last_summary
        current = self._observation_current
        if not self._closed:
            try:
                state = self._call(self._io.snapshot)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                current = False
                state = None
            if type(state) is dict and type(state.get("summary")) is dict:
                summary = state["summary"]
                self._last_summary = summary
                self._absorb_summary(summary)
                try:
                    self._measure_now()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    current = False
                else:
                    current = True
            else:
                current = False
            self._observation_current = current
        if type(summary) is not dict:
            summary = {}
        report = {
            "schema_version": SCHEMA_VERSION,
            "execution_authorized": False,
            "live_runtime_accepted": False,
            "loaded_runtime_code_verified": False,
            "observation_current": bool(current),
            "stopped": bool(self._stopped),
            "closed": bool(self._closed),
            "incomplete": self._incomplete(),
            "completed_pair_count": self._completed_pair_count(),
            "fit_attempt_count": _as_count(summary.get("model_fit_attempts")),
            "prediction_attempt_count": _as_count(
                summary.get("source_prediction_attempts")
            ),
            "recorded_active_wall_ns": self._recorded_wall_ns,
            "recorded_artifact_bytes": self._recorded_artifact_bytes,
            "observed_logical_bytes": self._measured_artifact_bytes,
            "measured_elapsed_ns": self._measured_wall_ns,
            "finalization_tail_ns": self._finalization_tail_ns,
            "head_sha256": summary.get("head_sha256"),
            "manifest_sha256": summary.get("manifest_sha256"),
            "journal_state": summary.get("journal_state"),
        }
        try:
            report["report_sha256"] = canonical_sha256(report)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            report["report_sha256"] = None
        return report

    def __repr__(self):
        return (
            f"<_SourceSession stopped={self._stopped} closed={self._closed} "
            f"busy={self._busy}>"
        )


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise SessionError("scientific_execution_not_authorized") from None
