"""P08 authenticated training-history binding for approved reporting.

This private analysis-draft module binds the authenticated training-fit
summaries produced by :mod:`p08_universal_evidence` (the authoritative source
of each neural fit's numerical history) to the per-fit epoch monitors that
observed those fits, then hands the accepted pair to the downstream
diagnostics aggregator.

It is read-only reporting IO.  It never trains, evaluates, selects, launches a
subprocess, touches the network, loads a checkpoint or unpickles data.  The
public prepared manifest returned by the accepted aggregator is unchanged and
carries no private filesystem path; every path, file digest and selection
decision is confined to the private receipt.  Runtime and resource
reverification remain the caller's responsibility and are explicitly *not*
claimed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat

from atlas_sers.evaluation import p08_plan, p08_universal_evidence
from atlas_sers.visualization import p08_live_monitor
from atlas_sers.visualization.p08_training_figure_data import (
    _validate_monitor_record,
    prepare_training_diagnostics,
)

__all__ = [
    "MAXIMUM_HISTORY",
    "MINIMUM_HISTORY",
    "SOURCE_EPOCH_BUDGET",
    "TRAINING_EVIDENCE_SCHEMA",
    "TrainingEvidenceError",
    "bind_training_histories",
]


TRAINING_EVIDENCE_SCHEMA = "p08_training_evidence_private_v1"

P08_JOB_PREFIX = "P08JOB-"

NEURAL_RECIPES = frozenset(("D0-M", "D1", "D2", "D3"))
NEURAL_POLICIES = frozenset(("PP-U-SG", "PP-U-ARPLS"))
NEURAL_STAGES = frozenset(("source_fit", "final_refit"))

SOURCE_EPOCH_BUDGET = 200
MINIMUM_HISTORY = 30
MAXIMUM_HISTORY = 200

MAXIMUM_MONITOR_BYTES = 2 * 1024 * 1024
MAXIMUM_MONITOR_ROOTS = 16

_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_EXPECTED_ENTRY_KEYS = frozenset(
    ("job", "summary", "summary_sha256", "receipt_sha256")
)

_SOURCE_SUMMARY_FIELDS = frozenset(
    (
        "status",
        "slot_id",
        "recipe_id",
        "recipe",
        "seed",
        "epochs_completed",
        "history",
    )
)
_FINAL_SUMMARY_FIELDS = frozenset(
    (
        "status",
        "fit_job_id",
        "recipe",
        "seed",
        "epochs",
        "epochs_completed",
        "history",
    )
)

_KNOWN_STATUSES = frozenset(
    (
        p08_live_monitor.STATUS_RUNNING,
        p08_live_monitor.STATUS_COMPLETE,
        p08_live_monitor.STATUS_FAILED,
        p08_live_monitor.STATUS_INTERRUPTED,
        p08_live_monitor.STATUS_CLOSED,
    )
)


class TrainingEvidenceError(ValueError):
    """Raised for malformed evidence, monitor control or refused checks."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _typed_equal(left, right):
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _require_equal(value, expected, code):
    if not _typed_equal(value, expected):
        raise TrainingEvidenceError(code)


def _is_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_refusal(check):
    outcome = check()
    if outcome is False:
        raise TrainingEvidenceError("check_refused")


def _stripped_job_id(job_id):
    if job_id.startswith(P08_JOB_PREFIX):
        return job_id[len(P08_JOB_PREFIX):]
    return job_id


def _matches_neural_filter(job):
    return (
        job.get("model_id") in NEURAL_RECIPES
        and job.get("policy_id") in NEURAL_POLICIES
        and job.get("stage") in NEURAL_STAGES
    )


def _validated_expected_jobs(jobs, check):
    expected = {}
    expected_keys = set(p08_plan.JOB_FIELDS) | {"job_id"}
    for job in jobs:
        _check_refusal(check)
        if not isinstance(job, dict):
            raise TrainingEvidenceError("job_must_be_mapping")
        if not _matches_neural_filter(job):
            continue
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id.startswith(P08_JOB_PREFIX):
            raise TrainingEvidenceError("job_id_invalid")
        if set(job.keys()) != expected_keys:
            raise TrainingEvidenceError("job_keys_invalid")
        fields = {name: job[name] for name in p08_plan.JOB_FIELDS}
        if job_id[len(P08_JOB_PREFIX):] != p08_plan._hash(fields):
            raise TrainingEvidenceError("job_id_hash_mismatch")
        if not _is_int(job["seed"]) or job["seed"] not in p08_plan.SEEDS:
            raise TrainingEvidenceError("job_seed_invalid")
        if job_id in expected:
            raise TrainingEvidenceError("duplicate_job_id")
        expected[job_id] = job
    return expected


def _validate_summary_entry(entry, job):
    if not isinstance(entry, dict):
        raise TrainingEvidenceError("training_summary_entry_invalid")
    if set(entry.keys()) != _EXPECTED_ENTRY_KEYS:
        raise TrainingEvidenceError("training_summary_entry_invalid")
    if not isinstance(entry["job"], dict):
        raise TrainingEvidenceError("training_summary_entry_invalid")
    if not _typed_equal(entry["job"], job):
        raise TrainingEvidenceError("training_summary_job_mismatch")
    if not isinstance(entry["summary"], dict):
        raise TrainingEvidenceError("training_summary_entry_invalid")
    if not _is_hex64(entry["summary_sha256"]):
        raise TrainingEvidenceError("summary_sha256_invalid")
    if not _is_hex64(entry["receipt_sha256"]):
        raise TrainingEvidenceError("receipt_sha256_invalid")


def _parse_monitor_roots(monitor_roots, allowed_roots):
    if not isinstance(monitor_roots, (list, tuple)) or isinstance(
        monitor_roots, (str, bytes)
    ):
        raise TrainingEvidenceError("monitor_roots_must_be_sequence")
    if not 1 <= len(monitor_roots) <= MAXIMUM_MONITOR_ROOTS:
        raise TrainingEvidenceError("monitor_roots_count_invalid")
    labels = set()
    reals = set()
    parsed = []
    for raw in monitor_roots:
        if not isinstance(raw, dict) or set(raw.keys()) != {"label", "root"}:
            raise TrainingEvidenceError("monitor_root_entry_invalid")
        label = raw["label"]
        if (
            not isinstance(label, str)
            or not _LABEL_RE.fullmatch(label)
            or ".." in label
        ):
            raise TrainingEvidenceError("monitor_label_invalid")
        if label in labels:
            raise TrainingEvidenceError("monitor_label_duplicate")
        labels.add(label)
        try:
            root_path = os.fspath(raw["root"])
        except TypeError:
            raise TrainingEvidenceError("monitor_root_invalid") from None
        if isinstance(root_path, bytes) or not root_path:
            raise TrainingEvidenceError("monitor_root_invalid")
        try:
            secure_root = os.fspath(
                p08_universal_evidence._secure_dir(root_path, allowed_roots)
            )
        except TrainingEvidenceError:
            raise
        except Exception as exc:
            raise TrainingEvidenceError("monitor_root_insecure") from exc
        real = os.path.realpath(secure_root)
        if real in reals:
            raise TrainingEvidenceError("monitor_root_duplicate")
        reals.add(real)
        parsed.append({"label": label, "root": secure_root})
    return parsed


def _read_bounded_file(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise TrainingEvidenceError("monitor_file_unreadable") from None
    try:
        try:
            before = os.fstat(descriptor)
        except OSError:
            raise TrainingEvidenceError("monitor_file_unreadable") from None
        if not stat.S_ISREG(before.st_mode):
            raise TrainingEvidenceError("monitor_file_not_regular")
        if before.st_nlink != 1:
            raise TrainingEvidenceError("monitor_file_not_single_link")
        if before.st_size > MAXIMUM_MONITOR_BYTES:
            raise TrainingEvidenceError("monitor_file_too_large")
        chunks = []
        remaining = MAXIMUM_MONITOR_BYTES + 1
        while remaining > 0:
            try:
                chunk = os.read(descriptor, min(remaining, 65536))
            except OSError:
                raise TrainingEvidenceError("monitor_file_unreadable") from None
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAXIMUM_MONITOR_BYTES:
            raise TrainingEvidenceError("monitor_file_too_large")
        try:
            after = os.fstat(descriptor)
            named = os.lstat(path)
        except OSError:
            raise TrainingEvidenceError("monitor_file_changed") from None
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ):
            raise TrainingEvidenceError("monitor_file_changed")
        if (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ):
            raise TrainingEvidenceError("monitor_file_changed")
        if len(raw) != after.st_size:
            raise TrainingEvidenceError("monitor_file_changed")
    finally:
        os.close(descriptor)
    return raw


def _read_monitor_document(root, job_id, allowed_roots, check):
    _check_refusal(check)
    try:
        secure_root = os.fspath(
            p08_universal_evidence._secure_dir(root, allowed_roots)
        )
    except TrainingEvidenceError:
        raise
    except Exception as exc:
        raise TrainingEvidenceError("monitor_root_insecure") from exc
    _check_refusal(check)
    job_dir = os.path.join(secure_root, job_id)
    if not os.path.lexists(job_dir):
        _check_refusal(check)
        return None
    _check_refusal(check)
    if os.path.islink(job_dir):
        raise TrainingEvidenceError("monitor_ancestor_insecure")
    try:
        os.fspath(p08_universal_evidence._secure_dir(job_dir, allowed_roots))
    except TrainingEvidenceError:
        raise
    except Exception as exc:
        raise TrainingEvidenceError("monitor_ancestor_insecure") from exc
    _check_refusal(check)
    path = os.path.join(job_dir, "semantic.json")
    if not os.path.lexists(path):
        _check_refusal(check)
        return None
    _check_refusal(check)
    try:
        secure = os.fspath(
            p08_universal_evidence._secure_file(path, allowed_roots)
        )
    except TrainingEvidenceError:
        raise
    except Exception as exc:
        raise TrainingEvidenceError("monitor_file_insecure") from exc
    _check_refusal(check)
    raw = _read_bounded_file(secure)
    _check_refusal(check)
    try:
        document = p08_universal_evidence._strict_loads(raw)
    except Exception as exc:
        raise TrainingEvidenceError("monitor_document_invalid") from exc
    if not isinstance(document, dict):
        raise TrainingEvidenceError("monitor_document_invalid")
    _check_refusal(check)
    return {
        "path": path,
        "real": secure,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document": document,
    }


def _verify_source_summary(summary, job):
    if not isinstance(summary, dict):
        raise TrainingEvidenceError("summary_invalid")
    if not _SOURCE_SUMMARY_FIELDS <= set(summary):
        raise TrainingEvidenceError("summary_fields_invalid")
    _require_equal(summary["status"], "complete", "summary_status_invalid")
    _require_equal(summary["slot_id"], job["job_id"], "summary_slot_id_mismatch")
    _require_equal(
        summary["recipe_id"], job["model_id"], "summary_recipe_id_mismatch"
    )
    _require_equal(
        summary["recipe"], job["model_id"], "summary_recipe_mismatch"
    )
    _require_equal(summary["seed"], job["seed"], "summary_seed_mismatch")
    history = summary["history"]
    if not isinstance(history, list):
        raise TrainingEvidenceError("summary_history_invalid")
    epochs_completed = summary["epochs_completed"]
    if not _is_int(epochs_completed) or epochs_completed != len(history):
        raise TrainingEvidenceError("summary_epochs_completed_invalid")
    if not MINIMUM_HISTORY <= len(history) <= MAXIMUM_HISTORY:
        raise TrainingEvidenceError("summary_history_length_invalid")
    return history


def _verify_final_summary(summary, job):
    if not isinstance(summary, dict):
        raise TrainingEvidenceError("summary_invalid")
    if not _FINAL_SUMMARY_FIELDS <= set(summary):
        raise TrainingEvidenceError("summary_fields_invalid")
    _require_equal(summary["status"], "complete", "summary_status_invalid")
    _require_equal(
        summary["fit_job_id"], job["job_id"], "summary_fit_job_id_mismatch"
    )
    _require_equal(
        summary["recipe"], job["model_id"], "summary_recipe_mismatch"
    )
    _require_equal(summary["seed"], job["seed"], "summary_seed_mismatch")
    history = summary["history"]
    if not isinstance(history, list):
        raise TrainingEvidenceError("summary_history_invalid")
    epochs = summary["epochs"]
    epochs_completed = summary["epochs_completed"]
    if not _is_int(epochs) or not _is_int(epochs_completed):
        raise TrainingEvidenceError("summary_epochs_invalid")
    if epochs != epochs_completed or epochs != len(history):
        raise TrainingEvidenceError("summary_epochs_invalid")
    if not MINIMUM_HISTORY <= epochs <= MAXIMUM_HISTORY:
        raise TrainingEvidenceError("summary_epochs_invalid")
    return history


def _verify_summary(job, summary):
    if job["stage"] == "source_fit":
        return _verify_source_summary(summary, job)
    return _verify_final_summary(summary, job)


def _verify_monitor_identity(document, job):
    if not isinstance(document, dict):
        raise TrainingEvidenceError("monitor_document_invalid")
    _require_equal(
        document.get("schema"),
        p08_live_monitor.SEMANTIC_SCHEMA,
        "monitor_schema_invalid",
    )
    _require_equal(
        document.get("job_id"),
        _stripped_job_id(job["job_id"]),
        "monitor_job_mismatch",
    )
    _require_equal(
        document.get("model_id"), job["model_id"], "monitor_model_mismatch"
    )
    _require_equal(
        document.get("policy_id"), job["policy_id"], "monitor_policy_mismatch"
    )
    _require_equal(
        document.get("seed"), job["seed"], "monitor_seed_mismatch"
    )
    _require_equal(
        document.get("stage"), job["stage"], "monitor_stage_mismatch"
    )
    _require_equal(
        document.get("validation_available"),
        job["stage"] == "source_fit",
        "monitor_validation_mismatch",
    )


def _verify_complete_history(job, document, history):
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise TrainingEvidenceError("monitor_rows_invalid")
    row_count = len(rows)
    if not MINIMUM_HISTORY <= row_count <= MAXIMUM_HISTORY:
        raise TrainingEvidenceError("monitor_history_length_invalid")
    epochs_completed = document.get("epochs_completed")
    if not _is_int(epochs_completed) or epochs_completed != row_count:
        raise TrainingEvidenceError("monitor_epochs_completed_invalid")
    budget = document.get("epoch_budget")
    if not _is_int(budget):
        raise TrainingEvidenceError("monitor_epoch_budget_invalid")
    if len(history) != row_count:
        raise TrainingEvidenceError("summary_history_length_invalid")

    stage = job["stage"]
    if stage == "source_fit":
        if budget != SOURCE_EPOCH_BUDGET:
            raise TrainingEvidenceError("monitor_budget_mismatch")
        expected_reason = (
            "epoch_limit" if row_count == SOURCE_EPOCH_BUDGET else "patience"
        )
        _require_equal(
            document.get("stop_reason"),
            expected_reason,
            "monitor_stop_reason_invalid",
        )
    else:
        if budget != len(history):
            raise TrainingEvidenceError("monitor_budget_mismatch")
        _require_equal(
            document.get("stop_reason"),
            "fixed_duration",
            "monitor_stop_reason_invalid",
        )

    validation_available = stage == "source_fit"
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TrainingEvidenceError("monitor_row_invalid")
        record = history[index]
        if not isinstance(record, dict):
            raise TrainingEvidenceError("summary_history_row_invalid")
        try:
            projected = p08_live_monitor._project_record(
                record,
                validation_available=validation_available,
                expected_epoch=index + 1,
                epoch_budget=budget,
                elapsed_seconds=row.get("elapsed_seconds"),
            )
        except (TypeError, ValueError) as exc:
            raise TrainingEvidenceError("summary_history_row_invalid") from exc
        if not _typed_equal(projected, row):
            raise TrainingEvidenceError("monitor_history_mismatch")


def bind_training_histories(
    evidence,
    *,
    monitor_roots,
    allowed_evidence_roots,
    check,
):
    """Bind authenticated fit summaries to observing epoch monitors.

    The caller has already completed an authenticated ``load_evidence`` and
    its own outer runtime/resource recheck.  This function does not claim that
    external verification.  It reads only the declared monitor roots, requires
    the expected neural fit set to match the authenticated summary set exactly,
    verifies every complete monitor history against the authoritative summary
    through the public projection, and then calls the accepted downstream
    diagnostics aggregator.  Nothing is written.  ``check`` is a mandatory
    callable refusal hook: a ``False`` return aborts before any further work.
    """
    if not isinstance(evidence, dict):
        raise TrainingEvidenceError("evidence_must_be_mapping")
    if not {"jobs", "training_fit_summaries", "diagnostics"} <= set(evidence):
        raise TrainingEvidenceError("evidence_keys_invalid")
    diagnostics = evidence["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise TrainingEvidenceError("diagnostics_must_be_mapping")
    if diagnostics.get("graph_sha256") != p08_universal_evidence.GRAPH_SHA256:
        raise TrainingEvidenceError("graph_sha256_mismatch")
    if diagnostics.get("outer_reverification_required") is not True:
        raise TrainingEvidenceError("outer_reverification_required_missing")
    if not callable(check):
        raise TypeError("check must be callable")

    _check_refusal(check)
    allowed_roots = p08_universal_evidence._prepare_roots(allowed_evidence_roots)
    _check_refusal(check)

    jobs = evidence["jobs"]
    if not isinstance(jobs, (list, tuple)) or isinstance(jobs, (str, bytes)):
        raise TrainingEvidenceError("jobs_must_be_sequence")
    expected = _validated_expected_jobs(jobs, check)

    sink = evidence["training_fit_summaries"]
    if not isinstance(sink, dict):
        raise TrainingEvidenceError("training_fit_summaries_must_be_mapping")
    if set(sink) != set(expected):
        raise TrainingEvidenceError("training_summary_job_set_mismatch")

    histories = {}
    source_bindings = []
    for job_id in sorted(expected):
        _check_refusal(check)
        job = expected[job_id]
        entry = sink[job_id]
        _validate_summary_entry(entry, job)
        histories[job_id] = _verify_summary(job, entry["summary"])
        source_bindings.append(
            {
                "job_id": job_id,
                "receipt_sha256": entry["receipt_sha256"],
                "summary_sha256": entry["summary_sha256"],
            }
        )

    _check_refusal(check)
    parsed_roots = _parse_monitor_roots(monitor_roots, allowed_roots)
    _check_refusal(check)

    monitor_docs = {}
    examined = []
    for job_id in sorted(expected):
        job = expected[job_id]
        projected_job = {
            "job_id": job_id,
            "policy_id": job["policy_id"],
            "model_id": job["model_id"],
            "seed": job["seed"],
            "stage": job["stage"],
        }
        chosen = None
        for monitor in parsed_roots:
            _check_refusal(check)
            candidate = _read_monitor_document(
                monitor["root"], job_id, allowed_roots, check
            )
            if candidate is None:
                continue
            candidate["label"] = monitor["label"]
            document = candidate["document"]
            _verify_monitor_identity(document, job)
            status = document.get("status")
            if not isinstance(status, str) or status not in _KNOWN_STATUSES:
                raise TrainingEvidenceError("monitor_status_invalid")
            if status != p08_live_monitor.STATUS_COMPLETE:
                candidate["selection"] = "excluded_noncomplete"
            else:
                try:
                    _validate_monitor_record(document, projected_job)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise TrainingEvidenceError("monitor_record_invalid") from exc
                _verify_complete_history(job, document, histories[job_id])
                if chosen is None:
                    candidate["selection"] = "selected"
                    chosen = candidate
                else:
                    candidate["selection"] = "exact_duplicate_numerical_history"
            examined.append(
                {
                    "path": candidate["path"],
                    "label": candidate["label"],
                    "job_id": job_id,
                    "file_sha256": candidate["sha256"],
                    "selection": candidate["selection"],
                }
            )
        if chosen is not None:
            monitor_docs[job_id] = chosen["document"]

    _check_refusal(check)
    projected = [
        {
            "job_id": job_id,
            "policy_id": expected[job_id]["policy_id"],
            "model_id": expected[job_id]["model_id"],
            "seed": expected[job_id]["seed"],
            "stage": expected[job_id]["stage"],
        }
        for job_id in sorted(expected)
    ]
    prepared = prepare_training_diagnostics(projected, monitor_docs)
    _check_refusal(check)

    receipt = {
        "schema": TRAINING_EVIDENCE_SCHEMA,
        "semantic_sha256": prepared["semantic_sha256"],
        "source_graph_sha256": diagnostics["graph_sha256"],
        "source_bindings": source_bindings,
        "examined_monitors": examined,
        "total_expected": len(expected),
        "monitored": len(monitor_docs),
        "missing_history": len(expected) - len(monitor_docs),
        "runtime_external_authentication_verified": False,
        "no_publication": True,
    }
    return {"prepared": prepared, "private_receipt": receipt}
