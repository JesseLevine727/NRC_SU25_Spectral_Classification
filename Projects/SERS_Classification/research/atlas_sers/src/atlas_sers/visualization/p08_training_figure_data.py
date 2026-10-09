"""P08 U1 training-diagnostic aggregation.

Pure in-memory, deterministic projection/aggregation of already-authenticated
complete live-monitor histories onto policy/recipe/stage descriptive curves.
No fitting, file IO, plotting, model selection or new uncertainty inference.
The caller owns monitor/graph/receipt authentication and historical coverage
audit; this module only summarizes what was authenticated and passed in.
"""

from __future__ import annotations

import math
import re

import numpy as np

from atlas_sers.visualization.p08_f02_render import _canonical_sha

__all__ = ["prepare_training_diagnostics"]

SCHEMA_VERSION = "nato-sers-p08-training-figure-data-v1"
MONITOR_SCHEMA = "p08_live_monitor_semantic_v1"
FIGURE_ID = "P08-U1-training-diagnostics"

POLICY_ORDER = ("PP-U-SG", "PP-U-ARPLS")
RECIPE_ORDER = ("D0-M", "D1", "D2", "D3")
STAGE_ORDER = ("source_fit", "calibration_model_fit", "final_refit")

METRIC_ORDER = (
    "chemical_ce",
    "total_loss",
    "supcon_loss",
    "paired_loss",
    "train_nll",
    "validation_nll",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
)

VALID_SEEDS = (20260805, 20260817, 20260829)
STOP_REASONS = ("patience", "epoch_limit", "fixed_duration")

POPULATION = {"spectra": 598, "masters": 69, "instruments": 10}

CAPTION = (
    "Authenticated complete monitor histories aggregated across unique fitting "
    "jobs into descriptive policy/recipe/stage curves. Curves summarize "
    "per-epoch minibatch-weighted training chemical cross-entropy and the "
    "composite total optimization objective; these sampled minibatch objectives "
    "differ from post-epoch evaluation negative log-likelihood. Auxiliary "
    "supcon/paired loss components are recipe-specific and are not directly "
    "comparable to the total objective across recipes. Validation metrics are "
    "recorded for source_fit runs only and never substitute for a held-out test "
    "set; calibration_model_fit and final_refit rows carry training-only "
    "diagnostics. Source_fit optimization uses a 30-epoch minimum stopping "
    "threshold and a 200-epoch maximum; refit stages reuse the caller-selected "
    "duration. Curves are fitting-run-weighted descriptive summaries over "
    "correlated runs (seeds/splits). Lines show medians; 10th/90th percentiles "
    "describe variation among saved fitting runs, not confidence intervals. "
    "Only runs that reached each epoch contribute; early stopping changes the "
    "composition, and stopped runs are not carried forward. Fitting jobs are "
    "not independent chemical samples: the dataset has 598 spectra from 69 "
    "physical samples and 10 instruments. Coverage is limited to saved SG/arPLS "
    "monitor histories; historical MIN losses are not pooled or invented. "
    "A missing history is not a zero loss or proof of a failed model. These curves do not support "
    "hypothesis tests, superiority claims, model selection, or new uncertainty "
    "inference. P08 learned-method performance is reported elsewhere."
)

# The saved monitor recipe is fixed by the (supcon_enabled, paired_enabled)
# flags. The expected-job ``model_id`` is the same recipe label so planned
# coverage can be attributed to a recipe even when a monitor is missing.
_RECIPE_BY_FLAGS = {
    (False, False): "D0-M",
    (True, False): "D1",
    (False, True): "D2",
    (True, True): "D3",
}

_JOB_RE = re.compile(r"^P08JOB-[0-9a-f]{64}$")
_SUFFIX_RE = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_FIELDS = frozenset({"job_id", "policy_id", "model_id", "seed", "stage"})

_MONITOR_FIELDS = frozenset(
    {
        "schema",
        "job_id",
        "model_id",
        "policy_id",
        "seed",
        "stage",
        "validation_available",
        "epoch_budget",
        "status",
        "stop_reason",
        "epochs_completed",
        "elapsed_seconds",
        "rows",
    }
)

_BASE_ROW_FIELDS = frozenset(
    {
        "epoch",
        "elapsed_seconds",
        "chemical_ce",
        "total_loss",
        "supcon_enabled",
        "paired_enabled",
        "supcon_loss",
        "paired_loss",
    }
)

_SOURCE_ROW_FIELDS = frozenset(
    {
        "train_nll",
        "validation_nll",
        "train_balanced_accuracy",
        "validation_balanced_accuracy",
        "best_epoch",
        "nonimproving_epochs",
    }
)

_OPTIONAL_ROW_FIELDS = frozenset({"total_optimizer_steps"})


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_expected_job(job):
    if not isinstance(job, dict):
        raise ValueError("expected job must be a dict")
    if set(job) != _EXPECTED_FIELDS:
        raise ValueError("expected job has an unexpected field set")
    job_id = job["job_id"]
    if not isinstance(job_id, str) or not _JOB_RE.match(job_id):
        raise ValueError("expected job_id must be P08JOB- plus 64 lowercase hex")
    if job["policy_id"] not in POLICY_ORDER:
        raise ValueError("unknown policy_id")
    if job["model_id"] not in RECIPE_ORDER:
        raise ValueError("unknown model_id")
    if not _is_int(job["seed"]) or job["seed"] not in VALID_SEEDS:
        raise ValueError("invalid seed")
    if job["stage"] not in STAGE_ORDER:
        raise ValueError("unknown stage")
    return job


def _validate_row(row, epoch_index, stage, previous_elapsed):
    if not isinstance(row, dict):
        raise ValueError("row must be a dict")
    required = set(_BASE_ROW_FIELDS)
    if stage == "source_fit":
        required |= set(_SOURCE_ROW_FIELDS)
    allowed = required | set(_OPTIONAL_ROW_FIELDS)
    keys = set(row)
    if not required <= keys:
        raise ValueError("row is missing required fields")
    if not keys <= allowed:
        raise ValueError("row has unexpected fields")

    if not _is_int(row["epoch"]) or row["epoch"] != epoch_index:
        raise ValueError("row epochs must be contiguous from 1")

    elapsed = row["elapsed_seconds"]
    if not _is_number(elapsed) or elapsed < 0 or elapsed < previous_elapsed:
        raise ValueError("row elapsed_seconds invalid")

    for field in ("chemical_ce", "total_loss"):
        if not _is_number(row[field]):
            raise ValueError("non-finite row metric: " + field)

    supcon = row["supcon_enabled"]
    paired = row["paired_enabled"]
    if not isinstance(supcon, bool) or not isinstance(paired, bool):
        raise ValueError("row recipe flags must be bool")
    if supcon:
        if not _is_number(row["supcon_loss"]):
            raise ValueError("enabled supcon_loss must be finite")
    elif row["supcon_loss"] is not None:
        raise ValueError("disabled supcon_loss must be None")
    if paired:
        if not _is_number(row["paired_loss"]):
            raise ValueError("enabled paired_loss must be finite")
    elif row["paired_loss"] is not None:
        raise ValueError("disabled paired_loss must be None")

    if "total_optimizer_steps" in row:
        steps = row["total_optimizer_steps"]
        if not _is_int(steps) or steps < 0:
            raise ValueError("total_optimizer_steps invalid")

    if stage == "source_fit":
        for field in ("train_nll", "validation_nll"):
            if not _is_number(row[field]):
                raise ValueError("non-finite " + field)
        for field in (
            "train_balanced_accuracy",
            "validation_balanced_accuracy",
        ):
            value = row[field]
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError("balanced accuracy out of range")
        best = row["best_epoch"]
        if not _is_int(best) or not 1 <= best <= epoch_index:
            raise ValueError("best_epoch out of range")
        if not _is_int(row["nonimproving_epochs"]) or row["nonimproving_epochs"] < 0:
            raise ValueError("nonimproving_epochs invalid")

    return float(elapsed), (supcon, paired)


def _validate_monitor_record(record, expected):
    if not isinstance(record, dict):
        raise ValueError("monitor record must be a dict")
    if set(record) != _MONITOR_FIELDS:
        raise ValueError("monitor record has an unexpected field set")
    if record["schema"] != MONITOR_SCHEMA:
        raise ValueError("unknown monitor schema")
    if record["status"] != "complete":
        raise ValueError("only complete monitor histories are accepted")
    if record["stop_reason"] not in STOP_REASONS:
        raise ValueError("unknown stop_reason")

    job_id = record["job_id"]
    if not isinstance(job_id, str) or not _SUFFIX_RE.match(job_id):
        raise ValueError("monitor job_id must be 64 lowercase hex")
    if "P08JOB-" + job_id != expected["job_id"]:
        raise ValueError("monitor job_id does not match expected job")

    for field in ("model_id", "policy_id", "seed", "stage"):
        if record[field] != expected[field]:
            raise ValueError("monitor identity mismatch: " + field)
    if not _is_int(record["seed"]) or record["seed"] not in VALID_SEEDS:
        raise ValueError("invalid seed")

    stage = expected["stage"]
    available = record["validation_available"]
    if not isinstance(available, bool) or available != (stage == "source_fit"):
        raise ValueError("validation_available mismatch")

    budget = record["epoch_budget"]
    if not _is_int(budget) or not 1 <= budget <= 200:
        raise ValueError("epoch_budget out of range")
    if stage == "source_fit" and budget != 200:
        raise ValueError("source_fit epoch_budget must be 200")

    rows = record["rows"]
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    completed = record["epochs_completed"]
    if not _is_int(completed) or completed < 1 or completed > budget:
        raise ValueError("epochs_completed out of range")
    if completed != len(rows):
        raise ValueError("epochs_completed must equal len(rows)")
    if stage == "source_fit" and not 30 <= completed <= 200:
        raise ValueError("source_fit epochs_completed out of range")

    flags = None
    last_elapsed = 0.0
    for index, row in enumerate(rows, start=1):
        last_elapsed, row_flags = _validate_row(row, index, stage, last_elapsed)
        if flags is None:
            flags = row_flags
        elif flags != row_flags:
            raise ValueError("inconsistent recipe flags across rows")

    recipe = _RECIPE_BY_FLAGS[flags]
    if recipe != expected["model_id"]:
        raise ValueError("row flags recipe does not match expected model_id")

    elapsed = record["elapsed_seconds"]
    if not _is_number(elapsed) or elapsed < 0 or elapsed < last_elapsed:
        raise ValueError("monitor elapsed_seconds invalid")

    return {"recipe": recipe, "n_epochs": completed, "rows": rows}


def prepare_training_diagnostics(expected_jobs, monitor_records):
    """Project authenticated complete monitor histories onto 24 groups/curves."""
    if not isinstance(expected_jobs, list):
        raise ValueError("expected_jobs must be a list")
    if not expected_jobs:
        raise ValueError("expected_jobs must be non-empty")
    if not isinstance(monitor_records, dict):
        raise ValueError("monitor_records must be a dict")

    validated = []
    seen = set()
    for job in expected_jobs:
        checked = _validate_expected_job(job)
        if checked["job_id"] in seen:
            raise ValueError("duplicate expected job_id")
        seen.add(checked["job_id"])
        validated.append(checked)

    if set(monitor_records) - seen:
        raise ValueError("monitor_records contains unrecognized jobs")

    plan = {}
    for job in validated:
        key = (job["policy_id"], job["model_id"], job["stage"])
        plan[key] = plan.get(key, 0) + 1

    runs = {}
    monitored_total = 0
    for job in validated:
        if job["job_id"] not in monitor_records:
            continue
        record = monitor_records[job["job_id"]]
        run = _validate_monitor_record(record, job)
        key = (job["policy_id"], run["recipe"], job["stage"])
        runs.setdefault(key, []).append(run)
        monitored_total += 1

    groups = []
    curves = []
    for policy in POLICY_ORDER:
        for recipe in RECIPE_ORDER:
            for stage in STAGE_ORDER:
                key = (policy, recipe, stage)
                planned = plan.get(key, 0)
                run_list = runs.get(key, [])
                monitored = len(run_list)
                missing = planned - monitored
                if planned == 0:
                    status = "no_registered_jobs"
                elif monitored == 0:
                    status = "missing_histories"
                elif monitored < planned:
                    status = "partial_history_coverage"
                else:
                    status = "complete_history_coverage"
                max_epoch = max((run["n_epochs"] for run in run_list), default=None)
                if monitored == 0:
                    coverage = "NA"
                elif missing == 0:
                    coverage = "complete"
                else:
                    coverage = "partial"
                groups.append(
                    {
                        "policy_id": policy,
                        "recipe": recipe,
                        "stage": stage,
                        "planned_jobs": planned,
                        "monitored_jobs": monitored,
                        "missing_history_jobs": missing,
                        "max_recorded_epoch": max_epoch,
                        "status": status,
                        "coverage": coverage,
                    }
                )
                if max_epoch is None:
                    continue
                for epoch in range(1, max_epoch + 1):
                    at_epoch = [run for run in run_list if run["n_epochs"] >= epoch]
                    n_runs = len(at_epoch)
                    for metric in METRIC_ORDER:
                        values = []
                        for run in at_epoch:
                            value = run["rows"][epoch - 1].get(metric)
                            if value is not None:
                                values.append(float(value))
                        finite_count = len(values)
                        if finite_count > 0:
                            array = np.asarray(values, dtype=float)
                            q10, median, q90 = np.quantile(array, [0.1, 0.5, 0.9], method="linear")
                            reason = None
                        else:
                            q10 = median = q90 = None
                            reason = "metric_not_recorded"
                        curves.append(
                            {
                                "policy_id": policy,
                                "recipe": recipe,
                                "stage": stage,
                                "epoch": epoch,
                                "metric": metric,
                                "n_runs_at_epoch": n_runs,
                                "finite_count": finite_count,
                                "undefined_count": n_runs - finite_count,
                                "median": median,
                                "q10": q10,
                                "q90": q90,
                                "reason": reason,
                            }
                        )

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "caption": CAPTION,
        "groups": groups,
        "curves": curves,
        "metric_order": list(METRIC_ORDER),
        "policy_order": list(POLICY_ORDER),
        "recipe_order": list(RECIPE_ORDER),
        "stage_order": list(STAGE_ORDER),
        "population": dict(POPULATION),
    }
    semantic_sha256 = _canonical_sha(semantic)
    manifest = {
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "external_authentication_verified": False,
        "semantic_sha256": semantic_sha256,
        "counts": {
            "total_expected": len(validated),
            "monitored": monitored_total,
            "missing_history": len(validated) - monitored_total,
            "groups": len(groups),
            "curve_rows": len(curves),
        },
    }
    return {
        "semantic": semantic,
        "semantic_sha256": semantic_sha256,
        "manifest": manifest,
    }
