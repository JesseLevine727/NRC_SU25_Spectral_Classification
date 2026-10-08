"""P08-U1 T276 source-only selection adapter.

Thin, pure adapter over already-authenticated frozen P08 graph job records and
in-memory source-fit summaries.  It never fits, never reads files, never reads
held outcomes and never authorizes execution.  A future controller must verify
actual receipt and persisted artifact bytes before calling it.

* ``select_classical`` consumes one ``select_hyperparameters`` job plus its exact
  ``source_validation_prediction`` dependencies and applies the unchanged
  inherited ``atlas_sers.evaluation.classical.select_lexicographic_candidate``
  objective to the job's own model family.
* ``select_epochs`` consumes one ``select_refit_epochs`` job plus its exact
  dependency set and computes the per-seed duration with the same
  ``clip(round(median(best_epoch)), 30, 200)`` rule as the frozen
  ``atlas_sers.evaluation.p05_selection.inherit_refit_epochs`` over the already
  frozen U1 subset.

Returned records remain PRIVATE (job/context/unit ids) and JSON-safe.  No error
message carries a private field.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from statistics import median

import pandas as pd

from atlas_sers.evaluation import p08_plan
from atlas_sers.evaluation.classical import select_lexicographic_candidate
from atlas_sers.governance.canonical import sha256_value

__all__ = [
    "SCHEMA_VERSION",
    "REGISTRY_SHA256",
    "SelectionAdapterError",
    "select_classical",
    "select_epochs",
]

SCHEMA_VERSION = "nato-sers-p08-u1-selection-adapter-v1"
REGISTRY_SHA256 = "046ebfa9023591ba91f48b797fe3bb037f6e7e82cb4bfb34800a9aa12b468bba"

PERMITTED_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
SVM_MODEL = "C-RBF-SVM"
SVM_SEED = "deterministic"
SEEDS = (20260805, 20260817, 20260829)
NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")

SELECT_HYPERPARAMETERS = "select_hyperparameters"
SELECT_REFIT_EPOCHS = "select_refit_epochs"
SOURCE_FIT = "source_fit"
SOURCE_VALIDATION_PREDICTION = "source_validation_prediction"
SELECTED_CANDIDATE_DEPENDENT = "source_selection_dependent"
EPOCH_DEPENDENT = "source_epoch_dependent"
NOT_APPLICABLE = "not_applicable"
FIXED_SPEC = "fixed_spec"
COMPLETE = "complete"

REFIT_EPOCH_MINIMUM = 30
REFIT_EPOCH_MAXIMUM = 200
MINIMUM_BEST_EPOCH = 1
MINIMUM_COMPLETED_EPOCHS = 30
MAXIMUM_COMPLETED_EPOCHS = 200

JOB_FIELDS = p08_plan.JOB_FIELDS
_JOB_KEYS = frozenset(JOB_FIELDS) | {"job_id"}

POLICY_REPRESENTATION = p08_plan.POLICY_REPRESENTATION

DEPENDENCY_FIELDS = ("fit_job", "prediction_job", "summary")

REGISTRY_COLUMNS = (
    "candidate_id",
    "model_id",
    "family_order",
    "family_candidate_order",
    "declared_candidate_order",
    "parameters_json",
    "hyperparameter_sha256",
    "complexity_rank",
    "stochastic",
    "technical_seeds",
    "seed_count",
)

CLASSICAL_SUMMARY_FIELDS = (
    "status",
    "fit_id",
    "model_id",
    "candidate_id",
    "seed",
    "fit_uid_sha256",
    "validation_uid_sha256",
    "fit_master_sha256",
    "validation_metrics",
)

NEURAL_SUMMARY_FIELDS = (
    "status",
    "best_epoch",
    "epochs_completed",
    "history",
    "seed",
    "slot_id",
    "unit_id",
)

_HEX_DIGITS = frozenset("0123456789abcdef")

_REASON_CODES = frozenset({
    "invalid_arguments",
    "selection_job_invalid",
    "policy_not_permitted",
    "resolution_invalid",
    "model_not_permitted",
    "seed_invalid",
    "seed_mismatch",
    "dependencies_not_mapping",
    "dependency_missing",
    "dependency_extra",
    "dependency_invalid",
    "dependency_duplicate",
    "dependency_coverage_mismatch",
    "fit_job_invalid",
    "prediction_job_invalid",
    "job_pair_mismatch",
    "selection_job_mismatch",
    "summary_invalid",
    "summary_status_not_complete",
    "summary_identity_mismatch",
    "summary_metric_invalid",
    "registry_bytes_invalid",
    "registry_sha256_mismatch",
    "registry_csv_invalid",
    "registry_header_invalid",
    "registry_candidate_invalid",
    "registry_candidate_hash_mismatch",
    "registry_parameters_invalid",
    "registry_order_invalid",
    "family_mismatch",
    "candidate_hash_mismatch",
    "selection_failed",
    "nonfinite_output",
    "unlisted_reason_code",
})


class SelectionAdapterError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise SelectionAdapterError(reason_code) from None


def _require_mapping(value, reason_code):
    if not isinstance(value, Mapping):
        _fail(reason_code)
    return value


def _require_sequence(value, reason_code):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(reason_code)
    return value


def _require_identifier(value, reason_code):
    if type(value) is not str or not value or value != value.strip():
        _fail(reason_code)
    return value


def _require_sha256(value, reason_code):
    if type(value) is not str or len(value) != 64:
        _fail(reason_code)
    for character in value:
        if character not in _HEX_DIGITS:
            _fail(reason_code)
    return value


def _require_integer(value, reason_code, minimum=None, maximum=None):
    if type(value) is bool or not hasattr(value, "__index__"):
        _fail(reason_code)
    result = int(value.__index__())
    if minimum is not None and result < minimum:
        _fail(reason_code)
    if maximum is not None and result > maximum:
        _fail(reason_code)
    return result


def _require_finite(value, reason_code):
    if type(value) is bool or not isinstance(value, numbers.Real):
        _fail(reason_code)
    try:
        number = float(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (OverflowError, ValueError):
        _fail(reason_code)
    if not math.isfinite(number):
        _fail(reason_code)
    return number


def _recompute_job_id(job):
    payload = {name: job[name] for name in JOB_FIELDS}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "P08JOB-" + hashlib.sha256(encoded).hexdigest()


def _require_job(raw, reason_code, expected_stage):
    job = _require_mapping(raw, reason_code)
    if set(job.keys()) != _JOB_KEYS:
        _fail(reason_code)
    if job["stage"] != expected_stage:
        _fail(reason_code)
    dependencies = _require_sequence(job["dependencies"], reason_code)
    for dependency in dependencies:
        _require_identifier(dependency, reason_code)
    if list(dependencies) != sorted(dependencies):
        _fail("dependency_invalid")
    if len(set(dependencies)) != len(dependencies):
        _fail("dependency_duplicate")
    for name in (
        "policy_id",
        "representation_id",
        "context_id",
        "model_id",
        "stage",
        "resolution",
        "evidence_status",
        "job_id",
    ):
        _require_identifier(job[name], reason_code)
    for name in ("array_sha256", "model_spec_sha256", "test_uid_sha256"):
        _require_sha256(job[name], reason_code)
    if job["job_id"] != _recompute_job_id(job):
        _fail(reason_code)
    return job


def _verify_selection_job(selection_job, *, stage, resolution, models, seed_required):
    job = _require_job(selection_job, "selection_job_invalid", stage)
    if job["policy_id"] not in PERMITTED_POLICIES:
        _fail("policy_not_permitted")
    if job["resolution"] != resolution:
        _fail("resolution_invalid")
    if job["model_id"] not in models:
        _fail("model_not_permitted")
    if job["unit_id"] != NOT_APPLICABLE:
        _fail("selection_job_invalid")
    if job["candidate_id"] != SELECTED_CANDIDATE_DEPENDENT:
        _fail("selection_job_invalid")
    if job["hyperparameter_sha256"] != NOT_APPLICABLE:
        _fail("selection_job_invalid")
    if job["fit_uid_sha256"] != NOT_APPLICABLE:
        _fail("selection_job_invalid")
    if job["validation_uid_sha256"] != NOT_APPLICABLE:
        _fail("selection_job_invalid")
    if seed_required:
        seed = _require_integer(job["seed"], "seed_invalid")
        if seed not in SEEDS:
            _fail("seed_invalid")
    elif job["seed"] != NOT_APPLICABLE:
        _fail("seed_invalid")
    return job


def _dependency_pairs(selection_job, dependencies):
    mapping = _require_mapping(dependencies, "dependencies_not_mapping")
    declared = list(selection_job["dependencies"])
    if len(declared) != len(set(declared)):
        _fail("dependency_duplicate")
    observed = set()
    for key in mapping.keys():
        _require_identifier(key, "dependency_invalid")
        observed.add(key)
    expected = set(declared)
    if observed - expected:
        _fail("dependency_extra")
    if expected - observed:
        _fail("dependency_missing")
    pairs = []
    for prediction_job_id in sorted(declared):
        entry = _require_mapping(mapping[prediction_job_id], "dependency_invalid")
        if set(entry.keys()) != set(DEPENDENCY_FIELDS):
            _fail("dependency_invalid")
        fit_job = _require_job(entry["fit_job"], "fit_job_invalid", SOURCE_FIT)
        prediction_job = _require_job(
            entry["prediction_job"], "prediction_job_invalid", SOURCE_VALIDATION_PREDICTION
        )
        if prediction_job["job_id"] != prediction_job_id:
            _fail("dependency_invalid")
        summary = _require_mapping(entry["summary"], "summary_invalid")
        pairs.append((prediction_job_id, fit_job, prediction_job, summary))
    return pairs


def _verify_pairing(selection_job, fit_job, prediction_job):
    for name in JOB_FIELDS:
        if name in ("stage", "dependencies"):
            continue
        if fit_job[name] != prediction_job[name]:
            _fail("job_pair_mismatch")
    if fit_job["resolution"] != FIXED_SPEC:
        _fail("resolution_invalid")
    if list(fit_job["dependencies"]):
        _fail("job_pair_mismatch")
    if list(prediction_job["dependencies"]) != [fit_job["job_id"]]:
        _fail("job_pair_mismatch")
    policy_id = _require_identifier(selection_job["policy_id"], "selection_job_invalid")
    if POLICY_REPRESENTATION.get(policy_id) != selection_job["representation_id"]:
        _fail("selection_job_mismatch")
    for name in (
        "policy_id",
        "representation_id",
        "array_sha256",
        "context_id",
        "model_spec_sha256",
        "model_id",
        "test_uid_sha256",
    ):
        if prediction_job[name] != selection_job[name]:
            _fail("selection_job_mismatch")


def _verify_classical_pair(selection_job, fit_job, prediction_job, summary):
    _verify_pairing(selection_job, fit_job, prediction_job)
    unit_id = _require_identifier(fit_job["unit_id"], "fit_job_invalid")
    model_id = selection_job["model_id"]
    seed = fit_job["seed"]
    if model_id == SVM_MODEL:
        if seed != SVM_SEED:
            _fail("seed_invalid")
    else:
        parsed_seed = _require_integer(seed, "seed_invalid")
        if parsed_seed not in SEEDS:
            _fail("seed_invalid")
    candidate_id = _require_identifier(fit_job["candidate_id"], "fit_job_invalid")
    if candidate_id in (SELECTED_CANDIDATE_DEPENDENT, "fixed_recipe"):
        _fail("fit_job_invalid")
    _require_sha256(fit_job["hyperparameter_sha256"], "fit_job_invalid")
    _require_sha256(fit_job["fit_uid_sha256"], "fit_job_invalid")
    _require_sha256(fit_job["validation_uid_sha256"], "fit_job_invalid")
    for name in CLASSICAL_SUMMARY_FIELDS:
        if name not in summary:
            _fail("summary_invalid")
    if summary["status"] != COMPLETE:
        _fail("summary_status_not_complete")
    for name, expected in (
        ("fit_id", fit_job["job_id"]),
        ("model_id", fit_job["model_id"]),
        ("candidate_id", fit_job["candidate_id"]),
        ("seed", fit_job["seed"]),
        ("fit_uid_sha256", fit_job["fit_uid_sha256"]),
        ("validation_uid_sha256", fit_job["validation_uid_sha256"]),
    ):
        if summary[name] != expected:
            _fail("summary_identity_mismatch")
    metrics = _require_mapping(summary["validation_metrics"], "summary_metric_invalid")
    balanced_accuracy = _require_finite(
        metrics.get("balanced_accuracy"), "summary_metric_invalid"
    )
    macro_f1 = _require_finite(metrics.get("macro_f1"), "summary_metric_invalid")
    if not (0.0 <= balanced_accuracy <= 1.0) or not (0.0 <= macro_f1 <= 1.0):
        _fail("summary_metric_invalid")
    return unit_id, candidate_id, seed, balanced_accuracy, macro_f1


def _verify_neural_pair(selection_job, fit_job, prediction_job, summary):
    _verify_pairing(selection_job, fit_job, prediction_job)
    unit_id = _require_identifier(fit_job["unit_id"], "fit_job_invalid")
    seed = selection_job["seed"]
    if fit_job["seed"] != seed:
        _fail("seed_mismatch")
    if fit_job["candidate_id"] != "fixed_recipe":
        _fail("fit_job_invalid")
    hyperparameter_sha256 = _require_sha256(
        fit_job["hyperparameter_sha256"], "fit_job_invalid"
    )
    model_spec_sha256 = _require_sha256(
        fit_job["model_spec_sha256"], "fit_job_invalid"
    )
    if hyperparameter_sha256 != model_spec_sha256:
        _fail("fit_job_invalid")
    _require_sha256(fit_job["fit_uid_sha256"], "fit_job_invalid")
    _require_sha256(fit_job["validation_uid_sha256"], "fit_job_invalid")
    for name in NEURAL_SUMMARY_FIELDS:
        if name not in summary:
            _fail("summary_invalid")
    if summary["status"] != COMPLETE:
        _fail("summary_status_not_complete")
    if summary["seed"] != seed:
        _fail("seed_mismatch")
    recipe_value = summary.get("recipe_id")
    if "recipe" in summary:
        if recipe_value is not None and summary["recipe"] != recipe_value:
            _fail("summary_identity_mismatch")
        recipe_value = summary["recipe"]
    if recipe_value is None or recipe_value != fit_job["model_id"]:
        _fail("summary_identity_mismatch")
    if summary["slot_id"] != fit_job["job_id"]:
        _fail("summary_identity_mismatch")
    if summary["unit_id"] != unit_id:
        _fail("summary_identity_mismatch")
    completed = _require_integer(
        summary["epochs_completed"],
        "summary_metric_invalid",
        minimum=MINIMUM_COMPLETED_EPOCHS,
        maximum=MAXIMUM_COMPLETED_EPOCHS,
    )
    best_epoch = _require_integer(
        summary["best_epoch"],
        "summary_metric_invalid",
        minimum=MINIMUM_BEST_EPOCH,
        maximum=completed,
    )
    _require_sequence(summary["history"], "summary_invalid")
    return unit_id, seed, best_epoch, completed


def _parse_int_string(value, reason_code):
    if type(value) is not str or not value:
        _fail(reason_code)
    try:
        return int(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail(reason_code)


def _parse_bool_string(value, reason_code):
    if value in ("true", "True", "1"):
        return True
    if value in ("false", "False", "0"):
        return False
    _fail(reason_code)


def _parse_seed_strings(value, reason_code):
    if type(value) is not str or not value:
        _fail(reason_code)
    parts = [
        part
        for part in value.replace("|", ",").replace(";", ",").replace(" ", ",").split(",")
        if part
    ]
    if not parts:
        _fail(reason_code)
    return parts


def _parse_parameters(raw, expected_sha256):
    if type(raw) is not str or not raw:
        _fail("registry_parameters_invalid")
    try:
        parsed = json.loads(raw)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("registry_parameters_invalid")
    if type(parsed) is not dict:
        _fail("registry_parameters_invalid")
    try:
        digest = sha256_value(parsed)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("registry_parameters_invalid")
    if digest != expected_sha256:
        _fail("registry_candidate_hash_mismatch")
    return parsed


def _parse_registry(candidate_registry_bytes):
    if type(candidate_registry_bytes) is not bytes or not candidate_registry_bytes:
        _fail("registry_bytes_invalid")
    if hashlib.sha256(candidate_registry_bytes).hexdigest() != REGISTRY_SHA256:
        _fail("registry_sha256_mismatch")
    try:
        frame = pd.read_csv(
            io.BytesIO(candidate_registry_bytes), dtype=str, keep_default_na=False
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("registry_csv_invalid")
    if frame.empty or tuple(frame.columns) != REGISTRY_COLUMNS:
        _fail("registry_header_invalid")
    rows = []
    for raw in frame.to_dict("records"):
        candidate_id = _require_identifier(raw["candidate_id"], "registry_candidate_invalid")
        model_id = _require_identifier(raw["model_id"], "registry_candidate_invalid")
        hyperparameter_sha256 = _require_sha256(
            raw["hyperparameter_sha256"], "registry_candidate_invalid"
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "family_order": _parse_int_string(
                    raw["family_order"], "registry_candidate_invalid"
                ),
                "family_candidate_order": _parse_int_string(
                    raw["family_candidate_order"], "registry_candidate_invalid"
                ),
                "declared_candidate_order": _parse_int_string(
                    raw["declared_candidate_order"], "registry_candidate_invalid"
                ),
                "parameters": _parse_parameters(
                    raw["parameters_json"], hyperparameter_sha256
                ),
                "hyperparameter_sha256": hyperparameter_sha256,
                "complexity_rank": _parse_int_string(
                    raw["complexity_rank"], "registry_candidate_invalid"
                ),
                "stochastic": _parse_bool_string(
                    raw["stochastic"], "registry_candidate_invalid"
                ),
                "technical_seeds": _parse_seed_strings(
                    raw["technical_seeds"], "registry_candidate_invalid"
                ),
                "seed_count": _parse_int_string(
                    raw["seed_count"], "registry_candidate_invalid"
                ),
            }
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        _fail("registry_candidate_invalid")
    if len({row["declared_candidate_order"] for row in rows}) != len(rows):
        _fail("registry_order_invalid")
    return rows


def _verify_family(family, expected_seed_strings, model_id):
    if len({row["family_order"] for row in family}) != 1:
        _fail("registry_order_invalid")
    if len({row["family_candidate_order"] for row in family}) != len(family):
        _fail("registry_order_invalid")
    expected_count = 1 if model_id == SVM_MODEL else len(SEEDS)
    expected_stochastic = model_id != SVM_MODEL
    for row in family:
        if row["seed_count"] != expected_count:
            _fail("registry_candidate_invalid")
        if row["stochastic"] != expected_stochastic:
            _fail("registry_candidate_invalid")
        if set(row["technical_seeds"]) != expected_seed_strings:
            _fail("registry_candidate_invalid")


def _median_duration(best_epochs):
    """Parity with the frozen p05_selection.inherit_refit_epochs clipping."""

    rounded = int(round(median(best_epochs)))
    return max(REFIT_EPOCH_MINIMUM, min(REFIT_EPOCH_MAXIMUM, rounded))


def _json_value(value):
    if type(value) is bool:
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        try:
            number = float(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OverflowError, ValueError):
            _fail("nonfinite_output")
        if not math.isfinite(number):
            _fail("nonfinite_output")
        return number
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (KeyboardInterrupt, SystemExit):
            raise
        except SelectionAdapterError:
            raise
        except Exception:
            _fail("nonfinite_output")
    _fail("nonfinite_output")


def _json_record(row):
    return {str(key): _json_value(value) for key, value in row.items()}


def _json_safe(value):
    try:
        json.dumps(value, allow_nan=False)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("nonfinite_output")
    return value


def _select_classical(selection_job, dependencies, candidate_registry_bytes):
    job = _verify_selection_job(
        selection_job,
        stage=SELECT_HYPERPARAMETERS,
        resolution=SELECTED_CANDIDATE_DEPENDENT,
        models=CLASSICAL_MODELS,
        seed_required=False,
    )
    model_id = job["model_id"]
    expected_seeds = (SVM_SEED,) if model_id == SVM_MODEL else SEEDS
    pairs = _dependency_pairs(job, dependencies)
    if not pairs:
        _fail("dependency_coverage_mismatch")
    records = []
    observed = []
    units = set()
    candidate_hashes = {}
    unit_provenance = {}
    evidence = []
    for prediction_job_id, fit_job, prediction_job, summary in pairs:
        unit_id, candidate_id, seed, balanced_accuracy, macro_f1 = _verify_classical_pair(
            job, fit_job, prediction_job, summary
        )
        unit_signature = (
            fit_job["fit_uid_sha256"],
            fit_job["validation_uid_sha256"],
        )
        if unit_provenance.setdefault(unit_id, unit_signature) != unit_signature:
            _fail("dependency_invalid")
        observed.append((unit_id, candidate_id, seed))
        units.add(unit_id)
        candidate_hashes.setdefault(candidate_id, set()).add(
            fit_job["hyperparameter_sha256"]
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "selection_unit_id": unit_id,
                "seed": seed,
                "status": COMPLETE,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
            }
        )
        evidence.append(
            {
                "prediction_job_id": prediction_job_id,
                "fit_job_id": fit_job["job_id"],
                "candidate_id": candidate_id,
                "selection_unit_id": unit_id,
                "seed": seed,
                "fit_uid_sha256": fit_job["fit_uid_sha256"],
                "validation_uid_sha256": fit_job["validation_uid_sha256"],
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
            }
        )
    if len(observed) != len(set(observed)):
        _fail("dependency_duplicate")
    if any(len(values) != 1 for values in candidate_hashes.values()):
        _fail("candidate_hash_mismatch")

    rows = _parse_registry(candidate_registry_bytes)
    family = [row for row in rows if row["model_id"] == model_id]
    if not family:
        _fail("family_mismatch")
    _verify_family(family, {str(seed) for seed in expected_seeds}, model_id)
    by_candidate_id = {row["candidate_id"]: row for row in family}
    expected_coverage = {
        (unit_id, candidate_id, seed)
        for unit_id in units
        for candidate_id in by_candidate_id
        for seed in expected_seeds
    }
    if set(observed) != expected_coverage:
        _fail("dependency_coverage_mismatch")
    for candidate_id, hashes in candidate_hashes.items():
        declared = by_candidate_id.get(candidate_id)
        if declared is None or declared["hyperparameter_sha256"] != next(iter(hashes)):
            _fail("candidate_hash_mismatch")

    registry = pd.DataFrame(
        [
            {
                "candidate_id": row["candidate_id"],
                "model_id": row["model_id"],
                "complexity_rank": row["complexity_rank"],
                "declared_candidate_order": row["declared_candidate_order"],
                "seed_count": row["seed_count"],
            }
            for row in family
        ]
    )
    try:
        winner, aggregation = select_lexicographic_candidate(
            pd.DataFrame(records), registry
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("selection_failed")
    selected_candidate_id = _require_identifier(
        str(winner["candidate_id"]), "selection_failed"
    )
    selected = by_candidate_id.get(selected_candidate_id)
    if selected is None:
        _fail("selection_failed")

    dependency_job_ids = sorted(item[0] for item in pairs)
    evidence_sha256 = sha256_value(evidence)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": SELECT_HYPERPARAMETERS,
        "policy_id": job["policy_id"],
        "context_id": job["context_id"],
        "representation_id": job["representation_id"],
        "array_sha256": job["array_sha256"],
        "model_id": model_id,
        "model_spec_sha256": job["model_spec_sha256"],
        "selection_job_id": job["job_id"],
        "dependency_job_ids": dependency_job_ids,
        "dependency_evidence_sha256": evidence_sha256,
        "candidate_registry_sha256": REGISTRY_SHA256,
        "selected_candidate_id": selected_candidate_id,
        "selected_hyperparameter_sha256": selected["hyperparameter_sha256"],
        "selected_parameters": selected["parameters"],
        "counts": {
            "dependency_count": len(pairs),
            "selection_unit_count": len(units),
            "family_candidate_count": len(family),
            "seed_count_per_family_candidate": (
                1 if model_id == SVM_MODEL else len(SEEDS)
            ),
        },
        "trace": [_json_record(row) for row in aggregation.to_dict("records")],
        "execution_authorized": False,
    }
    result["selection_state_sha256"] = sha256_value(
        {
            "selection_job_id": result["selection_job_id"],
            "dependency_job_ids": dependency_job_ids,
            "dependency_evidence_sha256": evidence_sha256,
            "selected_candidate_id": selected_candidate_id,
            "selected_hyperparameter_sha256": selected["hyperparameter_sha256"],
        }
    )
    return _json_safe(result)


def _select_epochs(selection_job, dependencies):
    job = _verify_selection_job(
        selection_job,
        stage=SELECT_REFIT_EPOCHS,
        resolution=EPOCH_DEPENDENT,
        models=NEURAL_RECIPES,
        seed_required=True,
    )
    pairs = _dependency_pairs(job, dependencies)
    unit_best_epochs = {}
    evidence = []
    for prediction_job_id, fit_job, prediction_job, summary in pairs:
        unit_id, seed, best_epoch, completed = _verify_neural_pair(
            job, fit_job, prediction_job, summary
        )
        if unit_id in unit_best_epochs:
            _fail("dependency_duplicate")
        unit_best_epochs[unit_id] = best_epoch
        evidence.append(
            {
                "prediction_job_id": prediction_job_id,
                "fit_job_id": fit_job["job_id"],
                "selection_unit_id": unit_id,
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_completed": completed,
                "fit_uid_sha256": fit_job["fit_uid_sha256"],
                "validation_uid_sha256": fit_job["validation_uid_sha256"],
            }
        )
    if not unit_best_epochs:
        _fail("dependency_coverage_mismatch")
    ordered_units = sorted(unit_best_epochs)
    best_epochs = [unit_best_epochs[unit_id] for unit_id in ordered_units]
    duration = _median_duration(best_epochs)
    dependency_job_ids = sorted(item[0] for item in pairs)
    evidence_sha256 = sha256_value(evidence)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": SELECT_REFIT_EPOCHS,
        "policy_id": job["policy_id"],
        "context_id": job["context_id"],
        "representation_id": job["representation_id"],
        "array_sha256": job["array_sha256"],
        "model_id": job["model_id"],
        "model_spec_sha256": job["model_spec_sha256"],
        "selection_job_id": job["job_id"],
        "recipe_id": job["model_id"],
        "seed": job["seed"],
        "dependency_job_ids": dependency_job_ids,
        "dependency_evidence_sha256": evidence_sha256,
        "unit_best_epochs": unit_best_epochs,
        "best_epochs": best_epochs,
        "epochs": duration,
        "duration": duration,
        "refit_epoch_minimum": REFIT_EPOCH_MINIMUM,
        "refit_epoch_maximum": REFIT_EPOCH_MAXIMUM,
        "counts": {
            "dependency_count": len(pairs),
            "selection_unit_count": len(ordered_units),
        },
        "execution_authorized": False,
    }
    result["selection_state_sha256"] = sha256_value(
        {
            "selection_job_id": result["selection_job_id"],
            "dependency_job_ids": dependency_job_ids,
            "dependency_evidence_sha256": evidence_sha256,
            "recipe_id": result["recipe_id"],
            "seed": result["seed"],
            "best_epochs": best_epochs,
            "duration": duration,
        }
    )
    return _json_safe(result)


def select_classical(*, selection_job, dependencies, candidate_registry_bytes):
    """Apply the frozen lexicographic classical selection to one family.

    Inputs are read only.  A returned record is a private, JSON-safe selection
    statement, not an execution capability.  Missing or failed evidence is a
    hard error; no rows are dropped and no fallback is applied.
    """

    try:
        return _select_classical(selection_job, dependencies, candidate_registry_bytes)
    except SelectionAdapterError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("selection_failed")


def select_epochs(*, selection_job, dependencies):
    """Return the frozen per-seed refit duration for one frozen U1 subset.

    Mirrors ``inherit_refit_epochs``: the median of the per-unit best epochs is
    rounded with the Python convention and clipped to ``[30, 200]``.  No recipe
    selection, fitting, history validation or file access happens here.
    """

    try:
        return _select_epochs(selection_job, dependencies)
    except SelectionAdapterError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("selection_failed")
