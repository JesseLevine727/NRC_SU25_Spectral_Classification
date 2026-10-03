"""Shared in-memory source-prediction semantic boundary (P08-T083).

The prospective U0 smoke needs one common check that a caller, *after*
independently authenticating bytes and extracting score arrays without
coercing or reordering them, presents a self-consistent source-fit /
source-validation-prediction job pair together with an ordered score matrix
whose row and column vocabularies match the declared validation UIDs and
ordered classes.

This module preserves the inherited P03 score-table / P05 NPZ formats: it does
not define, invent or load another persisted prediction format.  It receives
already-extracted in-memory arrays only.

Non-claims
----------
* This boundary does not authenticate bytes, artifacts, ledgers, receipts,
  physical-master independence, model provenance or completed fitting.
* ``declared_job_pair_verified`` and ``source_prediction_structure_verified``
  record only internal consistency against caller-supplied pins.  They are not
  evidence of provenance.
* ``external_authentication_verified``, ``physical_role_isolation_verified``,
  ``training_completion_verified`` and ``prediction_parity_verified`` are
  always ``False``.
* A same-shaped finite but wrong matrix is externally indistinguishable from a
  correct one here; this boundary does not detect it as model error.
* No observations, true labels, file paths, raw scores or observation ids
  leave this interface: only scalars, counts and hashes are returned.
* Job-role hash differences do not prove physical-master independence.
* Model-spec and candidate-grid membership must be authenticated separately by
  the caller; this module never recomputes model specifications or grids.
* Execution through this module is never authorized.
"""

from __future__ import annotations

import numpy as np

from atlas_sers.evaluation.p08_plan import (
    CLASSICAL_MODELS,
    EVIDENCE_FUTURE,
    FIXED_SPEC,
    JOB_FIELDS,
    NEURAL_RECIPES,
    POLICY_REPRESENTATION,
    SEEDS,
    SVM_SEED,
    _hash,
)
from atlas_sers.governance.canonical import sha256_value

__all__ = [
    "PredictionError",
    "SCHEMA_VERSION",
    "require_scientific_execution",
    "verify_source_prediction_values",
]

SCHEMA_VERSION = "nato-sers-p08-source-prediction-check-v1"

_FIT_STAGE = "source_fit"
_PREDICTION_STAGE = "source_validation_prediction"
_NEURAL_CANDIDATE = "fixed_recipe"
_SVM_MODEL = "C-RBF-SVM"

_SUPPORTED_POLICIES = ("PP-U-SG", "PP-U-ARPLS")

_JOB_KEYS = frozenset(JOB_FIELDS) | {"job_id"}

_HASH_FIELDS = (
    "array_sha256",
    "model_spec_sha256",
    "hyperparameter_sha256",
    "fit_uid_sha256",
    "validation_uid_sha256",
    "test_uid_sha256",
)

_IDENTIFIER_FIELDS = (
    "policy_id",
    "representation_id",
    "context_id",
    "model_id",
    "stage",
    "unit_id",
    "candidate_id",
    "resolution",
    "evidence_status",
)

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_input",
        "invalid_job",
        "job_identity_mismatch",
        "job_pair_mismatch",
        "invalid_identifiers",
        "validation_uid_mismatch",
        "class_order_mismatch",
        "invalid_scores",
        "nonfinite_scores",
        "verification_failed",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")


class PredictionError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_input"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise PredictionError(reason_code) from None


def _is_lower_hex64(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _is_identifier(value):
    if type(value) is not str:
        return False
    if not value or value != value.strip() or len(value) > 256:
        return False
    for character in value:
        if ord(character) < 0x20:
            return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _require_bounded_sequence(value, minimum, maximum):
    if type(value) is not list and type(value) is not tuple:
        _fail("invalid_input")
    if len(value) < minimum or len(value) > maximum:
        _fail("invalid_input")
    return list(value)


def _require_unique_labels(items):
    seen = set()
    for item in items:
        if not _is_identifier(item):
            _fail("invalid_identifiers")
        if item in seen:
            _fail("invalid_identifiers")
        seen.add(item)
    return items


def _validate_validation_uids(value):
    items = _require_bounded_sequence(value, 1, 598)
    return _require_unique_labels(items)


def _validate_classes(value):
    items = _require_bounded_sequence(value, 2, 3)
    return _require_unique_labels(items)


def _validate_job(job):
    """Validate structure and return a private bounded snapshot.

    The caller must not concurrently mutate the supplied containers while the
    snapshot is captured.  This is a defensive copy, not an atomic ownership or
    multi-owner authenticity proof.
    """
    if type(job) is not dict:
        _fail("invalid_job")
    if len(job) != len(_JOB_KEYS):
        _fail("invalid_job")
    for key in job:
        if type(key) is not str or key not in _JOB_KEYS:
            _fail("invalid_job")

    stage = job["stage"]
    if type(stage) is not str or stage not in (_FIT_STAGE, _PREDICTION_STAGE):
        _fail("invalid_job")
    expected_dependencies = 0 if stage == _FIT_STAGE else 1

    dependencies = job["dependencies"]
    if type(dependencies) is not list or len(dependencies) != expected_dependencies:
        _fail("job_pair_mismatch")

    snapshot = dict(job)
    snapshot["dependencies"] = list(dependencies)

    for name in _IDENTIFIER_FIELDS:
        if type(snapshot[name]) is not str:
            _fail("invalid_job")
    for name in _HASH_FIELDS:
        if not _is_lower_hex64(snapshot[name]):
            _fail("invalid_job")

    seed = snapshot["seed"]
    if snapshot["model_id"] == _SVM_MODEL:
        if type(seed) is not str:
            _fail("invalid_job")
    else:
        if type(seed) is not int:
            _fail("invalid_job")

    for dependency in snapshot["dependencies"]:
        if not _is_identifier(dependency):
            _fail("invalid_job")

    job_id = snapshot["job_id"]
    if type(job_id) is not str or len(job_id) != 71 or job_id[:7] != "P08JOB-":
        _fail("invalid_job")
    if not _is_lower_hex64(job_id[7:]):
        _fail("invalid_job")

    for name in _IDENTIFIER_FIELDS:
        if not _is_identifier(snapshot[name]):
            _fail("invalid_identifiers")

    return snapshot


def _check_job_hash(job):
    body = {name: job[name] for name in JOB_FIELDS}
    if job["job_id"] != "P08JOB-" + _hash(body):
        _fail("job_identity_mismatch")


def _validate_job_semantics(job):
    policy = job["policy_id"]
    if policy not in _SUPPORTED_POLICIES:
        _fail("invalid_job")
    if job["representation_id"] != POLICY_REPRESENTATION[policy]:
        _fail("invalid_job")
    if job["resolution"] != FIXED_SPEC:
        _fail("invalid_job")
    if job["evidence_status"] != EVIDENCE_FUTURE:
        _fail("invalid_job")

    model_id = job["model_id"]
    neural = model_id in NEURAL_RECIPES
    if not neural and model_id not in CLASSICAL_MODELS:
        _fail("invalid_job")

    seed = job["seed"]
    if model_id == _SVM_MODEL:
        if seed != SVM_SEED:
            _fail("invalid_job")
    else:
        if type(seed) is not int or seed not in SEEDS:
            _fail("invalid_job")

    candidate_id = job["candidate_id"]
    if neural:
        if candidate_id != _NEURAL_CANDIDATE:
            _fail("invalid_job")
        if job["hyperparameter_sha256"] != job["model_spec_sha256"]:
            _fail("invalid_job")
    elif candidate_id == _NEURAL_CANDIDATE:
        _fail("invalid_job")


def _validate_pair(fit_job, prediction_job):
    if fit_job["stage"] != _FIT_STAGE:
        _fail("invalid_job")
    if prediction_job["stage"] != _PREDICTION_STAGE:
        _fail("invalid_job")
    if fit_job["dependencies"] != []:
        _fail("job_pair_mismatch")
    if prediction_job["dependencies"] != [fit_job["job_id"]]:
        _fail("job_pair_mismatch")
    for name in JOB_FIELDS:
        if name in ("stage", "dependencies"):
            continue
        if fit_job[name] != prediction_job[name]:
            _fail("job_pair_mismatch")


def _verify_source_prediction_values(
    scores,
    *,
    observed_uids,
    observed_classes,
    expected_validation_uids,
    expected_classes,
    fit_job,
    prediction_job,
    expected_fit_job_id,
    expected_prediction_job_id,
):
    fit = _validate_job(fit_job)
    prediction = _validate_job(prediction_job)
    _check_job_hash(fit)
    _check_job_hash(prediction)
    _validate_job_semantics(fit)
    _validate_job_semantics(prediction)
    _validate_pair(fit, prediction)

    if type(expected_fit_job_id) is not str or expected_fit_job_id != fit["job_id"]:
        _fail("job_identity_mismatch")
    if (
        type(expected_prediction_job_id) is not str
        or expected_prediction_job_id != prediction["job_id"]
    ):
        _fail("job_identity_mismatch")

    uids = _validate_validation_uids(observed_uids)
    expected_uids = _validate_validation_uids(expected_validation_uids)
    classes = _validate_classes(observed_classes)
    expected_label_classes = _validate_classes(expected_classes)

    if uids != expected_uids:
        _fail("validation_uid_mismatch")
    if classes != expected_label_classes:
        _fail("class_order_mismatch")

    validation_uid_sha256 = sha256_value(sorted(expected_uids))
    if (
        fit["validation_uid_sha256"] != validation_uid_sha256
        or prediction["validation_uid_sha256"] != validation_uid_sha256
    ):
        _fail("validation_uid_mismatch")

    if type(scores) is not np.ndarray:
        _fail("invalid_scores")
    if scores.dtype != np.dtype("float64"):
        _fail("invalid_scores")
    if scores.ndim != 2:
        _fail("invalid_scores")
    if scores.shape != (len(uids), len(classes)):
        _fail("invalid_scores")

    snapshot_scores = scores.copy()
    snapshot_uids = list(uids)
    snapshot_classes = list(classes)

    if type(snapshot_scores) is not np.ndarray:
        _fail("invalid_scores")
    if snapshot_scores.dtype != np.dtype("float64"):
        _fail("invalid_scores")
    if snapshot_scores.ndim != 2:
        _fail("invalid_scores")
    if snapshot_scores.shape != (len(snapshot_uids), len(snapshot_classes)):
        _fail("invalid_scores")
    if not bool(np.isfinite(snapshot_scores).all()):
        _fail("nonfinite_scores")

    ordered_validation_uid_sha256 = sha256_value(snapshot_uids)
    ordered_class_sha256 = sha256_value(snapshot_classes)
    prediction_content_sha256 = sha256_value(
        {
            "classes": snapshot_classes,
            "uids": snapshot_uids,
            "scores": snapshot_scores.tolist(),
        }
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "declared_job_pair_verified": True,
        "source_prediction_structure_verified": True,
        "external_authentication_verified": False,
        "physical_role_isolation_verified": False,
        "training_completion_verified": False,
        "prediction_parity_verified": False,
        "row_count": len(snapshot_uids),
        "class_count": len(snapshot_classes),
        "fit_job_sha256": fit["job_id"][7:],
        "prediction_job_sha256": prediction["job_id"][7:],
        "validation_uid_sha256": validation_uid_sha256,
        "ordered_validation_uid_sha256": ordered_validation_uid_sha256,
        "ordered_class_sha256": ordered_class_sha256,
        "prediction_content_sha256": prediction_content_sha256,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def verify_source_prediction_values(
    scores,
    *,
    observed_uids,
    observed_classes,
    expected_validation_uids,
    expected_classes,
    fit_job,
    prediction_job,
    expected_fit_job_id,
    expected_prediction_job_id,
):
    """Verify an in-memory source-prediction presentation.

    Ordinary failures collapse to a static allowlisted
    :class:`PredictionError`; ``KeyboardInterrupt``, ``SystemExit`` and other
    base exceptions propagate unchanged as the same object.
    """
    try:
        return _verify_source_prediction_values(
            scores,
            observed_uids=observed_uids,
            observed_classes=observed_classes,
            expected_validation_uids=expected_validation_uids,
            expected_classes=expected_classes,
            fit_job=fit_job,
            prediction_job=prediction_job,
            expected_fit_job_id=expected_fit_job_id,
            expected_prediction_job_id=expected_prediction_job_id,
        )
    except PredictionError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution, regardless of forged flags."""
    _fail("scientific_execution_not_authorized")
