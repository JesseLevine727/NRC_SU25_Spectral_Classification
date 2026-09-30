"""P08-T007 metadata-only nested-QC feasibility auditor.

This module audits the *owner-approved* inner three-fold, class-preserving,
sample-separated model-selection support rule against caller-supplied
metadata records.  It performs no fitting, no fold construction, no
threshold or policy selection, and no file, network or random-number
activity.  It does not grant execution: ``execution_authorized`` is always
``False``.  ``amendment_status`` records the approved support rule; this
module never authenticates an approval from an input flag and is not an
execution permit.

The exported function is :func:`audit_nested_qc_support`.  It validates the
concrete membership of the supplied identifiers (never hash inequality
alone) and reports whether each pseudo-domain context can in principle host
an inner three-fold grouped split with at least three distinct fitting
masters per class in every registered selection unit.

Limitations the caller must keep in mind:

* Authentication of the supplied metadata bytes is entirely the caller's
  responsibility.  This function only proves internal consistency of the
  records it is handed.
* It is a metadata feasibility report, not evidence that nested selection
  would be scientifically useful, and not an execution permit.
* Inner folds group by master, never by pseudo-instrument.  Satisfying the
  per-class master count does not prove instrument-independent inner CV.
* ``require_scientific_execution`` always denies execution, even when handed
  a forged ``execution_authorized`` flag.

Nothing here writes files, draws random numbers, or imports scientific
stacks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = [
    "AMENDMENT_STATUS",
    "ERROR_REASON_CODES",
    "MASTER_MODE",
    "PSEUDO_MODE",
    "QCSupportError",
    "REASON_CODES",
    "REQUIRED_INNER_FOLDS",
    "SCHEMA_VERSION",
    "audit_nested_qc_support",
    "require_scientific_execution",
]


SCHEMA_VERSION = "nato-sers-p08-qc-nested-support-v1"
AMENDMENT_STATUS = "owner_approved_support_rule_no_execution_permit"
REQUIRED_INNER_FOLDS = 3

MASTER_MODE = "master_cv"
PSEUDO_MODE = "pseudo_domain"
SELECTION_MODES = (MASTER_MODE, PSEUDO_MODE)

OUTCOME_SUPPORTED = "nested_three_fold_supported"
OUTCOME_INSUFFICIENT = "insufficient_nested_class_masters"
OUTCOME_NO_PSEUDO = "no_source_pseudo_domains"
REASON_CODES = frozenset((OUTCOME_SUPPORTED, OUTCOME_INSUFFICIENT, OUTCOME_NO_PSEUDO))

CONTEXT_KEYS = (
    "context_id",
    "selection_mode",
    "classes",
    "outer_fit_uids",
    "outer_test_uids",
    "held_instrument",
    "selection_units",
)
UNIT_KEYS = ("unit_id", "fit_uids", "validation_uids")
ROLE_KEYS = ("observation_uid", "master_id", "instrument", "label")

ERROR_REASON_CODES = frozenset(
    (
        "scientific_execution_not_authorized",
        "contexts_must_be_sequence",
        "contexts_empty",
        "context_must_be_mapping",
        "context_keys_invalid",
        "context_id_invalid",
        "duplicate_context_id",
        "selection_mode_invalid",
        "classes_must_be_sequence",
        "classes_empty",
        "class_id_invalid",
        "duplicate_class_id",
        "classes_too_few",
        "outer_fit_uids_must_be_sequence",
        "outer_fit_uids_empty",
        "outer_test_uids_must_be_sequence",
        "outer_test_uids_empty",
        "uid_invalid",
        "duplicate_uid",
        "held_instrument_invalid",
        "selection_units_must_be_sequence",
        "selection_units_count_invalid",
        "unit_must_be_mapping",
        "unit_keys_invalid",
        "unit_id_invalid",
        "duplicate_unit_id",
        "fit_uids_must_be_sequence",
        "fit_uids_empty",
        "validation_uids_must_be_sequence",
        "validation_uids_empty",
        "roles_must_be_sequence",
        "role_must_be_mapping",
        "role_keys_invalid",
        "observation_uid_invalid",
        "duplicate_observation_uid",
        "role_field_invalid",
        "unknown_observation_uid",
        "ambiguous_master_label",
        "label_not_in_context_classes",
        "fit_test_uid_overlap",
        "fit_test_master_overlap",
        "held_instrument_in_outer_fit",
        "outer_test_instrument_mismatch",
        "outer_fit_missing_classes",
        "selection_fit_not_subset_outer_fit",
        "selection_validation_not_subset_outer_fit",
        "selection_fit_validation_uid_overlap",
        "selection_fit_validation_master_overlap",
        "selection_fit_missing_classes",
        "selection_validation_missing_classes",
        "master_folds_not_partition",
        "master_fit_not_outer_minus_fold",
        "pseudo_validation_multiple_instruments",
        "pseudo_validation_instrument_in_selection_fit",
        "pseudo_validation_instruments_not_distinct",
    )
)


class QCSupportError(ValueError):
    """Stable error for malformed metadata or attempted execution."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this readiness module."""
    raise QCSupportError("scientific_execution_not_authorized")


def _fail(code):
    if code not in ERROR_REASON_CODES:
        raise AssertionError(f"unregistered reason code: {code}")
    raise QCSupportError(code)


def _hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value, code):
    if not isinstance(value, Mapping):
        _fail(code)


def _require_sequence(value, code):
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail(code)


def _require_identifier(value, code):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)


def _require_exact_keys(mapping, expected, code):
    if set(mapping.keys()) != set(expected):
        _fail(code)


def _parse_string_collection(value, sequence_code, empty_code, item_code, duplicate_code):
    _require_sequence(value, sequence_code)
    if len(value) == 0:
        _fail(empty_code)
    seen = set()
    for item in value:
        _require_identifier(item, item_code)
        if item in seen:
            _fail(duplicate_code)
        seen.add(item)
    return frozenset(value)


def _parse_selection_units(raw_units, mode):
    _require_sequence(raw_units, "selection_units_must_be_sequence")
    if mode == MASTER_MODE:
        if len(raw_units) != 3:
            _fail("selection_units_count_invalid")
    else:
        if len(raw_units) < 2:
            _fail("selection_units_count_invalid")
    seen = set()
    units = []
    for raw in raw_units:
        _require_mapping(raw, "unit_must_be_mapping")
        _require_exact_keys(raw, UNIT_KEYS, "unit_keys_invalid")
        unit_id = raw["unit_id"]
        _require_identifier(unit_id, "unit_id_invalid")
        if unit_id in seen:
            _fail("duplicate_unit_id")
        seen.add(unit_id)
        fit_uids = _parse_string_collection(
            raw["fit_uids"],
            "fit_uids_must_be_sequence",
            "fit_uids_empty",
            "uid_invalid",
            "duplicate_uid",
        )
        validation_uids = _parse_string_collection(
            raw["validation_uids"],
            "validation_uids_must_be_sequence",
            "validation_uids_empty",
            "uid_invalid",
            "duplicate_uid",
        )
        units.append(
            {
                "unit_id": unit_id,
                "fit_uids": fit_uids,
                "validation_uids": validation_uids,
            }
        )
    units.sort(key=lambda unit: unit["unit_id"])
    return units


def _parse_contexts(raw_contexts):
    _require_sequence(raw_contexts, "contexts_must_be_sequence")
    if len(raw_contexts) == 0:
        _fail("contexts_empty")
    seen = set()
    parsed = []
    for raw in raw_contexts:
        _require_mapping(raw, "context_must_be_mapping")
        _require_exact_keys(raw, CONTEXT_KEYS, "context_keys_invalid")
        context_id = raw["context_id"]
        _require_identifier(context_id, "context_id_invalid")
        if context_id in seen:
            _fail("duplicate_context_id")
        seen.add(context_id)

        mode = raw["selection_mode"]
        _require_identifier(mode, "selection_mode_invalid")
        if mode not in SELECTION_MODES:
            _fail("selection_mode_invalid")

        classes = _parse_string_collection(
            raw["classes"],
            "classes_must_be_sequence",
            "classes_empty",
            "class_id_invalid",
            "duplicate_class_id",
        )
        if len(classes) < 2:
            _fail("classes_too_few")

        outer_fit_uids = _parse_string_collection(
            raw["outer_fit_uids"],
            "outer_fit_uids_must_be_sequence",
            "outer_fit_uids_empty",
            "uid_invalid",
            "duplicate_uid",
        )
        outer_test_uids = _parse_string_collection(
            raw["outer_test_uids"],
            "outer_test_uids_must_be_sequence",
            "outer_test_uids_empty",
            "uid_invalid",
            "duplicate_uid",
        )

        held_instrument = raw["held_instrument"]
        _require_identifier(held_instrument, "held_instrument_invalid")

        units = _parse_selection_units(raw["selection_units"], mode)

        parsed.append(
            {
                "context_id": context_id,
                "selection_mode": mode,
                "classes": classes,
                "outer_fit_uids": outer_fit_uids,
                "outer_test_uids": outer_test_uids,
                "held_instrument": held_instrument,
                "selection_units": units,
            }
        )
    parsed.sort(key=lambda context: context["context_id"])
    return parsed


def _parse_roles(raw_roles):
    _require_sequence(raw_roles, "roles_must_be_sequence")
    by_uid = {}
    for raw in raw_roles:
        _require_mapping(raw, "role_must_be_mapping")
        _require_exact_keys(raw, ROLE_KEYS, "role_keys_invalid")
        observation_uid = raw["observation_uid"]
        _require_identifier(observation_uid, "observation_uid_invalid")
        if observation_uid in by_uid:
            _fail("duplicate_observation_uid")
        for field in ("master_id", "instrument", "label"):
            _require_identifier(raw[field], "role_field_invalid")
        by_uid[observation_uid] = {
            "observation_uid": observation_uid,
            "master_id": raw["master_id"],
            "instrument": raw["instrument"],
            "label": raw["label"],
        }
    return by_uid


def _check_master_labels(by_uid):
    labels = {}
    for row in by_uid.values():
        labels.setdefault(row["master_id"], set()).add(row["label"])
    for master_id in sorted(labels):
        if len(labels[master_id]) != 1:
            _fail("ambiguous_master_label")


def _rows_for(uids, by_uid):
    rows = []
    for uid in sorted(uids):
        row = by_uid.get(uid)
        if row is None:
            _fail("unknown_observation_uid")
        rows.append(row)
    return rows


def _audit_context(ctx, by_uid):
    context_id = ctx["context_id"]
    mode = ctx["selection_mode"]
    classes = ctx["classes"]
    held_instrument = ctx["held_instrument"]
    outer_fit = ctx["outer_fit_uids"]
    outer_test = ctx["outer_test_uids"]

    outer_fit_rows = _rows_for(outer_fit, by_uid)
    outer_test_rows = _rows_for(outer_test, by_uid)

    unit_rows = []
    for unit in ctx["selection_units"]:
        fit_rows = _rows_for(unit["fit_uids"], by_uid)
        validation_rows = _rows_for(unit["validation_uids"], by_uid)
        unit_rows.append((unit, fit_rows, validation_rows))

    referenced_labels = {row["label"] for row in outer_fit_rows}
    referenced_labels.update(row["label"] for row in outer_test_rows)
    for _, fit_rows, validation_rows in unit_rows:
        referenced_labels.update(row["label"] for row in fit_rows)
        referenced_labels.update(row["label"] for row in validation_rows)
    if not referenced_labels <= classes:
        _fail("label_not_in_context_classes")

    if outer_fit & outer_test:
        _fail("fit_test_uid_overlap")
    outer_fit_masters = {row["master_id"] for row in outer_fit_rows}
    outer_test_masters = {row["master_id"] for row in outer_test_rows}
    if outer_fit_masters & outer_test_masters:
        _fail("fit_test_master_overlap")

    if any(row["instrument"] == held_instrument for row in outer_fit_rows):
        _fail("held_instrument_in_outer_fit")
    if any(row["instrument"] != held_instrument for row in outer_test_rows):
        _fail("outer_test_instrument_mismatch")

    if not classes <= {row["label"] for row in outer_fit_rows}:
        _fail("outer_fit_missing_classes")

    for unit, fit_rows, validation_rows in unit_rows:
        fit_uids = unit["fit_uids"]
        validation_uids = unit["validation_uids"]
        if not fit_uids <= outer_fit:
            _fail("selection_fit_not_subset_outer_fit")
        if not validation_uids <= outer_fit:
            _fail("selection_validation_not_subset_outer_fit")
        if fit_uids & validation_uids:
            _fail("selection_fit_validation_uid_overlap")
        fit_masters = {row["master_id"] for row in fit_rows}
        validation_masters = {row["master_id"] for row in validation_rows}
        if fit_masters & validation_masters:
            _fail("selection_fit_validation_master_overlap")
        if not classes <= {row["label"] for row in fit_rows}:
            _fail("selection_fit_missing_classes")
        if not classes <= {row["label"] for row in validation_rows}:
            _fail("selection_validation_missing_classes")

    if mode == MASTER_MODE:
        folds = [unit["validation_uids"] for unit in ctx["selection_units"]]
        for left in range(len(folds)):
            for right in range(left + 1, len(folds)):
                if folds[left] & folds[right]:
                    _fail("master_folds_not_partition")
        union = set()
        for fold in folds:
            union |= fold
        if union != outer_fit:
            _fail("master_folds_not_partition")
        for unit in ctx["selection_units"]:
            if unit["fit_uids"] != outer_fit - unit["validation_uids"]:
                _fail("master_fit_not_outer_minus_fold")
    else:
        validation_instruments = []
        for _, fit_rows, validation_rows in unit_rows:
            instruments = {row["instrument"] for row in validation_rows}
            if len(instruments) != 1:
                _fail("pseudo_validation_multiple_instruments")
            instrument = next(iter(instruments))
            if any(row["instrument"] == instrument for row in fit_rows):
                _fail("pseudo_validation_instrument_in_selection_fit")
            validation_instruments.append(instrument)
        if len(set(validation_instruments)) != len(validation_instruments):
            _fail("pseudo_validation_instruments_not_distinct")

    context_eligible = mode == PSEUDO_MODE
    unit_details = []
    for unit, fit_rows, _validation_rows in unit_rows:
        class_master_counts = {}
        minimum = None
        for class_id in sorted(classes):
            masters = {
                row["master_id"] for row in fit_rows if row["label"] == class_id
            }
            class_master_counts[class_id] = len(masters)
            if minimum is None or len(masters) < minimum:
                minimum = len(masters)
        meets = minimum >= REQUIRED_INNER_FOLDS
        if not meets:
            context_eligible = False
        unit_details.append(
            {
                "unit_id": unit["unit_id"],
                "class_master_counts": class_master_counts,
                "min_class_masters": minimum,
                "meets_inner_three_fold": meets,
                "fit_uid_sha256": _hash(sorted(unit["fit_uids"])),
                "validation_uid_sha256": _hash(sorted(unit["validation_uids"])),
            }
        )

    if mode == MASTER_MODE:
        reason_code = OUTCOME_NO_PSEUDO
        eligible = False
    elif context_eligible:
        reason_code = OUTCOME_SUPPORTED
        eligible = True
    else:
        reason_code = OUTCOME_INSUFFICIENT
        eligible = False

    return {
        "context_id": context_id,
        "selection_mode": mode,
        "classes": sorted(classes),
        "reason_code": reason_code,
        "eligible": eligible,
        "outer_fit_uid_sha256": _hash(sorted(outer_fit)),
        "outer_test_uid_sha256": _hash(sorted(outer_test)),
        "units": unit_details,
    }


def _build_summary(details):
    total_contexts = len(details)
    pseudo_contexts = 0
    mastercv_contexts = 0
    eligible_contexts = 0
    reason_counts = {
        OUTCOME_SUPPORTED: 0,
        OUTCOME_INSUFFICIENT: 0,
        OUTCOME_NO_PSEUDO: 0,
    }
    total_pseudo_units = 0
    individually_supported_pseudo_units = 0
    eligible_pseudo_units = 0
    histogram = {}

    for detail in details:
        reason_counts[detail["reason_code"]] += 1
        if detail["eligible"]:
            eligible_contexts += 1
        if detail["selection_mode"] == PSEUDO_MODE:
            pseudo_contexts += 1
            for unit in detail["units"]:
                total_pseudo_units += 1
                key = str(unit["min_class_masters"])
                histogram[key] = histogram.get(key, 0) + 1
                if unit["meets_inner_three_fold"]:
                    individually_supported_pseudo_units += 1
                    if detail["eligible"]:
                        eligible_pseudo_units += 1
        else:
            mastercv_contexts += 1

    ordered_histogram = {
        key: histogram[key] for key in sorted(histogram, key=lambda value: int(value))
    }
    return {
        "total_contexts": total_contexts,
        "pseudo_contexts": pseudo_contexts,
        "mastercv_contexts": mastercv_contexts,
        "eligible_contexts": eligible_contexts,
        "fallback_contexts": total_contexts - eligible_contexts,
        "reason_counts": reason_counts,
        "total_pseudo_units": total_pseudo_units,
        "individually_supported_pseudo_units": individually_supported_pseudo_units,
        "pseudo_unit_min_masters_histogram": ordered_histogram,
        "eligible_pseudo_units": eligible_pseudo_units,
    }


def audit_nested_qc_support(contexts, roles):
    """Audit metadata-only support for the proposed nested inner folds.

    All inputs are validated and copied; nothing is mutated.  The returned
    dictionary is JSON-safe and always marked ``execution_authorized``
    ``False``.  No scientific execution is possible through this module.
    """
    parsed_contexts = _parse_contexts(contexts)
    by_uid = _parse_roles(roles)
    _check_master_labels(by_uid)

    details = [_audit_context(ctx, by_uid) for ctx in parsed_contexts]
    details.sort(key=lambda detail: detail["context_id"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "amendment_status": AMENDMENT_STATUS,
        "required_inner_folds": REQUIRED_INNER_FOLDS,
        "contexts": details,
        "summary": _build_summary(details),
    }
    audit = dict(payload)
    audit["audit_sha256"] = _hash(payload)
    return audit
