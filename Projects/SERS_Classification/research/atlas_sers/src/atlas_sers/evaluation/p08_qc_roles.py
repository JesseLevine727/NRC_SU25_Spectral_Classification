"""P08-T009 no-fit nested QC role registry.

This module turns an *approved* metadata feasibility report into concrete
P08 nested-QC role assignments.  It calls
:func:`atlas_sers.evaluation.p08_qc_support.audit_nested_qc_support` to
validate every concrete membership and to decide eligibility; it never
duplicates or weakens those checks.  A support failure is re-raised as a
:class:`QCRoleError` carrying the same static reason code, so no private
value can escape through an error message.

The module performs no fitting, no predictions, no threshold or policy
selection, no random-number generation, and no file, network, clock or
scientific-library activity.  ``execution_authorized`` is always ``False``
and :func:`require_scientific_execution` always denies execution, even when
handed a forged authorization flag.

For each eligible pseudo-domain context and each registered policy unit it
assigns the unique masters of the parent fit to three nested estimator
folds.  The assignment is a deterministic, classifier-outcome-blind round
robin over a canonical JSON SHA256 ranking of
``{namespace, salt, context_id, parent_unit_id, class_label, master_id}``.
All observations of one master stay in the same fold, and every class is
stratified independently.  These nested folds are master-CV folds inside
the parent fit; they are *not* instrument-transfer folds and they never
alter the original P02/P04 registries.
"""

from __future__ import annotations

import hashlib
import json

from atlas_sers.evaluation.p08_qc_support import (
    QCSupportError,
    audit_nested_qc_support,
)

__all__ = [
    "ALGORITHM",
    "MASTER_NAMESPACE",
    "QCRoleError",
    "REQUIRED_INNER_FOLDS",
    "ROLE_PAIR_NAMESPACE",
    "ROLE_REASON_CODES",
    "SALT",
    "SCHEMA_VERSION",
    "build_nested_qc_roles",
    "require_scientific_execution",
]


SCHEMA_VERSION = "nato-sers-p08-qc-nested-roles-v1"
ALGORITHM = "classwise_canonical_hash_rank_round_robin_3"
SALT = 2026093004
REQUIRED_INNER_FOLDS = 3
MASTER_NAMESPACE = "p08-qc-inner-master-v1"
ROLE_PAIR_NAMESPACE = "p08-qc-inner-role-pair-v1"

ROLE_REASON_CODES = frozenset(
    (
        "scientific_execution_not_authorized",
        "parent_fit_class_masters_below_three",
        "inner_fold_not_partition_parent_fit",
        "inner_fold_fit_validation_uid_overlap",
        "inner_fold_fit_validation_master_overlap",
        "inner_fold_missing_class",
        "inner_fold_insufficient_fit_masters",
    )
)


class QCRoleError(ValueError):
    """Stable error for malformed metadata or attempted execution."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this role-registry module."""
    raise QCRoleError("scientific_execution_not_authorized")


def _fail(code):
    if code not in ROLE_REASON_CODES:
        raise AssertionError(f"unregistered reason code: {code}")
    raise QCRoleError(code)


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _audit(contexts, roles):
    try:
        return audit_nested_qc_support(contexts, roles)
    except QCSupportError as exc:
        raise QCRoleError(exc.reason_code) from None


def _master_rank_digest(context_id, unit_id, class_id, master_id):
    return _sha256(
        {
            "namespace": MASTER_NAMESPACE,
            "salt": SALT,
            "context_id": context_id,
            "parent_unit_id": unit_id,
            "class_label": class_id,
            "master_id": master_id,
        }
    )


def _role_pair_id(context_id, unit_id, fold_index, fit_hash, validation_hash):
    return "P08QCROLE-" + _sha256(
        {
            "namespace": ROLE_PAIR_NAMESPACE,
            "context_id": context_id,
            "parent_unit_id": unit_id,
            "fold_index": fold_index,
            "fit_uid_sha256": fit_hash,
            "validation_uid_sha256": validation_hash,
        }
    )


def _build_policy_unit(context_id, classes, unit, by_uid):
    unit_id = unit["unit_id"]
    parent_fit = sorted(unit["fit_uids"])
    parent_fit_set = set(parent_fit)

    master_of = {}
    master_label = {}
    for uid in parent_fit:
        row = by_uid[uid]
        master = row["master_id"]
        master_of[uid] = master
        master_label[master] = row["label"]

    all_masters = set(master_of.values())
    masters_by_class = {}
    for class_id in classes:
        masters = sorted(
            master for master in all_masters if master_label[master] == class_id
        )
        if len(masters) < REQUIRED_INNER_FOLDS:
            _fail("parent_fit_class_masters_below_three")
        masters_by_class[class_id] = masters

    fold_validation_masters = [set() for _ in range(REQUIRED_INNER_FOLDS)]
    for class_id in classes:
        ranked = sorted(
            masters_by_class[class_id],
            key=lambda master: (
                _master_rank_digest(context_id, unit_id, class_id, master),
                master,
            ),
        )
        for index, master in enumerate(ranked):
            fold_validation_masters[index % REQUIRED_INNER_FOLDS].add(master)

    folds = []
    for fold_index in range(REQUIRED_INNER_FOLDS):
        validation_masters = fold_validation_masters[fold_index]
        fit_masters = all_masters - validation_masters
        fit_uids = sorted(uid for uid in parent_fit if master_of[uid] in fit_masters)
        validation_uids = sorted(
            uid for uid in parent_fit if master_of[uid] in validation_masters
        )

        if set(fit_uids) | set(validation_uids) != parent_fit_set:
            _fail("inner_fold_not_partition_parent_fit")
        if set(fit_uids) & set(validation_uids):
            _fail("inner_fold_fit_validation_uid_overlap")
        if fit_masters & validation_masters:
            _fail("inner_fold_fit_validation_master_overlap")

        fit_class_counts = {}
        validation_class_counts = {}
        for class_id in classes:
            fit_for_class = {
                master_of[uid]
                for uid in fit_uids
                if master_label[master_of[uid]] == class_id
            }
            validation_for_class = {
                master_of[uid]
                for uid in validation_uids
                if master_label[master_of[uid]] == class_id
            }
            if not fit_for_class or not validation_for_class:
                _fail("inner_fold_missing_class")
            if len(fit_for_class) < 2:
                _fail("inner_fold_insufficient_fit_masters")
            fit_class_counts[class_id] = len(fit_for_class)
            validation_class_counts[class_id] = len(validation_for_class)

        fit_uid_sha256 = _sha256(fit_uids)
        validation_uid_sha256 = _sha256(validation_uids)
        folds.append(
            {
                "fold_index": fold_index,
                "role_pair_id": _role_pair_id(
                    context_id,
                    unit_id,
                    fold_index,
                    fit_uid_sha256,
                    validation_uid_sha256,
                ),
                "fit_uids": fit_uids,
                "validation_uids": validation_uids,
                "fit_uid_sha256": fit_uid_sha256,
                "validation_uid_sha256": validation_uid_sha256,
                "fit_masters": sorted(fit_masters),
                "validation_masters": sorted(validation_masters),
                "fit_class_master_counts": fit_class_counts,
                "validation_class_master_counts": validation_class_counts,
                "quantile_fit_uid_sha256": fit_uid_sha256,
            }
        )

    policy_fit_uids = parent_fit
    policy_validation_uids = sorted(unit["validation_uids"])
    policy_fit_hash = _sha256(policy_fit_uids)
    policy_validation_hash = _sha256(policy_validation_uids)
    return {
        "parent_unit_id": unit_id,
        "policy_fit_uids": policy_fit_uids,
        "policy_validation_uids": policy_validation_uids,
        "policy_fit_uid_sha256": policy_fit_hash,
        "policy_validation_uid_sha256": policy_validation_hash,
        "policy_refit_quantile_fit_uid_sha256": policy_fit_hash,
        "inner_folds": folds,
    }


def _summarize(entries):
    context_count = len(entries)
    eligible_contexts = 0
    fallback_contexts = 0
    policy_units = 0
    nested_folds = 0
    role_pair_ids = set()
    minimum_fit = None
    minimum_validation = None

    for entry in entries:
        if not entry["eligible"]:
            fallback_contexts += 1
            continue
        eligible_contexts += 1
        for unit in entry["policy_units"]:
            policy_units += 1
            for fold in unit["inner_folds"]:
                nested_folds += 1
                role_pair_ids.add(fold["role_pair_id"])
                fit_min = min(fold["fit_class_master_counts"].values())
                validation_min = min(fold["validation_class_master_counts"].values())
                if minimum_fit is None or fit_min < minimum_fit:
                    minimum_fit = fit_min
                if minimum_validation is None or validation_min < minimum_validation:
                    minimum_validation = validation_min

    return {
        "context_count": context_count,
        "eligible_contexts": eligible_contexts,
        "fallback_contexts": fallback_contexts,
        "policy_validation_units": policy_units,
        "nested_estimator_folds": nested_folds,
        "unique_role_pair_ids": len(role_pair_ids),
        "minimum_fit_masters_per_class": minimum_fit,
        "minimum_validation_masters_per_class": minimum_validation,
    }


def build_nested_qc_roles(contexts, roles):
    """Build no-fit nested QC estimator folds inside approved policy units.

    All inputs are validated by the P08 support auditor and copied into new
    structures; nothing is mutated.  The returned dictionary is JSON-safe
    and always marked ``execution_authorized`` ``False``.  No scientific
    execution is possible through this module.
    """
    support = _audit(contexts, roles)
    by_uid = {row["observation_uid"]: row for row in roles}
    contexts_by_id = {context["context_id"]: context for context in contexts}

    entries = []
    for detail in support["contexts"]:
        context_id = detail["context_id"]
        raw_context = contexts_by_id[context_id]

        if not detail["eligible"]:
            entries.append(
                {
                    "context_id": context_id,
                    "eligible": False,
                    "reason_code": detail["reason_code"],
                    "outer_fit_uid_sha256": detail["outer_fit_uid_sha256"],
                    "outer_test_uid_sha256": detail["outer_test_uid_sha256"],
                    "final_refit_quantile_fit_uid_sha256": None,
                    "policy_units": [],
                }
            )
            continue

        classes = detail["classes"]
        policy_units = [
            _build_policy_unit(context_id, classes, unit, by_uid)
            for unit in sorted(
                raw_context["selection_units"], key=lambda item: item["unit_id"]
            )
        ]
        entries.append(
            {
                "context_id": context_id,
                "eligible": True,
                "reason_code": detail["reason_code"],
                "outer_fit_uid_sha256": detail["outer_fit_uid_sha256"],
                "outer_test_uid_sha256": detail["outer_test_uid_sha256"],
                "final_refit_quantile_fit_uid_sha256": detail["outer_fit_uid_sha256"],
                "policy_units": policy_units,
            }
        )

    entries.sort(key=lambda entry: entry["context_id"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "support_audit_sha256": support["audit_sha256"],
        "algorithm": ALGORITHM,
        "salt": SALT,
        "contexts": entries,
        "summary": _summarize(entries),
    }
    registry = dict(payload)
    registry["registry_sha256"] = _sha256(payload)
    return registry
