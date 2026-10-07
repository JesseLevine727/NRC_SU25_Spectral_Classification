"""P08-T235 stress score-support metadata adapter (metadata only).

This module is an INTERNAL metadata adapter.  It consumes an authenticated P08
stress prediction catalog (``p08_perturbation_predictions``) together with
caller-authenticated per-context test memberships and derives a deterministic,
execution-disabled description of the per-context and pooled stress
score-support graph.

It performs **zero** scientific work: no scores, predictions, fits, thresholds,
routes, draws, quantiles, statistics, rendering or numerical parity checks are
computed, guessed or inferred.  Every dynamic outcome stays a deferred
downstream dependency.  The full stress job ledger is explicitly *not*
complete here, no scores are claimed and no new authority is granted.

The parent prediction catalog is validated through
``p08_perturbation_predictions.iter_stress_prediction_records``; the returned
lazy parent iterator is intentionally never consumed.  The returned root binds
the parent catalog and the supplied memberships by canonical digest and always
keeps every execution/verification/score flag ``False``.

Recorded master IDs keep their exact input type.  A master ID must be either an
exact ``int`` (never ``bool``) or a nonempty stripped ``str``; every row must
share one concrete master-ID type, and integers are never stringified.

Malformed metadata is sanitised to ``invalid_stress_score_support``.  Only
``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/``RecursionError``/
``OverflowError`` are caught, never ``BaseException`` or a bare ``Exception``.
"""

from __future__ import annotations

import json

from .p08_perturbation_join import FAMILY_POLICY, QC_POLICY
from .p08_perturbation_predictions import iter_stress_prediction_records
from .p08_plan import (
    CLASSICAL_MODELS,
    D0_STRATEGY,
    EVIDENCE_BY_POLICY,
    EVIDENCE_HISTORICAL,
    NOT_APPLICABLE,
    POLICIES,
    SELECTED_STRATEGY,
)
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "build_stress_score_support",
    "require_scientific_execution",
    "validate_stress_score_support",
]


SCHEMA_VERSION = "nato-sers-p08-stress-score-support-v1"
INVALID = "invalid_stress_score_support"
EXECUTION_DENIED = "scientific_execution_not_authorized"

SUPPORT_OPERATIONAL = "operational_260"
SUPPORT_QC_ELIGIBLE = "qc_eligible_54"

MODE_UNIVERSAL = "universal"
MODE_QC_FIXED_ROUTE = "qc_fixed_route"
MODE_QC_MINIMAL_FALLBACK = "qc_minimal_fallback"
MODE_FAMILY_MINIMAL_FALLBACK = "family_minimal_fallback"

_QC_SOURCE_ELIGIBLE = "fixed_clean_route"
_QC_SOURCE_FALLBACK = "disturbed_minimal_pipeline"

_ENDPOINTS = ("M01", "M06")
_FOLDS = (0, 1, 2, 3)

_UNIVERSAL_STRATEGIES = tuple(CLASSICAL_MODELS) + (D0_STRATEGY, SELECTED_STRATEGY)

_MIN_POLICY = next(
    policy for policy in POLICIES if EVIDENCE_BY_POLICY[policy] == EVIDENCE_HISTORICAL
)

_NA = NOT_APPLICABLE

_ROW_FIELDS = frozenset(
    {
        "context_id",
        "domain",
        "station",
        "instrument",
        "outer_repeat",
        "outer_fold",
        "observation_uid",
        "master_id",
        "label",
    }
)

_MEMBERSHIP_FIELDS = frozenset({"operational_contexts", "qc_eligible_contexts", "test_rows"})

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "scientific_scores_computed",
        "full_stress_job_ledger_complete",
        "prediction_catalog",
        "memberships",
        "context_records",
        "pool_groups",
        "context_views",
        "pooled_procedures",
        "pooled_views",
        "global_master_ids",
        "global_instrument_ids",
        "supports",
        "summary",
        "catalog_sha256",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this metadata-only adapter."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID)


def _require_mapping(value):
    if not isinstance(value, dict):
        _fail()
    return value


def _require_sequence(value):
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail()
    return list(value)


def _require_identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail()
    return value


def _require_master_id(value):
    """Return a master ID preserving its exact recorded value and type.

    Only an exact ``int`` (``bool`` excluded) or a nonempty already-stripped
    ``str`` is accepted.  The concrete input value/type is returned unchanged
    so recorded integer IDs are never stringified and ``1`` cannot be
    conflated with ``"1"``.
    """
    if type(value) is int:
        return value
    if isinstance(value, str) and value and value == value.strip():
        return value
    _fail()


def _require_uniform_master_id_type(master_ids):
    """Require every master ID to share one concrete type before sorting.

    Mixed ``int``/``str`` master IDs would both raise on ordering and risk
    conflating distinct IDs, so they are rejected here, ahead of any sorting or
    hashing of master IDs.
    """
    kinds = {type(master_id) for master_id in master_ids}
    if len(kinds) > 1:
        _fail()


def _require_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        _fail()
    for character in value:
        if character not in _HEX_DIGITS:
            _fail()
    return value


def _require_int_in_bounds(value, low, high):
    if type(value) is not int or value < low or value > high:
        _fail()
    return value


def _snapshot(value):
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _normalize_context_list(value):
    if not isinstance(value, list):
        _fail()
    seen = set()
    for item in value:
        _require_identifier(item)
        if item in seen:
            _fail()
        seen.add(item)
    return sorted(seen)


def _normalize_rows(rows, input_contexts):
    if not isinstance(rows, list) or not rows:
        _fail()
    seen = set()
    normalized = []
    for row in rows:
        _require_mapping(row)
        if set(row.keys()) != _ROW_FIELDS:
            _fail()
        context_id = _require_identifier(row["context_id"])
        if context_id not in input_contexts:
            _fail()
        domain = _require_identifier(row["domain"])
        station = _require_identifier(row["station"])
        instrument = _require_identifier(row["instrument"])
        observation_uid = _require_identifier(row["observation_uid"])
        master_id = _require_master_id(row["master_id"])
        label = _require_identifier(row["label"])
        outer_repeat = _require_int_in_bounds(row["outer_repeat"], 1, 5)
        outer_fold = _require_int_in_bounds(row["outer_fold"], 0, 3)
        key = (context_id, observation_uid)
        if key in seen:
            _fail()
        seen.add(key)
        normalized.append(
            {
                "context_id": context_id,
                "domain": domain,
                "station": station,
                "instrument": instrument,
                "outer_repeat": outer_repeat,
                "outer_fold": outer_fold,
                "observation_uid": observation_uid,
                "master_id": master_id,
                "label": label,
            }
        )
    _require_uniform_master_id_type(entry["master_id"] for entry in normalized)
    normalized.sort(key=lambda entry: (entry["context_id"], entry["observation_uid"]))
    return normalized


def _normalize_memberships(memberships, input_contexts, qc_eligible_set):
    mapping = _require_mapping(memberships)
    if set(mapping.keys()) != _MEMBERSHIP_FIELDS:
        _fail()
    operational = _normalize_context_list(mapping["operational_contexts"])
    eligible = _normalize_context_list(mapping["qc_eligible_contexts"])
    if set(operational) != set(input_contexts):
        _fail()
    if set(eligible) != qc_eligible_set:
        _fail()
    if not set(eligible) <= set(operational):
        _fail()
    rows = _normalize_rows(mapping["test_rows"], input_contexts)
    return {
        "operational_contexts": operational,
        "qc_eligible_contexts": eligible,
        "test_rows": rows,
    }


def _build_context_records(rows, input_contexts):
    by_context = {}
    for row in rows:
        by_context.setdefault(row["context_id"], []).append(row)

    records = {}
    for context_id in sorted(input_contexts):
        context_rows = by_context.get(context_id)
        if not context_rows:
            _fail()
        info = input_contexts[context_id]
        uids = sorted(row["observation_uid"] for row in context_rows)
        if uids != info["test_uids"]:
            _fail()
        if canonical_sha256(uids) != info["test_uid_sha256"]:
            _fail()

        first = context_rows[0]
        for row in context_rows:
            if (
                row["domain"] != first["domain"]
                or row["station"] != first["station"]
                or row["instrument"] != first["instrument"]
                or row["outer_repeat"] != first["outer_repeat"]
                or row["outer_fold"] != first["outer_fold"]
            ):
                _fail()

        classes = sorted({row["label"] for row in context_rows})
        class_spectrum_counts = {}
        masters_by_label = {}
        units = {}
        for row in context_rows:
            label = row["label"]
            class_spectrum_counts[label] = class_spectrum_counts.get(label, 0) + 1
            masters_by_label.setdefault(label, set()).add(row["master_id"])
            entry = units.setdefault(
                row["master_id"],
                {"label": row["label"], "observation_uids": []},
            )
            if entry["label"] != label:
                _fail()
            entry["observation_uids"].append(row["observation_uid"])

        class_master_counts = {label: len(masters) for label, masters in masters_by_label.items()}
        master_units = [
            {
                "master_id": master_id,
                "label": units[master_id]["label"],
                "observation_uids": sorted(units[master_id]["observation_uids"]),
            }
            for master_id in sorted(units)
        ]

        body = {
            "context_id": context_id,
            "domain": first["domain"],
            "station": first["station"],
            "instrument": first["instrument"],
            "outer_repeat": first["outer_repeat"],
            "outer_fold": first["outer_fold"],
            "test_uid_sha256": info["test_uid_sha256"],
            "test_uids": info["test_uids"],
            "classes": classes,
            "class_spectrum_counts": class_spectrum_counts,
            "class_master_counts": class_master_counts,
            "master_units": master_units,
        }
        record = dict(body)
        record["context_metadata_sha256"] = canonical_sha256(body)
        records[context_id] = record
    return records


def _validate_global_relations(records, rows):
    domain_to_station_instrument = {}
    station_instrument_to_domain = {}
    domain_repeat_fold = {}
    uid_values = {}
    master_labels = {}
    domain_repeat_uid_fold = {}
    domain_repeat_master_fold = {}

    for context_id in sorted(records):
        record = records[context_id]
        domain = record["domain"]
        station_instrument = (record["station"], record["instrument"])
        existing = domain_to_station_instrument.get(domain)
        if existing is None:
            domain_to_station_instrument[domain] = station_instrument
        elif existing != station_instrument:
            _fail()
        existing_domain = station_instrument_to_domain.get(station_instrument)
        if existing_domain is None:
            station_instrument_to_domain[station_instrument] = domain
        elif existing_domain != domain:
            _fail()
        fold_key = (domain, record["outer_repeat"], record["outer_fold"])
        if fold_key in domain_repeat_fold:
            _fail()
        domain_repeat_fold[fold_key] = context_id

    for row in rows:
        observation_uid = row["observation_uid"]
        value = (row["master_id"], row["label"], row["instrument"], row["station"])
        existing_value = uid_values.get(observation_uid)
        if existing_value is None:
            uid_values[observation_uid] = value
        elif existing_value != value:
            _fail()

        master_id = row["master_id"]
        label = row["label"]
        existing_label = master_labels.get(master_id)
        if existing_label is None:
            master_labels[master_id] = label
        elif existing_label != label:
            _fail()

        domain_repeat = (row["domain"], row["outer_repeat"])
        outer_fold = row["outer_fold"]
        uid_key = (domain_repeat, observation_uid)
        existing_uid_fold = domain_repeat_uid_fold.get(uid_key)
        if existing_uid_fold is None:
            domain_repeat_uid_fold[uid_key] = outer_fold
        elif existing_uid_fold != outer_fold:
            _fail()
        master_key = (domain_repeat, master_id)
        existing_master_fold = domain_repeat_master_fold.get(master_key)
        if existing_master_fold is None:
            domain_repeat_master_fold[master_key] = outer_fold
        elif existing_master_fold != outer_fold:
            _fail()


def _make_view(
    policy_id,
    context_id,
    strategy,
    recipe_id,
    target_procedure_id,
    mode,
    upstream_alias_id,
):
    body = {
        "policy_id": policy_id,
        "context_id": context_id,
        "strategy": strategy,
        "recipe_id": recipe_id,
        "target_procedure_id": target_procedure_id,
        "mode": mode,
        "upstream_alias_id": upstream_alias_id,
    }
    record = dict(body)
    record["view_id"] = "P08STRESSVIEW-" + canonical_sha256(body)
    return record


def _context_views(bundle, procedure_index, operational_contexts, operational_set):
    views = []

    universal_records = _require_sequence(bundle["universal_procedures"]["records"])
    universal_policies = sorted(
        {_require_identifier(record["policy_id"]) for record in universal_records}
    )
    universal_contexts = {_require_identifier(record["context_id"]) for record in universal_records}

    universal_aliases = {}
    for alias in _require_sequence(bundle["universal_procedures"]["strategy_aliases"]):
        _require_mapping(alias)
        policy_id = _require_identifier(alias["policy_id"])
        context_id = _require_identifier(alias["context_id"])
        strategy = _require_identifier(alias["strategy"])
        key = (policy_id, context_id, strategy)
        if key in universal_aliases:
            _fail()
        universal_aliases[key] = alias

    for context_id in operational_contexts:
        if context_id not in universal_contexts:
            _fail()
        for policy_id in universal_policies:
            for strategy in _UNIVERSAL_STRATEGIES:
                if strategy in CLASSICAL_MODELS:
                    target = procedure_index.get((policy_id, context_id, strategy))
                    if target is None:
                        _fail()
                    recipe_id = strategy
                    upstream_alias_id = _NA
                else:
                    alias = universal_aliases.get((policy_id, context_id, strategy))
                    if alias is None:
                        _fail()
                    target = _require_identifier(alias["target_procedure_id"])
                    recipe_id = _require_identifier(alias["recipe_id"])
                    upstream_alias_id = _require_identifier(alias["upstream_alias_id"])
                views.append(
                    _make_view(
                        policy_id,
                        context_id,
                        strategy,
                        recipe_id,
                        target,
                        MODE_UNIVERSAL,
                        upstream_alias_id,
                    )
                )

    for alias in _require_sequence(bundle["qc_procedures"]["strategy_aliases"]):
        _require_mapping(alias)
        context_id = _require_identifier(alias["context_id"])
        if context_id not in operational_set:
            _fail()
        source_mode = alias["mode"]
        if source_mode == _QC_SOURCE_ELIGIBLE:
            mode = MODE_QC_FIXED_ROUTE
        elif source_mode == _QC_SOURCE_FALLBACK:
            mode = MODE_QC_MINIMAL_FALLBACK
        else:
            _fail()
        views.append(
            _make_view(
                _require_identifier(alias["policy_id"]),
                context_id,
                _require_identifier(alias["strategy"]),
                _require_identifier(alias["recipe_id"]),
                _require_identifier(alias["target_procedure_id"]),
                mode,
                _require_identifier(alias["upstream_alias_id"]),
            )
        )

    min_index = {
        (context_id, model_id): procedure_index[(_MIN_POLICY, context_id, model_id)]
        for (policy_id, context_id, model_id) in procedure_index
        if policy_id == _MIN_POLICY
    }
    for alias in _require_sequence(bundle["family_aliases"]):
        _require_mapping(alias)
        context_id = _require_identifier(alias["context_id"])
        if context_id not in operational_set:
            _fail()
        recipe_id = _require_identifier(alias["recipe_id"])
        target = min_index.get((context_id, recipe_id))
        if target is None:
            _fail()
        views.append(
            _make_view(
                FAMILY_POLICY,
                context_id,
                _require_identifier(alias["strategy"]),
                recipe_id,
                target,
                MODE_FAMILY_MINIMAL_FALLBACK,
                _require_identifier(alias["alias_id"]),
            )
        )

    views.sort(key=lambda view: (view["policy_id"], view["context_id"], view["strategy"]))
    return views


def _support_summary(support_contexts, support_groups, records):
    domains = set()
    instruments = set()
    stations = set()
    distinct_uids = set()
    distinct_masters = set()
    appearances = 0
    master_context_units = 0
    for context_id in support_contexts:
        record = records[context_id]
        domains.add(record["domain"])
        instruments.add(record["instrument"])
        stations.add(record["station"])
        distinct_uids.update(record["test_uids"])
        appearances += len(record["test_uids"])
        masters = {unit["master_id"] for unit in record["master_units"]}
        distinct_masters.update(masters)
        master_context_units += len(masters)

    complete_groups = [group for group in support_groups if group["complete_four_fold"]]
    complete_contexts = 0
    complete_spectra = set()
    complete_masters = set()
    complete_appearances = 0
    for group in complete_groups:
        complete_contexts += len(group["context_ids"])
        for context_id in group["context_ids"]:
            record = records[context_id]
            complete_spectra.update(record["test_uids"])
            complete_appearances += len(record["test_uids"])
            complete_masters.update(unit["master_id"] for unit in record["master_units"])

    return {
        "contexts": len(support_contexts),
        "domains": len(domains),
        "instruments": len(instruments),
        "stations": sorted(stations),
        "distinct_test_spectra": len(distinct_uids),
        "distinct_test_masters": len(distinct_masters),
        "test_spectrum_appearances": appearances,
        "master_context_prediction_units": master_context_units,
        "complete_four_fold_domain_repeat_groups": len(complete_groups),
        "complete_four_fold_contexts": complete_contexts,
        "complete_four_fold_distinct_spectra": len(complete_spectra),
        "complete_four_fold_distinct_masters": len(complete_masters),
        "complete_four_fold_test_appearances": complete_appearances,
    }


def _build(parent, memberships):
    bundle = _require_mapping(parent["bundle"])
    input_catalog = _require_mapping(bundle["input_catalog"])
    contexts_raw = _require_sequence(input_catalog["contexts"])
    case_manifest = _require_mapping(input_catalog["case_manifest"])
    cases = _require_sequence(case_manifest["cases"])
    family_cases = _require_mapping(case_manifest["family_cases"])
    procedures = _require_sequence(bundle["procedures"])

    input_contexts = {}
    for context in contexts_raw:
        _require_mapping(context)
        context_id = _require_identifier(context["context_id"])
        if context_id in input_contexts:
            _fail()
        test_uids = sorted(
            _require_identifier(uid) for uid in _require_sequence(context["test_uids"])
        )
        if len(set(test_uids)) != len(test_uids):
            _fail()
        test_uid_sha256 = _require_hex64(context["test_uid_sha256"])
        input_contexts[context_id] = {
            "test_uids": test_uids,
            "test_uid_sha256": test_uid_sha256,
        }
    if not input_contexts:
        _fail()
    operational_contexts = sorted(input_contexts)
    operational_set = set(operational_contexts)

    procedure_index = {}
    for procedure in procedures:
        _require_mapping(procedure)
        policy_id = _require_identifier(procedure["policy_id"])
        context_id = _require_identifier(procedure["context_id"])
        model_id = _require_identifier(procedure["model_id"])
        procedure_id = _require_identifier(procedure["procedure_id"])
        key = (policy_id, context_id, model_id)
        if key in procedure_index:
            _fail()
        procedure_index[key] = procedure_id

    qc_records = _require_sequence(bundle["qc_procedures"]["records"])
    qc_eligible_set = {_require_identifier(record["context_id"]) for record in qc_records}

    normalized_memberships = _normalize_memberships(memberships, input_contexts, qc_eligible_set)
    rows = normalized_memberships["test_rows"]
    eligible_contexts = normalized_memberships["qc_eligible_contexts"]

    records = _build_context_records(rows, input_contexts)
    _validate_global_relations(records, rows)

    context_views = _context_views(bundle, procedure_index, operational_contexts, operational_set)
    view_target = {
        (view["context_id"], view["policy_id"], view["strategy"]): view["target_procedure_id"]
        for view in context_views
    }

    qc_aliases = _require_sequence(bundle["qc_procedures"]["strategy_aliases"])
    qc_strategy_set = sorted({_require_identifier(alias["strategy"]) for alias in qc_aliases})
    all_pairs = sorted({(view["policy_id"], view["strategy"]) for view in context_views})
    eligible_pairs = [(_MIN_POLICY, strategy) for strategy in qc_strategy_set] + [
        (QC_POLICY, strategy) for strategy in qc_strategy_set
    ]

    pool_groups = []
    pooled_procedure_by_id = {}
    pooled_views = []
    supports = {}

    for support_id, support_contexts, pairs in (
        (SUPPORT_OPERATIONAL, operational_contexts, all_pairs),
        (SUPPORT_QC_ELIGIBLE, eligible_contexts, eligible_pairs),
    ):
        grouped = {}
        for context_id in support_contexts:
            group_key = (
                records[context_id]["domain"],
                records[context_id]["outer_repeat"],
            )
            grouped.setdefault(group_key, []).append(context_id)

        support_groups = []
        complete_group_ids = []
        for group_key in sorted(grouped):
            domain, outer_repeat = group_key
            context_ids = sorted(grouped[group_key])
            folds = sorted({records[context_id]["outer_fold"] for context_id in context_ids})
            complete = len(context_ids) == 4 and folds == list(_FOLDS)
            station = records[context_ids[0]]["station"]
            instrument = records[context_ids[0]]["instrument"]

            uid_union = set()
            masters = set()
            for context_id in context_ids:
                uid_union.update(records[context_id]["test_uids"])
                masters.update(unit["master_id"] for unit in records[context_id]["master_units"])
            test_uids = sorted(uid_union)

            group_body = {
                "support_id": support_id,
                "domain": domain,
                "station": station,
                "instrument": instrument,
                "outer_repeat": outer_repeat,
                "folds": folds,
                "context_ids": context_ids,
                "complete_four_fold": complete,
                "test_uids": test_uids,
                "master_count": len(masters),
            }
            group = dict(group_body)
            group["group_id"] = "P08STRESSPOOLGROUP-" + canonical_sha256(group_body)
            pool_groups.append(group)
            support_groups.append(group)

            if not complete:
                continue
            complete_group_ids.append(group["group_id"])
            for policy_id, strategy in pairs:
                members = []
                for context_id in context_ids:
                    target = view_target.get((context_id, policy_id, strategy))
                    if target is None:
                        _fail()
                    members.append({"context_id": context_id, "procedure_id": target})
                procedure_body = {
                    "domain": domain,
                    "station": station,
                    "instrument": instrument,
                    "outer_repeat": outer_repeat,
                    "members": members,
                    "pooled_uid_sha256": canonical_sha256(test_uids),
                    "pooled_master_count": len(masters),
                }
                pooled_procedure_id = "P08STRESSPOOLPROC-" + canonical_sha256(procedure_body)
                if pooled_procedure_id not in pooled_procedure_by_id:
                    procedure = dict(procedure_body)
                    procedure["pooled_procedure_id"] = pooled_procedure_id
                    pooled_procedure_by_id[pooled_procedure_id] = procedure
                view_body = {
                    "support_id": support_id,
                    "pool_group_id": group["group_id"],
                    "domain": domain,
                    "outer_repeat": outer_repeat,
                    "policy_id": policy_id,
                    "strategy": strategy,
                    "target_pooled_procedure_id": pooled_procedure_id,
                }
                pooled_view = dict(view_body)
                pooled_view["view_id"] = "P08STRESSPOOLVIEW-" + canonical_sha256(view_body)
                pooled_views.append(pooled_view)

        supports[support_id] = {
            "context_ids": list(support_contexts),
            "complete_pool_group_ids": sorted(complete_group_ids),
            "summary": _support_summary(support_contexts, support_groups, records),
        }

    pool_groups.sort(
        key=lambda group: (
            group["support_id"],
            group["domain"],
            group["outer_repeat"],
        )
    )
    pooled_views.sort(
        key=lambda view: (
            view["support_id"],
            view["domain"],
            view["outer_repeat"],
            view["policy_id"],
            view["strategy"],
        )
    )
    pooled_procedures = [pooled_procedure_by_id[key] for key in sorted(pooled_procedure_by_id)]

    global_master_ids = sorted({row["master_id"] for row in rows})
    global_instrument_ids = sorted({row["instrument"] for row in rows})

    procedure_count = len(procedures)
    context_view_count = len(context_views)
    pooled_procedure_count = len(pooled_procedures)
    pooled_view_count = len(pooled_views)
    case_count = len(cases)
    family_count = len(family_cases)
    endpoint_count = len(_ENDPOINTS)

    stage_counts = {
        "context_case_score": procedure_count * case_count * endpoint_count,
        "context_family_curve": procedure_count * family_count * endpoint_count,
        "pooled_case_score": pooled_procedure_count * case_count * endpoint_count,
        "pooled_family_curve": pooled_procedure_count * family_count * endpoint_count,
    }
    alias_counts = {
        "context_case": context_view_count * case_count * endpoint_count,
        "context_family": context_view_count * family_count * endpoint_count,
        "pooled_case": pooled_view_count * case_count * endpoint_count,
        "pooled_family": pooled_view_count * family_count * endpoint_count,
    }
    summary = {
        "context_count": len(operational_contexts),
        "eligible_qc_context_count": len(eligible_contexts),
        "procedure_count": procedure_count,
        "context_view_count": context_view_count,
        "pooled_procedure_count": pooled_procedure_count,
        "pooled_view_count": pooled_view_count,
        "case_count": case_count,
        "endpoint_count": endpoint_count,
        "disturbance_family_count": family_count,
        "stage_counts": stage_counts,
        "score_job_count": sum(stage_counts.values()),
        "alias_counts": alias_counts,
        "reporting_alias_count": sum(alias_counts.values()),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "scientific_scores_computed": False,
        "full_stress_job_ledger_complete": False,
        "prediction_catalog": parent,
        "memberships": normalized_memberships,
        "context_records": [records[context_id] for context_id in sorted(records)],
        "pool_groups": pool_groups,
        "context_views": context_views,
        "pooled_procedures": pooled_procedures,
        "pooled_views": pooled_views,
        "global_master_ids": global_master_ids,
        "global_instrument_ids": global_instrument_ids,
        "supports": supports,
        "summary": summary,
    }


def build_stress_score_support(*, prediction_catalog, memberships):
    """Build the metadata-only stress score-support catalog.

    ``prediction_catalog`` and ``memberships`` are read and deep-copied only;
    they are never mutated.  The parent prediction catalog is validated through
    its lazy record iterator, which is intentionally never consumed.  The
    returned root always keeps every execution/verification/score flag
    ``False`` and performs no scientific work.
    """
    try:
        canonical_sha256(
            {
                "prediction_catalog": prediction_catalog,
                "memberships": memberships,
            }
        )
        iter_stress_prediction_records(prediction_catalog)
        parent = _snapshot(prediction_catalog)
        memberships_snapshot = _snapshot(memberships)
        body = _build(parent, memberships_snapshot)
        catalog = dict(body)
        catalog["catalog_sha256"] = canonical_sha256(body)
        return catalog
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None


def validate_stress_score_support(catalog):
    """Eagerly validate ``catalog`` and return an independent snapshot."""
    try:
        if not isinstance(catalog, dict) or set(catalog.keys()) != _ROOT_FIELDS:
            _fail()
        if catalog["schema_version"] != SCHEMA_VERSION:
            _fail()
        if catalog["execution_authorized"] is not False:
            _fail()
        operations = catalog["scientific_operations"]
        if type(operations) is not int or operations != 0:
            _fail()
        for flag in (
            "artifact_provenance_independently_verified",
            "scientific_scores_computed",
            "full_stress_job_ledger_complete",
        ):
            if catalog[flag] is not False:
                _fail()

        declared = _require_hex64(catalog["catalog_sha256"])
        content = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
        if canonical_sha256(content) != declared:
            _fail()

        rebuilt = build_stress_score_support(
            prediction_catalog=catalog["prediction_catalog"],
            memberships=catalog["memberships"],
        )
        if canonical_sha256(rebuilt) != canonical_sha256(catalog):
            _fail()
        return _snapshot(rebuilt)
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None
