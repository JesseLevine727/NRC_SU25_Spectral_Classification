"""P08-T247 stress contrast-binding metadata adapter (metadata only).

This module is an INTERNAL metadata adapter.  It consumes an authenticated P08
stress score-support catalog (``p08_perturbation_score_support``), the frozen
public perturbation-inference registry and a caller-authenticated instrument
platform-family mapping, and derives a deterministic, execution-disabled
binding of every declared stress contrast to its logical context/pooled views,
procedure targets and weight columns.

It performs **zero** scientific work: no scores, fits, predictions, weights,
resampling draws, quantiles, routes, statistics or numerical parity checks are
computed, guessed or inferred.  No comparison is estimated, no contrast is
evaluated and no scientific authority is granted.  The full stress job ledger
is explicitly *not* complete here.

The parent score catalog is validated through
``validate_stress_score_support`` (which returns an independent snapshot); the
returned lazy prediction graph is intentionally never consumed beyond that
validation.  The inference registry is bound verbatim by canonical digest;
this adapter is **not** a provenance validator for it and neither modifies nor
re-interprets its frozen scientific choices.  The platform-family mapping is
externally authenticated by the caller and bound by canonical digest.

Recorded master IDs keep their exact input type.  A master ID must be either an
exact ``int`` (never ``bool``) or a nonempty stripped ``str``; integers are
never stringified in the output, they are only converted to a string weight
identity for ordering and identity checks, with string collisions rejected.

Malformed metadata is sanitised to ``invalid_stress_contrast_metadata``.  Only
``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/``RecursionError``/
``OverflowError`` are caught, never ``BaseException`` or a bare ``Exception``.
"""

from __future__ import annotations

import json

from .p08_perturbation_score_support import validate_stress_score_support
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "build_stress_contrast_catalog",
    "validate_stress_contrast_catalog",
    "require_scientific_execution",
]

SCHEMA_VERSION = "nato-sers-p08-stress-contrast-catalog-v1"
INVALID = "invalid_stress_contrast_metadata"
EXECUTION_DENIED = "scientific_execution_not_authorized"

REGISTRY_SCHEMA_VERSION = "nato-sers-p08-perturbation-inference-v1"
REGISTRY_QC_MODE = "fixed_clean_route_sensitivity"

_DISTURBANCE_FAMILIES = ("shift", "slope", "quadratic", "gaussian", "impulse", "clipping")
_ENDPOINTS = ("M01", "M06")
_SCOPE_APPROVALS = ("P08-A10", "P08-A11")
_REFERENCE_POLICY = "PP-U-MIN"
_UNIVERSAL_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
_QC_POLICY = "PP-QC-SRC"
_UNIVERSAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED")
_ADAPTIVE_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED")
_NEURAL_MODELS = ("D0-M", "P05-SELECTED")
_UNIVERSAL_CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
_ADAPTIVE_CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST")

_COUNTER_FIELDS = (
    "authorized_model_fits",
    "authorized_new_predictions",
    "authorized_resampling_draws",
    "authorized_perturbation_runs",
)
_REGISTRY_FALSE_FLAGS = (
    "native_grid_gate_reaction_included",
    "source_noise_reestimated_during_inference",
    "synthetic_realizations_resampled_during_inference",
    "inference_implementation_accepted",
    "perturbation_runtime_accepted",
    "resource_proposal_approved",
    "original_G4_pass_implied",
)
_REGISTRY_TRUE_FLAGS = (
    "clean_reference_shared_between_disturbance_families",
    "recipe_identity_fixed_across_policies",
    "replicate_probability_ensemble_forbidden",
    "absolute_clean_and_stressed_BA_required",
)

_CONTRAST_COUNT = 456
_EFFECT_COUNT = 216
_INTERACTION_COUNT = 240

REGISTRY_PROTOCOL_VERSION = "nato-sers-p08-perturbation-inference-20261007-v1"

_OPERATIONAL_SUPPORT = "operational_260"
_QC_ELIGIBLE_SUPPORT = "qc_eligible_54"

_PANEL_UNIVERSAL = "universal"
_PANEL_QC_FIXED_ROUTE = "qc_fixed_route"

_FAMILY_UNIVERSAL_ROBUSTNESS_EFFECTS = "universal_robustness_effects"
_FAMILY_OPERATIONAL_QC_ROBUSTNESS_EFFECTS = "operational_qc_robustness_effects"
_FAMILY_OPERATIONAL_ROBUSTNESS_MODEL_INTERACTIONS = "operational_robustness_model_interactions"
_FAMILY_ELIGIBLE_QC_ROBUSTNESS_EFFECTS = "eligible_qc_robustness_effects"
_FAMILY_ELIGIBLE_QC_ROBUSTNESS_MODEL_INTERACTIONS = "eligible_qc_robustness_model_interactions"

_MULTIPLICITY_FAMILIES = {
    _FAMILY_UNIVERSAL_ROBUSTNESS_EFFECTS: 120,
    _FAMILY_OPERATIONAL_QC_ROBUSTNESS_EFFECTS: 48,
    _FAMILY_OPERATIONAL_ROBUSTNESS_MODEL_INTERACTIONS: 192,
    _FAMILY_ELIGIBLE_QC_ROBUSTNESS_EFFECTS: 48,
    _FAMILY_ELIGIBLE_QC_ROBUSTNESS_MODEL_INTERACTIONS: 48,
}
_MULTIPLICITY_METHOD = "Holm"
_MULTIPLICITY_SIGN_SENSITIVITIES = ["domain", "instrument_identity"]
_EFFECT_FIELDS = frozenset(
    {
        "contrast_id",
        "kind",
        "panel",
        "support",
        "disturbance_family",
        "policy",
        "method",
        "endpoint",
        "multiplicity_family",
    }
)
_INTERACTION_FIELDS = frozenset(
    {
        "contrast_id",
        "kind",
        "panel",
        "support",
        "disturbance_family",
        "policy",
        "neural",
        "classical",
        "endpoint",
        "multiplicity_family",
    }
)

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "numerical_inference_accepted",
        "full_stress_job_ledger_complete",
        "score_catalog",
        "inference_registry",
        "platform_families",
        "master_weight_columns",
        "instrument_weight_columns",
        "support_records",
        "contrasts",
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
    if type(value) is int:
        return value
    if isinstance(value, str) and value and value == value.strip():
        return value
    _fail()


def _require_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        _fail()
    for character in value:
        if character not in _HEX_DIGITS:
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


def _require_exact_list(registry, key, expected):
    value = registry[key]
    if not isinstance(value, list) or len(value) != len(expected):
        _fail()
    for item in value:
        if not isinstance(item, str):
            _fail()
    if tuple(value) != expected:
        _fail()


def _validate_multiplicity(registry):
    multiplicity = _require_mapping(registry["multiplicity"])

    families = _require_mapping(multiplicity["families"])
    if set(families.keys()) != set(_MULTIPLICITY_FAMILIES):
        _fail()
    for family, expected_count in _MULTIPLICITY_FAMILIES.items():
        count = families[family]
        if type(count) is not int or count != expected_count:
            _fail()

    total_contrasts = multiplicity["total_contrasts"]
    if type(total_contrasts) is not int or total_contrasts != _CONTRAST_COUNT:
        _fail()

    if multiplicity["procedure"] != _MULTIPLICITY_METHOD:
        _fail()

    if multiplicity["span_all_six_disturbance_families"] is not True:
        _fail()

    sensitivities = multiplicity["separate_sign_sensitivities"]
    if not isinstance(sensitivities, list) or sensitivities != _MULTIPLICITY_SIGN_SENSITIVITIES:
        _fail()

    if multiplicity["remove_structural_duplicates"] is not False:
        _fail()

    bookkeeping_p = multiplicity["unavailable_adjustment_bookkeeping_p"]
    if type(bookkeeping_p) is not int or bookkeeping_p != 1:
        _fail()

    if multiplicity["unavailable_estimate_and_raw_p"] is not None:
        _fail()

    if multiplicity["primary_family_sizes_changed"] is not False:
        _fail()

    if multiplicity["pointwise_or_directional_or_pooled_extra_families"] is not False:
        _fail()


def _validate_registry(registry):
    _require_mapping(registry)
    if registry["schema_version"] != REGISTRY_SCHEMA_VERSION:
        _fail()
    if registry["protocol_version"] != REGISTRY_PROTOCOL_VERSION:
        _fail()
    if registry["execution_authorized"] is not False:
        _fail()
    for field in _COUNTER_FIELDS:
        value = registry[field]
        if type(value) is not int or value != 0:
            _fail()
    _require_exact_list(registry, "scope_approvals", _SCOPE_APPROVALS)
    _require_exact_list(registry, "disturbance_families", _DISTURBANCE_FAMILIES)
    _require_exact_list(registry, "endpoints", _ENDPOINTS)
    _require_exact_list(registry, "comparison_universal_policies", _UNIVERSAL_POLICIES)
    _require_exact_list(registry, "universal_models", _UNIVERSAL_MODELS)
    _require_exact_list(registry, "adaptive_models", _ADAPTIVE_MODELS)
    _require_exact_list(registry, "neural_models", _NEURAL_MODELS)
    _require_exact_list(registry, "universal_classical_models", _UNIVERSAL_CLASSICAL_MODELS)
    _require_exact_list(registry, "adaptive_classical_models", _ADAPTIVE_CLASSICAL_MODELS)
    if registry["reference_policy"] != _REFERENCE_POLICY:
        _fail()
    if registry["comparison_qc_policy"] != _QC_POLICY:
        _fail()
    if registry["qc_mode"] != REGISTRY_QC_MODE:
        _fail()
    for field in _REGISTRY_FALSE_FLAGS:
        if registry[field] is not False:
            _fail()
    for field in _REGISTRY_TRUE_FLAGS:
        if registry[field] is not True:
            _fail()
    if not isinstance(registry["contrasts"], list) or not registry["contrasts"]:
        _fail()
    _validate_multiplicity(registry)


def _expected_contrast_semantics():
    records = []
    for disturbance_family in _DISTURBANCE_FAMILIES:
        for endpoint in _ENDPOINTS:
            for policy in _UNIVERSAL_POLICIES:
                for method in _UNIVERSAL_MODELS:
                    records.append(
                        {
                            "kind": "effect",
                            "panel": _PANEL_UNIVERSAL,
                            "support": _OPERATIONAL_SUPPORT,
                            "disturbance_family": disturbance_family,
                            "policy": policy,
                            "method": method,
                            "endpoint": endpoint,
                            "multiplicity_family": _FAMILY_UNIVERSAL_ROBUSTNESS_EFFECTS,
                        }
                    )
            for method in _ADAPTIVE_MODELS:
                records.append(
                    {
                        "kind": "effect",
                        "panel": _PANEL_QC_FIXED_ROUTE,
                        "support": _OPERATIONAL_SUPPORT,
                        "disturbance_family": disturbance_family,
                        "policy": _QC_POLICY,
                        "method": method,
                        "endpoint": endpoint,
                        "multiplicity_family": _FAMILY_OPERATIONAL_QC_ROBUSTNESS_EFFECTS,
                    }
                )
                records.append(
                    {
                        "kind": "effect",
                        "panel": _PANEL_QC_FIXED_ROUTE,
                        "support": _QC_ELIGIBLE_SUPPORT,
                        "disturbance_family": disturbance_family,
                        "policy": _QC_POLICY,
                        "method": method,
                        "endpoint": endpoint,
                        "multiplicity_family": _FAMILY_ELIGIBLE_QC_ROBUSTNESS_EFFECTS,
                    }
                )
            for policy in _UNIVERSAL_POLICIES:
                for neural in _NEURAL_MODELS:
                    for classical in _UNIVERSAL_CLASSICAL_MODELS:
                        records.append(
                            {
                                "kind": "interaction",
                                "panel": _PANEL_UNIVERSAL,
                                "support": _OPERATIONAL_SUPPORT,
                                "disturbance_family": disturbance_family,
                                "policy": policy,
                                "neural": neural,
                                "classical": classical,
                                "endpoint": endpoint,
                                "multiplicity_family": (
                                    _FAMILY_OPERATIONAL_ROBUSTNESS_MODEL_INTERACTIONS
                                ),
                            }
                        )
            for neural in _NEURAL_MODELS:
                for classical in _ADAPTIVE_CLASSICAL_MODELS:
                    records.append(
                        {
                            "kind": "interaction",
                            "panel": _PANEL_QC_FIXED_ROUTE,
                            "support": _OPERATIONAL_SUPPORT,
                            "disturbance_family": disturbance_family,
                            "policy": _QC_POLICY,
                            "neural": neural,
                            "classical": classical,
                            "endpoint": endpoint,
                            "multiplicity_family": (
                                _FAMILY_OPERATIONAL_ROBUSTNESS_MODEL_INTERACTIONS
                            ),
                        }
                    )
                    records.append(
                        {
                            "kind": "interaction",
                            "panel": _PANEL_QC_FIXED_ROUTE,
                            "support": _QC_ELIGIBLE_SUPPORT,
                            "disturbance_family": disturbance_family,
                            "policy": _QC_POLICY,
                            "neural": neural,
                            "classical": classical,
                            "endpoint": endpoint,
                            "multiplicity_family": (
                                _FAMILY_ELIGIBLE_QC_ROBUSTNESS_MODEL_INTERACTIONS
                            ),
                        }
                    )
    return records


def _validate_contrasts(registry):
    expected_records = _expected_contrast_semantics()
    expected_effects = 0
    expected_interactions = 0
    expected = set()
    for record in expected_records:
        if record["kind"] == "effect":
            expected_effects += 1
        else:
            expected_interactions += 1
        digest = canonical_sha256(record)
        if digest in expected:
            _fail()
        expected.add(digest)
    if expected_effects != _EFFECT_COUNT or expected_interactions != _INTERACTION_COUNT:
        _fail()
    if len(expected_records) != _CONTRAST_COUNT:
        _fail()

    seen_ids = set()
    actual = set()
    for record in registry["contrasts"]:
        _require_mapping(record)
        kind = record.get("kind")
        if kind == "effect":
            if set(record.keys()) != _EFFECT_FIELDS:
                _fail()
        elif kind == "interaction":
            if set(record.keys()) != _INTERACTION_FIELDS:
                _fail()
        else:
            _fail()

        contrast_id = _require_identifier(record["contrast_id"])
        if contrast_id in seen_ids:
            _fail()
        seen_ids.add(contrast_id)

        semantic = {key: value for key, value in record.items() if key != "contrast_id"}
        digest = canonical_sha256(semantic)
        if digest in actual:
            _fail()
        actual.add(digest)

    if actual != expected:
        _fail()


def _build_master_columns(master_ids):
    if not isinstance(master_ids, list) or not master_ids:
        _fail()
    identities = {}
    for master_id in master_ids:
        _require_master_id(master_id)
        identity = str(master_id)
        if identity in identities:
            _fail()
        identities[identity] = master_id
    columns = []
    for index, identity in enumerate(sorted(identities)):
        columns.append(
            {
                "master_id": identities[identity],
                "weight_identity": identity,
                "column": index,
            }
        )
    return columns, {row["weight_identity"]: row["column"] for row in columns}


def _build_instrument_columns(instrument_ids):
    if not isinstance(instrument_ids, list) or not instrument_ids:
        _fail()
    seen = set()
    for instrument in instrument_ids:
        _require_identifier(instrument)
        if instrument in seen:
            _fail()
        seen.add(instrument)
    columns = []
    for index, instrument in enumerate(sorted(seen)):
        columns.append(
            {
                "instrument": instrument,
                "weight_identity": instrument,
                "column": index,
            }
        )
    return columns, {row["instrument"]: row["column"] for row in columns}


def _build_support_records(
    parent,
    context_records,
    pool_groups,
    master_index,
    instrument_index,
    platform_families,
):
    supports = _require_mapping(parent["supports"])
    if not supports:
        _fail()
    records = []
    for support_id in sorted(supports):
        _require_identifier(support_id)
        support = _require_mapping(supports[support_id])

        context_ids = []
        seen_contexts = set()
        for context_id in _require_sequence(support["context_ids"]):
            _require_identifier(context_id)
            if context_id in seen_contexts or context_id not in context_records:
                _fail()
            seen_contexts.add(context_id)
            context_ids.append(context_id)
        context_ids.sort()

        complete_ids = []
        seen_groups = set()
        for group_id in _require_sequence(support["complete_pool_group_ids"]):
            _require_identifier(group_id)
            group = pool_groups.get(group_id)
            if group is None or group.get("support_id") != support_id:
                _fail()
            if not set(group["context_ids"]) <= seen_contexts:
                _fail()
            if group_id in seen_groups:
                _fail()
            seen_groups.add(group_id)
            complete_ids.append(group_id)
        complete_ids.sort()

        masters = set()
        instruments = set()
        domains = set()
        for context_id in context_ids:
            record = context_records[context_id]
            domains.add(_require_identifier(record["domain"]))
            instruments.add(_require_identifier(record["instrument"]))
            for unit in record["master_units"]:
                _require_mapping(unit)
                masters.add(_require_master_id(unit["master_id"]))

        master_columns = set()
        for master_id in masters:
            column = master_index.get(str(master_id))
            if column is None:
                _fail()
            master_columns.add(column)
        instrument_columns = set()
        for instrument in instruments:
            column = instrument_index.get(instrument)
            if column is None:
                _fail()
            instrument_columns.add(column)

        known = sorted(
            {
                platform_families[instrument]
                for instrument in instruments
                if platform_families[instrument] is not None
            }
        )
        unknown = sorted(
            instrument for instrument in instruments if platform_families[instrument] is None
        )

        records.append(
            {
                "support_id": support_id,
                "context_ids": context_ids,
                "complete_pool_group_ids": complete_ids,
                "master_weight_columns": sorted(master_columns),
                "instrument_weight_columns": sorted(instrument_columns),
                "domains": sorted(domains),
                "instruments": sorted(instruments),
                "known_platform_families": known,
                "unknown_family_instruments": unknown,
            }
        )
    return records


def _effect_terms(record):
    method = record["method"]
    return [
        {"coefficient": 1, "policy_id": _REFERENCE_POLICY, "strategy": method},
        {"coefficient": -1, "policy_id": record["policy"], "strategy": method},
    ]


def _interaction_terms(record):
    policy = record["policy"]
    neural = record["neural"]
    classical = record["classical"]
    return [
        {"coefficient": 1, "policy_id": _REFERENCE_POLICY, "strategy": neural},
        {"coefficient": -1, "policy_id": policy, "strategy": neural},
        {"coefficient": -1, "policy_id": _REFERENCE_POLICY, "strategy": classical},
        {"coefficient": 1, "policy_id": policy, "strategy": classical},
    ]


def _build_contrasts(registry, support_lookup, context_views, pooled_views):
    contrasts = []
    for record in registry["contrasts"]:
        kind = record["kind"]
        terms = _effect_terms(record) if kind == "effect" else _interaction_terms(record)

        support_id = record["support"]
        support = support_lookup.get(support_id)
        if support is None:
            _fail()

        context_bindings = []
        for context_id in support["context_ids"]:
            view_ids = []
            target_procedure_ids = []
            for term in terms:
                view = context_views.get((context_id, term["policy_id"], term["strategy"]))
                if view is None:
                    _fail()
                view_ids.append(_require_identifier(view["view_id"]))
                target_procedure_ids.append(_require_identifier(view["target_procedure_id"]))
            context_bindings.append(
                {
                    "context_id": context_id,
                    "view_ids": view_ids,
                    "target_procedure_ids": target_procedure_ids,
                }
            )
        context_bindings.sort(key=lambda entry: entry["context_id"])

        pooled_bindings = []
        for group_id in support["complete_pool_group_ids"]:
            view_ids = []
            target_pooled_procedure_ids = []
            for term in terms:
                view = pooled_views.get((support_id, group_id, term["policy_id"], term["strategy"]))
                if view is None:
                    _fail()
                view_ids.append(_require_identifier(view["view_id"]))
                target_pooled_procedure_ids.append(
                    _require_identifier(view["target_pooled_procedure_id"])
                )
            pooled_bindings.append(
                {
                    "pool_group_id": group_id,
                    "view_ids": view_ids,
                    "target_pooled_procedure_ids": target_pooled_procedure_ids,
                }
            )
        pooled_bindings.sort(key=lambda entry: entry["pool_group_id"])

        body = {
            "contrast_id": record["contrast_id"],
            "kind": kind,
            "panel": record["panel"],
            "support_id": support_id,
            "disturbance_family": record["disturbance_family"],
            "endpoint": record["endpoint"],
            "multiplicity_family": record["multiplicity_family"],
            "terms": terms,
            "context_bindings": context_bindings,
            "pooled_bindings": pooled_bindings,
        }
        contrast = dict(body)
        contrast["contrast_binding_sha256"] = canonical_sha256(body)
        contrasts.append(contrast)

    contrasts.sort(key=lambda entry: entry["contrast_id"])
    return contrasts


def _build_summary(parent, master_columns, instrument_columns, contrasts, platform_families):
    kind_counts = {}
    family_counts = {}
    support_counts = {}
    context_units = 0
    pooled_units = 0
    context_references = 0
    pooled_references = 0
    for contrast in contrasts:
        kind = contrast["kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        family = contrast["multiplicity_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
        support_id = contrast["support_id"]
        support_counts[support_id] = support_counts.get(support_id, 0) + 1
        context_count = len(contrast["context_bindings"])
        pooled_count = len(contrast["pooled_bindings"])
        context_units += context_count
        pooled_units += pooled_count
        context_references += len(contrast["terms"]) * context_count
        pooled_references += len(contrast["terms"]) * pooled_count

    return {
        "contrast_count": len(contrasts),
        "kind_counts": dict(sorted(kind_counts.items())),
        "multiplicity_family_counts": dict(sorted(family_counts.items())),
        "support_contrast_counts": dict(sorted(support_counts.items())),
        "context_comparison_units": context_units,
        "pooled_comparison_units": pooled_units,
        "signed_context_term_references": context_references,
        "signed_pooled_term_references": pooled_references,
        "global_masters": len(master_columns),
        "global_instruments": len(instrument_columns),
        "all_authorization_flags_false": True,
        "platform_family_mapping_sha256": canonical_sha256(platform_families),
    }


def _validate_platform_families(mapping, parent):
    if not isinstance(mapping, dict):
        _fail()
    expected = set()
    for instrument in parent["global_instrument_ids"]:
        _require_identifier(instrument)
        expected.add(instrument)
    if not expected or set(mapping.keys()) != expected:
        _fail()
    for value in mapping.values():
        if value is None:
            continue
        if not isinstance(value, str) or not value or value != value.strip():
            _fail()


def _build(parent, registry, platform_families):
    master_columns, master_index = _build_master_columns(parent["global_master_ids"])
    instrument_columns, instrument_index = _build_instrument_columns(
        parent["global_instrument_ids"]
    )

    context_records = {}
    for record in parent["context_records"]:
        _require_mapping(record)
        context_id = _require_identifier(record["context_id"])
        if context_id in context_records:
            _fail()
        context_records[context_id] = record

    pool_groups = {}
    for group in parent["pool_groups"]:
        _require_mapping(group)
        group_id = _require_identifier(group["group_id"])
        if group_id in pool_groups:
            _fail()
        pool_groups[group_id] = group

    context_views = {}
    for view in parent["context_views"]:
        _require_mapping(view)
        key = (
            _require_identifier(view["context_id"]),
            _require_identifier(view["policy_id"]),
            _require_identifier(view["strategy"]),
        )
        if key in context_views:
            _fail()
        context_views[key] = view

    pooled_views = {}
    for view in parent["pooled_views"]:
        _require_mapping(view)
        key = (
            _require_identifier(view["support_id"]),
            _require_identifier(view["pool_group_id"]),
            _require_identifier(view["policy_id"]),
            _require_identifier(view["strategy"]),
        )
        if key in pooled_views:
            _fail()
        pooled_views[key] = view

    support_records = _build_support_records(
        parent,
        context_records,
        pool_groups,
        master_index,
        instrument_index,
        platform_families,
    )
    support_lookup = {record["support_id"]: record for record in support_records}

    contrasts = _build_contrasts(registry, support_lookup, context_views, pooled_views)
    summary = _build_summary(
        parent, master_columns, instrument_columns, contrasts, platform_families
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "numerical_inference_accepted": False,
        "full_stress_job_ledger_complete": False,
        "score_catalog": parent,
        "inference_registry": registry,
        "platform_families": platform_families,
        "master_weight_columns": master_columns,
        "instrument_weight_columns": instrument_columns,
        "support_records": support_records,
        "contrasts": contrasts,
        "summary": summary,
    }


def build_stress_contrast_catalog(*, score_catalog, inference_registry, platform_families):
    """Build the metadata-only stress contrast-binding catalog."""
    try:
        canonical_sha256(
            {
                "score_catalog": score_catalog,
                "inference_registry": inference_registry,
                "platform_families": platform_families,
            }
        )
        parent = validate_stress_score_support(score_catalog)
        registry = _snapshot(inference_registry)
        _validate_registry(registry)
        _validate_contrasts(registry)
        families = _snapshot(platform_families)
        _validate_platform_families(families, parent)
        body = _build(parent, registry, families)
        catalog = dict(body)
        catalog["catalog_sha256"] = canonical_sha256(body)
        return catalog
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None


def validate_stress_contrast_catalog(catalog):
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
            "numerical_inference_accepted",
            "full_stress_job_ledger_complete",
        ):
            if catalog[flag] is not False:
                _fail()

        declared = _require_hex64(catalog["catalog_sha256"])
        body = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
        if canonical_sha256(body) != declared:
            _fail()

        rebuilt = build_stress_contrast_catalog(
            score_catalog=catalog["score_catalog"],
            inference_registry=catalog["inference_registry"],
            platform_families=catalog["platform_families"],
        )
        if canonical_sha256(rebuilt) != canonical_sha256(catalog):
            _fail()
        return _snapshot(rebuilt)
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None
