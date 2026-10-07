"""P08-T257 bounded stress inference operation ledger (metadata only).

This module is an INTERNAL metadata ledger.  It consumes an authenticated P08
stress contrast-binding catalog (``p08_stress_contrasts``) and derives a
deterministic, execution-disabled inventory of the finite future operations the
registered P08 robustness comparisons will require: shared positive-weight
authentication, per-contrast support assessment, point estimates, unit-weight
parity checks, positive-weight batches and summaries, the original hierarchical
realization and its batches, sign sensitivities, Holm bookkeeping, domain
stability and deletion stability.

It performs **zero** scientific work.  No scores, fits, predictions, weights,
resampling draws, quantiles, routes, statistics, arrays or numerical parity
checks are computed, guessed or inferred.  No operation is authorized, no
operation result is recorded and no comparison is evaluated.  The full stress
job ledger is explicitly *not* complete: probability-quality, spectral-
preservation and final-figure operation inventories remain separate.

The parent contrast catalog is validated through
``validate_stress_contrast_catalog`` (which returns an independent snapshot).
The parent prediction/score graph is never consumed.  Every external dependency
is an explicit reference to the parent catalog digest, a contrast binding digest
or a declared shared global-weight stream.

Recorded operations are metadata descriptors only.  Identifiers are derived
from content, never from iteration position, Python hash or any stochastic
source.

Malformed metadata is sanitised to ``invalid_stress_inference_plan``.  Only
``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/``RecursionError``/
``OverflowError`` are caught, never ``BaseException`` or a bare ``Exception``.
"""

from __future__ import annotations

import json

from .p08_qc_blocks import canonical_sha256
from .p08_stress_contrasts import validate_stress_contrast_catalog

__all__ = [
    "build_stress_inference_plan",
    "validate_stress_inference_plan",
    "iter_stress_inference_jobs",
    "require_scientific_execution",
]

SCHEMA_VERSION = "nato-sers-p08-stress-inference-plan-v1"
INVALID = "invalid_stress_inference_plan"
EXECUTION_DENIED = "scientific_execution_not_authorized"

_BATCH_SIZE = 128
_TOTAL_DRAWS = 10000
_BATCH_COUNT = 79
_TRAILING_DRAW_COUNT = 16

_VIEWS = ("fixed_context", "fixed_pooled", "paired_context", "paired_pooled")
_FIXED_VIEWS = ("fixed_context", "fixed_pooled")
_PAIRED_VIEWS = ("paired_context", "paired_pooled")
_SIGN_VIEWS = ("fixed_context", "paired_context")
_VIEW_TARGET_KIND = {
    "fixed_context": "context",
    "fixed_pooled": "pooled",
    "paired_context": "context",
    "paired_pooled": "pooled",
}
_VIEW_ACTIVATION = {
    "fixed_context": "unconditional",
    "fixed_pooled": "unconditional",
    "paired_context": "conditional",
    "paired_pooled": "conditional",
}
_PAIRED_ACTIVATION = "conditional_on_fixed_target_unavailable_and_nonempty_whole_curve_intersection"

_WEIGHT_MODES = ("crossed", "master_only", "instrument_only")
_WEIGHT_IDS = ("master", "instrument")
_WEIGHT_SEEDS = {"master": 2026093001, "instrument": 2026093002}
_HIERARCHY_SEED = 2026093003

_SIGN_UNITS = ("domain", "instrument_identity")

_MULTIPLICITY_FAMILIES = {
    "universal_robustness_effects": 120,
    "operational_qc_robustness_effects": 48,
    "operational_robustness_model_interactions": 192,
    "eligible_qc_robustness_effects": 48,
    "eligible_qc_robustness_model_interactions": 48,
}

_STAGES = (
    "global_weight_authentication",
    "support_assessment",
    "point_estimate",
    "unit_weight_parity",
    "weighted_batch",
    "weighted_summary",
    "hierarchy_realization_prepare",
    "hierarchy_batch",
    "hierarchy_summary",
    "sign_sensitivity",
    "holm_adjustment",
    "domain_stability",
    "deletion_stability",
)

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "numerical_inference_accepted",
        "full_stress_job_ledger_complete",
        "source_contrast_catalog_sha256",
        "activation_rules",
        "shared_batches",
        "global_weight_authentication",
        "contrasts",
        "operation_blocks",
        "summary",
        "plan_sha256",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this metadata-only ledger."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID)


def _require_mapping(value):
    if not isinstance(value, dict):
        _fail()
    return value


def _require_identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail()
    return value


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


def _require_strict_json(value):
    json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _job_id(stage, **fields):
    payload = {"stage": stage}
    payload.update(fields)
    return canonical_sha256(payload)


def _support_job_id(contrast_id, target_kind):
    return _job_id("support_assessment", contrast_id=contrast_id, target_kind=target_kind)


def _point_job_id(contrast_id, view_id):
    return _job_id("point_estimate", contrast_id=contrast_id, view_id=view_id)


def _parity_job_id(contrast_id, view_id, mode):
    return _job_id("unit_weight_parity", contrast_id=contrast_id, view_id=view_id, mode=mode)


def _batch_job_id(contrast_id, view_id, mode, batch_index):
    return _job_id(
        "weighted_batch",
        contrast_id=contrast_id,
        view_id=view_id,
        mode=mode,
        batch_index=batch_index,
    )


def _weighted_summary_job_id(contrast_id, view_id, mode):
    return _job_id("weighted_summary", contrast_id=contrast_id, view_id=view_id, mode=mode)


def _hierarchy_prepare_job_id(contrast_id):
    return _job_id("hierarchy_realization_prepare", contrast_id=contrast_id)


def _hierarchy_batch_job_id(contrast_id, batch_index):
    return _job_id("hierarchy_batch", contrast_id=contrast_id, batch_index=batch_index)


def _hierarchy_summary_job_id(contrast_id):
    return _job_id("hierarchy_summary", contrast_id=contrast_id)


def _sign_job_id(contrast_id, view_id, sign_unit):
    return _job_id(
        "sign_sensitivity",
        contrast_id=contrast_id,
        view_id=view_id,
        sign_unit=sign_unit,
    )


def _holm_job_id(multiplicity_family, sign_unit):
    return _job_id("holm_adjustment", multiplicity_family=multiplicity_family, sign_unit=sign_unit)


def _domain_stability_job_id(contrast_id):
    return _job_id("domain_stability", contrast_id=contrast_id)


def _deletion_job_id(contrast_id, deletion_kind, deletion_id):
    return _job_id(
        "deletion_stability",
        contrast_id=contrast_id,
        deletion_kind=deletion_kind,
        deletion_id=deletion_id,
    )


def _auth_logical_id(weight_id):
    return _job_id("global_weight_authentication", weight_id=weight_id)


def _shared_batches():
    ranges = [
        [start, min(start + _BATCH_SIZE, _TOTAL_DRAWS)]
        for start in range(0, _TOTAL_DRAWS, _BATCH_SIZE)
    ]
    return {
        "batch_size": _BATCH_SIZE,
        "batch_count": len(ranges),
        "total_draws": _TOTAL_DRAWS,
        "trailing_draw_count": _TRAILING_DRAW_COUNT,
        "ranges": ranges,
    }


def _global_weight_records(catalog):
    columns_by_weight = {
        "master": catalog["master_weight_columns"],
        "instrument": catalog["instrument_weight_columns"],
    }
    records = []
    for weight_id in _WEIGHT_IDS:
        columns = list(columns_by_weight[weight_id])
        identities = sorted(_require_identifier(entry["weight_identity"]) for entry in columns)
        if len(set(identities)) != len(identities):
            _fail()
        records.append(
            {
                "logical_id": _auth_logical_id(weight_id),
                "weight_id": weight_id,
                "seed": _WEIGHT_SEEDS[weight_id],
                "generator": "PCG64",
                "distribution": "Exponential(mean=1)",
                "draw_count": _TOTAL_DRAWS,
                "dtype": "float64",
                "positive_finite_only": True,
                "identity_order": "lexicographic",
                "identity_count": len(identities),
                "identity_sha256": canonical_sha256(identities),
                "reused_across_modes_views_cases_repetitions": True,
                "subset_redraw_forbidden": True,
            }
        )
    return records


def _block(stage, allocated, unconditional_allocated, conditional_allocated, **extra):
    block = {
        "stage": stage,
        "allocated": allocated,
        "unconditional_allocated": unconditional_allocated,
        "conditional_allocated": conditional_allocated,
    }
    for key in sorted(extra):
        block[key] = extra[key]
    return block


def _operation_blocks(contrast_records):
    count = len(contrast_records)
    deletion = sum(
        record["domain_count"] + record["instrument_count"] + record["known_platform_family_count"]
        for record in contrast_records
    )
    return [
        _block("global_weight_authentication", 2, 2, 0),
        _block("support_assessment", 2 * count, 2 * count, 0),
        _block("point_estimate", 4 * count, 2 * count, 2 * count),
        _block("unit_weight_parity", 12 * count, 6 * count, 6 * count),
        _block(
            "weighted_batch",
            12 * _BATCH_COUNT * count,
            6 * _BATCH_COUNT * count,
            6 * _BATCH_COUNT * count,
            batch_count=_BATCH_COUNT,
            modes=list(_WEIGHT_MODES),
            views=list(_VIEWS),
        ),
        _block("weighted_summary", 12 * count, 6 * count, 6 * count),
        _block(
            "hierarchy_realization_prepare",
            count,
            count,
            0,
            draw_count=_TOTAL_DRAWS,
            seed=_HIERARCHY_SEED,
        ),
        _block(
            "hierarchy_batch",
            _BATCH_COUNT * count,
            _BATCH_COUNT * count,
            0,
        ),
        _block("hierarchy_summary", count, count, 0),
        _block(
            "sign_sensitivity",
            4 * count,
            2 * count,
            2 * count,
            sign_units=list(_SIGN_UNITS),
            views=list(_SIGN_VIEWS),
        ),
        _block(
            "holm_adjustment",
            len(_MULTIPLICITY_FAMILIES) * len(_SIGN_UNITS),
            len(_MULTIPLICITY_FAMILIES) * len(_SIGN_UNITS),
            0,
            family_count=len(_MULTIPLICITY_FAMILIES),
            sign_unit_count=len(_SIGN_UNITS),
        ),
        _block("domain_stability", count, count, 0),
        _block("deletion_stability", deletion, deletion, 0),
    ]


def _summarize(contrast_records, blocks):
    stage_counts = {}
    stage_unconditional = {}
    stage_conditional = {}
    for block in blocks:
        stage = block["stage"]
        stage_counts[stage] = block["allocated"]
        stage_unconditional[stage] = block["unconditional_allocated"]
        stage_conditional[stage] = block["conditional_allocated"]
    total = sum(stage_counts.values())
    total_unconditional = sum(stage_unconditional.values())
    total_conditional = sum(stage_conditional.values())
    fixed_sign = sum(
        record["fixed_domain_sign_assignments"] + record["fixed_instrument_sign_assignments"]
        for record in contrast_records
    )
    paired_sign = sum(
        record["paired_domain_sign_assignment_upper_bound"]
        + record["paired_instrument_sign_assignment_upper_bound"]
        for record in contrast_records
    )
    count = len(contrast_records)
    return {
        "contrast_count": count,
        "stage_counts": stage_counts,
        "stage_unconditional_counts": stage_unconditional,
        "stage_conditional_counts": stage_conditional,
        "total_operation_descriptors": total,
        "unconditional_operation_descriptors": total_unconditional,
        "conditional_operation_descriptors": total_conditional,
        "conditional_activated_slots": None,
        "activation_status": "unresolved_until_runtime",
        "shared_batch_count": _BATCH_COUNT,
        "shared_batch_size": _BATCH_SIZE,
        "shared_total_draws": _TOTAL_DRAWS,
        "trailing_draw_count": _TRAILING_DRAW_COUNT,
        "weighted_terminal_scalar_upper_bound": (
            count * len(_VIEWS) * len(_WEIGHT_MODES) * _TOTAL_DRAWS
        ),
        "fixed_sign_assignment_exact_total": fixed_sign,
        "paired_sign_assignment_upper_bound": paired_sign,
        "holm_family_sizes": dict(_MULTIPLICITY_FAMILIES),
        "holm_job_count": len(_MULTIPLICITY_FAMILIES) * len(_SIGN_UNITS),
        "full_stress_job_ledger_complete": False,
        "scientific_operations": 0,
    }


def _build_plan(catalog):
    catalog_hash = _require_hex64(catalog["catalog_sha256"])

    supports = {}
    for support in catalog["support_records"]:
        _require_mapping(support)
        support_id = _require_identifier(support["support_id"])
        if support_id in supports:
            _fail()
        supports[support_id] = support

    contrast_records = []
    seen_contrast_ids = set()
    for contrast in catalog["contrasts"]:
        _require_mapping(contrast)
        contrast_id = _require_identifier(contrast["contrast_id"])
        if contrast_id in seen_contrast_ids:
            _fail()
        seen_contrast_ids.add(contrast_id)
        support_id = _require_identifier(contrast["support_id"])
        support = supports.get(support_id)
        if support is None:
            _fail()

        context_ids = list(support["context_ids"])
        pool_ids = list(support["complete_pool_group_ids"])
        context_sha = canonical_sha256(context_ids)
        pool_sha = canonical_sha256(pool_ids)

        domains = list(support["domains"])
        instruments = list(support["instruments"])
        known_families = list(support["known_platform_families"])

        contrast_records.append(
            {
                "contrast_id": contrast_id,
                "kind": _require_identifier(contrast["kind"]),
                "support_id": support_id,
                "multiplicity_family": _require_identifier(contrast["multiplicity_family"]),
                "contrast_binding_sha256": _require_hex64(contrast["contrast_binding_sha256"]),
                "fixed_context": {"count": len(context_ids), "sha256": context_sha},
                "fixed_pooled": {"count": len(pool_ids), "sha256": pool_sha},
                "paired_context": {
                    "count": len(context_ids),
                    "sha256": context_sha,
                    "candidate_support_nonempty": bool(context_ids),
                },
                "paired_pooled": {
                    "count": len(pool_ids),
                    "sha256": pool_sha,
                    "candidate_support_nonempty": bool(pool_ids),
                },
                "domain_count": len(domains),
                "instrument_count": len(instruments),
                "known_platform_family_count": len(known_families),
                "fixed_domain_sign_assignments": 1 << len(domains),
                "fixed_instrument_sign_assignments": 1 << len(instruments),
                "paired_domain_sign_assignment_upper_bound": 1 << len(domains),
                "paired_instrument_sign_assignment_upper_bound": 1 << len(instruments),
            }
        )

    contrast_records.sort(key=lambda record: record["contrast_id"])

    blocks = _operation_blocks(contrast_records)
    summary = _summarize(contrast_records, blocks)

    return {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "numerical_inference_accepted": False,
        "full_stress_job_ledger_complete": False,
        "source_contrast_catalog_sha256": catalog_hash,
        "activation_rules": {
            "fixed_views": "unconditional",
            "paired_views": _PAIRED_ACTIVATION,
            "runtime_resolved": False,
            "required_at_runtime": True,
        },
        "shared_batches": _shared_batches(),
        "global_weight_authentication": _global_weight_records(catalog),
        "contrasts": contrast_records,
        "operation_blocks": blocks,
        "summary": summary,
    }


def build_stress_inference_plan(*, contrast_catalog):
    """Build the metadata-only stress inference operation ledger."""
    try:
        catalog = validate_stress_contrast_catalog(contrast_catalog)
        body = _build_plan(catalog)
        plan = dict(body)
        plan["plan_sha256"] = canonical_sha256(body)
        return plan
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None


def _validated_pair(plan, contrast_catalog):
    if not isinstance(plan, dict) or set(plan.keys()) != _ROOT_FIELDS:
        _fail()
    if plan["schema_version"] != SCHEMA_VERSION:
        _fail()
    if plan["execution_authorized"] is not False:
        _fail()
    operations = plan["scientific_operations"]
    if type(operations) is not int or operations != 0:
        _fail()
    for flag in ("numerical_inference_accepted", "full_stress_job_ledger_complete"):
        if plan[flag] is not False:
            _fail()

    declared = _require_hex64(plan["plan_sha256"])
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require_strict_json(body)
    if canonical_sha256(body) != declared:
        _fail()

    catalog = validate_stress_contrast_catalog(contrast_catalog)
    rebuilt = _build_plan(catalog)
    if canonical_sha256(rebuilt) != canonical_sha256(body):
        _fail()
    full = dict(rebuilt)
    full["plan_sha256"] = canonical_sha256(rebuilt)
    return _snapshot(full), catalog


def validate_stress_inference_plan(plan, *, contrast_catalog):
    """Eagerly validate ``plan`` against the parent catalog and snapshot it."""
    try:
        snapshot, _ = _validated_pair(plan, contrast_catalog)
        return snapshot
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None


def _iter_unbound_jobs(plan, catalog):
    catalog_hash = plan["source_contrast_catalog_sha256"]
    ranges = plan["shared_batches"]["ranges"]
    supports = {record["support_id"]: record for record in catalog["support_records"]}
    contrast_records = plan["contrasts"]
    weight_records = {
        record["weight_id"]: record for record in plan["global_weight_authentication"]
    }

    def binding_reference(contrast_id, binding_sha256):
        return {
            "kind": "contrast_binding",
            "source_contrast_catalog_sha256": catalog_hash,
            "contrast_id": contrast_id,
            "contrast_binding_sha256": binding_sha256,
        }

    def weight_reference(weight_id):
        record = weight_records[weight_id]
        return {
            "kind": "shared_global_weight",
            "weight_id": weight_id,
            "seed": record["seed"],
            "generator": record["generator"],
            "distribution": record["distribution"],
            "draw_count": record["draw_count"],
            "dtype": record["dtype"],
            "identity_count": record["identity_count"],
            "identity_sha256": record["identity_sha256"],
            "positive_finite_only": record["positive_finite_only"],
            "identity_order": record["identity_order"],
        }

    def weight_declaration_reference(weight_id):
        record = weight_records[weight_id]
        return {
            "kind": "declared_shared_global_weight_recipe",
            "source_contrast_catalog_sha256": catalog_hash,
            "weight_id": weight_id,
            "seed": record["seed"],
            "generator": record["generator"],
            "distribution": record["distribution"],
            "draw_count": record["draw_count"],
            "dtype": record["dtype"],
            "identity_count": record["identity_count"],
            "identity_sha256": record["identity_sha256"],
        }

    for record in plan["global_weight_authentication"]:
        yield {
            "logical_id": record["logical_id"],
            "stage": "global_weight_authentication",
            "activation": "unconditional",
            "weight_id": record["weight_id"],
            "seed": record["seed"],
            "generator": record["generator"],
            "distribution": record["distribution"],
            "draw_count": record["draw_count"],
            "dtype": record["dtype"],
            "positive_finite_only": record["positive_finite_only"],
            "identity_order": record["identity_order"],
            "identity_count": record["identity_count"],
            "identity_sha256": record["identity_sha256"],
            "reused_across_modes_views_cases_repetitions": True,
            "subset_redraw_forbidden": True,
            "dependencies": [],
            "external_dependencies": [weight_declaration_reference(record["weight_id"])],
        }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        support = supports[record["support_id"]]
        binding_sha256 = record["contrast_binding_sha256"]
        for target_kind, fixed_key, candidate_key in (
            ("context", "fixed_context", "paired_context"),
            ("pooled", "fixed_pooled", "paired_pooled"),
        ):
            if target_kind == "context":
                target_ids = list(support["context_ids"])
            else:
                target_ids = list(support["complete_pool_group_ids"])
            yield {
                "logical_id": _support_job_id(contrast_id, target_kind),
                "stage": "support_assessment",
                "activation": "unconditional",
                "contrast_id": contrast_id,
                "target_kind": target_kind,
                "fixed_target_ids": target_ids,
                "fixed_target_count": record[fixed_key]["count"],
                "fixed_target_sha256": record[fixed_key]["sha256"],
                "paired_candidate_ids": target_ids,
                "paired_candidate_count": record[candidate_key]["count"],
                "paired_candidate_sha256": record[candidate_key]["sha256"],
                "paired_candidate_nonempty": record[candidate_key]["candidate_support_nonempty"],
                "whole_curve_requirements": {
                    "clean_case_repetition_seed_parity": True,
                    "dose_wise_intersection": False,
                },
                "dependencies": [],
                "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
            }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        for view_id in _VIEWS:
            target_kind = _VIEW_TARGET_KIND[view_id]
            yield {
                "logical_id": _point_job_id(contrast_id, view_id),
                "stage": "point_estimate",
                "activation": _VIEW_ACTIVATION[view_id],
                "contrast_id": contrast_id,
                "view_id": view_id,
                "target_kind": target_kind,
                "estimand": (
                    "absolute_clean_stressed_ba_curves_directional_areas_"
                    "stochastic_score_means_and_paired_area_descriptive_"
                    "domain_summaries"
                ),
                "pooled_four_fold_first": target_kind == "pooled",
                "cross_family_score_average": False,
                "interval_bounds_for_interactions": False,
                "per_dose_lowest_domain_ba_per_paired_pipeline": True,
                "minimum_paired_domain_effect": True,
                "weakest_domain_quantities_distinct": True,
                "weakest_domain_ties_retained": True,
                "weakest_domain_diagnostics_name": (
                    "all_required_procedure_and_policy_pair_weakest_domain_diagnostics"
                ),
                "bound_term_count": 2 if record["kind"] == "effect" else 4,
                "four_term_min_is_not_two_pipeline_comparison": (record["kind"] == "interaction"),
                "dependencies": [_support_job_id(contrast_id, target_kind)],
                "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
            }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        for view_id in _VIEWS:
            target_kind = _VIEW_TARGET_KIND[view_id]
            for mode in _WEIGHT_MODES:
                yield {
                    "logical_id": _parity_job_id(contrast_id, view_id, mode),
                    "stage": "unit_weight_parity",
                    "activation": _VIEW_ACTIVATION[view_id],
                    "contrast_id": contrast_id,
                    "view_id": view_id,
                    "mode": mode,
                    "tolerance": 1e-12,
                    "unused_factor_one": True,
                    "future_numeric_check_only": True,
                    "dependencies": [
                        _point_job_id(contrast_id, view_id),
                        _support_job_id(contrast_id, target_kind),
                        _auth_logical_id("master"),
                        _auth_logical_id("instrument"),
                    ],
                    "external_dependencies": [
                        binding_reference(contrast_id, binding_sha256),
                        weight_reference("master"),
                        weight_reference("instrument"),
                    ],
                }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        for view_id in _VIEWS:
            target_kind = _VIEW_TARGET_KIND[view_id]
            for mode in _WEIGHT_MODES:
                parity_id = _parity_job_id(contrast_id, view_id, mode)
                support_id = _support_job_id(contrast_id, target_kind)
                external = [
                    binding_reference(contrast_id, binding_sha256),
                    weight_reference("master"),
                    weight_reference("instrument"),
                ]
                for batch_index, (draw_start, draw_stop) in enumerate(ranges):
                    yield {
                        "logical_id": _batch_job_id(contrast_id, view_id, mode, batch_index),
                        "stage": "weighted_batch",
                        "activation": _VIEW_ACTIVATION[view_id],
                        "contrast_id": contrast_id,
                        "view_id": view_id,
                        "mode": mode,
                        "batch_index": batch_index,
                        "draw_start": draw_start,
                        "draw_stop": draw_stop,
                        "draw_count": draw_stop - draw_start,
                        "recomputation": (
                            "weighted_prediction_units_to_class_ba_to_curves_"
                            "areas_to_signed_contrast_within_draw"
                        ),
                        "fixed_noise_routes_calibration": True,
                        "hierarchy_rng": False,
                        "recompute_weakest_domain_minima_with_shared_weights": True,
                        "weakest_domain_minima_conditional_diagnostics_only": True,
                        "new_significance_family": False,
                        "dependencies": [
                            parity_id,
                            support_id,
                            _auth_logical_id("master"),
                            _auth_logical_id("instrument"),
                        ],
                        "external_dependencies": external,
                    }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        for view_id in _VIEWS:
            for mode in _WEIGHT_MODES:
                dependencies = [
                    _batch_job_id(contrast_id, view_id, mode, batch_index)
                    for batch_index in range(len(ranges))
                ]
                dependencies.append(_parity_job_id(contrast_id, view_id, mode))
                dependencies.append(_point_job_id(contrast_id, view_id))
                yield {
                    "logical_id": _weighted_summary_job_id(contrast_id, view_id, mode),
                    "stage": "weighted_summary",
                    "activation": _VIEW_ACTIVATION[view_id],
                    "contrast_id": contrast_id,
                    "view_id": view_id,
                    "mode": mode,
                    "required_draws": _TOTAL_DRAWS,
                    "quantiles": [0.025, 0.975],
                    "quantile_method": "linear",
                    "interval_kind": "marginal_conditional",
                    "failure_policy": (
                        "invalid_weights_or_nonfinite_failure_no_redraw_"
                        "no_partial_draw_average_no_bca"
                    ),
                    "dependencies": dependencies,
                    "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
                }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        yield {
            "logical_id": _hierarchy_prepare_job_id(contrast_id),
            "stage": "hierarchy_realization_prepare",
            "activation": "unconditional",
            "contrast_id": contrast_id,
            "view_id": "fixed_context",
            "seed": _HIERARCHY_SEED,
            "generator": "PCG64",
            "draw_count": _TOTAL_DRAWS,
            "reset_per_contrast": True,
            "domain_index_array_first": True,
            "domain_visit_order": "sorted_source_domains",
            "occurrence_visit_order": "row_major",
            "class_visit_order": "sorted_classes",
            "master_identity_order": "lexicographic",
            "class_draw_strategy": (
                "single_multinomial_per_domain_class_size_all_selected_occurrences"
            ),
            "all_realizations_before_score_batches": True,
            "counts_loaded": False,
            "dependencies": [_support_job_id(contrast_id, "context")],
            "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
        }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        prepare_id = _hierarchy_prepare_job_id(contrast_id)
        point_id = _point_job_id(contrast_id, "fixed_context")
        for batch_index, (draw_start, draw_stop) in enumerate(ranges):
            yield {
                "logical_id": _hierarchy_batch_job_id(contrast_id, batch_index),
                "stage": "hierarchy_batch",
                "activation": "unconditional",
                "contrast_id": contrast_id,
                "view_id": "fixed_context",
                "batch_index": batch_index,
                "draw_start": draw_start,
                "draw_stop": draw_stop,
                "draw_count": draw_stop - draw_start,
                "scope": "fixed_full_support_original_hierarchy",
                "preserves_original_context_class_cells": True,
                "undefined_draws_preserved": True,
                "paired_or_pooled_expansion": False,
                "dependencies": [prepare_id, point_id],
                "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
            }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        dependencies = [
            _hierarchy_batch_job_id(contrast_id, batch_index) for batch_index in range(len(ranges))
        ]
        yield {
            "logical_id": _hierarchy_summary_job_id(contrast_id),
            "stage": "hierarchy_summary",
            "activation": "unconditional",
            "contrast_id": contrast_id,
            "view_id": "fixed_context",
            "required_draws": _TOTAL_DRAWS,
            "interval_condition": "unconditional_only_if_all_10000_defined",
            "undefined_reason": "hierarchical_fixed_support_undefined",
            "surviving_draw_conditional_interval": False,
            "dependencies": dependencies,
            "external_dependencies": [],
        }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        domain_assignments = record["fixed_domain_sign_assignments"]
        instrument_assignments = record["fixed_instrument_sign_assignments"]
        for view_id in _SIGN_VIEWS:
            for sign_unit in _SIGN_UNITS:
                if sign_unit == "domain":
                    assignment_count = domain_assignments
                else:
                    assignment_count = instrument_assignments
                yield {
                    "logical_id": _sign_job_id(contrast_id, view_id, sign_unit),
                    "stage": "sign_sensitivity",
                    "activation": _VIEW_ACTIVATION[view_id],
                    "contrast_id": contrast_id,
                    "view_id": view_id,
                    "sign_unit": sign_unit,
                    "assignment_count": assignment_count,
                    "assignment_count_kind": (
                        "exact" if view_id == "fixed_context" else "upper_bound"
                    ),
                    "statistic": "absolute_equal_domain_mean_difference",
                    "include_observed_assignment": True,
                    "tolerance": 1e-12,
                    "instrument_sign_shared_across_station_domains": True,
                    "pooled_or_directional_tests": False,
                    "dependencies": [_point_job_id(contrast_id, view_id)],
                    "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
                }

    family_members = {}
    for record in contrast_records:
        family_members.setdefault(record["multiplicity_family"], []).append(record["contrast_id"])
    for multiplicity_family in sorted(_MULTIPLICITY_FAMILIES):
        members = sorted(family_members.get(multiplicity_family, []))
        for sign_unit in _SIGN_UNITS:
            yield {
                "logical_id": _holm_job_id(multiplicity_family, sign_unit),
                "stage": "holm_adjustment",
                "activation": "unconditional",
                "multiplicity_family": multiplicity_family,
                "sign_unit": sign_unit,
                "family_size": _MULTIPLICITY_FAMILIES[multiplicity_family],
                "slot_count": len(members),
                "dependencies": [
                    _sign_job_id(contrast_id, "fixed_context", sign_unit) for contrast_id in members
                ],
                "external_dependencies": [],
                "missing_policy": ("unavailable_estimate_and_raw_p_unavailable_bookkeeping_p_one"),
                "pooled_or_directional_family": False,
                "spans_all_six_disturbance_families": True,
                "authority": "none",
            }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        yield {
            "logical_id": _domain_stability_job_id(contrast_id),
            "stage": "domain_stability",
            "activation": "unconditional",
            "contrast_id": contrast_id,
            "view_id": "fixed_context",
            "quartile_method": "linear",
            "tolerance": 1e-12,
            "descriptive_only": True,
            "inference_promotion": False,
            "dependencies": [_point_job_id(contrast_id, "fixed_context")],
            "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
        }

    for record in contrast_records:
        contrast_id = record["contrast_id"]
        binding_sha256 = record["contrast_binding_sha256"]
        support = supports[record["support_id"]]
        targets = []
        for domain in sorted(support["domains"]):
            targets.append(("domain", domain))
        for instrument in sorted(support["instruments"]):
            targets.append(("instrument", instrument))
        for family in sorted(support["known_platform_families"]):
            targets.append(("known_platform_family", family))
        point_id = _point_job_id(contrast_id, "fixed_context")
        support_id = _support_job_id(contrast_id, "context")
        for deletion_kind, deletion_id in targets:
            yield {
                "logical_id": _deletion_job_id(contrast_id, deletion_kind, deletion_id),
                "stage": "deletion_stability",
                "activation": "unconditional",
                "contrast_id": contrast_id,
                "view_id": "fixed_context",
                "deletion_kind": deletion_kind,
                "deletion_id": deletion_id,
                "recompute": "equal_mean_without_refit",
                "empty_retained_domain": "undefined",
                "unknown_family_instruments_excluded_from_platform_family_deletions": True,
                "pooled_or_paired_expansion": False,
                "dependencies": [point_id, support_id],
                "external_dependencies": [binding_reference(contrast_id, binding_sha256)],
            }


def _iter_jobs(plan, catalog):
    catalog_hash = plan["source_contrast_catalog_sha256"]
    resolved = {}
    for raw in _iter_unbound_jobs(plan, catalog):
        logical_id = _require_identifier(raw.get("logical_id"))
        if logical_id in resolved:
            _fail()
        dependencies = raw.get("dependencies")
        if not isinstance(dependencies, list):
            _fail()
        mapped_dependencies = []
        for dependency in dependencies:
            mapped = resolved.get(dependency)
            if mapped is None:
                _fail()
            mapped_dependencies.append(mapped)
        body = {
            key: value for key, value in raw.items() if key not in ("logical_id", "dependencies")
        }
        body["source_contrast_catalog_sha256"] = catalog_hash
        body["dependencies"] = mapped_dependencies
        job_id = canonical_sha256(body)
        resolved[logical_id] = job_id
        job = dict(body)
        job["job_id"] = job_id
        yield _snapshot(job)


def iter_stress_inference_jobs(plan, *, contrast_catalog):
    """Validate eagerly and return a deterministic topological job iterator."""
    try:
        snapshot, catalog = _validated_pair(plan, contrast_catalog)
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None
    return _iter_jobs(snapshot, catalog)
