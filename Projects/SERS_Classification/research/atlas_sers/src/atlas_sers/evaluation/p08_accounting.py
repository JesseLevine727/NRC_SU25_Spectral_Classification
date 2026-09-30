"""Pure planning-slot accounting for the P08 universal policy panel.

This module is intentionally non-executable. It validates context and recipe
metadata and returns aggregate slot counts. It never loads data, fits models,
generates predictions, or authorizes execution.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["AccountingError", "plan_universal_accounting"]


class AccountingError(ValueError):
    """Malformed accounting input carrying a fixed, data-free ``reason_code``."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


_CONTEXT_KEYS = frozenset({"context_id", "selection_mode", "selection_unit_ids"})
_VALID_SELECTION_MODES = frozenset({"pseudo_domain", "master_cv"})
_VALID_RECIPES = frozenset({"D0-M", "D1", "D2", "D3"})
_RECIPE_ORDER = ("D0-M", "D1", "D2", "D3")
_MODEL_ORDER = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED")
_P05_MODEL_ACCOUNTING_BASIS = (
    "p05_selected_counts_are_additional_unique_neural_refits_beyond_d0_m_"
    "not_total_cost_or_selected_strategy_coverage"
)
_POLICY_ORDER = (
    ("PP-U-MIN", "R_MIN_400_1800", "existing_evidence_reuse_unverified"),
    ("PP-U-SG", "R_SG_400_1800", "new_policy_fitting_unapproved"),
    ("PP-U-ARPLS", "R_ARPLS_400_1800", "new_policy_fitting_unapproved"),
)
_LIMITATIONS = [
    "planning_slots_only_not_execution_authorization",
    "min_evidence_reuse_unverified_and_not_authenticated",
    "min_evidence_reuse_requires_complete_hashes_before_any_future_reuse",
    "sg_and_arpls_new_policy_fitting_unapproved",
    "source_cache_reuse_only_within_same_new_policy_candidate_context",
    "p05_selected_counts_are_additional_unique_neural_refits_beyond_d0_m_"
    "not_total_cost_or_selected_strategy_coverage",
    "no_family_qc_or_robustness_jobs_included",
    "no_retries_no_timings_no_storage_estimates",
    "does_not_complete_execution_ledger",
]


def _model_entry(
    model_id: str,
    *,
    source: int = 0,
    final: int = 0,
    calibration: int = 0,
    potential: int = 0,
    new_if_match: int = 0,
    scalar: int = 0,
) -> dict[str, int | str]:
    return {
        "model_id": model_id,
        "source_selection_fit_slots": source,
        "final_refit_slots": final,
        "calibration_model_fit_slots": calibration,
        "potentially_reusable_calibration_slots": potential,
        "new_calibration_model_fits_if_role_hashes_match": new_if_match,
        "scalar_calibration_operations": scalar,
        "source_fit_validation_prediction_jobs": source,
        "calibration_source_validation_prediction_jobs": calibration,
        "final_refit_test_prediction_jobs": final,
    }


def _build_models(
    context_count: int,
    selection_unit_count: int,
    master_cv_contexts: int,
    non_d0_contexts: int,
    non_d0_units: int,
) -> list[dict[str, int | str]]:
    pseudo_domain_contexts = context_count - master_cv_contexts
    entries = {
        "C-RBF-SVM": _model_entry(
            "C-RBF-SVM",
            source=selection_unit_count * 36,
            final=context_count,
            calibration=context_count * 3,
            potential=master_cv_contexts * 3,
            new_if_match=pseudo_domain_contexts * 3,
            scalar=context_count,
        ),
        "C-RANDOM-FOREST": _model_entry(
            "C-RANDOM-FOREST",
            source=selection_unit_count * 16 * 3,
            final=context_count * 3,
            calibration=context_count * 3 * 3,
            potential=master_cv_contexts * 3 * 3,
            new_if_match=pseudo_domain_contexts * 3 * 3,
            scalar=context_count,
        ),
        "C-EXTRA-TREES": _model_entry(
            "C-EXTRA-TREES",
            source=selection_unit_count * 16 * 3,
            final=context_count * 3,
            calibration=context_count * 3 * 3,
            potential=master_cv_contexts * 3 * 3,
            new_if_match=pseudo_domain_contexts * 3 * 3,
            scalar=context_count,
        ),
        "D0-M": _model_entry(
            "D0-M",
            source=selection_unit_count * 3,
            final=context_count * 3,
            scalar=context_count * 3,
        ),
        "P05-SELECTED": _model_entry(
            "P05-SELECTED",
            source=non_d0_units * 3,
            final=non_d0_contexts * 3,
            scalar=non_d0_contexts * 3,
        ),
    }
    return [entries[model_id] for model_id in _MODEL_ORDER]


def _policy_totals(
    models: list[dict[str, int | str]],
    context_count: int,
    non_d0_contexts: int,
    d0_contexts: int,
) -> dict[str, int]:
    literal_fit = 0
    literal_prediction = 0
    potential = 0
    scalar = 0
    for model in models:
        source = int(model["source_selection_fit_slots"])
        final = int(model["final_refit_slots"])
        calibration = int(model["calibration_model_fit_slots"])
        literal_fit += source + final + calibration
        potential += int(model["potentially_reusable_calibration_slots"])
        scalar += int(model["scalar_calibration_operations"])
        literal_prediction += (
            int(model["source_fit_validation_prediction_jobs"])
            + int(model["calibration_source_validation_prediction_jobs"])
            + int(model["final_refit_test_prediction_jobs"])
        )

    strategy_alias = 2 * context_count * 3
    unique_recipe = (context_count + non_d0_contexts) * 3
    duplicate_alias = d0_contexts * 3
    return {
        "literal_fit_slot_ceiling": literal_fit,
        "prospective_new_fit_slots_after_source_cache_match": literal_fit - potential,
        "scalar_calibration_operations": scalar,
        "literal_prediction_job_ceiling": literal_prediction,
        "prospective_prediction_jobs_after_source_cache_match": literal_prediction - potential,
        "strategy_alias_refit_slots": strategy_alias,
        "unique_recipe_refit_slots": unique_recipe,
        "same_context_same_recipe_alias_slots": duplicate_alias,
        "authorized_fits": 0,
        "retry_slots": 0,
    }


def _validate_contexts(contexts: object) -> tuple[list[dict[str, int | str]], set[str]]:
    if not isinstance(contexts, list | tuple):
        raise AccountingError("contexts_not_list_or_tuple")
    if not contexts:
        raise AccountingError("contexts_empty")

    normalized: list[dict[str, int | str]] = []
    seen_context_ids: set[str] = set()
    for raw in contexts:
        if not isinstance(raw, Mapping):
            raise AccountingError("context_not_mapping")
        if set(raw.keys()) != _CONTEXT_KEYS:
            raise AccountingError("context_keys_mismatch")

        context_id = raw["context_id"]
        if not isinstance(context_id, str):
            raise AccountingError("context_id_not_string")
        if context_id.strip() == "":
            raise AccountingError("context_id_blank")
        if context_id in seen_context_ids:
            raise AccountingError("context_id_duplicate")
        seen_context_ids.add(context_id)

        selection_mode = raw["selection_mode"]
        if not isinstance(selection_mode, str) or selection_mode not in _VALID_SELECTION_MODES:
            raise AccountingError("selection_mode_invalid")

        unit_ids = raw["selection_unit_ids"]
        if not isinstance(unit_ids, list | tuple):
            raise AccountingError("selection_unit_ids_not_list_or_tuple")
        if not unit_ids:
            raise AccountingError("selection_unit_ids_empty")

        seen_units: set[str] = set()
        for unit_id in unit_ids:
            if not isinstance(unit_id, str):
                raise AccountingError("selection_unit_id_not_string")
            if unit_id.strip() == "":
                raise AccountingError("selection_unit_id_blank")
            if unit_id in seen_units:
                raise AccountingError("selection_unit_id_duplicate")
            seen_units.add(unit_id)

        if selection_mode == "master_cv":
            if len(unit_ids) != 3:
                raise AccountingError("master_cv_unit_count_invalid")
        elif len(unit_ids) < 2:
            raise AccountingError("pseudo_domain_unit_count_invalid")

        normalized.append(
            {
                "context_id": context_id,
                "selection_mode": selection_mode,
                "unit_count": len(unit_ids),
            }
        )
    return normalized, seen_context_ids


def _validate_recipes(
    selected_recipes: object,
    seen_context_ids: set[str],
    mode_by_context: dict[str, str],
) -> dict[str, str]:
    if not isinstance(selected_recipes, Mapping):
        raise AccountingError("selected_recipes_not_mapping")
    if set(selected_recipes.keys()) != seen_context_ids:
        raise AccountingError("selected_recipes_keys_mismatch")

    recipe_by_context: dict[str, str] = {}
    for context_id, recipe in selected_recipes.items():
        if not isinstance(recipe, str) or recipe not in _VALID_RECIPES:
            raise AccountingError("selected_recipe_value_invalid")
        recipe_by_context[context_id] = recipe

    for context_id, mode in mode_by_context.items():
        if mode == "master_cv" and recipe_by_context[context_id] != "D0-M":
            raise AccountingError("master_cv_requires_d0_m")
    return recipe_by_context


def plan_universal_accounting(
    contexts: object,
    selected_recipes: object,
) -> dict[str, object]:
    """Validate metadata and return aggregate planning-slot counters.

    The returned report is aggregate-only and never authorizes execution.
    """

    normalized, seen_context_ids = _validate_contexts(contexts)
    mode_by_context = {ctx["context_id"]: ctx["selection_mode"] for ctx in normalized}
    recipe_by_context = _validate_recipes(selected_recipes, seen_context_ids, mode_by_context)

    context_count = len(normalized)
    selection_unit_count = sum(int(ctx["unit_count"]) for ctx in normalized)
    master_cv_contexts = sum(
        1 for ctx in normalized if ctx["selection_mode"] == "master_cv"
    )
    pseudo_domain_contexts = context_count - master_cv_contexts

    recipe_counts = {recipe: 0 for recipe in _RECIPE_ORDER}
    for recipe in recipe_by_context.values():
        recipe_counts[recipe] += 1
    non_d0_contexts = sum(1 for recipe in recipe_by_context.values() if recipe != "D0-M")
    non_d0_units = sum(
        int(ctx["unit_count"])
        for ctx in normalized
        if recipe_by_context[ctx["context_id"]] != "D0-M"
    )

    models = _build_models(
        context_count=context_count,
        selection_unit_count=selection_unit_count,
        master_cv_contexts=master_cv_contexts,
        non_d0_contexts=non_d0_contexts,
        non_d0_units=non_d0_units,
    )
    per_policy_totals = _policy_totals(
        models=models,
        context_count=context_count,
        non_d0_contexts=non_d0_contexts,
        d0_contexts=recipe_counts["D0-M"],
    )

    policy_reports = []
    for policy_id, representation_id, evidence_status in _POLICY_ORDER:
        policy_reports.append(
            {
                "policy_id": policy_id,
                "representation_id": representation_id,
                "evidence_status": evidence_status,
                "models": [dict(model) for model in models],
                "totals": dict(per_policy_totals),
            }
        )

    all_policy_totals = {
        key: sum(int(report["totals"][key]) for report in policy_reports)
        for key in per_policy_totals
    }
    nonminimal_reports = [
        report for report in policy_reports if report["policy_id"] != "PP-U-MIN"
    ]
    incremental_nonminimal_totals = {
        key: sum(int(report["totals"][key]) for report in nonminimal_reports)
        for key in per_policy_totals
    }

    return {
        "schema_version": 1,
        "scope": "universal_slot_accounting_not_execution_ledger",
        "authorized_for_execution": False,
        "authorized_model_fits": 0,
        "context_count": context_count,
        "selection_unit_count": selection_unit_count,
        "context_counts_by_selection_mode": {
            "master_cv": master_cv_contexts,
            "pseudo_domain": pseudo_domain_contexts,
        },
        "selected_recipe_counts": {
            recipe: recipe_counts[recipe] for recipe in _RECIPE_ORDER
        },
        "policy_reports": policy_reports,
        "all_policy_totals": all_policy_totals,
        "incremental_nonminimal_totals": incremental_nonminimal_totals,
        "model_accounting_basis": _P05_MODEL_ACCOUNTING_BASIS,
        "limitations": list(_LIMITATIONS),
    }
