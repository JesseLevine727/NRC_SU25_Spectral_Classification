"""P08-T161 N2 normalization-inference declaration contract.

Metadata-only validation of the reviewed public N2 normalization inference
registry, readiness declaration, slot ledger audit and reference-support
aggregate.  These tests fit no model, score no spectrum, draw no resample and
confer no execution authority; they compare exact declared keys, counts and
SHA256 pins across the reviewed artifacts only.  The inherited statistical
protocol is verified byte-for-byte through its pinned hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

NORMALIZATION_INFERENCE = PACKAGE / "plan/contracts/p08_normalization_inference.json"
READINESS_CONTRACT = PACKAGE / "plan/contracts/p08_readiness_contract.json"
NORMALIZATION_INPUT_AUDIT = (
    PACKAGE / "results/p08_readiness/normalization_input_audit.json"
)
SLOT_LEDGER_AUDIT = (
    PACKAGE / "results/p08_readiness/normalization_slot_ledger_audit.json"
)
REFERENCE_SUPPORT_AUDIT = (
    PACKAGE / "results/p08_readiness/normalization_reference_support_audit.json"
)
STATISTICAL_PROTOCOL = PACKAGE / "plan/P08_STATISTICAL_PROTOCOL.md"
NORMALIZATION_PROTOCOL = PACKAGE / "plan/P08_NORMALIZATION_INFERENCE.md"
FIGURE_PLAN = PACKAGE / "plan/P08_FIGURE_PLAN.csv"

CONTROL_POLICIES = ["PP-NORM-SNV", "PP-NORM-VECTOR", "PP-NORM-AREA", "PP-NORM-D1"]
CONTROL_ACTIONS = [
    "R_SNV_400_1800",
    "R_VECTOR_400_1800",
    "R_AREA_400_1800",
    "R_D1_400_1800",
]
POLICY_ACTION = dict(zip(CONTROL_POLICIES, CONTROL_ACTIONS, strict=True))
ENDPOINTS = ["M01", "M06"]


def _load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def normalization():
    return _load(NORMALIZATION_INFERENCE)


@pytest.fixture(scope="module")
def readiness():
    return _load(READINESS_CONTRACT)


@pytest.fixture(scope="module")
def ledger():
    return _load(SLOT_LEDGER_AUDIT)


@pytest.fixture(scope="module")
def support():
    return _load(REFERENCE_SUPPORT_AUDIT)


def test_schema_protocol_and_denied_execution(normalization):
    assert normalization["schema_version"] == "nato-sers-p08-normalization-inference-v1"
    assert (
        normalization["protocol_version"]
        == "nato-sers-p08-normalization-inference-20261004-v1"
    )
    assert normalization["date"] == "2026-10-04"
    assert (
        normalization["status"]
        == "pre_outcome_numerical_specification_implementation_review_pending"
    )
    assert normalization["protocol"] == "plan/P08_NORMALIZATION_INFERENCE.md"
    assert normalization["inherited_protocol"] == "plan/P08_STATISTICAL_PROTOCOL.md"
    assert normalization["owner_decision"] == "P08-A06"
    assert normalization["execution_authorized"] is False
    assert normalization["authorized_model_fits"] == 0
    assert normalization["authorized_new_predictions"] == 0
    assert normalization["authorized_resampling_draws"] == 0
    assert normalization["inference_implementation_accepted"] is False
    assert normalization["normalization_runtime_accepted"] is False
    assert normalization["resource_proposal_approved"] is False
    assert normalization["new_primary_hypothesis"] is False
    assert normalization["original_G4_pass_implied"] is False
    assert normalization["prior_minimal_benchmark_outcomes_seen"] is True
    assert normalization["figure_id"] == "P08-F10"
    assert normalization["endpoints"] == ["M01", "M06"]
    assert NORMALIZATION_PROTOCOL.is_file()


def test_protocol_and_audit_pins_are_byte_exact(normalization):
    assert _sha256(STATISTICAL_PROTOCOL) == normalization["inherited_protocol_sha256"]
    assert _sha256(NORMALIZATION_INPUT_AUDIT) == normalization["input_audit_sha256"]
    assert _sha256(SLOT_LEDGER_AUDIT) == normalization["ledger_audit_sha256"]
    assert (
        _sha256(REFERENCE_SUPPORT_AUDIT)
        == normalization["reference_support_audit_sha256"]
    )


def test_plan_and_family_map_bind_ledger_and_support(
    normalization, ledger, support
):
    assert ledger["plan_sha256"] == normalization["normalization_plan_sha256"]
    assert (
        ledger["source_family_mapping_sha256"]
        == normalization["source_family_mapping_sha256"]
    )
    assert (
        support["normalization_slot_plan_sha256"]
        == normalization["normalization_plan_sha256"]
    )
    assert support["input_bindings_sha256"] == ledger["private_input_bindings_sha256"]
    assert support["execution_authorized"] is False
    assert support["new_scientific_operations"] == 0
    assert support["predictions_recomputed"] is False
    assert support["score_columns_loaded"] is False
    assert support["all_input_bytes_unchanged"] is True
    assert support["historical_missing_endpoints_repaired"] is False
    assert ledger["execution_authorized"] is False
    assert ledger["new_scientific_operations"] == 0
    assert ledger["input_bytes_unchanged"] is True
    assert ledger["historical_missing_endpoints_repaired"] is False


def test_control_policy_action_mapping(normalization):
    assert normalization["control_policies"] == CONTROL_POLICIES
    assert normalization["control_actions"] == CONTROL_ACTIONS
    assert normalization["selector_scope"] == "frozen_context_local_MIN_selected_family"
    assert normalization["reference_policy"] == "PP-U-MIN"
    assert normalization["reference_action"] == "R_MIN_400_1800"
    assert normalization["reference_method"] == "C-SELECTED"
    assert normalization["comparison_method"] == "N2-MIN-FAMILY-SOURCE-RETUNED"
    for effect in normalization["effects"]:
        assert effect["action"] == POLICY_ACTION[effect["policy"]]


def test_native_scaling_and_no_reselection(normalization):
    assert normalization["native_control_scaling_preserved"] is True
    assert normalization["append_minmax"] is False
    assert normalization["destructive_control_policy"] == "PP-NORM-D1"
    assert normalization["destructive_control_promotion_allowed"] is False
    assert normalization["classical_hyperparameters_reselected_source_only"] is True
    assert normalization["MIN_hyperparameters_or_estimators_reused"] is False
    assert normalization["family_reselection_allowed"] is False


def test_effects_cartesian_and_direction(normalization):
    effects = normalization["effects"]
    assert len(effects) == 8
    assert [effect["contrast_id"] for effect in effects] == [
        f"N2-E{index:02d}" for index in range(1, 9)
    ]
    assert len({effect["contrast_id"] for effect in effects}) == 8
    observed = {
        (effect["policy"], effect["action"], effect["endpoint"]) for effect in effects
    }
    expected = {
        (policy, POLICY_ACTION[policy], endpoint)
        for policy in CONTROL_POLICIES
        for endpoint in ENDPOINTS
    }
    assert observed == expected
    assert {effect["support"] for effect in effects} == {
        "fixed_reference_available_252"
    }
    assert normalization["interactions"] == []
    assert (
        normalization["effect_direction"]
        == "control_minus_historical_MIN_same_context_selected_family"
    )
    assert normalization["point_estimator"] == (
        "equal_original_present_classes_then_equal_fixed_reference_contexts"
        "_then_equal_domains"
    )
    assert normalization["full_260_context_paired_effect_available"] is False


def test_population_support_and_hashes(normalization, support):
    population = normalization["population"]
    reference = support["fixed_reference_available_support"]
    pooled = support["fixed_complete_four_fold_support"]
    assert population["primary_spectra"] == 598
    assert population["registered_held_contexts"] == 260
    assert population["historical_missing_contexts"] == 8
    assert population["fixed_reference_contexts"] == 252
    assert population["fixed_pooled_contexts"] == 228
    assert population["domains"] == 13
    assert population["physical_masters"] == 69
    assert population["instruments"] == 10
    assert population["held_distinct_spectra"] == 557
    assert population["complete_four_fold_groups"] == 57
    assert population["incomplete_four_fold_groups"] == 8
    assert population["context_support_fixed_before_new_control_outcomes"] is True
    assert support["primary_registered_support"]["contexts"] == 260
    assert reference["contexts"] == 252
    assert pooled["contexts"] == 228
    assert support["historical_missing_contexts"] == 8
    assert support["complete_domain_repeat_groups"] == 57
    assert support["incomplete_domain_repeat_groups"] == 8
    for frame in (reference, pooled):
        assert frame["domains"] == 13
        assert frame["distinct_physical_masters"] == 69
        assert frame["instruments"] == 10
        assert frame["distinct_spectra"] == 557
        assert frame["stations"] == 3
    assert (
        population["fixed_reference_context_set_sha256"]
        == reference["context_set_sha256"]
    )
    assert (
        population["fixed_pooled_context_set_sha256"]
        == pooled["context_set_sha256"]
    )
    assert (
        population["fixed_pooled_group_set_sha256"]
        == support["complete_four_fold_group_set_sha256"]
    )


def test_reference_and_pooled_units_agree(normalization, support):
    population = normalization["population"]
    reference = support["fixed_reference_available_support"]
    pooled = support["fixed_complete_four_fold_support"]
    assert population["fixed_reference_spectrum_context_appearances"] == 2714
    assert population["fixed_reference_master_context_units"] == 1268
    assert population["fixed_pooled_spectrum_context_appearances"] == 2499
    assert population["fixed_pooled_master_context_units"] == 1150
    assert reference["spectrum_context_appearances"] == 2714
    assert reference["master_context_prediction_units"] == 1268
    assert pooled["spectrum_context_appearances"] == 2499
    assert pooled["master_context_prediction_units"] == 1150


def test_missing_reference_does_not_drop_jobs(normalization):
    assert normalization["pooled_four_fold_sensitivity"] is True
    assert normalization["pooled_support_fixed_before_new_outcomes"] is True
    assert normalization["pool_across_repeats"] is False
    assert (
        normalization["missingness"]["drop_new_control_jobs_for_missing_reference"]
        is False
    )


def test_slot_ledger_totals_and_stages(ledger):
    summary = ledger["summary"]
    totals = summary["totals"]
    assert ledger["schema_version"] == "nato-sers-p08-n2-slot-ledger-audit-v1"
    assert ledger["decision_id"] == "P08-A06"
    assert ledger["status"] == "pass_metadata_only"
    assert ledger["context_count"] == 260
    assert ledger["new_control_contexts"] == 260
    assert ledger["historical_reference_available_contexts"] == 252
    assert ledger["historical_reference_missing_contexts"] == 8
    assert ledger["source_units_matched"] == 681
    assert ledger["master_calibration_role_matches"] == 396
    assert ledger["authenticated_input_files"] == 22
    assert ledger["new_scientific_operations"] == 0
    assert ledger["execution_authorized"] is False
    assert summary["authorized_fit_slots"] == 0
    assert summary["execution_authorized"] is False
    assert summary["historical_reference_unavailable_contexts"] == 8
    assert totals["total_jobs"] == 89632
    assert totals["model_fit_slots"] == 42368
    assert totals["scalar_calibrations"] == 1040
    assert totals["calibration_prediction_aliases"] == 1776
    stages = totals["stage_counts"]
    assert stages["source_fit"] == 39504
    assert stages["source_validation_prediction"] == 39504
    assert stages["calibration_model_fit"] == 1704
    assert stages["calibration_validation_prediction"] == 1704
    assert stages["calibration_prediction_alias"] == 1776
    assert stages["final_refit"] == 1160
    assert stages["held_prediction"] == 1160
    assert stages["seed_ensemble_prediction"] == 1040
    assert stages["select_hyperparameters"] == 1040
    assert stages["scalar_calibration"] == 1040


def test_slot_ledger_policy_breakdown(ledger):
    by_policy = ledger["summary"]["by_policy"]
    assert set(by_policy) == set(CONTROL_POLICIES)
    for entry in by_policy.values():
        assert entry["model_fit_slots"] == 10592
        assert entry["scalar_calibrations"] == 260
        assert entry["calibration_prediction_aliases"] == 444
        assert entry["total_jobs"] == 22408
        predictions = entry["prediction_stage_counts"]
        assert predictions["source_validation_prediction"] == 9876
        assert predictions["calibration_validation_prediction"] == 426
        assert predictions["held_prediction"] == 290
        assert predictions["seed_ensemble_prediction"] == 260
        stage_counts = entry["stage_counts"]
        assert stage_counts["source_fit"] == 9876
        assert stage_counts["calibration_model_fit"] == 426
        assert stage_counts["final_refit"] == 290
        assert stage_counts["select_hyperparameters"] == 260
    assert sum(entry["total_jobs"] for entry in by_policy.values()) == 89632
    assert sum(entry["model_fit_slots"] for entry in by_policy.values()) == 42368
    assert sum(entry["scalar_calibrations"] for entry in by_policy.values()) == 1040


def test_weighted_uncertainty_declaration(normalization):
    weighted = normalization["weighted_uncertainty"]
    assert weighted["owner_amendment"] == "P08-A02"
    assert weighted["draws"] == 10000
    assert weighted["distribution"] == "Exponential_mean_1"
    assert weighted["generator"] == "PCG64"
    assert weighted["master_seed"] == 2026093001
    assert weighted["instrument_seed"] == 2026093002
    assert weighted["share_existing_global_P08_weight_arrays"] is True
    assert weighted["master_array_shape"] == [10000, 69]
    assert weighted["instrument_array_shape"] == [10000, 10]
    assert weighted["array_dtype"] == "float64"
    assert weighted["array_axes"] == ["draw", "lexicographically_sorted_identity"]
    assert weighted["subset_strategy"] == "index_global_sorted_identities_no_redraw"
    assert weighted["factors"] == [
        "master_and_instrument",
        "master_only",
        "instrument_only",
    ]
    assert weighted["maximum_batch_size"] == 128
    assert weighted["unit_weight_absolute_tolerance"] == 1e-12
    assert weighted["quantiles"] == [0.025, 0.975]
    assert weighted["quantile_method"] == "linear"
    assert weighted["intervals_simultaneous"] is False
    assert weighted["includes_retraining_uncertainty"] is False
    assert weighted["bca_available"] is False
    assert weighted["includes_historical_missingness_uncertainty"] is False
    assert weighted["includes_family_selection_uncertainty"] is False


def test_hierarchical_feasibility_declaration(normalization):
    hierarchy = normalization["hierarchical_feasibility"]
    assert hierarchy["draws"] == 10000
    assert hierarchy["generator"] == "PCG64"
    assert hierarchy["seed"] == 2026093003
    assert hierarchy["reset_per_contrast_endpoint"] is True
    assert hierarchy["draw_order"] == "inherited_P06P11_section_4"
    assert hierarchy["empty_original_context_class_rule"] == (
        "undefined_draw_no_drop_no_retry"
    )
    assert hierarchy["unconditional_interval_requires_all_draws_defined"] is True


def test_multiplicity_and_sign_sensitivities(normalization):
    multiplicity = normalization["multiplicity"]
    assert multiplicity["procedure"] == "Holm"
    assert multiplicity["families"] == {"normalization_effects": 8}
    assert multiplicity["scope"] == "fixed_reference_available_252"
    assert multiplicity["separate_sign_sensitivities"] == [
        "domain",
        "instrument_identity",
    ]
    assert multiplicity["remove_structural_duplicates"] is False
    assert multiplicity["unavailable_adjustment_bookkeeping_p"] == 1
    assert multiplicity["unavailable_estimate_and_raw_p"] is None
    assert multiplicity["primary_family_sizes_changed"] is False
    assert multiplicity["pooled_sensitivity_adds_family"] is False
    assert (
        multiplicity["additional_complete_case_p_values_fill_registered_slots"]
        is False
    )
    signs = normalization["sign_sensitivities"]
    assert signs["statistic"] == "absolute_equal_domain_effect_two_sided"
    assert signs["enumeration"] == "exhaustive_including_observed"
    assert signs["equality_tolerance"] == 1e-12
    assert signs["fixed_support_domain_assignments"] == 8192
    assert signs["fixed_support_instrument_assignments"] == 1024
    assert signs["randomized_treatment_interpretation"] is False


def test_missingness_declaration(normalization):
    missingness = normalization["missingness"]
    assert missingness["required_support"] == "fixed_reference_available_252"
    assert (
        missingness["fixed_support_effect_requires_all_registered_contexts_and_rows"]
        is True
    )
    assert missingness["pooled_effect_requires_all_fixed_228_contexts"] is True
    assert missingness["new_missing_cell_rule"] == "fixed_support_contrast_unavailable"
    assert (
        missingness["further_complete_case_sensitivity_separately_labelled"] is True
    )
    assert (
        missingness[
            "cross_control_reduced_support_requires_common_five_cell_intersection"
        ]
        is True
    )
    assert missingness["average_incomplete_seeds"] is False
    assert missingness["minimal_fallback"] is False
    assert missingness["family_substitution"] is False
    assert missingness["repair_historical_C_SELECTED_missing_endpoints"] is False
    assert missingness["drop_new_control_jobs_for_missing_reference"] is False
    assert missingness["missingness_assumed_random"] is False


def test_readiness_alternative_declaration(readiness):
    nested = readiness["normalization_fixed_family_alternative"]
    assert nested["numerical_inference_lock_complete"] is True
    assert nested["inference_implementation_accepted"] is False
    assert nested["normalization_runtime_accepted"] is False
    assert nested["execution_authorized"] is False
    assert nested["approved"] is True
    assert nested["approval_scope"] == (
        "scientific_design_only_not_resource_or_execution"
    )
    assert nested["resource_proposal"]["approved"] is False
    assert nested["inference_protocol"] == "plan/P08_NORMALIZATION_INFERENCE.md"
    assert (
        nested["inference_registry"]
        == "plan/contracts/p08_normalization_inference.json"
    )
    assert (
        nested["reference_support_audit"]
        == "results/p08_readiness/normalization_reference_support_audit.json"
    )


def test_figure_plan_declares_p08_f10():
    assert FIGURE_PLAN.is_file()
    with FIGURE_PLAN.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["figure_id"] == "P08-F10"]
    assert len(matches) == 1
    row = matches[0]
    assert row["stage"] == "normalization"
    assert row["status"] == "planned_no_new_results"
    assert row["formats"] == "native_tikz;offline_html;vector_pdf;png"
