"""P08 perturbation-inference declaration contract.

Metadata-only validation of the reviewed public perturbation-inference
registry, support audit and readiness contract.  These tests fit no model,
compute no weight, materialise no array and draw no resample; they check
declared panels, comparison grids, invariants, counts and SHA256 pins only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
from collections import Counter

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

REGISTRY = PACKAGE / "plan/contracts/p08_perturbation_inference.json"
READINESS = PACKAGE / "plan/contracts/p08_readiness_contract.json"
SUPPORT_AUDIT = PACKAGE / "results/p08_readiness/perturbation_inference_support_audit.json"
QC_PUBLIC_AUDIT = PACKAGE / "results/p08_readiness/qc_nested_support_audit.json"
DESIGN_REGISTRY = PACKAGE / "plan/contracts/p08_perturbation_design.json"
INHERITED_PROTOCOL = PACKAGE / "plan/P08_STATISTICAL_PROTOCOL.md"
POLICY_CONTRACT = PACKAGE / "plan/contracts/preprocessing_policy_contract.json"
RANGE_REGISTRY = PACKAGE / "plan/contracts/p08_range_inference.json"

UNIVERSAL_MODELS = ["C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED"]
ADAPTIVE_MODELS = ["C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED"]
NEURAL_MODELS = ["D0-M", "P05-SELECTED"]
UNIVERSAL_CLASSICAL_MODELS = ["C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES"]
ADAPTIVE_CLASSICAL_MODELS = ["C-RBF-SVM", "C-RANDOM-FOREST"]
DISTURBANCE_FAMILIES = ["shift", "slope", "quadratic", "gaussian", "impulse", "clipping"]
UNIVERSAL_POLICIES = ["PP-U-SG", "PP-U-ARPLS"]
QC_POLICY = "PP-QC-SRC"
REFERENCE_POLICY = "PP-U-MIN"
ENDPOINTS = ["M01", "M06"]

CONTRAST_FAMILIES = {
    "universal_robustness_effects": 120,
    "operational_qc_robustness_effects": 48,
    "operational_robustness_model_interactions": 192,
    "eligible_qc_robustness_effects": 48,
    "eligible_qc_robustness_model_interactions": 48,
}

SUPPORT_FIELD_MAP = {
    "contexts": "contexts",
    "domains": "domains",
    "instruments": "instruments",
    "held_spectra": "distinct_test_spectra",
    "masters": "distinct_test_masters",
    "spectrum_appearances": "test_spectrum_appearances",
    "master_context_units": "master_context_prediction_units",
    "complete_four_fold_groups": "complete_four_fold_domain_repeat_groups",
    "complete_four_fold_contexts": "complete_four_fold_contexts",
    "complete_four_fold_spectrum_appearances": "complete_four_fold_test_appearances",
    "domain_sign_assignments": "exhaustive_domain_sign_assignments",
    "instrument_sign_assignments": "exhaustive_instrument_sign_assignments",
}

EXPECTED_OPERATIONAL = {
    "contexts": 260,
    "domains": 13,
    "instruments": 10,
    "held_spectra": 557,
    "masters": 69,
    "spectrum_appearances": 2785,
    "master_context_units": 1310,
    "complete_four_fold_groups": 65,
    "complete_four_fold_contexts": 260,
    "complete_four_fold_spectrum_appearances": 2785,
    "domain_sign_assignments": 8192,
    "instrument_sign_assignments": 1024,
}

EXPECTED_ELIGIBLE = {
    "contexts": 54,
    "domains": 3,
    "instruments": 3,
    "held_spectra": 176,
    "masters": 24,
    "spectrum_appearances": 758,
    "master_context_units": 293,
    "complete_four_fold_groups": 9,
    "complete_four_fold_contexts": 36,
    "complete_four_fold_spectrum_appearances": 528,
    "domain_sign_assignments": 8,
    "instrument_sign_assignments": 8,
}

SHARED_WEIGHT_KEYS = (
    "draws",
    "generator",
    "master_seed",
    "instrument_seed",
    "distribution",
    "share_existing_global_P08_weight_arrays",
    "maximum_batch_size",
    "unit_weight_absolute_tolerance",
    "quantiles",
    "quantile_method",
    "includes_retraining_uncertainty",
    "bca_available",
    "intervals_simultaneous",
    "master_array_shape",
    "instrument_array_shape",
    "array_dtype",
    "array_axes",
)

QC_BOUND_SCOPE = [
    "operational_qc_effects",
    "operational_qc_interactions",
    "eligible_qc_effects",
    "eligible_qc_interactions",
]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def registry():
    return _load(REGISTRY)


@pytest.fixture(scope="module")
def readiness():
    return _load(READINESS)


@pytest.fixture(scope="module")
def audit():
    return _load(SUPPORT_AUDIT)


@pytest.fixture(scope="module")
def qc_public():
    return _load(QC_PUBLIC_AUDIT)


@pytest.fixture(scope="module")
def range_registry():
    return _load(RANGE_REGISTRY)


def _base(cid, kind, panel, support, family, policy, endpoint, family_name):
    return {
        "contrast_id": cid,
        "kind": kind,
        "panel": panel,
        "support": support,
        "disturbance_family": family,
        "policy": policy,
        "endpoint": endpoint,
        "multiplicity_family": family_name,
    }


def _expected_contrasts(registry):
    families = registry["disturbance_families"]
    u_policies = registry["comparison_universal_policies"]
    qc_policies = [registry["comparison_qc_policy"]]
    universal = registry["universal_models"]
    adaptive = registry["adaptive_models"]
    neural = registry["neural_models"]
    u_classical = registry["universal_classical_models"]
    a_classical = registry["adaptive_classical_models"]
    endpoints = registry["endpoints"]
    out = []
    for index, (fam, pol, method, endpoint) in enumerate(
        itertools.product(families, u_policies, universal, endpoints), 1
    ):
        record = _base(
            f"S-UE{index:03d}",
            "effect",
            "universal",
            "operational_260",
            fam,
            pol,
            endpoint,
            "universal_robustness_effects",
        )
        record["method"] = method
        out.append(record)
    for index, (fam, pol, method, endpoint) in enumerate(
        itertools.product(families, qc_policies, adaptive, endpoints), 1
    ):
        record = _base(
            f"S-QOE{index:03d}",
            "effect",
            "qc_fixed_route",
            "operational_260",
            fam,
            pol,
            endpoint,
            "operational_qc_robustness_effects",
        )
        record["method"] = method
        out.append(record)
    for index, (fam, pol, neu, classical, endpoint) in enumerate(
        itertools.product(families, u_policies, neural, u_classical, endpoints), 1
    ):
        record = _base(
            f"S-UI{index:03d}",
            "interaction",
            "universal",
            "operational_260",
            fam,
            pol,
            endpoint,
            "operational_robustness_model_interactions",
        )
        record["neural"] = neu
        record["classical"] = classical
        out.append(record)
    for index, (fam, pol, neu, classical, endpoint) in enumerate(
        itertools.product(families, qc_policies, neural, a_classical, endpoints), 1
    ):
        record = _base(
            f"S-QOI{index:03d}",
            "interaction",
            "qc_fixed_route",
            "operational_260",
            fam,
            pol,
            endpoint,
            "operational_robustness_model_interactions",
        )
        record["neural"] = neu
        record["classical"] = classical
        out.append(record)
    for index, (fam, pol, method, endpoint) in enumerate(
        itertools.product(families, qc_policies, adaptive, endpoints), 1
    ):
        record = _base(
            f"S-QSE{index:03d}",
            "effect",
            "qc_fixed_route",
            "qc_eligible_54",
            fam,
            pol,
            endpoint,
            "eligible_qc_robustness_effects",
        )
        record["method"] = method
        out.append(record)
    for index, (fam, pol, neu, classical, endpoint) in enumerate(
        itertools.product(families, qc_policies, neural, a_classical, endpoints), 1
    ):
        record = _base(
            f"S-QSI{index:03d}",
            "interaction",
            "qc_fixed_route",
            "qc_eligible_54",
            fam,
            pol,
            endpoint,
            "eligible_qc_robustness_model_interactions",
        )
        record["neural"] = neu
        record["classical"] = classical
        out.append(record)
    return out


def test_registry_metadata(registry):
    assert registry["schema_version"] == "nato-sers-p08-perturbation-inference-v1"
    assert registry["protocol_version"] == "nato-sers-p08-perturbation-inference-20261007-v1"
    assert registry["date"] == "2026-10-07"
    assert registry["status"] == "pre_outcome_numerical_specification_implementation_review_pending"


def test_registry_pins_protocol_design_and_audit(registry):
    assert registry["protocol"] == "plan/P08_PERTURBATION_INFERENCE.md"
    assert registry["inherited_protocol"] == "plan/P08_STATISTICAL_PROTOCOL.md"
    assert registry["design_registry"] == "plan/contracts/p08_perturbation_design.json"
    assert (
        registry["support_audit"]
        == "results/p08_readiness/perturbation_inference_support_audit.json"
    )
    assert registry["inherited_protocol_sha256"] == _sha(INHERITED_PROTOCOL)
    assert registry["design_registry_sha256"] == _sha(DESIGN_REGISTRY)
    assert registry["support_audit_sha256"] == _sha(SUPPORT_AUDIT)


def test_support_audit_pins_public_qc_support(audit, qc_public):
    hashes = audit["input_hashes"]
    assert hashes["qc_public_support"] == _sha(QC_PUBLIC_AUDIT)
    assert hashes["qc_private_support"] == qc_public["private_archive_sha256"]


def test_panels_families_policies_and_scope(registry):
    assert registry["universal_models"] == UNIVERSAL_MODELS
    assert registry["adaptive_models"] == ADAPTIVE_MODELS
    assert registry["neural_models"] == NEURAL_MODELS
    assert registry["universal_classical_models"] == UNIVERSAL_CLASSICAL_MODELS
    assert registry["adaptive_classical_models"] == ADAPTIVE_CLASSICAL_MODELS
    assert registry["disturbance_families"] == DISTURBANCE_FAMILIES
    assert registry["comparison_universal_policies"] == UNIVERSAL_POLICIES
    assert registry["comparison_qc_policy"] == QC_POLICY
    assert registry["reference_policy"] == REFERENCE_POLICY
    assert registry["endpoints"] == ENDPOINTS
    assert registry["scope_approvals"] == ["P08-A10", "P08-A11"]


def test_comparison_policy_ids_in_policy_contract(readiness):
    contract = _load(POLICY_CONTRACT)
    assert contract["qc_adaptive_policy"]["policy_id"] == QC_POLICY
    assert readiness["universal_policies"] == [REFERENCE_POLICY, *UNIVERSAL_POLICIES]
    assert set(contract["candidate_actions"]) == {
        "R_MIN_400_1800",
        "R_SG_400_1800",
        "R_ARPLS_400_1800",
    }


def test_contrasts_complete_ordered_and_unique(registry):
    expected = _expected_contrasts(registry)
    assert len(expected) == 456
    assert registry["contrasts"] == expected
    ids = [record["contrast_id"] for record in registry["contrasts"]]
    assert len(set(ids)) == len(ids) == 456
    counts = Counter(record["multiplicity_family"] for record in registry["contrasts"])
    assert dict(counts) == CONTRAST_FAMILIES
    assert registry["multiplicity"]["families"] == CONTRAST_FAMILIES
    assert registry["multiplicity"]["total_contrasts"] == 456


def test_supports_match_audit(registry, audit):
    for name, expected in (
        ("operational_260", EXPECTED_OPERATIONAL),
        ("qc_eligible_54", EXPECTED_ELIGIBLE),
    ):
        declared = registry["supports"][name]
        audited = audit["supports"][name]
        for reg_field, aud_field in SUPPORT_FIELD_MAP.items():
            assert declared[reg_field] == expected[reg_field]
            assert audited[aud_field] == expected[reg_field]
    eligible = registry["supports"]["qc_eligible_54"]
    assert eligible["contexts_per_domain"] == 18
    assert eligible["stations"] == ["cwa"]
    qc_eligible = audit["supports"]["qc_eligible_54"]
    by_domain = qc_eligible["by_domain"]
    assert len(by_domain) == 3
    assert all(record["contexts"] == 18 for record in by_domain)
    assert sum(record["contexts"] for record in by_domain) == 54
    assert audit["supports"]["qc_eligible_54"]["stations"] == ["cwa"]


def test_support_audit_denies_numerical_work(audit):
    assert audit["execution_authorized"] is False
    assert audit["new_scientific_operations"] == 0
    assert type(audit["new_scientific_operations"]) is int
    assert audit["spectral_arrays_or_predictions_loaded"] is False
    assert audit["scores_or_resampling_draws_computed"] is False
    assert audit["input_bytes_unchanged"] is True
    assert audit["inputs_verified"] == 5
    assert audit["status"] == "pass"
    assert audit["scope"] == "authenticated_identity_and_role_metadata_only"
    assert audit["schema_version"] == "nato-sers-p08-perturbation-inference-support-v1"


def test_weighted_uncertainty_matches_range_and_new_rules(registry, range_registry):
    weighted = registry["weighted_uncertainty"]
    inherited = range_registry["weighted_uncertainty"]
    for key in SHARED_WEIGHT_KEYS:
        assert weighted[key] == inherited[key], key
    assert weighted["unit_weight_absolute_tolerance"] == 1e-12
    assert weighted["quantiles"] == [0.025, 0.975]
    assert weighted["owner_amendment"] == "P08-A02"
    assert weighted["subset_strategy"] == "index_global_sorted_identities_no_redraw"
    assert weighted["shared_across_all_doses_and_repetitions"] is True
    assert weighted["recompute_curves_areas_and_contrasts_within_each_draw"] is True
    assert weighted["factors"] == ["master_and_instrument", "master_only", "instrument_only"]
    assert weighted["includes_source_noise_estimation_uncertainty"] is False
    assert weighted["includes_monte_carlo_integration_uncertainty"] is False


def test_curve_estimator_and_fixed_realization_rules(registry):
    curve = registry["curve_estimator"]
    assert curve["loss"] == "clean_BA_minus_mean_replicate_stressed_BA"
    assert curve["stochastic_repetitions_required"] == 10
    assert curve["area"] == "trapezoid_over_declared_0_1_severity_axis"
    assert (
        curve["signed_area"] == "average_positive_negative_at_matched_absolute_dose_then_integrate"
    )
    assert curve["worst_direction"] == "larger_of_two_directional_loss_areas_not_pointwise_envelope"
    assert curve["noise_axis"] == "zero_no_noise_then_quantile_rank_divided_by_0.95"
    assert (
        registry["point_estimator"]
        == "equal_original_present_classes_then_repetitions_then_dose_area"
        "_then_equal_contexts_then_equal_domains"
    )
    assert curve["negative_losses_clipped"] is False
    assert curve["combine_disturbance_families"] is False
    assert curve["noise_axis_is_physical_amplitude"] is False
    assert registry["effect_direction"] == "MIN_loss_area_minus_policy_loss_area"
    assert registry["interaction_direction"] == (
        "neural_robustness_benefit_minus_classical_robustness_benefit"
    )
    assert registry["absolute_clean_and_stressed_BA_required"] is True
    assert registry["qc_mode"] == "fixed_clean_route_sensitivity"
    assert registry["native_grid_gate_reaction_included"] is False
    assert registry["clean_reference_shared_between_disturbance_families"] is True
    assert registry["recipe_identity_fixed_across_policies"] is True
    assert registry["source_noise_reestimated_during_inference"] is False
    assert registry["synthetic_realizations_resampled_during_inference"] is False
    assert registry["replicate_probability_ensemble_forbidden"] is True
    assert registry["pooled_four_fold_sensitivity"] is True
    assert registry["pool_across_repeats"] is False
    assert registry["pooled_requires_all_original_four_folds_within_fixed_support"] is True


def test_hierarchy_sign_multiplicity_rules(registry):
    hierarchy = registry["hierarchical_feasibility"]
    assert hierarchy["draws"] == 10000
    assert hierarchy["generator"] == "PCG64"
    assert hierarchy["seed"] == 2026093003
    assert hierarchy["reset_per_contrast_endpoint"] is True
    assert hierarchy["draw_order"] == "inherited_P06P11_section_4"
    assert hierarchy["carry_all_curve_cells_together"] is True
    assert hierarchy["empty_original_context_class_rule"] == "undefined_draw_no_drop_no_retry"
    assert hierarchy["unconditional_interval_requires_all_draws_defined"] is True

    signs = registry["sign_sensitivities"]
    assert signs["statistic"] == "absolute_equal_domain_effect_two_sided"
    assert signs["enumeration"] == "exhaustive_including_observed"
    assert signs["equality_tolerance"] == 1e-12
    assert signs["randomized_treatment_interpretation"] is False
    assert signs["qc_maximum_nonzero_domains"] == 3
    assert signs["qc_maximum_nonzero_instrument_identities"] == 3
    assert signs["qc_minimum_two_sided_raw_p"] == 2 / (2**3)
    assert signs["qc_minimum_two_sided_raw_p"] == 0.25
    assert signs["qc_resolution_bound_is_measured_p_value"] is False
    assert signs["qc_resolution_bound_applies_to"] == QC_BOUND_SCOPE

    multiplicity = registry["multiplicity"]
    assert multiplicity["procedure"] == "Holm"
    assert multiplicity["families"] == CONTRAST_FAMILIES
    assert multiplicity["total_contrasts"] == 456
    assert multiplicity["span_all_six_disturbance_families"] is True
    assert multiplicity["separate_sign_sensitivities"] == ["domain", "instrument_identity"]
    assert multiplicity["remove_structural_duplicates"] is False
    assert multiplicity["unavailable_adjustment_bookkeeping_p"] == 1
    assert multiplicity["unavailable_estimate_and_raw_p"] is None
    assert multiplicity["primary_family_sizes_changed"] is False
    assert multiplicity["pointwise_or_directional_or_pooled_extra_families"] is False


def test_missingness_rules(registry):
    missing = registry["missingness"]
    assert missing["full_target_requires_all_registered_contexts_and_rows"] is True
    assert missing["whole_curve_required_including_all_repetitions"] is True
    assert missing["interaction_requires_common_four_procedure_curve_support"] is True
    assert missing["paired_support_sensitivity_separately_labelled"] is True
    assert missing["dose_specific_support_intersections_forbidden"] is True
    assert missing["interpolate_missing_dose"] is False
    assert missing["average_incomplete_repetitions"] is False
    assert missing["average_incomplete_seeds"] is False
    assert missing["fallback_for_failed_universal_cell"] is False
    assert missing["structural_qc_fallback_contexts"] == 206
    assert missing["family_structural_alias_contexts"] == 260
    assert missing["invalid_selected_qc_action_uses_MIN_input_same_QC_estimator"] is True
    assert missing["invalid_MIN_input_fatal"] is True
    assert missing["repair_historical_C_SELECTED_missing_endpoints"] is False


def test_readiness_perturbation_design(readiness):
    design = readiness["perturbation_design"]
    assert design["full_inference_registry_complete"] is True
    assert design["inference_protocol"] == "plan/P08_PERTURBATION_INFERENCE.md"
    assert design["inference_registry"] == "plan/contracts/p08_perturbation_inference.json"
    assert design["inference_support_audit"] == (
        "results/p08_readiness/perturbation_inference_support_audit.json"
    )
    assert design["inference_implementation_accepted"] is False
    assert design["exact_prediction_reconstruction_ledger_complete"] is True
    assert design["stress_prediction_layer_metadata_verified"] is True
    assert design["stress_prediction_catalog_audit"] == (
        "results/p08_readiness/stress_prediction_catalog_audit.json"
    )
    assert design["stress_prediction_operation_descriptors"] == 2013605
    assert design["stress_prediction_reporting_aliases"] == 574080
    assert design["stress_combined_procedures"] == 3413
    assert design["stress_seed_estimator_slots"] == 8571
    assert design["full_stress_job_ledger_complete"] is False
    assert design["stress_scoring_inference_rendering_ledger_complete"] is False
    assert design["stress_clean_probability_parity_accepted"] is False
    assert design["finite_resource_proposal_complete"] is False
    assert design["execution_authorized"] is False


def test_registry_authority_flags_closed(registry):
    assert registry["execution_authorized"] is False
    assert registry["authorized_model_fits"] == 0
    assert registry["authorized_new_predictions"] == 0
    assert registry["authorized_resampling_draws"] == 0
    assert registry["authorized_perturbation_runs"] == 0
    for key in (
        "authorized_model_fits",
        "authorized_new_predictions",
        "authorized_resampling_draws",
        "authorized_perturbation_runs",
    ):
        assert type(registry[key]) is int
        assert registry[key] == 0
    assert registry["inference_implementation_accepted"] is False
    assert registry["perturbation_runtime_accepted"] is False
    assert registry["resource_proposal_approved"] is False
    assert registry["original_G4_pass_implied"] is False
    assert registry["no_native_grid_gate_reaction_claim"] is True
    assert registry["no_chemical_disentanglement_claim"] is True
