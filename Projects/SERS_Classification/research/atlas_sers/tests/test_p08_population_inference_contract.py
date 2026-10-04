"""P08-T187 conditional filtered-population inference declaration contract.

Metadata-only validation of the reviewed public population inference registry,
readiness declaration, slot audit and public support aggregate.  These tests fit
no model, score no spectrum, draw no resample and confer no execution authority;
they compare exact declared keys, Cartesian identifier bookkeeping and SHA256
pins across the reviewed artifacts only.  The inherited statistical protocol and
the normalization weight declaration are verified byte-for-byte / field-for-
field through their pinned public counterparts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

POPULATION_INFERENCE = PACKAGE / "plan/contracts/p08_population_inference.json"
NORMALIZATION_INFERENCE = PACKAGE / "plan/contracts/p08_normalization_inference.json"
READINESS_CONTRACT = PACKAGE / "plan/contracts/p08_readiness_contract.json"
SUPPORT_AUDIT = PACKAGE / "results/p08_readiness/population_inference_support_audit.json"
SLOT_LEDGER_AUDIT = PACKAGE / "results/p08_readiness/population_slot_ledger_audit.json"
STATISTICAL_PROTOCOL = PACKAGE / "plan/P08_STATISTICAL_PROTOCOL.md"
POPULATION_PROTOCOL = PACKAGE / "plan/P08_POPULATION_INFERENCE.md"
FIGURE_PLAN = PACKAGE / "plan/P08_FIGURE_PLAN.csv"

NOTES = "notes_clear_500"
MIRA = "mira1_excluded_575"
PRIMARY = "primary_598"

POPULATION_FRAMES = {
    NOTES: (11, 10, 69, 449),
    MIRA: (12, 9, 69, 534),
}

SUPPORT_NAMES = ("classical", "common", "eligible", "neural")

# (fixed contexts, pooled contexts, complete four-fold groups, incomplete groups)
SUPPORT_SIZES = {
    NOTES: {
        "classical": (215, 208, 52, 3),
        "common": (215, 208, 52, 3),
        "eligible": (220, 220, 55, 0),
        "neural": (216, 212, 53, 2),
    },
    MIRA: {
        "classical": (240, 240, 60, 0),
        "common": (240, 240, 60, 0),
        "eligible": (240, 240, 60, 0),
        "neural": (240, 240, 60, 0),
    },
}

# support -> {"fixed": (appearances, master units), "pooled": (appearances, units)}
SUPPORT_APPEARANCES = {
    NOTES: {
        "classical": {"fixed": (2135, 1033), "pooled": (2032, 993)},
        "common": {"fixed": (2135, 1033), "pooled": (2032, 993)},
        "eligible": {"fixed": (2245, 1065), "pooled": (2245, 1065)},
        "neural": {"fixed": (2152, 1040), "pooled": (2103, 1017)},
    },
    MIRA: {
        "classical": {"fixed": (2670, 1225), "pooled": (2670, 1225)},
        "common": {"fixed": (2670, 1225), "pooled": (2670, 1225)},
        "eligible": {"fixed": (2670, 1225), "pooled": (2670, 1225)},
        "neural": {"fixed": (2670, 1225), "pooled": (2670, 1225)},
    },
}

NOTES_FIXED_CONTEXT_SHA = {
    "classical": "e75ab84bfc273b93bb50a1272f0c903a90611a926d28e7aef61cc094b8c1729d",
    "common": "e75ab84bfc273b93bb50a1272f0c903a90611a926d28e7aef61cc094b8c1729d",
    "eligible": "d514014e79d5e553487303b1eb650991241fcb2a79b17428654de6ba8f934355",
    "neural": "1257ecced3817262c85499c8090ca8cc1797bc505137c2f44d8f482695f5d554",
}
NOTES_POOLED_CONTEXT_SHA = {
    "classical": "5086edfadefca756410575072cf854044c40a7cd4d6249f1de884ed791605fc9",
    "common": "5086edfadefca756410575072cf854044c40a7cd4d6249f1de884ed791605fc9",
    "eligible": "d514014e79d5e553487303b1eb650991241fcb2a79b17428654de6ba8f934355",
    "neural": "75d8f20cecf5d6c8bb4b2e0cc34883b9f351d32615d09192b700224816e7b4a9",
}
NOTES_GROUP_SHA = {
    "classical": "52568098d2c1b491469b1d336871644abcdbf681da6434cdd47ba9693d564d31",
    "common": "52568098d2c1b491469b1d336871644abcdbf681da6434cdd47ba9693d564d31",
    "eligible": "7435d80f1fbab6c313df409152e82f88735e60712f01e003231271d35506b5b6",
    "neural": "ca8cddc5c000951aa649563098e8f23e27f45ecb922b1fbb709be7e0d80fd97c",
}
MIRA_CONTEXT_SHA = "9f267636a2247c29b7d89393149dd6ef4268454ef3e320e64c34d7c044eced3c"
MIRA_GROUP_SHA = "106ed53b76bb194095cf98ad9c14052875efdf31379ae118578912f486cccaa5"


def _load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _population(declaration, population_id):
    for entry in declaration["populations"]:
        if entry["population_id"] == population_id:
            return entry
    raise AssertionError(f"population {population_id!r} not declared")


def _audit_population(audit, population_id):
    for entry in audit["populations"]:
        if entry["population_id"] == population_id:
            return entry
    raise AssertionError(f"population {population_id!r} not audited")


def _effect_ids(declaration, panel_key):
    panel = declaration["panels"][panel_key]
    template = declaration["contrasts"]["effect_id_template"]
    return [
        template.format(
            population_id=entry["population_id"],
            method=method,
            nonminimal_policy=policy,
            endpoint=endpoint,
        )
        for entry in declaration["populations"]
        for method in panel["methods"]
        for policy in declaration["nonminimal_policies"]
        for endpoint in declaration["endpoints"]
    ]


def _interaction_ids(declaration, panel_key):
    panel = declaration["panels"][panel_key]
    template = declaration["contrasts"]["interaction_id_template"]
    return [
        template.format(
            population_id=entry["population_id"],
            neural_method=neural,
            classical_method=classical,
            nonminimal_policy=policy,
            endpoint=endpoint,
        )
        for entry in declaration["populations"]
        for neural in declaration["neural_methods"]
        for classical in panel["classical_methods"]
        for policy in declaration["nonminimal_policies"]
        for endpoint in declaration["endpoints"]
    ]


@pytest.fixture(scope="module")
def population():
    return _load(POPULATION_INFERENCE)


@pytest.fixture(scope="module")
def normalization():
    return _load(NORMALIZATION_INFERENCE)


@pytest.fixture(scope="module")
def audit():
    return _load(SUPPORT_AUDIT)


@pytest.fixture(scope="module")
def readiness():
    return _load(READINESS_CONTRACT)


def test_schema_status_and_denied_execution(population):
    assert population["schema_version"] == "nato-sers-p08-population-inference-v1"
    assert population["date"] == "2026-10-04"
    assert population["status"] == (
        "conditional_pre_outcome_specification_panel_clarification_"
        "and_implementation_review_pending"
    )
    assert population["execution_authorized"] is False
    assert population["authorized_model_fits"] == 0
    assert population["authorized_new_predictions"] == 0
    assert population["authorized_resampling_draws"] == 0
    assert population["inference_implementation_accepted"] is False
    assert population["population_runtime_accepted"] is False
    assert population["resource_proposal_approved"] is False
    assert population["new_primary_hypothesis"] is False
    assert population["original_G4_pass_implied"] is False
    assert population["prior_minimal_benchmark_outcomes_seen"] is True
    assert population["protocol"] == "plan/P08_POPULATION_INFERENCE.md"
    assert population["inherited_protocol"] == "plan/P08_STATISTICAL_PROTOCOL.md"
    assert POPULATION_PROTOCOL.is_file()


def test_scope_is_conditional_not_approval(population):
    assert population["owner_selection_decision"] == "P08-A07"
    assert population["panel_scope_clarification_pending"] is True
    assert population["selected_panel"] is None
    assert (
        population["support_audit"]
        == "results/p08_readiness/population_inference_support_audit.json"
    )
    assert population["figure_id"] == "P08-F11"
    assert len(population["populations"]) == 2
    assert {entry["population_id"] for entry in population["populations"]} == {
        NOTES,
        MIRA,
    }


def test_protocol_and_audit_pins_are_byte_exact(population):
    assert _sha256(STATISTICAL_PROTOCOL) == population["inherited_protocol_sha256"]
    assert _sha256(SUPPORT_AUDIT) == population["support_audit_sha256"]
    assert _sha256(SLOT_LEDGER_AUDIT) == population["slot_audit_sha256"]


def test_support_audit_metadata_only_flags(audit):
    assert audit["schema_version"] == ("nato-sers-p08-population-inference-support-audit-v1")
    assert audit["status"] == "pass_metadata_only"
    assert audit["execution_authorized"] is False
    assert audit["new_scientific_operations"] == 0
    assert audit["verified_input_file_count"] == 32
    assert audit["all_input_bytes_unchanged"] is True
    assert audit["arrays_loaded"] is False
    assert audit["scores_or_probabilities_loaded"] is False
    assert audit["statistical_inference_executed"] is False
    assert audit["prediction_content_validated"] is False
    assert set(audit["primary_input_pins"]) == {"contexts", "manifest", "roles"}


def test_support_audit_population_ids(audit):
    assert {entry["population_id"] for entry in audit["populations"]} == {
        PRIMARY,
        NOTES,
        MIRA,
    }


def test_panels_and_model_identifiers(population):
    panels = population["panels"]
    assert set(panels) == {"four_method", "five_method"}
    assert panels["four_method"]["methods"] == [
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
        "D0-M",
        "P05-SELECTED",
    ]
    assert panels["five_method"]["methods"] == [
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
        "C-EXTRA-TREES",
        "D0-M",
        "P05-SELECTED",
    ]
    assert panels["four_method"]["classical_methods"] == [
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
    ]
    assert panels["five_method"]["classical_methods"] == [
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
        "C-EXTRA-TREES",
    ]
    assert "C-EXTRA-TREES" not in panels["four_method"]["methods"]
    assert "C-EXTRA-TREES" in panels["five_method"]["methods"]
    assert set(panels["four_method"]["methods"]) < set(panels["five_method"]["methods"])


def test_neural_methods_policies_and_endpoints(population):
    assert population["neural_methods"] == ["D0-M", "P05-SELECTED"]
    assert len(population["neural_methods"]) == 2
    assert population["policies"] == ["PP-U-MIN", "PP-U-SG", "PP-U-ARPLS"]
    assert population["reference_policy"] == "PP-U-MIN"
    assert population["nonminimal_policies"] == ["PP-U-SG", "PP-U-ARPLS"]
    assert set(population["neural_methods"]) <= set(population["panels"]["five_method"]["methods"])
    assert population["endpoints"] == ["M01", "M06"]


def test_population_frame_identities(population):
    for population_id, frame in POPULATION_FRAMES.items():
        entry = _population(population, population_id)
        assert set(entry["support_sets"]) == set(SUPPORT_NAMES)
        domains, instruments, masters, spectra = frame
        for block in entry["support_sets"].values():
            for tier in ("fixed", "pooled"):
                tier_frame = block[tier]
                assert tier_frame["domains"] == domains
                assert tier_frame["instruments"] == instruments
                assert tier_frame["physical_masters"] == masters
                assert tier_frame["distinct_spectra"] == spectra


def test_population_support_context_counts(population):
    for population_id, table in SUPPORT_SIZES.items():
        entry = _population(population, population_id)
        for name, (fixed, pooled, complete, incomplete) in table.items():
            block = entry["support_sets"][name]
            assert block["fixed"]["contexts"] == fixed
            assert block["pooled"]["contexts"] == pooled
            assert block["complete_domain_repeat_groups"] == complete
            assert block["incomplete_domain_repeat_groups"] == incomplete


def test_population_support_appearances_and_units(population):
    for population_id, table in SUPPORT_APPEARANCES.items():
        entry = _population(population, population_id)
        for name, tiers in table.items():
            block = entry["support_sets"][name]
            for tier in ("fixed", "pooled"):
                appearances, units = tiers[tier]
                assert block[tier]["spectrum_context_appearances"] == appearances
                assert block[tier]["master_context_units"] == units


def test_context_and_group_hashes(population):
    notes = _population(population, NOTES)["support_sets"]
    for name in SUPPORT_NAMES:
        assert notes[name]["fixed"]["context_set_sha256"] == (NOTES_FIXED_CONTEXT_SHA[name])
        assert notes[name]["pooled"]["context_set_sha256"] == (NOTES_POOLED_CONTEXT_SHA[name])
        assert notes[name]["complete_group_set_sha256"] == NOTES_GROUP_SHA[name]
    for name in SUPPORT_NAMES:
        block = _population(population, MIRA)["support_sets"][name]
        assert block["fixed"]["context_set_sha256"] == MIRA_CONTEXT_SHA
        assert block["pooled"]["context_set_sha256"] == MIRA_CONTEXT_SHA
        assert block["complete_group_set_sha256"] == MIRA_GROUP_SHA


def test_common_support_equals_classical(population):
    for population_id in (NOTES, MIRA):
        supports = _population(population, population_id)["support_sets"]
        assert supports["common"] == supports["classical"]
    notes = _population(population, NOTES)["support_sets"]
    assert notes["neural"]["fixed"]["contexts"] != notes["classical"]["fixed"]["contexts"]
    assert notes["neural"]["pooled"]["contexts"] != notes["classical"]["pooled"]["contexts"]


def test_pooling_requires_complete_four_folds(population):
    for entry in population["populations"]:
        for block in entry["support_sets"].values():
            complete = block["complete_domain_repeat_groups"]
            assert block["pooled"]["contexts"] == complete * 4
            assert block["fixed"]["contexts"] >= block["pooled"]["contexts"]
    pooled = population["pooled_sensitivity"]
    assert pooled["require_complete_four_folds"] is True
    assert pooled["pool_across_repeats"] is False
    assert pooled["additional_hypothesis_family"] is False


def test_declaration_support_matches_public_audit(population, audit):
    for entry in population["populations"]:
        audited = _audit_population(audit, entry["population_id"])
        for key in (
            "population_plan_sha256",
            "neural_support_result_sha256",
            "private_primary_bridge_sha256",
        ):
            assert entry[key] == audited[key]
        for name, block in entry["support_sets"].items():
            audited_block = audited["support_sets"][name]
            assert (
                block["complete_domain_repeat_groups"]
                == audited_block["complete_domain_repeat_groups"]
            )
            assert (
                block["incomplete_domain_repeat_groups"]
                == audited_block["incomplete_domain_repeat_groups"]
            )
            assert block["complete_group_set_sha256"] == audited_block["complete_group_set_sha256"]
            for tier, audit_key in (
                ("fixed", "fixed_context_support"),
                ("pooled", "complete_four_fold_support"),
            ):
                audited_tier = audited_block[audit_key]
                audit_field_names = {
                    "physical_masters": "distinct_physical_masters",
                    "master_context_units": "master_context_prediction_units",
                }
                for field, value in block[tier].items():
                    audit_field = audit_field_names.get(field, field)
                    assert audited_tier[audit_field] == value


def test_primary_metadata_reference_unchanged(population, audit):
    primary = _audit_population(audit, PRIMARY)
    for block in primary["support_sets"].values():
        for tier in ("fixed_context_support", "complete_four_fold_support"):
            frame = block[tier]
            assert frame["contexts"] == 260
            assert frame["domains"] == 13
            assert frame["instruments"] == 10
            assert frame["distinct_physical_masters"] == 69
            assert frame["distinct_spectra"] == 557
    missingness = population["missingness"]
    assert (
        missingness[
            "primary_selected_classical_eight_missing_contexts_do_not_define_new_population_support"
        ]
        is True
    )
    assert missingness["repair_primary_missing_endpoints"] is False
    assert missingness["metadata_unsupported_contexts_retained_in_availability_report"] is True


def test_effect_family_is_cartesian_and_unique(population):
    assert population["contrasts"]["effect_axes"] == [
        "population_id",
        "method",
        "nonminimal_policy",
        "endpoint",
    ]
    assert population["contrasts"]["effect_id_template"] == (
        "POP-E/{population_id}/{method}/{nonminimal_policy}/{endpoint}"
    )
    for panel_key, size in (("four_method", 32), ("five_method", 40)):
        ids = _effect_ids(population, panel_key)
        assert len(ids) == size
        assert len(set(ids)) == size
        assert len(ids) == population["panels"][panel_key]["effect_family_size"]
    assert set(_effect_ids(population, "four_method")) <= set(
        _effect_ids(population, "five_method")
    )


def test_interaction_family_is_cartesian_and_unique(population):
    assert population["contrasts"]["interaction_axes"] == [
        "population_id",
        "neural_method",
        "classical_method",
        "nonminimal_policy",
        "endpoint",
    ]
    assert population["contrasts"]["interaction_id_template"] == (
        "POP-I/{population_id}/{neural_method}/{classical_method}/{nonminimal_policy}/{endpoint}"
    )
    for panel_key, size in (("four_method", 32), ("five_method", 48)):
        ids = _interaction_ids(population, panel_key)
        assert len(ids) == size
        assert len(set(ids)) == size
        assert len(ids) == population["panels"][panel_key]["interaction_family_size"]
    assert set(_interaction_ids(population, "four_method")) <= set(
        _interaction_ids(population, "five_method")
    )


def test_contrast_directions_and_common_support(population):
    contrasts = population["contrasts"]
    assert contrasts["effect_direction"] == "same_population_policy_minus_MIN"
    assert contrasts["classical_effect_support"] == "classical.fixed"
    assert contrasts["neural_effect_support"] == "neural.fixed"
    assert contrasts["interaction_direction"] == (
        "neural_policy_effect_minus_classical_policy_effect"
    )
    assert contrasts["interaction_support"] == "common.fixed"
    assert contrasts["interaction_recompute_all_four_cells_on_common_support"] is True
    assert contrasts["subtract_unmatched_method_means"] is False
    assert contrasts["paired_absolute_method_comparisons_use_common_support"] is True
    assert contrasts["point_estimator"] == (
        "equal_original_present_classes_then_equal_fixed_contexts_within_domain_then_equal_domains"
    )


def test_multiplicity_two_tiers_and_sign_families(population):
    multiplicity = population["multiplicity"]
    assert multiplicity["procedure"] == "Holm"
    assert multiplicity["scope"] == ("both_population_tiers_together_within_selected_panel")
    assert multiplicity["families"] == [
        "filtered_population_effects",
        "filtered_population_interactions",
    ]
    assert multiplicity["separate_sign_sensitivities"] == [
        "domain",
        "instrument_identity",
    ]
    assert multiplicity["remove_structural_duplicates"] is False
    assert multiplicity["unavailable_adjustment_bookkeeping_p"] == 1
    assert multiplicity["unavailable_estimate_and_raw_p"] is None
    assert multiplicity["primary_family_sizes_changed"] is False
    assert multiplicity["panel_selection_required_before_new_outcomes"] is True
    assert multiplicity["choose_panel_from_outcomes"] is False
    assert multiplicity["additional_complete_case_p_values_fill_registered_slots"] is False


def test_weighted_uncertainty_reuses_global_arrays(population, normalization):
    weighted = population["weighted_uncertainty"]
    reference = normalization["weighted_uncertainty"]
    for key in (
        "owner_amendment",
        "draws",
        "distribution",
        "generator",
        "master_seed",
        "instrument_seed",
        "share_existing_global_P08_weight_arrays",
        "master_array_shape",
        "instrument_array_shape",
        "array_dtype",
        "array_axes",
        "subset_strategy",
        "factors",
        "maximum_batch_size",
        "unit_weight_absolute_tolerance",
        "quantiles",
        "quantile_method",
        "intervals_simultaneous",
        "includes_retraining_uncertainty",
        "bca_available",
    ):
        assert weighted[key] == reference[key]
    assert weighted["master_array_shape"] == [10000, 69]
    assert weighted["instrument_array_shape"] == [10000, 10]
    assert weighted["maximum_batch_size"] == 128
    assert weighted["unit_weight_absolute_tolerance"] == 1e-12
    assert weighted["quantiles"] == [0.025, 0.975]
    assert weighted["includes_source_selection_uncertainty"] is False
    assert weighted["includes_support_or_missingness_uncertainty"] is False


def test_hierarchy_seed_and_undefined_rule(population, normalization):
    hierarchy = population["hierarchical_feasibility"]
    assert hierarchy == normalization["hierarchical_feasibility"]
    assert hierarchy["seed"] == 2026093003
    assert hierarchy["empty_original_context_class_rule"] == ("undefined_draw_no_drop_no_retry")
    assert hierarchy["unconditional_interval_requires_all_draws_defined"] is True


def test_sign_assignment_counts_are_two_powers(population):
    seen = 0
    for entry in population["populations"]:
        for block in entry["support_sets"].values():
            for tier in ("fixed", "pooled"):
                frame = block[tier]
                assert frame["domain_sign_assignments"] == 2 ** frame["domains"]
                assert frame["instrument_sign_assignments"] == 2 ** frame["instruments"]
                seen += 1
    assert seen == 16
    signs = population["sign_sensitivities"]
    assert signs["enumeration"] == "exhaustive_including_observed"
    assert signs["equality_tolerance"] == 1e-12
    assert signs["instrument_sign_shared_across_stations"] is True
    assert signs["assignment_counts_source"] == ("each_population_and_support_definition")
    assert signs["randomized_treatment_interpretation"] is False


def test_population_source_only_selection(population):
    selection = population["source_selection"]
    assert selection["fresh_population_MIN_neural_recipe_selection"] is True
    assert selection["copy_primary_recipe_mapping"] is False
    assert selection["recipe_fixed_across_population_policies"] is True
    assert selection["source_only_classical_grid_retuned_per_policy"] is True
    assert selection["primary_or_population_held_outcomes_used_for_selection"] is False
    assert selection["primary_MIN_models_reused"] is False
    assert selection["structural_ordinary_fallback_requires_valid_selection_readiness"] is True
    assert selection["unknown_or_failed_selection_is_fallback"] is False
    assert selection["neural_source_calibration_uses_inherited_units_only"] is True


def test_missingness_declaration(population):
    missingness = population["missingness"]
    assert missingness["support_fixed_before_new_outcomes"] is True
    assert missingness["fixed_support_contrast_requires_every_registered_cell_and_row"] is True
    assert missingness["new_missing_cell_rule"] == ("fixed_support_contrast_unavailable")
    assert missingness["average_incomplete_seeds"] is False
    assert missingness["complete_case_sensitivity_separately_labelled"] is True
    assert (
        missingness[
            "complete_case_cross_policy_comparison_requires_common_MIN_SG_ARPLS_intersection"
        ]
        is True
    )
    assert missingness["metadata_unsupported_contexts_retained_in_availability_report"] is True
    assert missingness["missingness_assumed_random"] is False


def test_pooled_sensitivity_declaration(population):
    pooled = population["pooled_sensitivity"]
    assert pooled["enabled"] is True
    assert pooled["require_complete_four_folds"] is True
    assert pooled["source"] == "same_method_or_common_support_pooled_set"
    assert (
        pooled["within_domain_repeat_average_then_equal_retained_repeats_then_equal_domains"]
        is True
    )
    assert pooled["pool_across_repeats"] is False
    assert pooled["changes_class_weighting_and_may_change_support"] is True
    assert pooled["additional_hypothesis_family"] is False


def test_cross_population_descriptive_only(population):
    cross = population["cross_population"]
    assert cross["comparison"] == "descriptive_matched_test_subset_only"
    assert cross["pair_by"] == ["domain", "outer_repeat", "outer_fold"]
    assert cross["primary_predictions_restricted_to_exact_filtered_test_uids"] is True
    assert cross["M06_recomputed_from_same_retained_views"] is True
    assert cross["requires_complete_content_validated_primary_and_filtered_predictions"] is True
    assert cross["no_new_fit_or_prediction_jobs_added"] is True
    assert cross["adds_inference_family"] is False
    assert cross["additional_p_values_or_intervals"] is False
    assert cross["whole_population_mean_difference_is_matched_filter_effect"] is False
    assert cross["selection_and_source_information_may_change"] is True
    assert cross["causal_clean_chemistry_claim"] is False


def test_figure_plan_registers_f11_like_f10():
    assert FIGURE_PLAN.is_file()
    with FIGURE_PLAN.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    f10 = [row for row in rows if row["figure_id"] == "P08-F10"]
    f11 = [row for row in rows if row["figure_id"] == "P08-F11"]
    assert len(f10) == 1
    assert len(f11) == 1
    assert f11[0]["status"] == "planned_no_new_results"
    assert f11[0]["formats"] == f10[0]["formats"]


def test_prose_and_readiness_links(readiness):
    assert POPULATION_PROTOCOL.is_file()
    block = readiness["population_membership_prescreen"]
    assert block["conditional_population_inference_specified"] is True
    assert block["population_inference_locked"] is False
    assert block["population_resource_proposal_complete"] is False
    assert block["execution_authorized"] is False
    for link in (
        "plan/P08_POPULATION_INFERENCE.md",
        "plan/contracts/p08_population_inference.json",
        "results/p08_readiness/population_inference_support_audit.json",
    ):
        assert link in block.values()
    master_plan = PACKAGE / "plan/MASTER_PLAN.md"
    population_support = PACKAGE / "plan/P08_POPULATION_SUPPORT.md"
    readiness_prose = PACKAGE / "plan/P08_READINESS.md"
    for path in (master_plan, population_support, readiness_prose):
        assert "P08_POPULATION_INFERENCE" in path.read_text(encoding="utf-8")


def test_public_audit_provenance(audit):
    assert (
        _sha256(PACKAGE / "results/p08_readiness/population_role_support_audit.json")
        == audit["upstream_role_audit_sha256"]
    )
    assert (
        _sha256(PACKAGE / "results/p08_readiness/population_neural_support_audit.json")
        == audit["upstream_neural_audit_sha256"]
    )
    populations = audit["populations"]
    records = populations.values() if isinstance(populations, dict) else populations
    for population in records:
        assert population["all_260_primary_context_mappings_verified"] is True
        assert population["test_rows_equal_primary_intersect_population"] is True
