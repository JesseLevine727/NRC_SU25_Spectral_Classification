import copy
import json

import pytest

from atlas_sers.evaluation.p08_accounting import AccountingError, plan_universal_accounting


def _mixed_fixture():
    contexts = [
        {
            "context_id": "ctx-a",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["u1", "u2", "u3"],
        },
        {
            "context_id": "ctx-b",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["u4", "u5"],
        },
        {
            "context_id": "ctx-c",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["u6", "u7", "u8"],
        },
    ]
    recipes = {"ctx-a": "D0-M", "ctx-b": "D1", "ctx-c": "D2"}
    return contexts, recipes


def _expect_error(contexts, recipes):
    with pytest.raises(AccountingError) as excinfo:
        plan_universal_accounting(contexts, recipes)
    return excinfo.value.reason_code


def test_accounting_error_is_value_error():
    assert issubclass(AccountingError, ValueError)


def test_scope_and_schema_version():
    report = plan_universal_accounting(*_mixed_fixture())
    assert report["schema_version"] == 1
    assert report["scope"] == "universal_slot_accounting_not_execution_ledger"


def test_mixed_fixture_hand_arithmetic():
    contexts, recipes = _mixed_fixture()
    report = plan_universal_accounting(contexts, recipes)

    assert report["authorized_for_execution"] is False
    assert report["authorized_model_fits"] == 0
    assert report["context_count"] == 3
    assert report["selection_unit_count"] == 8
    assert report["context_counts_by_selection_mode"] == {
        "master_cv": 1,
        "pseudo_domain": 2,
    }
    assert report["selected_recipe_counts"] == {
        "D0-M": 1,
        "D1": 1,
        "D2": 1,
        "D3": 0,
    }
    assert len(report["policy_reports"]) == 3

    for policy in report["policy_reports"]:
        models = {model["model_id"]: model for model in policy["models"]}

        svm = models["C-RBF-SVM"]
        assert svm["source_selection_fit_slots"] == 288
        assert svm["final_refit_slots"] == 3
        assert svm["calibration_model_fit_slots"] == 9
        assert svm["potentially_reusable_calibration_slots"] == 3
        assert svm["new_calibration_model_fits_if_role_hashes_match"] == 6
        assert svm["scalar_calibration_operations"] == 3

        for model_id in ("C-RANDOM-FOREST", "C-EXTRA-TREES"):
            tree = models[model_id]
            assert tree["source_selection_fit_slots"] == 384
            assert tree["final_refit_slots"] == 9
            assert tree["calibration_model_fit_slots"] == 27
            assert tree["potentially_reusable_calibration_slots"] == 9
            assert tree["new_calibration_model_fits_if_role_hashes_match"] == 18
            assert tree["scalar_calibration_operations"] == 3

        d0 = models["D0-M"]
        assert d0["source_selection_fit_slots"] == 24
        assert d0["final_refit_slots"] == 9
        assert d0["calibration_model_fit_slots"] == 0
        assert d0["scalar_calibration_operations"] == 9

        p05 = models["P05-SELECTED"]
        assert p05["source_selection_fit_slots"] == 15
        assert p05["final_refit_slots"] == 6
        assert p05["calibration_model_fit_slots"] == 0
        assert p05["scalar_calibration_operations"] == 6

        totals = policy["totals"]
        assert totals["literal_fit_slot_ceiling"] == 1194
        assert totals["prospective_new_fit_slots_after_source_cache_match"] == 1173
        assert totals["scalar_calibration_operations"] == 24
        assert totals["literal_prediction_job_ceiling"] == 1194
        assert totals["prospective_prediction_jobs_after_source_cache_match"] == 1173
        assert totals["strategy_alias_refit_slots"] == 18
        assert totals["unique_recipe_refit_slots"] == 15
        assert totals["same_context_same_recipe_alias_slots"] == 3
        assert totals["authorized_fits"] == 0
        assert totals["retry_slots"] == 0

    assert report["all_policy_totals"]["literal_fit_slot_ceiling"] == 3582
    assert (
        report["all_policy_totals"]["prospective_new_fit_slots_after_source_cache_match"]
        == 3519
    )
    assert report["incremental_nonminimal_totals"]["literal_fit_slot_ceiling"] == 2388
    assert (
        report["incremental_nonminimal_totals"][
            "prospective_new_fit_slots_after_source_cache_match"
        ]
        == 2346
    )


def test_d0_alias_and_non_d0_nonalias_counts():
    contexts = [
        {
            "context_id": "m1",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["m1a", "m1b", "m1c"],
        },
        {
            "context_id": "m2",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["m2a", "m2b", "m2c"],
        },
        {
            "context_id": "p1",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["p1a", "p1b"],
        },
    ]
    recipes = {"m1": "D0-M", "m2": "D0-M", "p1": "D1"}
    report = plan_universal_accounting(contexts, recipes)
    totals = report["policy_reports"][0]["totals"]
    assert totals["strategy_alias_refit_slots"] == 18
    assert totals["unique_recipe_refit_slots"] == 12
    assert totals["same_context_same_recipe_alias_slots"] == 6

    contexts2 = [
        {
            "context_id": "p1",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["x1", "x2"],
        },
        {
            "context_id": "p2",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["y1", "y2"],
        },
        {
            "context_id": "p3",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["z1", "z2"],
        },
    ]
    recipes2 = {"p1": "D1", "p2": "D2", "p3": "D3"}
    report2 = plan_universal_accounting(contexts2, recipes2)
    totals2 = report2["policy_reports"][0]["totals"]
    assert totals2["strategy_alias_refit_slots"] == 18
    assert totals2["unique_recipe_refit_slots"] == 18
    assert totals2["same_context_same_recipe_alias_slots"] == 0


def test_all_master_fallback():
    contexts = [
        {
            "context_id": "m1",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["a1", "a2", "a3"],
        },
        {
            "context_id": "m2",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["b1", "b2", "b3"],
        },
        {
            "context_id": "m3",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["c1", "c2", "c3"],
        },
    ]
    recipes = {"m1": "D0-M", "m2": "D0-M", "m3": "D0-M"}
    report = plan_universal_accounting(contexts, recipes)
    totals = report["policy_reports"][0]["totals"]
    assert report["selected_recipe_counts"] == {"D0-M": 3, "D1": 0, "D2": 0, "D3": 0}
    assert totals["literal_fit_slot_ceiling"] == 1308
    assert totals["prospective_new_fit_slots_after_source_cache_match"] == 1245
    assert totals["scalar_calibration_operations"] == 18
    assert totals["strategy_alias_refit_slots"] == 18
    assert totals["unique_recipe_refit_slots"] == 9
    assert totals["same_context_same_recipe_alias_slots"] == 9


def test_arbitrary_valid_counts_not_hardcoded_260():
    contexts = [
        {
            "context_id": "m0",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["m0-a", "m0-b", "m0-c"],
        },
        {
            "context_id": "p0",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ["p0-a", "p0-b", "p0-c", "p0-d"],
        },
    ]
    recipes = {"m0": "D0-M", "p0": "D3"}
    report = plan_universal_accounting(contexts, recipes)
    assert report["context_count"] == 2
    assert report["selection_unit_count"] == 7
    totals = report["policy_reports"][0]["totals"]
    assert totals["literal_fit_slot_ceiling"] == 1022
    assert totals["prospective_new_fit_slots_after_source_cache_match"] == 1001
    assert totals["scalar_calibration_operations"] == 15


def test_order_invariance():
    contexts, recipes = _mixed_fixture()
    report = plan_universal_accounting(contexts, recipes)
    reordered = [
        {
            "context_id": contexts[2]["context_id"],
            "selection_mode": contexts[2]["selection_mode"],
            "selection_unit_ids": list(reversed(contexts[2]["selection_unit_ids"])),
        },
        {
            "context_id": contexts[0]["context_id"],
            "selection_mode": contexts[0]["selection_mode"],
            "selection_unit_ids": list(reversed(contexts[0]["selection_unit_ids"])),
        },
        {
            "context_id": contexts[1]["context_id"],
            "selection_mode": contexts[1]["selection_mode"],
            "selection_unit_ids": list(reversed(contexts[1]["selection_unit_ids"])),
        },
    ]
    reordered_recipes = {
        "ctx-c": recipes["ctx-c"],
        "ctx-a": recipes["ctx-a"],
        "ctx-b": recipes["ctx-b"],
    }
    assert plan_universal_accounting(reordered, reordered_recipes) == report


def test_inputs_not_mutated():
    contexts, recipes = _mixed_fixture()
    contexts_before = copy.deepcopy(contexts)
    recipes_before = copy.deepcopy(recipes)
    plan_universal_accounting(contexts, recipes)
    assert contexts == contexts_before
    assert recipes == recipes_before


def test_json_roundtrip():
    report = plan_universal_accounting(*_mixed_fixture())
    assert json.loads(json.dumps(report)) == report


def test_identifier_privacy():
    contexts, recipes = _mixed_fixture()
    report = plan_universal_accounting(contexts, recipes)
    serialized = json.dumps(report, sort_keys=True)
    for context in contexts:
        assert context["context_id"] not in serialized
        for unit_id in context["selection_unit_ids"]:
            assert unit_id not in serialized


def test_model_counters_sum_to_policy_totals_independently():
    report = plan_universal_accounting(*_mixed_fixture())
    for policy in report["policy_reports"]:
        literal = sum(
            model["source_selection_fit_slots"]
            + model["final_refit_slots"]
            + model["calibration_model_fit_slots"]
            for model in policy["models"]
        )
        assert literal == policy["totals"]["literal_fit_slot_ceiling"]

        scalar = sum(model["scalar_calibration_operations"] for model in policy["models"])
        assert scalar == policy["totals"]["scalar_calibration_operations"]

        predictions = sum(
            model["source_fit_validation_prediction_jobs"]
            + model["calibration_source_validation_prediction_jobs"]
            + model["final_refit_test_prediction_jobs"]
            for model in policy["models"]
        )
        assert predictions == policy["totals"]["literal_prediction_job_ceiling"]

        potential = sum(
            model["potentially_reusable_calibration_slots"] for model in policy["models"]
        )
        assert (
            literal - potential
            == policy["totals"]["prospective_new_fit_slots_after_source_cache_match"]
        )


def test_no_execution_flags_and_evidence_labels():
    report = plan_universal_accounting(*_mixed_fixture())
    assert report["authorized_for_execution"] is False
    assert report["authorized_model_fits"] == 0

    labels = {
        policy["policy_id"]: policy["evidence_status"] for policy in report["policy_reports"]
    }
    assert labels["PP-U-MIN"] == "existing_evidence_reuse_unverified"
    assert labels["PP-U-SG"] == "new_policy_fitting_unapproved"
    assert labels["PP-U-ARPLS"] == "new_policy_fitting_unapproved"

    for policy in report["policy_reports"]:
        assert policy["totals"]["authorized_fits"] == 0
        assert policy["totals"]["retry_slots"] == 0
    assert report["all_policy_totals"]["authorized_fits"] == 0
    assert report["incremental_nonminimal_totals"]["authorized_fits"] == 0


def test_limitations_are_fixed_and_nonempty():
    report = plan_universal_accounting(*_mixed_fixture())
    assert isinstance(report["limitations"], list)
    assert report["limitations"]
    joined = " ".join(report["limitations"])
    assert "not_execution_authorization" in joined
    assert "unverified" in joined
    assert "does_not_complete_execution_ledger" in joined


def test_contexts_not_list_or_tuple():
    assert _expect_error({"context_id": "a"}, {}) == "contexts_not_list_or_tuple"
    assert _expect_error("abc", {}) == "contexts_not_list_or_tuple"
    assert _expect_error(None, {}) == "contexts_not_list_or_tuple"


def test_contexts_empty():
    assert _expect_error([], {}) == "contexts_empty"


def test_context_not_mapping():
    assert _expect_error([["not", "mapping"]], {}) == "context_not_mapping"


def test_context_keys_mismatch():
    missing = [{"context_id": "a", "selection_mode": "master_cv"}]
    extra = [
        {
            "context_id": "a",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["x", "y", "z"],
            "extra": 1,
        }
    ]
    assert _expect_error(missing, {}) == "context_keys_mismatch"
    assert _expect_error(extra, {}) == "context_keys_mismatch"


def test_context_id_invalid():
    base = {"selection_mode": "master_cv", "selection_unit_ids": ["x", "y", "z"]}
    assert _expect_error([{**base, "context_id": 7}], {}) == "context_id_not_string"
    assert _expect_error([{**base, "context_id": True}], {}) == "context_id_not_string"
    assert _expect_error([{**base, "context_id": ""}], {}) == "context_id_blank"
    assert _expect_error([{**base, "context_id": "   "}], {}) == "context_id_blank"


def test_context_id_duplicate():
    contexts = [
        {
            "context_id": "dup",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["x", "y", "z"],
        },
        {
            "context_id": "dup",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["a", "b", "c"],
        },
    ]
    assert _expect_error(contexts, {"dup": "D0-M"}) == "context_id_duplicate"


def test_selection_mode_invalid():
    base = {"context_id": "a", "selection_unit_ids": ["x", "y", "z"]}
    assert _expect_error([{**base, "selection_mode": "other"}], {}) == "selection_mode_invalid"
    assert _expect_error([{**base, "selection_mode": 1}], {}) == "selection_mode_invalid"
    assert _expect_error([{**base, "selection_mode": True}], {}) == "selection_mode_invalid"


def test_selection_unit_ids_invalid_container():
    base = {"context_id": "a", "selection_mode": "master_cv"}
    bad_string = [{**base, "selection_unit_ids": "xyz"}]
    bad_set = [{**base, "selection_unit_ids": {"x"}}]
    empty = [{**base, "selection_unit_ids": []}]
    assert _expect_error(bad_string, {}) == "selection_unit_ids_not_list_or_tuple"
    assert _expect_error(bad_set, {}) == "selection_unit_ids_not_list_or_tuple"
    assert _expect_error(empty, {}) == "selection_unit_ids_empty"


def test_selection_unit_id_invalid():
    base = {"context_id": "a", "selection_mode": "master_cv"}
    assert (
        _expect_error([{**base, "selection_unit_ids": ["x", 1, "z"]}], {})
        == "selection_unit_id_not_string"
    )
    assert (
        _expect_error([{**base, "selection_unit_ids": ["x", True, "z"]}], {})
        == "selection_unit_id_not_string"
    )
    assert (
        _expect_error([{**base, "selection_unit_ids": ["x", "", "z"]}], {})
        == "selection_unit_id_blank"
    )
    assert (
        _expect_error([{**base, "selection_unit_ids": ["x", "x", "z"]}], {})
        == "selection_unit_id_duplicate"
    )


def test_count_requirements():
    master = {"context_id": "a", "selection_mode": "master_cv"}
    assert (
        _expect_error([{**master, "selection_unit_ids": ["x", "y"]}], {})
        == "master_cv_unit_count_invalid"
    )
    assert (
        _expect_error([{**master, "selection_unit_ids": ["x", "y", "z", "w"]}], {})
        == "master_cv_unit_count_invalid"
    )
    pseudo = {"context_id": "p", "selection_mode": "pseudo_domain"}
    assert (
        _expect_error([{**pseudo, "selection_unit_ids": ["x"]}], {})
        == "pseudo_domain_unit_count_invalid"
    )


def test_selected_recipes_types_and_keys():
    contexts = [
        {
            "context_id": "a",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["x", "y", "z"],
        }
    ]
    assert _expect_error(contexts, None) == "selected_recipes_not_mapping"
    assert _expect_error(contexts, []) == "selected_recipes_not_mapping"
    assert _expect_error(contexts, {}) == "selected_recipes_keys_mismatch"
    assert (
        _expect_error(contexts, {"a": "D0-M", "b": "D0-M"})
        == "selected_recipes_keys_mismatch"
    )


def test_selected_recipe_value_invalid():
    contexts = [
        {
            "context_id": "a",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["x", "y", "z"],
        }
    ]
    assert _expect_error(contexts, {"a": "D4"}) == "selected_recipe_value_invalid"
    assert _expect_error(contexts, {"a": "d0-m"}) == "selected_recipe_value_invalid"
    assert _expect_error(contexts, {"a": None}) == "selected_recipe_value_invalid"


def test_master_cv_requires_d0_m():
    contexts = [
        {
            "context_id": "a",
            "selection_mode": "master_cv",
            "selection_unit_ids": ["x", "y", "z"],
        }
    ]
    for recipe in ("D1", "D2", "D3"):
        assert _expect_error(contexts, {"a": recipe}) == "master_cv_requires_d0_m"


def test_policy_representation_ids_exact_frozen_mapping():
    report = plan_universal_accounting(*_mixed_fixture())
    mapping = {
        policy["policy_id"]: policy["representation_id"]
        for policy in report["policy_reports"]
    }
    assert mapping == {
        "PP-U-MIN": "R_MIN_400_1800",
        "PP-U-SG": "R_SG_400_1800",
        "PP-U-ARPLS": "R_ARPLS_400_1800",
    }
    assert [policy["policy_id"] for policy in report["policy_reports"]] == [
        "PP-U-MIN",
        "PP-U-SG",
        "PP-U-ARPLS",
    ]


def test_p05_counts_are_additional_and_not_total_cost_or_coverage():
    contexts = [
        {
            "context_id": f"m{i}",
            "selection_mode": "master_cv",
            "selection_unit_ids": [f"m{i}a", f"m{i}b", f"m{i}c"],
        }
        for i in range(3)
    ]
    recipes = {f"m{i}": "D0-M" for i in range(3)}
    report = plan_universal_accounting(contexts, recipes)

    assert report["model_accounting_basis"] == (
        "p05_selected_counts_are_additional_unique_neural_refits_beyond_d0_m_"
        "not_total_cost_or_selected_strategy_coverage"
    )
    assert "additional_unique" in report["model_accounting_basis"]
    assert "not_total_cost" in report["model_accounting_basis"]

    policy = report["policy_reports"][0]
    models = {model["model_id"]: model for model in policy["models"]}
    p05 = models["P05-SELECTED"]
    assert p05["source_selection_fit_slots"] == 0
    assert p05["final_refit_slots"] == 0
    assert p05["calibration_model_fit_slots"] == 0
    assert p05["scalar_calibration_operations"] == 0

    # Selected-strategy coverage still spans every context: D0-M is the
    # selected strategy for these contexts and is refit per context and family.
    assert report["context_count"] == 3
    assert report["selected_recipe_counts"]["D0-M"] == 3
    assert models["D0-M"]["final_refit_slots"] == 3 * 3


def test_tuple_contexts_and_units_with_cross_context_unit_reuse():
    contexts = (
        {
            "context_id": "t-master",
            "selection_mode": "master_cv",
            "selection_unit_ids": ("shared", "b", "c"),
        },
        {
            "context_id": "t-pseudo",
            "selection_mode": "pseudo_domain",
            "selection_unit_ids": ("shared", "e"),
        },
    )
    recipes = {"t-master": "D0-M", "t-pseudo": "D1"}
    report = plan_universal_accounting(contexts, recipes)
    assert report["context_count"] == 2
    assert report["selection_unit_count"] == 5
    assert report["selected_recipe_counts"] == {
        "D0-M": 1,
        "D1": 1,
        "D2": 0,
        "D3": 0,
    }


def _build_full_size_contexts():
    contexts: list[dict[str, object]] = []
    recipes: dict[str, str] = {}

    def add(prefix, recipe, mode, groups):
        index = 0
        for unit_count, repeats in groups:
            for _ in range(repeats):
                context_id = f"{prefix}-{index:03d}"
                index += 1
                contexts.append(
                    {
                        "context_id": context_id,
                        "selection_mode": mode,
                        "selection_unit_ids": [
                            f"{context_id}-u{j}" for j in range(unit_count)
                        ],
                    }
                )
                recipes[context_id] = recipe

    add("master", "D0-M", "master_cv", ((3, 132),))
    add("pd0m", "D0-M", "pseudo_domain", ((2, 69), (3, 20)))
    add("pd1", "D1", "pseudo_domain", ((2, 6), (3, 8)))
    add("pd2", "D2", "pseudo_domain", ((2, 11),))
    add("pd3", "D3", "pseudo_domain", ((2, 13), (3, 1)))
    return contexts, recipes


def test_full_size_synthetic_aggregate():
    contexts, recipes = _build_full_size_contexts()
    report = plan_universal_accounting(contexts, recipes)

    assert report["context_count"] == 260
    assert report["selection_unit_count"] == 681
    assert report["context_counts_by_selection_mode"] == {
        "master_cv": 132,
        "pseudo_domain": 128,
    }
    assert report["selected_recipe_counts"] == {
        "D0-M": 221,
        "D1": 14,
        "D2": 11,
        "D3": 14,
    }

    base = report["policy_reports"][0]["totals"]
    assert base["literal_fit_slot_ceiling"] == 100373
    assert base["prospective_new_fit_slots_after_source_cache_match"] == 97601
    assert base["scalar_calibration_operations"] == 1677
    assert base["unique_recipe_refit_slots"] == 897
    assert base["strategy_alias_refit_slots"] == 1560
    assert base["same_context_same_recipe_alias_slots"] == 663

    for policy in report["policy_reports"]:
        assert policy["totals"] == base

    assert report["all_policy_totals"] == {
        key: 3 * value for key, value in base.items()
    }
    assert report["incremental_nonminimal_totals"] == {
        key: 2 * value for key, value in base.items()
    }
    assert report["all_policy_totals"]["literal_fit_slot_ceiling"] == 301119
    assert (
        report["all_policy_totals"][
            "prospective_new_fit_slots_after_source_cache_match"
        ]
        == 292803
    )
    assert (
        report["incremental_nonminimal_totals"]["literal_fit_slot_ceiling"]
        == 200746
    )
    assert (
        report["incremental_nonminimal_totals"][
            "prospective_new_fit_slots_after_source_cache_match"
        ]
        == 195202
    )


def test_malformed_input_error_text_and_repr_do_not_leak_sentinel():
    sentinel = "SENTINEL_LEAK_CANARY_9f3a"

    bad_count = [
        {
            "context_id": sentinel,
            "selection_mode": "master_cv",
            "selection_unit_ids": ["only", "two"],
        }
    ]
    with pytest.raises(AccountingError) as excinfo:
        plan_universal_accounting(bad_count, {sentinel: "D0-M"})
    error = excinfo.value
    assert error.reason_code == "master_cv_unit_count_invalid"
    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert error.reason_code in str(error)
    assert error.reason_code in repr(error)

    bad_units = [
        {
            "context_id": "ctx-ok",
            "selection_mode": "master_cv",
            "selection_unit_ids": [sentinel, "b", "c", "d"],
        }
    ]
    with pytest.raises(AccountingError) as excinfo2:
        plan_universal_accounting(bad_units, {"ctx-ok": "D0-M"})
    error2 = excinfo2.value
    assert error2.reason_code == "master_cv_unit_count_invalid"
    assert sentinel not in str(error2)
    assert sentinel not in repr(error2)

    bad_keys = [
        {
            "context_id": sentinel,
            "selection_mode": "master_cv",
            "selection_unit_ids": ["a", "b", "c"],
            "leak": sentinel,
        }
    ]
    with pytest.raises(AccountingError) as excinfo3:
        plan_universal_accounting(bad_keys, {sentinel: "D0-M"})
    error3 = excinfo3.value
    assert error3.reason_code == "context_keys_mismatch"
    assert sentinel not in str(error3)
    assert sentinel not in repr(error3)
