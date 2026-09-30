"""Tests for the bounded QC policy-selection subgraph builder."""

from __future__ import annotations

import copy
import hashlib

import pytest

from atlas_sers.evaluation import p08_qc_blocks, p08_qc_policy
from atlas_sers.evaluation.p08_plan import SEEDS, SVM_SEED
from atlas_sers.evaluation.p08_qc_policy import (
    PolicyBlockError,
    build_policy_blocks,
    require_scientific_execution,
)

_ALIASES = []
_HEXDIGITS = set("0123456789abcdef")
_SVM_MODEL_ID = "C-RBF-SVM"
_D0_MODEL_ID = "D0-M"
_ALLOWED_MODEL_IDS = {_SVM_MODEL_ID, _D0_MODEL_ID, "not_applicable"}

EXPECTED_STAGE_COUNTS = {
    "inner_quantile_fit": 6,
    "inner_route_pair": 6,
    "inner_source_fit": 12,
    "inner_source_prediction": 12,
    "policy_refit_quantile_fit": 2,
    "policy_route_pair": 2,
    "inner_select_hyperparameters": 2,
    "inner_select_refit_epochs": 2,
    "policy_scalar_calibration": 4,
    "policy_refit": 4,
    "policy_validation_prediction": 4,
    "policy_seed_ensemble_prediction": 4,
    "policy_panel_score": 2,
    "gate_objective": 1,
    "gate_selection": 1,
}

EXPECTED_SLOT_TOTALS = {
    "inner_quantile_fit": 6,
    "inner_route_pair": 744,
    "inner_source_fit": 29016,
    "inner_source_prediction": 29016,
    "policy_refit_quantile_fit": 2,
    "policy_route_pair": 248,
    "inner_select_hyperparameters": 248,
    "inner_select_refit_epochs": 744,
    "policy_scalar_calibration": 992,
    "policy_refit": 992,
    "policy_validation_prediction": 992,
    "policy_seed_ensemble_prediction": 496,
    "policy_panel_score": 248,
    "gate_objective": 124,
    "gate_selection": 1,
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding_obj() -> dict:
    return {"synthetic": "binding"}


def _binding_sha() -> str:
    return p08_qc_blocks.canonical_sha256(_binding_obj())


def _make_fold(unit_label: str, index: int) -> dict:
    role_pair_id = f"{unit_label}-fold-{index}"
    fit = _hash(f"{role_pair_id}-fit")
    validation = _hash(f"{role_pair_id}-validation")
    return {
        "fold_index": index,
        "role_pair_id": role_pair_id,
        "fit_uids": [],
        "validation_uids": [],
        "fit_uid_sha256": fit,
        "validation_uid_sha256": validation,
        "fit_masters": [],
        "validation_masters": [],
        "fit_class_master_counts": {},
        "validation_class_master_counts": {},
        "quantile_fit_uid_sha256": fit,
    }


def _make_unit(unit_id: str) -> dict:
    fit = _hash(f"{unit_id}-policy-fit")
    validation = _hash(f"{unit_id}-policy-validation")
    return {
        "parent_unit_id": unit_id,
        "policy_fit_uids": [],
        "policy_validation_uids": [],
        "policy_fit_uid_sha256": fit,
        "policy_validation_uid_sha256": validation,
        "policy_refit_quantile_fit_uid_sha256": fit,
        "inner_folds": [_make_fold(unit_id, index) for index in range(3)],
    }


def _make_context(unit_count: int = 2) -> dict:
    outer_fit = _hash("outer-fit")
    outer_test = _hash("outer-test")
    unit_ids = [f"unit-{chr(ord('a') + index)}" for index in range(unit_count)]
    return {
        "context_id": "ctx-0",
        "eligible": True,
        "reason_code": "eligible",
        "outer_fit_uid_sha256": outer_fit,
        "outer_test_uid_sha256": outer_test,
        "final_refit_quantile_fit_uid_sha256": outer_fit,
        "policy_units": [_make_unit(unit_id) for unit_id in unit_ids],
    }


def _make_candidates() -> list:
    return [
        {
            "candidate_id": f"cand-{index:02d}",
            "hyperparameter_sha256": _hash(f"cand-{index:02d}"),
        }
        for index in range(36)
    ]


def _build(context=None, candidates=None, binding=None) -> dict:
    context = _make_context() if context is None else context
    candidates = _make_candidates() if candidates is None else candidates
    binding = _binding_sha() if binding is None else binding
    return build_policy_blocks(binding, context, candidates)


def _raise(reason_code, context=None, candidates=None, binding=None):
    with pytest.raises(PolicyBlockError) as excinfo:
        _build(context=context, candidates=candidates, binding=binding)
    assert excinfo.value.reason_code == reason_code


def _get(block, name):
    return block[name]


def _block_id(block):
    return block["block_id"]


def _stage(block):
    return block["stage"]


def _model(block):
    return block["model_id"]


def _deps(block):
    return set(block["depends_on_blocks"])


def _blocks_for(result, stage):
    return [block for block in result["blocks"] if _stage(block) == stage]


def _by_id(result):
    return {_block_id(block): block for block in result["blocks"]}


def _slot_count(block):
    total = 1
    for values in block["axes"].values():
        total *= len(values)
    return total


def _ancestors(by_id, block):
    seen = set()
    stack = list(_deps(block))
    while stack:
        dependency = stack.pop()
        if dependency in seen:
            continue
        seen.add(dependency)
        stack.extend(_deps(by_id[dependency]))
    return seen


def _expected_d0_seeds():
    return sorted(int(seed) for seed in SEEDS)


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEXDIGITS
    )


def _assert_block_id(value):
    prefix = "P08QCBLOCK-"
    assert value.startswith(prefix)
    assert _is_hex64(value[len(prefix):])


def test_top_level_contract():
    result = _build()
    assert result["execution_authorized"] is False
    assert sorted(result) == [
        "blocks",
        "execution_authorized",
        "gate_selection_block_id",
        "policy_route_blocks_by_unit",
    ]
    assert len(result["blocks"]) == 64
    ids = [_block_id(block) for block in result["blocks"]]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    for value in ids:
        _assert_block_id(value)
    _assert_block_id(result["gate_selection_block_id"])
    assert list(result["policy_route_blocks_by_unit"]) == ["unit-a", "unit-b"]
    assert all(
        value in ids for value in result["policy_route_blocks_by_unit"].values()
    )


def test_sealed_catalog_summary_counts():
    result = _build()
    sealed = p08_qc_blocks.seal_catalog(
        _binding_obj(), result["blocks"], _ALIASES
    )
    summary = sealed["summary"]
    assert summary["stage_counts"] == EXPECTED_SLOT_TOTALS
    assert summary["stage_block_counts"] == EXPECTED_STAGE_COUNTS


def test_literal_slot_totals():
    result = _build()
    totals = {}
    for block in result["blocks"]:
        stage = _stage(block)
        totals[stage] = totals.get(stage, 0) + _slot_count(block)
    assert totals == EXPECTED_SLOT_TOTALS


def test_per_model_slot_counts():
    result = _build()
    for stage, expected in (
        ("inner_source_fit", {_SVM_MODEL_ID: 26784, _D0_MODEL_ID: 2232}),
        ("policy_refit", {_SVM_MODEL_ID: 248, _D0_MODEL_ID: 744}),
        ("policy_scalar_calibration", {_SVM_MODEL_ID: 248, _D0_MODEL_ID: 744}),
    ):
        per_model = {}
        for block in _blocks_for(result, stage):
            model = _model(block)
            per_model[model] = per_model.get(model, 0) + _slot_count(block)
        assert per_model == expected


def test_svm_candidate_axes_are_sorted_registry():
    candidates = _make_candidates()
    result = _build(candidates=candidates)
    expected = [
        {
            "candidate_id": candidate["candidate_id"],
            "hyperparameter_sha256": candidate["hyperparameter_sha256"],
        }
        for candidate in sorted(
            candidates, key=lambda item: item["candidate_id"]
        )
    ]
    for block in _blocks_for(result, "inner_source_fit"):
        if _model(block) != _SVM_MODEL_ID:
            continue
        axes = _get(block, "axes")
        assert axes["candidate"] == expected
        assert axes["seed"] == [SVM_SEED]
        assert len(axes["gate_id"]) == 124
        assert axes["gate_id"][0] == "QC-000-MIN"
        assert axes["gate_id"][-1] == "QC-123-DUAL"


def test_d0_candidate_and_seed_axes():
    result = _build()
    expected_candidate = [
        {"candidate_id": "fixed_spec", "hyperparameter_sha256": "not_applicable"}
    ]
    for block in _blocks_for(result, "inner_source_fit"):
        if _model(block) != _D0_MODEL_ID:
            continue
        axes = _get(block, "axes")
        assert axes["candidate"] == expected_candidate
        assert axes["seed"] == _expected_d0_seeds()


def test_no_block_carries_test_role():
    context = _make_context()
    result = _build(context=context)
    outer_test = context["outer_test_uid_sha256"]
    for block in result["blocks"]:
        assert _get(block, "test_uid_sha256") == "not_applicable"
        for field in ("fit_uid_sha256", "validation_uid_sha256"):
            assert _get(block, field) != outer_test


def test_selection_and_calibration_have_no_validation_hash():
    result = _build()
    stages = {
        "inner_select_hyperparameters",
        "inner_select_refit_epochs",
        "policy_scalar_calibration",
    }
    for block in result["blocks"]:
        if _stage(block) in stages:
            assert _get(block, "validation_uid_sha256") == "not_applicable"
            assert _get(block, "test_uid_sha256") == "not_applicable"


def test_inner_quantile_fit_singletons_and_role():
    result = _build()
    quantiles = _blocks_for(result, "inner_quantile_fit")
    assert len(quantiles) == 6
    for block in quantiles:
        axes = _get(block, "axes")
        assert axes["gate_id"] == ["not_applicable"]
        assert axes["candidate"] == [
            {
                "candidate_id": "not_applicable",
                "hyperparameter_sha256": "not_applicable",
            }
        ]
        assert axes["seed"] == ["not_applicable"]
        assert _model(block) == "not_applicable"
        assert _get(block, "validation_uid_sha256") == "not_applicable"
        assert _get(block, "fit_uid_sha256") != "not_applicable"
        assert "-fold-" in _get(block, "role_id")
        assert _deps(block) == set()
        assert _get(block, "resolution") == "source_fit_only_shared_all_gates"


def test_inner_route_pair_depends_on_quantile():
    result = _build()
    by_id = _by_id(result)
    routes = _blocks_for(result, "inner_route_pair")
    assert len(routes) == 6
    for route in routes:
        deps = _deps(route)
        assert len(deps) == 1
        parent = by_id[next(iter(deps))]
        assert _stage(parent) == "inner_quantile_fit"


def test_inner_selection_dependencies_and_resolutions():
    result = _build()
    by_id = _by_id(result)
    svm_select = _blocks_for(result, "inner_select_hyperparameters")
    d0_select = _blocks_for(result, "inner_select_refit_epochs")
    assert len(svm_select) == 2
    assert len(d0_select) == 2
    for block in svm_select:
        deps = _deps(block)
        assert len(deps) == 3
        assert all(_stage(by_id[dep]) == "inner_source_prediction" for dep in deps)
        assert all(_model(by_id[dep]) == _SVM_MODEL_ID for dep in deps)
        assert "selectedcandidate-all3folds" in _get(block, "resolution")
    for block in d0_select:
        deps = _deps(block)
        assert len(deps) == 3
        assert all(_stage(by_id[dep]) == "inner_source_fit" for dep in deps)
        assert all(_model(by_id[dep]) == _D0_MODEL_ID for dep in deps)
        assert (
            "samegate-sameseed-median3bestepochs-pythonround-clip30_200"
            in _get(block, "resolution")
        )


def test_scalar_calibration_dependencies_and_seed_order():
    result = _build()
    by_id = _by_id(result)
    scalars = _blocks_for(result, "policy_scalar_calibration")
    assert len(scalars) == 4
    for block in scalars:
        deps = _deps(block)
        if _model(block) == _SVM_MODEL_ID:
            stages = sorted(_stage(by_id[dep]) for dep in deps)
            assert stages.count("inner_source_prediction") == 3
            assert stages.count("inner_select_hyperparameters") == 1
            assert _get(block, "axes")["seed"] == [SVM_SEED]
        else:
            assert _get(block, "axes")["seed"] == _expected_d0_seeds()
            predictions = [
                dep
                for dep in deps
                if _stage(by_id[dep]) == "inner_source_prediction"
            ]
            assert len(predictions) == 3
            assert all(
                _model(by_id[dep]) == _D0_MODEL_ID for dep in predictions
            )


def test_policy_refit_dependencies_and_fit_only_resolution():
    result = _build()
    by_id = _by_id(result)
    refits = _blocks_for(result, "policy_refit")
    assert len(refits) == 4
    for block in refits:
        assert "fit_only_F" in _get(block, "resolution")
        assert _get(block, "validation_uid_sha256") == "not_applicable"
        routes = [
            dep for dep in _deps(block) if _stage(by_id[dep]) == "policy_route_pair"
        ]
        assert len(routes) == 1
        if _model(block) == _SVM_MODEL_ID:
            stages = {_stage(by_id[dep]) for dep in _deps(block)}
            assert "inner_select_hyperparameters" in stages
            assert "same_gate_selected_candidate" in _get(block, "resolution")
        else:
            stages = {_stage(by_id[dep]) for dep in _deps(block)}
            assert "inner_select_refit_epochs" in stages
            assert "same_gate_same_seed_source_selected_epochs" in _get(
                block, "resolution"
            )


def test_validation_prediction_and_ensemble_chain():
    result = _build()
    by_id = _by_id(result)
    for block in _blocks_for(result, "policy_validation_prediction"):
        stages = sorted(_stage(by_id[dep]) for dep in _deps(block))
        assert stages == ["policy_refit", "policy_scalar_calibration"]
        assert _get(block, "validation_uid_sha256") != "not_applicable"
    ensembles = _blocks_for(result, "policy_seed_ensemble_prediction")
    assert len(ensembles) == 4
    for block in ensembles:
        assert _get(block, "axes")["seed"] == ["not_applicable"]
        deps = _deps(block)
        assert len(deps) == 1
        parent = by_id[next(iter(deps))]
        assert _stage(parent) == "policy_validation_prediction"


def test_panel_score_and_gate_objective_parents():
    result = _build()
    by_id = _by_id(result)
    panels = _blocks_for(result, "policy_panel_score")
    assert len(panels) == 2
    for block in panels:
        assert _model(block) == "not_applicable"
        assert _get(block, "fit_uid_sha256") == "not_applicable"
        deps = _deps(block)
        assert len(deps) == 2
        assert {_stage(by_id[dep]) for dep in deps} == {
            "policy_seed_ensemble_prediction"
        }
        assert "M01" in _get(block, "resolution")
        assert "equal_svm_d0_weights" in _get(block, "resolution")

    objective = _blocks_for(result, "gate_objective")
    assert len(objective) == 1
    deps = _deps(objective[0])
    assert len(deps) == 4
    assert {_stage(by_id[dep]) for dep in deps} == {
        "policy_panel_score",
        "policy_route_pair",
    }
    assert _get(objective[0], "role_id") == "ctx-0"
    assert _get(objective[0], "fit_uid_sha256") == _hash("outer-fit")


def test_gate_selection_placeholder():
    result = _build()
    objective_id = _block_id(_blocks_for(result, "gate_objective")[0])
    selections = _blocks_for(result, "gate_selection")
    assert len(selections) == 1
    block = selections[0]
    assert _get(block, "axes")["gate_id"] == ["source_selected_gate"]
    assert _deps(block) == {objective_id}
    assert "no_winner_computed" in _get(block, "resolution")
    assert _get(block, "test_uid_sha256") == "not_applicable"
    assert _slot_count(block) == 1


def test_source_selected_placeholders_are_unresolved():
    result = _build()
    found = False
    for block in result["blocks"]:
        for candidate in _get(block, "axes")["candidate"]:
            if candidate["candidate_id"] == "source_selected_candidate":
                found = True
                assert candidate["hyperparameter_sha256"] == "not_applicable"
    assert found


def test_ordering_and_inputs_not_mutated():
    context = _make_context()
    candidates = _make_candidates()
    context_before = copy.deepcopy(context)
    candidates_before = copy.deepcopy(candidates)
    result = _build(context=context, candidates=candidates)
    assert context == context_before
    assert candidates == candidates_before
    ids = [_block_id(block) for block in result["blocks"]]
    assert ids == sorted(ids)
    unit_ids = list(result["policy_route_blocks_by_unit"])
    assert unit_ids == sorted(unit_ids)


def test_permutation_invariance():
    base = _build()
    context = _make_context()
    context["policy_units"].reverse()
    for unit in context["policy_units"]:
        unit["inner_folds"].reverse()
    candidates = _make_candidates()
    candidates.reverse()
    permuted = _build(context=context, candidates=candidates)
    assert [_block_id(block) for block in permuted["blocks"]] == [
        _block_id(block) for block in base["blocks"]
    ]
    assert permuted["gate_selection_block_id"] == base["gate_selection_block_id"]
    assert permuted["policy_route_blocks_by_unit"] == (
        base["policy_route_blocks_by_unit"]
    )


def test_binding_changes_block_ids():
    first = _build(binding=_binding_sha())
    second = _build(
        binding=p08_qc_blocks.canonical_sha256({"synthetic": "other"})
    )
    first_ids = {_block_id(block) for block in first["blocks"]}
    second_ids = {_block_id(block) for block in second["blocks"]}
    assert first_ids
    assert not (first_ids & second_ids)


def test_require_scientific_execution_always_raises():
    with pytest.raises(PolicyBlockError) as excinfo:
        require_scientific_execution()
    assert excinfo.value.reason_code == "scientific_execution_not_authorized"
    with pytest.raises(PolicyBlockError) as excinfo:
        require_scientific_execution("ctx", 1, extra=True)
    assert excinfo.value.reason_code == "scientific_execution_not_authorized"


def test_binding_invalid():
    _raise(p08_qc_policy.REASON_BINDING_SHA256_INVALID, binding="short")


def test_context_not_mapping():
    _raise(p08_qc_policy.REASON_CONTEXT_NOT_MAPPING, context=["nope"])


def test_unsupported_context_rejected():
    context = _make_context()
    context["eligible"] = False
    _raise(p08_qc_policy.REASON_UNSUPPORTED_CONTEXT, context=context)


def test_missing_eligible_rejected():
    context = _make_context()
    del context["eligible"]
    _raise(p08_qc_policy.REASON_UNSUPPORTED_CONTEXT, context=context)


def test_context_id_invalid():
    context = _make_context()
    context["context_id"] = " ctx "
    _raise(p08_qc_policy.REASON_CONTEXT_ID_INVALID, context=context)


def test_outer_fit_test_same():
    context = _make_context()
    context["outer_test_uid_sha256"] = context["outer_fit_uid_sha256"]
    _raise(p08_qc_policy.REASON_OUTER_FIT_TEST_SAME, context=context)


def test_too_few_units():
    context = _make_context()
    context["policy_units"] = context["policy_units"][:1]
    _raise(p08_qc_policy.REASON_POLICY_UNITS_COUNT_INVALID, context=context)


def test_duplicate_unit_ids():
    context = _make_context()
    context["policy_units"][1]["parent_unit_id"] = "unit-a"
    _raise(p08_qc_policy.REASON_UNIT_IDS_DUPLICATE, context=context)


def test_unit_hash_invalid():
    context = _make_context()
    context["policy_units"][0]["policy_fit_uid_sha256"] = "ZZ"
    _raise(p08_qc_policy.REASON_UNIT_FIT_UID_INVALID, context=context)


def test_unit_fit_validation_same():
    context = _make_context()
    unit = context["policy_units"][0]
    unit["policy_validation_uid_sha256"] = unit["policy_fit_uid_sha256"]
    _raise(p08_qc_policy.REASON_UNIT_FIT_VALIDATION_SAME, context=context)


def test_unit_hash_equals_outer_test():
    context = _make_context()
    context["policy_units"][0]["policy_fit_uid_sha256"] = (
        context["outer_test_uid_sha256"]
    )
    _raise(p08_qc_policy.REASON_UNIT_HASH_EQUALS_OUTER_TEST, context=context)


def test_folds_count_invalid():
    context = _make_context()
    unit = context["policy_units"][0]
    unit["inner_folds"] = unit["inner_folds"][:2]
    _raise(p08_qc_policy.REASON_INNER_FOLDS_INVALID, context=context)


def test_fold_index_bool_rejected():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][0]["fold_index"] = True
    _raise(p08_qc_policy.REASON_FOLD_INDEX_INVALID, context=context)


def test_fold_index_duplicate_rejected():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][1]["fold_index"] = 0
    _raise(p08_qc_policy.REASON_FOLD_INDEX_INVALID, context=context)


def test_duplicate_role_pair_ids_rejected():
    context = _make_context()
    role = context["policy_units"][0]["inner_folds"][0]["role_pair_id"]
    context["policy_units"][0]["inner_folds"][1]["role_pair_id"] = role
    _raise(p08_qc_policy.REASON_ROLE_PAIR_IDS_DUPLICATE, context=context)


def test_fold_hash_invalid():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][0]["fit_uid_sha256"] = "nothex"
    _raise(p08_qc_policy.REASON_FOLD_FIT_UID_INVALID, context=context)


def test_fold_hash_equals_outer_test():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][0]["fit_uid_sha256"] = (
        context["outer_test_uid_sha256"]
    )
    _raise(p08_qc_policy.REASON_FOLD_HASH_EQUALS_OUTER_TEST, context=context)


def test_candidate_count_invalid():
    _raise(
        p08_qc_policy.REASON_SVM_CANDIDATES_COUNT_INVALID,
        candidates=_make_candidates()[:35],
    )


def test_candidate_keys_invalid():
    candidates = _make_candidates()
    candidates[0]["model_id"] = "svm"
    _raise(p08_qc_policy.REASON_SVM_CANDIDATE_KEYS_INVALID, candidates=candidates)


def test_candidate_id_invalid():
    candidates = _make_candidates()
    candidates[0]["candidate_id"] = ""
    _raise(p08_qc_policy.REASON_SVM_CANDIDATE_ID_INVALID, candidates=candidates)


def test_candidate_hash_invalid():
    candidates = _make_candidates()
    candidates[0]["hyperparameter_sha256"] = "0x00"
    _raise(p08_qc_policy.REASON_SVM_CANDIDATE_HASH_INVALID, candidates=candidates)


def test_candidate_duplicate_ids():
    candidates = _make_candidates()
    candidates[1]["candidate_id"] = candidates[0]["candidate_id"]
    _raise(
        p08_qc_policy.REASON_SVM_CANDIDATE_IDS_DUPLICATE,
        candidates=candidates,
    )


def test_three_units_scale_without_hardcoding():
    result = _build(context=_make_context(unit_count=3))
    assert len(result["blocks"]) == 95
    totals = {}
    for block in result["blocks"]:
        stage = _stage(block)
        totals[stage] = totals.get(stage, 0) + _slot_count(block)
    assert totals["inner_source_fit"] == 43524
    assert totals["policy_refit"] == 1488
    assert totals["policy_scalar_calibration"] == 1488


def test_all_models_in_exact_allowed_set():
    result = _build()
    for block in result["blocks"]:
        assert _model(block) in _ALLOWED_MODEL_IDS


def test_policy_block_error_is_value_error():
    assert issubclass(PolicyBlockError, ValueError)


def test_context_none_rejected_direct():
    with pytest.raises(PolicyBlockError) as excinfo:
        build_policy_blocks(_binding_sha(), None, _make_candidates())
    assert excinfo.value.reason_code == p08_qc_policy.REASON_CONTEXT_NOT_MAPPING


def test_context_quantile_binding_mismatch():
    context = _make_context()
    context["final_refit_quantile_fit_uid_sha256"] = _hash("other")
    _raise(p08_qc_policy.REASON_CONTEXT_QUANTILE_BINDING_MISMATCH, context=context)


def test_context_quantile_uid_invalid():
    context = _make_context()
    context["final_refit_quantile_fit_uid_sha256"] = "zz"
    _raise(p08_qc_policy.REASON_CONTEXT_QUANTILE_UID_INVALID, context=context)


def test_unit_quantile_binding_mismatch():
    context = _make_context()
    context["policy_units"][0]["policy_refit_quantile_fit_uid_sha256"] = (
        _hash("other")
    )
    _raise(p08_qc_policy.REASON_UNIT_QUANTILE_BINDING_MISMATCH, context=context)


def test_fold_quantile_binding_mismatch():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][0]["quantile_fit_uid_sha256"] = (
        _hash("other")
    )
    _raise(p08_qc_policy.REASON_FOLD_QUANTILE_BINDING_MISMATCH, context=context)


def test_quantile_resolutions_are_source_fit_only():
    result = _build()
    for stage in ("inner_quantile_fit", "policy_refit_quantile_fit"):
        for block in _blocks_for(result, stage):
            assert _get(block, "resolution") == "source_fit_only_shared_all_gates"


def test_lone_surrogate_context_id_rejected():
    context = _make_context()
    context["context_id"] = "ctx-\ud800"
    _raise(p08_qc_policy.REASON_CONTEXT_ID_INVALID, context=context)


def test_lone_surrogate_unit_id_rejected():
    context = _make_context()
    context["policy_units"][0]["parent_unit_id"] = "unit-\ud800"
    _raise(p08_qc_policy.REASON_UNIT_ID_INVALID, context=context)


def test_valid_unicode_context_id_accepted():
    context = _make_context()
    context["context_id"] = "ctx-\u03a9-\u00e7"
    result = _build(context=context)
    assert result["execution_authorized"] is False
    assert result["blocks"]


def test_no_raw_membership_fields_leak():
    context = _make_context()
    context["policy_units"][0]["inner_folds"][0]["fit_uids"] = [
        "RAW-MEMBERSHIP"
    ]
    result = _build(context=context)
    assert "RAW-MEMBERSHIP" not in repr(result["blocks"])
    forbidden = {
        "fit_uids",
        "validation_uids",
        "fit_masters",
        "validation_masters",
        "fit_class_master_counts",
        "validation_class_master_counts",
    }
    for block in result["blocks"]:
        assert not (set(block) & forbidden)


def test_gate_objective_depends_on_all_unit_panels_and_same_unit_routes():
    result = _build()
    by_id = _by_id(result)
    objective = _blocks_for(result, "gate_objective")[0]
    deps = _deps(objective)
    panels = [dep for dep in deps if _stage(by_id[dep]) == "policy_panel_score"]
    routes = [dep for dep in deps if _stage(by_id[dep]) == "policy_route_pair"]
    assert len(panels) == 2
    assert set(routes) == set(result["policy_route_blocks_by_unit"].values())
    for panel in panels:
        assert _model(by_id[panel]) == "not_applicable"


def test_selection_and_calibration_exclude_parent_validation_ancestors():
    context = _make_context()
    result = _build(context=context)
    by_id = _by_id(result)
    parent_validation = {
        unit["policy_validation_uid_sha256"] for unit in context["policy_units"]
    }
    stages = {
        "inner_select_hyperparameters",
        "inner_select_refit_epochs",
        "policy_scalar_calibration",
    }
    for block in result["blocks"]:
        if _stage(block) not in stages:
            continue
        for ancestor in _ancestors(by_id, block):
            assert (
                by_id[ancestor]["validation_uid_sha256"] not in parent_validation
            )


def test_d0_inner_fit_binds_fold_validation_and_chain_agrees():
    context = _make_context()
    result = _build(context=context)
    by_id = _by_id(result)
    folds = {}
    for unit in context["policy_units"]:
        for fold in unit["inner_folds"]:
            folds[fold["role_pair_id"]] = fold
    fits = [
        block
        for block in result["blocks"]
        if _stage(block) == "inner_source_fit"
        and _model(block) == _D0_MODEL_ID
    ]
    assert len(fits) == 6
    for fit in fits:
        fold = folds[fit["role_id"]]
        assert fit["fit_uid_sha256"] == fold["fit_uid_sha256"]
        assert fit["validation_uid_sha256"] == fold["validation_uid_sha256"]
        assert fit["test_uid_sha256"] == "not_applicable"
        deps = _deps(fit)
        assert len(deps) == 1
        route = by_id[next(iter(deps))]
        assert _stage(route) == "inner_route_pair"
        assert route["role_id"] == fit["role_id"]
        assert route["fit_uid_sha256"] == fit["fit_uid_sha256"]
        child = [
            block
            for block in result["blocks"]
            if _stage(block) == "inner_source_prediction"
            and _model(block) == _D0_MODEL_ID
            and fit["block_id"] in _deps(block)
        ]
        assert len(child) == 1
        prediction = child[0]
        assert prediction["role_id"] == fit["role_id"]
        assert prediction["fit_uid_sha256"] == fit["fit_uid_sha256"]
        assert prediction["validation_uid_sha256"] == fit["validation_uid_sha256"]


def test_svm_inner_fit_is_not_validation_bound():
    result = _build()
    fits = [
        block
        for block in result["blocks"]
        if _stage(block) == "inner_source_fit"
        and _model(block) == _SVM_MODEL_ID
    ]
    assert len(fits) == 6
    for fit in fits:
        assert fit["validation_uid_sha256"] == "not_applicable"


def test_block_ids_carry_content_address_prefix():
    result = _build()
    for block in result["blocks"]:
        _assert_block_id(block["block_id"])
