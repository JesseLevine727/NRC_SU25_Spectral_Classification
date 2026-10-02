"""Structural tests for the pinned P08 range plan expansion."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from atlas_sers.evaluation import p08_range_plan as rp
from tests.test_p08_plan import (
    build_plan,
    make_candidates,
    make_master_context,
    make_pseudo_context,
)

_PRIMARY_POLICY_ID = "PP-U-MIN"
_RETAINED_MODEL_IDS = frozenset({"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "D1", "D2", "D3"})
_FIT_STAGES = frozenset({"source_fit", "calibration_model_fit", "final_refit"})
_PRIMARY_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"
_ARRAY = "1" * 64
_AXIS = "2" * 64
_FILE = "3" * 64


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _clone(value):
    return copy.deepcopy(value)


def _expect(reason, fn):
    with pytest.raises(rp.RangePlanError) as excinfo:
        fn()
    assert excinfo.value.reason_code == reason


def _job_key(job):
    return (
        job["context_id"],
        job["model_id"],
        job["stage"],
        job["unit_id"],
        job["seed"],
        job["candidate_id"],
    )


def _retained_jobs(primary):
    return [
        job
        for job in primary["jobs"]
        if job["policy_id"] == _PRIMARY_POLICY_ID and job["model_id"] in _RETAINED_MODEL_IDS
    ]


def _rehash(plan):
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = _hash(payload)
    return plan


def _min_job(plan):
    return next(
        job
        for job in plan["jobs"]
        if job["policy_id"] == _PRIMARY_POLICY_ID and job["model_id"] in _RETAINED_MODEL_IDS
    )


def _min_alias(plan):
    return next(alias for alias in plan["aliases"] if alias["policy_id"] == _PRIMARY_POLICY_ID)


def _repin(monkeypatch, plan):
    _rehash(plan)
    monkeypatch.setattr(rp, "PRIMARY_PLAN_SHA256", plan["plan_sha256"])


class _StrSubclass(str):
    pass


@pytest.fixture
def contexts():
    return [
        make_master_context(),
        make_pseudo_context(context_id="CTX-PSEUDO-BETA", recipe="D1"),
        make_pseudo_context(context_id="CTX-PSEUDO-GAMMA", recipe="D2"),
        make_pseudo_context(context_id="CTX-PSEUDO-DELTA", recipe="D3"),
    ]


@pytest.fixture
def primary(monkeypatch, contexts):
    plan = build_plan(contexts=contexts, candidates=make_candidates())
    monkeypatch.setattr(rp, "PRIMARY_PLAN_SHA256", plan["plan_sha256"])
    return plan


@pytest.fixture
def range_input():
    return {
        "representation_id": rp.REPRESENTATION_ID,
        "rows": 598,
        "features": 1450,
        "dtype": "float32",
        "axis_start_cm1": 400,
        "axis_end_cm1": 1849,
        "array_sha256": _ARRAY,
        "axis_sha256": _AXIS,
        "row_order_sha256": rp.PRIMARY_ROW_ORDER_SHA256,
        "file_sha256": _FILE,
        "invalid_rows": 0,
    }


@pytest.fixture
def plan(primary, range_input):
    return rp.build_range_plan(primary, range_input)


def test_public_constants():
    assert rp.PRIMARY_PLAN_SHA256 == _PRIMARY_PLAN_SHA256
    assert rp.PRIMARY_ROW_ORDER_SHA256 == (
        "b0d9ef9ae34a87443d951742bf1b295df522dd12674d5ca4a9785953cee6a5a7"
    )
    assert rp.POLICY_ID == "PP-RANGE-MIN"
    assert rp.REPRESENTATION_ID == "R_MIN_400_1849"
    assert rp.SCHEMA_VERSION == "nato-sers-p08-range-slot-dag-v1"


def test_pin_rejects_unrelated_plan(primary, range_input, monkeypatch):
    monkeypatch.setattr(rp, "PRIMARY_PLAN_SHA256", _PRIMARY_PLAN_SHA256)
    _expect(
        "primary_plan_hash_mismatch",
        lambda: rp.build_range_plan(primary, range_input),
    )


def test_payload_hash_mismatch(primary, range_input):
    bad = _clone(primary)
    bad["summary"]["totals"]["total_jobs"] += 1
    _expect(
        "primary_payload_hash_mismatch",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_forged_execution_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    bad["execution_authorized"] = True
    _repin(monkeypatch, bad)
    _expect(
        "primary_execution_must_be_denied",
        lambda: rp.build_range_plan(bad, range_input),
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("representation_id", "R_OTHER", "range_input_representation_invalid"),
        ("representation_id", 7, "range_input_representation_invalid"),
        ("dtype", "float64", "range_input_dtype_invalid"),
        ("dtype", None, "range_input_dtype_invalid"),
        ("rows", 597, "range_input_rows_invalid"),
        ("rows", True, "range_input_rows_invalid"),
        ("rows", 598.0, "range_input_rows_invalid"),
        ("features", 1449, "range_input_features_invalid"),
        ("features", 1450.0, "range_input_features_invalid"),
        ("axis_start_cm1", 401, "range_input_axis_start_invalid"),
        ("axis_end_cm1", 1850, "range_input_axis_end_invalid"),
        ("invalid_rows", 1, "range_input_invalid_rows_invalid"),
        ("invalid_rows", False, "range_input_invalid_rows_invalid"),
        ("array_sha256", "Z" * 64, "range_input_array_sha256_invalid"),
        ("array_sha256", "A" * 64, "range_input_array_sha256_invalid"),
        ("array_sha256", "1" * 63, "range_input_array_sha256_invalid"),
        ("axis_sha256", None, "range_input_axis_sha256_invalid"),
        ("row_order_sha256", "0" * 64, "range_input_row_order_sha256_invalid"),
        ("row_order_sha256", "B" * 64, "range_input_row_order_sha256_invalid"),
        ("file_sha256", "z" * 64, "range_input_file_sha256_invalid"),
    ],
)
def test_range_input_fields_rejected(primary, range_input, field, value, reason):
    bad = _clone(range_input)
    bad[field] = value
    _expect(reason, lambda: rp.build_range_plan(primary, bad))


@pytest.mark.parametrize(
    "field,reason",
    [
        ("representation_id", "range_input_representation_invalid"),
        ("dtype", "range_input_dtype_invalid"),
        ("row_order_sha256", "range_input_row_order_sha256_invalid"),
    ],
)
def test_range_input_rejects_str_subclass(primary, range_input, field, reason):
    bad = _clone(range_input)
    bad[field] = _StrSubclass(bad[field])
    _expect(reason, lambda: rp.build_range_plan(primary, bad))


def test_range_input_keys_rejected(primary, range_input):
    missing = _clone(range_input)
    del missing["file_sha256"]
    _expect("range_input_keys_invalid", lambda: rp.build_range_plan(primary, missing))
    extra = _clone(range_input)
    extra["unexpected"] = 1
    _expect("range_input_keys_invalid", lambda: rp.build_range_plan(primary, extra))


def test_inputs_immutable_and_deterministic(primary, range_input):
    primary_copy = _clone(primary)
    input_copy = _clone(range_input)
    first = rp.build_range_plan(primary, range_input)
    second = rp.build_range_plan(primary, range_input)
    assert first == second
    assert primary == primary_copy
    assert range_input == input_copy


def test_recomputed_hashes(plan):
    for job in plan["jobs"]:
        fields = {key: value for key, value in job.items() if key != "job_id"}
        assert job["job_id"] == "P08RANGEJOB-" + _hash(fields)
    for alias in plan["aliases"]:
        fields = {key: value for key, value in alias.items() if key != "alias_id"}
        assert alias["alias_id"] == "P08RANGEALIAS-" + _hash(fields)
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert plan["plan_sha256"] == _hash(payload)


def test_selection_and_transform_invariants(primary, range_input, plan):
    expected = _retained_jobs(primary)
    assert plan["summary"]["jobs_count"] == len(expected)
    expected_stages = {}
    for job in expected:
        expected_stages[job["stage"]] = expected_stages.get(job["stage"], 0) + 1
    assert plan["summary"]["stage_counts"] == expected_stages

    new_by_key = {_job_key(job): job for job in plan["jobs"]}
    assert set(new_by_key) == {_job_key(job) for job in expected}
    new_id_of = {job["job_id"]: new_by_key[_job_key(job)]["job_id"] for job in expected}
    assert len(new_id_of) == len(expected)
    old_ids = {job["job_id"] for job in primary["jobs"]}
    new_ids = {job["job_id"] for job in plan["jobs"]}
    assert old_ids.isdisjoint(new_ids)

    contract = _hash(range_input)
    base_specs = {}
    for job in expected:
        base_specs.setdefault(job["model_id"], job["model_spec_sha256"])

    for job in expected:
        new = new_by_key[_job_key(job)]
        expected_spec = _hash(
            {
                "base_model_spec_sha256": base_specs[job["model_id"]],
                "range_input_contract_sha256": contract,
            }
        )
        assert new["policy_id"] == rp.POLICY_ID
        assert new["representation_id"] == rp.REPRESENTATION_ID
        assert new["array_sha256"] == range_input["array_sha256"]
        assert new["model_spec_sha256"] == expected_spec
        assert new["evidence_status"] == rp.core.EVIDENCE_FUTURE
        if job["model_id"] in {"D0-M", "D1", "D2", "D3"} and job["stage"] in {
            "source_fit",
            "source_validation_prediction",
        }:
            assert new["hyperparameter_sha256"] == expected_spec
        else:
            assert new["hyperparameter_sha256"] == job["hyperparameter_sha256"]
        assert new["fit_uid_sha256"] == job["fit_uid_sha256"]
        assert new["validation_uid_sha256"] == job["validation_uid_sha256"]
        assert new["test_uid_sha256"] == job["test_uid_sha256"]
        assert new["resolution"] == job["resolution"]
        assert new["unit_id"] == job["unit_id"]
        assert new["seed"] == job["seed"]
        assert new["candidate_id"] == job["candidate_id"]
        assert new["model_id"] == job["model_id"]
        assert new["stage"] == job["stage"]
        assert new["context_id"] == job["context_id"]
        assert new["dependencies"] == sorted(new_id_of[dep] for dep in job["dependencies"])
        assert all(dep in new_ids for dep in new["dependencies"])

    for model, spec in base_specs.items():
        assert plan["model_spec_sha256"][model] == _hash(
            {
                "base_model_spec_sha256": spec,
                "range_input_contract_sha256": contract,
            }
        )


def test_only_registered_min_sources(primary, plan):
    base_models = {
        job["model_id"] for job in primary["jobs"] if job["policy_id"] == _PRIMARY_POLICY_ID
    }
    assert "C-EXTRA-TREES" in base_models
    assert _RETAINED_MODEL_IDS < base_models
    output_models = {job["model_id"] for job in plan["jobs"]}
    assert output_models == _RETAINED_MODEL_IDS
    assert "C-EXTRA-TREES" not in output_models
    primary_et = [
        job
        for job in primary["jobs"]
        if job["policy_id"] == _PRIMARY_POLICY_ID and job["model_id"] == "C-EXTRA-TREES"
    ]
    assert primary_et
    assert all(job["policy_id"] == rp.POLICY_ID for job in plan["jobs"])


def test_master_sharing_and_counts(primary, plan):
    expected = _retained_jobs(primary)
    new_by_key = {_job_key(job): job for job in plan["jobs"]}
    new_id_of = {job["job_id"]: new_by_key[_job_key(job)]["job_id"] for job in expected}
    old_lookup = {job["job_id"]: job for job in primary["jobs"]}
    old_by_new = {new_id_of[old_id]: old_lookup[old_id] for old_id in new_id_of}

    calibration_aliases = [
        job for job in plan["jobs"] if job["stage"] == "calibration_prediction_alias"
    ]
    assert calibration_aliases
    for job in calibration_aliases:
        assert job["context_id"] == "CTX-MASTER-ALPHA"
        assert job["dependencies"]
        dependency_jobs = [old_by_new[dependency] for dependency in job["dependencies"]]
        assert {dependency["stage"] for dependency in dependency_jobs} == {
            "select_hyperparameters",
            "source_validation_prediction",
        }
        selections = [
            dependency
            for dependency in dependency_jobs
            if dependency["stage"] == "select_hyperparameters"
        ]
        predictions = [
            dependency
            for dependency in dependency_jobs
            if dependency["stage"] == "source_validation_prediction"
        ]
        assert len(selections) == 1
        assert predictions
        for dependency in dependency_jobs:
            assert dependency["context_id"] == job["context_id"]
            assert dependency["model_id"] == job["model_id"]
        for dependency in predictions:
            assert dependency["unit_id"] == job["unit_id"]
            assert dependency["seed"] == job["seed"]
        expected_candidates = {
            candidate["candidate_id"]
            for candidate in primary["jobs"]
            if candidate["stage"] == "source_validation_prediction"
            and candidate["policy_id"] == _PRIMARY_POLICY_ID
            and candidate["model_id"] == job["model_id"]
            and candidate["context_id"] == job["context_id"]
            and candidate["unit_id"] == job["unit_id"]
            and candidate["seed"] == job["seed"]
        }
        assert {dependency["candidate_id"] for dependency in predictions} == expected_candidates

    master_aliases = [
        alias for alias in plan["aliases"] if alias["context_id"] == "CTX-MASTER-ALPHA"
    ]
    assert len(master_aliases) == 2
    assert len({alias["strategy"] for alias in master_aliases}) == 2
    assert len({alias["target_job_id"] for alias in master_aliases}) == 1
    target = {job["job_id"]: job for job in plan["jobs"]}[master_aliases[0]["target_job_id"]]
    assert target["stage"] == rp._SEED_STAGE
    assert target["context_id"] == "CTX-MASTER-ALPHA"
    assert target["model_id"] == "D0-M"

    expected_fit = sum(1 for job in expected if job["stage"] in _FIT_STAGES)
    expected_scalar = sum(1 for job in expected if job["stage"] == "scalar_calibration")
    plan_contexts = {job["context_id"] for job in plan["jobs"]}
    assert plan_contexts == {job["context_id"] for job in expected}
    assert len(plan_contexts) == 4
    assert plan["summary"]["model_fit_slots"] == expected_fit
    assert plan["summary"]["scalar_calibrations"] == expected_scalar
    base_aliases = [
        alias for alias in primary["aliases"] if alias["policy_id"] == _PRIMARY_POLICY_ID
    ]
    assert plan["summary"]["aliases_count"] == len(base_aliases)
    assert plan["summary"]["context_count"] == 4
    assert plan["summary"]["authorized_fit_slots"] == 0
    assert plan["execution_authorized"] is False
    assert len(plan["aliases"]) == len(base_aliases)


def test_alias_invariants(primary, plan):
    selected = [alias for alias in primary["aliases"] if alias["policy_id"] == _PRIMARY_POLICY_ID]
    assert plan["aliases"]
    assert len(plan["aliases"]) == len(selected)
    lookup = {job["job_id"]: job for job in plan["jobs"]}
    pairs, ids, contexts = [], [], set()
    for alias in plan["aliases"]:
        assert alias["policy_id"] == rp.POLICY_ID
        target = lookup[alias["target_job_id"]]
        assert target["stage"] == rp._SEED_STAGE
        assert target["context_id"] == alias["context_id"]
        assert target["model_id"] == alias["recipe_id"]
        pairs.append((alias["context_id"], alias["strategy"]))
        ids.append(alias["alias_id"])
        contexts.add(alias["context_id"])
    assert len(pairs) == len(set(pairs))
    assert len(ids) == len(set(ids))
    assert contexts == {alias["context_id"] for alias in selected}
    old_alias_ids = {alias["alias_id"] for alias in primary["aliases"]}
    assert old_alias_ids.isdisjoint(set(ids))


def test_array_hash_rekeys(primary, range_input, plan):
    other = _clone(range_input)
    other["array_sha256"] = "9" * 64
    rekeyed = rp.build_range_plan(primary, other)
    old_ids = {job["job_id"] for job in plan["jobs"]}
    new_ids = {job["job_id"] for job in rekeyed["jobs"]}
    assert old_ids.isdisjoint(new_ids)
    assert plan["model_spec_sha256"] != rekeyed["model_spec_sha256"]
    assert plan["plan_sha256"] != rekeyed["plan_sha256"]
    for job in rekeyed["jobs"]:
        assert all(dependency in new_ids for dependency in job["dependencies"])


def test_dependency_cycle_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    job = _min_job(bad)
    job["dependencies"] = [job["job_id"]]
    _repin(monkeypatch, bad)
    _expect(
        "primary_dependency_cycle",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_missing_dependency_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    _min_job(bad)["dependencies"] = ["P08-UNREGISTERED"]
    _repin(monkeypatch, bad)
    _expect(
        "primary_dependency_unregistered",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_duplicate_job_id_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    bad["jobs"].append(_clone(_min_job(bad)))
    _repin(monkeypatch, bad)
    _expect(
        "primary_job_ids_duplicated",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_wrong_recipe_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    alias = _min_alias(bad)
    alias["recipe_id"] = "D1" if alias["recipe_id"] != "D1" else "D2"
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_model_mismatch",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_wrong_context_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    alias = _min_alias(bad)
    alias["context_id"] = alias["context_id"] + "-X"
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_context_mismatch",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_wrong_stage_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    alias = _min_alias(bad)
    candidate = next(
        job
        for job in bad["jobs"]
        if job["policy_id"] == _PRIMARY_POLICY_ID
        and job["model_id"] in _RETAINED_MODEL_IDS
        and job["stage"] != rp._SEED_STAGE
        and job["context_id"] == alias["context_id"]
    )
    alias["target_job_id"] = candidate["job_id"]
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_target_stage_invalid",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_missing_target_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    _min_alias(bad)["target_job_id"] = "P08-UNREGISTERED"
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_target_unregistered",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_duplicate_pair_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    bad["aliases"].append(_clone(_min_alias(bad)))
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_selection_duplicated",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_ids_duplicated_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    selected = [alias for alias in bad["aliases"] if alias["policy_id"] == _PRIMARY_POLICY_ID]
    first, second = selected[0], selected[1]
    assert (first["context_id"], first["strategy"]) != (
        second["context_id"],
        second["strategy"],
    )
    second["alias_id"] = first["alias_id"]
    _repin(monkeypatch, bad)
    _expect(
        "primary_alias_ids_duplicated",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_alias_id_not_string_rejected(primary, range_input, monkeypatch):
    for hostile in (_StrSubclass("P08ALIAS-x"), 17, None):
        bad = _clone(primary)
        _min_alias(bad)["alias_id"] = hostile
        _repin(monkeypatch, bad)
        _expect(
            "primary_alias_id_not_string",
            lambda bad=bad: rp.build_range_plan(bad, range_input),
        )


def test_alias_selection_empty_rejected(primary, range_input, monkeypatch):
    bad = _clone(primary)
    for alias in bad["aliases"]:
        if alias["policy_id"] == _PRIMARY_POLICY_ID:
            alias["policy_id"] = "PP-U-MAX"
    _repin(monkeypatch, bad)
    _expect(
        "range_alias_selection_empty",
        lambda: rp.build_range_plan(bad, range_input),
    )


def test_reason_codes_are_static():
    assert rp._FALLBACK_REASON_CODE in rp._REASON_CODES
    for code in rp._REASON_CODES:
        assert rp.RangePlanError(code).reason_code == code
    hostile = (
        "caller_sentinel",
        123,
        None,
        [],
        {},
        _StrSubclass("primary_plan_keys_invalid"),
    )
    for value in hostile:
        error = rp.RangePlanError(value)
        assert error.reason_code == rp._FALLBACK_REASON_CODE
        assert str(error) == rp._FALLBACK_REASON_CODE


def test_require_scientific_execution_always_denies():
    for forged in (True, {"allow": True}, "yes"):
        with pytest.raises(rp.RangePlanError) as excinfo:
            rp.require_scientific_execution(forged, allow=True)
        assert excinfo.value.reason_code == "scientific_execution_not_authorized"


@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("use_projection", [False, True])
def test_acquisition_classifier_interface(class_count, use_projection):
    torch = pytest.importorskip("torch")
    from atlas_sers.models.acquisition import AcquisitionClassifier

    with torch.random.fork_rng(devices=[]):
        model = AcquisitionClassifier(class_count, use_projection=use_projection)
        model.cpu()
        model.eval()
        state_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count == (
            208691 + (class_count - 3) * 65 + (4160 if use_projection else 0)
        )
        for width in (1401, 1450):
            values = torch.zeros(2, 1, width, device="cpu")
            with torch.no_grad():
                logits, embedding, projection = model(values)
            assert tuple(logits.shape) == (2, class_count)
            assert torch.isfinite(logits).all()
            assert tuple(embedding.shape) == (2, 64)
            assert torch.isfinite(embedding).all()
            if use_projection:
                assert projection is not None
                assert tuple(projection.shape) == (2, 64)
                assert torch.isfinite(projection).all()
            else:
                assert projection is None
        assert {
            name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
        } == state_shapes
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
