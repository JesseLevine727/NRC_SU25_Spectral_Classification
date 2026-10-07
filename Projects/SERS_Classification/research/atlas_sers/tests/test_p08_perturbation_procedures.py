"""Invented-fixture tests for the P08 universal stress procedure metadata adapter.

Only the pure, data-free adapter ``build_universal_procedure_records`` and the
frozen P08 planner are exercised.  The plan and the minimum-operation bridge
are invented in memory; they carry no real identities, no dataset, no
filesystem access and no model.  Every hash below is a hash of an invented
tag, so the fixtures are provenance-free by construction.

The adapter is an internal metadata extractor over an externally
authenticated plan.  It is deliberately not a full upstream DAG validator
and not a real artifact authenticator.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from atlas_sers.evaluation.p08_perturbation_procedures import (
    build_universal_procedure_records,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_plan import (
    CLASSICAL_MODELS,
    NEURAL_RECIPES,
    POLICIES,
    POLICY_REPRESENTATION,
    SEEDS,
    SVM_MODEL,
    SVM_SEED,
    build_universal_plan,
)

OUTPUT_SCHEMA = "nato-sers-p08-universal-stress-procedures-v1"
BRIDGE_SCHEMA = "nato-sers-p08-minimum-operation-evidence-v1"
REASON_MESSAGE = "invalid_stress_procedure_metadata"
EXECUTION_MESSAGE = "scientific_execution_not_authorized"

MIN_POLICY = "PP-U-MIN"
FUTURE_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
CTX_A = "CTX-A"
CTX_B = "CTX-B"
RECIPE_D0 = "D0-M"
RECIPE_B = "D1"
STRATEGY_D0 = "D0-M"
STRATEGY_SELECTED = "P05-SELECTED"
NOT_APPLICABLE = "not_applicable"

BRIDGE_STAGES = (
    "final_refit",
    "held_prediction",
    "scalar_calibration",
    "seed_ensemble_prediction",
)
FINAL_STAGES = frozenset(BRIDGE_STAGES)

CLASSICAL_REFIT_STATUS = "completed_fit_record_not_persisted_estimator"
ENDPOINT_STATUS = "complete_saved_pipeline_endpoint"
BOUND_STATUS = "bound_existing_record_or_array"

OUTPUT_KEYS = frozenset(
    (
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "exact_prediction_ledger_complete",
        "parent_plan_sha256",
        "minimal_bridge_metadata_sha256",
        "records",
        "strategy_aliases",
        "summary",
        "catalog_sha256",
    )
)
RECORD_KEYS = frozenset(
    (
        "context_id",
        "policy_id",
        "model_id",
        "representation_id",
        "array_sha256",
        "model_spec_sha256",
        "fit_uid_sha256",
        "test_uid_sha256",
        "seeds",
        "parent_plan_sha256",
        "model_reuse_mode",
        "calibration_order",
        "refit_references",
        "calibration_references",
        "held_reference_jobs",
        "clean_endpoint_reference",
        "procedure_id",
    )
)
REFERENCE_KEYS = frozenset(
    ("job_id", "seed", "historical_binding_sha256", "resolved_source_values")
)
ALIAS_KEYS = frozenset(
    (
        "policy_id",
        "context_id",
        "strategy",
        "recipe_id",
        "target_procedure_id",
        "upstream_alias_id",
    )
)
SUMMARY_KEYS = frozenset(
    (
        "procedure_count",
        "seed_estimator_count",
        "historical_classical_reconstruction_slots",
        "historical_neural_checkpoint_slots",
        "future_seed_estimator_slots",
        "reporting_alias_count",
        "by_policy",
    )
)


# ---------------------------------------------------------------------------
# Pure stdlib canonical hashing (no governance, no numpy, no third party).
# ---------------------------------------------------------------------------


def sha(tag):
    return hashlib.sha256(f"p08-stress::{tag}".encode()).hexdigest()


def sha256_value(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upstream_sha(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Invented fixture construction.
# ---------------------------------------------------------------------------


def _alpha_id(prefix, index):
    letters = "abcdefghijklmnopqrstuvwxyz"
    text = ""
    value = index
    while True:
        text = letters[value % 26] + text
        value = value // 26
        if value == 0:
            break
        value -= 1
    return f"{prefix}{text}"


def make_unit(unit_id, tag):
    return {
        "unit_id": unit_id,
        "fit_uid_sha256": sha(f"{tag}::fit::{unit_id}"),
        "validation_uid_sha256": sha(f"{tag}::val::{unit_id}"),
    }


def make_pseudo_context(context_id, recipe):
    selection = [make_unit(f"{context_id}-SEL-{index}", context_id) for index in range(2)]
    calibration = [make_unit(f"{context_id}-CAL-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selection_mode": "pseudo_domain",
        "selected_recipe_id": recipe,
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": selection,
        "calibration_units": calibration,
    }


def make_candidates():
    candidates = []
    for index in range(36):
        candidates.append(
            {
                "candidate_id": _alpha_id("svm-candidate-", index),
                "model_id": SVM_MODEL,
                "hyperparameter_sha256": sha(f"svm-hp-{index}"),
            }
        )
    for index in range(16):
        candidates.append(
            {
                "candidate_id": _alpha_id("rf-candidate-", index),
                "model_id": "C-RANDOM-FOREST",
                "hyperparameter_sha256": sha(f"rf-hp-{index}"),
            }
        )
    for index in range(16):
        candidates.append(
            {
                "candidate_id": _alpha_id("et-candidate-", index),
                "model_id": "C-EXTRA-TREES",
                "hyperparameter_sha256": sha(f"et-hp-{index}"),
            }
        )
    return candidates


def make_actions():
    return {
        representation: sha(f"array::{representation}")
        for representation in POLICY_REPRESENTATION.values()
    }


def make_model_spec():
    names = tuple(CLASSICAL_MODELS) + tuple(NEURAL_RECIPES)
    return {name: sha(f"spec::{name}") for name in names}


def build_fixture_plan():
    contexts = [
        make_pseudo_context(CTX_A, RECIPE_D0),
        make_pseudo_context(CTX_B, RECIPE_B),
    ]
    return build_universal_plan(
        contexts,
        make_candidates(),
        make_actions(),
        make_model_spec(),
    )


def _is_neural(model_id):
    return model_id in NEURAL_RECIPES


def _resolved_source_values(context_id, model_id, stage, seed):
    if _is_neural(model_id):
        if stage == "seed_ensemble_prediction":
            return {}
        return {
            "refit_id": f"refit-{context_id}-{model_id}-{seed}",
            "epochs": 30,
            "calibration_state_sha256": sha(
                f"neural-calibration::{context_id}::{model_id}::{seed}"
            ),
        }
    values = {
        "candidate_id": f"chosen-{context_id}-{model_id}",
        "hyperparameter_sha256": sha(f"classical-hp::{context_id}::{model_id}"),
        "selection_state_sha256": sha(f"classical-selection::{context_id}::{model_id}"),
    }
    if stage == "scalar_calibration":
        values["calibration_state_sha256"] = sha(f"classical-calibration::{context_id}::{model_id}")
    return values


def _evidence_status(is_neural, stage):
    if stage == "seed_ensemble_prediction":
        return ENDPOINT_STATUS
    if stage == "final_refit" and not is_neural:
        return CLASSICAL_REFIT_STATUS
    return BOUND_STATUS


def _bridge_binding(job):
    context_id = job["context_id"]
    model_id = job["model_id"]
    stage = job["stage"]
    seed = job["seed"]
    record = {
        "job_id": job["job_id"],
        "context_id": context_id,
        "model_id": model_id,
        "stage": stage,
        "seed": seed,
        "scientific_execution_authorized": False,
        "evidence": [{"synthetic_reference": "invented-only"}],
        "resolved_source_values": _resolved_source_values(context_id, model_id, stage, seed),
        "evidence_status": _evidence_status(_is_neural(model_id), stage),
    }
    record["binding_sha256"] = sha256_value(record)
    return record


def build_fixture_bridge(plan):
    bindings = [
        _bridge_binding(job)
        for job in plan["jobs"]
        if job["policy_id"] == MIN_POLICY and job["stage"] in BRIDGE_STAGES
    ]
    bindings.sort(key=lambda binding: binding["job_id"])
    return {
        "schema_version": BRIDGE_SCHEMA,
        "execution_authorized": False,
        "parent_plan_sha256": plan["plan_sha256"],
        "frozen_minimal_input": {},
        "operation_bindings": bindings,
        "fallback_endpoint_aliases": [],
        "verified_file_hashes": {},
        "boundary": {},
    }


def build_fixture_inputs():
    plan = build_fixture_plan()
    bridge = build_fixture_bridge(plan)
    return plan, bridge


def build_output():
    plan, bridge = build_fixture_inputs()
    return build_universal_procedure_records(plan=plan, minimal_bridge=bridge)


def _matching_jobs(plan, **criteria):
    return [
        job for job in plan["jobs"] if all(job.get(key) == value for key, value in criteria.items())
    ]


def _first_job(plan, **criteria):
    jobs = _matching_jobs(plan, **criteria)
    assert jobs, f"no job for {criteria!r}"
    return jobs[0]


def _first_binding(bridge, predicate):
    for binding in bridge["operation_bindings"]:
        if predicate(binding):
            return binding
    raise AssertionError("no matching bridge binding")


def _rehash_binding(binding):
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = sha256_value(body)


def _references(record):
    references = []
    references.extend(record["refit_references"])
    references.extend(record["calibration_references"])
    references.extend(record["held_reference_jobs"])
    references.append(record["clean_endpoint_reference"])
    return references


# ---------------------------------------------------------------------------
# Mutation finalization.
#
# The adapter authenticates individual relevant final job IDs and all alias
# IDs using P08JOB-/P08ALIAS- plus the canonical hash.  A content edit that is
# meant to exercise a semantic failure must therefore be finalized so that
# identifiers and dependency strings are internally consistent again.  The
# plan byte hash is caller-owned and is deliberately never recomputed here.
# ---------------------------------------------------------------------------


def finalize_mutations(plan, bridge=None, max_iterations=64):
    jobs = plan["jobs"]
    aliases = plan.get("aliases", [])

    initial_by_id = {}
    for job in jobs:
        initial_by_id.setdefault(job["job_id"], job)

    alias_targets = {id(alias): initial_by_id.get(alias["target_job_id"]) for alias in aliases}
    binding_targets = {}
    if bridge is not None:
        for binding in bridge["operation_bindings"]:
            binding_targets[id(binding)] = initial_by_id.get(binding["job_id"])

    for _ in range(max_iterations):
        by_id = {}
        for job in jobs:
            by_id.setdefault(job["job_id"], job)
        resolved = {id(job): [(by_id.get(dep), dep) for dep in job["dependencies"]] for job in jobs}
        new_ids = {}
        for job in jobs:
            deps = sorted(
                (target["job_id"] if target is not None else original)
                for target, original in resolved[id(job)]
            )
            body = {
                key: value for key, value in job.items() if key not in ("job_id", "dependencies")
            }
            body["dependencies"] = deps
            new_ids[id(job)] = "P08JOB-" + upstream_sha(body)

        stable = all(new_ids[id(job)] == job["job_id"] for job in jobs)

        for job in jobs:
            job["job_id"] = new_ids[id(job)]
        for job in jobs:
            job["dependencies"] = sorted(
                (target["job_id"] if target is not None else original)
                for target, original in resolved[id(job)]
            )
        if stable:
            break
    else:
        raise AssertionError("plan job identifiers did not converge")

    for alias in aliases:
        target = alias_targets.get(id(alias))
        if target is not None:
            alias["target_job_id"] = target["job_id"]
        alias["alias_id"] = "P08ALIAS-" + upstream_sha(
            {key: value for key, value in alias.items() if key != "alias_id"}
        )

    if bridge is not None:
        for binding in bridge["operation_bindings"]:
            target = binding_targets.get(id(binding))
            if target is not None and binding["job_id"] != target["job_id"]:
                binding["job_id"] = target["job_id"]
                _rehash_binding(binding)

    # Independent post-finalization audit.  A semantic mutation is only
    # accepted once every emitted identifier hashes back to its own finalized
    # content, so stale identifiers can never stand in for a real change.
    for job in jobs:
        body = {key: value for key, value in job.items() if key not in ("job_id", "dependencies")}
        body["dependencies"] = sorted(job["dependencies"])
        assert job["job_id"] == "P08JOB-" + upstream_sha(body), job["job_id"]
    for alias in aliases:
        body = {key: value for key, value in alias.items() if key != "alias_id"}
        assert alias["alias_id"] == "P08ALIAS-" + upstream_sha(body), alias["alias_id"]
    if bridge is not None:
        for binding in bridge["operation_bindings"]:
            body = {key: value for key, value in binding.items() if key != "binding_sha256"}
            assert binding["binding_sha256"] == sha256_value(body), binding["binding_sha256"]


# ---------------------------------------------------------------------------
# Positive tests.
# ---------------------------------------------------------------------------


def test_top_level_shape_and_denied_execution():
    plan, bridge = build_fixture_inputs()
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)

    assert set(output) == OUTPUT_KEYS
    assert output["schema_version"] == OUTPUT_SCHEMA
    assert output["execution_authorized"] is False
    assert output["scientific_operations"] == 0
    assert output["artifact_provenance_independently_verified"] is False
    assert output["exact_prediction_ledger_complete"] is False
    assert output["parent_plan_sha256"] == plan["plan_sha256"]
    assert output["minimal_bridge_metadata_sha256"] == sha256_value(bridge)
    assert isinstance(output["records"], list)
    assert isinstance(output["strategy_aliases"], list)
    assert isinstance(output["summary"], dict)
    assert len(output["catalog_sha256"]) == 64


def test_execution_hook_always_denied():
    for forged in (
        None,
        {},
        {"execution_authorized": True},
        {"execution_authorized": True, "scientific_operations": 0},
    ):
        with pytest.raises(ValueError) as caught:
            require_scientific_execution(forged)
        assert str(caught.value) == EXECUTION_MESSAGE


def test_exact_summary():
    output = build_output()
    summary = output["summary"]
    assert set(summary) == SUMMARY_KEYS
    assert summary["procedure_count"] == 27
    assert summary["seed_estimator_count"] == 69
    assert summary["historical_classical_reconstruction_slots"] == 14
    assert summary["historical_neural_checkpoint_slots"] == 9
    assert summary["future_seed_estimator_slots"] == 46
    assert summary["reporting_alias_count"] == 12
    assert set(summary["by_policy"]) == set(POLICIES)
    for policy in POLICIES:
        assert summary["by_policy"][policy] == {
            "procedures": 9,
            "seed_estimators": 23,
        }


def test_non_ascii_contexts_use_utf8_new_procedure_ids():
    plan = build_universal_plan(
        [
            make_pseudo_context("CTX-é", RECIPE_D0),
            make_pseudo_context("CTX-Ω", RECIPE_B),
        ],
        make_candidates(),
        make_actions(),
        make_model_spec(),
    )
    bridge = build_fixture_bridge(plan)
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)

    assert output["summary"]["procedure_count"] == 27
    assert output["summary"]["seed_estimator_count"] == 69
    context_ids = {record["context_id"] for record in output["records"]}
    assert {"CTX-é", "CTX-Ω"} <= context_ids

    for record in output["records"]:
        body = {key: value for key, value in record.items() if key != "procedure_id"}
        utf8_hash = sha256_value(body)
        ascii_hash = upstream_sha(body)
        assert ascii_hash != utf8_hash
        assert record["procedure_id"].endswith(utf8_hash)
        assert not record["procedure_id"].endswith(ascii_hash)


def test_benign_nonsemantic_unit_label_repair_preserves_dependencies():
    plan, bridge = build_fixture_inputs()
    refit = _first_job(
        plan,
        policy_id=MIN_POLICY,
        model_id=SVM_MODEL,
        stage="final_refit",
        seed=SVM_SEED,
    )
    refit["unit_id"] = "invented-publication-label"

    finalize_mutations(plan, bridge)

    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    assert output["summary"]["procedure_count"] == 27
    assert output["summary"]["seed_estimator_count"] == 69


def test_resolved_source_values_are_deep_copied_into_references():
    plan, bridge = build_fixture_inputs()
    binding = _first_binding(
        bridge,
        lambda item: (
            item["stage"] == "final_refit"
            and item["model_id"] == SVM_MODEL
            and item["seed"] == SVM_SEED
        ),
    )
    binding["resolved_source_values"]["audit_note"] = {"items": ["original"]}
    _rehash_binding(binding)

    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)

    reference = None
    for record in output["records"]:
        for candidate in record["refit_references"]:
            if candidate["job_id"] == binding["job_id"]:
                reference = candidate
    assert reference is not None
    assert reference["resolved_source_values"]["audit_note"] == {"items": ["original"]}

    binding["resolved_source_values"]["audit_note"]["items"].append("original-mutated")
    assert reference["resolved_source_values"]["audit_note"]["items"] == ["original"]
    assert binding["resolved_source_values"]["audit_note"]["items"] == [
        "original",
        "original-mutated",
    ]

    reference["resolved_source_values"]["audit_note"]["items"].append("output-mutated")
    assert binding["resolved_source_values"]["audit_note"]["items"] == [
        "original",
        "original-mutated",
    ]

    _rehash_binding(binding)

    second = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    second_reference = None
    for record in second["records"]:
        for candidate in record["refit_references"]:
            if candidate["job_id"] == binding["job_id"]:
                second_reference = candidate
    assert second_reference is not None
    assert second_reference is not reference
    assert second_reference["resolved_source_values"]["audit_note"]["items"] == [
        "original",
        "original-mutated",
    ]


def test_record_keys_and_ordering():
    output = build_output()
    records = output["records"]
    assert len(records) == 27
    for record in records:
        assert set(record) == RECORD_KEYS
    order = [(r["policy_id"], r["context_id"], r["model_id"]) for r in records]
    assert order == sorted(order)
    procedure_ids = [record["procedure_id"] for record in records]
    assert len(procedure_ids) == len(set(procedure_ids))
    for procedure_id in procedure_ids:
        assert procedure_id.startswith("P08STRESSPROC-")
        assert len(procedure_id) == len("P08STRESSPROC-") + 64


def test_procedure_id_and_catalog_hash_recomputed():
    output = build_output()
    for record in output["records"]:
        body = {key: value for key, value in record.items() if key != "procedure_id"}
        assert record["procedure_id"] == "P08STRESSPROC-" + sha256_value(body)
    body = {key: value for key, value in output.items() if key != "catalog_sha256"}
    assert output["catalog_sha256"] == sha256_value(body)


def test_modes_and_calibration_orders():
    output = build_output()
    for record in output["records"]:
        neural = record["model_id"] in NEURAL_RECIPES
        if neural:
            assert record["calibration_order"] == ("per_seed_temperature_then_seed_average")
        else:
            assert record["calibration_order"] == ("seed_average_then_single_temperature")
        if record["policy_id"] == MIN_POLICY:
            expected = (
                "historical_neural_checkpoint" if neural else "historical_classical_reconstruction"
            )
        else:
            assert record["policy_id"] in FUTURE_POLICIES
            expected = "future_retained_estimator"
        assert record["model_reuse_mode"] == expected


def test_seeds_and_seed_estimator_totals():
    output = build_output()
    per_policy = {}
    for record in output["records"]:
        if record["model_id"] == SVM_MODEL:
            assert list(record["seeds"]) == [SVM_SEED]
        else:
            assert list(record["seeds"]) == list(SEEDS)
        per_policy[record["policy_id"]] = per_policy.get(record["policy_id"], 0) + len(
            record["seeds"]
        )
    for policy in POLICIES:
        assert per_policy[policy] == 23


def test_record_provenance_bindings_match_plan():
    plan, bridge = build_fixture_inputs()
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    for record in output["records"]:
        assert record["representation_id"] == POLICY_REPRESENTATION[record["policy_id"]]
        assert record["parent_plan_sha256"] == output["parent_plan_sha256"]
        assert len(record["array_sha256"]) == 64
        assert len(record["model_spec_sha256"]) == 64
        assert len(record["fit_uid_sha256"]) == 64
        assert len(record["test_uid_sha256"]) == 64


def test_context_outer_hashes_shared_across_policies():
    output = build_output()
    for context_id in (CTX_A, CTX_B):
        records = [r for r in output["records"] if r["context_id"] == context_id]
        assert records
        assert len({record["fit_uid_sha256"] for record in records}) == 1
        assert len({record["test_uid_sha256"] for record in records}) == 1


def test_reference_shape_and_historical_mapping():
    plan, bridge = build_fixture_inputs()
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    bindings = {b["job_id"]: b for b in bridge["operation_bindings"]}
    for record in output["records"]:
        for reference in _references(record):
            assert set(reference) == REFERENCE_KEYS
            if record["policy_id"] == MIN_POLICY:
                binding = bindings[reference["job_id"]]
                assert reference["historical_binding_sha256"] == binding["binding_sha256"]
                assert reference["resolved_source_values"] == binding["resolved_source_values"]
            else:
                assert reference["historical_binding_sha256"] is None
                assert reference["resolved_source_values"] is None
        if record["policy_id"] == MIN_POLICY:
            for reference in record["refit_references"]:
                if record["model_id"] in CLASSICAL_MODELS:
                    assert (
                        bindings[reference["job_id"]]["evidence_status"] == CLASSICAL_REFIT_STATUS
                    )
            assert (
                bindings[record["clean_endpoint_reference"]["job_id"]]["evidence_status"]
                == ENDPOINT_STATUS
            )


def test_reference_order_follows_seed_order():
    output = build_output()
    for record in output["records"]:
        assert [ref["seed"] for ref in record["refit_references"]] == list(record["seeds"])
        assert [ref["seed"] for ref in record["held_reference_jobs"]] == list(record["seeds"])
        if record["model_id"] in NEURAL_RECIPES:
            assert [ref["seed"] for ref in record["calibration_references"]] == list(
                record["seeds"]
            )
        else:
            assert len(record["calibration_references"]) == 1
            assert record["calibration_references"][0]["seed"] == NOT_APPLICABLE
        assert record["clean_endpoint_reference"]["seed"] == NOT_APPLICABLE


def test_historical_bridge_values_are_deep_copies_not_aliases():
    plan, bridge = build_fixture_inputs()
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    for binding in bridge["operation_bindings"]:
        binding["resolved_source_values"] = {"mutated": True}
    for record in output["records"]:
        if record["policy_id"] != MIN_POLICY:
            continue
        for reference in _references(record):
            assert reference["resolved_source_values"] != {"mutated": True}


def test_strategy_aliases_and_cross_policy_recipes():
    plan, bridge = build_fixture_inputs()
    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    aliases = output["strategy_aliases"]
    assert len(aliases) == 12
    for alias in aliases:
        assert set(alias) == ALIAS_KEYS
    procedures = {
        (record["policy_id"], record["context_id"], record["model_id"]): record["procedure_id"]
        for record in output["records"]
    }
    plan_alias_ids = {alias["alias_id"] for alias in plan["aliases"]}
    for alias in aliases:
        assert alias["upstream_alias_id"] in plan_alias_ids
        assert (
            alias["target_procedure_id"]
            == procedures[(alias["policy_id"], alias["context_id"], alias["recipe_id"])]
        )
    for policy in POLICIES:
        context_a = [
            alias
            for alias in aliases
            if alias["policy_id"] == policy and alias["context_id"] == CTX_A
        ]
        assert len(context_a) == 2
        assert {alias["strategy"] for alias in context_a} == {
            STRATEGY_D0,
            STRATEGY_SELECTED,
        }
        assert len({alias["target_procedure_id"] for alias in context_a}) == 1
        assert {alias["recipe_id"] for alias in context_a} == {RECIPE_D0}

        context_b = [
            alias
            for alias in aliases
            if alias["policy_id"] == policy and alias["context_id"] == CTX_B
        ]
        assert len(context_b) == 2
        assert {alias["strategy"] for alias in context_b} == {
            STRATEGY_D0,
            STRATEGY_SELECTED,
        }
        assert len({alias["target_procedure_id"] for alias in context_b}) == 2
        assert {alias["recipe_id"] for alias in context_b} == {RECIPE_D0, RECIPE_B}


def test_extra_source_bindings_and_ignored_stages_are_tolerated():
    base = build_output()
    plan, bridge = build_fixture_inputs()

    source_job = _first_job(plan, policy_id=MIN_POLICY, stage="source_fit")
    extra_binding = {
        "job_id": source_job["job_id"],
        "context_id": source_job["context_id"],
        "model_id": source_job["model_id"],
        "stage": source_job["stage"],
        "seed": source_job["seed"],
        "scientific_execution_authorized": False,
        "evidence": [{"synthetic_reference": "invented-only"}],
        "resolved_source_values": {},
        "evidence_status": BOUND_STATUS,
    }
    _rehash_binding(extra_binding)
    bridge["operation_bindings"].append(extra_binding)
    bridge["operation_bindings"].sort(key=lambda binding: binding["job_id"])

    template = _first_job(plan, stage="source_validation_prediction")
    ignored_job = copy.deepcopy(template)
    ignored_job["unit_id"] = "CTX-A-IGNORED-EXTRA-UNIT"
    ignored_job["job_id"] = "P08JOB-" + sha256_value(
        {key: value for key, value in ignored_job.items() if key != "job_id"}
    )
    plan["jobs"].append(ignored_job)
    plan["jobs"].sort(key=lambda job: job["job_id"])

    output = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    assert output["records"] == base["records"]
    assert output["strategy_aliases"] == base["strategy_aliases"]
    assert output["summary"] == base["summary"]


def test_extra_binding_duplicate_ids_are_still_rejected():
    plan, bridge = build_fixture_inputs()
    source_job = _first_job(plan, policy_id=MIN_POLICY, stage="source_fit")
    binding = {
        "job_id": source_job["job_id"],
        "context_id": source_job["context_id"],
        "model_id": source_job["model_id"],
        "stage": source_job["stage"],
        "seed": source_job["seed"],
        "scientific_execution_authorized": False,
        "evidence": [{"synthetic_reference": "invented-only"}],
        "resolved_source_values": {},
        "evidence_status": BOUND_STATUS,
    }
    _rehash_binding(binding)
    bridge["operation_bindings"].append(binding)
    bridge["operation_bindings"].append(copy.deepcopy(binding))
    with pytest.raises(ValueError) as caught:
        build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    assert str(caught.value) == REASON_MESSAGE


def test_semantic_stability_under_reversed_inputs():
    plan, bridge = build_fixture_inputs()
    base = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)

    shuffled_plan = copy.deepcopy(plan)
    shuffled_plan["jobs"] = list(reversed(shuffled_plan["jobs"]))
    shuffled_plan["aliases"] = list(reversed(shuffled_plan["aliases"]))
    shuffled_bridge = copy.deepcopy(bridge)
    shuffled_bridge["operation_bindings"] = list(reversed(shuffled_bridge["operation_bindings"]))
    shuffled = build_universal_procedure_records(plan=shuffled_plan, minimal_bridge=shuffled_bridge)

    assert base["records"] == shuffled["records"]
    assert base["strategy_aliases"] == shuffled["strategy_aliases"]
    assert shuffled["minimal_bridge_metadata_sha256"] == sha256_value(shuffled_bridge)
    assert shuffled["catalog_sha256"] == sha256_value(
        {key: value for key, value in shuffled.items() if key != "catalog_sha256"}
    )


def test_repeatability_nonmutation_and_no_output_aliasing():
    plan, bridge = build_fixture_inputs()
    plan_copy = copy.deepcopy(plan)
    bridge_copy = copy.deepcopy(bridge)

    first = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    second = build_universal_procedure_records(plan=plan, minimal_bridge=bridge)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert plan == plan_copy
    assert bridge == bridge_copy
    assert first["records"][0] is not second["records"][0]
    original = second["records"][0]["context_id"]
    first["records"][0]["context_id"] = "MUTATED-CONTEXT"
    assert second["records"][0]["context_id"] == original


def test_strict_json_roundtrip():
    output = build_output()
    encoded = json.dumps(output, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["catalog_sha256"] == output["catalog_sha256"]
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


# ---------------------------------------------------------------------------
# Negative fixtures.  Each mutation is explicit and self-describing.
# Mutation functions that are meant to exercise a semantic failure are
# finalized by the test harness so that identifiers and dependency strings
# are consistent and the failure is not a trivial stale-hash error.
# ---------------------------------------------------------------------------


def mutate_plan_execution_flag_true(plan, bridge):
    plan["execution_authorized"] = True


def mutate_bridge_execution_flag_true(plan, bridge):
    bridge["execution_authorized"] = True


def mutate_bridge_parent_mismatch(plan, bridge):
    bridge["parent_plan_sha256"] = sha("wrong-parent-plan")


def mutate_duplicate_plan_job_id(plan, bridge):
    job = _first_job(plan, policy_id=MIN_POLICY, stage="held_prediction")
    plan["jobs"].append(copy.deepcopy(job))


def mutate_duplicate_bridge_entry(plan, bridge):
    bridge["operation_bindings"].append(copy.deepcopy(bridge["operation_bindings"][0]))


def mutate_remove_min_bridge_entry(plan, bridge):
    bridge["operation_bindings"].pop(0)


def mutate_tamper_binding_hash(plan, bridge):
    bridge["operation_bindings"][0]["binding_sha256"] = sha("tampered-binding-hash")


def mutate_stale_job_id(plan, bridge):
    job = _first_job(
        plan,
        policy_id=MIN_POLICY,
        context_id=CTX_A,
        stage="held_prediction",
    )
    job["seed"] = 19990101


def mutate_stale_alias_id(plan, bridge):
    alias = next(alias for alias in plan["aliases"] if alias["strategy"] == STRATEGY_SELECTED)
    alias["recipe_id"] = "D3"


def mutate_wrong_bridge_seed_rehashed(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: (
            item["stage"] == "final_refit"
            and item["model_id"] in ("C-RANDOM-FOREST", "C-EXTRA-TREES")
        ),
    )
    binding["seed"] = 20260101
    _rehash_binding(binding)


def mutate_endpoint_representation(plan, bridge):
    job = _first_job(
        plan,
        policy_id=MIN_POLICY,
        context_id=CTX_A,
        stage="seed_ensemble_prediction",
    )
    job["representation_id"] = "R-NOT-A-REPRESENTATION"


def mutate_inconsistent_fit_role(plan, bridge):
    group = _matching_jobs(
        plan,
        policy_id=MIN_POLICY,
        context_id=CTX_A,
        model_id="C-RANDOM-FOREST",
        stage="final_refit",
    )
    assert len(group) == 3
    group[0]["fit_uid_sha256"] = sha("inconsistent-fit-role")


def mutate_drop_refit_seed(plan, bridge):
    for index, job in enumerate(plan["jobs"]):
        if (
            job["policy_id"] == MIN_POLICY
            and job["context_id"] == CTX_A
            and job["model_id"] == "C-RANDOM-FOREST"
            and job["stage"] == "final_refit"
        ):
            del plan["jobs"][index]
            return
    raise AssertionError("no classical refit seed to drop")


def mutate_bool_seed(plan, bridge):
    job = _first_job(plan, policy_id=MIN_POLICY, context_id=CTX_A, stage="final_refit")
    job["seed"] = True


def mutate_missing_extraordinary_seed(plan, bridge):
    for index, job in enumerate(plan["jobs"]):
        if (
            job["policy_id"] == MIN_POLICY
            and job["context_id"] == CTX_A
            and job["model_id"] == SVM_MODEL
            and job["stage"] == "final_refit"
            and job["seed"] == SVM_SEED
        ):
            del plan["jobs"][index]
            return
    raise AssertionError("no deterministic SVM refit seed to remove")


def mutate_wrong_held_dependency(plan, bridge):
    source = _first_job(plan, policy_id=MIN_POLICY, context_id=CTX_A, stage="source_fit")
    job = _first_job(
        plan,
        policy_id=MIN_POLICY,
        context_id=CTX_A,
        model_id=SVM_MODEL,
        stage="held_prediction",
    )
    job["dependencies"] = [source["job_id"]]


def mutate_duplicate_endpoint_dependency(plan, bridge):
    job = _first_job(
        plan,
        policy_id=MIN_POLICY,
        context_id=CTX_A,
        model_id=SVM_MODEL,
        stage="seed_ensemble_prediction",
    )
    dependencies = list(job["dependencies"])
    job["dependencies"] = sorted(dependencies + [dependencies[0]])


def mutate_wrong_neural_calibration_cross_seed(plan, bridge):
    scalars = {
        job["seed"]: job
        for job in _matching_jobs(
            plan,
            policy_id=MIN_POLICY,
            context_id=CTX_A,
            model_id=RECIPE_D0,
            stage="scalar_calibration",
        )
    }
    assert len(scalars) == len(SEEDS)
    scalar_ids = {job["job_id"] for job in scalars.values()}
    for job in plan["jobs"]:
        if (
            job["policy_id"] == MIN_POLICY
            and job["context_id"] == CTX_A
            and job["model_id"] == RECIPE_D0
            and job["stage"] == "held_prediction"
            and job["seed"] == SEEDS[0]
        ):
            remaining = [dep for dep in job["dependencies"] if dep not in scalar_ids]
            job["dependencies"] = sorted(remaining + [scalars[SEEDS[1]]["job_id"]])
            return
    raise AssertionError("no neural held prediction to retarget")


def mutate_min_classical_refit_status(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["stage"] == "final_refit" and item["model_id"] in CLASSICAL_MODELS,
    )
    binding["evidence_status"] = BOUND_STATUS
    _rehash_binding(binding)


def mutate_min_classical_candidate_mismatch(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["stage"] == "final_refit" and item["model_id"] == "C-RANDOM-FOREST",
    )
    binding["resolved_source_values"]["candidate_id"] = "chosen-somewhere-else"
    _rehash_binding(binding)


def mutate_neural_wrong_epoch_rehashed(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["model_id"] in NEURAL_RECIPES and item["stage"] == "scalar_calibration",
    )
    binding["resolved_source_values"]["epochs"] = 31
    _rehash_binding(binding)


def mutate_neural_wrong_refit_id_rehashed(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["model_id"] in NEURAL_RECIPES and item["stage"] == "final_refit",
    )
    binding["resolved_source_values"]["refit_id"] = "refit-somewhere-else"
    _rehash_binding(binding)


def mutate_neural_bool_epoch_rehashed(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["model_id"] in NEURAL_RECIPES and item["stage"] == "held_prediction",
    )
    binding["resolved_source_values"]["epochs"] = True
    _rehash_binding(binding)


def mutate_neural_nonpositive_epoch_rehashed(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["model_id"] in NEURAL_RECIPES and item["stage"] == "scalar_calibration",
    )
    binding["resolved_source_values"]["epochs"] = 0
    _rehash_binding(binding)


def mutate_classical_missing_calibration_state(plan, bridge):
    binding = _first_binding(
        bridge,
        lambda item: item["model_id"] in CLASSICAL_MODELS and item["stage"] == "scalar_calibration",
    )
    del binding["resolved_source_values"]["calibration_state_sha256"]
    _rehash_binding(binding)


def mutate_bad_alias_target(plan, bridge):
    plan["aliases"][0]["target_job_id"] = "P08JOB-" + "0" * 64


def mutate_duplicate_d0_alias(plan, bridge):
    alias = next(alias for alias in plan["aliases"] if alias["strategy"] == STRATEGY_D0)
    plan["aliases"].append(copy.deepcopy(alias))


def mutate_selected_recipe_inconsistent(plan, bridge):
    for alias in plan["aliases"]:
        if (
            alias["strategy"] == STRATEGY_SELECTED
            and alias["context_id"] == CTX_B
            and alias["policy_id"] == POLICIES[0]
        ):
            alias["recipe_id"] = "D2"
            return
    raise AssertionError("no selected-strategy alias to retarget")


def mutate_missing_classical_group(plan, bridge):
    plan["jobs"] = [
        job
        for job in plan["jobs"]
        if not (
            job["policy_id"] == MIN_POLICY
            and job["context_id"] == CTX_A
            and job["model_id"] == SVM_MODEL
            and job["stage"] in FINAL_STAGES
        )
    ]


def mutate_cross_policy_fit_mismatch(plan, bridge):
    new_fit = sha("mutated-cross-policy-fit")
    touched = 0
    for job in plan["jobs"]:
        if (
            job["policy_id"] == FUTURE_POLICIES[0]
            and job["context_id"] == CTX_A
            and job["stage"] in ("final_refit", "held_prediction")
        ):
            job["fit_uid_sha256"] = new_fit
            touched += 1
    assert touched, "no cross-policy fit binding to mutate"


def mutate_cross_model_test_mismatch(plan, bridge):
    new_test = sha("mutated-cross-model-test")
    touched = 0
    for job in plan["jobs"]:
        if (
            job["policy_id"] == MIN_POLICY
            and job["context_id"] == CTX_A
            and job["model_id"] == SVM_MODEL
        ):
            job["test_uid_sha256"] = new_test
            touched += 1
    assert touched, "no cross-model test binding to mutate"


NEGATIVE_MUTATIONS = (
    (mutate_plan_execution_flag_true, False),
    (mutate_bridge_execution_flag_true, False),
    (mutate_bridge_parent_mismatch, False),
    (mutate_duplicate_plan_job_id, False),
    (mutate_duplicate_bridge_entry, False),
    (mutate_remove_min_bridge_entry, False),
    (mutate_tamper_binding_hash, False),
    (mutate_stale_job_id, False),
    (mutate_stale_alias_id, False),
    (mutate_wrong_bridge_seed_rehashed, False),
    (mutate_endpoint_representation, True),
    (mutate_inconsistent_fit_role, True),
    (mutate_drop_refit_seed, True),
    (mutate_bool_seed, True),
    (mutate_missing_extraordinary_seed, True),
    (mutate_wrong_held_dependency, True),
    (mutate_duplicate_endpoint_dependency, True),
    (mutate_wrong_neural_calibration_cross_seed, True),
    (mutate_min_classical_refit_status, False),
    (mutate_min_classical_candidate_mismatch, False),
    (mutate_neural_wrong_epoch_rehashed, False),
    (mutate_neural_wrong_refit_id_rehashed, False),
    (mutate_neural_bool_epoch_rehashed, False),
    (mutate_neural_nonpositive_epoch_rehashed, False),
    (mutate_classical_missing_calibration_state, False),
    (mutate_bad_alias_target, True),
    (mutate_duplicate_d0_alias, False),
    (mutate_selected_recipe_inconsistent, True),
    (mutate_missing_classical_group, True),
    (mutate_cross_policy_fit_mismatch, True),
    (mutate_cross_model_test_mismatch, True),
)


@pytest.mark.parametrize(
    "mutate,repair",
    NEGATIVE_MUTATIONS,
    ids=[mutation.__name__ for mutation, _ in NEGATIVE_MUTATIONS],
)
def test_malformed_metadata_is_rejected(mutate, repair):
    plan, bridge = build_fixture_inputs()
    mutate(plan, bridge)
    if repair:
        finalize_mutations(plan, bridge)
    with pytest.raises(ValueError) as caught:
        build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    assert str(caught.value) == REASON_MESSAGE


# ---------------------------------------------------------------------------
# Malformed adapter inputs.
# ---------------------------------------------------------------------------


MALFORMED_INPUT_CASES = (
    "plan_not_mapping",
    "bridge_not_mapping",
    "bindings_not_sequence",
    "binding_not_mapping",
    "binding_model_unhashable",
    "binding_seed_list",
    "binding_nonfinite_resolved_value",
)


@pytest.mark.parametrize("case", MALFORMED_INPUT_CASES, ids=list(MALFORMED_INPUT_CASES))
def test_malformed_adapter_inputs_raise_sanitized_error(case):
    plan, bridge = build_fixture_inputs()
    if case == "plan_not_mapping":
        plan = []
    elif case == "bridge_not_mapping":
        bridge = []
    elif case == "bindings_not_sequence":
        bridge["operation_bindings"] = {}
    elif case == "binding_not_mapping":
        bridge["operation_bindings"][0] = "not-a-binding"
    elif case == "binding_model_unhashable":
        binding = bridge["operation_bindings"][0]
        binding["model_id"] = ["unhashable"]
        _rehash_binding(binding)
    elif case == "binding_seed_list":
        binding = bridge["operation_bindings"][0]
        binding["seed"] = []
        _rehash_binding(binding)
    elif case == "binding_nonfinite_resolved_value":
        binding = bridge["operation_bindings"][0]
        binding["resolved_source_values"] = {"value": float("inf")}
        binding["binding_sha256"] = "0" * 64
    else:
        raise AssertionError(f"unhandled malformed case: {case}")

    with pytest.raises(ValueError) as caught:
        build_universal_procedure_records(plan=plan, minimal_bridge=bridge)
    assert str(caught.value) == REASON_MESSAGE
