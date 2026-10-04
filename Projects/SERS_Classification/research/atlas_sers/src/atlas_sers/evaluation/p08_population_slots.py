"""P08-T178 population operation catalog (metadata only).

This module enumerates the *universal* MIN / SG / arPLS comparison operations
for a regenerated population without executing, fitting, calibrating, scoring
or selecting anything.  It is a catalogue of future, outcome-dependent
operations:

* classical tuning is repeated inside every registered family/policy and is
  delegated to the frozen ``p08_plan`` helpers;
* the neural recipe is chosen afresh from a context's MIN source evidence
  using the locked P05 G3 rule, then frozen across the three policies;
* two panel-planning alternatives are exposed through ``include_extra_trees``
  (four methods versus five); the flag is planning metadata only and never an
  approval, a launch or a scientific decision.

The module writes nothing, draws no random numbers and performs no scientific
computation; it imports no training runtime and relies only on metadata
helpers.  Caller-provided hashes only prove internal binding.  They do not
establish provenance, physical disjointness, master isolation or that a
selection will succeed; the caller's prior role audit remains mandatory.

``require_scientific_execution`` always refuses, even for forged permission
flags.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

from atlas_sers.governance.canonical import sha256_value

from . import p08_plan as core
from .p05_core_plan import validate_core_contract

__all__ = ["build_population_slots", "require_scientific_execution"]


SCHEMA_VERSION = "nato-sers-p08-population-slot-catalog-v1"

UNSUPPORTED_MODE = "unsupported"
NOT_APPLICABLE_MODE = "not_applicable"
PLAN_MODES = (core.MASTER_MODE, core.PSEUDO_MODE, UNSUPPORTED_MODE, NOT_APPLICABLE_MODE)

JOB_NAMESPACE = "P08POPJOB-"
ALIAS_NAMESPACE = "P08POPALIAS-"
EXCLUDED_NAMESPACE = "P08POPEXCL-"

SELECT_NEURAL_RECIPE_STAGE = "select_neural_recipe"
GUARD_SOURCE_FIT_STAGE = "guard_source_fit"
GUARD_VALIDATION_PREDICTION_STAGE = "guard_validation_prediction"
SELECT_NEURAL_RECIPE_RESOLUTION = "source_only_recipe_selection"

NEURAL_SCALAR_RESOLUTION = "master_equal_source_logits_then_seed_probability_average"
NEURAL_SCALAR_ROLE_PURPOSE = "inherited_selection_only"
ALIAS_READY_RULE = "await_ready_for_refit"

NON_D0_RECIPES = ("D1", "D2", "D3")
POP_MODEL_FIT_STAGES = frozenset(
    ("source_fit", GUARD_SOURCE_FIT_STAGE, "calibration_model_fit", "final_refit")
)

CONTEXT_FIELDS = (
    "context_id",
    "selection_mode",
    "domain_eligible",
    "classical_supported",
    "neural_supported",
    "g3_comparable",
    "unavailable_reasons",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
    "selection_units",
    "calibration_units",
    "guard_units",
)

UNIT_FIELDS = ("unit_id", "fit_uid_sha256", "validation_uid_sha256")
GUARD_FIELDS = UNIT_FIELDS + ("support", "exclusion_reason")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this readiness catalogue."""
    raise ValueError("scientific_execution_not_authorized")


def _fail(code):
    raise core.PlanError(code)


_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_mapping(value, code):
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _require_sequence(value, code):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _require_identifier(value, code):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    return value


def _require_sha256(value, code):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX_DIGITS for character in value)
    ):
        _fail(code)
    return value


def _require_exact_keys(value, keys, code):
    if set(value) != set(keys):
        _fail(code)
    return value


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def _parse_units(raw_units, expected_fields, sequence_code, guard=False):
    _require_sequence(raw_units, sequence_code)
    seen = set()
    parsed = []
    for raw in raw_units:
        _require_mapping(raw, "unit_must_be_mapping")
        _require_exact_keys(raw, expected_fields, "unit_keys_invalid")
        unit_id = raw["unit_id"]
        _require_identifier(unit_id, "unit_id_invalid")
        if unit_id in seen:
            _fail("duplicate_unit_id")
        seen.add(unit_id)
        _require_sha256(raw["fit_uid_sha256"], "fit_uid_hash_invalid")
        _require_sha256(raw["validation_uid_sha256"], "validation_uid_hash_invalid")
        if raw["fit_uid_sha256"] == raw["validation_uid_sha256"]:
            _fail("role_fit_validation_hashes_equal")
        unit = {
            "unit_id": unit_id,
            "fit_uid_sha256": raw["fit_uid_sha256"],
            "validation_uid_sha256": raw["validation_uid_sha256"],
        }
        if guard:
            support = raw["support"]
            if not isinstance(support, bool):
                _fail("guard_support_invalid")
            reason = raw["exclusion_reason"]
            if support:
                if reason is not None:
                    _fail("guard_exclusion_reason_invalid")
            elif not isinstance(reason, str) or not reason or reason != reason.strip():
                _fail("guard_exclusion_reason_invalid")
            unit["support"] = support
            unit["exclusion_reason"] = reason
        parsed.append(unit)
    parsed.sort(key=lambda unit: unit["unit_id"])
    return parsed


def _derive_classical_supported(ctx):
    if not ctx["domain_eligible"]:
        return False
    mode = ctx["selection_mode"]
    if mode == core.MASTER_MODE:
        selection_ok = len(ctx["selection_units"]) == 3
    elif mode == core.PSEUDO_MODE:
        selection_ok = len(ctx["selection_units"]) >= 2
    else:
        return False
    return selection_ok and len(ctx["calibration_units"]) == 3


def _derive_neural_supported(ctx):
    if not ctx["domain_eligible"]:
        return False
    mode = ctx["selection_mode"]
    if mode == core.MASTER_MODE:
        return len(ctx["selection_units"]) == 3 and len(ctx["guard_units"]) == 0
    if mode == core.PSEUDO_MODE:
        return len(ctx["selection_units"]) >= 2 and len(ctx["guard_units"]) == 3
    return False


def _derive_g3_comparable(ctx):
    return (
        ctx["neural_supported"]
        and ctx["selection_mode"] == core.PSEUDO_MODE
        and len(ctx["guard_units"]) == 3
        and all(guard["support"] for guard in ctx["guard_units"])
    )


def _parse_population_contexts(contexts):
    core._require_sequence(contexts, "contexts_must_be_sequence")
    if not contexts:
        _fail("contexts_empty")
    seen = set()
    parsed = []
    for raw in contexts:
        _require_mapping(raw, "context_must_be_mapping")
        _require_exact_keys(raw, CONTEXT_FIELDS, "context_keys_invalid")

        context_id = raw["context_id"]
        _require_identifier(context_id, "context_id_invalid")
        if context_id in seen:
            _fail("duplicate_context_id")
        seen.add(context_id)

        mode = raw["selection_mode"]
        _require_identifier(mode, "selection_mode_invalid")
        if mode not in PLAN_MODES:
            _fail("selection_mode_invalid")

        for flag in ("domain_eligible", "classical_supported", "neural_supported", "g3_comparable"):
            if not isinstance(raw[flag], bool):
                _fail("support_flag_invalid")

        eligible = raw["domain_eligible"]
        if eligible and mode == NOT_APPLICABLE_MODE:
            _fail("eligibility_mode_contradiction")
        if not eligible and mode != NOT_APPLICABLE_MODE:
            _fail("eligibility_mode_contradiction")

        reasons = raw["unavailable_reasons"]
        _require_sequence(reasons, "unavailable_reasons_invalid")
        cleaned_reasons = []
        seen_reasons = set()
        for reason in reasons:
            _require_identifier(reason, "unavailable_reasons_invalid")
            if reason in seen_reasons:
                _fail("unavailable_reasons_duplicate")
            seen_reasons.add(reason)
            cleaned_reasons.append(reason)
        cleaned_reasons.sort()

        outer_fit = raw["outer_fit_uid_sha256"]
        outer_test = raw["outer_test_uid_sha256"]
        _require_sha256(outer_fit, "outer_fit_uid_hash_invalid")
        _require_sha256(outer_test, "outer_test_uid_hash_invalid")
        if outer_fit == outer_test:
            _fail("outer_fit_test_equal")

        selection_units = _parse_units(
            raw["selection_units"], UNIT_FIELDS, "selection_units_must_be_sequence"
        )
        calibration_units = _parse_units(
            raw["calibration_units"], UNIT_FIELDS, "calibration_units_must_be_sequence"
        )
        guard_units = _parse_units(
            raw["guard_units"], GUARD_FIELDS, "guard_units_must_be_sequence", guard=True
        )

        for unit in selection_units + calibration_units + guard_units:
            if outer_test in (unit["fit_uid_sha256"], unit["validation_uid_sha256"]):
                _fail("role_hash_matches_outer_test")

        if not eligible or mode in (UNSUPPORTED_MODE, NOT_APPLICABLE_MODE):
            if selection_units or calibration_units or guard_units:
                _fail("roles_declared_for_unavailable_context")

        selection_ids = {unit["unit_id"] for unit in selection_units}
        calibration_ids = {unit["unit_id"] for unit in calibration_units}
        guard_ids = {unit["unit_id"] for unit in guard_units}
        if guard_ids & (selection_ids | calibration_ids):
            _fail("duplicate_unit_id")

        selection_by_id = {unit["unit_id"]: unit for unit in selection_units}
        calibration_by_id = {unit["unit_id"]: unit for unit in calibration_units}
        for unit_id in selection_ids & calibration_ids:
            if mode != core.MASTER_MODE:
                _fail("selection_calibration_id_collision")
            if selection_by_id[unit_id] != calibration_by_id[unit_id]:
                _fail("selection_calibration_alias_mismatch")
        if mode == core.MASTER_MODE and calibration_units:
            if calibration_by_id != selection_by_id:
                _fail("master_calibration_roles_must_alias_selection")

        ctx = {
            "context_id": context_id,
            "selection_mode": mode,
            "domain_eligible": eligible,
            "classical_supported": raw["classical_supported"],
            "neural_supported": raw["neural_supported"],
            "g3_comparable": raw["g3_comparable"],
            "unavailable_reasons": cleaned_reasons,
            "outer_fit_uid_sha256": outer_fit,
            "outer_test_uid_sha256": outer_test,
            "selection_units": selection_units,
            "calibration_units": calibration_units,
            "guard_units": guard_units,
        }

        if ctx["classical_supported"] != _derive_classical_supported(ctx):
            _fail("classical_support_contradiction")
        if ctx["neural_supported"] != _derive_neural_supported(ctx):
            _fail("neural_support_contradiction")
        if ctx["g3_comparable"] != _derive_g3_comparable(ctx):
            _fail("g3_comparable_contradiction")

        if not ctx["neural_supported"]:
            ctx["allowed_recipes"] = []
        elif ctx["g3_comparable"]:
            ctx["allowed_recipes"] = list(core.NEURAL_RECIPES)
        else:
            ctx["allowed_recipes"] = [core.D0_RECIPE]
        parsed.append(ctx)

    parsed.sort(key=lambda ctx: ctx["context_id"])
    return parsed


def _validated_contract(core_contract):
    _require_mapping(core_contract, "core_contract_must_be_mapping")
    try:
        validate_core_contract(core_contract)
    except (TypeError, KeyError) as error:
        raise core.PlanError("core_contract_invalid") from error
    g3 = core_contract.get("g3")
    if not isinstance(g3, Mapping) or not g3:
        _fail("core_contract_g3_invalid")
    return core_contract


# ---------------------------------------------------------------------------
# job / alias plumbing
# ---------------------------------------------------------------------------


def _binding(population_id, population_sha256, population_plan_sha256, neural_support_sha256):
    return {
        "population_id": population_id,
        "population_sha256": population_sha256,
        "population_plan_sha256": population_plan_sha256,
        "neural_support_sha256": neural_support_sha256,
    }


def _nfields(
    policy,
    representation,
    array_sha,
    context_id,
    model_id,
    model_spec_sha,
    stage,
    unit_id,
    seed,
    candidate_id,
    hyperparameter_sha,
    fit_sha,
    validation_sha,
    test_sha,
    resolution,
):
    return {
        "policy_id": policy,
        "representation_id": representation,
        "array_sha256": array_sha,
        "context_id": context_id,
        "model_id": model_id,
        "model_spec_sha256": model_spec_sha,
        "stage": stage,
        "unit_id": unit_id,
        "seed": seed,
        "candidate_id": candidate_id,
        "hyperparameter_sha256": hyperparameter_sha,
        "fit_uid_sha256": fit_sha,
        "validation_uid_sha256": validation_sha,
        "test_uid_sha256": test_sha,
        "resolution": resolution,
        "evidence_status": core.EVIDENCE_FUTURE,
    }


def _add_job(jobs, binding, fields, dependencies, activation=None, metadata=None):
    payload = {}
    payload.update(binding)
    payload.update(fields)
    dependency_ids = set(dependencies)
    if activation is not None:
        selection_job_id = activation.get("selection_job_id")
        if selection_job_id is not None:
            dependency_ids.add(selection_job_id)
    payload["dependencies"] = sorted(dependency_ids)
    payload["activation"] = activation
    if metadata is not None:
        payload["metadata"] = metadata
    job = dict(payload)
    job["job_id"] = JOB_NAMESPACE + sha256_value(payload)
    jobs.append(job)
    return job["job_id"]


def _new_alias(
    binding,
    policy_id,
    context_id,
    strategy,
    recipe_id,
    target_job_id,
    activation,
    conditional_targets=None,
    metadata=None,
):
    fields = {}
    fields.update(binding)
    fields.update(
        {
            "policy_id": policy_id,
            "context_id": context_id,
            "strategy": strategy,
            "recipe_id": recipe_id,
            "target_job_id": target_job_id,
            "activation": activation,
        }
    )
    if conditional_targets is not None:
        fields["conditional_targets"] = sorted(
            conditional_targets, key=lambda item: item["recipe_id"]
        )
    if metadata is not None:
        fields["metadata"] = dict(metadata)
    alias = dict(fields)
    alias["alias_id"] = ALIAS_NAMESPACE + sha256_value(fields)
    return alias


def _excluded_slot(
    binding, policy, representation, array_sha, context_id, recipe, model_spec_sha, guard, seed
):
    fields = {}
    fields.update(binding)
    fields.update(
        {
            "policy_id": policy,
            "representation_id": representation,
            "array_sha256": array_sha,
            "context_id": context_id,
            "model_id": recipe,
            "model_spec_sha256": model_spec_sha,
            "stage": GUARD_SOURCE_FIT_STAGE,
            "unit_id": guard["unit_id"],
            "fit_uid_sha256": guard["fit_uid_sha256"],
            "validation_uid_sha256": guard["validation_uid_sha256"],
            "role_fit_uid_sha256": guard["fit_uid_sha256"],
            "role_validation_uid_sha256": guard["validation_uid_sha256"],
            "seed": seed,
            "exclusion_reason": guard["exclusion_reason"],
            "numerical_job": False,
            "counted_as_attempted_fit": False,
        }
    )
    slot = dict(fields)
    slot["slot_id"] = EXCLUDED_NAMESPACE + sha256_value(fields)
    return slot


def _wrap_core_jobs(core_jobs, binding):
    """Wrap frozen classical jobs into the population namespace, remapping IDs."""
    id_map = {}
    wrapped = []
    for job in core_jobs:
        fields = {name: job[name] for name in core.JOB_FIELDS if name != "dependencies"}
        fields["evidence_status"] = core.EVIDENCE_FUTURE
        payload = {}
        payload.update(binding)
        payload.update(fields)
        payload["dependencies"] = sorted(id_map[dep] for dep in job["dependencies"])
        payload["activation"] = None
        new_job = dict(payload)
        new_job["job_id"] = JOB_NAMESPACE + sha256_value(payload)
        id_map[job["job_id"]] = new_job["job_id"]
        wrapped.append(new_job)
    return wrapped


# ---------------------------------------------------------------------------
# classical operations (delegated to the frozen core)
# ---------------------------------------------------------------------------


def _classical_scratch(ctx):
    return {
        "context_id": ctx["context_id"],
        "selection_mode": ctx["selection_mode"],
        "selected_recipe_id": core.D0_RECIPE,
        "outer_fit_uid_sha256": ctx["outer_fit_uid_sha256"],
        "outer_test_uid_sha256": ctx["outer_test_uid_sha256"],
        "selection_units": ctx["selection_units"],
        "calibration_units": ctx["calibration_units"],
    }


def _classical_jobs(ctx, actions, spec, candidates_by_model, include_extra_trees, binding):
    if not ctx["classical_supported"]:
        return []
    scratch = core._parse_contexts([_classical_scratch(ctx)])[0]
    jobs = []
    for policy in core.POLICIES:
        representation = core.POLICY_REPRESENTATION[policy]
        core_jobs, _aliases = core._build_context(
            policy,
            representation,
            actions[representation],
            core.EVIDENCE_FUTURE,
            scratch,
            candidates_by_model,
            spec,
        )
        retained = []
        for job in core_jobs:
            if job["model_id"] not in core.CLASSICAL_MODELS:
                continue
            if not include_extra_trees and job["model_id"] == "C-EXTRA-TREES":
                continue
            retained.append(job)
        jobs.extend(_wrap_core_jobs(retained, binding))
    return jobs


# ---------------------------------------------------------------------------
# neural operations
# ---------------------------------------------------------------------------


def _fresh_sources(
    jobs, binding, spec, ctx, policy, representation, array_sha, recipe, min_sel_id, conditional
):
    activation = {"selection_job_id": min_sel_id, "recipe_id": recipe} if conditional else None
    spec_sha = spec[recipe]
    context_id = ctx["context_id"]
    test_sha = ctx["outer_test_uid_sha256"]
    by_seed = {}
    for unit in ctx["selection_units"]:
        for seed in core.SEEDS:
            fit_id = _add_job(
                jobs,
                binding,
                _nfields(
                    policy,
                    representation,
                    array_sha,
                    context_id,
                    recipe,
                    spec_sha,
                    "source_fit",
                    unit["unit_id"],
                    seed,
                    "fixed_recipe",
                    spec_sha,
                    unit["fit_uid_sha256"],
                    unit["validation_uid_sha256"],
                    test_sha,
                    core.FIXED_SPEC,
                ),
                [min_sel_id],
                activation,
            )
            pred_id = _add_job(
                jobs,
                binding,
                _nfields(
                    policy,
                    representation,
                    array_sha,
                    context_id,
                    recipe,
                    spec_sha,
                    "source_validation_prediction",
                    unit["unit_id"],
                    seed,
                    "fixed_recipe",
                    spec_sha,
                    unit["fit_uid_sha256"],
                    unit["validation_uid_sha256"],
                    test_sha,
                    core.FIXED_SPEC,
                ),
                [fit_id],
                activation,
            )
            by_seed.setdefault(seed, []).append(pred_id)
    return by_seed


def _pipeline(
    jobs,
    binding,
    spec,
    ctx,
    policy,
    representation,
    array_sha,
    recipe,
    sources,
    min_sel_id,
    conditional,
):
    activation = {"selection_job_id": min_sel_id, "recipe_id": recipe} if conditional else None
    spec_sha = spec[recipe]
    context_id = ctx["context_id"]
    test_sha = ctx["outer_test_uid_sha256"]
    outer_fit = ctx["outer_fit_uid_sha256"]
    held_ids = []
    for seed in core.SEEDS:
        source_ids = list(sources[seed])
        select_deps = list(source_ids)
        if conditional:
            select_deps.append(min_sel_id)
        select_id = _add_job(
            jobs,
            binding,
            _nfields(
                policy,
                representation,
                array_sha,
                context_id,
                recipe,
                spec_sha,
                "select_refit_epochs",
                core.NOT_APPLICABLE,
                seed,
                core.SOURCE_SELECTION_DEPENDENT,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                test_sha,
                core.SOURCE_EPOCH_DEPENDENT,
            ),
            select_deps,
            activation,
        )
        scalar_id = _add_job(
            jobs,
            binding,
            _nfields(
                policy,
                representation,
                array_sha,
                context_id,
                recipe,
                spec_sha,
                core.SCALAR_STAGE,
                core.NOT_APPLICABLE,
                seed,
                core.SOURCE_SELECTION_DEPENDENT,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                test_sha,
                NEURAL_SCALAR_RESOLUTION,
            ),
            source_ids + [min_sel_id],
            activation,
            metadata={
                "role_purpose": NEURAL_SCALAR_ROLE_PURPOSE,
                "excludes_guards": True,
                "excludes_classical_calibration": True,
                "excludes_test_evidence": True,
            },
        )
        refit_id = _add_job(
            jobs,
            binding,
            _nfields(
                policy,
                representation,
                array_sha,
                context_id,
                recipe,
                spec_sha,
                "final_refit",
                core.NOT_APPLICABLE,
                seed,
                core.SOURCE_SELECTION_DEPENDENT,
                core.NOT_APPLICABLE,
                outer_fit,
                core.NOT_APPLICABLE,
                test_sha,
                core.SOURCE_EPOCH_DEPENDENT,
            ),
            [select_id, min_sel_id],
            activation,
        )
        held_id = _add_job(
            jobs,
            binding,
            _nfields(
                policy,
                representation,
                array_sha,
                context_id,
                recipe,
                spec_sha,
                "held_prediction",
                core.NOT_APPLICABLE,
                seed,
                core.SOURCE_SELECTION_DEPENDENT,
                core.NOT_APPLICABLE,
                outer_fit,
                core.NOT_APPLICABLE,
                test_sha,
                core.SOURCE_EPOCH_DEPENDENT,
            ),
            [refit_id, scalar_id],
            activation,
        )
        held_ids.append(held_id)
    return _add_job(
        jobs,
        binding,
        _nfields(
            policy,
            representation,
            array_sha,
            context_id,
            recipe,
            spec_sha,
            "seed_ensemble_prediction",
            core.NOT_APPLICABLE,
            core.NOT_APPLICABLE,
            core.SOURCE_SELECTION_DEPENDENT,
            core.NOT_APPLICABLE,
            core.NOT_APPLICABLE,
            core.NOT_APPLICABLE,
            test_sha,
            core.SOURCE_EPOCH_DEPENDENT,
        ),
        held_ids,
        activation,
    )


def _build_neural(ctxs, actions, spec, binding, core_contract_sha, g3):
    jobs = []
    excluded = []
    aliases = []
    select_index = {}
    ensemble_index = {}
    dev_pred = {}

    min_policy = core.POLICIES[0]
    min_rep = core.POLICY_REPRESENTATION[min_policy]
    min_sha = actions[min_rep]

    for ctx in ctxs:
        if not ctx["neural_supported"]:
            continue
        context_id = ctx["context_id"]
        test_sha = ctx["outer_test_uid_sha256"]
        pred_index = {}
        guard_index = {}
        for unit in ctx["selection_units"]:
            for recipe in core.NEURAL_RECIPES:
                spec_sha = spec[recipe]
                for seed in core.SEEDS:
                    fit_id = _add_job(
                        jobs,
                        binding,
                        _nfields(
                            min_policy,
                            min_rep,
                            min_sha,
                            context_id,
                            recipe,
                            spec_sha,
                            "source_fit",
                            unit["unit_id"],
                            seed,
                            "fixed_recipe",
                            spec_sha,
                            unit["fit_uid_sha256"],
                            unit["validation_uid_sha256"],
                            test_sha,
                            core.FIXED_SPEC,
                        ),
                        [],
                    )
                    pred_id = _add_job(
                        jobs,
                        binding,
                        _nfields(
                            min_policy,
                            min_rep,
                            min_sha,
                            context_id,
                            recipe,
                            spec_sha,
                            "source_validation_prediction",
                            unit["unit_id"],
                            seed,
                            "fixed_recipe",
                            spec_sha,
                            unit["fit_uid_sha256"],
                            unit["validation_uid_sha256"],
                            test_sha,
                            core.FIXED_SPEC,
                        ),
                        [fit_id],
                    )
                    pred_index.setdefault((recipe, seed), []).append(pred_id)
        for guard in ctx["guard_units"]:
            if not guard["support"]:
                continue
            for recipe in core.NEURAL_RECIPES:
                spec_sha = spec[recipe]
                for seed in core.SEEDS:
                    fit_id = _add_job(
                        jobs,
                        binding,
                        _nfields(
                            min_policy,
                            min_rep,
                            min_sha,
                            context_id,
                            recipe,
                            spec_sha,
                            GUARD_SOURCE_FIT_STAGE,
                            guard["unit_id"],
                            seed,
                            "fixed_recipe",
                            spec_sha,
                            guard["fit_uid_sha256"],
                            guard["validation_uid_sha256"],
                            test_sha,
                            core.FIXED_SPEC,
                        ),
                        [],
                    )
                    pred_id = _add_job(
                        jobs,
                        binding,
                        _nfields(
                            min_policy,
                            min_rep,
                            min_sha,
                            context_id,
                            recipe,
                            spec_sha,
                            GUARD_VALIDATION_PREDICTION_STAGE,
                            guard["unit_id"],
                            seed,
                            "fixed_recipe",
                            spec_sha,
                            guard["fit_uid_sha256"],
                            guard["validation_uid_sha256"],
                            test_sha,
                            core.FIXED_SPEC,
                        ),
                        [fit_id],
                    )
                    guard_index.setdefault((recipe, seed), []).append(pred_id)

        excluded_ids = []
        for guard in ctx["guard_units"]:
            if guard["support"]:
                continue
            for recipe in core.NEURAL_RECIPES:
                for seed in core.SEEDS:
                    slot = _excluded_slot(
                        binding,
                        min_policy,
                        min_rep,
                        min_sha,
                        context_id,
                        recipe,
                        spec[recipe],
                        guard,
                        seed,
                    )
                    excluded.append(slot)
                    excluded_ids.append(slot["slot_id"])

        dependencies = []
        for ids in pred_index.values():
            dependencies.extend(ids)
        for ids in guard_index.values():
            dependencies.extend(ids)
        metadata = {
            "source_only": True,
            "core_contract_sha256": core_contract_sha,
            "g3": dict(g3) if isinstance(g3, dict) else {},
            "allowed_recipes": list(ctx["allowed_recipes"]),
            "excluded_slot_ids": sorted(excluded_ids),
            "ready_rule": "inherit_p05_ready_for_refit",
            "requires_all_numerical_status_resolved": True,
        }
        select_index[context_id] = _add_job(
            jobs,
            binding,
            _nfields(
                min_policy,
                min_rep,
                min_sha,
                context_id,
                core.D0_RECIPE,
                spec[core.D0_RECIPE],
                SELECT_NEURAL_RECIPE_STAGE,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                core.SOURCE_SELECTION_DEPENDENT,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                core.NOT_APPLICABLE,
                test_sha,
                SELECT_NEURAL_RECIPE_RESOLUTION,
            ),
            dependencies,
            metadata=metadata,
        )
        dev_pred[context_id] = pred_index

    for ctx in ctxs:
        if not ctx["neural_supported"]:
            continue
        context_id = ctx["context_id"]
        min_sel_id = select_index[context_id]
        dev = dev_pred[context_id]
        for policy in core.POLICIES:
            representation = core.POLICY_REPRESENTATION[policy]
            array_sha = actions[representation]
            is_min = policy == min_policy
            if is_min:
                d0_sources = {seed: list(dev[(core.D0_RECIPE, seed)]) for seed in core.SEEDS}
            else:
                d0_sources = _fresh_sources(
                    jobs,
                    binding,
                    spec,
                    ctx,
                    policy,
                    representation,
                    array_sha,
                    core.D0_RECIPE,
                    min_sel_id,
                    False,
                )
            ensemble_index[(context_id, policy, core.D0_RECIPE)] = _pipeline(
                jobs,
                binding,
                spec,
                ctx,
                policy,
                representation,
                array_sha,
                core.D0_RECIPE,
                d0_sources,
                min_sel_id,
                False,
            )
            if not ctx["g3_comparable"]:
                continue
            for recipe in NON_D0_RECIPES:
                if is_min:
                    sources = {seed: list(dev[(recipe, seed)]) for seed in core.SEEDS}
                else:
                    sources = _fresh_sources(
                        jobs,
                        binding,
                        spec,
                        ctx,
                        policy,
                        representation,
                        array_sha,
                        recipe,
                        min_sel_id,
                        True,
                    )
                ensemble_index[(context_id, policy, recipe)] = _pipeline(
                    jobs,
                    binding,
                    spec,
                    ctx,
                    policy,
                    representation,
                    array_sha,
                    recipe,
                    sources,
                    min_sel_id,
                    True,
                )

    for ctx in ctxs:
        if not ctx["neural_supported"]:
            continue
        context_id = ctx["context_id"]
        min_sel_id = select_index[context_id]
        for policy in core.POLICIES:
            d0_ensemble = ensemble_index[(context_id, policy, core.D0_RECIPE)]
            aliases.append(
                _new_alias(
                    binding,
                    policy,
                    context_id,
                    core.D0_STRATEGY,
                    core.D0_RECIPE,
                    d0_ensemble,
                    None,
                    metadata={
                        "ready_rule": ALIAS_READY_RULE,
                        "target_is_structural": True,
                        "ensemble_follows_min_ready": True,
                    },
                )
            )
            if ctx["g3_comparable"]:
                targets = [
                    {
                        "recipe_id": recipe,
                        "target_job_id": ensemble_index[(context_id, policy, recipe)],
                    }
                    for recipe in NON_D0_RECIPES
                ]
                aliases.append(
                    _new_alias(
                        binding,
                        policy,
                        context_id,
                        core.SELECTED_STRATEGY,
                        None,
                        d0_ensemble,
                        {
                            "selection_job_id": min_sel_id,
                            "recipe_id": None,
                            "ready_rule": ALIAS_READY_RULE,
                        },
                        conditional_targets=targets,
                        metadata={
                            "ready_rule": ALIAS_READY_RULE,
                            "unknown_recipe_until_selection": True,
                            "allowed_conditional_recipes": list(NON_D0_RECIPES),
                            "no_unknown_or_failure_fallback": True,
                        },
                    )
                )
            else:
                aliases.append(
                    _new_alias(
                        binding,
                        policy,
                        context_id,
                        core.SELECTED_STRATEGY,
                        core.D0_RECIPE,
                        d0_ensemble,
                        {
                            "selection_job_id": min_sel_id,
                            "recipe_id": core.D0_RECIPE,
                            "ready_rule": ALIAS_READY_RULE,
                        },
                        metadata={
                            "ready_rule": ALIAS_READY_RULE,
                            "d0_default_requires_selection": True,
                            "d0_default_requires_ready_for_refit": True,
                            "no_unknown_or_failure_fallback": True,
                        },
                    )
                )

    return jobs, excluded, aliases


# ---------------------------------------------------------------------------
# graph validation and summary
# ---------------------------------------------------------------------------


def _min_selection_jobs(jobs):
    min_select = {}
    for job in jobs:
        if job["stage"] != SELECT_NEURAL_RECIPE_STAGE:
            continue
        context_id = job["context_id"]
        if context_id in min_select:
            raise AssertionError("duplicate min selection job")
        if job["policy_id"] != core.POLICIES[0]:
            raise AssertionError("min_selection_policy_not_primary")
        if job["activation"] is not None:
            raise AssertionError("min_selection_activation_not_none")
        min_select[context_id] = job["job_id"]
    return min_select


def _validate_job_dependencies(jobs, by_id, min_select):
    for job in jobs:
        context_id = job["context_id"]
        activation = job["activation"]
        for dependency in job["dependencies"]:
            if dependency not in by_id:
                raise AssertionError("unregistered dependency")
            dependency_job = by_id[dependency]
            if dependency_job["context_id"] != context_id:
                raise AssertionError("cross-context dependency")
            if dependency_job["policy_id"] != job["policy_id"]:
                allowed_min = (
                    dependency_job["job_id"] == min_select.get(context_id)
                    and dependency_job["stage"] == SELECT_NEURAL_RECIPE_STAGE
                )
                if not allowed_min:
                    raise AssertionError("cross-policy dependency")
            if activation is None and dependency_job["activation"] is not None:
                raise AssertionError("unconditional depends on conditional")
        if activation is None:
            continue
        if not isinstance(activation, dict):
            raise AssertionError("activation_must_be_mapping")
        selection_job_id = activation.get("selection_job_id")
        recipe_id = activation.get("recipe_id")
        if selection_job_id is None or selection_job_id not in by_id:
            raise AssertionError("activation_missing_selection_job")
        if selection_job_id != min_select.get(context_id):
            raise AssertionError("activation_selection_job_not_context_min")
        if recipe_id not in NON_D0_RECIPES:
            raise AssertionError("activation_recipe_not_conditional")
        if recipe_id != job["model_id"]:
            raise AssertionError("activation_recipe_model_mismatch")
        if selection_job_id not in job["dependencies"]:
            raise AssertionError("activation_missing_direct_dependency")
        for dependency in job["dependencies"]:
            dependency_activation = by_id[dependency]["activation"]
            if dependency_activation is None:
                continue
            if not isinstance(dependency_activation, dict):
                raise AssertionError("activation_must_be_mapping")
            if dependency_activation.get("recipe_id") != recipe_id:
                raise AssertionError("cross-recipe conditional dependency")


def _validate_and_order_graph(jobs):
    by_id = {job["job_id"]: job for job in jobs}
    if len(by_id) != len(jobs):
        raise AssertionError("duplicate job ids")
    min_select = _min_selection_jobs(jobs)
    _validate_job_dependencies(jobs, by_id, min_select)

    in_degree = {job["job_id"]: len(job["dependencies"]) for job in jobs}
    consumers = {job["job_id"]: [] for job in jobs}
    for job in jobs:
        for dependency in job["dependencies"]:
            consumers[dependency].append(job["job_id"])

    ready = [job_id for job_id, degree in in_degree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for consumer in consumers[node]:
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(ordered) != len(jobs):
        raise AssertionError("dependency cycle")

    position = {job_id: index for index, job_id in enumerate(ordered)}
    for job in jobs:
        for dependency in job["dependencies"]:
            if position[dependency] >= position[job["job_id"]]:
                raise AssertionError("dependency order violated")
    return [by_id[job_id] for job_id in ordered]


def _validate_aliases(aliases, jobs, ctxs):
    by_id = {job["job_id"]: job for job in jobs}
    neural_contexts = {ctx["context_id"] for ctx in ctxs if ctx["neural_supported"]}
    min_select = _min_selection_jobs(jobs)
    seen_keys = set()
    grouped = {}
    for alias in aliases:
        context_id = alias["context_id"]
        policy_id = alias["policy_id"]
        strategy = alias["strategy"]
        logical_key = (context_id, policy_id, strategy)
        if logical_key in seen_keys:
            raise AssertionError("duplicate_alias_logical_key")
        seen_keys.add(logical_key)
        grouped.setdefault((context_id, policy_id), []).append(alias)
        if context_id not in neural_contexts:
            raise AssertionError("alias_context_not_neural")
        target = by_id.get(alias["target_job_id"])
        if target is None:
            raise AssertionError("alias_target_unregistered")
        if target["context_id"] != context_id or target["policy_id"] != policy_id:
            raise AssertionError("alias_target_binding_mismatch")
        if target["stage"] != "seed_ensemble_prediction":
            raise AssertionError("alias_target_not_ensemble")
        recipe_id = alias["recipe_id"]
        if recipe_id is None:
            if target["model_id"] != core.D0_RECIPE:
                raise AssertionError("alias_default_target_not_d0")
        elif target["model_id"] != recipe_id:
            raise AssertionError("alias_recipe_target_mismatch")
        activation = alias["activation"]
        metadata = alias.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("ready_rule") != ALIAS_READY_RULE:
            raise AssertionError("alias_missing_ready_rule")
        if strategy == core.SELECTED_STRATEGY:
            if not isinstance(activation, dict):
                raise AssertionError("selected_alias_activation_missing")
            if activation.get("selection_job_id") != min_select.get(context_id):
                raise AssertionError("alias_selection_job_not_context_min")
            if activation.get("ready_rule") != ALIAS_READY_RULE:
                raise AssertionError("selected_alias_ready_rule_invalid")
            if not metadata.get("no_unknown_or_failure_fallback"):
                raise AssertionError("selected_alias_missing_no_fallback")
            if recipe_id is None:
                conditional_recipes = [
                    entry["recipe_id"] for entry in alias.get("conditional_targets", [])
                ]
                if len(conditional_recipes) != len(NON_D0_RECIPES) or set(
                    conditional_recipes
                ) != set(NON_D0_RECIPES):
                    raise AssertionError("g3_conditional_targets_not_exact")
            else:
                if recipe_id != core.D0_RECIPE:
                    raise AssertionError("selected_alias_recipe_invalid")
                if alias.get("conditional_targets"):
                    raise AssertionError("structural_alias_has_conditional_targets")
        else:
            if recipe_id != core.D0_RECIPE:
                raise AssertionError("ordinary_alias_recipe_not_d0")
            if activation is not None:
                raise AssertionError("ordinary_alias_has_activation")
            if alias.get("conditional_targets"):
                raise AssertionError("ordinary_alias_has_conditional_targets")
        for entry in alias.get("conditional_targets", []):
            if entry["recipe_id"] not in NON_D0_RECIPES:
                raise AssertionError("alias_conditional_recipe_invalid")
            conditional_target = by_id.get(entry["target_job_id"])
            if conditional_target is None:
                raise AssertionError("alias_conditional_target_unregistered")
            if (
                conditional_target["context_id"] != context_id
                or conditional_target["policy_id"] != policy_id
            ):
                raise AssertionError("alias_conditional_target_binding_mismatch")
            if conditional_target["stage"] != "seed_ensemble_prediction":
                raise AssertionError("alias_conditional_target_not_ensemble")
            if conditional_target["model_id"] != entry["recipe_id"]:
                raise AssertionError("alias_conditional_target_recipe_mismatch")
            entry_activation = conditional_target["activation"]
            if not isinstance(entry_activation, dict):
                raise AssertionError("alias_conditional_target_activation_missing")
            if entry_activation.get("recipe_id") != entry["recipe_id"]:
                raise AssertionError("alias_conditional_target_activation_recipe_mismatch")
            if entry_activation.get("selection_job_id") != min_select.get(context_id):
                raise AssertionError("alias_conditional_target_selection_job_not_context_min")
    for context_id in neural_contexts:
        for policy_id in core.POLICIES:
            if len(grouped.get((context_id, policy_id), [])) != 2:
                raise AssertionError("alias_count_not_two")


def _stage_counts(jobs):
    counts = {}
    for job in jobs:
        counts[job["stage"]] = counts.get(job["stage"], 0) + 1
    return {stage: counts[stage] for stage in sorted(counts)}


def _bounds_stats(jobs):
    counts = _stage_counts(jobs)
    return {
        "stage_counts": counts,
        "model_fit_slots": sum(
            count for stage, count in counts.items() if stage in POP_MODEL_FIT_STAGES
        ),
        "scalar_calibrations": counts.get(core.SCALAR_STAGE, 0),
        "total_jobs": len(jobs),
    }


def _model_stats(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["model_id"], []).append(job)
    return {model: _bounds_stats(grouped[model]) for model in sorted(grouped)}


def _summary(jobs, excluded, ctxs):
    unconditional = [job for job in jobs if job["activation"] is None]
    conditional = [job for job in jobs if job["activation"] is not None]
    upper_jobs = list(unconditional)

    groups = {}
    for job in conditional:
        groups.setdefault(job["context_id"], {}).setdefault(
            job["activation"]["recipe_id"], []
        ).append(job)
    for context_id in sorted(groups):
        recipe_groups = groups[context_id]
        if sorted(recipe_groups) != sorted(NON_D0_RECIPES):
            raise AssertionError("conditional context recipes invalid")
        policies = sorted({job["policy_id"] for branch in recipe_groups.values() for job in branch})
        for policy in policies:
            reference = None
            for recipe in sorted(recipe_groups):
                counts = _stage_counts(
                    [job for job in recipe_groups[recipe] if job["policy_id"] == policy]
                )
                if reference is None:
                    reference = counts
                elif counts != reference:
                    raise AssertionError("conditional branch stage counts differ")
        upper_jobs.extend(recipe_groups[sorted(recipe_groups)[0]])

    per_policy = {}
    for policy in core.POLICIES:
        per_policy[policy] = {
            "lower": _bounds_stats([job for job in unconditional if job["policy_id"] == policy]),
            "upper": _bounds_stats([job for job in upper_jobs if job["policy_id"] == policy]),
        }

    neural_contexts = sum(1 for ctx in ctxs if ctx["neural_supported"])
    classical_ensembles = sum(
        1
        for job in jobs
        if job["stage"] == "seed_ensemble_prediction" and job["model_id"] in core.CLASSICAL_MODELS
    )
    logical_neural_endpoints = 2 * neural_contexts * 3
    per_model_upper_jobs = list(unconditional) + list(conditional)
    return {
        "catalog_stage_counts": _stage_counts(jobs),
        "lower": _bounds_stats(unconditional),
        "upper": _bounds_stats(upper_jobs),
        "bounds_are_nonadditive": True,
        "per_policy": per_policy,
        "per_model": {
            "bounds_are_nonadditive": True,
            "catalog": _model_stats(jobs),
            "lower": _model_stats(unconditional),
            "upper": _model_stats(per_model_upper_jobs),
        },
        "conditional_context_count": len(groups),
        "logical_neural_endpoints": logical_neural_endpoints,
        "logical_total_endpoints": classical_ensembles + logical_neural_endpoints,
        "classical_family_ensembles": classical_ensembles,
        "excluded_source_fit_slots": len(excluded),
        "authorized_scientific_operations": 0,
    }


def _availability(ctx, include_extra_trees):
    classical_models = []
    if ctx["classical_supported"]:
        classical_models = [
            model
            for model in core.CLASSICAL_MODELS
            if include_extra_trees or model != "C-EXTRA-TREES"
        ]
    neural_methods = [core.D0_STRATEGY, core.SELECTED_STRATEGY] if ctx["neural_supported"] else []
    return {
        "context_id": ctx["context_id"],
        "selection_mode": ctx["selection_mode"],
        "domain_eligible": ctx["domain_eligible"],
        "classical_supported": ctx["classical_supported"],
        "neural_supported": ctx["neural_supported"],
        "g3_comparable": ctx["g3_comparable"],
        "unavailable_reasons": list(ctx["unavailable_reasons"]),
        "classical_models": classical_models,
        "neural_methods": neural_methods,
        "allowed_recipes": list(ctx["allowed_recipes"]) if ctx["neural_supported"] else [],
    }


def build_population_slots(
    *,
    population_id,
    population_sha256,
    population_plan_sha256,
    neural_support_sha256,
    contexts,
    candidates,
    actions,
    model_spec_sha256,
    core_contract,
    include_extra_trees,
):
    """Build the data-free population slot catalogue (metadata only)."""
    _require_identifier(population_id, "population_id_invalid")
    _require_sha256(population_sha256, "population_hash_invalid")
    _require_sha256(population_plan_sha256, "population_plan_hash_invalid")
    _require_sha256(neural_support_sha256, "neural_support_hash_invalid")
    if not isinstance(include_extra_trees, bool):
        _fail("include_extra_trees_invalid")

    contract = _validated_contract(core_contract)
    core_contract_sha = sha256_value(contract)
    g3 = contract["g3"]

    parsed_actions = core._parse_actions(actions)
    parsed_spec = core._parse_model_spec(model_spec_sha256)
    parsed_candidates = core._parse_candidates(candidates)
    parsed_contexts = _parse_population_contexts(contexts)

    candidates_by_model = {model: [] for model in core.CLASSICAL_MODELS}
    for candidate in parsed_candidates:
        candidates_by_model[candidate["model_id"]].append(candidate)

    binding = _binding(
        population_id, population_sha256, population_plan_sha256, neural_support_sha256
    )

    jobs = []
    for ctx in parsed_contexts:
        jobs.extend(
            _classical_jobs(
                ctx, parsed_actions, parsed_spec, candidates_by_model, include_extra_trees, binding
            )
        )

    neural_jobs, excluded, aliases = _build_neural(
        parsed_contexts, parsed_actions, parsed_spec, binding, core_contract_sha, g3
    )
    jobs.extend(neural_jobs)

    jobs = _validate_and_order_graph(jobs)
    _validate_aliases(aliases, jobs, parsed_contexts)
    excluded.sort(key=lambda slot: slot["slot_id"])
    aliases.sort(key=lambda alias: alias["alias_id"])

    availability = [_availability(ctx, include_extra_trees) for ctx in parsed_contexts]
    summary = _summary(jobs, excluded, parsed_contexts)

    panel_methods = ["C-RBF-SVM", "C-RANDOM-FOREST"]
    if include_extra_trees:
        panel_methods.append("C-EXTRA-TREES")
    panel_methods.extend([core.D0_STRATEGY, core.SELECTED_STRATEGY])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "numerical_readiness_verified": False,
        "panel_choice_is_planning_only": True,
        "population_id": population_id,
        "population_sha256": population_sha256,
        "population_plan_sha256": population_plan_sha256,
        "neural_support_sha256": neural_support_sha256,
        "include_extra_trees": include_extra_trees,
        "panel_methods": panel_methods,
        "jobs": jobs,
        "excluded_source_fit_slots": excluded,
        "endpoint_aliases": aliases,
        "availability": availability,
        "summary": summary,
    }
    plan = dict(payload)
    plan["plan_sha256"] = sha256_value(payload)
    return plan
