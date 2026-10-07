"""P08 universal robustness procedure references (metadata only).

This module is an INTERNAL adapter of previously authenticated upstream
metadata.  It maps a data-free universal P08 slot DAG plus a frozen
minimum-operation evidence bridge into deterministic *procedure records* that
later adapters can join into a complete stress ledger.

It is deliberately narrow:

* It is **not** a full upstream graph validator and **not** an artifact
  authenticator.  Callers must separately authenticate the original files, the
  complete upstream DAG, the exclusion sets and every artifact before any
  scientific use.  The hashes recorded here only bind the metadata that was
  actually supplied to this function.
* It performs **zero** scientific operations.  No arrays, fits, epochs,
  temperatures, thresholds, exclusions or predictions are computed, guessed,
  drawn or inferred.
* ``require_scientific_execution`` always denies execution, even when handed a
  forged authority flag.

Calibration ordering is copied from the universal plan conventions rather than
invented here.  Classical records average the per-seed uncalibrated held scores
and then apply one source-fitted temperature
(``seed_average_then_single_temperature``); neural records apply a per-seed
temperature first and only then average
(``per_seed_temperature_then_seed_average``).

The historical (``MIN``) policy is the one the universal plan marked as
``historical_reuse_requires_authentication``.  Only its final operations may
carry a bridge binding; every other policy's references remain unresolved
future outcomes (``historical_binding_sha256`` and ``resolved_source_values``
are ``None``), which is *not* a missing-evidence error.

Nothing here writes files, imports filesystem or scientific stacks, catches
``BaseException``, draws randomness or reproduces the heavy plan hash.  Strict
JSON safety is checked through the stdlib-only
``p08_qc_blocks.canonical_sha256`` helper.
"""

from __future__ import annotations

import copy
import hashlib
import json

from .p08_plan import (
    CLASSICAL_MODELS,
    D0_RECIPE,
    D0_STRATEGY,
    EVIDENCE_BY_POLICY,
    EVIDENCE_HISTORICAL,
    JOB_FIELDS,
    NEURAL_RECIPES,
    NOT_APPLICABLE,
    POLICIES,
    POLICY_REPRESENTATION,
    SCALAR_STAGE,
    SEEDS,
    SELECTED_STRATEGY,
    SVM_MODEL,
    SVM_SEED,
)
from .p08_plan import (
    SCHEMA_VERSION as PLAN_SCHEMA_VERSION,
)
from .p08_qc_blocks import canonical_sha256 as sha256_value

__all__ = [
    "build_universal_procedure_records",
    "require_scientific_execution",
]


PROCEDURE_SCHEMA_VERSION = "nato-sers-p08-universal-stress-procedures-v1"
BRIDGE_SCHEMA_VERSION = "nato-sers-p08-minimum-operation-evidence-v1"

INVALID_METADATA = "invalid_stress_procedure_metadata"
EXECUTION_DENIED = "scientific_execution_not_authorized"

FINAL_REFIT_STAGE = "final_refit"
HELD_STAGE = "held_prediction"
ENSEMBLE_STAGE = "seed_ensemble_prediction"
FINAL_STAGES = frozenset((FINAL_REFIT_STAGE, HELD_STAGE, SCALAR_STAGE, ENSEMBLE_STAGE))

REUSE_HISTORICAL_CLASSICAL = "historical_classical_reconstruction"
REUSE_HISTORICAL_NEURAL = "historical_neural_checkpoint"
REUSE_FUTURE = "future_retained_estimator"

CALIBRATION_CLASSICAL = "seed_average_then_single_temperature"
CALIBRATION_NEURAL = "per_seed_temperature_then_seed_average"

CLASSICAL_REFIT_STATUS = "completed_fit_record_not_persisted_estimator"
ENDPOINT_STATUS = "complete_saved_pipeline_endpoint"

CLASSICAL_SHARED_KEYS = (
    "candidate_id",
    "hyperparameter_sha256",
    "selection_state_sha256",
)
NEURAL_RESOLVED_KEYS = ("refit_id", "epochs", "calibration_state_sha256")

MIN_POLICY = next(
    policy for policy in POLICIES if EVIDENCE_BY_POLICY[policy] == EVIDENCE_HISTORICAL
)

_HEX_DIGITS = frozenset("0123456789abcdef")

_PLAN_KEYS = frozenset(
    (
        "schema_version",
        "execution_authorized",
        "jobs",
        "aliases",
        "summary",
        "plan_sha256",
    )
)
_JOB_KEYS = frozenset(JOB_FIELDS) | {"job_id"}
_ALIAS_KEYS = frozenset(
    (
        "policy_id",
        "context_id",
        "strategy",
        "recipe_id",
        "target_job_id",
        "alias_id",
    )
)
_BRIDGE_KEYS = frozenset(
    (
        "schema_version",
        "execution_authorized",
        "parent_plan_sha256",
        "frozen_minimal_input",
        "operation_bindings",
        "fallback_endpoint_aliases",
        "verified_file_hashes",
        "boundary",
    )
)
_BINDING_KEYS = (
    "job_id",
    "context_id",
    "model_id",
    "stage",
    "seed",
    "evidence_status",
    "resolved_source_values",
    "evidence",
    "scientific_execution_authorized",
    "binding_sha256",
)


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this metadata-only adapter."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID_METADATA)


def _require_mapping(value):
    if not isinstance(value, dict):
        _fail()
    return value


def _require_sequence(value):
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail()
    return list(value)


def _require_identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail()


def _require_lower_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        _fail()
    for character in value:
        if character not in _HEX_DIGITS:
            _fail()


def _require_seed(value):
    if isinstance(value, bool):
        _fail()
    if isinstance(value, int):
        return
    if isinstance(value, str) and value and value == value.strip():
        return
    _fail()


def _require_exact_keys(mapping, expected):
    if set(mapping.keys()) != set(expected):
        _fail()


def _model_seeds(model_id):
    if model_id == SVM_MODEL:
        return [SVM_SEED]
    return list(SEEDS)


def _parse_plan(plan):
    _require_mapping(plan)
    _require_exact_keys(plan, _PLAN_KEYS)
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        _fail()
    if plan["execution_authorized"] is not False:
        _fail()
    plan_sha = plan["plan_sha256"]
    _require_lower_hex64(plan_sha)
    jobs = _parse_jobs(plan["jobs"])
    aliases = _parse_aliases(plan["aliases"])
    return plan_sha, jobs, aliases


def _parse_final_job(raw):
    _require_exact_keys(raw, _JOB_KEYS)
    policy = raw["policy_id"]
    if policy not in POLICIES:
        _fail()
    if raw["representation_id"] != POLICY_REPRESENTATION[policy]:
        _fail()
    _require_identifier(raw["context_id"])
    model = raw["model_id"]
    if model not in CLASSICAL_MODELS and model not in NEURAL_RECIPES:
        _fail()
    _require_identifier(raw["stage"])
    _require_seed(raw["seed"])
    _require_lower_hex64(raw["array_sha256"])
    _require_lower_hex64(raw["model_spec_sha256"])
    _require_lower_hex64(raw["test_uid_sha256"])
    _require_identifier(raw["resolution"])
    if raw["evidence_status"] != EVIDENCE_BY_POLICY[policy]:
        _fail()
    dependencies = _require_sequence(raw["dependencies"])
    if len(dependencies) != len(set(dependencies)):
        _fail()
    for dependency in dependencies:
        _require_identifier(dependency)


def _upstream_content_id(body):
    """Reproduce upstream p08_plan content hashing (ASCII-escaped JSON)."""
    sha256_value(body)
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _require_job_content_id(raw):
    body = {key: value for key, value in raw.items() if key != "job_id"}
    if raw["job_id"] != "P08JOB-" + _upstream_content_id(body):
        _fail()


def _parse_jobs(raw_jobs):
    raw_jobs = _require_sequence(raw_jobs)
    seen = set()
    jobs = []
    for raw in raw_jobs:
        _require_mapping(raw)
        job_id = raw.get("job_id")
        _require_identifier(job_id)
        if job_id in seen:
            _fail()
        seen.add(job_id)
        stage = raw.get("stage")
        _require_identifier(stage)
        if stage in FINAL_STAGES:
            _parse_final_job(raw)
            _require_job_content_id(raw)
        jobs.append(raw)
    return jobs


def _parse_aliases(raw_aliases):
    raw_aliases = _require_sequence(raw_aliases)
    seen = set()
    aliases = []
    for raw in raw_aliases:
        _require_mapping(raw)
        _require_exact_keys(raw, _ALIAS_KEYS)
        if raw["policy_id"] not in POLICIES:
            _fail()
        _require_identifier(raw["context_id"])
        _require_identifier(raw["strategy"])
        _require_identifier(raw["recipe_id"])
        _require_identifier(raw["target_job_id"])
        alias_id = raw["alias_id"]
        _require_identifier(alias_id)
        if alias_id in seen:
            _fail()
        seen.add(alias_id)
        body = {key: value for key, value in raw.items() if key != "alias_id"}
        if alias_id != "P08ALIAS-" + _upstream_content_id(body):
            _fail()
        aliases.append(raw)
    return aliases


def _parse_bridge(minimal_bridge, plan_sha):
    _require_mapping(minimal_bridge)
    _require_exact_keys(minimal_bridge, _BRIDGE_KEYS)
    if minimal_bridge["schema_version"] != BRIDGE_SCHEMA_VERSION:
        _fail()
    if minimal_bridge["execution_authorized"] is not False:
        _fail()
    parent_plan_sha = minimal_bridge["parent_plan_sha256"]
    _require_lower_hex64(parent_plan_sha)
    if parent_plan_sha != plan_sha:
        _fail()

    raw_bindings = _require_sequence(minimal_bridge["operation_bindings"])
    index = {}
    for raw in raw_bindings:
        _require_mapping(raw)
        job_id = raw.get("job_id")
        _require_identifier(job_id)
        if job_id in index:
            _fail()
        index[job_id] = raw
    return index


def _seeded(jobs, expected_seeds):
    if len(jobs) != len(expected_seeds):
        _fail()
    by_seed = {}
    for job in jobs:
        seed = job["seed"]
        if seed in by_seed:
            _fail()
        by_seed[seed] = job
    if set(by_seed) != set(expected_seeds):
        _fail()
    return by_seed


def _normalise_group(key, members):
    policy, context, model = key
    is_classical = model in CLASSICAL_MODELS
    model_seeds = _model_seeds(model)
    scalar_seeds = [NOT_APPLICABLE] if is_classical else list(SEEDS)

    by_stage = {}
    for job in members:
        by_stage.setdefault(job["stage"], []).append(job)
    if set(by_stage) != FINAL_STAGES:
        _fail()

    refit_map = _seeded(by_stage[FINAL_REFIT_STAGE], model_seeds)
    held_map = _seeded(by_stage[HELD_STAGE], model_seeds)
    scalar_map = _seeded(by_stage[SCALAR_STAGE], scalar_seeds)

    endpoints = by_stage[ENSEMBLE_STAGE]
    if len(endpoints) != 1:
        _fail()
    endpoint = endpoints[0]
    if endpoint["seed"] != NOT_APPLICABLE:
        _fail()

    array_shas = {job["array_sha256"] for job in members}
    model_specs = {job["model_spec_sha256"] for job in members}
    tests = {job["test_uid_sha256"] for job in members}
    if len(array_shas) != 1 or len(model_specs) != 1 or len(tests) != 1:
        _fail()
    test_sha = next(iter(tests))

    refit_fits = {job["fit_uid_sha256"] for job in refit_map.values()}
    if len(refit_fits) != 1:
        _fail()
    fit_sha = next(iter(refit_fits))
    _require_lower_hex64(fit_sha)
    if fit_sha == test_sha:
        _fail()

    for job in held_map.values():
        if job["fit_uid_sha256"] != fit_sha:
            _fail()
    for job in list(scalar_map.values()) + [endpoint]:
        job_fit = job["fit_uid_sha256"]
        if job_fit != NOT_APPLICABLE and job_fit != fit_sha:
            _fail()

    refit_id_by_seed = {seed: job["job_id"] for seed, job in refit_map.items()}
    held_id_by_seed = {seed: job["job_id"] for seed, job in held_map.items()}
    scalar_id_by_seed = {seed: job["job_id"] for seed, job in scalar_map.items()}

    for seed, job in held_map.items():
        dependencies = list(job["dependencies"])
        if len(dependencies) != len(set(dependencies)):
            _fail()
        if is_classical:
            if dependencies != [refit_id_by_seed[seed]]:
                _fail()
        else:
            if len(dependencies) != 2 or set(dependencies) != {
                refit_id_by_seed[seed],
                scalar_id_by_seed[seed],
            }:
                _fail()

    endpoint_dependencies = list(endpoint["dependencies"])
    if len(endpoint_dependencies) != len(set(endpoint_dependencies)):
        _fail()
    if is_classical:
        expected_dependencies = set(held_id_by_seed.values()) | {scalar_id_by_seed[NOT_APPLICABLE]}
    else:
        expected_dependencies = set(held_id_by_seed.values())
    if (
        len(endpoint_dependencies) != len(expected_dependencies)
        or set(endpoint_dependencies) != expected_dependencies
    ):
        _fail()

    return {
        "policy_id": policy,
        "context_id": context,
        "model_id": model,
        "is_classical": is_classical,
        "model_seeds": model_seeds,
        "scalar_seeds": scalar_seeds,
        "refit_map": refit_map,
        "held_map": held_map,
        "scalar_map": scalar_map,
        "endpoint": endpoint,
        "array_sha256": next(iter(array_shas)),
        "model_spec_sha256": next(iter(model_specs)),
        "test_uid_sha256": test_sha,
        "fit_uid_sha256": fit_sha,
    }


def _build_groups(jobs):
    raw_groups = {}
    for job in jobs:
        if job["stage"] in FINAL_STAGES:
            key = (job["policy_id"], job["context_id"], job["model_id"])
            raw_groups.setdefault(key, []).append(job)
    return {key: _normalise_group(key, members) for key, members in raw_groups.items()}


def _validate_global(groups):
    array_by_policy = {}
    spec_by_model = {}
    fit_by_context = {}
    test_by_context = {}
    for group in groups.values():
        policy = group["policy_id"]
        if array_by_policy.setdefault(policy, group["array_sha256"]) != group["array_sha256"]:
            _fail()
        model = group["model_id"]
        if (
            spec_by_model.setdefault(model, group["model_spec_sha256"])
            != group["model_spec_sha256"]
        ):
            _fail()
        context = group["context_id"]
        if fit_by_context.setdefault(context, group["fit_uid_sha256"]) != group["fit_uid_sha256"]:
            _fail()
        if (
            test_by_context.setdefault(context, group["test_uid_sha256"])
            != group["test_uid_sha256"]
        ):
            _fail()


def _validate_aliases_and_models(groups, aliases):
    alias_table = {}
    for alias in aliases:
        alias_table.setdefault((alias["policy_id"], alias["context_id"]), []).append(alias)

    group_policy_context = {(key[0], key[1]) for key in groups}
    if set(alias_table) != group_policy_context:
        _fail()

    endpoint_by_key = {key: group["endpoint"]["job_id"] for key, group in groups.items()}

    selected_by_context = {}
    for (policy, context), entries in alias_table.items():
        if len(entries) != 2:
            _fail()
        by_strategy = {}
        for entry in entries:
            strategy = entry["strategy"]
            if strategy in by_strategy:
                _fail()
            by_strategy[strategy] = entry
        if set(by_strategy) != {D0_STRATEGY, SELECTED_STRATEGY}:
            _fail()

        d0_entry = by_strategy[D0_STRATEGY]
        selected_entry = by_strategy[SELECTED_STRATEGY]
        if d0_entry["recipe_id"] != D0_RECIPE:
            _fail()
        selected_recipe = selected_entry["recipe_id"]
        if selected_recipe not in NEURAL_RECIPES:
            _fail()
        if endpoint_by_key.get((policy, context, D0_RECIPE)) != d0_entry["target_job_id"]:
            _fail()
        if (
            endpoint_by_key.get((policy, context, selected_recipe))
            != selected_entry["target_job_id"]
        ):
            _fail()
        if selected_by_context.setdefault(context, selected_recipe) != selected_recipe:
            _fail()

    context_models = {}
    for policy, context, model in groups:
        context_models.setdefault(context, {}).setdefault(policy, set()).add(model)

    for context, policy_models in context_models.items():
        selected_recipe = selected_by_context.get(context)
        expected = set(CLASSICAL_MODELS) | {D0_RECIPE, selected_recipe}
        for policy in POLICIES:
            if policy_models.get(policy) != expected:
                _fail()


def _group_final_members(group):
    return (
        list(group["refit_map"].values())
        + list(group["held_map"].values())
        + list(group["scalar_map"].values())
        + [group["endpoint"]]
    )


def _classical_shared_values(resolved):
    for key in CLASSICAL_SHARED_KEYS:
        if key not in resolved:
            _fail()
    _require_identifier(resolved["candidate_id"])
    _require_lower_hex64(resolved["hyperparameter_sha256"])
    _require_lower_hex64(resolved["selection_state_sha256"])
    return (
        resolved["candidate_id"],
        resolved["hyperparameter_sha256"],
        resolved["selection_state_sha256"],
    )


def _neural_seed_values(resolved):
    for key in NEURAL_RESOLVED_KEYS:
        if key not in resolved:
            _fail()
    _require_identifier(resolved["refit_id"])
    epochs = resolved["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        _fail()
    _require_lower_hex64(resolved["calibration_state_sha256"])
    return (resolved["refit_id"], epochs, resolved["calibration_state_sha256"])


def _validate_referenced_binding(job, binding):
    _require_exact_keys(binding, _BINDING_KEYS)
    if not isinstance(binding["job_id"], str) or binding["job_id"] != job["job_id"]:
        _fail()
    if binding["context_id"] != job["context_id"]:
        _fail()
    if binding["model_id"] != job["model_id"]:
        _fail()
    if binding["stage"] != job["stage"]:
        _fail()
    if binding["seed"] != job["seed"]:
        _fail()
    if binding["scientific_execution_authorized"] is not False:
        _fail()
    _require_identifier(binding["evidence_status"])
    _require_mapping(binding["resolved_source_values"])
    evidence = _require_sequence(binding["evidence"])
    if not evidence:
        _fail()
    declared = binding["binding_sha256"]
    _require_lower_hex64(declared)
    unsigned = {key: binding[key] for key in _BINDING_KEYS if key != "binding_sha256"}
    if sha256_value(unsigned) != declared:
        _fail()


def _validate_minimum_evidence(groups, binding_index):
    for group in groups.values():
        if group["policy_id"] != MIN_POLICY:
            continue
        is_classical = group["is_classical"]
        classical_shared = []
        neural_by_seed = {}
        for job in _group_final_members(group):
            binding = binding_index.get(job["job_id"])
            if binding is None:
                _fail()
            _validate_referenced_binding(job, binding)

            status = binding["evidence_status"]
            if is_classical and job["stage"] == FINAL_REFIT_STAGE:
                if status != CLASSICAL_REFIT_STATUS:
                    _fail()
            if job["stage"] == ENSEMBLE_STAGE:
                if status != ENDPOINT_STATUS:
                    _fail()

            resolved = binding["resolved_source_values"]
            if is_classical:
                classical_shared.append(_classical_shared_values(resolved))
                if job["stage"] == SCALAR_STAGE:
                    if "calibration_state_sha256" not in resolved:
                        _fail()
                    _require_lower_hex64(resolved["calibration_state_sha256"])
            elif job["stage"] != ENSEMBLE_STAGE:
                seed = job["seed"]
                seed_values = _neural_seed_values(resolved)
                if seed in neural_by_seed and neural_by_seed[seed] != seed_values:
                    _fail()
                neural_by_seed[seed] = seed_values
        if is_classical and len(set(classical_shared)) > 1:
            _fail()


def _make_reference(job, seed, policy, binding_index):
    if policy == MIN_POLICY:
        binding = binding_index.get(job["job_id"])
        if binding is None:
            _fail()
        return {
            "job_id": job["job_id"],
            "seed": seed,
            "historical_binding_sha256": binding["binding_sha256"],
            "resolved_source_values": copy.deepcopy(binding["resolved_source_values"]),
        }
    return {
        "job_id": job["job_id"],
        "seed": seed,
        "historical_binding_sha256": None,
        "resolved_source_values": None,
    }


def _build_record(group, plan_sha, binding_index):
    policy = group["policy_id"]
    is_classical = group["is_classical"]

    if policy == MIN_POLICY:
        reuse_mode = REUSE_HISTORICAL_CLASSICAL if is_classical else REUSE_HISTORICAL_NEURAL
    else:
        reuse_mode = REUSE_FUTURE

    calibration_order = CALIBRATION_CLASSICAL if is_classical else CALIBRATION_NEURAL

    refit_references = [
        _make_reference(group["refit_map"][seed], seed, policy, binding_index)
        for seed in group["model_seeds"]
    ]
    calibration_references = [
        _make_reference(group["scalar_map"][seed], seed, policy, binding_index)
        for seed in group["scalar_seeds"]
    ]
    held_reference_jobs = [
        _make_reference(group["held_map"][seed], seed, policy, binding_index)
        for seed in group["model_seeds"]
    ]
    clean_endpoint_reference = _make_reference(
        group["endpoint"], group["endpoint"]["seed"], policy, binding_index
    )

    record = {
        "context_id": group["context_id"],
        "policy_id": policy,
        "model_id": group["model_id"],
        "representation_id": POLICY_REPRESENTATION[policy],
        "array_sha256": group["array_sha256"],
        "model_spec_sha256": group["model_spec_sha256"],
        "fit_uid_sha256": group["fit_uid_sha256"],
        "test_uid_sha256": group["test_uid_sha256"],
        "seeds": list(group["model_seeds"]),
        "parent_plan_sha256": plan_sha,
        "model_reuse_mode": reuse_mode,
        "calibration_order": calibration_order,
        "refit_references": refit_references,
        "calibration_references": calibration_references,
        "held_reference_jobs": held_reference_jobs,
        "clean_endpoint_reference": clean_endpoint_reference,
    }
    record["procedure_id"] = "P08STRESSPROC-" + sha256_value(record)
    return record


def _build_records(groups, plan_sha, binding_index):
    return [_build_record(groups[key], plan_sha, binding_index) for key in sorted(groups)]


def _build_strategy_aliases(aliases, records):
    endpoint_to_procedure = {}
    for record in records:
        endpoint_id = record["clean_endpoint_reference"]["job_id"]
        endpoint_to_procedure[endpoint_id] = record["procedure_id"]

    result = []
    for alias in sorted(
        aliases,
        key=lambda entry: (entry["policy_id"], entry["context_id"], entry["strategy"]),
    ):
        target = endpoint_to_procedure.get(alias["target_job_id"])
        if target is None:
            _fail()
        result.append(
            {
                "policy_id": alias["policy_id"],
                "context_id": alias["context_id"],
                "strategy": alias["strategy"],
                "recipe_id": alias["recipe_id"],
                "target_procedure_id": target,
                "upstream_alias_id": alias["alias_id"],
            }
        )
    return result


def _build_summary(records, strategy_aliases):
    by_policy = {}
    for policy in POLICIES:
        policy_records = [record for record in records if record["policy_id"] == policy]
        by_policy[policy] = {
            "procedures": len(policy_records),
            "seed_estimators": sum(len(record["seeds"]) for record in policy_records),
        }

    historical_classical = sum(
        len(record["seeds"])
        for record in records
        if record["model_reuse_mode"] == REUSE_HISTORICAL_CLASSICAL
    )
    historical_neural = sum(
        len(record["seeds"])
        for record in records
        if record["model_reuse_mode"] == REUSE_HISTORICAL_NEURAL
    )
    future_seed_estimators = sum(
        len(record["seeds"]) for record in records if record["model_reuse_mode"] == REUSE_FUTURE
    )

    return {
        "procedure_count": len(records),
        "seed_estimator_count": sum(len(record["seeds"]) for record in records),
        "historical_classical_reconstruction_slots": historical_classical,
        "historical_neural_checkpoint_slots": historical_neural,
        "future_seed_estimator_slots": future_seed_estimators,
        "reporting_alias_count": len(strategy_aliases),
        "by_policy": by_policy,
    }


def _build_universal_procedure_records(plan, minimal_bridge):
    plan_sha, jobs, aliases = _parse_plan(plan)
    binding_index = _parse_bridge(minimal_bridge, plan_sha)
    bridge_sha = sha256_value(minimal_bridge)

    groups = _build_groups(jobs)
    if not groups:
        _fail()

    _validate_global(groups)
    _validate_aliases_and_models(groups, aliases)
    _validate_minimum_evidence(groups, binding_index)

    records = _build_records(groups, plan_sha, binding_index)
    strategy_aliases = _build_strategy_aliases(aliases, records)
    summary = _build_summary(records, strategy_aliases)

    payload = {
        "schema_version": PROCEDURE_SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "exact_prediction_ledger_complete": False,
        "parent_plan_sha256": plan_sha,
        "minimal_bridge_metadata_sha256": bridge_sha,
        "records": records,
        "strategy_aliases": strategy_aliases,
        "summary": summary,
    }
    payload["catalog_sha256"] = sha256_value(payload)
    return payload


def build_universal_procedure_records(*, plan, minimal_bridge):
    """Build metadata-only universal stress procedure records.

    The supplied ``plan`` and ``minimal_bridge`` are only read and copied from;
    they are never mutated.  The returned dictionary binds the supplied
    metadata and always keeps ``execution_authorized`` ``False``.
    """
    try:
        return _build_universal_procedure_records(plan, minimal_bridge)
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None
