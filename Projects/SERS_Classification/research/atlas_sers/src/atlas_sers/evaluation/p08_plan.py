"""P08-T006 universal no-fit slot DAG planner.

This module builds a deterministic, data-free DAG of scientific *slots* for
the P08 protocol.  It never fits a model, never selects epochs, scalars,
hyperparameters, thresholds or durations, and never authorizes execution.
Every dynamic value is deferred to registered future source outcomes.

Frozen identifiers recorded here:

* policy ids ``PP-U-MIN``, ``PP-U-SG``, ``PP-U-ARPLS``;
* representation ids ``R_MIN_400_1800``, ``R_SG_400_1800``,
  ``R_ARPLS_400_1800``;
* schema namespace ``nato-sers-p08-universal-slot-dag-v1``;
* selection modes ``master_cv`` and ``pseudo_domain`` only.

Calibration ordering is recorded, not invented.  The classical forest
procedure averages the per-seed *uncalibrated* held probabilities, maps the
mean through ``log(clip(mean, 1e-7, 1 - 1e-7))`` and then applies one
source-fitted temperature.  Consequently a classical ``held_prediction``
depends only on its ``final_refit`` (``uncalibrated_scores_only``) and the
classical ``seed_ensemble_prediction`` depends on every raw held prediction
plus the single ``scalar_calibration``
(``seed_average_then_single_temperature``).  Neural recipes keep the opposite
order: a per-seed temperature is applied first, each neural
``held_prediction`` depends on that seed's ``scalar_calibration``, and the
neural ``seed_ensemble_prediction`` depends only on the held predictions.

Master calibration identifiers: upstream evidence may canonicalise a raw label
such as ``calibration_master_cv:N`` to the matching selection id
``master_cv:N`` only after proving an exact fit/validation set-hash match.
This API is deliberately stricter: master calibration units must already carry
the exact selection ``unit_id`` together with both identical hashes.  Pseudo
contexts always build new calibration fits; no alias is created from label
text.

Limitations that the caller must keep in mind:

* No observations enter this interface.  Authentication of upstream roles,
  source/validation/test units and data provenance is entirely the caller's
  responsibility.
* Distinct hashes are only a necessary recording condition.  Hash difference
  alone does not prove that units, calibration sets or outer partitions are
  disjoint; a full caller-side role audit is still required before any
  scientific use.
* ``require_scientific_execution`` always denies execution from this readiness
  module, even when handed a forged ``execution_authorized`` flag.

Nothing here writes files, draws random numbers, or imports scientific stacks.
"""

from __future__ import annotations

import hashlib
import json

__all__ = [
    "PlanError",
    "REASON_CODES",
    "SCHEMA_VERSION",
    "build_universal_plan",
    "require_scientific_execution",
]


SCHEMA_VERSION = "nato-sers-p08-universal-slot-dag-v1"

POLICIES = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")
POLICY_REPRESENTATION = {
    "PP-U-MIN": "R_MIN_400_1800",
    "PP-U-SG": "R_SG_400_1800",
    "PP-U-ARPLS": "R_ARPLS_400_1800",
}

SEEDS = (20260805, 20260817, 20260829)
SVM_SEED = "deterministic"

SVM_MODEL = "C-RBF-SVM"
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")

D0_RECIPE = "D0-M"
D0_STRATEGY = "D0-M"
SELECTED_STRATEGY = "P05-SELECTED"

MASTER_MODE = "master_cv"
PSEUDO_MODE = "pseudo_domain"

EVIDENCE_HISTORICAL = "historical_reuse_requires_authentication"
EVIDENCE_FUTURE = "unapproved_future_job"
EVIDENCE_BY_POLICY = {
    "PP-U-MIN": EVIDENCE_HISTORICAL,
    "PP-U-SG": EVIDENCE_FUTURE,
    "PP-U-ARPLS": EVIDENCE_FUTURE,
}

NOT_APPLICABLE = "not_applicable"
FIXED_SPEC = "fixed_spec"
SOURCE_SELECTION_DEPENDENT = "source_selection_dependent"
SOURCE_EPOCH_DEPENDENT = "source_epoch_dependent"
RESOLVE_SELECTED_CANDIDATE = "resolve_selected_candidate_after_source_selection"

RESOLUTION_UNCALIBRATED = "uncalibrated_scores_only"
RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE = "seed_average_then_single_temperature"
RESOLUTION_CALIBRATE_SEED_AVERAGED = "calibrate_seed_averaged_source_scores"

MODEL_FIT_STAGES = frozenset(("source_fit", "calibration_model_fit", "final_refit"))
SCALAR_STAGE = "scalar_calibration"
CALIBRATION_ALIAS_STAGE = "calibration_prediction_alias"

JOB_FIELDS = (
    "policy_id",
    "representation_id",
    "array_sha256",
    "context_id",
    "model_id",
    "model_spec_sha256",
    "stage",
    "unit_id",
    "seed",
    "candidate_id",
    "hyperparameter_sha256",
    "fit_uid_sha256",
    "validation_uid_sha256",
    "test_uid_sha256",
    "dependencies",
    "resolution",
    "evidence_status",
)

REASON_CODES = frozenset(
    (
        "scientific_execution_not_authorized",
        "actions_must_be_mapping",
        "actions_keys_invalid",
        "action_hash_invalid",
        "model_spec_must_be_mapping",
        "model_spec_keys_invalid",
        "model_spec_hash_invalid",
        "candidates_must_be_sequence",
        "candidates_empty",
        "candidate_must_be_mapping",
        "candidate_keys_invalid",
        "candidate_id_invalid",
        "candidate_model_id_invalid",
        "candidate_model_not_permitted",
        "duplicate_candidate_id",
        "candidate_hyperparameter_hash_invalid",
        "candidate_counts_invalid",
        "contexts_must_be_sequence",
        "contexts_empty",
        "context_must_be_mapping",
        "context_keys_invalid",
        "context_id_invalid",
        "duplicate_context_id",
        "selection_mode_invalid",
        "selected_recipe_id_invalid",
        "selected_recipe_not_permitted",
        "master_recipe_must_be_d0",
        "outer_fit_uid_hash_invalid",
        "outer_test_uid_hash_invalid",
        "outer_fit_test_equal",
        "selection_units_must_be_sequence",
        "selection_units_too_few",
        "master_selection_units_count_invalid",
        "calibration_units_must_be_sequence",
        "calibration_units_too_few",
        "calibration_units_count_invalid",
        "unit_must_be_mapping",
        "unit_keys_invalid",
        "unit_id_invalid",
        "duplicate_unit_id",
        "fit_uid_hash_invalid",
        "validation_uid_hash_invalid",
        "unit_fit_validation_equal",
        "unit_conflicts_outer_test",
        "master_calibration_id_mismatch",
        "master_calibration_hash_mismatch",
    )
)

_HEX_DIGITS = frozenset("0123456789abcdef")


class PlanError(ValueError):
    """Raised for malformed plan inputs or any attempted execution."""


def require_scientific_execution(plan):
    """Always deny scientific execution for this readiness module.

    The argument is deliberately ignored, so even a forged
    ``{"execution_authorized": True}`` payload cannot open the gate.
    """
    raise PlanError("scientific_execution_not_authorized")


def _hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(code):
    if code not in REASON_CODES:
        raise AssertionError(f"unregistered reason code: {code}")
    raise PlanError(code)


def _require_mapping(value, code):
    if not isinstance(value, dict):
        _fail(code)


def _require_sequence(value, code):
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail(code)


def _require_identifier(value, code):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)


def _require_sha256(value, code):
    if not isinstance(value, str) or len(value) != 64:
        _fail(code)
    for character in value:
        if character not in _HEX_DIGITS:
            _fail(code)


def _require_exact_keys(mapping, expected, code):
    if set(mapping.keys()) != set(expected):
        _fail(code)


def _parse_actions(actions):
    _require_mapping(actions, "actions_must_be_mapping")
    if set(actions.keys()) != set(POLICY_REPRESENTATION.values()):
        _fail("actions_keys_invalid")
    parsed = {}
    for key in sorted(actions):
        _require_sha256(actions[key], "action_hash_invalid")
        parsed[key] = actions[key]
    return parsed


def _parse_model_spec(model_spec_sha256):
    _require_mapping(model_spec_sha256, "model_spec_must_be_mapping")
    if set(model_spec_sha256.keys()) != set(CLASSICAL_MODELS) | set(NEURAL_RECIPES):
        _fail("model_spec_keys_invalid")
    parsed = {}
    for key in sorted(model_spec_sha256):
        _require_sha256(model_spec_sha256[key], "model_spec_hash_invalid")
        parsed[key] = model_spec_sha256[key]
    return parsed


def _parse_candidates(candidates):
    _require_sequence(candidates, "candidates_must_be_sequence")
    if not candidates:
        _fail("candidates_empty")
    counts = {model: 0 for model in CLASSICAL_MODELS}
    seen = set()
    parsed = []
    for raw in candidates:
        _require_mapping(raw, "candidate_must_be_mapping")
        _require_exact_keys(
            raw,
            ("candidate_id", "model_id", "hyperparameter_sha256"),
            "candidate_keys_invalid",
        )
        candidate_id = raw["candidate_id"]
        model_id = raw["model_id"]
        _require_identifier(candidate_id, "candidate_id_invalid")
        _require_identifier(model_id, "candidate_model_id_invalid")
        if model_id not in CLASSICAL_MODELS:
            _fail("candidate_model_not_permitted")
        if candidate_id in seen:
            _fail("duplicate_candidate_id")
        _require_sha256(raw["hyperparameter_sha256"], "candidate_hyperparameter_hash_invalid")
        seen.add(candidate_id)
        counts[model_id] += 1
        parsed.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "hyperparameter_sha256": raw["hyperparameter_sha256"],
            }
        )
    if (
        counts[SVM_MODEL] != 36
        or counts["C-RANDOM-FOREST"] != 16
        or counts["C-EXTRA-TREES"] != 16
    ):
        _fail("candidate_counts_invalid")
    parsed.sort(key=lambda candidate: candidate["candidate_id"])
    return parsed


def _parse_units(raw_units, minimum, sequence_code, too_few_code):
    _require_sequence(raw_units, sequence_code)
    if len(raw_units) < minimum:
        _fail(too_few_code)
    seen = set()
    parsed = []
    for raw in raw_units:
        _require_mapping(raw, "unit_must_be_mapping")
        _require_exact_keys(
            raw,
            ("unit_id", "fit_uid_sha256", "validation_uid_sha256"),
            "unit_keys_invalid",
        )
        unit_id = raw["unit_id"]
        _require_identifier(unit_id, "unit_id_invalid")
        if unit_id in seen:
            _fail("duplicate_unit_id")
        seen.add(unit_id)
        _require_sha256(raw["fit_uid_sha256"], "fit_uid_hash_invalid")
        _require_sha256(raw["validation_uid_sha256"], "validation_uid_hash_invalid")
        parsed.append(
            {
                "unit_id": unit_id,
                "fit_uid_sha256": raw["fit_uid_sha256"],
                "validation_uid_sha256": raw["validation_uid_sha256"],
            }
        )
    parsed.sort(key=lambda unit: unit["unit_id"])
    return parsed


def _parse_contexts(contexts):
    _require_sequence(contexts, "contexts_must_be_sequence")
    if not contexts:
        _fail("contexts_empty")
    seen = set()
    parsed = []
    for raw in contexts:
        _require_mapping(raw, "context_must_be_mapping")
        _require_exact_keys(
            raw,
            (
                "context_id",
                "selection_mode",
                "selected_recipe_id",
                "outer_fit_uid_sha256",
                "outer_test_uid_sha256",
                "selection_units",
                "calibration_units",
            ),
            "context_keys_invalid",
        )
        context_id = raw["context_id"]
        _require_identifier(context_id, "context_id_invalid")
        if context_id in seen:
            _fail("duplicate_context_id")
        seen.add(context_id)

        mode = raw["selection_mode"]
        _require_identifier(mode, "selection_mode_invalid")
        if mode not in (MASTER_MODE, PSEUDO_MODE):
            _fail("selection_mode_invalid")

        recipe = raw["selected_recipe_id"]
        _require_identifier(recipe, "selected_recipe_id_invalid")
        if recipe not in NEURAL_RECIPES:
            _fail("selected_recipe_not_permitted")
        if mode == MASTER_MODE and recipe != D0_RECIPE:
            _fail("master_recipe_must_be_d0")

        outer_fit = raw["outer_fit_uid_sha256"]
        outer_test = raw["outer_test_uid_sha256"]
        _require_sha256(outer_fit, "outer_fit_uid_hash_invalid")
        _require_sha256(outer_test, "outer_test_uid_hash_invalid")
        if outer_fit == outer_test:
            _fail("outer_fit_test_equal")

        selection_minimum = 3 if mode == MASTER_MODE else 2
        selection_units = _parse_units(
            raw["selection_units"],
            selection_minimum,
            "selection_units_must_be_sequence",
            "selection_units_too_few",
        )
        if mode == MASTER_MODE and len(selection_units) != 3:
            _fail("master_selection_units_count_invalid")
        calibration_units = _parse_units(
            raw["calibration_units"],
            3,
            "calibration_units_must_be_sequence",
            "calibration_units_too_few",
        )
        if len(calibration_units) != 3:
            _fail("calibration_units_count_invalid")

        for unit in selection_units + calibration_units:
            if unit["fit_uid_sha256"] == unit["validation_uid_sha256"]:
                _fail("unit_fit_validation_equal")
            if unit["fit_uid_sha256"] == outer_test or unit["validation_uid_sha256"] == outer_test:
                _fail("unit_conflicts_outer_test")

        if mode == MASTER_MODE:
            selection_by_id = {unit["unit_id"]: unit for unit in selection_units}
            if set(selection_by_id) != {unit["unit_id"] for unit in calibration_units}:
                _fail("master_calibration_id_mismatch")
            for unit in calibration_units:
                selection_unit = selection_by_id[unit["unit_id"]]
                if (
                    selection_unit["fit_uid_sha256"] != unit["fit_uid_sha256"]
                    or selection_unit["validation_uid_sha256"] != unit["validation_uid_sha256"]
                ):
                    _fail("master_calibration_hash_mismatch")

        parsed.append(
            {
                "context_id": context_id,
                "selection_mode": mode,
                "selected_recipe_id": recipe,
                "outer_fit_uid_sha256": outer_fit,
                "outer_test_uid_sha256": outer_test,
                "selection_units": selection_units,
                "calibration_units": calibration_units,
            }
        )
    parsed.sort(key=lambda context: context["context_id"])
    return parsed


def _new_job(fields, dependencies):
    job = dict(fields)
    job["dependencies"] = sorted(dependencies)
    job["job_id"] = "P08JOB-" + _hash(job)
    return job


def _new_alias(policy_id, context_id, strategy, recipe_id, target_job_id):
    fields = {
        "policy_id": policy_id,
        "context_id": context_id,
        "strategy": strategy,
        "recipe_id": recipe_id,
        "target_job_id": target_job_id,
    }
    alias = dict(fields)
    alias["alias_id"] = "P08ALIAS-" + _hash(fields)
    return alias


def _build_context(policy, representation, array_sha, evidence, ctx, candidates_by_model, spec):
    jobs = []
    aliases = []
    context_id = ctx["context_id"]
    outer_fit_sha = ctx["outer_fit_uid_sha256"]
    test_sha = ctx["outer_test_uid_sha256"]

    def fields(
        model_id,
        stage,
        unit_id,
        seed,
        candidate_id,
        hyperparameter_sha256,
        fit_sha,
        validation_sha,
        resolution,
    ):
        return {
            "policy_id": policy,
            "representation_id": representation,
            "array_sha256": array_sha,
            "context_id": context_id,
            "model_id": model_id,
            "model_spec_sha256": spec[model_id],
            "stage": stage,
            "unit_id": unit_id,
            "seed": seed,
            "candidate_id": candidate_id,
            "hyperparameter_sha256": hyperparameter_sha256,
            "fit_uid_sha256": fit_sha,
            "validation_uid_sha256": validation_sha,
            "test_uid_sha256": test_sha,
            "resolution": resolution,
            "evidence_status": evidence,
        }

    for model_id in CLASSICAL_MODELS:
        model_seeds = (SVM_SEED,) if model_id == SVM_MODEL else SEEDS
        model_candidates = candidates_by_model[model_id]
        source_prediction_ids = {}

        for unit in ctx["selection_units"]:
            for candidate in model_candidates:
                for seed in model_seeds:
                    fit_fields = fields(
                        model_id,
                        "source_fit",
                        unit["unit_id"],
                        seed,
                        candidate["candidate_id"],
                        candidate["hyperparameter_sha256"],
                        unit["fit_uid_sha256"],
                        unit["validation_uid_sha256"],
                        FIXED_SPEC,
                    )
                    fit_job = _new_job(fit_fields, [])
                    jobs.append(fit_job)
                    prediction_fields = dict(fit_fields)
                    prediction_fields["stage"] = "source_validation_prediction"
                    prediction_job = _new_job(prediction_fields, [fit_job["job_id"]])
                    jobs.append(prediction_job)
                    source_prediction_ids[
                        (unit["unit_id"], seed, candidate["candidate_id"])
                    ] = prediction_job["job_id"]

        select_job = _new_job(
            fields(
                model_id,
                "select_hyperparameters",
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_SELECTION_DEPENDENT,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_SELECTION_DEPENDENT,
            ),
            sorted(source_prediction_ids.values()),
        )
        jobs.append(select_job)

        scalar_dependencies = []
        if ctx["selection_mode"] == PSEUDO_MODE:
            for unit in ctx["calibration_units"]:
                for seed in model_seeds:
                    calibration_fields = fields(
                        model_id,
                        "calibration_model_fit",
                        unit["unit_id"],
                        seed,
                        SOURCE_SELECTION_DEPENDENT,
                        NOT_APPLICABLE,
                        unit["fit_uid_sha256"],
                        unit["validation_uid_sha256"],
                        SOURCE_SELECTION_DEPENDENT,
                    )
                    calibration_fit = _new_job(calibration_fields, [select_job["job_id"]])
                    jobs.append(calibration_fit)
                    prediction_fields = dict(calibration_fields)
                    prediction_fields["stage"] = "calibration_validation_prediction"
                    prediction_job = _new_job(prediction_fields, [calibration_fit["job_id"]])
                    jobs.append(prediction_job)
                    scalar_dependencies.append(prediction_job["job_id"])
        else:
            for unit in ctx["calibration_units"]:
                for seed in model_seeds:
                    dependency_ids = [select_job["job_id"]]
                    dependency_ids.extend(
                        source_prediction_ids[
                            (unit["unit_id"], seed, candidate["candidate_id"])
                        ]
                        for candidate in model_candidates
                    )
                    alias_job = _new_job(
                        fields(
                            model_id,
                            CALIBRATION_ALIAS_STAGE,
                            unit["unit_id"],
                            seed,
                            SOURCE_SELECTION_DEPENDENT,
                            NOT_APPLICABLE,
                            unit["fit_uid_sha256"],
                            unit["validation_uid_sha256"],
                            RESOLVE_SELECTED_CANDIDATE,
                        ),
                        dependency_ids,
                    )
                    jobs.append(alias_job)
                    scalar_dependencies.append(alias_job["job_id"])

        scalar_job = _new_job(
            fields(
                model_id,
                SCALAR_STAGE,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_SELECTION_DEPENDENT,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                RESOLUTION_CALIBRATE_SEED_AVERAGED,
            ),
            scalar_dependencies,
        )
        jobs.append(scalar_job)

        held_ids = []
        for seed in model_seeds:
            refit_job = _new_job(
                fields(
                    model_id,
                    "final_refit",
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    outer_fit_sha,
                    NOT_APPLICABLE,
                    SOURCE_SELECTION_DEPENDENT,
                ),
                [select_job["job_id"]],
            )
            jobs.append(refit_job)
            held_job = _new_job(
                fields(
                    model_id,
                    "held_prediction",
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    outer_fit_sha,
                    NOT_APPLICABLE,
                    RESOLUTION_UNCALIBRATED,
                ),
                [refit_job["job_id"]],
            )
            jobs.append(held_job)
            held_ids.append(held_job["job_id"])

        ensemble_job = _new_job(
            fields(
                model_id,
                "seed_ensemble_prediction",
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_SELECTION_DEPENDENT,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE,
            ),
            held_ids + [scalar_job["job_id"]],
        )
        jobs.append(ensemble_job)

    recipes = [D0_RECIPE]
    if ctx["selected_recipe_id"] != D0_RECIPE:
        recipes.append(ctx["selected_recipe_id"])

    neural_ensemble_by_recipe = {}
    for recipe in sorted(recipes):
        source_prediction_by_seed = {}
        for unit in ctx["selection_units"]:
            for seed in SEEDS:
                fit_fields = fields(
                    recipe,
                    "source_fit",
                    unit["unit_id"],
                    seed,
                    "fixed_recipe",
                    spec[recipe],
                    unit["fit_uid_sha256"],
                    unit["validation_uid_sha256"],
                    FIXED_SPEC,
                )
                fit_job = _new_job(fit_fields, [])
                jobs.append(fit_job)
                prediction_fields = dict(fit_fields)
                prediction_fields["stage"] = "source_validation_prediction"
                prediction_job = _new_job(prediction_fields, [fit_job["job_id"]])
                jobs.append(prediction_job)
                source_prediction_by_seed.setdefault(seed, []).append(prediction_job["job_id"])

        held_ids = []
        for seed in SEEDS:
            inherited = sorted(source_prediction_by_seed[seed])
            select_refit_job = _new_job(
                fields(
                    recipe,
                    "select_refit_epochs",
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    NOT_APPLICABLE,
                    NOT_APPLICABLE,
                    SOURCE_EPOCH_DEPENDENT,
                ),
                inherited,
            )
            jobs.append(select_refit_job)
            scalar_job = _new_job(
                fields(
                    recipe,
                    SCALAR_STAGE,
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    NOT_APPLICABLE,
                    NOT_APPLICABLE,
                    SOURCE_EPOCH_DEPENDENT,
                ),
                inherited,
            )
            jobs.append(scalar_job)
            refit_job = _new_job(
                fields(
                    recipe,
                    "final_refit",
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    outer_fit_sha,
                    NOT_APPLICABLE,
                    SOURCE_EPOCH_DEPENDENT,
                ),
                [select_refit_job["job_id"]],
            )
            jobs.append(refit_job)
            held_job = _new_job(
                fields(
                    recipe,
                    "held_prediction",
                    NOT_APPLICABLE,
                    seed,
                    SOURCE_SELECTION_DEPENDENT,
                    NOT_APPLICABLE,
                    outer_fit_sha,
                    NOT_APPLICABLE,
                    SOURCE_EPOCH_DEPENDENT,
                ),
                [refit_job["job_id"], scalar_job["job_id"]],
            )
            jobs.append(held_job)
            held_ids.append(held_job["job_id"])

        ensemble_job = _new_job(
            fields(
                recipe,
                "seed_ensemble_prediction",
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_SELECTION_DEPENDENT,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                NOT_APPLICABLE,
                SOURCE_EPOCH_DEPENDENT,
            ),
            held_ids,
        )
        jobs.append(ensemble_job)
        neural_ensemble_by_recipe[recipe] = ensemble_job["job_id"]

    aliases.append(
        _new_alias(
            policy,
            context_id,
            D0_STRATEGY,
            D0_RECIPE,
            neural_ensemble_by_recipe[D0_RECIPE],
        )
    )
    selected_recipe = ctx["selected_recipe_id"]
    aliases.append(
        _new_alias(
            policy,
            context_id,
            SELECTED_STRATEGY,
            selected_recipe,
            neural_ensemble_by_recipe[selected_recipe],
        )
    )

    return jobs, aliases


def _sorted_stage_counts(jobs):
    counts = {}
    for job in jobs:
        counts[job["stage"]] = counts.get(job["stage"], 0) + 1
    return {stage: counts[stage] for stage in sorted(counts)}


def _aggregate_policies(by_policy, policy_names):
    stage_counts = {}
    aliases = 0
    total_jobs = 0
    for policy in policy_names:
        entry = by_policy[policy]
        total_jobs += entry["total_jobs"]
        aliases += entry["aliases"]
        for stage, count in entry["stage_counts"].items():
            stage_counts[stage] = stage_counts.get(stage, 0) + count
    ordered_stages = {stage: stage_counts[stage] for stage in sorted(stage_counts)}
    model_fit_slots = sum(
        count for stage, count in ordered_stages.items() if stage in MODEL_FIT_STAGES
    )
    prediction_stage_counts = {
        stage: count for stage, count in ordered_stages.items() if stage.endswith("_prediction")
    }
    return {
        "total_jobs": total_jobs,
        "model_fit_slots": model_fit_slots,
        "scalar_calibrations": ordered_stages.get(SCALAR_STAGE, 0),
        "stage_counts": ordered_stages,
        "prediction_stage_counts": prediction_stage_counts,
        "calibration_prediction_aliases": ordered_stages.get(CALIBRATION_ALIAS_STAGE, 0),
        "aliases": aliases,
    }


def _build_summary(jobs, aliases):
    by_policy = {}
    for policy in POLICIES:
        policy_jobs = [job for job in jobs if job["policy_id"] == policy]
        stage_counts = _sorted_stage_counts(policy_jobs)
        model_fit_slots = sum(
            count for stage, count in stage_counts.items() if stage in MODEL_FIT_STAGES
        )
        prediction_stage_counts = {
            stage: count
            for stage, count in stage_counts.items()
            if stage.endswith("_prediction")
        }
        aliases_count = sum(1 for alias in aliases if alias["policy_id"] == policy)
        by_policy[policy] = {
            "total_jobs": len(policy_jobs),
            "model_fit_slots": model_fit_slots,
            "scalar_calibrations": stage_counts.get(SCALAR_STAGE, 0),
            "stage_counts": stage_counts,
            "prediction_stage_counts": prediction_stage_counts,
            "calibration_prediction_aliases": stage_counts.get(CALIBRATION_ALIAS_STAGE, 0),
            "aliases": aliases_count,
        }

    return {
        "by_policy": by_policy,
        "totals": _aggregate_policies(by_policy, POLICIES),
        "nonminimal_totals": _aggregate_policies(by_policy, ("PP-U-SG", "PP-U-ARPLS")),
        "authorized_fit_slots": 0,
    }


def build_universal_plan(contexts, candidates, actions, model_spec_sha256):
    """Build the data-free universal P08 slot DAG.

    All inputs are validated and copied; nothing is mutated.  The returned
    dictionary is JSON-safe and is always marked ``execution_authorized``
    ``False``.  No scientific execution is possible through this module.
    """
    parsed_actions = _parse_actions(actions)
    parsed_spec = _parse_model_spec(model_spec_sha256)
    parsed_candidates = _parse_candidates(candidates)
    parsed_contexts = _parse_contexts(contexts)

    candidates_by_model = {model: [] for model in CLASSICAL_MODELS}
    for candidate in parsed_candidates:
        candidates_by_model[candidate["model_id"]].append(candidate)

    jobs = []
    aliases = []
    for policy in POLICIES:
        representation = POLICY_REPRESENTATION[policy]
        array_sha = parsed_actions[representation]
        evidence = EVIDENCE_BY_POLICY[policy]
        for ctx in parsed_contexts:
            context_jobs, context_aliases = _build_context(
                policy,
                representation,
                array_sha,
                evidence,
                ctx,
                candidates_by_model,
                parsed_spec,
            )
            jobs.extend(context_jobs)
            aliases.extend(context_aliases)

    jobs.sort(key=lambda job: job["job_id"])
    aliases.sort(key=lambda alias: alias["alias_id"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "jobs": jobs,
        "aliases": aliases,
        "summary": _build_summary(jobs, aliases),
    }
    plan = dict(payload)
    plan["plan_sha256"] = _hash(payload)
    return plan
