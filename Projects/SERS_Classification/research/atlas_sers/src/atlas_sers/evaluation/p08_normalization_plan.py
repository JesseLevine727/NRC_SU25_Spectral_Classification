"""P08-T159 approved N2 no-fit normalization/control slot DAG planner.

This pure module builds a deterministic, data-free DAG of scientific *slots*
for the owner-approved N2 branch of P08.  Each original context keeps the
classical family selected by its saved MIN source state, and that family's
existing full hyperparameter grid is re-expanded separately under four
normalization/control representations.  No family is reselected, no neural
recipe is emitted, no previously fitted estimator is reused, no additional
grid is invented, and no scientific execution, training permission or
missing-result repair is authorized here.

Frozen identifiers recorded here:

* schema namespace ``nato-sers-p08-n2-slot-dag-v1``;
* representation ids ``R_SNV_400_1800``, ``R_VECTOR_400_1800``,
  ``R_AREA_400_1800``, ``R_D1_400_1800`` (D1 is a derivative *input*, not the
  neural recipe of the universal planner);
* policy ids ``PP-NORM-SNV``, ``PP-NORM-VECTOR``, ``PP-NORM-AREA``,
  ``PP-NORM-D1``;
* selection modes ``master_cv`` and ``pseudo_domain`` only.

The classical calibration ordering is inherited unchanged from the universal
planner: raw uncalibrated held probabilities are seed-averaged and then one
single temperature is applied.  Consequently a ``held_prediction`` job depends
only on its ``final_refit``, the single ``scalar_calibration`` consumes every
step-3 calibration output, and the ``seed_ensemble_prediction`` consumes every
raw held prediction plus that one scalar.  Deterministic families are simply
the one-seed special case.

Authentication limitations the caller must keep in mind:

* No observations, outcomes or fitted values enter this interface.  The
  ``source_family_mapping_sha256`` argument is only recorded; the caller must
  authenticate it against the saved MIN selections.
* Distinct hashes are a necessary recording condition only.  Hash difference
  alone does not prove role disjointness; a full upstream physical-master,
  instrument and row-set audit remains independently required.
* ``require_scientific_execution`` always denies execution, even for a forged
  ``execution_authorized`` payload.

Nothing here imports scientific stacks, draws random numbers, touches the
filesystem, reads clocks/environment, or spawns subprocesses.
"""

from __future__ import annotations

import hashlib
import json

__all__ = [
    "NormalizationPlanError",
    "REASON_CODES",
    "SCHEMA_VERSION",
    "POLICIES",
    "POLICY_REPRESENTATION",
    "FAMILY_MODELS",
    "FAMILY_CANDIDATE_COUNTS",
    "STOCHASTIC_FAMILIES",
    "STOCHASTIC_SEEDS",
    "DETERMINISTIC_SEED",
    "MASTER_MODE",
    "PSEUDO_MODE",
    "build_normalization_plan",
    "require_scientific_execution",
]


SCHEMA_VERSION = "nato-sers-p08-n2-slot-dag-v1"

POLICIES = ("PP-NORM-SNV", "PP-NORM-VECTOR", "PP-NORM-AREA", "PP-NORM-D1")
POLICY_REPRESENTATION = {
    "PP-NORM-SNV": "R_SNV_400_1800",
    "PP-NORM-VECTOR": "R_VECTOR_400_1800",
    "PP-NORM-AREA": "R_AREA_400_1800",
    "PP-NORM-D1": "R_D1_400_1800",
}

FAMILY_MODELS = (
    "C-SPECTRAL-MATCH",
    "C-NEAREST-CENTROID",
    "C-PCA-LDA",
    "C-PLS-DA",
    "C-LOGREG-EN",
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
)
FAMILY_CANDIDATE_COUNTS = {
    "C-SPECTRAL-MATCH": 3,
    "C-NEAREST-CENTROID": 8,
    "C-PCA-LDA": 10,
    "C-PLS-DA": 5,
    "C-LOGREG-EN": 30,
    "C-RBF-SVM": 36,
    "C-RANDOM-FOREST": 16,
    "C-EXTRA-TREES": 16,
}
STOCHASTIC_FAMILIES = ("C-RANDOM-FOREST", "C-EXTRA-TREES")
STOCHASTIC_SEEDS = (20260805, 20260817, 20260829)
DETERMINISTIC_SEED = "deterministic"

MASTER_MODE = "master_cv"
PSEUDO_MODE = "pseudo_domain"

EVIDENCE_FUTURE = "unapproved_future_job"

NOT_APPLICABLE = "not_applicable"
FIXED_SPEC = "fixed_spec"
SOURCE_SELECTION_DEPENDENT = "source_selection_dependent"
RESOLVE_SELECTED_CANDIDATE = "resolve_selected_candidate_after_source_selection"

RESOLUTION_UNCALIBRATED = "uncalibrated_scores_only"
RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE = "seed_average_then_single_temperature"
RESOLUTION_CALIBRATE_SEED_AVERAGED = "calibrate_seed_averaged_source_scores"

SOURCE_FIT_STAGE = "source_fit"
SOURCE_PREDICTION_STAGE = "source_validation_prediction"
SELECT_STAGE = "select_hyperparameters"
CALIBRATION_FIT_STAGE = "calibration_model_fit"
CALIBRATION_PREDICTION_STAGE = "calibration_validation_prediction"
SCALAR_STAGE = "scalar_calibration"
CALIBRATION_ALIAS_STAGE = "calibration_prediction_alias"
FINAL_REFIT_STAGE = "final_refit"
HELD_STAGE = "held_prediction"
ENSEMBLE_STAGE = "seed_ensemble_prediction"

MODEL_FIT_STAGES = frozenset((SOURCE_FIT_STAGE, CALIBRATION_FIT_STAGE, FINAL_REFIT_STAGE))

CONTEXT_FIELDS = (
    "context_id",
    "selected_model_id",
    "selection_state_sha256",
    "selection_mode",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
    "selection_units",
    "calibration_units",
    "historical_reference_available",
)
UNIT_FIELDS = ("unit_id", "fit_uid_sha256", "validation_uid_sha256")
CANDIDATE_FIELDS = ("candidate_id", "model_id", "hyperparameter_sha256")

REASON_CODES = frozenset(
    (
        "scientific_execution_not_authorized",
        "actions_must_be_mapping",
        "actions_keys_invalid",
        "action_hash_invalid",
        "model_spec_must_be_mapping",
        "model_spec_keys_invalid",
        "model_spec_hash_invalid",
        "source_family_mapping_hash_invalid",
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
        "selected_model_id_invalid",
        "selected_model_not_permitted",
        "selection_state_hash_invalid",
        "selection_mode_invalid",
        "historical_reference_invalid",
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
        "dependency_missing",
        "duplicate_job_id",
        "dependency_cycle",
        "dependency_context_mismatch",
    )
)

_HEX_DIGITS = frozenset("0123456789abcdef")


class NormalizationPlanError(ValueError):
    """Raised for malformed plan inputs or any attempted execution."""


def require_scientific_execution(plan):
    """Always deny scientific execution for this readiness module.

    The argument is deliberately ignored, so even a forged
    ``{"execution_authorized": True}`` payload cannot open the gate.
    """
    raise NormalizationPlanError("scientific_execution_not_authorized")


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
    raise NormalizationPlanError(code)


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


def _require_bool(value, code):
    if not isinstance(value, bool):
        _fail(code)


def _require_exact_keys(mapping, expected, code):
    if set(mapping.keys()) != set(expected):
        _fail(code)


def _family_seeds(model_id):
    if model_id in STOCHASTIC_FAMILIES:
        return STOCHASTIC_SEEDS
    return (DETERMINISTIC_SEED,)


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
    if set(model_spec_sha256.keys()) != set(FAMILY_MODELS):
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
    counts = {model_id: 0 for model_id in FAMILY_MODELS}
    seen = set()
    parsed = []
    for raw in candidates:
        _require_mapping(raw, "candidate_must_be_mapping")
        _require_exact_keys(raw, CANDIDATE_FIELDS, "candidate_keys_invalid")
        candidate_id = raw["candidate_id"]
        model_id = raw["model_id"]
        _require_identifier(candidate_id, "candidate_id_invalid")
        _require_identifier(model_id, "candidate_model_id_invalid")
        if model_id not in FAMILY_MODELS:
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
    for model_id, expected in FAMILY_CANDIDATE_COUNTS.items():
        if counts[model_id] != expected:
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
        _require_exact_keys(raw, UNIT_FIELDS, "unit_keys_invalid")
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
        _require_exact_keys(raw, CONTEXT_FIELDS, "context_keys_invalid")

        context_id = raw["context_id"]
        _require_identifier(context_id, "context_id_invalid")
        if context_id in seen:
            _fail("duplicate_context_id")
        seen.add(context_id)

        model_id = raw["selected_model_id"]
        _require_identifier(model_id, "selected_model_id_invalid")
        if model_id not in FAMILY_MODELS:
            _fail("selected_model_not_permitted")

        state_sha = raw["selection_state_sha256"]
        _require_sha256(state_sha, "selection_state_hash_invalid")

        mode = raw["selection_mode"]
        _require_identifier(mode, "selection_mode_invalid")
        if mode not in (MASTER_MODE, PSEUDO_MODE):
            _fail("selection_mode_invalid")

        _require_bool(raw["historical_reference_available"], "historical_reference_invalid")

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
            if (
                unit["fit_uid_sha256"] == outer_test
                or unit["validation_uid_sha256"] == outer_test
            ):
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
                "selected_model_id": model_id,
                "selection_state_sha256": state_sha,
                "selection_mode": mode,
                "outer_fit_uid_sha256": outer_fit,
                "outer_test_uid_sha256": outer_test,
                "selection_units": selection_units,
                "calibration_units": calibration_units,
                "historical_reference_available": raw["historical_reference_available"],
            }
        )
    parsed.sort(key=lambda context: context["context_id"])
    return parsed


def _new_job(fields, dependencies):
    job = dict(fields)
    job["dependencies"] = sorted(set(dependencies))
    job["job_id"] = "P08N2JOB-" + _hash(job)
    return job


def _build_context(
    policy,
    representation,
    array_sha,
    source_map_sha,
    context,
    model_id,
    family_candidates,
    spec,
):
    jobs = []
    context_id = context["context_id"]
    outer_fit_sha = context["outer_fit_uid_sha256"]
    test_sha = context["outer_test_uid_sha256"]
    state_sha = context["selection_state_sha256"]
    family_seeds = _family_seeds(model_id)

    def fields(
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
            "evidence_status": EVIDENCE_FUTURE,
            "source_family_mapping_sha256": source_map_sha,
            "saved_selection_state_sha256": state_sha,
        }

    source_prediction_ids = {}
    for unit in context["selection_units"]:
        for candidate in family_candidates:
            for seed in family_seeds:
                fit_fields = fields(
                    SOURCE_FIT_STAGE,
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
                prediction_fields["stage"] = SOURCE_PREDICTION_STAGE
                prediction_job = _new_job(prediction_fields, [fit_job["job_id"]])
                jobs.append(prediction_job)
                source_prediction_ids[
                    (unit["unit_id"], seed, candidate["candidate_id"])
                ] = prediction_job["job_id"]

    select_job = _new_job(
        fields(
            SELECT_STAGE,
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
    if context["selection_mode"] == PSEUDO_MODE:
        for unit in context["calibration_units"]:
            for seed in family_seeds:
                calibration_fields = fields(
                    CALIBRATION_FIT_STAGE,
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
                prediction_fields["stage"] = CALIBRATION_PREDICTION_STAGE
                prediction_job = _new_job(prediction_fields, [calibration_fit["job_id"]])
                jobs.append(prediction_job)
                scalar_dependencies.append(prediction_job["job_id"])
    else:
        for unit in context["calibration_units"]:
            for seed in family_seeds:
                dependency_ids = [select_job["job_id"]]
                dependency_ids.extend(
                    source_prediction_ids[
                        (unit["unit_id"], seed, candidate["candidate_id"])
                    ]
                    for candidate in family_candidates
                )
                alias_job = _new_job(
                    fields(
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
    for seed in family_seeds:
        refit_job = _new_job(
            fields(
                FINAL_REFIT_STAGE,
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
                HELD_STAGE,
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
            ENSEMBLE_STAGE,
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
    return jobs


def _validate_graph(jobs):
    lookup = {}
    for job in jobs:
        job_id = job["job_id"]
        if job_id in lookup:
            _fail("duplicate_job_id")
        lookup[job_id] = job
    for job in jobs:
        for dependency_id in job["dependencies"]:
            if dependency_id not in lookup:
                _fail("dependency_missing")
            dependency = lookup[dependency_id]
            if (
                dependency["context_id"] != job["context_id"]
                or dependency["policy_id"] != job["policy_id"]
                or dependency["representation_id"] != job["representation_id"]
            ):
                _fail("dependency_context_mismatch")

    visiting = set()
    visited = set()

    def visit(job_id):
        if job_id in visited:
            return
        if job_id in visiting:
            _fail("dependency_cycle")
        visiting.add(job_id)
        for dependency_id in lookup[job_id]["dependencies"]:
            visit(dependency_id)
        visiting.discard(job_id)
        visited.add(job_id)

    for job_id in lookup:
        visit(job_id)


def _sorted_stage_counts(jobs):
    counts = {}
    for job in jobs:
        counts[job["stage"]] = counts.get(job["stage"], 0) + 1
    return {stage: counts[stage] for stage in sorted(counts)}


def _policy_summary(policy_jobs):
    stage_counts = _sorted_stage_counts(policy_jobs)
    model_fit_slots = sum(
        count for stage, count in stage_counts.items() if stage in MODEL_FIT_STAGES
    )
    prediction_stage_counts = {
        stage: count
        for stage, count in stage_counts.items()
        if stage.endswith("_prediction")
    }
    return {
        "total_jobs": len(policy_jobs),
        "model_fit_slots": model_fit_slots,
        "scalar_calibrations": stage_counts.get(SCALAR_STAGE, 0),
        "calibration_prediction_aliases": stage_counts.get(CALIBRATION_ALIAS_STAGE, 0),
        "stage_counts": stage_counts,
        "prediction_stage_counts": prediction_stage_counts,
    }


def _aggregate(entries):
    stage_counts = {}
    total_jobs = 0
    model_fit_slots = 0
    scalar_calibrations = 0
    calibration_prediction_aliases = 0
    for entry in entries:
        total_jobs += entry["total_jobs"]
        model_fit_slots += entry["model_fit_slots"]
        scalar_calibrations += entry["scalar_calibrations"]
        calibration_prediction_aliases += entry["calibration_prediction_aliases"]
        for stage, count in entry["stage_counts"].items():
            stage_counts[stage] = stage_counts.get(stage, 0) + count
    ordered_stages = {stage: stage_counts[stage] for stage in sorted(stage_counts)}
    prediction_stage_counts = {
        stage: count for stage, count in ordered_stages.items() if stage.endswith("_prediction")
    }
    return {
        "total_jobs": total_jobs,
        "model_fit_slots": model_fit_slots,
        "scalar_calibrations": scalar_calibrations,
        "calibration_prediction_aliases": calibration_prediction_aliases,
        "stage_counts": ordered_stages,
        "prediction_stage_counts": prediction_stage_counts,
    }


def _build_summary(jobs, contexts):
    by_policy = {}
    for policy in POLICIES:
        by_policy[policy] = _policy_summary(
            [job for job in jobs if job["policy_id"] == policy]
        )
    context_rows = sorted(
        (
            {
                "context_id": context["context_id"],
                "selected_model_id": context["selected_model_id"],
                "selection_mode": context["selection_mode"],
                "selection_state_sha256": context["selection_state_sha256"],
                "historical_reference_available": context["historical_reference_available"],
            }
            for context in contexts
        ),
        key=lambda row: row["context_id"],
    )
    unavailable = sum(
        1 for row in context_rows if not row["historical_reference_available"]
    )
    return {
        "by_policy": by_policy,
        "totals": _aggregate([by_policy[policy] for policy in POLICIES]),
        "contexts": context_rows,
        "historical_reference_unavailable_contexts": unavailable,
        "authorized_fit_slots": 0,
        "execution_authorized": False,
    }


def build_normalization_plan(
    contexts,
    candidates,
    actions,
    model_spec_sha256,
    source_family_mapping_sha256,
):
    """Build the data-free approved N2 normalization/control slot DAG.

    All inputs are validated and copied; nothing is mutated.  The returned
    dictionary is JSON-safe and is always marked ``execution_authorized``
    ``False``.  No scientific execution is possible through this module.
    """
    parsed_actions = _parse_actions(actions)
    parsed_spec = _parse_model_spec(model_spec_sha256)
    _require_sha256(source_family_mapping_sha256, "source_family_mapping_hash_invalid")
    parsed_candidates = _parse_candidates(candidates)
    parsed_contexts = _parse_contexts(contexts)

    candidates_by_model = {model_id: [] for model_id in FAMILY_MODELS}
    for candidate in parsed_candidates:
        candidates_by_model[candidate["model_id"]].append(candidate)

    jobs = []
    for policy in POLICIES:
        representation = POLICY_REPRESENTATION[policy]
        array_sha = parsed_actions[representation]
        for context in parsed_contexts:
            model_id = context["selected_model_id"]
            jobs.extend(
                _build_context(
                    policy,
                    representation,
                    array_sha,
                    source_family_mapping_sha256,
                    context,
                    model_id,
                    candidates_by_model[model_id],
                    parsed_spec,
                )
            )

    _validate_graph(jobs)
    jobs.sort(key=lambda job: job["job_id"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "source_family_mapping_sha256": source_family_mapping_sha256,
        "jobs": jobs,
        "aliases": [],
        "summary": _build_summary(jobs, parsed_contexts),
    }
    plan = dict(payload)
    plan["plan_sha256"] = _hash(payload)
    return plan
