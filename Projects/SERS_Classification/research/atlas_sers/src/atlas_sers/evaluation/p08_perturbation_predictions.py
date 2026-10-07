"""P08 exact stress prediction / reuse / reporting graph (metadata only).

This module is an INTERNAL metadata adapter.  It consumes the exact snapshot
bundle produced by ``p08_perturbation_join.bind_stress_prediction_inputs`` and
derives a deterministic, execution-disabled description of the stress
prediction, reuse and reporting graph.

It performs **zero** scientific work: no arrays, models, scores, thresholds,
routes, draws, calibrations or fits are computed, guessed or inferred.  Every
dynamic outcome stays a deferred upstream dependency.  The full stress job
ledger is explicitly *not* complete here, and no statistical, scoring,
rendering, resource or numerical-validation ledger is produced.

The catalog binds the supplied bundle by canonical digest and always keeps
``execution_authorized`` ``False``.  ``require_scientific_execution`` always
denies execution.  Malformed metadata is sanitised to
``invalid_stress_prediction_metadata``; only ``ValueError``/``TypeError``/
``KeyError``/``UnicodeError``/``RecursionError``/``OverflowError`` are caught,
never ``BaseException`` or blanket ``Exception``.
"""

from __future__ import annotations

import copy
import json

from .p08_perturbation_inputs import (
    CLEAN_CASE_ID,
    NOT_APPLICABLE,
    iter_stress_input_jobs,
)
from .p08_perturbation_join import bind_stress_prediction_inputs
from .p08_perturbation_procedures import (
    MIN_POLICY,
    REUSE_HISTORICAL_CLASSICAL,
)
from .p08_perturbation_qc_procedures import QC_POLICY
from .p08_plan import (
    CLASSICAL_MODELS,
    D0_STRATEGY,
    POLICY_REPRESENTATION,
    SELECTED_STRATEGY,
)
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "build_stress_prediction_catalog",
    "iter_stress_prediction_records",
    "require_scientific_execution",
]


CATALOG_SCHEMA_VERSION = "nato-sers-p08-stress-prediction-catalog-v1"
BUNDLE_SCHEMA_VERSION = "nato-sers-p08-stress-prediction-binding-v1"
JOB_PREFIX = "P08STRESSPRED-"
ALIAS_PREFIX = "P08STRESSALIAS-"
FAMILY_POLICY = "PP-FAMILY-SRC"

INVALID_METADATA = "invalid_stress_prediction_metadata"
EXECUTION_DENIED = "scientific_execution_not_authorized"

_NA = NOT_APPLICABLE

STAGE_MODEL_RECONSTRUCTION = "model_reconstruction"
STAGE_RETAINED_MODEL_AUTHENTICATION = "retained_model_authentication"
STAGE_CALIBRATOR_AUTHENTICATION = "calibrator_authentication"
STAGE_QC_CLEAN_ROUTE_AUTHENTICATION = "qc_clean_route_authentication"
STAGE_QC_MIXED_INPUT_ASSEMBLY = "qc_mixed_input_assembly"
STAGE_RAW_PREDICTION = "raw_prediction"
STAGE_CLASSICAL_SEED_AVERAGE = "classical_seed_average"
STAGE_TEMPERATURE_APPLY = "temperature_apply"
STAGE_NEURAL_SEED_AVERAGE = "neural_seed_average"
STAGE_PREDICTION_UNITS = "prediction_units"
STAGE_CLEAN_PROBABILITY_PARITY = "clean_probability_parity"

STAGE_ACTION_TRANSFORM = "action_transform"
STAGE_ZERO_INPUT_PARITY = "zero_input_parity"
STAGE_CONTEXT_ACTION_ASSEMBLY = "context_action_assembly"

_INPUT_STAGES = frozenset(
    (STAGE_ACTION_TRANSFORM, STAGE_ZERO_INPUT_PARITY, STAGE_CONTEXT_ACTION_ASSEMBLY)
)

RES_MODEL_RECONSTRUCTION = "fixed_saved_selection_source_rows_seed_no_retuning"
RES_RETAINED_MODEL = "retained_upstream_estimator_no_new_fit"
RES_CALIBRATOR = "source_fitted_temperature_no_new_fit"
RES_QC_CLEAN_ROUTE = "fixed_native_clean_route_not_gate_reaction"
RES_QC_MIXED_INPUT = (
    "candidate_row_receipts_select_clean_route_action_or_invalid_action_MIN_input_"
    "require_selected_clean_parity_unselected_invalid_not_fatal_invalid_MIN_fatal"
)
RES_RAW_PREDICTION = "uncalibrated_same_model_case_scores"
RES_CLASSICAL_SEED_AVERAGE = "average_uncalibrated_seed_scores"
RES_TEMPERATURE_CLASSICAL = "seed_average_then_single_temperature_logclip1e_7"
RES_TEMPERATURE_NEURAL = "same_seed_temperature_before_ensemble"
RES_NEURAL_SEED_AVERAGE = "average_calibrated_seed_probabilities"
RES_PREDICTION_UNITS = (
    "M01_spectrum_and_M06_master_instrument_probability_combination_no_repeat_ensemble"
)
RES_CLEAN_PROBABILITY_PARITY = (
    "absolute_1e_7_relative_0_exact_classes_M01_M06_and_retained_seed_values"
)

_QC_ALIAS_SOURCE_ELIGIBLE = "fixed_clean_route"
_QC_ALIAS_SOURCE_FALLBACK = "disturbed_minimal_pipeline"

_ALIAS_MODE_UNIVERSAL = "universal"
_ALIAS_MODE_QC_ELIGIBLE = "qc_fixed_route"
_ALIAS_MODE_QC_FALLBACK = "qc_minimal_fallback"
_ALIAS_MODE_FAMILY = "family_minimal_fallback"
_ALIAS_MODES = (
    _ALIAS_MODE_UNIVERSAL,
    _ALIAS_MODE_QC_ELIGIBLE,
    _ALIAS_MODE_QC_FALLBACK,
    _ALIAS_MODE_FAMILY,
)

_UNIVERSAL_STRATEGIES = tuple(CLASSICAL_MODELS) + (D0_STRATEGY, SELECTED_STRATEGY)
_ACTIONS = tuple(sorted(set(POLICY_REPRESENTATION.values())))

_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "scientific_predictions_computed",
        "clean_probability_parity_accepted",
        "full_stress_job_ledger_complete",
        "bundle",
        "summary",
        "catalog_sha256",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this metadata-only adapter."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID_METADATA)


def _is_hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_hex64(value):
    if not _is_hex64(value):
        _fail()
    return value


def _is_classical(model_id):
    return model_id in CLASSICAL_MODELS


def _snapshot(value):
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _ref_by_seed(references, seed):
    found = [reference for reference in references if reference.get("seed") == seed]
    if len(found) != 1:
        _fail()
    return found[0]


def _case_index(bundle):
    manifest = bundle["input_catalog"]["case_manifest"]
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        _fail()
    index = {}
    for case in cases:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id != case_id.strip():
            _fail()
        if case_id in index:
            _fail()
        index[case_id] = case["family"]
    return index


def _context_test_uids(bundle):
    contexts = bundle["input_catalog"]["contexts"]
    if not isinstance(contexts, list) or not contexts:
        _fail()
    result = {}
    for context in contexts:
        context_id = context["context_id"]
        if context_id in result:
            _fail()
        test_uids = context["test_uids"]
        if not isinstance(test_uids, list) or not test_uids:
            _fail()
        result[context_id] = list(test_uids)
    return result


def _sorted_procedures(bundle):
    procedures = bundle["procedures"]
    if not isinstance(procedures, list) or not procedures:
        _fail()
    return sorted(
        procedures,
        key=lambda procedure: (
            procedure["policy_id"],
            procedure["context_id"],
            procedure["model_id"],
        ),
    )


def _universal_alias_groups(bundle):
    aliases = bundle["universal_procedures"]["strategy_aliases"]
    if not isinstance(aliases, list):
        _fail()
    groups = {}
    for alias in aliases:
        key = (alias["policy_id"], alias["context_id"])
        by_strategy = groups.setdefault(key, {})
        strategy = alias["strategy"]
        if strategy in by_strategy:
            _fail()
        by_strategy[strategy] = alias
    if not groups:
        _fail()
    for by_strategy in groups.values():
        if set(by_strategy) != {D0_STRATEGY, SELECTED_STRATEGY}:
            _fail()
    return groups


def _build_summary(bundle):
    cases = _case_index(bundle)
    case_count = len(cases)
    procedures = _sorted_procedures(bundle)
    procedure_count = len(procedures)

    seed_estimator_count = 0
    historical_fit_count = 0
    calibrator_count = 0
    classical_procedure_count = 0
    neural_procedure_count = 0
    neural_seed_estimator_count = 0
    qc_contexts = set()
    for procedure in procedures:
        seeds = procedure["seeds"]
        seed_estimator_count += len(seeds)
        if procedure["model_reuse_mode"] == REUSE_HISTORICAL_CLASSICAL:
            historical_fit_count += len(seeds)
        calibrator_count += len(procedure["calibration_references"])
        if _is_classical(procedure["model_id"]):
            classical_procedure_count += 1
        else:
            neural_procedure_count += 1
            neural_seed_estimator_count += len(seeds)
        if procedure["policy_id"] == QC_POLICY:
            qc_contexts.add(procedure["context_id"])
    eligible_qc_context_count = len(qc_contexts)

    stage_counts = {
        STAGE_MODEL_RECONSTRUCTION: historical_fit_count,
        STAGE_RETAINED_MODEL_AUTHENTICATION: seed_estimator_count - historical_fit_count,
        STAGE_CALIBRATOR_AUTHENTICATION: calibrator_count,
        STAGE_QC_CLEAN_ROUTE_AUTHENTICATION: eligible_qc_context_count,
        STAGE_QC_MIXED_INPUT_ASSEMBLY: eligible_qc_context_count * case_count,
        STAGE_RAW_PREDICTION: seed_estimator_count * case_count,
        STAGE_CLASSICAL_SEED_AVERAGE: classical_procedure_count * case_count,
        STAGE_TEMPERATURE_APPLY: (
            classical_procedure_count * case_count + neural_seed_estimator_count * case_count
        ),
        STAGE_NEURAL_SEED_AVERAGE: neural_procedure_count * case_count,
        STAGE_PREDICTION_UNITS: procedure_count * case_count,
        STAGE_CLEAN_PROBABILITY_PARITY: procedure_count,
    }

    universal_groups = _universal_alias_groups(bundle)
    qc_aliases = bundle["qc_procedures"]["strategy_aliases"]
    if not isinstance(qc_aliases, list):
        _fail()
    family_aliases = bundle["family_aliases"]
    if not isinstance(family_aliases, list):
        _fail()

    qc_eligible = 0
    qc_fallback = 0
    for alias in qc_aliases:
        if alias["mode"] == _QC_ALIAS_SOURCE_ELIGIBLE:
            qc_eligible += 1
        elif alias["mode"] == _QC_ALIAS_SOURCE_FALLBACK:
            qc_fallback += 1
        else:
            _fail()

    alias_counts = {
        _ALIAS_MODE_UNIVERSAL: len(universal_groups) * len(_UNIVERSAL_STRATEGIES) * case_count,
        _ALIAS_MODE_QC_ELIGIBLE: qc_eligible * case_count,
        _ALIAS_MODE_QC_FALLBACK: qc_fallback * case_count,
        _ALIAS_MODE_FAMILY: len(family_aliases) * case_count,
    }

    return {
        "case_count": case_count,
        "procedure_count": procedure_count,
        "seed_estimator_count": seed_estimator_count,
        "historical_reconstruction_fit_count": historical_fit_count,
        "new_calibration_fit_count": 0,
        "eligible_qc_context_count": eligible_qc_context_count,
        "stage_counts": stage_counts,
        "prediction_job_count": sum(stage_counts.values()),
        "reporting_alias_count": sum(alias_counts.values()),
        "alias_counts_by_mode": alias_counts,
    }


def _make_job(
    binding_sha256,
    stage,
    procedure_id,
    context_id,
    policy_id,
    model_id,
    case_id,
    seed,
    depends_on_job_ids,
    depends_on_input_job_ids,
    upstream_references,
    resolution,
):
    references = copy.deepcopy(upstream_references)
    body = {
        "record_type": "job",
        "binding_sha256": binding_sha256,
        "stage": stage,
        "procedure_id": procedure_id,
        "context_id": context_id,
        "policy_id": policy_id,
        "model_id": model_id,
        "case_id": case_id,
        "seed": seed,
        "depends_on_job_ids": sorted(set(depends_on_job_ids)),
        "depends_on_input_job_ids": sorted(set(depends_on_input_job_ids)),
        "upstream_references": references,
        "resolution": resolution,
    }
    record = dict(body)
    record["job_id"] = JOB_PREFIX + canonical_sha256(body)
    return record


def _make_alias(
    binding_sha256,
    policy_id,
    context_id,
    strategy,
    recipe_id,
    case_id,
    target_procedure_id,
    target_prediction_units_job_id,
    mode,
    upstream_alias_id,
):
    body = {
        "record_type": "alias",
        "binding_sha256": binding_sha256,
        "policy_id": policy_id,
        "context_id": context_id,
        "strategy": strategy,
        "recipe_id": recipe_id,
        "case_id": case_id,
        "target_procedure_id": target_procedure_id,
        "target_prediction_units_job_id": target_prediction_units_job_id,
        "mode": mode,
        "upstream_alias_id": upstream_alias_id,
    }
    record = dict(body)
    record["alias_id"] = ALIAS_PREFIX + canonical_sha256(body)
    return record


def _needed_input_keys(procedures, case_index, context_test_uids, qc_context_ids):
    case_ids = sorted(case_index)
    needed = set()
    for procedure in procedures:
        if procedure["policy_id"] == QC_POLICY:
            continue
        representation = procedure["representation_id"]
        context_id = procedure["context_id"]
        for case_id in case_ids:
            needed.add((STAGE_CONTEXT_ACTION_ASSEMBLY, context_id, _NA, case_id, representation))
    for context_id in qc_context_ids:
        test_uids = context_test_uids[context_id]
        for case_id in case_ids:
            transform_context = context_id if case_index[case_id] == "gaussian" else _NA
            for uid in test_uids:
                for action in _ACTIONS:
                    needed.add((STAGE_ACTION_TRANSFORM, transform_context, uid, case_id, action))
        for uid in test_uids:
            for action in _ACTIONS:
                needed.add((STAGE_ZERO_INPUT_PARITY, _NA, uid, CLEAN_CASE_ID, action))
    return needed


def _build_input_index(bundle, needed):
    index = {}
    for job in iter_stress_input_jobs(bundle["input_catalog"]):
        stage = job["stage"]
        if stage not in _INPUT_STAGES:
            continue
        key = (
            stage,
            job["context_id"],
            job["observation_uid"],
            job["case_id"],
            job["representation_id"],
        )
        if key in needed:
            if key in index:
                _fail()
            index[key] = job["job_id"]
    for key in needed:
        if key not in index:
            _fail()
    return index


def _input_lookup(index, key):
    found = index.get(key)
    if found is None:
        _fail()
    return found


def _iter_records(bundle):
    try:
        binding_sha256 = bundle["binding_sha256"]
        procedures = _sorted_procedures(bundle)
        case_index = _case_index(bundle)
        case_ids = sorted(case_index)
        context_test_uids = _context_test_uids(bundle)

        qc_context_ids = sorted(
            {
                procedure["context_id"]
                for procedure in procedures
                if procedure["policy_id"] == QC_POLICY
            }
        )

        needed = _needed_input_keys(procedures, case_index, context_test_uids, qc_context_ids)
        input_index = _build_input_index(bundle, needed)

        model_ready = {}
        calibrator = {}
        qc_context_ready = {}
        qc_record_by_context = {}

        for procedure in procedures:
            procedure_id = procedure["procedure_id"]
            policy_id = procedure["policy_id"]
            context_id = procedure["context_id"]
            model_id = procedure["model_id"]
            if policy_id == QC_POLICY:
                existing = qc_record_by_context.get(context_id)
                if existing is None or model_id < existing["model_id"]:
                    qc_record_by_context[context_id] = procedure
            historical = procedure["model_reuse_mode"] == REUSE_HISTORICAL_CLASSICAL
            for seed in procedure["seeds"]:
                reference = _ref_by_seed(procedure["refit_references"], seed)
                if historical:
                    stage = STAGE_MODEL_RECONSTRUCTION
                    resolution = RES_MODEL_RECONSTRUCTION
                else:
                    stage = STAGE_RETAINED_MODEL_AUTHENTICATION
                    resolution = RES_RETAINED_MODEL
                job = _make_job(
                    binding_sha256,
                    stage,
                    procedure_id,
                    context_id,
                    policy_id,
                    model_id,
                    _NA,
                    seed,
                    (),
                    (),
                    {"refit": reference},
                    resolution,
                )
                model_ready[(procedure_id, seed)] = job["job_id"]
                if policy_id == QC_POLICY:
                    qc_context_ready.setdefault(context_id, set()).add(job["job_id"])
                yield job

        for procedure in procedures:
            procedure_id = procedure["procedure_id"]
            for reference in procedure["calibration_references"]:
                job = _make_job(
                    binding_sha256,
                    STAGE_CALIBRATOR_AUTHENTICATION,
                    procedure_id,
                    procedure["context_id"],
                    procedure["policy_id"],
                    procedure["model_id"],
                    _NA,
                    reference["seed"],
                    (),
                    (),
                    {"calibration": reference},
                    RES_CALIBRATOR,
                )
                calibrator[(procedure_id, reference["seed"])] = job["job_id"]
                yield job

        qc_route_auth = {}
        for context_id in sorted(qc_context_ready):
            record = qc_record_by_context[context_id]
            upstream = {
                "source_gate_reference": record["source_gate_reference"],
                "source_threshold_reference": record["source_threshold_reference"],
                "source_route_reference": record["source_route_reference"],
                "clean_route_reference": record["clean_route_reference"],
            }
            job = _make_job(
                binding_sha256,
                STAGE_QC_CLEAN_ROUTE_AUTHENTICATION,
                _NA,
                context_id,
                QC_POLICY,
                _NA,
                _NA,
                _NA,
                tuple(qc_context_ready[context_id]),
                (),
                upstream,
                RES_QC_CLEAN_ROUTE,
            )
            qc_route_auth[context_id] = job["job_id"]
            yield job

        qc_assembly = {}
        for context_id in qc_context_ids:
            test_uids = context_test_uids[context_id]
            for case_id in case_ids:
                family = case_index[case_id]
                transform_context = context_id if family == "gaussian" else _NA
                input_dependencies = set()
                for uid in test_uids:
                    for action in _ACTIONS:
                        input_dependencies.add(
                            _input_lookup(
                                input_index,
                                (
                                    STAGE_ACTION_TRANSFORM,
                                    transform_context,
                                    uid,
                                    case_id,
                                    action,
                                ),
                            )
                        )
                        input_dependencies.add(
                            _input_lookup(
                                input_index,
                                (
                                    STAGE_ZERO_INPUT_PARITY,
                                    _NA,
                                    uid,
                                    CLEAN_CASE_ID,
                                    action,
                                ),
                            )
                        )
                job = _make_job(
                    binding_sha256,
                    STAGE_QC_MIXED_INPUT_ASSEMBLY,
                    _NA,
                    context_id,
                    QC_POLICY,
                    _NA,
                    case_id,
                    _NA,
                    (qc_route_auth[context_id],),
                    tuple(input_dependencies),
                    {},
                    RES_QC_MIXED_INPUT,
                )
                qc_assembly[(context_id, case_id)] = job["job_id"]
                yield job

        prediction_units = {}
        ordered_cases = [CLEAN_CASE_ID] + [
            case_id for case_id in case_ids if case_id != CLEAN_CASE_ID
        ]

        for procedure in procedures:
            procedure_id = procedure["procedure_id"]
            context_id = procedure["context_id"]
            policy_id = procedure["policy_id"]
            model_id = procedure["model_id"]
            classical = _is_classical(model_id)
            seeds = list(procedure["seeds"])
            parity_id = None
            for case_id in ordered_cases:
                is_clean = case_id == CLEAN_CASE_ID
                raw_ids = {}
                for seed in seeds:
                    dependencies = {model_ready[(procedure_id, seed)]}
                    if not is_clean:
                        dependencies.add(parity_id)
                    input_dependencies = set()
                    if policy_id == QC_POLICY:
                        dependencies.add(qc_assembly[(context_id, case_id)])
                    else:
                        input_dependencies.add(
                            _input_lookup(
                                input_index,
                                (
                                    STAGE_CONTEXT_ACTION_ASSEMBLY,
                                    context_id,
                                    _NA,
                                    case_id,
                                    procedure["representation_id"],
                                ),
                            )
                        )
                    job = _make_job(
                        binding_sha256,
                        STAGE_RAW_PREDICTION,
                        procedure_id,
                        context_id,
                        policy_id,
                        model_id,
                        case_id,
                        seed,
                        tuple(dependencies),
                        tuple(input_dependencies),
                        {},
                        RES_RAW_PREDICTION,
                    )
                    raw_ids[seed] = job["job_id"]
                    yield job

                if classical:
                    average_job = _make_job(
                        binding_sha256,
                        STAGE_CLASSICAL_SEED_AVERAGE,
                        procedure_id,
                        context_id,
                        policy_id,
                        model_id,
                        case_id,
                        _NA,
                        tuple(raw_ids.values()),
                        (),
                        {},
                        RES_CLASSICAL_SEED_AVERAGE,
                    )
                    yield average_job
                    temperature_job = _make_job(
                        binding_sha256,
                        STAGE_TEMPERATURE_APPLY,
                        procedure_id,
                        context_id,
                        policy_id,
                        model_id,
                        case_id,
                        _NA,
                        (average_job["job_id"], calibrator[(procedure_id, _NA)]),
                        (),
                        {},
                        RES_TEMPERATURE_CLASSICAL,
                    )
                    yield temperature_job
                    producer_id = temperature_job["job_id"]
                    parity_dependencies = set(raw_ids.values())
                else:
                    temperature_ids = {}
                    for seed in seeds:
                        temperature_job = _make_job(
                            binding_sha256,
                            STAGE_TEMPERATURE_APPLY,
                            procedure_id,
                            context_id,
                            policy_id,
                            model_id,
                            case_id,
                            seed,
                            (raw_ids[seed], calibrator[(procedure_id, seed)]),
                            (),
                            {},
                            RES_TEMPERATURE_NEURAL,
                        )
                        temperature_ids[seed] = temperature_job["job_id"]
                        yield temperature_job
                    average_job = _make_job(
                        binding_sha256,
                        STAGE_NEURAL_SEED_AVERAGE,
                        procedure_id,
                        context_id,
                        policy_id,
                        model_id,
                        case_id,
                        _NA,
                        tuple(temperature_ids.values()),
                        (),
                        {},
                        RES_NEURAL_SEED_AVERAGE,
                    )
                    yield average_job
                    producer_id = average_job["job_id"]
                    parity_dependencies = set(temperature_ids.values())

                units_job = _make_job(
                    binding_sha256,
                    STAGE_PREDICTION_UNITS,
                    procedure_id,
                    context_id,
                    policy_id,
                    model_id,
                    case_id,
                    _NA,
                    (producer_id,),
                    (),
                    {},
                    RES_PREDICTION_UNITS,
                )
                prediction_units[(procedure_id, case_id)] = units_job["job_id"]
                yield units_job

                if is_clean:
                    parity_dependencies.add(units_job["job_id"])
                    parity_job = _make_job(
                        binding_sha256,
                        STAGE_CLEAN_PROBABILITY_PARITY,
                        procedure_id,
                        context_id,
                        policy_id,
                        model_id,
                        CLEAN_CASE_ID,
                        _NA,
                        tuple(parity_dependencies),
                        (),
                        {
                            "clean_endpoint_reference": procedure["clean_endpoint_reference"],
                            "held_reference_jobs": list(procedure["held_reference_jobs"]),
                        },
                        RES_CLEAN_PROBABILITY_PARITY,
                    )
                    parity_id = parity_job["job_id"]
                    yield parity_job

        universal_groups = _universal_alias_groups(bundle)
        procedure_index = {
            (procedure["policy_id"], procedure["context_id"], procedure["model_id"]): procedure[
                "procedure_id"
            ]
            for procedure in procedures
        }
        for policy_id, context_id in sorted(universal_groups):
            by_strategy = universal_groups[(policy_id, context_id)]
            for strategy in _UNIVERSAL_STRATEGIES:
                if strategy in CLASSICAL_MODELS:
                    target_procedure_id = procedure_index.get((policy_id, context_id, strategy))
                    if target_procedure_id is None:
                        _fail()
                    recipe_id = strategy
                    upstream_alias_id = _NA
                else:
                    alias = by_strategy.get(strategy)
                    if alias is None:
                        _fail()
                    target_procedure_id = alias["target_procedure_id"]
                    recipe_id = alias["recipe_id"]
                    upstream_alias_id = alias["upstream_alias_id"]
                for case_id in case_ids:
                    units_id = prediction_units.get((target_procedure_id, case_id))
                    if units_id is None:
                        _fail()
                    yield _make_alias(
                        binding_sha256,
                        policy_id,
                        context_id,
                        strategy,
                        recipe_id,
                        case_id,
                        target_procedure_id,
                        units_id,
                        _ALIAS_MODE_UNIVERSAL,
                        upstream_alias_id,
                    )

        qc_aliases = bundle["qc_procedures"]["strategy_aliases"]
        for alias in sorted(qc_aliases, key=lambda entry: (entry["context_id"], entry["strategy"])):
            source_mode = alias["mode"]
            if source_mode == _QC_ALIAS_SOURCE_ELIGIBLE:
                mode = _ALIAS_MODE_QC_ELIGIBLE
            elif source_mode == _QC_ALIAS_SOURCE_FALLBACK:
                mode = _ALIAS_MODE_QC_FALLBACK
            else:
                _fail()
            target_procedure_id = alias["target_procedure_id"]
            for case_id in case_ids:
                units_id = prediction_units.get((target_procedure_id, case_id))
                if units_id is None:
                    _fail()
                yield _make_alias(
                    binding_sha256,
                    alias["policy_id"],
                    alias["context_id"],
                    alias["strategy"],
                    alias["recipe_id"],
                    case_id,
                    target_procedure_id,
                    units_id,
                    mode,
                    alias["upstream_alias_id"],
                )

        family_aliases = bundle["family_aliases"]
        minimal_index = {
            (procedure["context_id"], procedure["model_id"]): procedure["procedure_id"]
            for procedure in procedures
            if procedure["policy_id"] == MIN_POLICY
        }
        for alias in sorted(
            family_aliases, key=lambda entry: (entry["context_id"], entry["strategy"])
        ):
            target_procedure_id = minimal_index.get((alias["context_id"], alias["recipe_id"]))
            if target_procedure_id is None:
                _fail()
            for case_id in case_ids:
                units_id = prediction_units.get((target_procedure_id, case_id))
                if units_id is None:
                    _fail()
                yield _make_alias(
                    binding_sha256,
                    FAMILY_POLICY,
                    alias["context_id"],
                    alias["strategy"],
                    alias["recipe_id"],
                    case_id,
                    target_procedure_id,
                    units_id,
                    _ALIAS_MODE_FAMILY,
                    alias["alias_id"],
                )
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None


def build_stress_prediction_catalog(
    *, input_catalog, universal_procedures, qc_catalog, minimal_bridge
):
    """Build the metadata-only stress prediction / reuse / reporting catalog.

    ``input_catalog``, ``universal_procedures``, ``qc_catalog`` and
    ``minimal_bridge`` are read and copied from only; they are never mutated.
    The returned dictionary binds the supplied bundle by canonical digest and
    always keeps ``execution_authorized`` ``False``.
    """
    try:
        bundle = bind_stress_prediction_inputs(
            input_catalog=input_catalog,
            universal_procedures=universal_procedures,
            qc_catalog=qc_catalog,
            minimal_bridge=minimal_bridge,
        )
        summary = _build_summary(bundle)
        body = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "execution_authorized": False,
            "scientific_operations": 0,
            "artifact_provenance_independently_verified": False,
            "scientific_predictions_computed": False,
            "clean_probability_parity_accepted": False,
            "full_stress_job_ledger_complete": False,
            "bundle": bundle,
            "summary": summary,
        }
        catalog = dict(body)
        catalog["catalog_sha256"] = canonical_sha256(body)
        return catalog
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None


def iter_stress_prediction_records(catalog):
    """Rebuild and verify ``catalog``, then lazily stream exact jobs/aliases."""
    try:
        if not isinstance(catalog, dict) or set(catalog.keys()) != _CATALOG_FIELDS:
            _fail()
        if catalog["schema_version"] != CATALOG_SCHEMA_VERSION:
            _fail()
        if catalog["execution_authorized"] is not False:
            _fail()
        operations = catalog["scientific_operations"]
        if type(operations) is not int or operations != 0:
            _fail()
        for flag in (
            "artifact_provenance_independently_verified",
            "scientific_predictions_computed",
            "clean_probability_parity_accepted",
            "full_stress_job_ledger_complete",
        ):
            if catalog[flag] is not False:
                _fail()

        declared = _require_hex64(catalog["catalog_sha256"])
        content = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
        if canonical_sha256(content) != declared:
            _fail()

        bundle = catalog["bundle"]
        if not isinstance(bundle, dict):
            _fail()
        rebuilt = build_stress_prediction_catalog(
            input_catalog=bundle["input_catalog"],
            universal_procedures=bundle["universal_procedures"],
            qc_catalog=bundle["qc_catalog"],
            minimal_bridge=bundle["minimal_bridge"],
        )
        if canonical_sha256(rebuilt) != canonical_sha256(catalog):
            _fail()
        snapshot = _snapshot(rebuilt["bundle"])
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None
    return _iter_records(snapshot)
