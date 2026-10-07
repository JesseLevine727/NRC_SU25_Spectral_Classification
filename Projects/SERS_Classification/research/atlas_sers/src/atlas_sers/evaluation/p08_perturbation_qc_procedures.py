"""P08-T218 fixed-route QC stress procedure and MIN fallback adapter.

This module is an INTERNAL adapter over separately authenticated upstream
metadata.  It joins a sealed P08 QC catalog
(``nato-sers-p08-qc-catalog-v1``) with the universal stress procedure records
(``nato-sers-p08-universal-stress-procedures-v1``) into declarative,
execution-disabled QC stress procedure records and reporting aliases.

It performs **zero** scientific work: no fits, predictions, thresholds,
routing, gate choices, exclusions, sampling, randomness or numerical parity
checks.  Every dynamic outcome stays a deferred upstream dependency.  The
native clean route is deliberately *not* resolved here; this adapter only
preserves the supplied routing metadata as references.

Scope and trust boundaries:

* Inputs are read and copied only; they are never mutated.
* The QC catalog is validated through ``p08_qc_blocks.iter_slots``, which
  eagerly re-seals the complete graph (schema, hashes, references, acyclicity)
  before returning a *generator* that this module never consumes.
* Canonical hashes and DAG consistency do **not** prove physical exclusion,
  native QC validity, model artifacts or numerical parity.  A caller must
  independently authenticate every upstream artifact before scientific use.
* ``require_scientific_execution`` always denies execution.

Malformed metadata is sanitised to ``invalid_qc_stress_procedure_metadata``.
Only ``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/
``RecursionError``/``OverflowError`` are caught; ``BaseException`` and blanket
``Exception`` are never caught.
"""

from __future__ import annotations

from .p08_plan import (
    CLASSICAL_MODELS,
    D0_RECIPE,
    D0_STRATEGY,
    EVIDENCE_BY_POLICY,
    EVIDENCE_HISTORICAL,
    NEURAL_RECIPES,
    NOT_APPLICABLE,
    POLICIES,
    POLICY_REPRESENTATION,
    SEEDS,
    SELECTED_STRATEGY,
    SVM_MODEL,
    SVM_SEED,
    _parse_contexts,
)
from .p08_qc_blocks import FALLBACK_TARGET, canonical_sha256, iter_slots

__all__ = [
    "build_qc_procedure_records",
    "require_scientific_execution",
]


QC_STRESS_SCHEMA_VERSION = "nato-sers-p08-qc-stress-procedures-v1"
UNIVERSAL_SCHEMA_VERSION = "nato-sers-p08-universal-stress-procedures-v1"
QC_CATALOG_NAMESPACE = "nato-sers-p08-qc-catalog-v1"

QC_POLICY = "PP-QC-SRC"
PROCEDURE_ID_PREFIX = "P08QCSTRESSPROC-"
UNIVERSAL_PROCEDURE_ID_PREFIX = "P08STRESSPROC-"

INVALID_METADATA = "invalid_qc_stress_procedure_metadata"
EXECUTION_DENIED = "scientific_execution_not_authorized"

_GATE_AXIS = "source_selected_gate"
_RF_MODEL = "C-RANDOM-FOREST"
_MIN_ARRAY = "R_MIN_400_1800"

_MODE_ELIGIBLE = "fixed_clean_route"
_MODE_FALLBACK = "disturbed_minimal_pipeline"
_ELIGIBLE_REASON = "eligible"
_ELIGIBLE_STATUS = "unapproved_future_job"
_FALLBACK_STATUS = "requires_authenticated_minimal_endpoint"

_REUSE_MODE = "future_retained_mixed_route_estimator"
_CLEAN_ROUTE_MODE = "fixed_native_clean_route_not_gate_reaction"
_CALIBRATION_CLASSICAL = "seed_average_then_single_temperature"
_CALIBRATION_NEURAL = "per_seed_temperature_then_seed_average"

_RESOLUTION_REFIT_CLASSICAL = "source_frozen_selected_candidate_selection_only"
_RESOLUTION_REFIT_NEURAL = "same_seed_selected_epochs_selection_only"
_RESOLUTION_ROUTE = "source_frozen_thresholds_selection_only"
_RESOLUTION_THRESHOLD = "calibration_quantile_fit_role_only"
_RESOLUTION_SOURCE_PREDICTION = "same_gate_candidate_seed"
_RESOLUTION_CALIBRATION_MODEL = "selected_candidate_same_gate_same_seed"
_RESOLUTION_TEST_ROUTE = "source_frozen_thresholds_row_local_after_all_final_models_frozen"
_RESOLUTION_SCALAR_CLASSICAL = "seed_average_then_single_temperature_master_equal"
_RESOLUTION_SCALAR_NEURAL = "same_gate_same_seed_source_logits_master_equal"
_RESOLUTION_HELD_CLASSICAL = "raw_scores"
_RESOLUTION_HELD_NEURAL = "same_seed_calibrated_scores"
_RESOLUTION_ENSEMBLE_CLASSICAL = "seed_average_then_single_temperature_logclip1e_7"
_RESOLUTION_ENSEMBLE_NEURAL = "average_calibrated_seed_probabilities"

_SELECT_HYPERPARAMETERS_STAGE = "final_select_hyperparameters"
_SELECT_REFIT_EPOCHS_STAGE = "final_select_refit_epochs"
_SOURCE_PREDICTION_STAGE = "final_source_prediction"
_CALIBRATION_MODEL_PREDICTION_STAGE = "final_calibration_model_prediction"

_MODEL_FINAL_STAGES = (
    "final_refit",
    "final_scalar_calibration",
    "final_held_prediction",
    "final_seed_ensemble_prediction",
)

_QC_ALIAS_STRATEGIES = ("C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED")

_NA_CANDIDATE = {
    "candidate_id": NOT_APPLICABLE,
    "hyperparameter_sha256": NOT_APPLICABLE,
}
_CLASSICAL_CANDIDATE = {
    "candidate_id": "source_selected_candidate",
    "hyperparameter_sha256": NOT_APPLICABLE,
}
_NEURAL_CANDIDATE = {
    "candidate_id": "fixed_spec",
    "hyperparameter_sha256": NOT_APPLICABLE,
}

MIN_POLICY = next(
    policy for policy in POLICIES if EVIDENCE_BY_POLICY[policy] == EVIDENCE_HISTORICAL
)

_HEX_DIGITS = frozenset("0123456789abcdef")

_BINDING_KEYS = frozenset(
    (
        "protocol_namespace",
        "nested_registry_sha256",
        "universal_contexts",
        "candidates",
        "actions",
        "model_spec_sha256",
        "gate_library_sha256",
    )
)
_UNIVERSAL_KEYS = frozenset(
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
_UNIVERSAL_ALIAS_KEYS = frozenset(
    (
        "policy_id",
        "context_id",
        "strategy",
        "recipe_id",
        "target_procedure_id",
        "upstream_alias_id",
    )
)
_ALIAS_METADATA_KEYS = frozenset(
    (
        "binding_sha256",
        "outer_fit_uid_sha256",
        "outer_test_uid_sha256",
        "minimal_array_sha256",
        "model_spec_sha256",
    )
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


def _require_exact_keys(mapping, expected):
    if set(mapping.keys()) != set(expected):
        _fail()


def _is_classical(model_id):
    return model_id in CLASSICAL_MODELS


def _model_seeds(model_id):
    if model_id == SVM_MODEL:
        return [SVM_SEED]
    return list(SEEDS)


def _candidate_for(model_id):
    if _is_classical(model_id):
        return dict(_CLASSICAL_CANDIDATE)
    return dict(_NEURAL_CANDIDATE)


def _strategy_model(strategy, selected_recipe):
    if strategy == SELECTED_STRATEGY:
        return selected_recipe
    return strategy


def _check_seed(value):
    if isinstance(value, bool):
        _fail()
    if isinstance(value, int):
        return
    if value in (SVM_SEED, NOT_APPLICABLE):
        return
    _fail()


def _check_axes(block, candidate, seeds):
    axes = block["axes"]
    if axes["gate_id"] != [_GATE_AXIS]:
        _fail()
    if axes["candidate"] != [candidate]:
        _fail()
    seed_list = list(axes["seed"])
    if seed_list != list(seeds):
        _fail()
    for seed in seed_list:
        _check_seed(seed)


def _check_block_identity(block, context_id, role_id, fit, validation, test):
    if block["context_id"] != context_id:
        _fail()
    if block["role_id"] != role_id:
        _fail()
    if block["fit_uid_sha256"] != fit:
        _fail()
    if block["validation_uid_sha256"] != validation:
        _fail()
    if block["test_uid_sha256"] != test:
        _fail()


def _select_block(context_blocks, stage, model_id):
    found = [
        block
        for block in context_blocks
        if block["stage"] == stage and block["model_id"] == model_id
    ]
    if len(found) != 1:
        _fail()
    return found[0]


def _select_unique_stage_block(context_blocks, stage):
    found = [block for block in context_blocks if block["stage"] == stage]
    if len(found) != 1:
        _fail()
    return found[0]


def _select_model_stage_blocks(context_blocks, stage, model_id):
    return [
        block
        for block in context_blocks
        if block["stage"] == stage and block["model_id"] == model_id
    ]


def _reference(block, seed):
    candidate = dict(block["axes"]["candidate"][0])
    slot_id = "P08QCSLOT-" + canonical_sha256(
        {
            "block_id": block["block_id"],
            "gate_id": _GATE_AXIS,
            "candidate": candidate,
            "seed": seed,
        }
    )
    return {
        "block_id": block["block_id"],
        "slot_id": slot_id,
        "seed": seed,
        "resolution": block["resolution"],
    }


def _parse_actions_binding(actions):
    _require_mapping(actions)
    if set(actions.keys()) != set(POLICY_REPRESENTATION.values()):
        _fail()
    for value in actions.values():
        _require_lower_hex64(value)
    return dict(actions)


def _parse_model_spec_binding(model_spec):
    _require_mapping(model_spec)
    if set(model_spec.keys()) != set(CLASSICAL_MODELS) | set(NEURAL_RECIPES):
        _fail()
    for value in model_spec.values():
        _require_lower_hex64(value)
    return dict(model_spec)


def _parse_candidates_binding(candidates):
    candidates = _require_sequence(candidates)
    seen = set()
    for raw in candidates:
        _require_mapping(raw)
        _require_exact_keys(raw, ("candidate_id", "model_id", "hyperparameter_sha256"))
        _require_identifier(raw["candidate_id"])
        model_id = raw["model_id"]
        _require_identifier(model_id)
        if model_id not in CLASSICAL_MODELS:
            _fail()
        _require_lower_hex64(raw["hyperparameter_sha256"])
        if raw["candidate_id"] in seen:
            _fail()
        seen.add(raw["candidate_id"])
    return candidates


def _parse_qc_catalog(qc_catalog):
    iter_slots(qc_catalog)
    bindings = qc_catalog["bindings"]
    _require_mapping(bindings)
    _require_exact_keys(bindings, _BINDING_KEYS)
    if bindings["protocol_namespace"] != QC_CATALOG_NAMESPACE:
        _fail()
    _require_lower_hex64(bindings["nested_registry_sha256"])
    _require_lower_hex64(bindings["gate_library_sha256"])

    raw_compact = bindings["universal_contexts"]
    compact = _parse_contexts(raw_compact)
    if compact != raw_compact:
        _fail()
    if not compact:
        _fail()

    actions = _parse_actions_binding(bindings["actions"])
    model_specs = _parse_model_spec_binding(bindings["model_spec_sha256"])
    _parse_candidates_binding(bindings["candidates"])
    binding_sha = canonical_sha256(bindings)
    return bindings, compact, actions, model_specs, binding_sha


def _parse_universal_records(raw_records):
    raw_records = _require_sequence(raw_records)
    seen_procedure_ids = set()
    seen_keys = set()
    parsed = []
    for raw in raw_records:
        _require_mapping(raw)
        procedure_id = raw.get("procedure_id")
        _require_identifier(procedure_id)
        body = {key: value for key, value in raw.items() if key != "procedure_id"}
        if procedure_id != UNIVERSAL_PROCEDURE_ID_PREFIX + canonical_sha256(body):
            _fail()
        if procedure_id in seen_procedure_ids:
            _fail()
        seen_procedure_ids.add(procedure_id)

        policy = raw["policy_id"]
        if policy not in POLICIES:
            _fail()
        context_id = raw["context_id"]
        _require_identifier(context_id)
        model_id = raw["model_id"]
        if model_id not in set(CLASSICAL_MODELS) | set(NEURAL_RECIPES):
            _fail()
        if raw["representation_id"] != POLICY_REPRESENTATION[policy]:
            _fail()
        _require_lower_hex64(raw["array_sha256"])
        _require_lower_hex64(raw["model_spec_sha256"])
        _require_lower_hex64(raw["fit_uid_sha256"])
        _require_lower_hex64(raw["test_uid_sha256"])

        key = (policy, context_id, model_id)
        if key in seen_keys:
            _fail()
        seen_keys.add(key)
        parsed.append(raw)
    return parsed


def _parse_universal_aliases(raw_aliases):
    raw_aliases = _require_sequence(raw_aliases)
    parsed = []
    for raw in raw_aliases:
        _require_mapping(raw)
        _require_exact_keys(raw, _UNIVERSAL_ALIAS_KEYS)
        if raw["policy_id"] not in POLICIES:
            _fail()
        _require_identifier(raw["context_id"])
        _require_identifier(raw["strategy"])
        _require_identifier(raw["recipe_id"])
        _require_identifier(raw["target_procedure_id"])
        _require_identifier(raw["upstream_alias_id"])
        parsed.append(raw)
    return parsed


def _parse_universal(universal_procedures):
    _require_mapping(universal_procedures)
    _require_exact_keys(universal_procedures, _UNIVERSAL_KEYS)
    if universal_procedures["schema_version"] != UNIVERSAL_SCHEMA_VERSION:
        _fail()
    if universal_procedures["execution_authorized"] is not False:
        _fail()
    if universal_procedures["artifact_provenance_independently_verified"] is not False:
        _fail()
    if universal_procedures["exact_prediction_ledger_complete"] is not False:
        _fail()
    scientific_operations = universal_procedures["scientific_operations"]
    if (
        isinstance(scientific_operations, bool)
        or not isinstance(scientific_operations, int)
        or scientific_operations != 0
    ):
        _fail()
    _require_lower_hex64(universal_procedures["parent_plan_sha256"])
    _require_lower_hex64(universal_procedures["minimal_bridge_metadata_sha256"])
    declared_catalog = universal_procedures["catalog_sha256"]
    _require_lower_hex64(declared_catalog)
    body = {key: value for key, value in universal_procedures.items() if key != "catalog_sha256"}
    if canonical_sha256(body) != declared_catalog:
        _fail()

    records = _parse_universal_records(universal_procedures["records"])
    aliases = _parse_universal_aliases(universal_procedures["strategy_aliases"])
    return records, aliases


def _build_eligible_records(
    context_id,
    ctx,
    context_blocks,
    alias_by_strategy,
    selected_recipe,
    outer_fit,
    outer_test,
    actions,
    model_specs,
    binding_sha,
    qc_catalog_sha256,
    universal_catalog_sha256,
):
    gate = _select_unique_stage_block(context_blocks, "gate_selection")
    threshold = _select_unique_stage_block(context_blocks, "final_refit_quantile_fit")
    source_route = _select_unique_stage_block(context_blocks, "final_source_route")
    test_route = _select_unique_stage_block(context_blocks, "final_test_route")

    for block, fit, test in (
        (gate, outer_fit, NOT_APPLICABLE),
        (threshold, outer_fit, NOT_APPLICABLE),
        (source_route, outer_fit, NOT_APPLICABLE),
        (test_route, NOT_APPLICABLE, outer_test),
    ):
        _check_block_identity(block, context_id, context_id, fit, NOT_APPLICABLE, test)
        if block["model_id"] != NOT_APPLICABLE:
            _fail()
        _check_axes(block, _NA_CANDIDATE, [NOT_APPLICABLE])

    if list(threshold["depends_on_blocks"]) != [gate["block_id"]]:
        _fail()
    if threshold["resolution"] != _RESOLUTION_THRESHOLD:
        _fail()
    if set(source_route["depends_on_blocks"]) != {
        gate["block_id"],
        threshold["block_id"],
    }:
        _fail()
    if len(source_route["depends_on_blocks"]) != 2:
        _fail()
    if source_route["resolution"] != _RESOLUTION_ROUTE:
        _fail()
    if test_route["resolution"] != _RESOLUTION_TEST_ROUTE:
        _fail()

    qc_models = [SVM_MODEL, _RF_MODEL, D0_RECIPE]
    if selected_recipe != D0_RECIPE:
        qc_models.append(selected_recipe)

    model_blocks = {}
    refit_ids = []
    for model_id in qc_models:
        classical = _is_classical(model_id)
        selection_stage = _SELECT_HYPERPARAMETERS_STAGE if classical else _SELECT_REFIT_EPOCHS_STAGE
        refit = _select_block(context_blocks, "final_refit", model_id)
        scalar = _select_block(context_blocks, "final_scalar_calibration", model_id)
        held = _select_block(context_blocks, "final_held_prediction", model_id)
        ensemble = _select_block(context_blocks, "final_seed_ensemble_prediction", model_id)
        selection = _select_block(context_blocks, selection_stage, model_id)
        model_blocks[model_id] = {
            "refit": refit,
            "scalar": scalar,
            "held": held,
            "ensemble": ensemble,
            "selection": selection,
        }
        refit_ids.append(refit["block_id"])

    seen_selection = set()
    for block in context_blocks:
        if block["stage"] in (_SELECT_HYPERPARAMETERS_STAGE, _SELECT_REFIT_EPOCHS_STAGE):
            key = (block["model_id"], block["stage"])
            if key in seen_selection:
                _fail()
            seen_selection.add(key)
    expected_selection = {
        (
            model_id,
            _SELECT_HYPERPARAMETERS_STAGE
            if _is_classical(model_id)
            else _SELECT_REFIT_EPOCHS_STAGE,
        )
        for model_id in qc_models
    }
    if seen_selection != expected_selection:
        _fail()

    seen_final = set()
    for block in context_blocks:
        if block["stage"] in _MODEL_FINAL_STAGES:
            key = (block["model_id"], block["stage"])
            if key in seen_final:
                _fail()
            seen_final.add(key)
    expected_final = {(model_id, stage) for model_id in qc_models for stage in _MODEL_FINAL_STAGES}
    if seen_final != expected_final:
        _fail()

    expected_route_dependencies = {
        gate["block_id"],
        threshold["block_id"],
    } | set(refit_ids)
    if set(test_route["depends_on_blocks"]) != expected_route_dependencies:
        _fail()
    if len(test_route["depends_on_blocks"]) != len(expected_route_dependencies):
        _fail()

    ensemble_by_model = {
        model_id: model_blocks[model_id]["ensemble"]["block_id"] for model_id in qc_models
    }
    for strategy in _QC_ALIAS_STRATEGIES:
        model_id = _strategy_model(strategy, selected_recipe)
        if alias_by_strategy[strategy]["target_block_id"] != ensemble_by_model[model_id]:
            _fail()

    records = []
    for model_id in qc_models:
        classical = _is_classical(model_id)
        candidate = _candidate_for(model_id)
        model_seeds = _model_seeds(model_id)
        scalar_seeds = [NOT_APPLICABLE] if classical else list(SEEDS)
        blocks = model_blocks[model_id]
        refit = blocks["refit"]
        scalar = blocks["scalar"]
        held = blocks["held"]
        ensemble = blocks["ensemble"]
        selection = blocks["selection"]

        _check_block_identity(
            selection,
            context_id,
            context_id,
            outer_fit,
            NOT_APPLICABLE,
            NOT_APPLICABLE,
        )
        _check_axes(
            selection,
            candidate,
            [NOT_APPLICABLE] if classical else list(SEEDS),
        )

        _check_block_identity(
            refit,
            context_id,
            context_id,
            outer_fit,
            NOT_APPLICABLE,
            NOT_APPLICABLE,
        )
        _check_axes(refit, candidate, model_seeds)
        if set(refit["depends_on_blocks"]) != {
            source_route["block_id"],
            selection["block_id"],
        }:
            _fail()
        if len(refit["depends_on_blocks"]) != 2:
            _fail()
        expected_refit_resolution = (
            _RESOLUTION_REFIT_CLASSICAL if classical else _RESOLUTION_REFIT_NEURAL
        )
        if refit["resolution"] != expected_refit_resolution:
            _fail()

        _check_block_identity(
            scalar,
            context_id,
            context_id,
            outer_fit,
            NOT_APPLICABLE,
            NOT_APPLICABLE,
        )
        _check_axes(scalar, candidate, scalar_seeds)
        expected_scalar_resolution = (
            _RESOLUTION_SCALAR_CLASSICAL if classical else _RESOLUTION_SCALAR_NEURAL
        )
        if scalar["resolution"] != expected_scalar_resolution:
            _fail()

        prediction_stage = (
            _CALIBRATION_MODEL_PREDICTION_STAGE if classical else _SOURCE_PREDICTION_STAGE
        )
        prediction_units = ctx["calibration_units"] if classical else ctx["selection_units"]
        expected_prediction_resolution = (
            _RESOLUTION_CALIBRATION_MODEL if classical else _RESOLUTION_SOURCE_PREDICTION
        )
        units_by_id = {unit["unit_id"]: unit for unit in prediction_units}
        if len(units_by_id) != len(prediction_units):
            _fail()
        prediction_blocks = _select_model_stage_blocks(context_blocks, prediction_stage, model_id)
        if len(prediction_blocks) != len(units_by_id):
            _fail()
        prediction_ids = set()
        seen_prediction_roles = set()
        for prediction_block in prediction_blocks:
            role_id = prediction_block["role_id"]
            if role_id in seen_prediction_roles:
                _fail()
            seen_prediction_roles.add(role_id)
            unit = units_by_id.get(role_id)
            if unit is None:
                _fail()
            _check_block_identity(
                prediction_block,
                context_id,
                role_id,
                unit["fit_uid_sha256"],
                unit["validation_uid_sha256"],
                NOT_APPLICABLE,
            )
            _check_axes(prediction_block, candidate, model_seeds)
            if prediction_block["resolution"] != expected_prediction_resolution:
                _fail()
            prediction_ids.add(prediction_block["block_id"])
        if seen_prediction_roles != set(units_by_id):
            _fail()
        scalar_dependencies = {selection["block_id"]} | prediction_ids
        if set(scalar["depends_on_blocks"]) != scalar_dependencies:
            _fail()
        if len(scalar["depends_on_blocks"]) != len(scalar_dependencies):
            _fail()

        _check_block_identity(
            held,
            context_id,
            context_id,
            outer_fit,
            NOT_APPLICABLE,
            outer_test,
        )
        _check_axes(held, candidate, model_seeds)
        held_dependencies = {refit["block_id"], test_route["block_id"]}
        if not classical:
            held_dependencies.add(scalar["block_id"])
        if set(held["depends_on_blocks"]) != held_dependencies:
            _fail()
        if len(held["depends_on_blocks"]) != len(held_dependencies):
            _fail()
        expected_held_resolution = (
            _RESOLUTION_HELD_CLASSICAL if classical else _RESOLUTION_HELD_NEURAL
        )
        if held["resolution"] != expected_held_resolution:
            _fail()

        _check_block_identity(
            ensemble,
            context_id,
            context_id,
            outer_fit,
            NOT_APPLICABLE,
            outer_test,
        )
        _check_axes(ensemble, candidate, [NOT_APPLICABLE])
        ensemble_dependencies = {held["block_id"]}
        if classical:
            ensemble_dependencies.add(scalar["block_id"])
        if set(ensemble["depends_on_blocks"]) != ensemble_dependencies:
            _fail()
        if len(ensemble["depends_on_blocks"]) != len(ensemble_dependencies):
            _fail()
        expected_ensemble_resolution = (
            _RESOLUTION_ENSEMBLE_CLASSICAL if classical else _RESOLUTION_ENSEMBLE_NEURAL
        )
        if ensemble["resolution"] != expected_ensemble_resolution:
            _fail()

        record = {
            "context_id": context_id,
            "policy_id": QC_POLICY,
            "model_id": model_id,
            "model_spec_sha256": model_specs[model_id],
            "action_array_sha256": dict(actions),
            "fit_uid_sha256": outer_fit,
            "test_uid_sha256": outer_test,
            "seeds": list(model_seeds),
            "parent_qc_catalog_sha256": qc_catalog_sha256,
            "parent_universal_catalog_sha256": universal_catalog_sha256,
            "binding_sha256": binding_sha,
            "model_reuse_mode": _REUSE_MODE,
            "calibration_order": (_CALIBRATION_CLASSICAL if classical else _CALIBRATION_NEURAL),
            "clean_route_mode": _CLEAN_ROUTE_MODE,
            "refit_references": [_reference(refit, seed) for seed in model_seeds],
            "calibration_references": [_reference(scalar, seed) for seed in scalar_seeds],
            "held_reference_jobs": [_reference(held, seed) for seed in model_seeds],
            "clean_endpoint_reference": _reference(ensemble, NOT_APPLICABLE),
            "source_gate_reference": _reference(gate, NOT_APPLICABLE),
            "source_threshold_reference": _reference(threshold, NOT_APPLICABLE),
            "source_route_reference": _reference(source_route, NOT_APPLICABLE),
            "clean_route_reference": _reference(test_route, NOT_APPLICABLE),
        }
        record["procedure_id"] = PROCEDURE_ID_PREFIX + canonical_sha256(record)
        records.append(record)

    return records


def _build(qc_catalog, universal_procedures):
    bindings, compact, actions, model_specs, binding_sha = _parse_qc_catalog(qc_catalog)
    universal_records, universal_aliases = _parse_universal(universal_procedures)

    compact_by_id = {ctx["context_id"]: ctx for ctx in compact}
    if len(compact_by_id) != len(compact):
        _fail()

    policy_context_models = {}
    for record in universal_records:
        policy_context_models.setdefault((record["policy_id"], record["context_id"]), {})[
            record["model_id"]
        ] = record

    expected_groups = {(policy, context_id) for policy in POLICIES for context_id in compact_by_id}
    if set(policy_context_models) != expected_groups:
        _fail()

    min_contexts = {
        record["context_id"] for record in universal_records if record["policy_id"] == MIN_POLICY
    }
    if min_contexts != set(compact_by_id):
        _fail()

    alias_index = {}
    seen_upstream_alias_ids = set()
    for alias in universal_aliases:
        key = (alias["policy_id"], alias["context_id"], alias["strategy"])
        if key in alias_index:
            _fail()
        alias_index[key] = alias
        upstream_alias_id = alias["upstream_alias_id"]
        if upstream_alias_id in seen_upstream_alias_ids:
            _fail()
        seen_upstream_alias_ids.add(upstream_alias_id)
    expected_alias_keys = {
        (policy, context_id, strategy)
        for policy in POLICIES
        for context_id in compact_by_id
        for strategy in (D0_STRATEGY, SELECTED_STRATEGY)
    }
    if set(alias_index) != expected_alias_keys:
        _fail()

    required_models_by_context = {}
    for context_id, ctx in compact_by_id.items():
        required = set(CLASSICAL_MODELS) | {D0_RECIPE}
        if ctx["selected_recipe_id"] != D0_RECIPE:
            required.add(ctx["selected_recipe_id"])
        required_models_by_context[context_id] = required

    for policy in POLICIES:
        expected_array = actions[POLICY_REPRESENTATION[policy]]
        for context_id, ctx in compact_by_id.items():
            models = policy_context_models.get((policy, context_id))
            if not models:
                _fail()
            if set(models) != required_models_by_context[context_id]:
                _fail()
            for model_id, record in models.items():
                if record["fit_uid_sha256"] != ctx["outer_fit_uid_sha256"]:
                    _fail()
                if record["test_uid_sha256"] != ctx["outer_test_uid_sha256"]:
                    _fail()
                if record["model_spec_sha256"] != model_specs[model_id]:
                    _fail()
                if record["array_sha256"] != expected_array:
                    _fail()

            d0_alias = alias_index.get((policy, context_id, D0_STRATEGY))
            selected_alias = alias_index.get((policy, context_id, SELECTED_STRATEGY))
            if d0_alias is None or selected_alias is None:
                _fail()
            if d0_alias["recipe_id"] != D0_RECIPE:
                _fail()
            if selected_alias["recipe_id"] != ctx["selected_recipe_id"]:
                _fail()
            if d0_alias["target_procedure_id"] != models[D0_RECIPE]["procedure_id"]:
                _fail()
            if (
                selected_alias["target_procedure_id"]
                != models[ctx["selected_recipe_id"]]["procedure_id"]
            ):
                _fail()

    known_contexts = set(compact_by_id)
    qc_blocks = qc_catalog["blocks"]
    qc_aliases = qc_catalog["aliases"]

    for block in qc_blocks:
        if block["context_id"] not in known_contexts:
            _fail()
    for alias in qc_aliases:
        if alias["context_id"] not in known_contexts:
            _fail()

    blocks_by_context = {}
    for block in qc_blocks:
        blocks_by_context.setdefault(block["context_id"], []).append(block)

    aliases_by_context = {}
    for alias in qc_aliases:
        aliases_by_context.setdefault(alias["context_id"], []).append(alias)

    min_record_by_context_model = {}
    for record in universal_records:
        if record["policy_id"] == MIN_POLICY:
            min_record_by_context_model[(record["context_id"], record["model_id"])] = record

    records_out = []
    aliases_out = []
    eligible_context_count = 0
    fallback_context_count = 0
    fallback_procedure_ids = set()

    for context_id in sorted(compact_by_id):
        ctx = compact_by_id[context_id]
        selected_recipe = ctx["selected_recipe_id"]
        outer_fit = ctx["outer_fit_uid_sha256"]
        outer_test = ctx["outer_test_uid_sha256"]

        context_aliases = aliases_by_context.get(context_id)
        if context_aliases is None or len(context_aliases) != 4:
            _fail()
        alias_by_strategy = {}
        for alias in context_aliases:
            strategy = alias["strategy"]
            if strategy in alias_by_strategy:
                _fail()
            alias_by_strategy[strategy] = alias
        if set(alias_by_strategy) != set(_QC_ALIAS_STRATEGIES):
            _fail()

        for strategy in _QC_ALIAS_STRATEGIES:
            alias = alias_by_strategy[strategy]
            expected_recipe = _strategy_model(strategy, selected_recipe)
            if alias["recipe_id"] != expected_recipe:
                _fail()
            metadata = alias["metadata"]
            if not isinstance(metadata, dict):
                _fail()
            _require_exact_keys(metadata, _ALIAS_METADATA_KEYS)
            if metadata["binding_sha256"] != binding_sha:
                _fail()
            if metadata["outer_fit_uid_sha256"] != outer_fit:
                _fail()
            if metadata["outer_test_uid_sha256"] != outer_test:
                _fail()
            if metadata["minimal_array_sha256"] != actions[_MIN_ARRAY]:
                _fail()
            if metadata["model_spec_sha256"] != model_specs[expected_recipe]:
                _fail()

        targets = [
            alias_by_strategy[strategy]["target_block_id"] for strategy in _QC_ALIAS_STRATEGIES
        ]
        all_fallback = all(target == FALLBACK_TARGET for target in targets)
        any_fallback = any(target == FALLBACK_TARGET for target in targets)
        if all_fallback and any_fallback:
            is_fallback = True
        elif (not all_fallback) and (not any_fallback):
            is_fallback = False
        else:
            _fail()

        statuses = {
            alias_by_strategy[strategy]["evidence_status"] for strategy in _QC_ALIAS_STRATEGIES
        }
        reasons = {alias_by_strategy[strategy]["reason_code"] for strategy in _QC_ALIAS_STRATEGIES}
        context_blocks = blocks_by_context.get(context_id, [])

        if is_fallback:
            if context_blocks:
                _fail()
            if statuses != {_FALLBACK_STATUS}:
                _fail()
            if len(reasons) != 1:
                _fail()
            reason = next(iter(reasons))
            if not isinstance(reason, str) or not reason or reason == _ELIGIBLE_REASON:
                _fail()
            fallback_context_count += 1
            for strategy in _QC_ALIAS_STRATEGIES:
                model_id = _strategy_model(strategy, selected_recipe)
                min_record = min_record_by_context_model.get((context_id, model_id))
                if min_record is None:
                    _fail()
                fallback_procedure_ids.add(min_record["procedure_id"])
                aliases_out.append(
                    {
                        "context_id": context_id,
                        "policy_id": QC_POLICY,
                        "strategy": strategy,
                        "recipe_id": _strategy_model(strategy, selected_recipe),
                        "mode": _MODE_FALLBACK,
                        "target_policy_id": MIN_POLICY,
                        "target_procedure_id": min_record["procedure_id"],
                        "upstream_alias_id": alias_by_strategy[strategy]["alias_id"],
                        "reason_code": alias_by_strategy[strategy]["reason_code"],
                        "same_case_target_required": True,
                    }
                )
            continue

        if statuses != {_ELIGIBLE_STATUS}:
            _fail()
        if reasons != {_ELIGIBLE_REASON}:
            _fail()

        eligible_context_count += 1
        context_records = _build_eligible_records(
            context_id,
            ctx,
            context_blocks,
            alias_by_strategy,
            selected_recipe,
            outer_fit,
            outer_test,
            actions,
            model_specs,
            binding_sha,
            qc_catalog["catalog_sha256"],
            universal_procedures["catalog_sha256"],
        )
        records_out.extend(context_records)
        records_by_model = {record["model_id"]: record for record in context_records}
        for strategy in _QC_ALIAS_STRATEGIES:
            model_id = _strategy_model(strategy, selected_recipe)
            aliases_out.append(
                {
                    "context_id": context_id,
                    "policy_id": QC_POLICY,
                    "strategy": strategy,
                    "recipe_id": _strategy_model(strategy, selected_recipe),
                    "mode": _MODE_ELIGIBLE,
                    "target_policy_id": QC_POLICY,
                    "target_procedure_id": records_by_model[model_id]["procedure_id"],
                    "upstream_alias_id": alias_by_strategy[strategy]["alias_id"],
                    "reason_code": alias_by_strategy[strategy]["reason_code"],
                    "same_case_target_required": True,
                }
            )

    records_out.sort(key=lambda record: (record["context_id"], record["model_id"]))
    aliases_out.sort(key=lambda alias: (alias["context_id"], alias["strategy"]))

    eligible_reporting_alias_count = sum(
        1 for alias in aliases_out if alias["mode"] == _MODE_ELIGIBLE
    )
    fallback_reporting_alias_count = sum(
        1 for alias in aliases_out if alias["mode"] == _MODE_FALLBACK
    )

    summary = {
        "context_count": len(compact_by_id),
        "eligible_context_count": eligible_context_count,
        "fallback_context_count": fallback_context_count,
        "procedure_count": len(records_out),
        "seed_estimator_count": sum(len(record["seeds"]) for record in records_out),
        "calibration_reference_count": sum(
            len(record["calibration_references"]) for record in records_out
        ),
        "reporting_alias_count": len(aliases_out),
        "eligible_reporting_alias_count": eligible_reporting_alias_count,
        "fallback_reporting_alias_count": fallback_reporting_alias_count,
        "unique_fallback_minimal_procedure_count": len(fallback_procedure_ids),
    }

    payload = {
        "schema_version": QC_STRESS_SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "exact_prediction_ledger_complete": False,
        "clean_routes_resolved": False,
        "parent_qc_catalog_sha256": qc_catalog["catalog_sha256"],
        "parent_universal_catalog_sha256": universal_procedures["catalog_sha256"],
        "records": records_out,
        "strategy_aliases": aliases_out,
        "summary": summary,
    }
    payload["catalog_sha256"] = canonical_sha256(payload)
    return payload


def build_qc_procedure_records(*, qc_catalog, universal_procedures):
    """Build metadata-only QC stress procedure records and reporting aliases.

    Both inputs are read and copied from only; they are never mutated.  The
    returned dictionary binds the supplied metadata and always keeps
    ``execution_authorized`` and ``clean_routes_resolved`` ``False``.
    """
    try:
        return _build(qc_catalog, universal_procedures)
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None
