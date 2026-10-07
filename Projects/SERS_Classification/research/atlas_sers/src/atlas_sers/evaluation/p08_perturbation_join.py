"""P08-T229 metadata-only joint stress-prediction binding.

This module is an INTERNAL, metadata-only join of caller-authenticated
upstream metadata:

* a sealed P08 stress input catalog (``p08_perturbation_inputs``);
* universal robustness procedure records
  (``nato-sers-p08-universal-stress-procedures-v1``);
* a sealed P08 QC catalog joined into QC stress procedure records
  (``nato-sers-p08-qc-stress-procedures-v1``);
* the caller-authenticated minimum-operation evidence bridge
  (``nato-sers-p08-minimum-operation-evidence-v1``).

It performs **zero** scientific work: no fits, predictions, thresholds,
routing, gate choices, exclusions, sampling, randomness, seed generation or
numerical parity checks.  It only revalidates and re-binds already supplied
metadata and preserves source-resolved values unchanged.

This is explicitly **not** artifact provenance validation and **not** a
complete physical leakage or disjointness proof.  Canonical hashes and DAG
consistency only bind the metadata that was actually supplied.  Callers must
independently authenticate every upstream artifact and the complete original
graph before any scientific use.

Scope and trust boundaries:

* Inputs are read and deep-copied only; they are never mutated.
* The stress input catalog is validated through
  ``p08_perturbation_inputs.iter_stress_input_jobs``; its lazy job generator
  is intentionally never consumed.
* The QC/universal join is delegated to
  ``p08_perturbation_qc_procedures.build_qc_procedure_records`` so the full QC
  graph checks and universal model/alias coverage are reused.
* ``require_scientific_execution`` always denies execution.

Malformed metadata is sanitised to ``invalid_stress_prediction_binding``.
Only ``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/
``RecursionError``/``OverflowError`` are caught; ``BaseException`` and a bare
``Exception`` are never caught.
"""

from __future__ import annotations

import copy

from .p08_perturbation_inputs import iter_stress_input_jobs
from .p08_perturbation_qc_procedures import build_qc_procedure_records
from .p08_plan import (
    CLASSICAL_MODELS,
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
)
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "bind_stress_prediction_inputs",
    "require_scientific_execution",
]


SCHEMA_VERSION = "nato-sers-p08-stress-prediction-binding-v1"
BINDING_NAMESPACE = "nato-sers-p08-stress-prediction-v1"
BRIDGE_SCHEMA_VERSION = "nato-sers-p08-minimum-operation-evidence-v1"

FAMILY_POLICY = "PP-FAMILY-SRC"
QC_POLICY = "PP-QC-SRC"

INVALID = "invalid_stress_prediction_binding"
EXECUTION_DENIED = "scientific_execution_not_authorized"

U_PREFIX = "P08STRESSPROC-"
QC_PREFIX = "P08QCSTRESSPROC-"

_CALIBRATION_CLASSICAL = "seed_average_then_single_temperature"
_CALIBRATION_NEURAL = "per_seed_temperature_then_seed_average"
_REUSE_HISTORICAL_CLASSICAL = "historical_classical_reconstruction"
_REUSE_HISTORICAL_NEURAL = "historical_neural_checkpoint"
_REUSE_FUTURE = "future_retained_estimator"
_QC_REUSE_MODE = "future_retained_mixed_route_estimator"
_QC_CLEAN_ROUTE_MODE = "fixed_native_clean_route_not_gate_reaction"

_FAMILY_STRATEGIES = (
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    D0_STRATEGY,
    SELECTED_STRATEGY,
)

_MIN_POLICY = next(
    policy for policy in POLICIES if EVIDENCE_BY_POLICY[policy] == EVIDENCE_HISTORICAL
)

_MODEL_IDS = frozenset(CLASSICAL_MODELS) | frozenset(NEURAL_RECIPES)

_U_REF_KEYS = frozenset(("job_id", "seed", "historical_binding_sha256", "resolved_source_values"))
_QC_REF_KEYS = frozenset(("block_id", "slot_id", "seed", "resolution"))

_QC_SHARED_REF_KEYS = (
    "source_gate_reference",
    "source_threshold_reference",
    "source_route_reference",
    "clean_route_reference",
)

_U_RECORD_REQUIRED = (
    "context_id",
    "policy_id",
    "model_id",
    "representation_id",
    "array_sha256",
    "model_spec_sha256",
    "fit_uid_sha256",
    "test_uid_sha256",
    "seeds",
    "model_reuse_mode",
    "calibration_order",
    "refit_references",
    "calibration_references",
    "held_reference_jobs",
    "clean_endpoint_reference",
    "procedure_id",
)

_QC_RECORD_REQUIRED = (
    "context_id",
    "policy_id",
    "model_id",
    "model_spec_sha256",
    "action_array_sha256",
    "fit_uid_sha256",
    "test_uid_sha256",
    "seeds",
    "binding_sha256",
    "model_reuse_mode",
    "calibration_order",
    "clean_route_mode",
    "refit_references",
    "calibration_references",
    "held_reference_jobs",
    "clean_endpoint_reference",
    "source_gate_reference",
    "source_threshold_reference",
    "source_route_reference",
    "clean_route_reference",
    "procedure_id",
)

_HEX_DIGITS = frozenset("0123456789abcdef")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID)


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
    return value


def _require_lower_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        _fail()
    for character in value:
        if character not in _HEX_DIGITS:
            _fail()
    return value


def _check_ref_seed(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        _fail()
    if type(actual) is not type(expected) or actual != expected:
        _fail()


def _check_seeds(seeds, expected):
    seeds = _require_sequence(seeds)
    if len(seeds) != len(expected):
        _fail()
    for actual, wanted in zip(seeds, expected, strict=True):
        _check_ref_seed(actual, wanted)


def _model_seeds(model_id):
    if model_id == SVM_MODEL:
        return [SVM_SEED]
    return list(SEEDS)


def _check_u_ref(ref, expected_seed, policy):
    _require_mapping(ref)
    if set(ref.keys()) != _U_REF_KEYS:
        _fail()
    _require_identifier(ref["job_id"])
    _check_ref_seed(ref["seed"], expected_seed)
    historical = ref["historical_binding_sha256"]
    resolved = ref["resolved_source_values"]
    if policy == _MIN_POLICY:
        _require_lower_hex64(historical)
        _require_mapping(resolved)
    else:
        if historical is not None or resolved is not None:
            _fail()


def _check_u_refs(refs, expected_seeds, policy):
    refs = _require_sequence(refs)
    if len(refs) != len(expected_seeds):
        _fail()
    for ref, seed in zip(refs, expected_seeds, strict=True):
        _check_u_ref(ref, seed, policy)


def _check_qc_ref(ref, expected_seed):
    _require_mapping(ref)
    if set(ref.keys()) != _QC_REF_KEYS:
        _fail()
    _require_identifier(ref["block_id"])
    _require_identifier(ref["slot_id"])
    _check_ref_seed(ref["seed"], expected_seed)
    _require_identifier(ref["resolution"])


def _check_qc_refs(refs, expected_seeds):
    refs = _require_sequence(refs)
    if len(refs) != len(expected_seeds):
        _fail()
    for ref, seed in zip(refs, expected_seeds, strict=True):
        _check_qc_ref(ref, seed)


def _validate_universal_record(record, input_contexts, actions):
    _require_mapping(record)
    for key in _U_RECORD_REQUIRED:
        if key not in record:
            _fail()

    context_id = record["context_id"]
    _require_identifier(context_id)
    if context_id not in input_contexts:
        _fail()

    policy = record["policy_id"]
    if policy not in POLICIES:
        _fail()

    model_id = record["model_id"]
    if model_id not in _MODEL_IDS:
        _fail()

    representation_id = POLICY_REPRESENTATION[policy]
    if record["representation_id"] != representation_id:
        _fail()
    if record["array_sha256"] != actions[representation_id]:
        _fail()
    _require_lower_hex64(record["model_spec_sha256"])
    _require_lower_hex64(record["fit_uid_sha256"])
    _require_lower_hex64(record["test_uid_sha256"])

    input_context = input_contexts[context_id]
    if record["fit_uid_sha256"] != input_context["fit_uid_sha256"]:
        _fail()
    if record["test_uid_sha256"] != input_context["test_uid_sha256"]:
        _fail()

    seeds = _model_seeds(model_id)
    _check_seeds(record["seeds"], seeds)

    classical = model_id in CLASSICAL_MODELS
    if policy == _MIN_POLICY:
        expected_reuse = _REUSE_HISTORICAL_CLASSICAL if classical else _REUSE_HISTORICAL_NEURAL
    else:
        expected_reuse = _REUSE_FUTURE
    if record["model_reuse_mode"] != expected_reuse:
        _fail()
    expected_calibration = _CALIBRATION_CLASSICAL if classical else _CALIBRATION_NEURAL
    if record["calibration_order"] != expected_calibration:
        _fail()

    _check_u_refs(record["refit_references"], seeds, policy)
    _check_u_refs(record["held_reference_jobs"], seeds, policy)
    if classical:
        _check_u_refs(record["calibration_references"], [NOT_APPLICABLE], policy)
    else:
        _check_u_refs(record["calibration_references"], seeds, policy)
    _check_u_ref(record["clean_endpoint_reference"], NOT_APPLICABLE, policy)

    body = {key: value for key, value in record.items() if key != "procedure_id"}
    if record["procedure_id"] != U_PREFIX + canonical_sha256(body):
        _fail()


def _validate_qc_record(record, input_contexts, actions):
    _require_mapping(record)
    for key in _QC_RECORD_REQUIRED:
        if key not in record:
            _fail()

    context_id = record["context_id"]
    _require_identifier(context_id)
    if context_id not in input_contexts:
        _fail()

    if record["policy_id"] != QC_POLICY:
        _fail()

    model_id = record["model_id"]
    if model_id not in _MODEL_IDS or model_id == "C-EXTRA-TREES":
        _fail()

    _require_lower_hex64(record["model_spec_sha256"])
    _require_lower_hex64(record["fit_uid_sha256"])
    _require_lower_hex64(record["test_uid_sha256"])
    _require_lower_hex64(record["binding_sha256"])
    if record["action_array_sha256"] != actions:
        _fail()

    input_context = input_contexts[context_id]
    if record["fit_uid_sha256"] != input_context["fit_uid_sha256"]:
        _fail()
    if record["test_uid_sha256"] != input_context["test_uid_sha256"]:
        _fail()

    seeds = _model_seeds(model_id)
    _check_seeds(record["seeds"], seeds)

    classical = model_id in CLASSICAL_MODELS
    if record["model_reuse_mode"] != _QC_REUSE_MODE:
        _fail()
    if record["clean_route_mode"] != _QC_CLEAN_ROUTE_MODE:
        _fail()
    expected_calibration = _CALIBRATION_CLASSICAL if classical else _CALIBRATION_NEURAL
    if record["calibration_order"] != expected_calibration:
        _fail()

    _check_qc_refs(record["refit_references"], seeds)
    _check_qc_refs(record["held_reference_jobs"], seeds)
    if classical:
        _check_qc_refs(record["calibration_references"], [NOT_APPLICABLE])
    else:
        _check_qc_refs(record["calibration_references"], seeds)
    _check_qc_ref(record["clean_endpoint_reference"], NOT_APPLICABLE)
    for key in _QC_SHARED_REF_KEYS:
        _check_qc_ref(record[key], NOT_APPLICABLE)

    body = {key: value for key, value in record.items() if key != "procedure_id"}
    if record["procedure_id"] != QC_PREFIX + canonical_sha256(body):
        _fail()


def _check_unique(records):
    seen_ids = set()
    seen_keys = set()
    for record in records:
        procedure_id = record["procedure_id"]
        _require_identifier(procedure_id)
        if procedure_id in seen_ids:
            _fail()
        seen_ids.add(procedure_id)
        key = (record["policy_id"], record["context_id"], record["model_id"])
        if key in seen_keys:
            _fail()
        seen_keys.add(key)


def _check_shared_qc_refs(records):
    seen = {}
    for record in records:
        context_id = record["context_id"]
        shared = [record[key] for key in _QC_SHARED_REF_KEYS]
        existing = seen.get(context_id)
        if existing is None:
            seen[context_id] = shared
        elif existing != shared:
            _fail()


def _alias_hash(alias, names):
    for name in names:
        if name in alias:
            return alias[name]
    _fail()


def _collect_family_aliases(minimal_bridge, input_contexts):
    raw_aliases = _require_sequence(minimal_bridge["fallback_endpoint_aliases"])
    family = []
    seen_ids = set()
    seen_pairs = set()
    per_context = {}
    for alias in raw_aliases:
        _require_mapping(alias)
        if alias.get("policy_id") != FAMILY_POLICY:
            continue
        alias_id = alias["alias_id"]
        _require_identifier(alias_id)
        if alias_id in seen_ids:
            _fail()
        seen_ids.add(alias_id)

        context_id = alias["context_id"]
        _require_identifier(context_id)
        if context_id not in input_contexts:
            _fail()

        strategy = alias["strategy"]
        _require_identifier(strategy)
        if strategy not in _FAMILY_STRATEGIES:
            _fail()

        pair = (context_id, strategy)
        if pair in seen_pairs:
            _fail()
        seen_pairs.add(pair)
        per_context.setdefault(context_id, set()).add(strategy)
        family.append(alias)

    for context_id in input_contexts:
        if per_context.get(context_id) != set(_FAMILY_STRATEGIES):
            _fail()
    return family


def _validate_family_alias(alias, context_id, model_id, min_record, minimal_bridge):
    if alias["context_id"] != context_id:
        _fail()
    if alias["recipe_id"] != model_id:
        _fail()
    if (
        _alias_hash(alias, ("fit_uid_sha256", "outer_fit_uid_sha256"))
        != min_record["fit_uid_sha256"]
    ):
        _fail()
    if (
        _alias_hash(alias, ("test_uid_sha256", "outer_test_uid_sha256"))
        != min_record["test_uid_sha256"]
    ):
        _fail()
    if alias["model_spec_sha256"] != min_record["model_spec_sha256"]:
        _fail()
    if _alias_hash(alias, ("array_sha256", "minimal_array_sha256")) != min_record["array_sha256"]:
        _fail()
    if alias["representation_id"] != min_record["representation_id"]:
        _fail()

    endpoint = min_record["clean_endpoint_reference"]
    if alias["target_job_id"] != endpoint["job_id"]:
        _fail()
    if alias["target_binding_sha256"] != endpoint["historical_binding_sha256"]:
        _fail()

    if type(alias["new_fits"]) is not int or alias["new_fits"] != 0:
        _fail()
    if type(alias["new_predictions"]) is not int or alias["new_predictions"] != 0:
        _fail()

    pipeline = alias["training_and_test_pipeline"]
    if "training_and_test_pipeline" in minimal_bridge:
        if pipeline != minimal_bridge["training_and_test_pipeline"]:
            _fail()
    else:
        _require_identifier(pipeline)

    _require_identifier(alias["reason"])
    if "target_evidence" not in alias:
        _fail()


def _validate_family_aliases(family_aliases, u_records, u_strategy_aliases, minimal_bridge):
    u_index = {}
    for record in u_records:
        u_index[(record["policy_id"], record["context_id"], record["model_id"])] = record

    min_alias = {}
    for alias in _require_sequence(u_strategy_aliases):
        _require_mapping(alias)
        if alias["policy_id"] != _MIN_POLICY:
            continue
        _require_identifier(alias["context_id"])
        _require_identifier(alias["strategy"])
        _require_identifier(alias["recipe_id"])
        min_alias[(alias["context_id"], alias["strategy"])] = alias

    for alias in family_aliases:
        context_id = alias["context_id"]
        strategy = alias["strategy"]
        if strategy in CLASSICAL_MODELS:
            model_id = strategy
        else:
            entry = min_alias.get((context_id, strategy))
            if entry is None:
                _fail()
            model_id = entry["recipe_id"]
            if model_id not in NEURAL_RECIPES:
                _fail()
        min_record = u_index.get((_MIN_POLICY, context_id, model_id))
        if min_record is None:
            _fail()
        _validate_family_alias(alias, context_id, model_id, min_record, minimal_bridge)


def _build(input_catalog, universal_procedures, qc_catalog, minimal_bridge):
    _require_mapping(input_catalog)
    _require_mapping(universal_procedures)
    _require_mapping(qc_catalog)
    _require_mapping(minimal_bridge)

    # Eager validation of the input catalog; the lazy job generator is
    # intentionally never consumed.
    iter_stress_input_jobs(input_catalog)

    input_binding = _require_mapping(input_catalog["input_binding"])
    actions = _require_mapping(input_binding["actions"])

    input_contexts = {}
    for context in _require_sequence(input_catalog["contexts"]):
        _require_mapping(context)
        input_contexts[context["context_id"]] = context

    bridge_sha256 = canonical_sha256(minimal_bridge)
    if universal_procedures.get("minimal_bridge_metadata_sha256") != bridge_sha256:
        _fail()
    if minimal_bridge.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        _fail()
    if minimal_bridge.get("execution_authorized") is not False:
        _fail()

    qc_procedures = build_qc_procedure_records(
        qc_catalog=qc_catalog,
        universal_procedures=universal_procedures,
    )
    _require_mapping(qc_procedures)

    qc_bindings = _require_mapping(qc_catalog["bindings"])
    if qc_bindings["actions"] != actions:
        _fail()

    compact_by_id = {}
    for context in _require_sequence(qc_bindings["universal_contexts"]):
        _require_mapping(context)
        compact_by_id[context["context_id"]] = context
    if set(compact_by_id) != set(input_contexts):
        _fail()
    for context_id, compact in compact_by_id.items():
        input_context = input_contexts[context_id]
        if compact["outer_fit_uid_sha256"] != input_context["fit_uid_sha256"]:
            _fail()
        if compact["outer_test_uid_sha256"] != input_context["test_uid_sha256"]:
            _fail()

    u_records = _require_sequence(universal_procedures["records"])
    for record in u_records:
        _validate_universal_record(record, input_contexts, actions)

    qc_records = _require_sequence(qc_procedures["records"])
    for record in qc_records:
        _validate_qc_record(record, input_contexts, actions)

    u_contexts = {record["context_id"] for record in u_records}
    if u_contexts != set(input_contexts):
        _fail()

    _check_unique(list(u_records) + list(qc_records))
    _check_shared_qc_refs(qc_records)

    family_aliases = _collect_family_aliases(minimal_bridge, input_contexts)
    _validate_family_aliases(
        family_aliases,
        u_records,
        universal_procedures["strategy_aliases"],
        minimal_bridge,
    )

    procedures = [copy.deepcopy(record) for record in u_records]
    procedures.extend(copy.deepcopy(record) for record in qc_records)
    procedures.sort(
        key=lambda record: (
            record["policy_id"],
            record["context_id"],
            record["model_id"],
        )
    )

    family_out = [copy.deepcopy(alias) for alias in family_aliases]
    family_out.sort(key=lambda alias: (alias["context_id"], alias["strategy"]))

    binding_sha256 = canonical_sha256(
        {
            "namespace": BINDING_NAMESPACE,
            "input_catalog_sha256": input_catalog["catalog_sha256"],
            "universal_procedure_sha256": universal_procedures["catalog_sha256"],
            "qc_catalog_sha256": qc_catalog["catalog_sha256"],
            "qc_procedure_sha256": qc_procedures["catalog_sha256"],
            "minimal_bridge_sha256": bridge_sha256,
        }
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "input_catalog": copy.deepcopy(input_catalog),
        "universal_procedures": copy.deepcopy(universal_procedures),
        "qc_catalog": copy.deepcopy(qc_catalog),
        "minimal_bridge": copy.deepcopy(minimal_bridge),
        "qc_procedures": copy.deepcopy(qc_procedures),
        "procedures": procedures,
        "family_aliases": family_out,
        "binding_sha256": binding_sha256,
    }
    bundle = dict(payload)
    bundle["bundle_sha256"] = canonical_sha256(payload)
    return bundle


def bind_stress_prediction_inputs(
    *,
    input_catalog,
    universal_procedures,
    qc_catalog,
    minimal_bridge,
):
    """Join caller-authenticated upstream metadata into a snapshot bundle.

    All four inputs are read and deep-copied only; they are never mutated.
    The returned bundle always keeps ``execution_authorized``,
    ``artifact_provenance_independently_verified`` ``False`` and
    ``scientific_operations`` ``0``.  Source-resolved values are preserved
    exactly; nothing is recomputed.
    """
    try:
        canonical_sha256(
            {
                "input_catalog": input_catalog,
                "universal_procedures": universal_procedures,
                "qc_catalog": qc_catalog,
                "minimal_bridge": minimal_bridge,
            }
        )
        return _build(input_catalog, universal_procedures, qc_catalog, minimal_bridge)
    except ValueError:
        raise ValueError(INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID) from None
