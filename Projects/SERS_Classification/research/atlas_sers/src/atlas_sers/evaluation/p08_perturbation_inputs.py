"""Exact metadata-only stress input dependency catalog for P08.

``build_stress_input_catalog`` canonicalises declarative input bindings,
case descriptors and context membership.  ``iter_stress_input_jobs``
validates and rebuilds that catalog and then streams the exact
input-operation job graph lazily in topological stage order.

Nothing numeric is computed: no spectral arrays, draws, scores or
thresholds.  This layer is an input-operation plan only and grants no
scientific authority.
"""

from __future__ import annotations

import hashlib
import json

from atlas_sers.evaluation.p08_perturbation_design import (
    build_perturbation_case_manifest,
)
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

__all__ = [
    "SCHEMA_VERSION",
    "JOB_PREFIX",
    "build_stress_input_catalog",
    "iter_stress_input_jobs",
    "require_scientific_execution",
]

SCHEMA_VERSION = "nato-sers-p08-stress-input-catalog-v1"
JOB_PREFIX = "P08STRESSINPUT-"
NOT_APPLICABLE = "not_applicable"
CLEAN_CASE_ID = "P08-STRESS-CLEAN"

_INVALID = "invalid_stress_input_metadata"

_FIVE_METHODS = [
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
    "D0-M",
    "P05-SELECTED",
]
_QC_STRESS_MODE = "fixed_clean_route_sensitivity"
_SHARED_EVIDENCE_KEYS = (
    "P01_contract_sha256",
    "P01_transform_source_sha256",
    "native_QC_source_sha256",
)
_BINDING_HASH_KEYS = (
    "raw_archive_sha256",
    "native_qc_file_sha256",
    "P01_contract_sha256",
    "P01_transform_source_sha256",
    "native_QC_source_sha256",
    "primary_manifest_file_sha256",
    "contexts_file_sha256",
    "roles_file_sha256",
    "raw_row_order_sha256",
    "frozen_action_row_order_sha256",
)
_ACTION_IDS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
_STARTING_AXIS_CM1 = [400, 1849, 1]
_OUTPUT_AXIS_CM1 = [400, 1800, 1]
_STOCHASTIC_FAMILIES = ("gaussian", "impulse")
_BINDING_FIELDS = frozenset(_BINDING_HASH_KEYS) | frozenset(
    {
        "ordered_population_uids",
        "ordered_raw_uids",
        "row_order_sha256",
        "actions",
        "starting_axis_cm1",
        "output_axis_cm1",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "context_id",
        "fit_uid_sha256",
        "test_uid_sha256",
        "fit_uids",
        "test_uids",
    }
)
_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "numerical_inputs_computed",
        "full_stress_job_ledger_complete",
        "design",
        "case_manifest",
        "input_binding",
        "contexts",
        "binding_sha256",
        "summary",
        "catalog_sha256",
    }
)
_HEX_CHARS = frozenset("0123456789abcdef")


def _fail():
    raise ValueError(_INVALID)


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX_CHARS for ch in value)


def _require_hex64(value):
    if not _is_hex64(value):
        _fail()
    return value


def _require_uid(value):
    if not isinstance(value, str) or value == "" or value != value.strip():
        _fail()
    value.encode("utf-8")
    return value


def _require_axis(value):
    if not isinstance(value, list) or len(value) != 3:
        _fail()
    for item in value:
        if type(item) is not int:
            _fail()
    return list(value)


def _snapshot(value):
    return json.loads(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    )


def _validate_binding(input_binding):
    if not isinstance(input_binding, dict):
        _fail()
    if set(input_binding.keys()) != _BINDING_FIELDS:
        _fail()
    for key in _BINDING_HASH_KEYS:
        _require_hex64(input_binding[key])
    population = input_binding["ordered_population_uids"]
    if not isinstance(population, list) or not population:
        _fail()
    seen_uids = set()
    for uid in population:
        _require_uid(uid)
        if uid in seen_uids:
            _fail()
        seen_uids.add(uid)
    row_order_sha256 = _require_hex64(input_binding["row_order_sha256"])
    if canonical_sha256(population) != row_order_sha256:
        _fail()
    raw_uids = input_binding["ordered_raw_uids"]
    if not isinstance(raw_uids, list) or len(raw_uids) != len(population):
        _fail()
    seen_raw_uids = set()
    for uid in raw_uids:
        _require_uid(uid)
        if uid in seen_raw_uids:
            _fail()
        seen_raw_uids.add(uid)
    raw_row_order_sha256 = _require_hex64(input_binding["raw_row_order_sha256"])
    if canonical_sha256(raw_uids) != raw_row_order_sha256:
        _fail()
    frozen_action_row_order_sha256 = _require_hex64(input_binding["frozen_action_row_order_sha256"])
    if (
        hashlib.sha256("\n".join(population).encode("utf-8")).hexdigest()
        != frozen_action_row_order_sha256
    ):
        _fail()
    starting_axis = _require_axis(input_binding["starting_axis_cm1"])
    if starting_axis != _STARTING_AXIS_CM1:
        _fail()
    output_axis = _require_axis(input_binding["output_axis_cm1"])
    if output_axis != _OUTPUT_AXIS_CM1:
        _fail()
    actions = input_binding["actions"]
    if not isinstance(actions, dict) or set(actions.keys()) != set(_ACTION_IDS):
        _fail()
    normalized_actions = {}
    for key in _ACTION_IDS:
        normalized_actions[key] = _require_hex64(actions[key])
    return {
        "raw_archive_sha256": input_binding["raw_archive_sha256"],
        "native_qc_file_sha256": input_binding["native_qc_file_sha256"],
        "P01_contract_sha256": input_binding["P01_contract_sha256"],
        "P01_transform_source_sha256": input_binding["P01_transform_source_sha256"],
        "native_QC_source_sha256": input_binding["native_QC_source_sha256"],
        "primary_manifest_file_sha256": input_binding["primary_manifest_file_sha256"],
        "contexts_file_sha256": input_binding["contexts_file_sha256"],
        "roles_file_sha256": input_binding["roles_file_sha256"],
        "ordered_population_uids": list(population),
        "ordered_raw_uids": list(raw_uids),
        "row_order_sha256": row_order_sha256,
        "raw_row_order_sha256": raw_row_order_sha256,
        "frozen_action_row_order_sha256": frozen_action_row_order_sha256,
        "actions": normalized_actions,
        "starting_axis_cm1": starting_axis,
        "output_axis_cm1": output_axis,
    }


def _validate_design_extras(design, binding):
    if not isinstance(design, dict):
        _fail()
    if design.get("selected_universal_panel") != _FIVE_METHODS:
        _fail()
    if design.get("selected_qc_stress_mode") != _QC_STRESS_MODE:
        _fail()
    if design.get("native_grid_gate_reaction_experiment_included") is not False:
        _fail()
    evidence = design.get("immutable_source_evidence")
    if not isinstance(evidence, dict):
        _fail()
    for key in _SHARED_EVIDENCE_KEYS:
        if _require_hex64(evidence.get(key)) != binding[key]:
            _fail()
    design_input = design.get("input")
    if not isinstance(design_input, dict):
        _fail()
    if _require_axis(design_input.get("starting_axis_cm1")) != binding["starting_axis_cm1"]:
        _fail()
    if _require_axis(design_input.get("output_axis_cm1")) != binding["output_axis_cm1"]:
        _fail()


def _validate_uid_list(value, population):
    if not isinstance(value, list) or not value:
        _fail()
    seen = set()
    for uid in value:
        _require_uid(uid)
        if uid not in population or uid in seen:
            _fail()
        seen.add(uid)
    if value != sorted(value):
        _fail()
    return list(value)


def _validate_contexts(contexts, population):
    if not isinstance(contexts, list) or not contexts:
        _fail()
    normalized = []
    seen_ids = set()
    for context in contexts:
        if not isinstance(context, dict) or set(context.keys()) != _CONTEXT_FIELDS:
            _fail()
        context_id = context["context_id"]
        if (
            not isinstance(context_id, str)
            or context_id == ""
            or context_id != context_id.strip()
            or context_id == NOT_APPLICABLE
        ):
            _fail()
        if context_id in seen_ids:
            _fail()
        seen_ids.add(context_id)
        fit_uids = _validate_uid_list(context["fit_uids"], population)
        test_uids = _validate_uid_list(context["test_uids"], population)
        if set(fit_uids) & set(test_uids):
            _fail()
        fit_uid_sha256 = _require_hex64(context["fit_uid_sha256"])
        test_uid_sha256 = _require_hex64(context["test_uid_sha256"])
        if canonical_sha256(fit_uids) != fit_uid_sha256:
            _fail()
        if canonical_sha256(test_uids) != test_uid_sha256:
            _fail()
        normalized.append(
            {
                "context_id": context_id,
                "fit_uid_sha256": fit_uid_sha256,
                "test_uid_sha256": test_uid_sha256,
                "fit_uids": fit_uids,
                "test_uids": test_uids,
            }
        )
    normalized.sort(key=lambda item: item["context_id"])
    return normalized


def _build_summary(design, manifest, contexts, binding):
    distinct_test_uids = sorted({uid for context in contexts for uid in context["test_uids"]})
    unique_count = len(distinct_test_uids)
    appearances = sum(len(context["test_uids"]) for context in contexts)
    context_count = len(contexts)
    cases = manifest["cases"]
    case_count = len(cases)
    gaussian_count = 0
    for case in cases:
        if case["family"] == "gaussian":
            gaussian_count += 1
    independent_count = case_count - gaussian_count
    action_count = len(binding["actions"])
    stochastic_replicates = design["case_definition"]["stochastic_replicates"]
    raw_case_count = independent_count * unique_count + gaussian_count * appearances
    stage_counts = {
        "row_prepare": unique_count,
        "source_noise_reference": context_count,
        "stochastic_realization": len(_STOCHASTIC_FAMILIES) * stochastic_replicates * unique_count,
        "raw_case": raw_case_count,
        "action_transform": action_count * raw_case_count,
        "zero_input_parity": action_count * unique_count,
        "context_action_assembly": context_count * case_count * action_count,
    }
    return {
        "context_count": context_count,
        "distinct_test_uid_count": unique_count,
        "test_uid_appearances": appearances,
        "case_count": case_count,
        "context_independent_case_count": independent_count,
        "context_dependent_case_count": gaussian_count,
        "action_count": action_count,
        "stage_counts": stage_counts,
        "total_job_count": sum(stage_counts.values()),
    }


def _build(design, input_binding, contexts):
    binding = _validate_binding(input_binding)
    _validate_design_extras(design, binding)
    population = set(binding["ordered_population_uids"])
    normalized_contexts = _validate_contexts(contexts, population)
    manifest = build_perturbation_case_manifest(design=design)
    if not isinstance(manifest, dict):
        _fail()
    if manifest.get("execution_authorized") is not False:
        _fail()
    operations = manifest.get("scientific_operations")
    if type(operations) is not int or operations != 0:
        _fail()

    design_copy = _snapshot(design)
    binding_copy = _snapshot(binding)
    contexts_copy = _snapshot(normalized_contexts)
    manifest_copy = _snapshot(manifest)

    binding_sha256 = canonical_sha256(
        {
            "design": design_copy,
            "case_manifest": manifest_copy,
            "input_binding": binding_copy,
            "contexts": contexts_copy,
        }
    )
    summary = _build_summary(design_copy, manifest_copy, contexts_copy, binding_copy)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "scientific_operations": 0,
        "artifact_provenance_independently_verified": False,
        "numerical_inputs_computed": False,
        "full_stress_job_ledger_complete": False,
        "design": design_copy,
        "case_manifest": manifest_copy,
        "input_binding": binding_copy,
        "contexts": contexts_copy,
        "binding_sha256": binding_sha256,
        "summary": summary,
    }
    catalog = dict(payload)
    catalog["catalog_sha256"] = canonical_sha256(payload)
    return catalog


def build_stress_input_catalog(*, design, input_binding, contexts):
    """Build the compact deterministic metadata-only input catalog.

    ``ordered_raw_uids[i]`` maps to ``ordered_population_uids[i]``; the
    caller remains responsible for the physical raw-archive mapping and
    for authenticating the referenced files.
    """
    try:
        canonical_sha256(
            {
                "design": design,
                "input_binding": input_binding,
                "contexts": contexts,
            }
        )
        return _build(design, input_binding, contexts)
    except ValueError:
        raise ValueError(_INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(_INVALID) from None


def _make_job(
    binding_sha256,
    stage,
    context_id,
    observation_uid,
    case_id,
    representation_id,
    family,
    replicate_index,
    depends_on_job_ids,
):
    body = {
        "binding_sha256": binding_sha256,
        "stage": stage,
        "context_id": context_id,
        "observation_uid": observation_uid,
        "case_id": case_id,
        "representation_id": representation_id,
        "family": family,
        "replicate_index": replicate_index,
        "depends_on_job_ids": sorted(set(depends_on_job_ids)),
    }
    record = dict(body)
    record["job_id"] = JOB_PREFIX + canonical_sha256(body)
    return record


def _iter_jobs(design, manifest, contexts, binding_sha256):
    na = NOT_APPLICABLE
    cases = manifest["cases"]
    clean_case = None
    for case in cases:
        if case["case_id"] == CLEAN_CASE_ID:
            clean_case = case
            break
    if clean_case is None:
        _fail()
    non_gaussian_cases = [case for case in cases if case["family"] != "gaussian"]
    gaussian_cases = [case for case in cases if case["family"] == "gaussian"]
    stochastic_replicates = design["case_definition"]["stochastic_replicates"]
    distinct_test_uids = sorted({uid for context in contexts for uid in context["test_uids"]})

    def row_prepare(uid):
        return _make_job(binding_sha256, "row_prepare", na, uid, na, na, na, None, ())

    def noise_reference(context_id):
        return _make_job(
            binding_sha256, "source_noise_reference", context_id, na, na, na, na, None, ()
        )

    def stochastic(uid, family, replicate_index):
        return _make_job(
            binding_sha256, "stochastic_realization", na, uid, na, na, family, replicate_index, ()
        )

    def raw_shared(uid, case):
        deps = [row_prepare(uid)["job_id"]]
        if case["family"] == "impulse":
            deps.append(stochastic(uid, "impulse", case["replicate_index"])["job_id"])
        return _make_job(
            binding_sha256,
            "raw_case",
            na,
            uid,
            case["case_id"],
            na,
            case["family"],
            case["replicate_index"],
            deps,
        )

    def raw_context(context_id, uid, case):
        deps = [
            row_prepare(uid)["job_id"],
            stochastic(uid, "gaussian", case["replicate_index"])["job_id"],
            noise_reference(context_id)["job_id"],
        ]
        return _make_job(
            binding_sha256,
            "raw_case",
            context_id,
            uid,
            case["case_id"],
            na,
            "gaussian",
            case["replicate_index"],
            deps,
        )

    def action_transform(context_id, uid, case, action, raw_job_id):
        return _make_job(
            binding_sha256,
            "action_transform",
            context_id,
            uid,
            case["case_id"],
            action,
            case["family"],
            case["replicate_index"],
            [raw_job_id],
        )

    def parity(uid, action):
        clean_raw = raw_shared(uid, clean_case)
        clean_action = action_transform(na, uid, clean_case, action, clean_raw["job_id"])
        return _make_job(
            binding_sha256,
            "zero_input_parity",
            na,
            uid,
            clean_case["case_id"],
            action,
            "clean",
            None,
            [clean_action["job_id"]],
        )

    for uid in distinct_test_uids:
        yield row_prepare(uid)

    for context in contexts:
        yield noise_reference(context["context_id"])

    for uid in distinct_test_uids:
        for family in _STOCHASTIC_FAMILIES:
            for replicate_index in range(stochastic_replicates):
                yield stochastic(uid, family, replicate_index)

    for case in non_gaussian_cases:
        for uid in distinct_test_uids:
            yield raw_shared(uid, case)

    for context in contexts:
        context_id = context["context_id"]
        for uid in context["test_uids"]:
            for case in gaussian_cases:
                yield raw_context(context_id, uid, case)

    for case in non_gaussian_cases:
        for uid in distinct_test_uids:
            raw_job_id = raw_shared(uid, case)["job_id"]
            for action in _ACTION_IDS:
                yield action_transform(na, uid, case, action, raw_job_id)

    for context in contexts:
        context_id = context["context_id"]
        for uid in context["test_uids"]:
            for case in gaussian_cases:
                raw_job_id = raw_context(context_id, uid, case)["job_id"]
                for action in _ACTION_IDS:
                    yield action_transform(context_id, uid, case, action, raw_job_id)

    parity_ids = {}
    for uid in distinct_test_uids:
        for action in _ACTION_IDS:
            record = parity(uid, action)
            parity_ids[(uid, action)] = record["job_id"]
            yield record

    for context in contexts:
        context_id = context["context_id"]
        for case in cases:
            context_bound = case["family"] == "gaussian"
            for action in _ACTION_IDS:
                deps = []
                for uid in context["test_uids"]:
                    if context_bound:
                        raw_job_id = raw_context(context_id, uid, case)["job_id"]
                        deps.append(
                            action_transform(context_id, uid, case, action, raw_job_id)["job_id"]
                        )
                    else:
                        raw_job_id = raw_shared(uid, case)["job_id"]
                        deps.append(action_transform(na, uid, case, action, raw_job_id)["job_id"])
                    deps.append(parity_ids[(uid, action)])
                yield _make_job(
                    binding_sha256,
                    "context_action_assembly",
                    context_id,
                    na,
                    case["case_id"],
                    action,
                    case["family"],
                    case["replicate_index"],
                    deps,
                )


def iter_stress_input_jobs(catalog):
    """Rebuild and verify ``catalog``, then lazily stream exact jobs."""
    try:
        if not isinstance(catalog, dict) or set(catalog.keys()) != _CATALOG_FIELDS:
            _fail()
        if catalog.get("schema_version") != SCHEMA_VERSION:
            _fail()
        if catalog.get("execution_authorized") is not False:
            _fail()
        operations = catalog.get("scientific_operations")
        if type(operations) is not int or operations != 0:
            _fail()
        for flag in (
            "artifact_provenance_independently_verified",
            "numerical_inputs_computed",
            "full_stress_job_ledger_complete",
        ):
            if catalog.get(flag) is not False:
                _fail()
        catalog_sha256 = _require_hex64(catalog.get("catalog_sha256"))
        content = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
        if canonical_sha256(content) != catalog_sha256:
            _fail()
        rebuilt = build_stress_input_catalog(
            design=catalog["design"],
            input_binding=catalog["input_binding"],
            contexts=catalog["contexts"],
        )
        if rebuilt != catalog:
            _fail()
        design = _snapshot(rebuilt["design"])
        manifest = _snapshot(rebuilt["case_manifest"])
        contexts = _snapshot(rebuilt["contexts"])
        binding_sha256 = rebuilt["binding_sha256"]
    except ValueError:
        raise ValueError(_INVALID) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(_INVALID) from None
    return _iter_jobs(design, manifest, contexts, binding_sha256)


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""
    raise ValueError("scientific_execution_not_authorized")
