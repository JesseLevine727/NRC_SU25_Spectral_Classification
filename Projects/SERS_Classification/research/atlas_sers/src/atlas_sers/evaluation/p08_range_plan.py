"""Pure plan expansion of the pinned universal slot DAG to the 400-1849 range.

Planning metadata only. The hashes in this module bind reviewed planning
metadata only: file bytes, normalization, and sample/instrument isolation
remain independently authenticated upstream. This module never fits, predicts,
reads arrays, writes files, mutates inputs, or grants scientific execution
permission. It validates only the pinned primary plan shape, not arbitrary
graphs.
"""

from __future__ import annotations

import heapq
import re

from . import p08_plan as core

PRIMARY_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"
PRIMARY_ROW_ORDER_SHA256 = "b0d9ef9ae34a87443d951742bf1b295df522dd12674d5ca4a9785953cee6a5a7"
POLICY_ID = "PP-RANGE-MIN"
REPRESENTATION_ID = "R_MIN_400_1849"
SCHEMA_VERSION = "nato-sers-p08-range-slot-dag-v1"

_BASE_POLICY_ID = "PP-U-MIN"
_BASE_REPRESENTATION_ID = "R_MIN_400_1800"
_BASE_EVIDENCE_STATUS = "historical_reuse_requires_authentication"
_FIXED_RECIPE = "fixed_recipe"
_SEED_STAGE = "seed_ensemble_prediction"
_DTYPE = "float32"

_PRIMARY_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "jobs",
        "aliases",
        "summary",
        "plan_sha256",
    }
)
_JOB_KEYS = frozenset(core.JOB_FIELDS) | frozenset({"job_id"})
_ALIAS_KEYS = frozenset(
    {
        "policy_id",
        "context_id",
        "strategy",
        "recipe_id",
        "target_job_id",
        "alias_id",
    }
)
_RANGE_KEYS = frozenset(
    {
        "representation_id",
        "rows",
        "features",
        "dtype",
        "axis_start_cm1",
        "axis_end_cm1",
        "array_sha256",
        "axis_sha256",
        "row_order_sha256",
        "file_sha256",
        "invalid_rows",
    }
)
_MODEL_IDS = frozenset({"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "D1", "D2", "D3"})
_NEURAL_MODEL_IDS = frozenset({"D0-M", "D1", "D2", "D3"})
_NEURAL_SOURCE_STAGES = frozenset({"source_fit", "source_validation_prediction"})
_ALIAS_STRATEGIES = frozenset({"D0-M", "P05-SELECTED"})

_HEX64 = re.compile(r"[0-9a-f]{64}")

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "range_plan_input_unexpected",
        "primary_plan_not_plain_dict",
        "primary_plan_keys_invalid",
        "primary_schema_unregistered",
        "primary_execution_must_be_denied",
        "primary_plan_hash_mismatch",
        "primary_payload_hash_mismatch",
        "primary_jobs_not_list",
        "primary_aliases_not_list",
        "primary_summary_not_dict",
        "range_input_not_plain_dict",
        "range_input_keys_invalid",
        "range_input_representation_invalid",
        "range_input_dtype_invalid",
        "range_input_rows_invalid",
        "range_input_features_invalid",
        "range_input_axis_start_invalid",
        "range_input_axis_end_invalid",
        "range_input_invalid_rows_invalid",
        "range_input_array_sha256_invalid",
        "range_input_axis_sha256_invalid",
        "range_input_row_order_sha256_invalid",
        "range_input_file_sha256_invalid",
        "primary_job_not_plain_dict",
        "primary_job_keys_invalid",
        "primary_job_id_not_string",
        "primary_job_ids_duplicated",
        "primary_representation_invalid",
        "primary_evidence_invalid",
        "range_selection_empty",
        "primary_dependencies_not_list",
        "primary_dependency_invalid",
        "primary_dependency_duplicated",
        "primary_dependency_unregistered",
        "primary_model_spec_invalid",
        "primary_model_spec_inconsistent",
        "primary_neural_candidate_invalid",
        "primary_neural_hyperparameter_mismatch",
        "primary_dependency_cycle",
        "primary_alias_not_plain_dict",
        "primary_alias_keys_invalid",
        "primary_alias_strategy_invalid",
        "primary_alias_recipe_invalid",
        "primary_alias_target_stage_invalid",
        "primary_alias_context_mismatch",
        "primary_alias_target_unregistered",
        "primary_alias_model_mismatch",
        "primary_alias_id_not_string",
        "primary_alias_ids_duplicated",
        "primary_alias_selection_duplicated",
        "range_alias_selection_empty",
    }
)
_FALLBACK_REASON_CODE = "range_plan_input_unexpected"


class RangePlanError(ValueError):
    """Static, input-independent plan expansion failure."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = _FALLBACK_REASON_CODE
        super().__init__(reason_code)
        self.reason_code = reason_code


def require_scientific_execution(*args, **kwargs):
    raise RangePlanError("scientific_execution_not_authorized")


def build_range_plan(primary_plan, range_input):
    try:
        return _build_range_plan(primary_plan, range_input)
    except RangePlanError:
        raise
    except Exception:
        raise RangePlanError("range_plan_input_unexpected") from None


def _hex64(value):
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _exact_int(value, expected):
    return type(value) is int and value == expected


def _validate_primary_plan(primary_plan):
    if type(primary_plan) is not dict:
        raise RangePlanError("primary_plan_not_plain_dict")
    if set(primary_plan) != _PRIMARY_KEYS:
        raise RangePlanError("primary_plan_keys_invalid")
    if primary_plan["schema_version"] != core.SCHEMA_VERSION:
        raise RangePlanError("primary_schema_unregistered")
    if primary_plan["execution_authorized"] is not False:
        raise RangePlanError("primary_execution_must_be_denied")
    if primary_plan["plan_sha256"] != PRIMARY_PLAN_SHA256:
        raise RangePlanError("primary_plan_hash_mismatch")
    payload = {key: value for key, value in primary_plan.items() if key != "plan_sha256"}
    if core._hash(payload) != PRIMARY_PLAN_SHA256:
        raise RangePlanError("primary_payload_hash_mismatch")
    if type(primary_plan["jobs"]) is not list:
        raise RangePlanError("primary_jobs_not_list")
    if type(primary_plan["aliases"]) is not list:
        raise RangePlanError("primary_aliases_not_list")
    if type(primary_plan["summary"]) is not dict:
        raise RangePlanError("primary_summary_not_dict")


def _validate_range_input(range_input):
    if type(range_input) is not dict:
        raise RangePlanError("range_input_not_plain_dict")
    if set(range_input) != _RANGE_KEYS:
        raise RangePlanError("range_input_keys_invalid")
    if (
        type(range_input["representation_id"]) is not str
        or range_input["representation_id"] != REPRESENTATION_ID
    ):
        raise RangePlanError("range_input_representation_invalid")
    if type(range_input["dtype"]) is not str or range_input["dtype"] != _DTYPE:
        raise RangePlanError("range_input_dtype_invalid")
    if not _exact_int(range_input["rows"], 598):
        raise RangePlanError("range_input_rows_invalid")
    if not _exact_int(range_input["features"], 1450):
        raise RangePlanError("range_input_features_invalid")
    if not _exact_int(range_input["axis_start_cm1"], 400):
        raise RangePlanError("range_input_axis_start_invalid")
    if not _exact_int(range_input["axis_end_cm1"], 1849):
        raise RangePlanError("range_input_axis_end_invalid")
    if not _exact_int(range_input["invalid_rows"], 0):
        raise RangePlanError("range_input_invalid_rows_invalid")
    if not _hex64(range_input["array_sha256"]):
        raise RangePlanError("range_input_array_sha256_invalid")
    if not _hex64(range_input["axis_sha256"]):
        raise RangePlanError("range_input_axis_sha256_invalid")
    if (
        not _hex64(range_input["row_order_sha256"])
        or range_input["row_order_sha256"] != PRIMARY_ROW_ORDER_SHA256
    ):
        raise RangePlanError("range_input_row_order_sha256_invalid")
    if not _hex64(range_input["file_sha256"]):
        raise RangePlanError("range_input_file_sha256_invalid")


def _select_jobs(jobs):
    retained = {}
    for job in jobs:
        if type(job) is not dict:
            raise RangePlanError("primary_job_not_plain_dict")
        if set(job) != _JOB_KEYS:
            raise RangePlanError("primary_job_keys_invalid")
        if job["policy_id"] != _BASE_POLICY_ID:
            continue
        if job["model_id"] not in _MODEL_IDS:
            continue
        job_id = job["job_id"]
        if type(job_id) is not str:
            raise RangePlanError("primary_job_id_not_string")
        if job_id in retained:
            raise RangePlanError("primary_job_ids_duplicated")
        if job["representation_id"] != _BASE_REPRESENTATION_ID:
            raise RangePlanError("primary_representation_invalid")
        if job["evidence_status"] != _BASE_EVIDENCE_STATUS:
            raise RangePlanError("primary_evidence_invalid")
        retained[job_id] = job
    if not retained:
        raise RangePlanError("range_selection_empty")
    registered = set(retained)
    for job in retained.values():
        dependencies = job["dependencies"]
        if type(dependencies) is not list:
            raise RangePlanError("primary_dependencies_not_list")
        seen = set()
        for dependency in dependencies:
            if type(dependency) is not str:
                raise RangePlanError("primary_dependency_invalid")
            if dependency in seen:
                raise RangePlanError("primary_dependency_duplicated")
            if dependency not in registered:
                raise RangePlanError("primary_dependency_unregistered")
            seen.add(dependency)
    return retained


def _model_base_specs(retained):
    model_base = {}
    for job in retained.values():
        model = job["model_id"]
        spec = job["model_spec_sha256"]
        if not _hex64(spec):
            raise RangePlanError("primary_model_spec_invalid")
        if model in model_base:
            if model_base[model] != spec:
                raise RangePlanError("primary_model_spec_inconsistent")
        else:
            model_base[model] = spec
    return model_base


def _derive_model_specs(model_base, input_hash):
    specifications = {}
    digests = {}
    for model in sorted(model_base):
        payload = {
            "base_model_spec_sha256": model_base[model],
            "range_input_contract_sha256": input_hash,
        }
        specifications[model] = payload
        digests[model] = core._hash(payload)
    return specifications, digests


def _transform_fields(job, model_spec_sha256, new_id_of, array_sha256):
    model = job["model_id"]
    fields = dict(job)
    del fields["job_id"]
    fields["policy_id"] = POLICY_ID
    fields["representation_id"] = REPRESENTATION_ID
    fields["array_sha256"] = array_sha256
    fields["evidence_status"] = core.EVIDENCE_FUTURE
    fields["model_spec_sha256"] = model_spec_sha256[model]
    if model in _NEURAL_MODEL_IDS and job["stage"] in _NEURAL_SOURCE_STAGES:
        if job["candidate_id"] != _FIXED_RECIPE:
            raise RangePlanError("primary_neural_candidate_invalid")
        if job["hyperparameter_sha256"] != job["model_spec_sha256"]:
            raise RangePlanError("primary_neural_hyperparameter_mismatch")
        fields["hyperparameter_sha256"] = model_spec_sha256[model]
    fields["dependencies"] = sorted(new_id_of[dep] for dep in job["dependencies"])
    return fields


def _topological_transform(retained, model_spec_sha256, array_sha256):
    indegree = {job_id: len(job["dependencies"]) for job_id, job in retained.items()}
    reverse = {job_id: [] for job_id in retained}
    for job_id, job in retained.items():
        for dependency in job["dependencies"]:
            reverse[dependency].append(job_id)
    ready = [job_id for job_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    new_id_of = {}
    transformed = {}
    while ready:
        job_id = heapq.heappop(ready)
        job = retained[job_id]
        fields = _transform_fields(job, model_spec_sha256, new_id_of, array_sha256)
        new_id = "P08RANGEJOB-" + core._hash(fields)
        new_id_of[job_id] = new_id
        output = dict(fields)
        output["job_id"] = new_id
        transformed[job_id] = output
        for child in reverse[job_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(new_id_of) != len(retained):
        raise RangePlanError("primary_dependency_cycle")
    return new_id_of, transformed


def _alias_recipe_valid(recipe):
    return type(recipe) is str and recipe != ""


def _build_aliases(aliases, retained, new_id_of):
    output = []
    selected_ids = set()
    selected_pairs = set()
    for alias in aliases:
        if type(alias) is not dict:
            raise RangePlanError("primary_alias_not_plain_dict")
        if set(alias) != _ALIAS_KEYS:
            raise RangePlanError("primary_alias_keys_invalid")
        if alias["policy_id"] != _BASE_POLICY_ID:
            continue
        if alias["strategy"] not in _ALIAS_STRATEGIES:
            raise RangePlanError("primary_alias_strategy_invalid")
        if not _alias_recipe_valid(alias["recipe_id"]):
            raise RangePlanError("primary_alias_recipe_invalid")
        target = alias["target_job_id"]
        if type(target) is not str or target not in retained:
            raise RangePlanError("primary_alias_target_unregistered")
        target_job = retained[target]
        if target_job["stage"] != _SEED_STAGE:
            raise RangePlanError("primary_alias_target_stage_invalid")
        if target_job["context_id"] != alias["context_id"]:
            raise RangePlanError("primary_alias_context_mismatch")
        if target_job["model_id"] != alias["recipe_id"]:
            raise RangePlanError("primary_alias_model_mismatch")
        pair = (alias["context_id"], alias["strategy"])
        if pair in selected_pairs:
            raise RangePlanError("primary_alias_selection_duplicated")
        selected_pairs.add(pair)
        original_id = alias["alias_id"]
        if type(original_id) is not str:
            raise RangePlanError("primary_alias_id_not_string")
        if original_id in selected_ids:
            raise RangePlanError("primary_alias_ids_duplicated")
        selected_ids.add(original_id)
        fields = dict(alias)
        del fields["alias_id"]
        fields["policy_id"] = POLICY_ID
        fields["target_job_id"] = new_id_of[target]
        fields["alias_id"] = "P08RANGEALIAS-" + core._hash(fields)
        output.append(fields)
    if not output:
        raise RangePlanError("range_alias_selection_empty")
    return sorted(output, key=lambda item: item["alias_id"])


def _summarize(jobs, aliases):
    contexts = set()
    stage_counts = {}
    by_model = {}
    model_fit_slots = 0
    scalar_calibrations = 0
    for job in jobs:
        contexts.add(job["context_id"])
        stage = job["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        is_fit = stage in core.MODEL_FIT_STAGES
        is_scalar = stage == core.SCALAR_STAGE
        if is_fit:
            model_fit_slots += 1
        if is_scalar:
            scalar_calibrations += 1
        model = job["model_id"]
        entry = by_model.get(model)
        if entry is None:
            entry = {
                "jobs_count": 0,
                "model_fit_slots": 0,
                "scalar_calibrations": 0,
                "stage_counts": {},
            }
            by_model[model] = entry
        entry["jobs_count"] += 1
        entry["stage_counts"][stage] = entry["stage_counts"].get(stage, 0) + 1
        if is_fit:
            entry["model_fit_slots"] += 1
        if is_scalar:
            entry["scalar_calibrations"] += 1
    return {
        "context_count": len(contexts),
        "jobs_count": len(jobs),
        "aliases_count": len(aliases),
        "model_fit_slots": model_fit_slots,
        "scalar_calibrations": scalar_calibrations,
        "stage_counts": stage_counts,
        "authorized_fit_slots": 0,
        "by_model": by_model,
    }


def _build_range_plan(primary_plan, range_input):
    _validate_primary_plan(primary_plan)
    _validate_range_input(range_input)
    input_hash = core._hash(range_input)
    retained = _select_jobs(primary_plan["jobs"])
    model_base = _model_base_specs(retained)
    specifications, digests = _derive_model_specs(model_base, input_hash)
    new_id_of, transformed = _topological_transform(retained, digests, range_input["array_sha256"])
    jobs = sorted(transformed.values(), key=lambda job: job["job_id"])
    aliases = _build_aliases(primary_plan["aliases"], retained, new_id_of)
    summary = _summarize(jobs, aliases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "source_primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "input_contract": dict(range_input),
        "input_contract_sha256": input_hash,
        "model_specifications": specifications,
        "model_spec_sha256": digests,
        "jobs": jobs,
        "aliases": aliases,
        "summary": summary,
    }
    result["plan_sha256"] = core._hash(result)
    return result
