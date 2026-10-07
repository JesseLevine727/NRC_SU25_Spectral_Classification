"""P08 exact stress score graph (metadata only).

This module is an INTERNAL metadata adapter.  It consumes the exact score
support catalog produced by
``p08_perturbation_score_support.validate_stress_score_support`` and derives a
deterministic, execution-disabled description of the stress scoring DAG.

It performs **zero** scientific work: no arrays, models, scores, thresholds,
routes, draws, calibrations, fits, probabilities or statistics are computed,
guessed or inferred.  Every dynamic score stays a deferred upstream dependency.
The returned records are metadata only, the full stress job ledger is
explicitly *not* complete, and ``require_scientific_execution`` always denies
execution.

Malformed metadata is sanitised to ``invalid_stress_score_metadata``; only
``ValueError``/``TypeError``/``KeyError``/``UnicodeError``/``RecursionError``/
``OverflowError`` are caught, never ``BaseException`` or blanket ``Exception``.
"""

from __future__ import annotations

from .p08_perturbation_predictions import iter_stress_prediction_records
from .p08_perturbation_score_support import validate_stress_score_support
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "iter_stress_score_records",
    "require_scientific_execution",
]


INVALID_METADATA = "invalid_stress_score_metadata"
EXECUTION_DENIED = "scientific_execution_not_authorized"

_NA = "not_applicable"

JOB_PREFIX = "P08STRESSSCORE-"
ALIAS_PREFIX = "P08STRESSSCOREALIAS-"

ENDPOINTS = ("M01", "M06")

STAGE_CONTEXT_CASE = "context_case_score"
STAGE_CONTEXT_FAMILY = "context_family_curve"
STAGE_POOLED_CASE = "pooled_case_score"
STAGE_POOLED_FAMILY = "pooled_family_curve"

MODE_CONTEXT_CASE = "context_case"
MODE_CONTEXT_FAMILY = "context_family"
MODE_POOLED_CASE = "pooled_case"
MODE_POOLED_FAMILY = "pooled_family"

_INPUT_STAGE_PREDICTION_UNITS = "prediction_units"
_INPUT_STAGE_CLEAN_PARITY = "clean_probability_parity"

_FAMILY_KEYS = (
    "shift",
    "slope",
    "quadratic",
    "gaussian",
    "impulse",
    "clipping",
)
_CLEAN_CASE_ID = "P08-STRESS-CLEAN"

RESOLUTION_CONTEXT_CASE = (
    "registered_context_present_class_scores_and_probability_diagnostics_no_resampling"
)
RESOLUTION_CURVE = (
    "all_registered_cases_mean_replicate_scores_signed_directions_normalized_area_"
    "negative_losses_retained"
)
RESOLUTION_POOLED_CASE = (
    "concatenate_four_disjoint_folds_rebuild_M01_M06_present_class_scores_no_cross_repeat_pooling"
)


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution for this metadata-only adapter."""
    raise ValueError(EXECUTION_DENIED)


def _fail():
    raise ValueError(INVALID_METADATA)


def _lookup(mapping, key):
    found = mapping.get(key)
    if found is None:
        _fail()
    return found


def _index_parent(prediction_catalog):
    units = {}
    parity = {}
    for record in iter_stress_prediction_records(prediction_catalog):
        if not isinstance(record, dict) or record.get("record_type") != "job":
            continue
        stage = record.get("stage")
        if stage == _INPUT_STAGE_PREDICTION_UNITS:
            key = (record["procedure_id"], record["case_id"])
            if key in units:
                _fail()
            units[key] = record["job_id"]
        elif stage == _INPUT_STAGE_CLEAN_PARITY:
            procedure_id = record["procedure_id"]
            if procedure_id in parity:
                _fail()
            parity[procedure_id] = record["job_id"]
    if not units or not parity:
        _fail()
    return units, parity


def _manifest_case_ids(manifest):
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail()
    case_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            _fail()
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id != case_id.strip():
            _fail()
        if case_id in case_ids:
            _fail()
        case_ids.add(case_id)
    return case_ids


def _family_case_ids(prediction_catalog):
    manifest = prediction_catalog["bundle"]["input_catalog"]["case_manifest"]
    if not isinstance(manifest, dict):
        _fail()
    family_cases = manifest.get("family_cases")
    if not isinstance(family_cases, dict):
        _fail()
    if set(family_cases) != set(_FAMILY_KEYS):
        _fail()
    case_ids = _manifest_case_ids(manifest)
    families = {}
    for family in _FAMILY_KEYS:
        members = family_cases[family]
        if not isinstance(members, list) or not members:
            _fail()
        seen = set()
        for member in members:
            if not isinstance(member, str) or not member or member != member.strip():
                _fail()
            if member in seen:
                _fail()
            if member not in case_ids:
                _fail()
            seen.add(member)
        if _CLEAN_CASE_ID not in seen:
            _fail()
        families[family] = sorted(seen)
    return families


def _procedure_ids(prediction_catalog):
    procedures = prediction_catalog["bundle"]["procedures"]
    if not isinstance(procedures, list) or not procedures:
        _fail()
    ids = []
    seen = set()
    for procedure in procedures:
        procedure_id = procedure["procedure_id"]
        if not isinstance(procedure_id, str) or not procedure_id:
            _fail()
        if procedure_id in seen:
            _fail()
        seen.add(procedure_id)
        ids.append(procedure_id)
    return sorted(ids)


def _make_job(
    binding_sha256,
    stage,
    target_id,
    endpoint,
    case_id,
    disturbance_family,
    internal_dependencies,
    prediction_dependencies,
    resolution,
):
    body = {
        "record_type": "job",
        "binding_sha256": binding_sha256,
        "stage": stage,
        "target_id": target_id,
        "endpoint": endpoint,
        "case_id": case_id,
        "disturbance_family": disturbance_family,
        "depends_on_job_ids": sorted(set(internal_dependencies)),
        "depends_on_prediction_job_ids": sorted(set(prediction_dependencies)),
        "resolution": resolution,
    }
    record = dict(body)
    record["job_id"] = JOB_PREFIX + canonical_sha256(body)
    return record


def _make_alias(
    binding_sha256,
    mode,
    view_id,
    endpoint,
    case_id,
    disturbance_family,
    target_score_job_id,
):
    body = {
        "record_type": "alias",
        "binding_sha256": binding_sha256,
        "mode": mode,
        "view_id": view_id,
        "endpoint": endpoint,
        "case_id": case_id,
        "disturbance_family": disturbance_family,
        "target_score_job_id": target_score_job_id,
    }
    record = dict(body)
    record["alias_id"] = ALIAS_PREFIX + canonical_sha256(body)
    return record


def _iter_records(snapshot):
    try:
        binding_sha256 = snapshot["catalog_sha256"]
        prediction_catalog = snapshot["prediction_catalog"]
        context_views = snapshot["context_views"]
        pooled_procedures = snapshot["pooled_procedures"]
        pooled_views = snapshot["pooled_views"]

        units, parity = _index_parent(prediction_catalog)
        case_ids = sorted({case_id for (_procedure_id, case_id) in units})
        if not case_ids:
            _fail()
        families = _family_case_ids(prediction_catalog)
        family_names = sorted(families)
        procedures = _procedure_ids(prediction_catalog)

        context_case = {}
        for procedure_id in procedures:
            for endpoint in ENDPOINTS:
                for case_id in case_ids:
                    prediction_dependencies = (
                        _lookup(units, (procedure_id, case_id)),
                        _lookup(parity, procedure_id),
                    )
                    job = _make_job(
                        binding_sha256,
                        STAGE_CONTEXT_CASE,
                        procedure_id,
                        endpoint,
                        case_id,
                        _NA,
                        (),
                        prediction_dependencies,
                        RESOLUTION_CONTEXT_CASE,
                    )
                    context_case[(procedure_id, endpoint, case_id)] = job["job_id"]
                    yield job

        context_family = {}
        for procedure_id in procedures:
            for endpoint in ENDPOINTS:
                for family in family_names:
                    internal_dependencies = [
                        _lookup(context_case, (procedure_id, endpoint, case_id))
                        for case_id in families[family]
                    ]
                    job = _make_job(
                        binding_sha256,
                        STAGE_CONTEXT_FAMILY,
                        procedure_id,
                        endpoint,
                        _NA,
                        family,
                        internal_dependencies,
                        (),
                        RESOLUTION_CURVE,
                    )
                    context_family[(procedure_id, endpoint, family)] = job["job_id"]
                    yield job

        pooled_ordered = sorted(
            pooled_procedures, key=lambda procedure: procedure["pooled_procedure_id"]
        )
        pooled_case = {}
        for pooled in pooled_ordered:
            pooled_id = pooled["pooled_procedure_id"]
            members = pooled["members"]
            for endpoint in ENDPOINTS:
                for case_id in case_ids:
                    prediction_dependencies = set()
                    for member in members:
                        procedure_id = member["procedure_id"]
                        prediction_dependencies.add(_lookup(units, (procedure_id, case_id)))
                        prediction_dependencies.add(_lookup(parity, procedure_id))
                    job = _make_job(
                        binding_sha256,
                        STAGE_POOLED_CASE,
                        pooled_id,
                        endpoint,
                        case_id,
                        _NA,
                        (),
                        prediction_dependencies,
                        RESOLUTION_POOLED_CASE,
                    )
                    pooled_case[(pooled_id, endpoint, case_id)] = job["job_id"]
                    yield job

        pooled_family = {}
        for pooled in pooled_ordered:
            pooled_id = pooled["pooled_procedure_id"]
            for endpoint in ENDPOINTS:
                for family in family_names:
                    internal_dependencies = [
                        _lookup(pooled_case, (pooled_id, endpoint, case_id))
                        for case_id in families[family]
                    ]
                    job = _make_job(
                        binding_sha256,
                        STAGE_POOLED_FAMILY,
                        pooled_id,
                        endpoint,
                        _NA,
                        family,
                        internal_dependencies,
                        (),
                        RESOLUTION_CURVE,
                    )
                    pooled_family[(pooled_id, endpoint, family)] = job["job_id"]
                    yield job

        ordered_context_views = sorted(
            context_views,
            key=lambda view: view["view_id"],
        )
        for view in ordered_context_views:
            target_id = view["target_procedure_id"]
            view_id = view["view_id"]
            for endpoint in ENDPOINTS:
                for case_id in case_ids:
                    yield _make_alias(
                        binding_sha256,
                        MODE_CONTEXT_CASE,
                        view_id,
                        endpoint,
                        case_id,
                        _NA,
                        _lookup(context_case, (target_id, endpoint, case_id)),
                    )
        for view in ordered_context_views:
            target_id = view["target_procedure_id"]
            view_id = view["view_id"]
            for endpoint in ENDPOINTS:
                for family in family_names:
                    yield _make_alias(
                        binding_sha256,
                        MODE_CONTEXT_FAMILY,
                        view_id,
                        endpoint,
                        _NA,
                        family,
                        _lookup(context_family, (target_id, endpoint, family)),
                    )

        ordered_pooled_views = sorted(
            pooled_views,
            key=lambda view: view["view_id"],
        )
        for view in ordered_pooled_views:
            target_id = view["target_pooled_procedure_id"]
            view_id = view["view_id"]
            for endpoint in ENDPOINTS:
                for case_id in case_ids:
                    yield _make_alias(
                        binding_sha256,
                        MODE_POOLED_CASE,
                        view_id,
                        endpoint,
                        case_id,
                        _NA,
                        _lookup(pooled_case, (target_id, endpoint, case_id)),
                    )
        for view in ordered_pooled_views:
            target_id = view["target_pooled_procedure_id"]
            view_id = view["view_id"]
            for endpoint in ENDPOINTS:
                for family in family_names:
                    yield _make_alias(
                        binding_sha256,
                        MODE_POOLED_FAMILY,
                        view_id,
                        endpoint,
                        _NA,
                        family,
                        _lookup(pooled_family, (target_id, endpoint, family)),
                    )
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None


def iter_stress_score_records(catalog):
    """Validate ``catalog`` eagerly, then lazily stream score jobs and aliases.

    ``catalog`` is the exact score support catalog produced by
    ``build_stress_score_support``.  Validation runs eagerly against an
    independent snapshot; the returned generator expands the deterministic
    stress scoring jobs (all before the aliases) lazily.
    """
    try:
        snapshot = validate_stress_score_support(catalog)
    except ValueError:
        raise ValueError(INVALID_METADATA) from None
    except (TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError(INVALID_METADATA) from None
    return _iter_records(snapshot)
