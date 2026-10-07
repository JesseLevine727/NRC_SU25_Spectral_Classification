"""Metadata-only P08 population resource proposal.

Pure planning arithmetic over audited public aggregate ledgers. This module
performs no training, model loading, array access, I/O, measurement, execution
or provenance inference; the caller remains responsible for provenance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from atlas_sers.governance.canonical import sha256_value

_SCHEMA = "nato-sers-p08-population-resource-proposal-v1"

_POPULATIONS = ("notes_clear_500", "mira1_excluded_575")
_CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
_NEURAL_MODELS = ("D0-M", "D1", "D2", "D3")
_CONDITIONAL_NEURAL_MODELS = ("D1", "D2", "D3")
_CLASSICAL_PANEL = ("C-RANDOM-FOREST", "C-RBF-SVM")

_CLASSICAL_STAGES = ("inner_selection", "calibration_crossfit", "final_family_refit")
_NEURAL_STAGES = ("source_fit", "guard_source_fit", "final_refit")
_CLASSICAL_FIT_STAGES = ("source_fit", "calibration_model_fit", "final_refit")
_NEURAL_FIT_STAGES = ("source_fit", "guard_source_fit", "final_refit")
_CONDITIONAL_FIT_STAGES = ("source_fit", "final_refit")

_FIT_STAGES_ALL = ("source_fit", "guard_source_fit", "calibration_model_fit", "final_refit")

_ALLOWED_STAGES = (
    "source_fit",
    "source_validation_prediction",
    "guard_source_fit",
    "guard_validation_prediction",
    "calibration_model_fit",
    "calibration_validation_prediction",
    "calibration_prediction_alias",
    "final_refit",
    "held_prediction",
    "scalar_calibration",
    "seed_ensemble_prediction",
    "select_hyperparameters",
    "select_neural_recipe",
    "select_refit_epochs",
)

_MARGIN = 2
_DAY = 24 * 3600
_UNIT = 16 * 2**30
_GIB = 2**30
_RESERVE_GIB = 30

_CATALOG_ORDER = {
    ("notes_clear_500", False): 0,
    ("notes_clear_500", True): 1,
    ("mira1_excluded_575", False): 2,
    ("mira1_excluded_575", True): 3,
}

_EXPECTED_SOURCE_FITS = 12780
_EXPECTED_REFITS = 1635
_EXPECTED_TOTAL = 14415
_EXPECTED_HELD = 260
_EXPECTED_DEVELOPMENT = 60
_EXPECTED_PILOT = 36


def _map(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _seq(value, name):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence.")
    return list(value)


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer count.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _num(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _ensure_canonical(value, name):
    try:
        sha256_value(value)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError(f"{name} is not canonical JSON: {exc}") from None
    return value


def _false(container, key, name):
    if container.get(key) is not False:
        raise ValueError(f"{name}.{key} must be present and exactly false.")


def _zero(container, key, name):
    if key not in container:
        raise ValueError(f"{name}.{key} must be present.")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{name}.{key} must be exactly integer 0.")


def _argmax(values):
    best_key, best = None, None
    for key in values:
        if best is None or values[key] > best:
            best_key, best = key, values[key]
    return best_key


def _stage_id(population, include_extra_trees):
    prefix = "POP-NOTES" if population == "notes_clear_500" else "POP-MIRA"
    return f"{prefix}-{'5' if include_extra_trees else '4'}"


def _validate_no_execution_flags(slot_audit, historical_classical, historical_neural):
    _false(slot_audit, "execution_authorized", "slot_audit")
    _zero(slot_audit, "scientific_operations", "slot_audit")
    _zero(slot_audit, "scientific_fit_count", "slot_audit")
    _false(slot_audit, "arrays_loaded", "slot_audit")
    _false(slot_audit, "outcomes_loaded", "slot_audit")
    _false(slot_audit, "numerical_readiness_verified", "slot_audit")
    _zero(historical_classical, "scientific_operations_started", "historical_classical")
    _false(historical_classical, "measured_p08_timing", "historical_classical")
    _false(historical_neural, "execution_authorized", "historical_neural")
    _zero(historical_neural, "new_scientific_operations", "historical_neural")
    for key in (
        "measured_P08_costs",
        "charged_recovery_seconds_used_as_fit_durations",
        "prediction_arrays_or_model_tensors_loaded",
        "outcome_fields_used",
    ):
        _false(historical_neural, key, "historical_neural")


def _counts(value, name):
    result = {}
    for stage, count in _map(value, name).items():
        if not isinstance(stage, str) or stage not in _ALLOWED_STAGES:
            raise ValueError(f"{name} contains an unknown stage name.")
        result[stage] = _int(count, f"{name}.{stage}")
    return result


def _prepare_catalogs(slot_audit):
    catalogs = _seq(slot_audit.get("catalogs"), "slot_audit.catalogs")
    if len(catalogs) != 4:
        raise ValueError("slot_audit.catalogs must contain exactly four catalogs.")
    prepared = []
    for index, raw in enumerate(catalogs):
        prefix = f"slot_audit.catalogs[{index}]"
        catalog = _map(raw, prefix)
        population = catalog.get("population_id")
        extra = catalog.get("include_extra_trees")
        if population not in _POPULATIONS or not isinstance(extra, bool):
            raise ValueError(f"{prefix} has an unknown population or non-bool include_extra_trees.")
        if (population, extra) not in _CATALOG_ORDER:
            raise ValueError(f"{prefix} is not one of the four required catalogs.")

        summary = _map(catalog.get("summary"), f"{prefix}.summary")
        if summary.get("bounds_are_nonadditive") is not True:
            raise ValueError(f"{prefix}.summary.bounds_are_nonadditive must be true.")
        _zero(summary, "authorized_scientific_operations", f"{prefix}.summary")
        per_model = _map(summary.get("per_model"), f"{prefix}.summary.per_model")
        if per_model.get("bounds_are_nonadditive") is not True:
            raise ValueError(f"{prefix}.summary.per_model.bounds_are_nonadditive must be true.")

        classical_models = list(_CLASSICAL_PANEL)
        if extra:
            classical_models.append("C-EXTRA-TREES")
        panel_models = classical_models + list(_NEURAL_MODELS)
        lower_models = _map(per_model.get("lower"), f"{prefix}.summary.per_model.lower")
        if set(lower_models) != set(panel_models):
            raise ValueError(
                f"{prefix}.summary.per_model.lower must contain exactly the panel models."
            )

        lower = _map(summary.get("lower"), f"{prefix}.summary.lower")
        upper = _map(summary.get("upper"), f"{prefix}.summary.upper")
        catalog_counts = _counts(
            summary.get("catalog_stage_counts"), f"{prefix}.summary.catalog_stage_counts"
        )
        catalog_jobs = sum(catalog_counts.values())
        lower_counts = _counts(lower.get("stage_counts"), f"{prefix}.summary.lower.stage_counts")
        upper_counts = _counts(upper.get("stage_counts"), f"{prefix}.summary.upper.stage_counts")

        aggregate_lower_counts = {}
        fit_total = 0
        for model in panel_models:
            entry = _map(lower_models[model], f"{prefix}.summary.per_model.lower.{model}")
            model_counts = _counts(
                entry.get("stage_counts"), f"{prefix}.summary.per_model.lower.{model}.stage_counts"
            )
            if model in _NEURAL_MODELS:
                if "calibration_model_fit" in model_counts:
                    raise ValueError("calibration_model_fit is forbidden for neural models.")
                fit_stages = _NEURAL_FIT_STAGES
            else:
                if "guard_source_fit" in model_counts:
                    raise ValueError("guard_source_fit is forbidden for classical models.")
                fit_stages = _CLASSICAL_FIT_STAGES

            model_total_jobs = _int(
                entry.get("total_jobs"),
                f"{prefix}.summary.per_model.lower.{model}.total_jobs",
                0,
            )
            if model_total_jobs != sum(model_counts.values()):
                raise ValueError(f"{model} total_jobs must equal its stage counts.")

            model_scalar = _int(
                entry.get("scalar_calibrations"),
                f"{prefix}.summary.per_model.lower.{model}.scalar_calibrations",
                0,
            )
            if model_scalar != model_counts.get("scalar_calibration", 0):
                raise ValueError(
                    f"{model} scalar_calibrations must equal its scalar_calibration stage count."
                )

            fit_sum = sum(model_counts.get(stage, 0) for stage in fit_stages)
            model_fit_slots = _int(
                entry.get("model_fit_slots"),
                f"{prefix}.summary.per_model.lower.{model}.model_fit_slots",
            )
            if fit_sum != model_fit_slots:
                raise ValueError(f"{model} fit count must equal its fit-stage counts.")
            fit_total += model_fit_slots

            for stage, count in model_counts.items():
                aggregate_lower_counts[stage] = aggregate_lower_counts.get(stage, 0) + count

        for stage in _ALLOWED_STAGES:
            if aggregate_lower_counts.get(stage, 0) != lower_counts.get(stage, 0):
                raise ValueError(
                    f"{prefix}.summary.lower.stage_counts must equal "
                    "the aggregate per-model stage counts."
                )

        lower_fit_slots = _int(
            lower.get("model_fit_slots"), f"{prefix}.summary.lower.model_fit_slots"
        )
        if sum(lower_counts.get(stage, 0) for stage in _FIT_STAGES_ALL) != lower_fit_slots:
            raise ValueError(
                f"{prefix}.summary.lower.model_fit_slots must equal its fit-stage counts."
            )
        if fit_total != lower_fit_slots:
            raise ValueError("Per-model fit counts must sum to summary.lower.model_fit_slots.")

        upper_fit_slots = _int(
            upper.get("model_fit_slots"), f"{prefix}.summary.upper.model_fit_slots"
        )
        if sum(upper_counts.get(stage, 0) for stage in _FIT_STAGES_ALL) != upper_fit_slots:
            raise ValueError(
                f"{prefix}.summary.upper.model_fit_slots must equal its fit-stage counts."
            )

        scalar_lower = _int(
            lower.get("scalar_calibrations"), f"{prefix}.summary.lower.scalar_calibrations"
        )
        if scalar_lower != lower_counts.get("scalar_calibration", 0):
            raise ValueError(
                f"{prefix}.summary.lower.scalar_calibrations must equal "
                "its scalar_calibration stage count."
            )
        scalar_upper = _int(
            upper.get("scalar_calibrations"), f"{prefix}.summary.upper.scalar_calibrations"
        )
        if scalar_upper != upper_counts.get("scalar_calibration", 0):
            raise ValueError(
                f"{prefix}.summary.upper.scalar_calibrations must equal "
                "its scalar_calibration stage count."
            )
        if scalar_upper < scalar_lower:
            raise ValueError("scalar_calibration ceiling must be >= lower.")

        jobs_lower = _int(lower.get("total_jobs"), f"{prefix}.summary.lower.total_jobs")
        if sum(lower_counts.values()) != jobs_lower:
            raise ValueError(f"{prefix}.summary.lower.total_jobs must equal its stage counts.")
        jobs_upper = _int(upper.get("total_jobs"), f"{prefix}.summary.upper.total_jobs")
        if sum(upper_counts.values()) != jobs_upper:
            raise ValueError(f"{prefix}.summary.upper.total_jobs must equal its stage counts.")
        if jobs_upper < jobs_lower:
            raise ValueError("total_job ceiling must be >= lower.")
        if fit_total > 0 and jobs_upper <= 0:
            raise ValueError("total_job ceiling must be positive when any fits are present.")

        for stage in _ALLOWED_STAGES:
            if upper_counts.get(stage, 0) < lower_counts.get(stage, 0):
                raise ValueError(f"{prefix} upper stage counts must be >= lower stage counts.")
            if catalog_counts.get(stage, 0) < upper_counts.get(stage, 0):
                raise ValueError(f"{prefix} catalog stage counts must be >= upper stage counts.")
        if catalog_jobs < jobs_upper:
            raise ValueError(f"{prefix} catalog total jobs must be >= upper total jobs.")

        deltas = {}
        for stage in _CONDITIONAL_FIT_STAGES:
            delta = upper_counts.get(stage, 0) - lower_counts.get(stage, 0)
            if delta < 0:
                raise ValueError("Conditional fit deltas must be nonnegative.")
            deltas[stage] = delta
        for stage in ("guard_source_fit", "calibration_model_fit"):
            if upper_counts.get(stage, 0) != lower_counts.get(stage, 0):
                raise ValueError(f"{stage} delta must be zero for the conditional envelope.")

        if upper_fit_slots - lower_fit_slots != deltas["source_fit"] + deltas["final_refit"]:
            raise ValueError("Upper-minus-lower fit slots must equal source plus refit deltas.")

        prepared.append(
            {
                "population_id": population,
                "include_extra_trees": extra,
                "classical_models": classical_models,
                "panel_models": panel_models,
                "lower_models": lower_models,
                "deltas": deltas,
                "upper_model_fit_slots": upper_fit_slots,
                "scalar_calibration_ceiling": scalar_upper,
                "total_job_ceiling": jobs_upper,
                "total_catalog_jobs": catalog_jobs,
            }
        )

    if {(item["population_id"], item["include_extra_trees"]) for item in prepared} != set(
        _CATALOG_ORDER
    ):
        raise ValueError(
            "Catalogs must cover both populations and both include_extra_trees values."
        )
    prepared.sort(
        key=lambda item: _CATALOG_ORDER[(item["population_id"], item["include_extra_trees"])]
    )
    return prepared


def _parse_classical(historical_classical):
    rows = _seq(historical_classical.get("classical"), "historical_classical.classical")
    if len(rows) != 9:
        raise ValueError("historical_classical.classical must contain exactly nine rows.")
    rates, sizes = {}, {}
    for index, raw in enumerate(rows):
        name = f"historical_classical.classical[{index}]"
        row = _map(raw, name)
        model, stage = row.get("model"), row.get("stage")
        if model not in _CLASSICAL_MODELS or stage not in _CLASSICAL_STAGES:
            raise ValueError(f"{name} has an unknown model or stage.")
        key = (model, stage)
        if key in rates:
            raise ValueError(f"{name} duplicates a classical model/stage pair.")
        seconds = _num(row.get("seconds"), f"{name}.seconds")
        recorded = _int(row.get("seconds_recorded"), f"{name}.seconds_recorded", 1)
        slots = _int(row.get("slots"), f"{name}.slots")
        reuse = _int(row.get("cache_reuse_verified"), f"{name}.cache_reuse_verified")
        size_sum = _num(
            row.get("serialized_model_size_bytes_sum"),
            f"{name}.serialized_model_size_bytes_sum",
            positive=True,
        )
        size_recorded = _int(row.get("size_recorded"), f"{name}.size_recorded", 1)
        if reuse + recorded != slots:
            raise ValueError(f"{name} cache reuse plus recorded must equal slots.")
        if size_recorded != recorded:
            raise ValueError(f"{name} size_recorded must equal seconds_recorded.")
        rates[key] = seconds / recorded
        sizes[key] = size_sum / size_recorded
    if set(rates) != {(m, s) for m in _CLASSICAL_MODELS for s in _CLASSICAL_STAGES}:
        raise ValueError("historical_classical.classical is missing model/stage rows.")
    return rates, sizes


def _parse_neural(historical_neural):
    rows = _seq(historical_neural.get("neural"), "historical_neural.neural")
    if len(rows) != 12:
        raise ValueError("historical_neural.neural must contain exactly twelve rows.")
    rates, sizes = {}, {}
    source_fits = refits = reused = 0
    for index, raw in enumerate(rows):
        name = f"historical_neural.neural[{index}]"
        row = _map(raw, name)
        model, stage = row.get("model"), row.get("stage")
        if model not in _NEURAL_MODELS or stage not in _NEURAL_STAGES:
            raise ValueError(f"{name} has an unknown model or stage.")
        key = (model, stage)
        if key in rates:
            raise ValueError(f"{name} duplicates a neural model/stage pair.")
        fits = _int(row.get("fits"), f"{name}.fits", 1)
        total = _num(row.get("elapsed_seconds_sum"), f"{name}.elapsed_seconds_sum", positive=True)
        mean = _num(row.get("elapsed_seconds_mean"), f"{name}.elapsed_seconds_mean", positive=True)
        computed = total / fits
        if abs(mean - computed) > 1e-12 + 1e-12 * abs(mean):
            raise ValueError(f"{name} stored mean does not match sum/fits.")
        byte_max = _int(
            row.get("retained_checkpoint_bytes_max"), f"{name}.retained_checkpoint_bytes_max"
        )
        byte_mean = _num(
            row.get("retained_checkpoint_bytes_mean"), f"{name}.retained_checkpoint_bytes_mean"
        )
        byte_sum = _int(
            row.get("retained_checkpoint_bytes_sum"), f"{name}.retained_checkpoint_bytes_sum"
        )
        computed_bytes_mean = byte_sum / fits
        if abs(byte_mean - computed_bytes_mean) > 1e-12 + 1e-12 * abs(byte_mean):
            raise ValueError(f"{name} stored retained checkpoint mean does not match sum/fits.")
        if byte_max > byte_sum or byte_sum > byte_max * fits:
            raise ValueError(f"{name} retained checkpoint size statistics are inconsistent.")
        reused_pilot_fits = _int(row.get("reused_pilot_fits"), f"{name}.reused_pilot_fits")
        if reused_pilot_fits > fits:
            raise ValueError(f"{name}.reused_pilot_fits must not exceed fits.")
        if stage == "final_refit" and reused_pilot_fits != 0:
            raise ValueError(f"{name}.reused_pilot_fits must be zero for final_refit rows.")
        reused += reused_pilot_fits
        if stage == "final_refit":
            refits += fits
        else:
            source_fits += fits
        rates[key] = computed
        sizes[key] = byte_max
    if set(rates) != {(m, s) for m in _NEURAL_MODELS for s in _NEURAL_STAGES}:
        raise ValueError("historical_neural.neural is missing model/stage rows.")

    declared_source = _int(
        historical_neural.get("historical_source_fits"), "historical_neural.historical_source_fits"
    )
    declared_refits = _int(
        historical_neural.get("historical_refits"), "historical_neural.historical_refits"
    )
    declared_total = _int(
        historical_neural.get("successful_record_count"),
        "historical_neural.successful_record_count",
    )
    if source_fits != declared_source or source_fits != _EXPECTED_SOURCE_FITS:
        raise ValueError("historical source fit counts are inconsistent.")
    if refits != declared_refits or refits != _EXPECTED_REFITS:
        raise ValueError("historical refit counts are inconsistent.")
    if declared_source + declared_refits != declared_total or declared_total != _EXPECTED_TOTAL:
        raise ValueError("historical total record count is inconsistent.")
    if (
        _int(historical_neural.get("held_contexts"), "historical_neural.held_contexts")
        != _EXPECTED_HELD
    ):
        raise ValueError("historical held context count is inconsistent.")
    if (
        _int(
            historical_neural.get("development_contexts_excluded"),
            "historical_neural.development_contexts_excluded",
        )
        != _EXPECTED_DEVELOPMENT
    ):
        raise ValueError("historical development exclusion count is inconsistent.")
    if (
        _int(
            historical_neural.get("historical_pilot_fits_reused_not_added"),
            "historical_neural.historical_pilot_fits_reused_not_added",
        )
        != _EXPECTED_PILOT
    ):
        raise ValueError("historical pilot reuse count is inconsistent.")
    if reused != _EXPECTED_PILOT:
        raise ValueError("per-row reused pilot fits must sum to the declared pilot reuse count.")
    return rates, sizes


def _conditional_maxima(neural_rates, neural_sizes):
    maxima = {}
    for stage in _CONDITIONAL_FIT_STAGES:
        rates = {m: neural_rates[(m, stage)] for m in _CONDITIONAL_NEURAL_MODELS}
        sizes = {m: neural_sizes[(m, stage)] for m in _CONDITIONAL_NEURAL_MODELS}
        rate_recipe = _argmax(rates)
        size_recipe = _argmax(sizes)
        maxima[stage] = {
            "max_mean_seconds": rates[rate_recipe],
            "max_mean_seconds_recipe": rate_recipe,
            "max_checkpoint_bytes": sizes[size_recipe],
            "max_checkpoint_bytes_recipe": size_recipe,
        }
    return maxima


def _build_proposal(catalog, classical_rates, classical_sizes, neural_rates, neural_sizes):
    lower_models = catalog["lower_models"]
    classical_seconds = 0.0
    classical_bytes = 0.0
    neural_base_seconds = 0.0
    neural_base_bytes = 0.0
    components = {}

    for model in catalog["classical_models"]:
        counts = _map(lower_models[model].get("stage_counts"), f"lower.{model}.stage_counts")
        source_count = counts.get("source_fit", 0)
        calibration_count = counts.get("calibration_model_fit", 0)
        refit_count = counts.get("final_refit", 0)
        seconds = (
            source_count * classical_rates[(model, "inner_selection")]
            + calibration_count * classical_rates[(model, "calibration_crossfit")]
            + refit_count * classical_rates[(model, "final_family_refit")]
        )
        retained = refit_count * classical_sizes[(model, "final_family_refit")]
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(
                f"lower.{model} classical kernel seconds are not finite and nonnegative."
            )
        if not math.isfinite(retained) or retained < 0:
            raise ValueError(
                f"lower.{model} classical retained bytes are not finite and nonnegative."
            )
        classical_seconds += seconds
        classical_bytes += retained
        components[model] = {
            "family": "classical",
            "source_fit_count": source_count,
            "calibration_model_fit_count": calibration_count,
            "final_refit_count": refit_count,
            "kernel_seconds": seconds,
            "retained_estimator_bytes": retained,
        }

    for model in _NEURAL_MODELS:
        counts = _map(lower_models[model].get("stage_counts"), f"lower.{model}.stage_counts")
        source_count = counts.get("source_fit", 0)
        guard_count = counts.get("guard_source_fit", 0)
        refit_count = counts.get("final_refit", 0)
        seconds = (
            source_count * neural_rates[(model, "source_fit")]
            + guard_count * neural_rates[(model, "guard_source_fit")]
            + refit_count * neural_rates[(model, "final_refit")]
        )
        retained = (
            source_count * neural_sizes[(model, "source_fit")]
            + guard_count * neural_sizes[(model, "guard_source_fit")]
            + refit_count * neural_sizes[(model, "final_refit")]
        )
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"lower.{model} neural kernel seconds are not finite and nonnegative.")
        if not math.isfinite(retained) or retained < 0:
            raise ValueError(f"lower.{model} neural retained bytes are not finite and nonnegative.")
        neural_base_seconds += seconds
        neural_base_bytes += retained
        components[model] = {
            "family": "neural",
            "source_fit_count": source_count,
            "guard_source_fit_count": guard_count,
            "final_refit_count": refit_count,
            "kernel_seconds": seconds,
            "retained_checkpoint_bytes": retained,
        }

    maxima = _conditional_maxima(neural_rates, neural_sizes)
    delta_source = catalog["deltas"]["source_fit"]
    delta_final = catalog["deltas"]["final_refit"]
    conditional_seconds = (
        delta_source * maxima["source_fit"]["max_mean_seconds"]
        + delta_final * maxima["final_refit"]["max_mean_seconds"]
    )
    conditional_bytes = (
        delta_source * maxima["source_fit"]["max_checkpoint_bytes"]
        + delta_final * maxima["final_refit"]["max_checkpoint_bytes"]
    )
    neural_envelope_seconds = neural_base_seconds + conditional_seconds
    serial_seconds = classical_seconds + neural_envelope_seconds
    storage_bytes = classical_bytes + neural_base_bytes + conditional_bytes
    if not math.isfinite(serial_seconds) or serial_seconds <= 0:
        raise ValueError("serial kernel seconds envelope must be finite and strictly positive.")
    if not math.isfinite(storage_bytes) or storage_bytes <= 0:
        raise ValueError("component storage bytes envelope must be finite and strictly positive.")
    walltime_margin_seconds = _MARGIN * serial_seconds
    storage_margin_bytes = _MARGIN * storage_bytes
    if not math.isfinite(walltime_margin_seconds) or not math.isfinite(storage_margin_bytes):
        raise ValueError("resource margin products must be finite.")
    walltime_seconds = math.ceil(walltime_margin_seconds / _DAY) * _DAY
    allowance_bytes = math.ceil(storage_margin_bytes / _UNIT) * _UNIT
    starting_free_bytes = allowance_bytes + _RESERVE_GIB * _GIB

    return {
        "stage_id": _stage_id(catalog["population_id"], catalog["include_extra_trees"]),
        "population_id": catalog["population_id"],
        "include_extra_trees": catalog["include_extra_trees"],
        "classical_models": list(catalog["classical_models"]),
        "neural_models": list(_NEURAL_MODELS),
        "accounted_fit_models": list(catalog["panel_models"]),
        "panel_alternatives_mutually_exclusive_within_population": True,
        "selected_panel": None,
        "resource_proposal_approved": False,
        "auto_launch": False,
        "logical_neural_strategies_in_panel": 2,
        "neural_recipes_charged_in_fresh_source_selection": 4,
        "classical_seconds": classical_seconds,
        "neural_base_seconds": neural_base_seconds,
        "neural_conditional_envelope_seconds": conditional_seconds,
        "neural_envelope_seconds": neural_envelope_seconds,
        "serial_kernel_seconds_envelope": serial_seconds,
        "classical_retained_estimator_bytes": classical_bytes,
        "neural_retained_checkpoint_bytes": neural_base_bytes,
        "conditional_storage_bytes_envelope": conditional_bytes,
        "component_storage_bytes_envelope": storage_bytes,
        "per_model_components": components,
        "conditional_envelope": {
            "source_fit": {
                "count": delta_source,
                "max_mean_seconds": maxima["source_fit"]["max_mean_seconds"],
                "max_mean_seconds_recipe": maxima["source_fit"]["max_mean_seconds_recipe"],
                "max_checkpoint_bytes": maxima["source_fit"]["max_checkpoint_bytes"],
                "max_checkpoint_bytes_recipe": maxima["source_fit"]["max_checkpoint_bytes_recipe"],
            },
            "final_refit": {
                "count": delta_final,
                "max_mean_seconds": maxima["final_refit"]["max_mean_seconds"],
                "max_mean_seconds_recipe": maxima["final_refit"]["max_mean_seconds_recipe"],
                "max_checkpoint_bytes": maxima["final_refit"]["max_checkpoint_bytes"],
                "max_checkpoint_bytes_recipe": maxima["final_refit"]["max_checkpoint_bytes_recipe"],
            },
            "guard_source_fit": {"count": 0},
            "calibration_model_fit": {"count": 0},
            "is_conservative_historical_rate_envelope": True,
            "may_combine_recipe_maxima_across_stages": True,
            "is_future_runtime_upper_bound": False,
            "single_selection_map_realizability_asserted": False,
        },
        "active_walltime_ceiling_seconds": walltime_seconds,
        "new_artifact_allowance_bytes": allowance_bytes,
        "starting_free_space_requirement_bytes": starting_free_bytes,
        "active_walltime_formula": {
            "serial_kernel_seconds_envelope": serial_seconds,
            "margin_factor": _MARGIN,
            "period_seconds": _DAY,
            "periods_rounded_up": walltime_seconds // _DAY,
            "result_seconds": walltime_seconds,
        },
        "new_artifact_allowance_formula": {
            "component_storage_bytes_envelope": storage_bytes,
            "margin_factor": _MARGIN,
            "allocation_unit_bytes": _UNIT,
            "allocation_units_rounded_up": allowance_bytes // _UNIT,
            "result_bytes": allowance_bytes,
        },
        "model_fit_ceiling": catalog["upper_model_fit_slots"],
        "scalar_calibration_ceiling": catalog["scalar_calibration_ceiling"],
        "total_job_ceiling": catalog["total_job_ceiling"],
        "total_catalog_jobs": catalog["total_catalog_jobs"],
        "per_stage_resources": {
            "ram_gib": 24,
            "allocated_gpu_gib": 8,
            "max_single_thread_cpu_workers": 4,
            "max_gpu_workers": 1,
            "free_space_reserve_gib": _RESERVE_GIB,
            "starting_free_space_requirement_bytes": starting_free_bytes,
        },
        "margin_factors_are_supervisor_planning_choices": True,
        "margin_factors_are_statistical_bounds": False,
        "completion_guaranteed": False,
        "estimated_parallel_walltime_provided": False,
        "automatic_retries": 0,
        "evidence_deletion_permitted": False,
        "prior_permits_transferable": False,
        "failures_and_interruptions_consume_allowances": True,
        "serial_sum_is_walltime_prediction": False,
        "serial_sum_is_gpu_profile_time": False,
        "excluded_from_serial_sum": [
            "external_calibration",
            "io",
            "orchestration",
            "persistence",
            "reporting",
            "new_data_or_stopping_changes",
        ],
        "historical_source_kernels_include_internal_validation": True,
        "classical_grid_and_calibration_estimators_retained_by_default": False,
    }


def build_population_resource_proposal(*, slot_audit, historical_classical, historical_neural):
    """Return a deterministic, JSON-safe, metadata-only resource proposal.

    All three inputs are audited public aggregate mappings. No arrays, models,
    checkpoints, predictions or histories are loaded and no input is mutated.
    """

    _ensure_canonical(slot_audit, "slot_audit")
    _ensure_canonical(historical_classical, "historical_classical")
    _ensure_canonical(historical_neural, "historical_neural")

    slot_audit = _map(slot_audit, "slot_audit")
    historical_classical = _map(historical_classical, "historical_classical")
    historical_neural = _map(historical_neural, "historical_neural")

    _validate_no_execution_flags(slot_audit, historical_classical, historical_neural)
    try:
        catalogs = _prepare_catalogs(slot_audit)
        classical_rates, classical_sizes = _parse_classical(historical_classical)
        neural_rates, neural_sizes = _parse_neural(historical_neural)

        proposals = [
            _build_proposal(catalog, classical_rates, classical_sizes, neural_rates, neural_sizes)
            for catalog in catalogs
        ]
    except OverflowError as exc:
        raise ValueError(
            "population resource arithmetic overflowed the supported numeric range."
        ) from exc

    result = {
        "schema_version": _SCHEMA,
        "execution_authorized": False,
        "resource_proposal_approved": False,
        "scientific_operations": 0,
        "measured_new_population_costs": False,
        "input_provenance_independently_verified": False,
        "actual_arrays_loaded": False,
        "serialized_estimator_sizes_are_in_memory_estimates_not_persisted_checkpoints": True,
        "resources_live_inspected": False,
        "capacities_reserved": False,
        "input_canonical_sha256": {
            "slot_audit": sha256_value(slot_audit),
            "historical_classical": sha256_value(historical_classical),
            "historical_neural": sha256_value(historical_neural),
        },
        "proposals": proposals,
    }
    result["report_sha256"] = sha256_value(result)
    return result
