"""Public aggregate semantic tables for P08 figures F02/F03/F04.

This module is a bounded, pure in-memory adapter over the accepted
``p08_universal_analysis.analyze_panel`` result.  It whitelists approved
aggregate fields only, never serialises the complete input, never copies
private master/observation/context cells, draw vectors or arbitrary nested
objects, and never scores, fits, calibrates, resamples, ranks or selects a
model or policy.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_universal_analysis as _analysis

__all__ = ["prepare_model_figures"]

SCHEMA_VERSION = "nato-sers-p08-model-figure-data-v1"
RESEARCH_QUESTION = "RQ-S01"
EXPECTED_DOMAINS = 13

_POLICY_COEFFICIENTS = (1.0, -1.0)
_INTERACTION_COEFFICIENTS = (1.0, -1.0, -1.0, 1.0)

_WEIGHTED_MODES = ("crossed", "master_only", "instrument_only")

_COUNT_COLUMNS = (
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
)

_METRIC_REQUIRED_COLUMNS = (
    "model_id",
    "domain",
    "station",
    "instrument",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
    "balanced_accuracy",
)

_M01_LABEL = "Individual-spectrum predictions."
_M06_LABEL = (
    "Combined predictions per sample: mean model probabilities within "
    "master/instrument, then equal instrument mean; never mean input spectra "
    "or hard labels."
)
_POLICY_LABEL_MIN = "minimally processed minmax"
_POLICY_LABEL_SG = "impulse replacement + smoothing + minmax"
_POLICY_LABEL_ARPLS = "impulse replacement + baseline correction + minmax"

_BOUNDARY_FLAGS = {
    "conditional_on_saved_fits_observed_support": True,
    "descriptive_sign_symmetry_not_randomized": True,
    "no_G4_decision": True,
    "no_model_or_policy_selection": True,
}

_POPULATION = {
    "primary_spectra": 598,
    "held_spectra": 557,
    "masters": 69,
    "instruments": 10,
    "held_domains": 13,
    "contexts": 260,
}

_INTERVAL_CAPTION = (
    "Intervals are 95% marginal conditional percentile intervals; they do not "
    "represent retraining or new-instrument uncertainty, do not remove causal "
    "nuisance effects, are not a G4 decision and do not select a held winner."
)
_INTERACTION_CAPTION = (
    "A positive interaction means preprocessing helped the deep model more "
    "(or hurt it less) than the comparator; it does not mean the deep model "
    "has higher absolute accuracy."
)


# ---------------------------------------------------------------------------
# Small typed helpers
# ---------------------------------------------------------------------------


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_real(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_count(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
    )


def _finite_required(value, name):
    _require(_is_real(value) and math.isfinite(float(value)), f"{name} must be finite")
    return float(value)


def _count_or_none(value, name):
    if value is None:
        return None
    _require(_is_count(value), f"{name} must be a non-negative integer")
    return int(value)


def _text_or_none(value, name):
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    raise ValueError(f"{name} has an unsupported type")


def _number_or_none(value, name):
    if value is None:
        return None
    _require(_is_real(value), f"{name} must be numeric or None")
    number = float(value)
    return number if math.isfinite(number) else None


def _close(left, right, label):
    scale = max(1.0, abs(left), abs(right))
    _require(abs(left - right) <= 1e-12 * scale, f"{label} mismatch")


def _aggregate_map(value, name):
    """Only a single finite p-value is approved, never a nested payload."""
    if value is None:
        return None
    number = _finite_required(value, name)
    _require(0 <= number <= 1, f"{name} outside [0,1]")
    return number


# ---------------------------------------------------------------------------
# Registry and declared dimensions
# ---------------------------------------------------------------------------


def _validated_registry():
    registry = list(_analysis.contrast_registry())
    _require(len(registry) == 52, "contrast registry must contain 52 entries")
    policy_entries = [
        entry
        for entry in registry
        if entry["family_id"] == _analysis.FAMILY_UNIVERSAL_POLICY
    ]
    interaction_entries = [
        entry
        for entry in registry
        if entry["family_id"] == _analysis.FAMILY_POLICY_MODEL_INTERACTION
    ]
    available = [entry for entry in interaction_entries if entry["available"]]
    unavailable = [entry for entry in interaction_entries if not entry["available"]]
    _require(len(policy_entries) == 20, "registry needs 20 universal policy effects")
    _require(len(available) == 24, "registry needs 24 available interactions")
    _require(len(unavailable) == 8, "registry needs 8 future-QC interactions")
    identifiers = [entry["contrast_id"] for entry in registry]
    _require(len(identifiers) == len(set(identifiers)), "registry ids must be unique")
    return (
        tuple(registry),
        tuple(policy_entries),
        tuple(available),
        tuple(unavailable),
    )


def _estimands():
    estimands = tuple(_analysis.ESTIMANDS)
    _require(len(estimands) == 2, "exactly two estimands are required")
    return estimands


def _policies(policy_entries):
    reference = policy_entries[0]["procedure_labels"][1][0]
    nonminimal = tuple(
        dict.fromkeys(entry["procedure_labels"][0][0] for entry in policy_entries)
    )
    derived = (reference,) + nonminimal
    declared = getattr(_analysis, "POLICIES", None)
    if declared is None:
        return derived
    declared = tuple(declared)
    _require(set(declared) == set(derived), "declared policies do not match registry")
    return declared


def _models(policy_entries):
    derived = []
    for entry in policy_entries:
        for label in entry["procedure_labels"]:
            if label[1] not in derived:
                derived.append(label[1])
    declared = getattr(_analysis, "MODELS", None)
    if declared is None:
        return tuple(derived)
    declared = tuple(declared)
    _require(set(declared) == set(derived), "declared models do not match registry")
    return declared


def _endpoints(policy_entries):
    derived = []
    for entry in policy_entries:
        if entry["endpoint"] not in derived:
            derived.append(entry["endpoint"])
    declared = getattr(_analysis, "ENDPOINTS", None)
    if declared is None:
        return tuple(derived)
    declared = tuple(declared)
    _require(set(declared) == set(derived), "declared endpoints do not match registry")
    return declared


def _coefficients_for(entry):
    if entry["family_id"] == _analysis.FAMILY_UNIVERSAL_POLICY:
        return _POLICY_COEFFICIENTS
    return _INTERACTION_COEFFICIENTS


def _endpoint_label(endpoint):
    text = str(endpoint)
    if "M01" in text:
        return _M01_LABEL
    if "M06" in text:
        return _M06_LABEL
    return text


def _policy_label(policy_id):
    text = str(policy_id)
    if "ARPLS" in text:
        return _POLICY_LABEL_ARPLS
    if "SG" in text:
        return _POLICY_LABEL_SG
    if "MIN" in text:
        return _POLICY_LABEL_MIN
    return text


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_metric_frame(frame, models, estimand, policy, endpoint):
    _require(
        not frame.empty,
        f"empty domain_metrics for {estimand}/{policy}/{endpoint}",
    )
    missing = [
        column for column in _METRIC_REQUIRED_COLUMNS if column not in frame.columns
    ]
    _require(not missing, f"domain_metrics missing columns: {missing}")
    _require(
        not frame.duplicated(["model_id", "domain"]).any(),
        f"duplicate model/domain rows for {estimand}/{policy}/{endpoint}",
    )
    _require(
        set(frame["model_id"].astype(str)) == {str(model) for model in models},
        f"model mismatch for {estimand}/{policy}/{endpoint}",
    )
    domains = set()
    for record in frame.itertuples(index=False):
        domain = record.domain
        _require(
            isinstance(domain, str) and domain,
            f"domain id must be a non-empty string for {estimand}/{policy}/{endpoint}",
        )
        domains.add(str(domain))
        value = record.balanced_accuracy
        _require(
            _is_real(value)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"balanced_accuracy out of range for {domain}",
        )
        for column in _COUNT_COLUMNS:
            _require(
                _is_count(getattr(record, column)) and int(getattr(record, column)) > 0,
                f"{column} must be a non-negative integer for {domain}",
            )
        _require(isinstance(record.station, str), f"station missing for {domain}")
        _require(isinstance(record.instrument, str), f"instrument missing for {domain}")
    _require(len(frame) == len(models) * len(domains), "incomplete model/domain grid")
    return domains


def _validate_metrics(result, estimands, policies, endpoints, models):
    metrics = result.get("metrics")
    _require(isinstance(metrics, dict), "analysis metrics must be a mapping")
    _require(set(metrics) == set(estimands), "analysis metrics estimand mismatch")
    ba_index = {}
    domain_meta = {}
    table_domains = []
    for estimand in estimands:
        per_estimand = metrics[estimand]
        _require(
            isinstance(per_estimand, dict),
            f"metrics[{estimand}] must be a mapping",
        )
        _require(
            set(per_estimand) == set(policies),
            f"metrics[{estimand}] policy mismatch",
        )
        for policy in policies:
            per_policy = per_estimand[policy]
            _require(
                isinstance(per_policy, dict),
                f"metrics[{estimand}][{policy}] must be a mapping",
            )
            _require(
                set(per_policy) == set(endpoints),
                f"metrics[{estimand}][{policy}] endpoint mismatch",
            )
            for endpoint in endpoints:
                subtable = per_policy[endpoint]
                _require(
                    isinstance(subtable, dict),
                    f"metrics[{estimand}][{policy}][{endpoint}] must be a mapping",
                )
                frame = subtable.get("domain_metrics")
                _require(
                    isinstance(frame, pd.DataFrame),
                    f"domain_metrics for {estimand}/{policy}/{endpoint} must be a DataFrame",
                )
                domains_here = _validate_metric_frame(
                    frame, models, estimand, policy, endpoint
                )
                table_domains.append(domains_here)
                for record in frame.itertuples(index=False):
                    domain = str(record.domain)
                    ba_index[
                        (
                            estimand,
                            policy,
                            endpoint,
                            str(record.model_id),
                            domain,
                        )
                    ] = float(record.balanced_accuracy)
                    meta = {
                        "station": str(record.station),
                        "instrument": str(record.instrument),
                        "contexts": int(record.contexts),
                        "unit_appearances": int(record.unit_appearances),
                        "physical_masters": int(record.physical_masters),
                        "distinct_units": int(record.distinct_units),
                    }
                    key = (estimand, endpoint, domain)
                    if key in domain_meta:
                        _require(
                            domain_meta[key] == meta,
                            f"inconsistent support metadata for {key}",
                        )
                    else:
                        domain_meta[key] = meta
    union = set().union(*table_domains) if table_domains else set()
    _require(union, "analysis produced no domains")
    for domains_here in table_domains:
        _require(
            domains_here == union,
            "domain coverage differs across metric tables",
        )
    expected = EXPECTED_DOMAINS
    _require(len(union) == expected, f"expected {expected} domains, found {len(union)}")
    identities = {}
    for (_, _, domain), meta in domain_meta.items():
        identity = (meta["station"], meta["instrument"])
        _require(
            domain not in identities or identities[domain] == identity,
            "domain station/instrument changes across estimands or endpoints",
        )
        identities[domain] = identity
    return ba_index, domain_meta, tuple(sorted(union))


def _validate_contrasts(result, registry, estimands):
    contrasts = result.get("contrasts")
    _require(isinstance(contrasts, dict), "analysis contrasts must be a mapping")
    _require(set(contrasts) == set(estimands), "analysis contrasts estimand mismatch")
    identifiers = [entry["contrast_id"] for entry in registry]
    for estimand in estimands:
        table = contrasts[estimand]
        _require(
            isinstance(table, dict),
            f"contrasts[{estimand}] must be a mapping",
        )
        _require(
            set(table) == set(identifiers),
            f"contrasts[{estimand}] does not match the registry",
        )
    return contrasts


# ---------------------------------------------------------------------------
# Approved contrast extraction
# ---------------------------------------------------------------------------


def _domain_effects(point, domains, label):
    effects = point.get("domain_effects")
    _require(
        isinstance(effects, dict),
        f"{label} point.domain_effects must be a mapping",
    )
    _require(
        set(effects) == set(domains),
        f"{label} point.domain_effects domain mismatch",
    )
    return {
        domain: _finite_required(effects[domain], f"{label} domain effect {domain}")
        for domain in domains
    }


def _procedure_effects(point, count, label):
    values = point.get("procedure_effects")
    _require(
        isinstance(values, (list, tuple)) and len(values) == count,
        f"{label} procedure_effects length mismatch",
    )
    return [_finite_required(value, f"{label} procedure_effects") for value in values]


def _procedure_domain_effects(point, domains, count, label):
    blocks = point.get("procedure_domain_effects")
    _require(
        isinstance(blocks, (list, tuple)) and len(blocks) == count,
        f"{label} procedure_domain_effects length mismatch",
    )
    copied = []
    for index, block in enumerate(blocks):
        _require(
            isinstance(block, dict) and set(block) == set(domains),
            f"{label} procedure_domain_effects[{index}] domain mismatch",
        )
        copied.append(
            {
                domain: _finite_required(
                    block[domain],
                    f"{label} procedure_domain_effects[{index}][{domain}]",
                )
                for domain in domains
            }
        )
    return copied


def _match_combination(overall, values, coefficients, label):
    expected = math.fsum(
        coefficient * value for coefficient, value in zip(coefficients, values, strict=True)
    )
    _close(overall, expected, label)


def _interval(summary, label):
    lower = summary.get("lower")
    upper = summary.get("upper")
    if lower is None and upper is None:
        return None, None
    lower_value = _finite_required(lower, f"{label} lower")
    upper_value = _finite_required(upper, f"{label} upper")
    _require(lower_value <= upper_value, f"{label} lower exceeds upper")
    return lower_value, upper_value


def _weighted_intervals(result, label):
    weighted = result.get("weighted")
    _require(isinstance(weighted, dict), f"{label} missing weighted intervals")
    intervals = {}
    for mode in _WEIGHTED_MODES:
        _require(mode in weighted, f"{label} missing weighted mode {mode}")
        block = weighted[mode]
        _require(
            isinstance(block, dict) and "overall" in block,
            f"{label} weighted {mode} missing overall",
        )
        summary = block["overall"].get("summary")
        _require(
            isinstance(summary, dict),
            f"{label} weighted {mode} missing summary",
        )
        lower, upper = _interval(summary, f"{label} weighted {mode}")
        intervals[mode] = {
            "lower": lower,
            "upper": upper,
            "reason_code": _text_or_none(
                summary.get("reason_code"), f"{label} weighted {mode} reason"
            ),
        }
    return intervals


def _hierarchy(result, label):
    hierarchy = result.get("hierarchy")
    _require(
        isinstance(hierarchy, dict) and "combined" in hierarchy,
        f"{label} missing hierarchy combined",
    )
    summary = hierarchy["combined"].get("summary")
    _require(isinstance(summary, dict), f"{label} missing hierarchy summary")
    lower, upper = _interval(summary, f"{label} hierarchy")
    return {
        "planned": _count_or_none(
            summary.get("planned_draws"), f"{label} planned_draws"
        ),
        "defined": _count_or_none(
            summary.get("defined_draws"), f"{label} defined_draws"
        ),
        "undefined": _count_or_none(
            summary.get("undefined_draws"), f"{label} undefined_draws"
        ),
        "lower": lower,
        "upper": upper,
        "reason_code": _text_or_none(
            summary.get("reason_code"), f"{label} hierarchy reason"
        ),
    }


def _adjustment(result):
    adjustment = result.get("sensitivity_adjustment") or {}
    _require(
        isinstance(adjustment, dict),
        "sensitivity_adjustment must be a mapping",
    )
    size = adjustment.get("family_size")
    if size is None:
        size = adjustment.get("size")
    return {
        "family_size": _count_or_none(size, "family_size"),
        "adjustment": _text_or_none(adjustment.get("adjustment"), "adjustment"),
        "domain_raw_p": _aggregate_map(adjustment.get("domain_raw_p"), "domain_raw_p"),
        "domain_adjusted_p": _aggregate_map(
            adjustment.get("domain_adjusted_p"), "domain_adjusted_p"
        ),
        "instrument_raw_p": _aggregate_map(
            adjustment.get("instrument_raw_p"), "instrument_raw_p"
        ),
        "instrument_adjusted_p": _aggregate_map(
            adjustment.get("instrument_adjusted_p"), "instrument_adjusted_p"
        ),
    }


def _extract_contrast(result, domains, coefficients, label):
    _require(isinstance(result, dict), f"{label} contrast must be a mapping")
    point = result.get("point")
    _require(isinstance(point, dict), f"{label} contrast missing point")
    overall = _finite_required(point.get("overall"), f"{label} point.overall")
    domain_effects = _domain_effects(point, domains, label)
    procedure_effects = _procedure_effects(point, len(coefficients), label)
    procedure_domains = _procedure_domain_effects(
        point, domains, len(coefficients), label
    )
    _match_combination(overall, procedure_effects, coefficients, f"{label} overall")
    for domain in domains:
        expected = math.fsum(
            coefficient * procedure_domains[index][domain]
            for index, coefficient in enumerate(coefficients)
        )
        _close(domain_effects[domain], expected, f"{label} domain {domain}")
    return {
        "overall": overall,
        "domain_effects": domain_effects,
        "procedure_effects": procedure_effects,
        "procedure_domains": procedure_domains,
        "intervals": _weighted_intervals(result, label),
        "hierarchy": _hierarchy(result, label),
        "adjustment": _adjustment(result),
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _build_f02(policy_entries, estimands, domains, ba_index, domain_meta, extracted):
    rows = []
    for estimand in estimands:
        for entry in policy_entries:
            endpoint = entry["endpoint"]
            policy_id = entry["policy_id"]
            model_id = entry["model_id"]
            reference_id = entry["procedure_labels"][1][0]
            data = extracted[(estimand, entry["contrast_id"])]
            for domain in domains:
                x_value = data["procedure_domains"][1][domain]
                y_value = data["procedure_domains"][0][domain]
                effect = data["domain_effects"][domain]
                _close(
                    effect,
                    y_value - x_value,
                    f"f02 {entry['contrast_id']} {domain}",
                )
                _close(
                    x_value,
                    ba_index[(estimand, reference_id, endpoint, model_id, domain)],
                    f"f02 reference {domain}",
                )
                _close(
                    y_value,
                    ba_index[(estimand, policy_id, endpoint, model_id, domain)],
                    f"f02 policy {domain}",
                )
                meta = domain_meta[(estimand, endpoint, domain)]
                rows.append(
                    {
                        "estimand": estimand,
                        "contrast_id": entry["contrast_id"],
                        "family_id": entry["family_id"],
                        "endpoint": endpoint,
                        "model_id": model_id,
                        "policy_id": policy_id,
                        "domain": domain,
                        "station": meta["station"],
                        "instrument": meta["instrument"],
                        "contexts": meta["contexts"],
                        "unit_appearances": meta["unit_appearances"],
                        "physical_masters": meta["physical_masters"],
                        "distinct_units": meta["distinct_units"],
                        "x_balanced_accuracy": x_value,
                        "y_balanced_accuracy": y_value,
                        "effect": effect,
                    }
                )
    return rows


def _build_f03(policy_entries, estimands, domains, domain_meta, extracted):
    effects_rows = []
    domain_rows = []
    for estimand in estimands:
        for entry in policy_entries:
            endpoint = entry["endpoint"]
            data = extracted[(estimand, entry["contrast_id"])]
            adjustment = data["adjustment"]
            hierarchy = data["hierarchy"]
            row = {
                "estimand": estimand,
                "contrast_id": entry["contrast_id"],
                "family_id": entry["family_id"],
                "endpoint": endpoint,
                "model_id": entry["model_id"],
                "policy_id": entry["policy_id"],
                "available": True,
                "reason": None,
                "family_size": adjustment["family_size"],
                "adjustment": adjustment["adjustment"],
                "point_effect": data["overall"],
                "domain_raw_p": adjustment["domain_raw_p"],
                "domain_adjusted_p": adjustment["domain_adjusted_p"],
                "instrument_raw_p": adjustment["instrument_raw_p"],
                "instrument_adjusted_p": adjustment["instrument_adjusted_p"],
                "hierarchy_planned": hierarchy["planned"],
                "hierarchy_defined": hierarchy["defined"],
                "hierarchy_undefined": hierarchy["undefined"],
                "hierarchy_lower": hierarchy["lower"],
                "hierarchy_upper": hierarchy["upper"],
                "hierarchy_reason": hierarchy["reason_code"],
            }
            for mode in _WEIGHTED_MODES:
                interval = data["intervals"][mode]
                row[f"{mode}_lower"] = interval["lower"]
                row[f"{mode}_upper"] = interval["upper"]
                row[f"{mode}_reason"] = interval["reason_code"]
            effects_rows.append(row)
            for domain in domains:
                meta = domain_meta[(estimand, endpoint, domain)]
                domain_rows.append(
                    {
                        "estimand": estimand,
                        "contrast_id": entry["contrast_id"],
                        "family_id": entry["family_id"],
                        "endpoint": endpoint,
                        "model_id": entry["model_id"],
                        "policy_id": entry["policy_id"],
                        "domain": domain,
                        "station": meta["station"],
                        "instrument": meta["instrument"],
                        "point_effect": data["domain_effects"][domain],
                    }
                )
    return effects_rows, domain_rows


def _build_f04(registry, estimands, domains, domain_meta, extracted):
    interaction_entries = [
        entry
        for entry in registry
        if entry["family_id"] == _analysis.FAMILY_POLICY_MODEL_INTERACTION
    ]
    overall_rows = []
    domain_rows = []
    for estimand in estimands:
        for entry in interaction_entries:
            available = bool(entry["available"])
            procedure_labels = [
                [str(policy), str(model), str(endpoint)]
                for policy, model, endpoint in entry["procedure_labels"]
            ]
            row = {
                "estimand": estimand,
                "contrast_id": entry["contrast_id"],
                "family_id": entry["family_id"],
                "endpoint": entry["endpoint"],
                "policy_id": entry["policy_id"],
                "deep_model_id": entry["deep_model_id"],
                "comparator_model_id": entry["comparator_model_id"],
                "model_id": None,
                "available": available,
                "reason": None,
                "procedure_labels": procedure_labels,
                "family_size": None,
                "adjustment": None,
                "point_effect": None,
                "domain_raw_p": None,
                "domain_adjusted_p": None,
                "instrument_raw_p": None,
                "instrument_adjusted_p": None,
                "hierarchy_planned": None,
                "hierarchy_defined": None,
                "hierarchy_undefined": None,
                "hierarchy_lower": None,
                "hierarchy_upper": None,
                "hierarchy_reason": None,
            }
            for mode in _WEIGHTED_MODES:
                row[f"{mode}_lower"] = None
                row[f"{mode}_upper"] = None
                row[f"{mode}_reason"] = None
            if available:
                data = extracted[(estimand, entry["contrast_id"])]
                adjustment = data["adjustment"]
                hierarchy = data["hierarchy"]
                row["family_size"] = adjustment["family_size"]
                row["adjustment"] = adjustment["adjustment"]
                row["point_effect"] = data["overall"]
                row["domain_raw_p"] = adjustment["domain_raw_p"]
                row["domain_adjusted_p"] = adjustment["domain_adjusted_p"]
                row["instrument_raw_p"] = adjustment["instrument_raw_p"]
                row["instrument_adjusted_p"] = adjustment["instrument_adjusted_p"]
                row["hierarchy_planned"] = hierarchy["planned"]
                row["hierarchy_defined"] = hierarchy["defined"]
                row["hierarchy_undefined"] = hierarchy["undefined"]
                row["hierarchy_lower"] = hierarchy["lower"]
                row["hierarchy_upper"] = hierarchy["upper"]
                row["hierarchy_reason"] = hierarchy["reason_code"]
                for mode in _WEIGHTED_MODES:
                    interval = data["intervals"][mode]
                    row[f"{mode}_lower"] = interval["lower"]
                    row[f"{mode}_upper"] = interval["upper"]
                    row[f"{mode}_reason"] = interval["reason_code"]
                for domain in domains:
                    meta = domain_meta[(estimand, entry["endpoint"], domain)]
                    domain_rows.append(
                        {
                            "estimand": estimand,
                            "contrast_id": entry["contrast_id"],
                            "family_id": entry["family_id"],
                            "endpoint": entry["endpoint"],
                            "policy_id": entry["policy_id"],
                            "deep_model_id": entry["deep_model_id"],
                            "comparator_model_id": entry["comparator_model_id"],
                            "domain": domain,
                            "station": meta["station"],
                            "instrument": meta["instrument"],
                            "point_effect": data["domain_effects"][domain],
                        }
                    )
            else:
                row["reason"] = (
                    entry.get("reason") or "outside_universal_execution_scope"
                )
            overall_rows.append(row)
    return overall_rows, domain_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def prepare_model_figures(analysis):
    """Return approved aggregate figure tables for the P08 F02/F03/F04 set.

    ``analysis`` is the exact accepted ``p08_universal_analysis.analyze_panel``
    result.  The function never mutates it and only whitelists aggregate
    fields.  The returned mapping holds ``semantic``, ``semantic_sha256`` and
    ``manifest``.
    """

    _require(isinstance(analysis, dict), "analysis result must be a mapping")
    registry, policy_entries, available, unavailable = _validated_registry()
    _require(
        analysis.get("registry") == list(registry),
        "input registry differs from frozen registry",
    )
    for key in _BOUNDARY_FLAGS:
        _require(
            analysis.get("boundary", {}).get(key) is True and analysis.get(key) is True,
            f"missing analysis boundary: {key}",
        )
    estimands = _estimands()
    policies = _policies(policy_entries)
    models = _models(policy_entries)
    endpoints = _endpoints(policy_entries)

    ba_index, domain_meta, domains = _validate_metrics(
        analysis, estimands, policies, endpoints, models
    )
    contrasts = _validate_contrasts(analysis, registry, estimands)

    extracted = {}
    for estimand in estimands:
        for entry in registry:
            if not entry["available"]:
                continue
            label = entry["contrast_id"]
            extracted[(estimand, label)] = _extract_contrast(
                contrasts[estimand][label],
                domains,
                _coefficients_for(entry),
                label,
            )
            data = extracted[(estimand, label)]
            adjustment = data["adjustment"]
            family_size = (
                20 if entry["family_id"] == _analysis.FAMILY_UNIVERSAL_POLICY else 32
            )
            _require(
                adjustment["family_size"] == family_size
                and adjustment["adjustment"] == "holm",
                "multiplicity family differs from registered family",
            )
            point = contrasts[estimand][label]["point"]
            _require(
                point.get("domain_instrument")
                == {
                    d: domain_meta[(estimand, entry["endpoint"], d)]["instrument"]
                    for d in domains
                },
                "contrast instrument mapping mismatch",
            )
            for index, (policy, model, endpoint) in enumerate(
                entry["procedure_labels"]
            ):
                values = data["procedure_domains"][index]
                for domain in domains:
                    _close(
                        values[domain],
                        ba_index[(estimand, policy, endpoint, model, domain)],
                        "contrast procedure/domain score",
                    )
                _close(
                    data["procedure_effects"][index],
                    math.fsum(values.values()) / len(domains),
                    "contrast procedure equal-domain mean",
                )
            _close(
                data["overall"],
                math.fsum(data["domain_effects"].values()) / len(domains),
                "contrast equal-domain mean",
            )

    f02_pairs = _build_f02(
        policy_entries, estimands, domains, ba_index, domain_meta, extracted
    )
    f03_effects, f03_domains = _build_f03(
        policy_entries, estimands, domains, domain_meta, extracted
    )
    f04_interactions, f04_domains = _build_f04(
        registry, estimands, domains, domain_meta, extracted
    )
    for row in f03_domains + f04_domains:
        meta = domain_meta[(row["estimand"], row["endpoint"], row["domain"])]
        row.update({key: meta[key] for key in _COUNT_COLUMNS})
    for row in f04_interactions:
        row["procedure_balanced_accuracies"] = (
            list(extracted[(row["estimand"], row["contrast_id"])]["procedure_effects"])
            if row["available"]
            else None
        )
        if not row["available"]:
            row["family_size"] = 32
            row["adjustment"] = "holm"
    for row in f04_domains:
        data = extracted[(row["estimand"], row["contrast_id"])]
        row["procedure_balanced_accuracies"] = [
            values[row["domain"]] for values in data["procedure_domains"]
        ]

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "research_question": RESEARCH_QUESTION,
        "metadata": {
            "independent_unit": "physical_master",
            "population": dict(_POPULATION),
            "counts_reference": "original_population_not_independent_replicates",
            "display": (
                "per-domain physical masters, contexts, unit appearances and "
                "distinct units"
            ),
            "interval_caption": _INTERVAL_CAPTION,
            "interaction_caption": _INTERACTION_CAPTION,
            "selection": (
                "source-only selection/calibration; frozen selected CNN "
                "identity across policies; no held-target fitting"
            ),
        },
        "boundary_flags": dict(_BOUNDARY_FLAGS),
        "labels": {
            "endpoints": {
                endpoint: _endpoint_label(endpoint) for endpoint in endpoints
            },
            "policies": {policy: _policy_label(policy) for policy in policies},
        },
        "f02_pairs": f02_pairs,
        "f03_effects": f03_effects,
        "f03_domains": f03_domains,
        "f04_interactions": f04_interactions,
        "f04_domains": f04_domains,
    }

    payload = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = {
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "counts": {
            "domains": len(domains),
            "registry_entries": len(registry),
            "f02_pairs": len(f02_pairs),
            "f03_effects": len(f03_effects),
            "f03_domains": len(f03_domains),
            "f04_interactions": len(f04_interactions),
            "f04_domains": len(f04_domains),
            "future_qc_unavailable": len(unavailable),
            "available_interactions": len(available),
        },
        "semantic_sha256": digest,
    }

    return {
        "semantic": semantic,
        "semantic_sha256": digest,
        "manifest": manifest,
    }
