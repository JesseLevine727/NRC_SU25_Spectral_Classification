"""Compact release export for P08 contrast diagnostics.

This module is a thin inference-summary adapter.  It performs no fitting,
score computation, resampling, rendering or file writing.  Existing contrast
results produced by :mod:`p08_universal_analysis` are copied into three tidy
tables plus a JSON-safe diagnostics payload, preserving every existing
aggregate scientific diagnostic while excluding private rows and identifiers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_universal_analysis as _analysis
from atlas_sers.visualization import p08_model_figure_data

__all__ = ["prepare_contrast_tables"]


# Exact shape of a draw summary.  Draw arrays themselves are never exported.
_SUMMARY_FIELDS = (
    "planned_draws",
    "defined_draws",
    "undefined_draws",
    "lower",
    "upper",
    "reason_code",
)

_STAT_FIELDS = (
    "mean",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "positive",
    "negative",
    "tied",
)

_POINT_FIELDS = (
    "overall",
    "domain_effects",
    "cell_effects",
    "domain_instrument",
    "procedure_effects",
    "procedure_domain_effects",
)

_DESCRIPTIVE_FIELDS = (
    "contrast_domain",
    "procedure_domain",
    "procedure_min_ties",
    "leave_one_domain",
    "leave_one_instrument",
    "leave_one_family",
    "minimum_paired_effect_ties",
)

_WEIGHTED_MODES = ("crossed", "master_only", "instrument_only")

_WEIGHTED_MODE_FIELDS = (
    "overall",
    "procedure_overall",
    "procedure_min_ba",
    "min_ba_difference",
    "minimum_paired_domain_effect",
)

_DRAW_BLOCK_FIELDS = ("summary", "draws")

_HIERARCHY_FIELDS = (
    "term_draws",
    "term_summaries",
    "combined",
    "sampled_domains",
    "empty_cells",
    "undefined_domain_occurrences",
    "affected_domains",
    "empty_cells_total",
    "undefined_domain_occurrences_total",
    "affected_domains_total",
)

_SIGN_FIELDS = (
    "label",
    "statistic",
    "observed_delta",
    "observed_statistic",
    "domain",
    "instrument",
)

_SIGN_GROUP_FIELDS = (
    "groups",
    "assignments",
    "observed_delta",
    "statistic",
    "hits",
    "p_descriptive",
)

_ADJUSTMENT_FIELDS = (
    "estimand",
    "family_id",
    "family_size",
    "adjustment",
    "domain_raw_p",
    "domain_adjusted_p",
    "instrument_raw_p",
    "instrument_adjusted_p",
)

_PAIRED_COLUMNS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "deep_model_id",
    "comparator_model_id",
    "domain",
    "point_effect",
    "instrument",
)

_PROCEDURE_COLUMNS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "deep_model_id",
    "comparator_model_id",
    "term_index",
    "term_policy_id",
    "term_model_id",
    "coefficient",
    "domain",
    "procedure_overall_balanced_accuracy",
    "procedure_domain_balanced_accuracy",
)

# ---------------------------------------------------------------------------
# Primitive validators.  Only finite scalars, None, scalar labels, bools and
# explicitly shaped lists are accepted; numeric dict/list payloads are not.
# ---------------------------------------------------------------------------


def _check_keys(value, allowed, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    extras = sorted(set(value) - set(allowed))
    if extras:
        raise ValueError(f"{name} has unexpected keys: {extras}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _optional_number(value, name):
    if value is None:
        return None
    return _number(value, name)


def _label(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string label")
    return value


def _optional_label(value, name):
    if value is None:
        return None
    return _label(value, name)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _label_list(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of labels")
    return [_label(item, f"{name}[]") for item in value]


def _label_number_map(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {
        _label(key, f"{name} key"): _number(item, f"{name}[{key}]") for key, item in value.items()
    }


def _label_optional_number_map(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {
        _label(key, f"{name} key"): _optional_number(item, f"{name}[{key}]")
        for key, item in value.items()
    }


def _extract_stats(value, name):
    _check_keys(value, _STAT_FIELDS, name)
    stats = {}
    for field in _STAT_FIELDS:
        if field not in value:
            raise ValueError(f"{name} missing {field!r}")
        if field in ("positive", "negative", "tied"):
            stats[field] = _integer(value[field], f"{name}.{field}")
        else:
            stats[field] = _number(value[field], f"{name}.{field}")
    return stats


def _extract_summary(value, name):
    _check_keys(value, _SUMMARY_FIELDS, name)
    summary = {}
    for field in _SUMMARY_FIELDS:
        if field not in value:
            raise ValueError(f"{name} missing {field!r}")
        if field in ("planned_draws", "defined_draws", "undefined_draws"):
            summary[field] = _integer(value[field], f"{name}.{field}")
        elif field == "reason_code":
            summary[field] = _optional_label(value[field], f"{name}.{field}")
        else:
            summary[field] = _optional_number(value[field], f"{name}.{field}")
    return summary


def _draw_block_summary(block, name):
    """Return the six-key summary of a draw block, dropping raw draws."""
    if not isinstance(block, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(block) != set(_DRAW_BLOCK_FIELDS):
        raise ValueError(f"{name} draw-block schema changed")
    return _extract_summary(block["summary"], name)


# ---------------------------------------------------------------------------
# Section extractors.  Each whitelists its own level so unexpected extras are
# rejected rather than leaked.
# ---------------------------------------------------------------------------


def _extract_point(value):
    _check_keys(value, _POINT_FIELDS, "diagnostics.point")
    return {
        "overall": _number(value["overall"], "point.overall"),
        "domain_effects": _label_number_map(value["domain_effects"], "point.domain_effects"),
        "domain_instrument": {
            _label(domain, "domain"): _label(instrument, "instrument")
            for domain, instrument in value["domain_instrument"].items()
        },
        "procedure_effects": [
            _number(item, "point.procedure_effects") for item in value["procedure_effects"]
        ],
        "procedure_domain_effects": [
            _label_number_map(item, "point.procedure_domain_effects")
            for item in value["procedure_domain_effects"]
        ],
    }


def _extract_descriptive(value):
    _check_keys(value, _DESCRIPTIVE_FIELDS, "diagnostics.descriptive")
    descriptive = {
        "contrast_domain": _extract_stats(value["contrast_domain"], "descriptive.contrast_domain"),
        "procedure_domain": [
            _extract_stats(item, "descriptive.procedure_domain")
            for item in value["procedure_domain"]
        ],
        "procedure_min_ties": [
            _label_list(item, "descriptive.procedure_min_ties")
            for item in value["procedure_min_ties"]
        ],
        "leave_one_domain": _label_optional_number_map(
            value["leave_one_domain"], "descriptive.leave_one_domain"
        ),
        "leave_one_instrument": _label_optional_number_map(
            value["leave_one_instrument"], "descriptive.leave_one_instrument"
        ),
        "leave_one_family": _label_optional_number_map(
            value["leave_one_family"], "descriptive.leave_one_family"
        ),
    }
    if "minimum_paired_effect_ties" in value:
        descriptive["minimum_paired_effect_ties"] = _label_list(
            value["minimum_paired_effect_ties"],
            "descriptive.minimum_paired_effect_ties",
        )
    return descriptive


def _extract_weighted(value):
    _check_keys(value, _WEIGHTED_MODES, "diagnostics.weighted")
    weighted = {}
    for mode in _WEIGHTED_MODES:
        block = value[mode]
        _check_keys(block, _WEIGHTED_MODE_FIELDS, f"weighted.{mode}")
        entry = {
            "overall": {
                "summary": _draw_block_summary(block["overall"], f"weighted.{mode}.overall")
            },
            "procedure_overall": [
                {"summary": _draw_block_summary(item, f"weighted.{mode}.procedure_overall")}
                for item in block["procedure_overall"]
            ],
            "procedure_min_ba": [
                {"summary": _draw_block_summary(item, f"weighted.{mode}.procedure_min_ba")}
                for item in block["procedure_min_ba"]
            ],
        }
        if "min_ba_difference" in block:
            entry["min_ba_difference"] = {
                "summary": _draw_block_summary(
                    block["min_ba_difference"], f"weighted.{mode}.min_ba_difference"
                )
            }
        if "minimum_paired_domain_effect" in block:
            entry["minimum_paired_domain_effect"] = {
                "summary": _draw_block_summary(
                    block["minimum_paired_domain_effect"],
                    f"weighted.{mode}.minimum_paired_domain_effect",
                )
            }
        weighted[mode] = entry
    return weighted


def _extract_hierarchy(value):
    _check_keys(value, _HIERARCHY_FIELDS, "diagnostics.hierarchy")
    return {
        "combined": {"summary": _draw_block_summary(value["combined"], "hierarchy.combined")},
        "term_summaries": [
            _extract_summary(item, "hierarchy.term_summaries") for item in value["term_summaries"]
        ],
        "empty_cells_total": _integer(value["empty_cells_total"], "hierarchy.empty_cells_total"),
        "undefined_domain_occurrences_total": _integer(
            value["undefined_domain_occurrences_total"],
            "hierarchy.undefined_domain_occurrences_total",
        ),
        "affected_domains_total": _integer(
            value["affected_domains_total"], "hierarchy.affected_domains_total"
        ),
    }


def _extract_sign_group(value, name):
    _check_keys(value, _SIGN_GROUP_FIELDS, name)
    return {
        "groups": _integer(value["groups"], f"{name}.groups"),
        "assignments": _integer(value["assignments"], f"{name}.assignments"),
        "observed_delta": _number(value["observed_delta"], f"{name}.observed_delta"),
        "statistic": _number(value["statistic"], f"{name}.statistic"),
        "hits": _integer(value["hits"], f"{name}.hits"),
        "p_descriptive": _number(value["p_descriptive"], f"{name}.p_descriptive"),
    }


def _extract_sign(value):
    _check_keys(value, _SIGN_FIELDS, "diagnostics.sign_sensitivity")
    return {
        "label": _label(value["label"], "sign_sensitivity.label"),
        "statistic": _label(value["statistic"], "sign_sensitivity.statistic"),
        "observed_delta": _number(value["observed_delta"], "sign_sensitivity.observed_delta"),
        "observed_statistic": _number(
            value["observed_statistic"], "sign_sensitivity.observed_statistic"
        ),
        "domain": _extract_sign_group(value["domain"], "sign_sensitivity.domain"),
        "instrument": _extract_sign_group(value["instrument"], "sign_sensitivity.instrument"),
    }


def _extract_adjustment(value):
    _check_keys(value, _ADJUSTMENT_FIELDS, "diagnostics.sensitivity_adjustment")
    return {
        "estimand": _label(value["estimand"], "adjustment.estimand"),
        "family_id": _label(value["family_id"], "adjustment.family_id"),
        "family_size": _integer(value["family_size"], "adjustment.family_size"),
        "adjustment": _label(value["adjustment"], "adjustment.adjustment"),
        "domain_raw_p": _optional_number(value["domain_raw_p"], "adjustment.domain_raw_p"),
        "domain_adjusted_p": _optional_number(
            value["domain_adjusted_p"], "adjustment.domain_adjusted_p"
        ),
        "instrument_raw_p": _optional_number(
            value["instrument_raw_p"], "adjustment.instrument_raw_p"
        ),
        "instrument_adjusted_p": _optional_number(
            value["instrument_adjusted_p"], "adjustment.instrument_adjusted_p"
        ),
    }


def _extract_sections(contrast):
    sections = {
        "point": _extract_point(contrast["point"]),
        "descriptive": _extract_descriptive(contrast["descriptive"]),
        "weighted": _extract_weighted(contrast["weighted"]),
        "hierarchy": _extract_hierarchy(contrast["hierarchy"]),
        "sign_sensitivity": _extract_sign(contrast["sign_sensitivity"]),
    }
    adjustment = contrast.get("sensitivity_adjustment")
    if adjustment is not None:
        sections["sensitivity_adjustment"] = _extract_adjustment(adjustment)
    return sections


# ---------------------------------------------------------------------------
# Contrast location and public metadata.
# ---------------------------------------------------------------------------


def _resolve_contrasts(data):
    return data["contrasts"]


def _public_metadata(entry):
    return {
        "contrast_id": _label(entry["contrast_id"], "contrast_id"),
        "family_id": _label(entry["family_id"], "family_id"),
        "endpoint": _label(entry["endpoint"], "endpoint"),
        "model_id": _optional_label(entry["model_id"], "model_id"),
        "policy_id": _label(entry["policy_id"], "policy_id"),
        "deep_model_id": _optional_label(entry["deep_model_id"], "deep_model_id"),
        "comparator_model_id": _optional_label(entry["comparator_model_id"], "comparator_model_id"),
    }


def _is_available(contrast, entry):
    if contrast is None:
        return False
    if "available" in contrast:
        return _flag(contrast["available"], "available")
    return _flag(entry["available"], "available")


def _reason(contrast, entry):
    if contrast is not None and "reason" in contrast:
        return _optional_label(contrast.get("reason"), "reason")
    return _optional_label(entry.get("reason"), "reason")


# ---------------------------------------------------------------------------
# Table row builders.
# ---------------------------------------------------------------------------


def _paired_rows(estimand, entry, contrast):
    point = contrast["point"]
    domains = contrast["domains"]
    domain_effects = point["domain_effects"]
    domain_instrument = point["domain_instrument"]
    base = {"estimand": estimand, **_public_metadata(entry)}
    rows = []
    for domain in domains:
        instrument = _label(domain_instrument[domain], "instrument")
        rows.append(
            {
                **base,
                "domain": _label(domain, "domain"),
                "point_effect": _number(domain_effects[domain], f"point.domain_effects[{domain}]"),
                "instrument": instrument,
            }
        )
    return rows


def _procedure_rows(estimand, entry, contrast):
    point = contrast["point"]
    domains = contrast["domains"]
    labels = entry["procedure_labels"]
    coefficients = entry["coefficients"]
    if len(labels) != len(coefficients):
        raise ValueError("procedure labels and coefficients length mismatch")
    base = {"estimand": estimand, **_public_metadata(entry)}
    rows = []
    for term_index, (procedure_label, coefficient) in enumerate(
        zip(labels, coefficients, strict=True)
    ):
        term_policy_id = _label(procedure_label[0], "procedure_label.policy")
        term_model_id = _label(procedure_label[1], "procedure_label.model")
        overall = _number(point["procedure_effects"][term_index], "point.procedure_effects")
        domain_effects = point["procedure_domain_effects"][term_index]
        for domain in domains:
            rows.append(
                {
                    **base,
                    "term_index": term_index,
                    "term_policy_id": term_policy_id,
                    "term_model_id": term_model_id,
                    "coefficient": _number(coefficient, "coefficient"),
                    "domain": _label(domain, "domain"),
                    "procedure_overall_balanced_accuracy": overall,
                    "procedure_domain_balanced_accuracy": _number(
                        domain_effects[domain],
                        f"procedure_domain_effects[{term_index}][{domain}]",
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def prepare_contrast_tables(analysis):
    """Build the compact release tables, diagnostics and manifest.

    The existing figure-data validation is invoked first for its registry,
    boundary and paired-metric checks; any render preparation is discarded and
    no renderer is called. Exported tables are copied; input arrays are not duplicated.
    """
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping")

    snapshot = analysis
    p08_model_figure_data.prepare_model_figures(snapshot)

    estimands = tuple(_analysis.ESTIMANDS)
    registry = tuple(_analysis.contrast_registry())
    contrasts = _resolve_contrasts(snapshot)

    source_summary = snapshot["contrast_summary"]
    if not isinstance(source_summary, pd.DataFrame):
        raise TypeError("contrast_summary must be a dataframe")
    if list(source_summary.columns) != list(_analysis._SUMMARY_COLUMNS):
        raise ValueError("contrast summary schema changed")
    # Existing flattening only: no inference or numerical score recalculation.
    expected_summary = _analysis._build_summary(registry, contrasts)
    pd.testing.assert_frame_equal(source_summary, expected_summary, check_exact=True)
    contrast_summary = source_summary.copy(deep=True)

    paired_rows = []
    procedure_rows = []
    diagnostics = {}
    counts = {}
    for estimand in estimands:
        if estimand not in contrasts:
            raise KeyError(f"missing estimand {estimand!r} in contrast results")
        per_estimand = contrasts[estimand]
        diagnostics[estimand] = {}
        available = 0
        for entry in registry:
            cid = entry["contrast_id"]
            contrast = per_estimand.get(cid)
            metadata = _public_metadata(entry)
            if not _is_available(contrast, entry):
                diagnostics[estimand][cid] = {
                    "available": False,
                    "reason": _reason(contrast, entry),
                    **metadata,
                }
                continue
            available += 1
            record = {
                "available": True,
                "reason": _reason(contrast, entry),
                **metadata,
            }
            record.update(_extract_sections(contrast))
            diagnostics[estimand][cid] = record
            paired_rows.extend(_paired_rows(estimand, entry, contrast))
            procedure_rows.extend(_procedure_rows(estimand, entry, contrast))
        counts[estimand] = {
            "registry": len(registry),
            "available": available,
            "unavailable": len(registry) - available,
        }

    tables = {
        "contrast_summary": contrast_summary,
        "paired_domains": pd.DataFrame(paired_rows, columns=list(_PAIRED_COLUMNS)),
        "procedure_domains": pd.DataFrame(procedure_rows, columns=list(_PROCEDURE_COLUMNS)),
    }

    manifest = {
        "estimands": list(estimands),
        "counts": counts,
        "external_authentication_verified": False,
        "reviewed": False,
        "published": False,
        "no_G4_decision": True,
        "no_model_or_policy_selection": True,
        "conditional_on_saved_fits_observed_support": True,
    }

    return {"tables": tables, "diagnostics": diagnostics, "manifest": manifest}
