"""T299 pure universal contrast engine (private slice).

This module evaluates a single, already-authenticated COMPLETE endpoint/estimand
panel in wide form.  It deliberately performs no support or completeness
inference: the caller authenticates exact registered roles and supplies
externally generated shared weights.  Panel validation and compilation reuse the
inherited :func:`compile_pair`; scoring reuses :func:`score_weights`; feasibility
reuses :func:`hierarchical_draws`; interval construction reuses
:func:`summarize_draws`.

Boundary notes
--------------
* ``hierarchical_draws`` is the number of original hierarchical-bootstrap
  draws and must be in ``1..10000``.  Tests may pass a smaller positive fixture
  value; the production caller enforces 10000.
* No probability aggregation or scoring from probabilities is performed here.
* Undefined hierarchical draws (NaN) are preserved and never redrawn, dropped
  or repaired; inherited summaries leave intervals missing when any draw is
  undefined.
* Sign enumeration is a finite symmetry sensitivity, not randomized causality.
  Domain signs and instrument signs are two separate registered sensitivities
  with separate Holm families; they are never multiplied together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p06p11_hierarchy import (
    hierarchical_draws as _hierarchical_draws,
)
from atlas_sers.evaluation.p06p11_inference import (
    compile_pair,
    score_weights,
    summarize_draws,
)

__all__ = ["analyze_contrast", "holm_fixed_family"]

_ID_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
)
_RESERVED = ("correct_model", "correct_reference")
_MAX_DRAWS = 10000
_MAX_SIGN_DOMAINS = 13
_MAX_SIGN_INSTRUMENTS = 10
_TOL = 1e-12
_TWO_TERM = (1.0, -1.0)
_FOUR_TERM = (1.0, -1.0, -1.0, 1.0)


def _as_int(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _identities(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of distinct nonempty strings")
    try:
        sequence = list(values)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of distinct nonempty strings"
        ) from None
    if not sequence:
        raise ValueError(f"{name} must be nonempty")
    seen = set()
    for value in sequence:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must contain trimmed nonempty strings")
        if value in seen:
            raise ValueError(f"{name} must be unique")
        seen.add(value)
    if sequence != sorted(sequence):
        raise ValueError(f"{name} must be sorted lexicographically")
    return sequence


def _weight_matrix(values, name):
    if isinstance(values, (bool, np.bool_, str, complex)):
        raise TypeError(f"{name} must be a finite positive numeric 2D array")
    array = np.asarray(values)
    if array.dtype == object or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise TypeError(f"{name} must be numeric")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    n_draws = array.shape[0]
    if n_draws < 1 or n_draws > _MAX_DRAWS:
        raise ValueError(f"{name} must have 1..{_MAX_DRAWS} draws")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.all(array > 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.asarray(array, dtype=float)


def _select_columns(weights, ids, global_ids, name):
    index = {value: position for position, value in enumerate(global_ids)}
    try:
        columns = [index[value] for value in ids]
    except KeyError:
        raise ValueError(
            f"{name} ids are missing from the global identity sequence"
        ) from None
    return weights[:, columns]


def _validate_coefficients(coefficients):
    if isinstance(
        coefficients, (bool, np.bool_, str, bytes, complex, np.complexfloating)
    ):
        raise TypeError("coefficients must be a sequence of exact 1/-1 real values")
    try:
        sequence = list(coefficients)
    except TypeError:
        raise TypeError(
            "coefficients must be a sequence of exact 1/-1 real values"
        ) from None
    if len(sequence) not in (2, 4):
        raise ValueError("coefficients must have length 2 or 4")
    expected = _TWO_TERM if len(sequence) == 2 else _FOUR_TERM
    result = []
    for value, target in zip(sequence, expected, strict=True):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("coefficients must be real numbers, not bool")
        if isinstance(value, (str, bytes)):
            raise TypeError("coefficients must be real numbers, not strings")
        if isinstance(value, (complex, np.complexfloating)):
            raise TypeError("coefficients must be real numbers")
        if isinstance(value, np.ndarray):
            raise TypeError("coefficients must be scalar real numbers")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TypeError("coefficients must be scalar real numbers") from None
        if not np.isfinite(number):
            raise ValueError("coefficients must be finite")
        if number != target:
            raise ValueError("coefficients must be exactly (1,-1) or (1,-1,-1,1)")
        result.append(float(target))
    return tuple(result)


def _validate_columns(columns, count):
    if isinstance(columns, (str, bytes)):
        raise TypeError("correct_columns must be a sequence of unique column names")
    try:
        sequence = list(columns)
    except TypeError:
        raise TypeError(
            "correct_columns must be a sequence of unique column names"
        ) from None
    if len(sequence) != count:
        raise ValueError("correct_columns length must match coefficients")
    seen = set()
    result = []
    for value in sequence:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("correct_columns must be trimmed nonempty strings")
        if value in seen:
            raise ValueError("correct_columns must be unique")
        seen.add(value)
        result.append(value)
    return result


def _assert_shared_structure(designs):
    base = designs[0]
    for other in designs[1:]:
        if (
            other.masters != base.masters
            or other.instruments != base.instruments
            or other.domains != base.domains
            or other.cell_keys != base.cell_keys
        ):
            raise ValueError("inconsistent_panel_structure")
        for attribute in ("counts", "cell_domain", "domain_instrument", "cell_factor"):
            if not np.array_equal(getattr(other, attribute), getattr(base, attribute)):
                raise ValueError("inconsistent_panel_structure")


def _binary_column(series, name):
    raw = series.to_numpy()
    result = np.empty(raw.shape[0], dtype=float)
    for index, value in enumerate(raw.tolist()):
        if isinstance(value, (bool, np.bool_)):
            result[index] = float(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{name} must be bool or finite 0/1 numeric")
            if number not in (0.0, 1.0):
                raise ValueError(f"{name} must contain only 0/1")
            result[index] = number
            continue
        raise ValueError(f"{name} must be bool or finite 0/1 numeric")
    return result


def _group_point(contexts, labels, domain_values, values, domain_order, cell_order):
    n = len(values)
    if not (len(contexts) == len(labels) == len(domain_values) == n):
        raise ValueError("row grouping arrays must align")

    cell_sum = {}
    cell_count = {}
    for i in range(n):
        key = (contexts[i], labels[i])
        cell_sum[key] = cell_sum.get(key, 0.0) + float(values[i])
        cell_count[key] = cell_count.get(key, 0) + 1
    cell_effect = {key: cell_sum[key] / cell_count[key] for key in cell_sum}

    context_sum = {}
    context_count = {}
    context_domain = {}
    for i in range(n):
        context_domain[contexts[i]] = domain_values[i]
    for (context, _label), value in cell_effect.items():
        context_sum[context] = context_sum.get(context, 0.0) + value
        context_count[context] = context_count.get(context, 0) + 1

    domain_sum = {}
    domain_count = {}
    for context, total in context_sum.items():
        mean = total / context_count[context]
        domain = context_domain[context]
        domain_sum[domain] = domain_sum.get(domain, 0.0) + mean
        domain_count[domain] = domain_count.get(domain, 0) + 1
    domain_effect = {
        domain: domain_sum[domain] / domain_count[domain] for domain in domain_sum
    }

    if set(cell_effect) != set(cell_order):
        raise RuntimeError("raw_grouping_mismatch")
    if set(domain_effect) != set(domain_order):
        raise RuntimeError("raw_grouping_mismatch")

    overall = float(sum(domain_effect.values())) / len(domain_effect)
    domain_array = np.array([domain_effect[name] for name in domain_order], dtype=float)
    cell_array = {key: float(cell_effect[key]) for key in cell_order}
    return overall, domain_array, cell_array


def _check_parity(design, point, domain, ones_m, ones_i):
    overall, domain_out = score_weights(design, ones_m, ones_i)
    if not np.allclose(overall, np.array([point]), rtol=0.0, atol=_TOL):
        raise RuntimeError("unit_weight_parity_failed")
    if not np.allclose(
        domain_out[0], np.asarray(domain, dtype=float), rtol=0.0, atol=_TOL
    ):
        raise RuntimeError("unit_weight_parity_failed")


def _draw_block(values):
    array = np.asarray(values, dtype=float)
    return {"draws": array, "summary": summarize_draws(array)}


def _stats(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive": int(np.sum(array > _TOL)),
        "negative": int(np.sum(array < -_TOL)),
        "tied": int(np.sum(np.abs(array) <= _TOL)),
    }


def _descriptive_section(
    contrast_domain,
    procedure_domains,
    domain_names,
    instrument_names,
    domain_instrument_index,
    domain_families,
):
    n = len(domain_names)
    values = np.asarray(contrast_domain, dtype=float)
    section = {
        "contrast_domain": _stats(values),
        "procedure_domain": [_stats(dom) for dom in procedure_domains],
        "procedure_min_ties": [],
        "leave_one_domain": {},
        "leave_one_instrument": {},
        "leave_one_family": {},
    }
    for dom in procedure_domains:
        array = np.asarray(dom, dtype=float)
        low = float(array.min())
        section["procedure_min_ties"].append(
            [domain_names[i] for i in range(n) if abs(array[i] - low) <= _TOL]
        )
    if len(procedure_domains) == 2:
        paired = np.asarray(procedure_domains[0], dtype=float) - np.asarray(
            procedure_domains[1], dtype=float
        )
        low = float(paired.min())
        section["minimum_paired_effect_ties"] = [
            domain_names[i] for i in range(n) if abs(paired[i] - low) <= _TOL
        ]
    if n > 1:
        total = float(values.sum())
        section["leave_one_domain"] = {
            domain_names[i]: float((total - values[i]) / (n - 1)) for i in range(n)
        }
    else:
        section["leave_one_domain"] = {domain_names[0]: None}
    index = np.asarray(domain_instrument_index, dtype=np.intp)
    for instrument in instrument_names:
        mask = np.array(
            [instrument_names[int(index[i])] != instrument for i in range(n)]
        )
        section["leave_one_instrument"][instrument] = (
            float(values[mask].mean()) if bool(mask.any()) else None
        )
    mapping = {}
    if domain_families is not None:
        if not isinstance(domain_families, Mapping):
            raise TypeError("domain_families must be a mapping or None")
        for domain, family in domain_families.items():
            if not isinstance(domain, str) or domain not in domain_names:
                continue
            if not isinstance(family, str):
                continue
            cleaned = family.strip()
            if not cleaned or cleaned.lower() == "unknown":
                continue
            mapping[domain] = cleaned
    for family in sorted(set(mapping.values())):
        mask = np.array([mapping.get(domain_names[i]) != family for i in range(n)])
        section["leave_one_family"][family] = (
            float(values[mask].mean()) if bool(mask.any()) else None
        )
    return section


def _sign_sensitivity(domain_effects, domain_instrument_index, n_instruments):
    effects = np.asarray(domain_effects, dtype=float)
    if effects.ndim != 1:
        raise ValueError("domain effects must be one-dimensional")
    n_domains = effects.size
    if n_domains < 1:
        raise ValueError("at least one domain is required")
    if n_domains > _MAX_SIGN_DOMAINS:
        raise ValueError(
            f"sign enumeration supports at most {_MAX_SIGN_DOMAINS} domains"
        )
    if isinstance(n_instruments, (bool, np.bool_)) or not isinstance(
        n_instruments, (int, np.integer)
    ):
        raise TypeError("n_instruments must be an integer")
    n_instruments = int(n_instruments)
    if n_instruments < 1:
        raise ValueError("at least one instrument is required")
    if n_instruments > _MAX_SIGN_INSTRUMENTS:
        raise ValueError(
            f"sign enumeration supports at most {_MAX_SIGN_INSTRUMENTS} instruments"
        )
    if not np.all(np.isfinite(effects)):
        raise ValueError("domain effects must be finite")
    groups = np.asarray(domain_instrument_index, dtype=np.intp)
    if groups.shape != (n_domains,):
        raise ValueError("domain instrument index must align with domains")
    if np.any(groups < 0) or np.any(groups >= n_instruments):
        raise ValueError("domain instrument index out of range")

    observed_delta = float(effects.mean())
    observed_statistic = float(abs(observed_delta))

    domain_assignments = 1 << n_domains
    domain_bits = (
        (np.arange(domain_assignments)[:, None] >> np.arange(n_domains)) & 1
    ).astype(float)
    domain_signs = 1.0 - 2.0 * domain_bits
    domain_statistics = np.abs(domain_signs @ effects) / n_domains
    domain_hits = int(np.count_nonzero(domain_statistics >= observed_statistic - _TOL))

    instrument_assignments = 1 << n_instruments
    instrument_bits = (
        (np.arange(instrument_assignments)[:, None] >> np.arange(n_instruments)) & 1
    ).astype(float)
    instrument_signs = 1.0 - 2.0 * instrument_bits
    factors = instrument_signs[:, groups]
    instrument_statistics = np.abs((factors * effects).sum(axis=1)) / n_domains
    instrument_hits = int(
        np.count_nonzero(instrument_statistics >= observed_statistic - _TOL)
    )

    return {
        "label": "sign_symmetry_sensitivity_not_randomized_causality",
        "statistic": "absolute_equal_domain_mean",
        "observed_delta": observed_delta,
        "observed_statistic": observed_statistic,
        "domain": {
            "groups": n_domains,
            "assignments": domain_assignments,
            "observed_delta": observed_delta,
            "statistic": observed_statistic,
            "hits": domain_hits,
            "p_descriptive": domain_hits / domain_assignments,
        },
        "instrument": {
            "groups": n_instruments,
            "assignments": instrument_assignments,
            "observed_delta": observed_delta,
            "statistic": observed_statistic,
            "hits": instrument_hits,
            "p_descriptive": instrument_hits / instrument_assignments,
        },
    }


def analyze_contrast(
    rows,
    correct_columns,
    coefficients,
    *,
    global_masters,
    global_instruments,
    master_weights,
    instrument_weights,
    hierarchical_draws=10000,
    hierarchical_seed=2026093003,
    domain_families=None,
):
    """Evaluate one authenticated complete panel contrast.

    ``rows`` is a wide, complete endpoint/estimand panel with the identity
    columns used by :func:`compile_pair`.  ``correct_columns`` is either two
    procedure columns with coefficients ``(1,-1)`` or four columns with
    ``(1,-1,-1,1)``.  Support/completeness is never inferred here.

    Global weights are supplied once by the caller and shared across all
    terms; they are validated and column-selected against the global sorted
    identity sequences, never regenerated.  ``hierarchical_draws`` is capped at
    10000 draws; tests may pass a smaller positive fixture value.  Point
    estimates are computed independently from the raw rows by equal class mean
    per context, equal context mean per domain and equal domain mean, then
    checked against ``score_weights`` with unit weights.
    """
    coeffs = _validate_coefficients(coefficients)
    columns = _validate_columns(correct_columns, len(coeffs))
    n_hier = _as_int(hierarchical_draws, "hierarchical_draws", 1, _MAX_DRAWS)
    hier_seed = _as_int(hierarchical_seed, "hierarchical_seed", 0)

    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    if rows.shape[0] == 0:
        raise ValueError("rows must be nonempty")
    if not rows.columns.is_unique:
        raise ValueError("rows must have unique columns")
    missing = [column for column in _ID_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"missing required identity columns: {missing}")
    for column in columns:
        if column not in rows.columns:
            raise ValueError(f"missing correct column: {column}")
        if column in _ID_COLUMNS or column in _RESERVED:
            raise ValueError("correct columns must not reuse reserved columns")

    g_masters = _identities(global_masters, "global_masters")
    g_instruments = _identities(global_instruments, "global_instruments")
    master_matrix = _weight_matrix(master_weights, "master_weights")
    instrument_matrix = _weight_matrix(instrument_weights, "instrument_weights")
    if master_matrix.shape[0] != instrument_matrix.shape[0]:
        raise ValueError("master_weights and instrument_weights must share draw count")
    if master_matrix.shape[1] != len(g_masters):
        raise ValueError("master_weights width must match global_masters")
    if instrument_matrix.shape[1] != len(g_instruments):
        raise ValueError("instrument_weights width must match global_instruments")

    designs = []
    for column in columns:
        panel = rows.copy()
        panel["correct_model"] = panel[column]
        panel["correct_reference"] = 0
        designs.append(compile_pair(panel))
    _assert_shared_structure(designs)
    base = designs[0]
    mw_panel = _select_columns(master_matrix, base.masters, g_masters, "master")
    iw_panel = _select_columns(
        instrument_matrix, base.instruments, g_instruments, "instrument"
    )

    contrast_delta = np.zeros_like(np.asarray(base.delta_correct, dtype=float))
    for coefficient, design in zip(coeffs, designs, strict=True):
        contrast_delta = contrast_delta + coefficient * np.asarray(
            design.delta_correct, dtype=float
        )
    contrast_design = replace(base, delta_correct=contrast_delta)

    domain_names = list(base.domains)
    instrument_names = list(base.instruments)
    cell_keys = list(base.cell_keys)
    domain_instrument = {
        domain_names[i]: instrument_names[int(base.domain_instrument[i])]
        for i in range(len(domain_names))
    }

    contexts = rows["context_id"].tolist()
    labels = rows["true_label"].tolist()
    domain_values = rows["domain"].tolist()
    procedure_values = [_binary_column(rows[column], column) for column in columns]

    procedure_points = []
    procedure_domains = []
    for values in procedure_values:
        point, domain, _cells = _group_point(
            contexts, labels, domain_values, values, domain_names, cell_keys
        )
        procedure_points.append(point)
        procedure_domains.append(domain)

    contrast_values = np.zeros(len(rows), dtype=float)
    for coefficient, values in zip(coeffs, procedure_values, strict=True):
        contrast_values = contrast_values + coefficient * values
    contrast_point, contrast_domain, contrast_cells = _group_point(
        contexts, labels, domain_values, contrast_values, domain_names, cell_keys
    )

    ones_m = np.ones((1, len(base.masters)))
    ones_i = np.ones((1, len(base.instruments)))
    _check_parity(contrast_design, contrast_point, contrast_domain, ones_m, ones_i)
    for design, point, domain in zip(
        designs, procedure_points, procedure_domains, strict=True
    ):
        _check_parity(design, point, domain, ones_m, ones_i)

    point_output = {
        "overall": contrast_point,
        "domain_effects": {
            name: float(value)
            for name, value in zip(domain_names, contrast_domain, strict=True)
        },
        "cell_effects": {key: float(value) for key, value in contrast_cells.items()},
        "domain_instrument": domain_instrument,
        "procedure_effects": [float(value) for value in procedure_points],
        "procedure_domain_effects": [
            {name: float(value) for name, value in zip(domain_names, dom, strict=True)}
            for dom in procedure_domains
        ],
    }

    two_term = len(coeffs) == 2
    modes = {
        "crossed": (mw_panel, iw_panel),
        "master_only": (mw_panel, np.ones_like(iw_panel)),
        "instrument_only": (np.ones_like(mw_panel), iw_panel),
    }
    weighted = {}
    for mode_name, (mode_masters, mode_instruments) in modes.items():
        entry = {
            "overall": _draw_block(
                score_weights(contrast_design, mode_masters, mode_instruments)[0]
            )
        }
        procedure_overall = []
        procedure_min_ba = []
        procedure_domain_out = []
        for design in designs:
            overall, domain_out = score_weights(design, mode_masters, mode_instruments)
            procedure_overall.append(_draw_block(overall))
            procedure_min_ba.append(_draw_block(domain_out.min(axis=1)))
            procedure_domain_out.append(domain_out)
        entry["procedure_overall"] = procedure_overall
        entry["procedure_min_ba"] = procedure_min_ba
        if two_term:
            difference = procedure_min_ba[0]["draws"] - procedure_min_ba[1]["draws"]
            entry["min_ba_difference"] = _draw_block(difference)
            paired = (procedure_domain_out[0] - procedure_domain_out[1]).min(axis=1)
            entry["minimum_paired_domain_effect"] = _draw_block(paired)
        weighted[mode_name] = entry

    term_results = [
        _hierarchical_draws(design, draws=n_hier, seed=hier_seed) for design in designs
    ]
    reference = term_results[0]
    for result in term_results[1:]:
        if not np.array_equal(result["sampled_domains"], reference["sampled_domains"]):
            raise RuntimeError("hierarchy_shared_support_mismatch")
        if not np.array_equal(result["empty_cells"], reference["empty_cells"]):
            raise RuntimeError("hierarchy_shared_support_mismatch")
        if not np.array_equal(
            result["undefined_domain_occurrences"],
            reference["undefined_domain_occurrences"],
        ):
            raise RuntimeError("hierarchy_shared_support_mismatch")
        if not np.array_equal(
            result["affected_domains"], reference["affected_domains"]
        ):
            raise RuntimeError("hierarchy_shared_support_mismatch")

    combined = np.zeros(n_hier, dtype=float)
    for coefficient, result in zip(coeffs, term_results, strict=True):
        combined = combined + coefficient * np.asarray(result["draws"], dtype=float)
    hierarchy = {
        "term_draws": [
            np.array(result["draws"], dtype=float, copy=True) for result in term_results
        ],
        "term_summaries": [summarize_draws(result["draws"]) for result in term_results],
        "combined": _draw_block(combined),
        "sampled_domains": np.array(reference["sampled_domains"], copy=True),
        "empty_cells": np.array(reference["empty_cells"], copy=True),
        "undefined_domain_occurrences": np.array(
            reference["undefined_domain_occurrences"], copy=True
        ),
        "affected_domains": np.array(reference["affected_domains"], copy=True),
        "empty_cells_total": int(np.asarray(reference["empty_cells"]).sum()),
        "undefined_domain_occurrences_total": int(
            np.asarray(reference["undefined_domain_occurrences"]).sum()
        ),
        "affected_domains_total": int(np.asarray(reference["affected_domains"]).sum()),
    }

    descriptive = _descriptive_section(
        contrast_domain,
        procedure_domains,
        domain_names,
        instrument_names,
        base.domain_instrument,
        domain_families,
    )
    sign = _sign_sensitivity(
        contrast_domain,
        np.asarray(base.domain_instrument, dtype=np.intp),
        len(instrument_names),
    )

    return {
        "contrast": {
            "columns": list(columns),
            "coefficients": list(coeffs),
            "term_count": len(coeffs),
        },
        "domains": domain_names,
        "instruments": instrument_names,
        "masters": list(base.masters),
        "cell_keys": list(base.cell_keys),
        "draw_counts": int(master_matrix.shape[0]),
        "labels": {
            "weighted": "conditional_on_supplied_shared_weights",
            "hierarchy": "original_hierarchical_bootstrap",
            "sign_sensitivity": "sign_symmetry_sensitivity_not_randomized_causality",
            "interval_method": "linear_2.5_97.5",
            "bca": False,
            "tail_p_values": False,
        },
        "point": point_output,
        "weighted": weighted,
        "hierarchy": hierarchy,
        "descriptive": descriptive,
        "sign_sensitivity": sign,
    }


def holm_fixed_family(pvalues, family_size):
    """Holm step-down adjustment over a fixed, caller-declared family.

    Values are returned in input order.  Unavailable ``None`` entries stay
    ``None`` but contribute ``p=1`` to the step-down calculation.  Ties are
    stable.  ``family_size`` must equal ``len(pvalues)``; no family reduction
    or raw-value modification is performed.
    """
    if isinstance(pvalues, (str, bytes)):
        raise TypeError("pvalues must be a sequence")
    try:
        sequence = list(pvalues)
    except TypeError:
        raise TypeError("pvalues must be a sequence") from None
    if not sequence:
        raise ValueError("pvalues must be nonempty")
    if isinstance(family_size, (bool, np.bool_)) or not isinstance(
        family_size, (int, np.integer)
    ):
        raise TypeError("family_size must be an integer")
    if int(family_size) != len(sequence):
        raise ValueError("family_size must equal len(pvalues)")

    n = len(sequence)
    values = np.ones(n, dtype=float)
    available = np.zeros(n, dtype=bool)
    for position, value in enumerate(sequence):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise TypeError("pvalues must be probabilities or None")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TypeError("pvalues must be probabilities or None") from None
        if not np.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError("pvalues must be finite probabilities in 0..1")
        values[position] = number
        available[position] = True

    order = np.argsort(values, kind="stable")
    adjusted = np.ones(n, dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (n - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return [None if not available[i] else float(adjusted[i]) for i in range(n)]
