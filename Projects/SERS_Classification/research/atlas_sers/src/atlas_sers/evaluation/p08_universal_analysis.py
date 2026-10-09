"""P08 universal preprocessing analysis driver (private draft, pure in-memory).

This module is a thin orchestration layer over two reviewed private adapters:
``p08_universal_units`` (validated M01/M06 full-panel units and descriptive
summaries) and ``p08_universal_inference`` (one complete contrast with
externally supplied shared weights).  It performs no fitting, calibration,
model selection, plotting, file or network access and no new preprocessing.

The caller authenticates frozen files, registered roles and production
population dimensions upstream.  This layer additionally enforces exact common
identity support across every model/policy comparison without intersecting or
dropping rows, and it never recalibrates, reselects or averages seeds.

The 44 available registry entries are evaluated under each of the two estimands
(``equal_context`` and ``pooled_four_fold``) with the *same two weight array
objects*, generated exactly once.  The eight future-QC interaction entries are
retained as explicit missing rows and never call the contrast engine.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_universal_inference as _inference
from atlas_sers.evaluation import p08_universal_units as _units
from atlas_sers.evaluation.p06p11_hierarchy import (
    _class_pools,
    _validated_arrays,
)
from atlas_sers.evaluation.p06p11_inference import compile_pair

analyze_contrast = _inference.analyze_contrast
holm_fixed_family = _inference.holm_fixed_family

__all__ = [
    "DRAW_COUNT",
    "analyze_panel",
    "contrast_registry",
    "hierarchy_support",
]

# Fixed module constant.  Tests may monkeypatch it to a small positive value;
# no user-facing override is exposed.
DRAW_COUNT = 10000

MASTER_SEED = 2026093001
INSTRUMENT_SEED = 2026093002
HIERARCHY_SEED = 2026093003

MAX_DRAWS = 10000
_BATCH = 128

POLICIES = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")
MODELS = (
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
    "D0-M",
    "P05-SELECTED",
)
NONMINIMAL_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
ESTIMANDS = ("equal_context", "pooled_four_fold")
ENDPOINTS = ("M01", "M06")
DEEP_MODELS = ("D0-M", "P05-SELECTED")
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
QC_CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST")

QC_POLICY = "PP-QC-SRC"
QC_REASON = "outside_universal_execution_scope"

FAMILY_UNIVERSAL_POLICY = "universal_policy"
FAMILY_POLICY_MODEL_INTERACTION = "policy_model_interaction"

_ID_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
)

_EFFECT_COEFFICIENTS = (1.0, -1.0)
_INTERACTION_COEFFICIENTS = (1.0, -1.0, -1.0, 1.0)

_SUMMARY_COLUMNS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "deep_model_id",
    "comparator_model_id",
    "available",
    "reason",
    "point_effect",
    "crossed_lower",
    "crossed_upper",
    "crossed_reason",
    "master_only_lower",
    "master_only_upper",
    "master_only_reason",
    "instrument_only_lower",
    "instrument_only_upper",
    "instrument_only_reason",
    "hierarchy_planned",
    "hierarchy_defined",
    "hierarchy_undefined",
    "hierarchy_reason",
    "domain_raw_p",
    "domain_adjusted_p",
    "instrument_raw_p",
    "instrument_adjusted_p",
    "family_size",
    "adjustment",
)

_BOUNDARY_LABELS = (
    "conditional_on_saved_fits_observed_support",
    "descriptive_sign_symmetry_not_randomized",
    "no_G4_decision",
    "no_model_or_policy_selection",
)


def _require_int(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _identity_list(values, name):
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


def _check_weights(values, name):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} weights must be finite")
    if not np.all(values > 0.0):
        raise ValueError(f"{name} weights must be positive")


def _procedure_column(policy_id, model_id):
    return f"{policy_id}::{model_id}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _effect_entry(policy_id, model_id, endpoint):
    return {
        "contrast_id": f"universal_policy::{policy_id}::{model_id}::{endpoint}",
        "family_id": FAMILY_UNIVERSAL_POLICY,
        "endpoint": endpoint,
        "policy_id": policy_id,
        "model_id": model_id,
        "deep_model_id": None,
        "comparator_model_id": None,
        "procedure_labels": (
            (policy_id, model_id, endpoint),
            ("PP-U-MIN", model_id, endpoint),
        ),
        "coefficients": _EFFECT_COEFFICIENTS,
        "available": True,
        "reason": None,
    }


def _interaction_entry(
    policy_id, deep_model_id, comparator_model_id, endpoint, *, available, reason
):
    return {
        "contrast_id": (
            f"policy_model_interaction::{policy_id}::{deep_model_id}"
            f"::{comparator_model_id}::{endpoint}"
        ),
        "family_id": FAMILY_POLICY_MODEL_INTERACTION,
        "endpoint": endpoint,
        "policy_id": policy_id,
        "model_id": None,
        "deep_model_id": deep_model_id,
        "comparator_model_id": comparator_model_id,
        "procedure_labels": (
            (policy_id, deep_model_id, endpoint),
            ("PP-U-MIN", deep_model_id, endpoint),
            (policy_id, comparator_model_id, endpoint),
            ("PP-U-MIN", comparator_model_id, endpoint),
        ),
        "coefficients": _INTERACTION_COEFFICIENTS,
        "available": bool(available),
        "reason": reason,
    }


def contrast_registry():
    """Return the finite, deterministic 52-entry P08 contrast registry.

    The registry is independent of any result and is never reduced for
    coincident D0-M/P05-SELECTED predictions.  It contains 20 universal policy
    effects, 24 universal policy-model interactions and eight future-QC
    interactions marked unavailable.  No QC policy effects, family fallbacks,
    adaptive/QC Extra Trees entries or other studies are added.
    """

    entries = []
    for policy_id in NONMINIMAL_POLICIES:
        for model_id in MODELS:
            for endpoint in ENDPOINTS:
                entries.append(_effect_entry(policy_id, model_id, endpoint))
    for policy_id in NONMINIMAL_POLICIES:
        for deep_model_id in DEEP_MODELS:
            for comparator_model_id in CLASSICAL_MODELS:
                for endpoint in ENDPOINTS:
                    entries.append(
                        _interaction_entry(
                            policy_id,
                            deep_model_id,
                            comparator_model_id,
                            endpoint,
                            available=True,
                            reason=None,
                        )
                    )
    for deep_model_id in DEEP_MODELS:
        for comparator_model_id in QC_CLASSICAL_MODELS:
            for endpoint in ENDPOINTS:
                entries.append(
                    _interaction_entry(
                        QC_POLICY,
                        deep_model_id,
                        comparator_model_id,
                        endpoint,
                        available=False,
                        reason=QC_REASON,
                    )
                )
    return entries


def _unavailable_contrast(entry):
    return {
        "available": False,
        "reason": entry["reason"],
        "contrast_id": entry["contrast_id"],
        "family_id": entry["family_id"],
        "endpoint": entry["endpoint"],
        "estimate": None,
        "point": None,
        "weighted": None,
        "hierarchy": None,
        "descriptive": None,
        "sign_sensitivity": None,
    }


# ---------------------------------------------------------------------------
# Panel validation and exact wide alignment
# ---------------------------------------------------------------------------


def _validate_panels(panels):
    if not isinstance(panels, Mapping):
        raise TypeError("panels must be a mapping")
    if set(panels.keys()) != set(ESTIMANDS):
        raise ValueError(
            "panels must contain exactly equal_context and pooled_four_fold"
        )
    for estimand in ESTIMANDS:
        by_policy = panels[estimand]
        if not isinstance(by_policy, Mapping) or set(by_policy.keys()) != set(POLICIES):
            raise ValueError("panels policy coverage mismatch")
        for policy_id in POLICIES:
            by_endpoint = by_policy[policy_id]
            if not isinstance(by_endpoint, Mapping) or set(by_endpoint.keys()) != set(
                ENDPOINTS
            ):
                raise ValueError("panels endpoint coverage mismatch")
            for endpoint in ENDPOINTS:
                frame = by_endpoint[endpoint]
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("panel endpoint must be a pandas DataFrame")
                if frame.shape[0] == 0:
                    raise ValueError("panel endpoint must be nonempty")
                if frame.columns.duplicated().any():
                    raise ValueError("panel endpoint has duplicate columns")
                missing = [
                    column for column in _ID_COLUMNS if column not in frame.columns
                ]
                for required in ("model_id", "correct"):
                    if required not in frame.columns:
                        missing.append(required)
                if missing:
                    raise ValueError(f"panel endpoint is missing columns: {missing}")
                present = sorted(set(frame["model_id"].astype(str).tolist()))
                if present != sorted(MODELS):
                    raise ValueError("panel endpoint model coverage mismatch")


def _align_wide(panels, estimand, endpoint):
    """Return one wide frame with exact sorted row alignment on the identity key.

    Every policy/model slice must expose exactly the same identity set.  Missing,
    extra, changed or duplicated identities raise instead of being intersected or
    silently dropped.
    """

    slices = {}
    reference_keys = None
    reference_identity = None
    for policy_id in POLICIES:
        frame = panels[estimand][policy_id][endpoint]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("panel endpoint must be a pandas DataFrame")
        if frame.columns.duplicated().any():
            raise ValueError("panel endpoint has duplicate columns")
        model_values = frame["model_id"].astype(str)
        for model_id in MODELS:
            sub = frame.loc[
                model_values == model_id, list(_ID_COLUMNS) + ["correct"]
            ].copy()
            sub = sub.sort_values(list(_ID_COLUMNS), kind="stable").reset_index(
                drop=True
            )
            keys = list(
                zip(
                    *(sub[column].astype(str).tolist() for column in _ID_COLUMNS),
                    strict=True,
                )
            )
            if len(set(keys)) != len(keys):
                raise ValueError("panel duplicate identity key")
            if reference_keys is None:
                reference_keys = keys
                reference_identity = sub.loc[:, list(_ID_COLUMNS)].copy()
            elif keys != reference_keys:
                raise ValueError("panel identity support mismatch")
            slices[(policy_id, model_id)] = sub
    wide = reference_identity.copy()
    for (policy_id, model_id), sub in slices.items():
        wide[_procedure_column(policy_id, model_id)] = sub["correct"].to_numpy()
    return wide


def _base_design(wide):
    frame = wide.loc[:, list(_ID_COLUMNS)].copy()
    frame["correct_model"] = 0
    frame["correct_reference"] = 0
    return compile_pair(frame)


def _check_wide_identities(wide, global_masters, global_instruments):
    """Refuse a wide frame whose identities lie outside the global lists."""

    masters = set(wide["master_sample_id"].astype(str).tolist())
    instruments = set(wide["instrument"].astype(str).tolist())
    unknown_masters = sorted(masters.difference(global_masters))
    if unknown_masters:
        raise ValueError(
            f"wide frame masters not covered by global_masters: {unknown_masters}"
        )
    unknown_instruments = sorted(instruments.difference(global_instruments))
    if unknown_instruments:
        raise ValueError(
            "wide frame instruments not covered by global_instruments: "
            f"{unknown_instruments}"
        )
    return wide


# ---------------------------------------------------------------------------
# Support-only hierarchical replay
# ---------------------------------------------------------------------------


def hierarchy_support(design, *, draws, seed):
    """Replay the historical hierarchy for domain-name support diagnostics.

    This is a support-only replay.  It reproduces the identical historical RNG
    order (full ``(draws, n_domains)`` domain-index array first, then source
    domains in order, sorted class pools and multinomial master weights, batches
    of at most 128) without using any outcome.  It returns the same per-draw
    totals as the inherited sampler plus the affected-domain names and flags
    that the inherited sampler does not expose.  No draw is repaired, dropped or
    rerandomized, and no estimate is derived from undefined draws.
    """

    n_draws = _require_int(draws, "draws", 1, MAX_DRAWS)
    n_seed = _require_int(seed, "seed", 0)
    counts, _delta, _factor, cell_domain = _validated_arrays(design)
    master_classes = tuple(design.master_classes)
    domains = tuple(design.domains)
    n_domains = len(domains)
    n_masters = counts.shape[1]

    rng = np.random.Generator(np.random.PCG64(n_seed))
    sampled = rng.integers(0, n_domains, size=(n_draws, n_domains))

    empty = np.zeros((n_draws, n_domains), dtype=np.intp)
    bad = np.zeros((n_draws, n_domains), dtype=bool)
    affected = np.zeros((n_draws, n_domains), dtype=bool)
    per_domain_empty = np.zeros(n_domains, dtype=np.intp)
    per_domain_undefined = np.zeros(n_domains, dtype=np.intp)

    for source in range(n_domains):
        domain_cells = np.flatnonzero(cell_domain == source)
        locations = np.argwhere(sampled == source)
        if domain_cells.size:
            present = np.any(counts[domain_cells] > 0.0, axis=0)
        else:
            present = np.zeros(n_masters, dtype=bool)

        weights = np.zeros((locations.shape[0], n_masters), dtype=float)
        for _label, pool in _class_pools(master_classes, present):
            size = len(pool)
            weights[:, pool] = rng.multinomial(
                size, np.full(size, 1.0 / size), size=locations.shape[0]
            )
        if locations.shape[0] == 0:
            continue

        domain_counts = counts[domain_cells]
        for start in range(0, locations.shape[0], _BATCH):
            stop = min(start + _BATCH, locations.shape[0])
            block = weights[start:stop]
            denominator = block @ domain_counts.T
            if not np.all(np.isfinite(denominator)):
                raise ValueError("numerical overflow")
            missing = denominator == 0.0
            n_empty = missing.sum(axis=1)
            flags = n_empty > 0
            rows = locations[start:stop, 0]
            slots = locations[start:stop, 1]
            empty[rows, slots] = n_empty
            bad[rows, slots] = flags
            per_domain_empty[source] += int(n_empty.sum())
            per_domain_undefined[source] += int(flags.sum())
            if flags.any():
                affected[rows[flags], source] = True

    empty_per_draw = empty.sum(axis=1)
    undefined_per_draw = bad.sum(axis=1)
    affected_per_draw = affected.sum(axis=1)
    defined = undefined_per_draw == 0
    affected_names = [
        domains[index] for index in range(n_domains) if bool(affected[:, index].any())
    ]
    return {
        "seed": n_seed,
        "draws": n_draws,
        "domains": list(domains),
        "sampled_domains": sampled,
        "empty_cells": empty_per_draw,
        "undefined_domain_occurrences": undefined_per_draw,
        "affected_domains": affected_per_draw,
        "affected_domain_flags": affected,
        "affected_domain_names": sorted(affected_names),
        "affected_draws_by_domain": {
            domains[index]: int(affected[:, index].sum()) for index in range(n_domains)
        },
        "undefined_occurrences_by_domain": {
            domains[index]: int(per_domain_undefined[index])
            for index in range(n_domains)
        },
        "empty_cells_by_domain": {
            domains[index]: int(per_domain_empty[index]) for index in range(n_domains)
        },
        "planned_draws": n_draws,
        "defined_draws": int(defined.sum()),
        "undefined_draws": int(n_draws - int(defined.sum())),
    }


# ---------------------------------------------------------------------------
# Multiplicity and summary
# ---------------------------------------------------------------------------


def _cross_check_support(registry, contrasts, support):
    for estimand in ESTIMANDS:
        for endpoint in ENDPOINTS:
            replay = support[estimand][endpoint]
            for entry in registry:
                if entry["endpoint"] != endpoint or not entry["available"]:
                    continue
                hierarchy = contrasts[estimand][entry["contrast_id"]]["hierarchy"]
                for key in (
                    "sampled_domains",
                    "empty_cells",
                    "undefined_domain_occurrences",
                    "affected_domains",
                ):
                    if not np.array_equal(hierarchy[key], replay[key]):
                        raise RuntimeError(
                            "hierarchy_support_cross_check_failed:"
                            f"{estimand}:{entry['contrast_id']}:{key}"
                        )


def _adjust_multiplicity(registry, contrasts):
    for estimand in ESTIMANDS:
        for family_id in (
            FAMILY_UNIVERSAL_POLICY,
            FAMILY_POLICY_MODEL_INTERACTION,
        ):
            family = [entry for entry in registry if entry["family_id"] == family_id]
            domain_raw = []
            instrument_raw = []
            for entry in family:
                result = contrasts[estimand][entry["contrast_id"]]
                sign = result.get("sign_sensitivity")
                if entry["available"] and sign:
                    domain_raw.append(sign["domain"]["p_descriptive"])
                    instrument_raw.append(sign["instrument"]["p_descriptive"])
                else:
                    domain_raw.append(None)
                    instrument_raw.append(None)
            domain_adjusted = holm_fixed_family(domain_raw, family_size=len(family))
            instrument_adjusted = holm_fixed_family(
                instrument_raw, family_size=len(family)
            )
            for entry, draw, daj, iraw, iaj in zip(
                family,
                domain_raw,
                domain_adjusted,
                instrument_raw,
                instrument_adjusted,
                strict=True,
            ):
                contrasts[estimand][entry["contrast_id"]]["sensitivity_adjustment"] = {
                    "estimand": estimand,
                    "family_id": family_id,
                    "family_size": len(family),
                    "adjustment": "holm",
                    "domain_raw_p": draw,
                    "domain_adjusted_p": daj,
                    "instrument_raw_p": iraw,
                    "instrument_adjusted_p": iaj,
                }


def _summary_row(estimand, entry, result):
    row = {column: None for column in _SUMMARY_COLUMNS}
    row["estimand"] = estimand
    row["contrast_id"] = entry["contrast_id"]
    row["family_id"] = entry["family_id"]
    row["endpoint"] = entry["endpoint"]
    row["model_id"] = entry["model_id"]
    row["policy_id"] = entry["policy_id"]
    row["deep_model_id"] = entry["deep_model_id"]
    row["comparator_model_id"] = entry["comparator_model_id"]
    row["available"] = bool(entry["available"])
    row["reason"] = None if entry["available"] else entry["reason"]

    adjustment = result.get("sensitivity_adjustment") or {}
    row["family_size"] = adjustment.get("family_size")
    row["adjustment"] = adjustment.get("adjustment")
    row["domain_raw_p"] = adjustment.get("domain_raw_p")
    row["domain_adjusted_p"] = adjustment.get("domain_adjusted_p")
    row["instrument_raw_p"] = adjustment.get("instrument_raw_p")
    row["instrument_adjusted_p"] = adjustment.get("instrument_adjusted_p")

    if not entry["available"]:
        return row

    row["point_effect"] = float(result["point"]["overall"])
    weighted = result["weighted"]
    for mode in ("crossed", "master_only", "instrument_only"):
        summary = weighted[mode]["overall"]["summary"]
        row[f"{mode}_lower"] = summary["lower"]
        row[f"{mode}_upper"] = summary["upper"]
        row[f"{mode}_reason"] = summary["reason_code"]

    hierarchy = result["hierarchy"]["combined"]["summary"]
    row["hierarchy_planned"] = hierarchy["planned_draws"]
    row["hierarchy_defined"] = hierarchy["defined_draws"]
    row["hierarchy_undefined"] = hierarchy["undefined_draws"]
    row["hierarchy_reason"] = hierarchy["reason_code"]
    return row


def _build_summary(registry, contrasts):
    rows = []
    for estimand in ESTIMANDS:
        for entry in registry:
            rows.append(
                _summary_row(estimand, entry, contrasts[estimand][entry["contrast_id"]])
            )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_panel(panels, *, global_masters, global_instruments, domain_families=None):
    """Analyze the approved universal preprocessing results in memory.

    ``panels`` is the exact output of ``p08_universal_units.build_units``.  The
    two global weight arrays are generated once and shared, unchanged, by every
    available contrast under both estimands.
    """

    g_masters = _identity_list(global_masters, "global_masters")
    g_instruments = _identity_list(global_instruments, "global_instruments")
    if domain_families is not None and not isinstance(domain_families, Mapping):
        raise TypeError("domain_families must be a mapping or None")
    _validate_panels(panels)
    draw_count = _require_int(DRAW_COUNT, "DRAW_COUNT", 1, MAX_DRAWS)

    master_weights = np.random.Generator(np.random.PCG64(MASTER_SEED)).exponential(
        1.0, size=(draw_count, len(g_masters))
    )
    instrument_weights = np.random.Generator(
        np.random.PCG64(INSTRUMENT_SEED)
    ).exponential(1.0, size=(draw_count, len(g_instruments)))
    _check_weights(master_weights, "master")
    _check_weights(instrument_weights, "instrument")

    registry = contrast_registry()

    metrics = {}
    contrasts = {}
    support = {}
    for estimand in ESTIMANDS:
        metrics[estimand] = {}
        contrasts[estimand] = {}
        support[estimand] = {}
        for policy_id in POLICIES:
            metrics[estimand][policy_id] = {}
            for endpoint in ENDPOINTS:
                frame = panels[estimand][policy_id][endpoint]
                metrics[estimand][policy_id][endpoint] = _units.summarize_units(frame)

    wide_cache = {}
    for estimand in ESTIMANDS:
        wide_cache[estimand] = {}
        for endpoint in ENDPOINTS:
            wide = _align_wide(panels, estimand, endpoint)
            _check_wide_identities(wide, g_masters, g_instruments)
            wide_cache[estimand][endpoint] = wide

    for estimand in ESTIMANDS:
        for endpoint in ENDPOINTS:
            wide = wide_cache[estimand][endpoint]
            base_design = _base_design(wide)
            support[estimand][endpoint] = hierarchy_support(
                base_design, draws=draw_count, seed=HIERARCHY_SEED
            )
            for entry in registry:
                if entry["endpoint"] != endpoint:
                    continue
                contrast_id = entry["contrast_id"]
                if not entry["available"]:
                    contrasts[estimand][contrast_id] = _unavailable_contrast(entry)
                    continue
                columns = [
                    _procedure_column(label[0], label[1])
                    for label in entry["procedure_labels"]
                ]
                frame = wide.loc[:, list(_ID_COLUMNS) + columns].copy()
                contrasts[estimand][contrast_id] = analyze_contrast(
                    frame,
                    columns,
                    tuple(entry["coefficients"]),
                    global_masters=g_masters,
                    global_instruments=g_instruments,
                    master_weights=master_weights,
                    instrument_weights=instrument_weights,
                    hierarchical_draws=draw_count,
                    hierarchical_seed=HIERARCHY_SEED,
                    domain_families=domain_families,
                )

    _cross_check_support(registry, contrasts, support)
    _adjust_multiplicity(registry, contrasts)
    contrast_summary = _build_summary(registry, contrasts)

    boundary = {label: True for label in _BOUNDARY_LABELS}
    result = {
        "registry": registry,
        "weights": {
            "number_draws": draw_count,
            "master_seed": MASTER_SEED,
            "instrument_seed": INSTRUMENT_SEED,
            "hierarchy_seed": HIERARCHY_SEED,
            "global_masters": list(g_masters),
            "global_instruments": list(g_instruments),
            "master_weights": master_weights,
            "instrument_weights": instrument_weights,
            "generated_once": (
                "both weight arrays are generated exactly once per analyze_panel "
                "call and shared unchanged by all available contrasts and both "
                "estimands"
            ),
        },
        "metrics": metrics,
        "contrasts": contrasts,
        "contrast_summary": contrast_summary,
        "hierarchy_support": support,
        "boundary": boundary,
    }
    for label in _BOUNDARY_LABELS:
        result[label] = True
    return result
