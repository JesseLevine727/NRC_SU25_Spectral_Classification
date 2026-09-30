"""Reduced pure weighted-score engine for p06-p11 inference (no hierarchical sampling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ID_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
)
_REQUIRED_COLUMNS = _ID_COLUMNS + ("correct_model", "correct_reference")
_MAX_DRAWS = 10000
_BATCH_ROWS = 128


@dataclass(frozen=True)
class PairedDesign:
    masters: tuple
    instruments: tuple
    domains: tuple
    cell_keys: tuple
    counts: np.ndarray
    delta_correct: np.ndarray
    cell_domain: np.ndarray
    domain_instrument: np.ndarray
    cell_factor: np.ndarray
    master_classes: tuple
    master_stations: tuple


def _bind(mapping, key, value, name):
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = value
    elif existing != value:
        raise ValueError(f"inconsistent_{name}_identity")


def _identity_values(rows, column):
    for value in rows[column].tolist():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{column} must contain trimmed nonempty strings")


def _binary_values(rows, column):
    array = rows[column].to_numpy()
    result = np.empty(array.shape[0], dtype=np.int64)
    for index, value in enumerate(array.tolist()):
        if isinstance(value, (bool, np.bool_)):
            result[index] = int(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{column} must be bool or finite 0/1 numeric")
            if number not in (0.0, 1.0):
                raise ValueError(f"{column} must contain only 0/1")
            result[index] = int(number)
            continue
        raise ValueError(f"{column} must be bool or finite 0/1 numeric")
    return result


def compile_pair(rows):
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    if rows.shape[0] == 0:
        raise ValueError("rows must be nonempty")
    if not rows.columns.is_unique:
        raise ValueError("rows must have unique columns")
    missing = [c for c in _REQUIRED_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for column in _ID_COLUMNS:
        _identity_values(rows, column)
    model = _binary_values(rows, "correct_model")
    reference = _binary_values(rows, "correct_reference")
    ctx = rows["context_id"].tolist()
    dom = rows["domain"].tolist()
    station = rows["station"].tolist()
    instrument = rows["instrument"].tolist()
    master = rows["master_sample_id"].tolist()
    unit = rows["unit_id"].tolist()
    label = rows["true_label"].tolist()
    context_map, domain_map, master_map, unit_map = {}, {}, {}, {}
    seen_pairs = set()
    for i in range(len(ctx)):
        pair = (ctx[i], unit[i])
        if pair in seen_pairs:
            raise ValueError("duplicate_context_unit")
        seen_pairs.add(pair)
        _bind(context_map, ctx[i], (dom[i], station[i], instrument[i]), "context")
        _bind(domain_map, dom[i], (station[i], instrument[i]), "domain")
        _bind(master_map, master[i], (station[i], label[i]), "master")
        _bind(unit_map, unit[i], (master[i], station[i], instrument[i], label[i]), "unit")
    masters = tuple(sorted(master_map))
    instruments = tuple(sorted(set(instrument)))
    domains = tuple(sorted(domain_map))
    cell_keys = tuple(sorted({(ctx[i], label[i]) for i in range(len(ctx))}))
    master_index = {m: i for i, m in enumerate(masters)}
    instrument_index = {v: i for i, v in enumerate(instruments)}
    domain_index = {d: i for i, d in enumerate(domains)}
    cell_index = {k: i for i, k in enumerate(cell_keys)}
    counts = np.zeros((len(cell_keys), len(masters)), dtype=float)
    delta = np.zeros_like(counts)
    diff = model - reference
    for i in range(len(ctx)):
        c = cell_index[(ctx[i], label[i])]
        m = master_index[master[i]]
        counts[c, m] += 1.0
        delta[c, m] += float(diff[i])
    cell_domain = np.array([domain_index[context_map[k[0]][0]] for k in cell_keys], dtype=np.int64)
    domain_instrument = np.array(
        [instrument_index[domain_map[d][1]] for d in domains], dtype=np.int64
    )
    context_classes, domain_contexts = {}, {}
    for context, _ in cell_keys:
        context_classes[context] = context_classes.get(context, 0) + 1
    for context in context_map:
        d = context_map[context][0]
        domain_contexts[d] = domain_contexts.get(d, 0) + 1
    cell_factor = np.array(
        [1.0 / (context_classes[k[0]] * domain_contexts[context_map[k[0]][0]]) for k in cell_keys],
        dtype=float,
    )
    master_classes = tuple(master_map[m][1] for m in masters)
    master_stations = tuple(master_map[m][0] for m in masters)
    for array in (counts, delta, cell_domain, domain_instrument, cell_factor):
        array.setflags(write=False)
    return PairedDesign(
        masters=masters,
        instruments=instruments,
        domains=domains,
        cell_keys=cell_keys,
        counts=counts,
        delta_correct=delta,
        cell_domain=cell_domain,
        domain_instrument=domain_instrument,
        cell_factor=cell_factor,
        master_classes=master_classes,
        master_stations=master_stations,
    )


def _positive_matrix(values, width, name):
    if isinstance(values, (bool, np.bool_, str, complex)):
        raise TypeError(f"{name} must be a finite positive numeric 2D array")
    array = np.asarray(values)
    if array.dtype == object or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must have at least one row")
    if array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.all(array > 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.asarray(array, dtype=float)


def score_weights(design, master_weights, instrument_weights):
    master_matrix = _positive_matrix(master_weights, len(design.masters), "master_weights")
    instrument_matrix = _positive_matrix(
        instrument_weights, len(design.instruments), "instrument_weights"
    )
    if master_matrix.shape[0] != instrument_matrix.shape[0]:
        raise ValueError("master_weights and instrument_weights must share batch size")
    cells = len(design.cell_keys)
    domains = len(design.domains)
    onehot = np.zeros((cells, domains), dtype=float)
    onehot[np.arange(cells), design.cell_domain] = 1.0
    batch = master_matrix.shape[0]
    overall = np.empty(batch, dtype=float)
    domain_out = np.empty((batch, domains), dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, batch, _BATCH_ROWS):
                stop = min(start + _BATCH_ROWS, batch)
                mw = master_matrix[start:stop]
                iw = instrument_matrix[start:stop]
                numerator = mw @ design.delta_correct.T
                denominator = mw @ design.counts.T
                if not (np.all(np.isfinite(numerator)) and np.all(np.isfinite(denominator))):
                    raise ValueError("nonfinite_score_weights")
                cell_delta = numerator / denominator
                domain_delta = (cell_delta * design.cell_factor) @ onehot
                weights = iw[:, design.domain_instrument]
                weight_sum = weights.sum(axis=1)
                overall[start:stop] = (domain_delta * weights).sum(axis=1) / weight_sum
                domain_out[start:stop] = domain_delta
                chunk_ok = (
                    np.all(np.isfinite(cell_delta))
                    and np.all(np.isfinite(domain_delta))
                    and np.all(np.isfinite(weight_sum))
                    and np.all(np.isfinite(overall[start:stop]))
                    and np.all(np.isfinite(domain_out[start:stop]))
                )
                if not chunk_ok:
                    raise ValueError("nonfinite_score_weights")
    except FloatingPointError:
        raise ValueError("nonfinite_score_weights") from None
    return overall, domain_out


def _identity_sequence(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of distinct nonempty strings")
    try:
        sequence = list(values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of distinct nonempty strings") from None
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


def positive_weights(
    masters, instruments, *, draws=10000, master_seed=2026092901, instrument_seed=2026092902
):
    master_ids = _identity_sequence(masters, "masters")
    instrument_ids = _identity_sequence(instruments, "instruments")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)):
        raise TypeError("draws must be an integer")
    draws = int(draws)
    if draws < 1 or draws > _MAX_DRAWS:
        raise ValueError(f"draws must be in 1..{_MAX_DRAWS}")
    for seed, name in ((master_seed, "master_seed"), (instrument_seed, "instrument_seed")):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"{name} must be a nonnegative integer")
        if int(seed) < 0:
            raise ValueError(f"{name} must be nonnegative")
    master_draws = np.random.Generator(np.random.PCG64(int(master_seed))).exponential(
        1.0, size=(draws, len(master_ids))
    )
    instrument_draws = np.random.Generator(np.random.PCG64(int(instrument_seed))).exponential(
        1.0, size=(draws, len(instrument_ids))
    )
    if not (np.all(master_draws > 0) and np.all(np.isfinite(master_draws))):
        raise ValueError("master draws must be finite and positive")
    if not (np.all(instrument_draws > 0) and np.all(np.isfinite(instrument_draws))):
        raise ValueError("instrument draws must be finite and positive")
    return master_draws, instrument_draws


def _float_vector(values, name):
    if isinstance(values, (bool, np.bool_, str, complex)):
        raise TypeError(f"{name} must be numeric")
    array = np.asarray(values)
    if array.dtype == object or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return np.asarray(array, dtype=float)


def summarize_draws(values):
    array = _float_vector(values, "values")
    if array.size == 0:
        raise ValueError("values must be nonempty")
    if array.size > _MAX_DRAWS:
        raise ValueError(f"values must hold at most {_MAX_DRAWS} draws")
    if np.isinf(array).any():
        raise ValueError("values must not contain infinite draws")
    planned = int(array.size)
    defined = int(np.isfinite(array).sum())
    undefined = planned - defined
    if undefined:
        return {
            "planned_draws": planned,
            "defined_draws": defined,
            "undefined_draws": undefined,
            "lower": None,
            "upper": None,
            "reason_code": "hierarchical_fixed_support_undefined",
        }
    lower = float(np.quantile(array, 0.025, method="linear"))
    upper = float(np.quantile(array, 0.975, method="linear"))
    reason = "degenerate_distribution" if bool(np.all(array == array[0])) else "ok"
    return {
        "planned_draws": planned,
        "defined_draws": defined,
        "undefined_draws": undefined,
        "lower": lower,
        "upper": upper,
        "reason_code": reason,
    }
