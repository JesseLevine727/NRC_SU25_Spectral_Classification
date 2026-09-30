"""Original hierarchical-bootstrap feasibility sampler (T013).

Source domains are resampled with replacement.  For every resampled source
domain occurrence, each original class pool receives a single multinomial
occurrence-weight draw.  Cell-level ratios are combined with the supplied
cell factors.  A zero denominator marks an originally represented class
cell as empty and leaves the whole domain occurrence undefined (NaN); cells
are never dropped or zero-imputed.
"""

from __future__ import annotations

import numpy as np

from atlas_sers.evaluation.p06p11_inference import PairedDesign

__all__ = ["hierarchical_draws"]

_DEFAULT_DRAWS = 10000
_DEFAULT_SEED = 2026092903
_MAX_DRAWS = 10000
_BATCH = 128


def _as_int(value, *, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return integer


def _valid_identities(values):
    if not values:
        return False
    if any(not isinstance(value, str) or not value for value in values):
        return False
    return list(values) == sorted(values) and len(set(values)) == len(values)


def _numeric_matrix(value, name):
    array = np.asarray(value)
    if array.dtype == bool or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    array = array.astype(float, copy=False)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _domain_index(value, n_cells, n_domains):
    raw = np.asarray(value)
    if raw.dtype == bool or raw.dtype.kind not in "iu":
        raise ValueError("cell_domain must be an integer array")
    cell_domain = raw.astype(np.intp, copy=False)
    if cell_domain.shape != (n_cells,):
        raise ValueError("cell_domain must align with cells")
    if np.any(cell_domain < 0) or np.any(cell_domain >= n_domains):
        raise ValueError("cell_domain values out of range")
    if np.unique(cell_domain).size != n_domains:
        raise ValueError("every domain must be represented by a cell")
    return cell_domain


def _factor_vector(value, n_cells):
    raw = np.asarray(value)
    if raw.dtype == bool or raw.dtype.kind not in "iuf":
        raise ValueError("cell_factor must be a real numeric array")
    cell_factor = raw.astype(float, copy=False)
    if cell_factor.shape != (n_cells,):
        raise ValueError("cell_factor must align with cells")
    if not np.isfinite(cell_factor).all() or np.any(cell_factor <= 0.0):
        raise ValueError("cell_factor must be positive and finite")
    return cell_factor


def _validated_arrays(design):
    if not isinstance(design, PairedDesign):
        raise TypeError("design must be a PairedDesign instance")

    domains = tuple(design.domains)
    masters = tuple(design.masters)
    if not _valid_identities(domains):
        raise ValueError("domains must be sorted unique non-empty strings")
    if not _valid_identities(masters):
        raise ValueError("masters must be sorted unique non-empty strings")
    master_classes = tuple(design.master_classes)
    if len(master_classes) != len(masters):
        raise ValueError("master_classes must align with masters")
    if any(not isinstance(name, str) or not name for name in master_classes):
        raise ValueError("master_classes must be non-empty strings")

    counts = _numeric_matrix(design.counts, "counts")
    delta = _numeric_matrix(design.delta_correct, "delta_correct")
    if counts.shape != delta.shape:
        raise ValueError("counts and delta_correct must be matching matrices")
    n_cells, n_masters = counts.shape
    if n_cells < 1 or n_masters < 1:
        raise ValueError("counts must be a non-empty matrix")
    if len(domains) < 1:
        raise ValueError("design must contain at least one domain")
    if n_masters != len(masters):
        raise ValueError("counts columns must align with masters")
    if n_cells != len(tuple(design.cell_keys)):
        raise ValueError("counts rows must align with cell_keys")
    if not np.isfinite(counts).all() or not np.isfinite(delta).all():
        raise ValueError("counts and delta_correct must be finite")
    if np.any(counts < 0.0):
        raise ValueError("counts must be non-negative")
    if np.any(counts != np.floor(counts)):
        raise ValueError("counts must be integer-valued")
    if np.any(np.abs(delta) > counts):
        raise ValueError("delta_correct must not exceed counts")
    if np.any(counts.sum(axis=1) <= 0.0):
        raise ValueError("every cell must have a positive total")

    cell_domain = _domain_index(design.cell_domain, n_cells, len(domains))
    cell_factor = _factor_vector(design.cell_factor, n_cells)
    for domain in range(len(domains)):
        if not np.isclose(
            cell_factor[cell_domain == domain].sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("cell_factor must sum to one within each domain")
    return counts, delta, cell_factor, cell_domain


def _class_pools(master_classes, present):
    pools = {}
    for master in np.flatnonzero(present):
        pools.setdefault(master_classes[int(master)], []).append(int(master))
    return sorted(pools.items())


def hierarchical_draws(design, *, draws=_DEFAULT_DRAWS, seed=_DEFAULT_SEED):
    """Return the historical hierarchical-bootstrap feasibility draws."""
    n_draws = _as_int(draws, name="draws", minimum=1, maximum=_MAX_DRAWS)
    n_seed = _as_int(seed, name="seed", minimum=0)
    counts, delta, cell_factor, cell_domain = _validated_arrays(design)

    master_classes = tuple(design.master_classes)
    n_domains = len(tuple(design.domains))
    if n_domains < 1:
        raise ValueError("design must contain at least one domain")
    n_masters = counts.shape[1]

    rng = np.random.Generator(np.random.PCG64(n_seed))
    sampled = rng.integers(0, n_domains, size=(n_draws, n_domains))

    scores = np.full((n_draws, n_domains), np.nan, dtype=float)
    empty = np.zeros((n_draws, n_domains), dtype=np.intp)
    bad = np.zeros((n_draws, n_domains), dtype=bool)
    affected = np.zeros((n_draws, n_domains), dtype=bool)

    for source in range(n_domains):
        domain_cells = np.flatnonzero(cell_domain == source)
        locations = np.argwhere(sampled == source)
        if domain_cells.size:
            present = np.any(counts[domain_cells] > 0.0, axis=0)
        else:
            present = np.zeros(n_masters, dtype=bool)

        weights = np.zeros((locations.shape[0], n_masters), dtype=float)
        for _, pool in _class_pools(master_classes, present):
            size = len(pool)
            weights[:, pool] = rng.multinomial(
                size, np.full(size, 1.0 / size), size=locations.shape[0]
            )
        if locations.shape[0] == 0:
            continue

        domain_counts = counts[domain_cells]
        domain_delta = delta[domain_cells]
        domain_factor = cell_factor[domain_cells]
        for start in range(0, locations.shape[0], _BATCH):
            stop = min(start + _BATCH, locations.shape[0])
            block = weights[start:stop]
            denominator = block @ domain_counts.T
            numerator = block @ domain_delta.T
            if not np.isfinite(denominator).all() or not np.isfinite(numerator).all():
                raise ValueError("numerical overflow while scoring")
            missing = denominator == 0.0
            n_empty = missing.sum(axis=1)
            defined = n_empty == 0
            block_score = np.full(stop - start, np.nan, dtype=float)
            if defined.any():
                ratio = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(denominator),
                    where=~missing,
                )
                block_score[defined] = (ratio * domain_factor)[defined].sum(axis=1)
            if defined.any() and not np.isfinite(block_score[defined]).all():
                raise ValueError("numerical overflow while scoring")
            rows = locations[start:stop, 0]
            slots = locations[start:stop, 1]
            scores[rows, slots] = block_score
            empty[rows, slots] = n_empty
            flags = n_empty > 0
            bad[rows, slots] = flags
            if flags.any():
                affected[rows[flags], source] = True

    mean_scores = scores.mean(axis=1)
    finite = np.isfinite(mean_scores)
    if finite.any() and (
        np.any(mean_scores[finite] < -1.0 - 1e-12)
        or np.any(mean_scores[finite] > 1.0 + 1e-12)
    ):
        raise ValueError("draws fall outside the expected bounds")
    return {
        "draws": mean_scores,
        "empty_cells": empty.sum(axis=1),
        "undefined_domain_occurrences": bad.sum(axis=1),
        "affected_domains": affected.sum(axis=1),
        "sampled_domains": sampled,
    }
