"""Synthetic tests for the T013 hierarchical-bootstrap sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from atlas_sers.evaluation.p06p11_hierarchy import hierarchical_draws
from atlas_sers.evaluation.p06p11_inference import PairedDesign


def _make_design(
    *,
    masters,
    classes,
    stations,
    instruments,
    domains,
    cell_keys,
    counts,
    cell_domain,
    domain_instrument,
    cell_factor,
    delta_correct=None,
):
    counts = np.asarray(counts, dtype=float)
    return PairedDesign(
        masters=tuple(masters),
        instruments=tuple(instruments),
        domains=tuple(domains),
        cell_keys=tuple(cell_keys),
        counts=counts,
        delta_correct=(
            counts if delta_correct is None else np.asarray(delta_correct, dtype=float)
        ),
        cell_domain=np.asarray(cell_domain, dtype=np.intp),
        domain_instrument=np.asarray(domain_instrument, dtype=np.intp),
        cell_factor=np.asarray(cell_factor, dtype=float),
        master_classes=tuple(classes),
        master_stations=tuple(stations),
    )


def _singleton_design():
    return _make_design(
        masters=("ma", "mb"),
        classes=("c1", "c1"),
        stations=("s1", "s2"),
        instruments=("i0",),
        domains=("d0",),
        cell_keys=(("x0", "c1"), ("x1", "c1")),
        counts=[[1.0, 0.0], [0.0, 1.0]],
        cell_domain=[0, 0],
        domain_instrument=[0],
        cell_factor=[0.5, 0.5],
    )


def _reference_design():
    return _make_design(
        masters=("m0", "m1", "m2"),
        classes=("c0", "c0", "c1"),
        stations=("s0", "s1", "s2"),
        instruments=("i0", "i1"),
        domains=("d0", "d1"),
        cell_keys=(("x0", "c0"), ("x1", "c0"), ("x2", "c0"), ("x2", "c1")),
        counts=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        delta_correct=[
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        cell_domain=[0, 0, 1, 1],
        domain_instrument=[0, 1],
        cell_factor=[0.5, 0.5, 0.5, 0.5],
    )


def _with_delta(design, delta):
    return PairedDesign(
        masters=design.masters,
        instruments=design.instruments,
        domains=design.domains,
        cell_keys=design.cell_keys,
        counts=design.counts,
        delta_correct=np.asarray(delta, dtype=float),
        cell_domain=design.cell_domain,
        domain_instrument=design.domain_instrument,
        cell_factor=design.cell_factor,
        master_classes=design.master_classes,
        master_stations=design.master_stations,
    )


def _reference(design, *, draws, seed):
    counts = np.asarray(design.counts, dtype=float)
    delta = np.asarray(design.delta_correct, dtype=float)
    cell_domain = np.asarray(design.cell_domain, dtype=np.intp)
    cell_factor = np.asarray(design.cell_factor, dtype=float)
    classes = tuple(design.master_classes)
    n_domains = len(tuple(design.domains))
    n_masters = counts.shape[1]
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, n_domains, size=(draws, n_domains))
    weights_by_domain = {}
    for source in range(n_domains):
        locations = np.argwhere(sampled == source)
        cells = np.flatnonzero(cell_domain == source)
        present = (
            np.any(counts[cells] > 0.0, axis=0)
            if cells.size
            else np.zeros(n_masters, dtype=bool)
        )
        pools = {}
        for master in np.flatnonzero(present):
            pools.setdefault(classes[int(master)], []).append(int(master))
        weights = np.zeros((locations.shape[0], n_masters), dtype=float)
        for name in sorted(pools):
            pool = pools[name]
            size = len(pool)
            weights[:, pool] = rng.multinomial(
                size, np.full(size, 1.0 / size), size=locations.shape[0]
            )
        weights_by_domain[source] = (locations, weights, cells)

    score = np.full((draws, n_domains), np.nan, dtype=float)
    empty = np.zeros((draws, n_domains), dtype=int)
    bad = np.zeros((draws, n_domains), dtype=bool)
    affected = np.zeros((draws, n_domains), dtype=bool)
    for source in range(n_domains):
        locations, weights, cells = weights_by_domain[source]
        for occurrence in range(locations.shape[0]):
            draw, slot = int(locations[occurrence, 0]), int(locations[occurrence, 1])
            n_empty = 0
            total = 0.0
            for cell in cells:
                denominator = 0.0
                numerator = 0.0
                for master in range(weights.shape[1]):
                    denominator += weights[occurrence, master] * counts[cell, master]
                    numerator += weights[occurrence, master] * delta[cell, master]
                if denominator == 0.0:
                    n_empty += 1
                else:
                    total += (numerator / denominator) * cell_factor[cell]
            empty[draw, slot] = n_empty
            if n_empty == 0:
                score[draw, slot] = total
            else:
                bad[draw, slot] = True
                affected[draw, source] = True
    return {
        "draws": score.mean(axis=1),
        "empty_cells": empty.sum(axis=1),
        "undefined_domain_occurrences": bad.sum(axis=1),
        "affected_domains": affected.sum(axis=1),
        "sampled_domains": sampled,
    }


def test_singleton_contexts_produce_undefined_draws():
    design = _singleton_design()
    result = hierarchical_draws(design, draws=64, seed=12345)
    expected = _reference(design, draws=64, seed=12345)
    np.testing.assert_array_equal(result["empty_cells"], expected["empty_cells"])
    np.testing.assert_array_equal(
        result["undefined_domain_occurrences"],
        expected["undefined_domain_occurrences"],
    )
    np.testing.assert_array_equal(result["sampled_domains"], expected["sampled_domains"])
    assert np.any(result["empty_cells"] > 0)
    assert np.isnan(result["draws"]).any()


def test_class_with_both_masters_in_one_context_stays_defined():
    design = _make_design(
        masters=("ma", "mb"),
        classes=("c1", "c1"),
        stations=("s1", "s2"),
        instruments=("i0",),
        domains=("d0",),
        cell_keys=(("x0", "c1"),),
        counts=[[2.0, 3.0]],
        cell_domain=[0],
        domain_instrument=[0],
        cell_factor=[1.0],
    )
    result = hierarchical_draws(design, draws=32, seed=999)
    assert not np.isnan(result["draws"]).any()
    assert result["empty_cells"].sum() == 0
    assert result["undefined_domain_occurrences"].sum() == 0


def test_single_master_class_is_always_retained():
    design = _make_design(
        masters=("ma",),
        classes=("c1",),
        stations=("s1",),
        instruments=("i0",),
        domains=("d0",),
        cell_keys=(("x0", "c1"),),
        counts=[[3.0]],
        cell_domain=[0],
        domain_instrument=[0],
        cell_factor=[1.0],
    )
    result = hierarchical_draws(design, draws=16, seed=7)
    assert not np.isnan(result["draws"]).any()
    np.testing.assert_allclose(result["draws"], 1.0)


def test_matches_manual_reference_across_batching():
    design = _reference_design()
    result = hierarchical_draws(design, draws=200, seed=2026092903)
    expected = _reference(design, draws=200, seed=2026092903)
    for key in (
        "empty_cells",
        "undefined_domain_occurrences",
        "affected_domains",
        "sampled_domains",
    ):
        np.testing.assert_array_equal(result[key], expected[key])
    np.testing.assert_allclose(result["draws"], expected["draws"], equal_nan=True)
    assert np.any(result["undefined_domain_occurrences"] > 0)
    assert np.any(result["affected_domains"] < result["undefined_domain_occurrences"])


def test_two_domains_share_a_master_with_distinct_instruments():
    design = _reference_design()
    assert design.domain_instrument[0] != design.domain_instrument[1]
    assert design.counts[0, 0] > 0.0 and design.counts[2, 0] > 0.0
    result = hierarchical_draws(design, draws=40, seed=17)
    assert result["draws"].shape == (40,)
    assert result["sampled_domains"].shape == (40, 2)
    assert set(np.unique(result["sampled_domains"])) <= {0, 1}


def test_deterministic_for_fixed_seed():
    design = _reference_design()
    first = hierarchical_draws(design, draws=50, seed=42)
    second = hierarchical_draws(design, draws=50, seed=42)
    for key, value in first.items():
        np.testing.assert_array_equal(value, second[key])


def test_negating_delta_reverses_defined_scores():
    design = _reference_design()
    flipped = _with_delta(design, -np.asarray(design.delta_correct, dtype=float))
    base = hierarchical_draws(design, draws=100, seed=5)
    other = hierarchical_draws(flipped, draws=100, seed=5)
    finite = np.isfinite(base["draws"])
    np.testing.assert_array_equal(finite, np.isfinite(other["draws"]))
    np.testing.assert_allclose(other["draws"][finite], -base["draws"][finite], atol=1e-12)


def test_zero_delta_gives_zero_defined_scores():
    design = _reference_design()
    zeroed = _with_delta(design, np.zeros_like(design.delta_correct))
    result = hierarchical_draws(zeroed, draws=64, seed=11)
    finite = np.isfinite(result["draws"])
    assert finite.any()
    np.testing.assert_allclose(result["draws"][finite], 0.0, atol=0.0)


def test_design_arrays_are_not_mutated():
    design = _reference_design()
    copies = {
        "counts": np.array(design.counts, copy=True),
        "delta_correct": np.array(design.delta_correct, copy=True),
        "cell_factor": np.array(design.cell_factor, copy=True),
        "cell_domain": np.array(design.cell_domain, copy=True),
    }
    hierarchical_draws(design, draws=20, seed=3)
    for name, value in copies.items():
        np.testing.assert_array_equal(getattr(design, name), value)


def test_global_numpy_rng_is_not_used():
    design = _singleton_design()
    np.random.seed(1234)
    before = np.random.get_state()
    hierarchical_draws(design, draws=10, seed=8)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("draws", [0, -1, 10001, 1.5, True, False, "10"])
def test_invalid_draws_are_rejected(draws):
    with pytest.raises((TypeError, ValueError)):
        hierarchical_draws(_singleton_design(), draws=draws, seed=1)


@pytest.mark.parametrize("seed", [-1, 1.5, True, False, "0"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises((TypeError, ValueError)):
        hierarchical_draws(_singleton_design(), draws=5, seed=seed)


def test_design_type_is_enforced():
    with pytest.raises(TypeError):
        hierarchical_draws(object(), draws=5, seed=1)


def test_non_finite_inputs_are_rejected():
    bad_counts = _make_design(
        masters=("ma",),
        classes=("c1",),
        stations=("s1",),
        instruments=("i0",),
        domains=("d0",),
        cell_keys=(("x0", "c1"),),
        counts=[[np.inf]],
        cell_domain=[0],
        domain_instrument=[0],
        cell_factor=[1.0],
    )
    with pytest.raises(ValueError):
        hierarchical_draws(bad_counts, draws=5, seed=1)


def test_integer_like_draws_are_accepted():
    design = _singleton_design()
    result = hierarchical_draws(design, draws=np.int64(6), seed=np.int64(2))
    assert result["draws"].shape == (6,)


@pytest.mark.parametrize(
    "change",
    [
        {"counts": np.array([[-1.0, 0.0], [0.0, 1.0]])},
        {"counts": np.array([[0.5, 0.0], [0.0, 1.0]])},
        {"delta_correct": np.array([[2.0, 0.0], [0.0, 1.0]])},
        {
            "counts": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "delta_correct": np.array([[0.0, 0.0], [0.0, 1.0]]),
        },
        {"domains": ("d0", "d1")},
        {"cell_domain": np.array([0.5, 0.5])},
        {"master_classes": ("c1",)},
        {"cell_factor": np.array([-0.5, 1.5])},
        {"cell_factor": np.array([0.4, 0.4])},
        {"counts": np.array([[np.inf, 0.0], [0.0, 1.0]])},
        {"counts": np.array([["a", "b"], ["c", "d"]])},
        {"counts": np.array([[1 + 0j, 0j], [0j, 1 + 0j]])},
    ],
)
def test_malformed_designs_are_rejected(change):
    design = replace(_singleton_design(), **change)
    with pytest.raises((TypeError, ValueError)):
        hierarchical_draws(design, draws=5, seed=1)
