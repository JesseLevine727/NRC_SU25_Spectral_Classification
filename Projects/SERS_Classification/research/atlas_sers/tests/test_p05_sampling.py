"""Synthetic tests for the P05 all-master two-view sampler."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

import pytest

from atlas_sers.evaluation.p05_sampling import (
    SAMPLER_VERSION,
    Observation,
    sample_master_views,
    validate_rows,
)


def build(specification, station="S1"):
    rows = []
    for master, target, instrument, count in specification:
        for repeat in range(count):
            rows.append(
                Observation(
                    uid=f"{master}-{instrument}-{repeat}",
                    master=master,
                    station=station,
                    target=target,
                    instrument=instrument,
                    substrate="",
                )
            )
    return rows


def uids_of(rows, batch):
    return tuple(rows[index].uid for index in batch.indices)


DENSE_SPEC = [
    ("M1", "A", "I1", 4),
    ("M2", "A", "I2", 3),
    ("M3", "A", "I3", 2),
    ("M4", "B", "I4", 5),
    ("M5", "B", "I5", 2),
    ("M6", "C", "I6", 3),
    ("M7", "C", "I7", 1),
    ("M8", "C", "I8a", 2),
    ("M8", "C", "I8b", 2),
    ("M8", "C", "I8c", 2),
]

WEIGHT_SPEC = [
    ("M1", "A", "I1", 5),
    ("M2", "A", "I2", 1),
    ("M2", "A", "I3", 1),
    ("M3", "B", "I4", 3),
]

SINGLE_INSTRUMENT_SPEC = [
    ("M1", "A", "I1", 3),
    ("M2", "A", "I2", 2),
    ("M3", "B", "I3", 4),
]


def test_deterministic_replay():
    rows = build(DENSE_SPEC)
    kwargs = dict(role_id="R1", seed=20260805, epoch=3, batch_ordinal=2)
    assert sample_master_views(rows, **kwargs) == sample_master_views(rows, **kwargs)


def test_different_seed_epoch_ordinal_and_role_change_draw():
    rows = build(DENSE_SPEC)
    base = sample_master_views(rows, role_id="R1", seed=1, epoch=1, batch_ordinal=0)
    variants = [
        dict(role_id="R1", seed=2, epoch=1, batch_ordinal=0),
        dict(role_id="R1", seed=1, epoch=2, batch_ordinal=0),
        dict(role_id="R1", seed=1, epoch=1, batch_ordinal=1),
        dict(role_id="R2", seed=1, epoch=1, batch_ordinal=0),
    ]
    for kwargs in variants:
        assert sample_master_views(rows, **kwargs).draw_sha256 != base.draw_sha256


def test_input_permutation_preserves_selection_and_weights():
    rows = build(DENSE_SPEC)
    shuffled = list(rows)
    random.Random(20260925).shuffle(shuffled)
    kwargs = dict(role_id="R1", seed=7, epoch=2, batch_ordinal=1)
    reference = sample_master_views(rows, **kwargs)
    permuted = sample_master_views(shuffled, **kwargs)
    assert uids_of(rows, reference) == uids_of(shuffled, permuted)
    assert reference.weights == permuted.weights
    assert reference.draw_sha256 == permuted.draw_sha256


def test_every_master_included_sorted_unique_and_instrument_cap():
    rows = build(DENSE_SPEC)
    batch = sample_master_views(rows, role_id="R1", seed=3, epoch=1, batch_ordinal=0)
    selected = [rows[index] for index in batch.indices]
    assert {row.master for row in selected} == {row.master for row in rows}
    uids = [row.uid for row in selected]
    assert len(uids) == len(set(uids))
    assert uids == sorted(uids)
    per_master = defaultdict(set)
    for row in selected:
        per_master[row.master].add(row.instrument)
    assert all(1 <= len(instruments) <= 2 for instruments in per_master.values())


def test_weights_sum_to_one_and_equal_master_contribution():
    rows = build(WEIGHT_SPEC)
    batch = sample_master_views(rows, role_id="R1", seed=11, epoch=1, batch_ordinal=2)
    assert abs(sum(batch.weights) - 1.0) < 1e-9
    assert all(weight > 0 for weight in batch.weights)
    per_master = defaultdict(float)
    for index, weight in zip(batch.indices, batch.weights, strict=True):
        per_master[rows[index].master] += weight
    assert per_master["M1"] == pytest.approx(0.25)
    assert per_master["M2"] == pytest.approx(0.25)
    assert per_master["M3"] == pytest.approx(0.5)


def test_all_single_instrument_role_selects_one_row_per_master():
    rows = build(SINGLE_INSTRUMENT_SPEC)
    batch = sample_master_views(rows, role_id="R", seed=5, epoch=1, batch_ordinal=0)
    assert len(batch.indices) == 3
    assert len(set(batch.indices)) == 3
    assert abs(sum(batch.weights) - 1.0) < 1e-9


def test_capacity_failure_is_rejected_before_sampling():
    specification = []
    for index in range(5):
        specification.append((f"M{index}", "A", f"I{index}a", 1))
        specification.append((f"M{index}", "A", f"I{index}b", 1))
    rows = build(specification)
    with pytest.raises(ValueError):
        sample_master_views(
            rows, role_id="R", seed=1, epoch=1, batch_ordinal=0, max_batch_size=9
        )
    batch = sample_master_views(
        rows, role_id="R", seed=1, epoch=1, batch_ordinal=0, max_batch_size=10
    )
    assert len(batch.indices) == 10


def test_unknown_substrate_strings_are_allowed():
    rows = [
        Observation("u1", "M1", "S", "A", "I1", "mystery"),
        Observation("u2", "M2", "S", "A", "I2", "unknown"),
    ]
    validate_rows(rows)
    batch = sample_master_views(rows, role_id="R", seed=1, epoch=1, batch_ordinal=0)
    assert len(batch.indices) == 2


def test_invalid_metadata_is_rejected():
    base = build(DENSE_SPEC)
    with pytest.raises(ValueError):
        validate_rows(base + [base[0]])
    with pytest.raises(ValueError):
        validate_rows([])
    with pytest.raises(ValueError) as duplicate_error:
        validate_rows(base + [base[0]])
    assert base[0].uid not in str(duplicate_error.value)
    with pytest.raises(ValueError):
        validate_rows([Observation("", "M", "S", "A", "I", "")])
    with pytest.raises(ValueError):
        validate_rows([Observation(" padded", "M", "S", "A", "I", "")])
    with pytest.raises(ValueError):
        validate_rows(
            [
                Observation("u1", "M", "S", "A", "I", ""),
                Observation("u2", "M", "S", "B", "J", ""),
            ]
        )
    with pytest.raises(ValueError):
        validate_rows([Observation("u1", "M", "S", "A", "", "")])
    with pytest.raises(ValueError):
        validate_rows(
            [
                Observation("u1", "M", "S1", "A", "I", ""),
                Observation("u2", "N", "S2", "A", "J", ""),
            ]
        )


def test_sampling_and_validation_do_not_mutate_inputs():
    rows = build(DENSE_SPEC)
    snapshot = list(rows)
    validate_rows(rows)
    sample_master_views(rows, role_id="R", seed=1, epoch=1, batch_ordinal=0)
    assert rows == snapshot


def test_invalid_argument_types_and_ranges():
    rows = build(DENSE_SPEC)
    with pytest.raises(TypeError):
        sample_master_views(rows, role_id="R", seed=1.5, epoch=1, batch_ordinal=0)
    with pytest.raises(TypeError):
        sample_master_views(rows, role_id="R", seed=True, epoch=1, batch_ordinal=0)
    with pytest.raises(TypeError):
        sample_master_views(rows, role_id="R", seed=1, epoch=True, batch_ordinal=0)
    with pytest.raises(ValueError):
        sample_master_views(rows, role_id="R", seed=1, epoch=0, batch_ordinal=0)
    with pytest.raises(ValueError):
        sample_master_views(rows, role_id="R", seed=1, epoch=1, batch_ordinal=-1)
    with pytest.raises(ValueError):
        sample_master_views(rows, role_id="", seed=1, epoch=1, batch_ordinal=0)
    with pytest.raises(ValueError):
        sample_master_views(rows, role_id=" R", seed=1, epoch=1, batch_ordinal=0)
    with pytest.raises(ValueError):
        sample_master_views(rows, role_id="   ", seed=1, epoch=1, batch_ordinal=0)
    with pytest.raises(TypeError):
        sample_master_views(
            rows, role_id="R", seed=1, epoch=1, batch_ordinal=0, max_batch_size=1.5
        )


def test_canonical_json_digest_and_delimiter_containing_inputs():
    rows = build(DENSE_SPEC)
    role_id = 'R"|1'
    batch = sample_master_views(
        rows, role_id=role_id, seed=20260805, epoch=1, batch_ordinal=1
    )
    selected_uids = [rows[index].uid for index in batch.indices]
    payload = json.dumps(
        [SAMPLER_VERSION, role_id, 20260805, 1, 1, selected_uids],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert batch.draw_sha256 == expected
    other = sample_master_views(
        rows, role_id="R", seed=20260805, epoch=1, batch_ordinal=1
    )
    assert other.draw_sha256 != batch.draw_sha256


def test_chosen_uid_varies_across_seeds_for_repeated_observations():
    rows = build([("M1", "A", "I1", 12)])
    chosen = {
        rows[index].uid
        for seed in range(1, 21)
        for index in sample_master_views(
            rows, role_id="R", seed=seed, epoch=1, batch_ordinal=0
        ).indices
    }
    assert len(chosen) > 1


def test_master_with_three_instruments_selects_two_distinct():
    rows = build(
        [("M1", "A", "I1", 2), ("M1", "A", "I2", 2), ("M1", "A", "I3", 2)]
    )
    seen_pairs = set()
    for seed in range(1, 21):
        batch = sample_master_views(
            rows, role_id="R", seed=seed, epoch=1, batch_ordinal=0
        )
        instruments = [rows[index].instrument for index in batch.indices]
        assert len(instruments) == 2
        assert len(set(instruments)) == 2
        seen_pairs.add(tuple(sorted(instruments)))
    assert len(seen_pairs) > 1


def test_weights_independent_of_repeat_counts():
    dense = [("M1", "A", "I1", 5), ("M2", "A", "I2", 1), ("M3", "B", "I3", 9)]
    sparse = [("M1", "A", "I1", 1), ("M2", "A", "I2", 7), ("M3", "B", "I3", 2)]
    for specification in (dense, sparse):
        rows = build(specification)
        batch = sample_master_views(
            rows, role_id="R", seed=4, epoch=1, batch_ordinal=0
        )
        assert abs(sum(batch.weights) - 1.0) < 1e-9
        per_master = defaultdict(float)
        for index, weight in zip(batch.indices, batch.weights, strict=True):
            per_master[rows[index].master] += weight
        assert per_master["M1"] == pytest.approx(0.25)
        assert per_master["M2"] == pytest.approx(0.25)
        assert per_master["M3"] == pytest.approx(0.5)


def test_single_class_role_is_supported_by_sampler():
    rows = build([("M1", "A", "I1", 2), ("M2", "A", "I2", 1)])
    batch = sample_master_views(rows, role_id="R", seed=1, epoch=1, batch_ordinal=0)
    assert len(batch.indices) == 2
    assert abs(sum(batch.weights) - 1.0) < 1e-9


def test_empty_rows_rejected():
    with pytest.raises(ValueError):
        validate_rows([])
    with pytest.raises(ValueError):
        sample_master_views([], role_id="R", seed=1, epoch=1, batch_ordinal=0)
