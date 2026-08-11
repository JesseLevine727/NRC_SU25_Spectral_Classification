"""Deterministic P03 batch-partition tests."""

from __future__ import annotations

import pytest

from atlas_sers.governance.p03_execution import _partition_batch_ids


def test_worker_partitions_are_disjoint_and_complete() -> None:
    identifiers = list(range(225))
    partitions = [
        _partition_batch_ids(
            identifiers,
            worker_index=worker,
            worker_count=4,
            start_index=0,
            stop_index=None,
            max_tasks=None,
        )
        for worker in range(4)
    ]

    assert sorted(identifier for part in partitions for identifier in part) == identifiers
    assert sum(len(set(part)) for part in partitions) == len(identifiers)
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            assert set(partitions[left]).isdisjoint(partitions[right])


def test_range_and_pilot_limit_are_applied_before_execution() -> None:
    assert _partition_batch_ids(
        list(range(20)),
        worker_index=1,
        worker_count=3,
        start_index=4,
        stop_index=17,
        max_tasks=2,
    ) == [4, 7]


@pytest.mark.parametrize(
    ("worker_index", "worker_count", "start_index", "stop_index", "max_tasks"),
    [
        (0, 0, 0, None, None),
        (-1, 2, 0, None, None),
        (2, 2, 0, None, None),
        (0, 1, -1, None, None),
        (0, 1, 5, 4, None),
        (0, 1, 0, None, 0),
    ],
)
def test_invalid_worker_contract_is_rejected(
    worker_index: int,
    worker_count: int,
    start_index: int,
    stop_index: int | None,
    max_tasks: int | None,
) -> None:
    with pytest.raises(ValueError):
        _partition_batch_ids(
            list(range(10)),
            worker_index=worker_index,
            worker_count=worker_count,
            start_index=start_index,
            stop_index=stop_index,
            max_tasks=max_tasks,
        )
