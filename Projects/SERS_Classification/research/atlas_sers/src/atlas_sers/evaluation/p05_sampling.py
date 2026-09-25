"""All-master two-view sampler for the P05 core protocol.

Standard library only so the sampler can be audited without torch installed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass

SAMPLER_VERSION = "p05-all-master-two-view-v1"
MAXIMUM_INSTRUMENTS_PER_MASTER = 2
DEFAULT_BATCH_SIZE_CEILING = 48


@dataclass(frozen=True)
class Observation:
    """One stored measurement with its physical and instrument metadata."""

    uid: str
    master: str
    station: str
    target: str
    instrument: str
    substrate: str


@dataclass(frozen=True)
class MasterBatch:
    """Sampler output: positions into the input rows, row weights and digest."""

    indices: tuple[int, ...]
    weights: tuple[float, ...]
    draw_sha256: str


def _require_observation(row: object) -> Observation:
    if not isinstance(row, Observation):
        raise TypeError("rows must contain Observation instances")
    uid = row.uid
    if not isinstance(uid, str):
        raise TypeError("uid must be a string")
    if uid != uid.strip():
        raise ValueError("UID must not contain surrounding whitespace")
    if not uid:
        raise ValueError("UID must be a non-empty string")
    for name in ("master", "station", "target", "instrument"):
        value = getattr(row, name)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(row.substrate, str):
        raise TypeError("substrate must be a string")
    return row


def validate_rows(rows: Iterable[Observation]) -> None:
    """Validate role metadata without mutating the caller's rows.

    Rejects empty input, duplicate/blank/trim-altered UIDs, contradictory
    master/class assignments, blank identifying fields and rows spanning
    multiple stations.  Unknown substrate strings are allowed.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("rows must not be empty")
    seen_uids: set[str] = set()
    master_targets: dict[str, str] = {}
    station: str | None = None
    for row in materialized:
        observation = _require_observation(row)
        if observation.uid in seen_uids:
            raise ValueError("duplicate UID in rows")
        seen_uids.add(observation.uid)
        previous = master_targets.get(observation.master)
        if previous is None:
            master_targets[observation.master] = observation.target
        elif previous != observation.target:
            raise ValueError("a master is assigned to conflicting classes")
        if station is None:
            station = observation.station
        elif station != observation.station:
            raise ValueError("rows must belong to a single station")
    return None


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def sample_master_views(
    rows: Iterable[Observation],
    *,
    role_id: str,
    seed: int,
    epoch: int,
    batch_ordinal: int,
    max_batch_size: int = DEFAULT_BATCH_SIZE_CEILING,
) -> MasterBatch:
    """Draw one deterministic all-master two-view batch.

    Every master contributes, at most two distinct instruments are selected per
    master and one UID per selected instrument.  Row weights follow
    ``1 / (number_of_classes * masters_in_class * sampled_views_of_master)``.
    """

    if not isinstance(role_id, str) or not role_id.strip():
        raise ValueError("role_id must be a nonblank string")
    if role_id != role_id.strip():
        raise ValueError("role_id must not contain surrounding whitespace")
    seed = _require_int("seed", seed)
    epoch = _require_int("epoch", epoch)
    batch_ordinal = _require_int("batch_ordinal", batch_ordinal)
    max_batch_size = _require_int("max_batch_size", max_batch_size)
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if batch_ordinal < 0:
        raise ValueError("batch_ordinal must be non-negative")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")

    materialized = list(rows)
    validate_rows(materialized)

    grouped: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for index, row in enumerate(materialized):
        grouped.setdefault(row.master, {}).setdefault(row.instrument, []).append(
            (row.uid, index)
        )
    if not grouped:
        raise ValueError("cannot sample from a role with no rows")
    for instruments in grouped.values():
        for entries in instruments.values():
            entries.sort(key=lambda item: item[0])

    capacity = sum(
        min(MAXIMUM_INSTRUMENTS_PER_MASTER, len(instruments))
        for instruments in grouped.values()
    )
    if capacity > max_batch_size:
        raise ValueError(
            f"sampling capacity {capacity} exceeds max_batch_size {max_batch_size}"
        )

    seed_payload = json.dumps(
        [SAMPLER_VERSION, role_id, seed, epoch, batch_ordinal],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    seed_digest = hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_digest, 16))

    selected: list[tuple[str, int, str, int]] = []
    for master in sorted(grouped):
        instruments = grouped[master]
        names = sorted(instruments)
        views = min(MAXIMUM_INSTRUMENTS_PER_MASTER, len(names))
        chosen = rng.sample(names, views)
        chosen.sort()
        for instrument in chosen:
            uid, index = rng.choice(instruments[instrument])
            selected.append((uid, index, master, views))

    selected.sort(key=lambda item: item[0])
    uids = [item[0] for item in selected]

    targets = {row.master: row.target for row in materialized}
    class_count = len({targets[master] for master in grouped})
    masters_per_class: dict[str, int] = {}
    for master in grouped:
        target = targets[master]
        masters_per_class[target] = masters_per_class.get(target, 0) + 1

    indices = tuple(item[1] for item in selected)
    weights = tuple(
        1.0 / (class_count * masters_per_class[targets[item[2]]] * item[3])
        for item in selected
    )
    draw_payload = json.dumps(
        [SAMPLER_VERSION, role_id, seed, epoch, batch_ordinal, list(uids)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    draw_sha256 = hashlib.sha256(draw_payload.encode("utf-8")).hexdigest()
    return MasterBatch(indices=indices, weights=weights, draw_sha256=draw_sha256)


__all__ = [
    "DEFAULT_BATCH_SIZE_CEILING",
    "MAXIMUM_INSTRUMENTS_PER_MASTER",
    "MasterBatch",
    "Observation",
    "SAMPLER_VERSION",
    "sample_master_views",
    "validate_rows",
]
