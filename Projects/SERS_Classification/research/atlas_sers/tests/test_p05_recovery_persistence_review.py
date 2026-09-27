"""Independent regression checks for bounded, evidence-preserving copying."""

import hashlib
import time

import pytest

from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_persistence as persistence
from atlas_sers.evaluation.p05_comprehensive_storage import StorageBudget


@pytest.mark.parametrize(
    "record",
    [
        {"sha256": "z" * 64, "size_bytes": 1},
        {"sha256": "a" * 64, "size_bytes": 1, "extra": 1},
        {"sha256": "a" * 64, "size_bytes": True},
        {"sha256": "a" * 64, "size_bytes": -1},
    ],
)
def test_inventory_records_are_strict(record):
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._record_size(record)


def test_raw_dotdot_is_not_normalized_away():
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._abs("one/../two")


def test_duplicate_ledger_units_rejected():
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._ledger_index(
            {"ledger": {"units": [{"unit_id": "duplicate"}, {"unit_id": "duplicate"}], "slots": []}}
        )


@pytest.mark.parametrize("kind", ["ceiling", "artifact_root"])
def test_budget_cannot_widen_approved_scope(tmp_path, kind):
    artifact = tmp_path / "artifacts"
    run_root = artifact / "p05comprehensive/runs/synthetic"
    run_root.mkdir(parents=True)
    budget = StorageBudget(artifact, run_root)
    if kind == "ceiling":
        budget._ceiling = authority.PRIVATE_STORAGE_CEILING_BYTES + 1
    else:
        budget._artifact_root = tmp_path / "other"
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._check_budget_binding(budget, run_root)


def test_source_growth_stops_before_excess_bytes_written(tmp_path):
    artifact = tmp_path / "artifacts"
    run_root = artifact / "p05comprehensive/runs/synthetic"
    run_root.mkdir(parents=True)
    source = tmp_path / "source.bin"
    source.write_bytes(b"AB")
    budget = StorageBudget(artifact, run_root)
    unit = budget.activate_unit(run_root / "unit")
    destination = unit / "copied.bin"
    record = {"sha256": hashlib.sha256(b"A").hexdigest(), "size_bytes": 1}
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._copy_evidence(source, destination, record, budget, time.perf_counter() + 30)
    assert source.read_bytes() == b"AB"
    assert destination.exists()
    assert destination.stat().st_size <= record["size_bytes"]


@pytest.mark.parametrize("occupied", ["regular", "symlink", "parent_symlink", "source_symlink"])
def test_copy_rejects_existing_or_linked_paths_without_overwriting(tmp_path, occupied):
    artifact = tmp_path / "artifacts"
    run_root = artifact / "p05comprehensive/runs/synthetic"
    run_root.mkdir(parents=True)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source evidence")
    outsider = tmp_path / "untouched.bin"
    outsider.write_bytes(b"untouched")
    budget = StorageBudget(artifact, run_root)
    unit = budget.activate_unit(run_root / "unit")
    destination = unit / "copied.bin"
    if occupied == "regular":
        destination.write_bytes(b"existing evidence")
    elif occupied == "symlink":
        destination.symlink_to(outsider)
    elif occupied == "parent_symlink":
        linked = unit / "linked"
        linked.symlink_to(tmp_path, target_is_directory=True)
        destination = linked / outsider.name
    else:
        linked = tmp_path / "linked-source.bin"
        linked.symlink_to(source)
        source = linked
    record = {"sha256": hashlib.sha256(b"source evidence").hexdigest(), "size_bytes": 15}
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._run(
            lambda: persistence._copy_evidence(
                source, destination, record, budget, time.perf_counter() + 30
            )
        )
    assert source.read_bytes() == b"source evidence"
    assert outsider.read_bytes() == b"untouched"
    if occupied == "regular":
        assert destination.read_bytes() == b"existing evidence"
