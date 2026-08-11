from __future__ import annotations

from pathlib import Path

import pytest

from atlas_sers.governance.p03_store import P03ShardStore


def test_shard_store_commits_verifies_and_quarantines_corruption(tmp_path: Path) -> None:
    store = P03ShardStore(run_root=tmp_path / "p03-run")
    protected = "a" * 64
    lease = store.begin(shard_id=7, protected_state_sha256=protected)
    assert lease.action == "new"
    assert lease.temporary_dir is not None
    (lease.temporary_dir / "fit_status.csv").write_text("fit_id,status\nf1,complete\n")
    final = store.commit(lease)
    assert final.is_dir()
    repeated = store.begin(shard_id=7, protected_state_sha256=protected)
    assert repeated.action == "verified_skip"
    assert store.validation_table({7: protected})[0]["valid"] is True

    (final / "fit_status.csv").write_text("corrupt")
    replacement = store.begin(shard_id=7, protected_state_sha256=protected)
    assert replacement.action == "new"
    assert any(store.quarantine.iterdir())
    assert store.validation_table({7: protected})[0]["valid"] is False
    store.abort(replacement, reason="synthetic_interrupt")
    assert not replacement.lock_path.exists()


def test_shard_store_prevents_concurrent_lease(tmp_path: Path) -> None:
    store = P03ShardStore(run_root=tmp_path / "p03-run")
    first = store.begin(shard_id=2, protected_state_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="already leased"):
        store.begin(shard_id=2, protected_state_sha256="b" * 64)
    store.abort(first, reason="test_cleanup")
