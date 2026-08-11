from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from atlas_sers.evaluation.p03_collect import (
    collect_selection_evidence,
    load_selection_predictions,
)
from atlas_sers.evaluation.p03_plan import assign_selection_shards
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore


def _manifest_and_shards() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.DataFrame(
        [
            {
                "fit_id": f"fit-{index}",
                "stage": "inner_selection",
                "accounting": "new_fit",
            }
            for index in range(3)
        ]
    )
    assigned = assign_selection_shards(manifest, target=2)
    shards = assigned.groupby("selection_shard_id", as_index=False).agg(
        selection_kind=("selection_kind", "first"),
        stage_count=("stage", "nunique"),
        fit_count=("fit_id", "size"),
        first_fit_id=("fit_id", "first"),
        last_fit_id=("fit_id", "last"),
        fit_id_sha256=("fit_id", lambda values: sha256_value(sorted(values))),
    )
    return manifest, shards


def _write_shards(root: Path, protected: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, shards = _manifest_and_shards()
    assigned = assign_selection_shards(manifest, target=2)
    store = P03ShardStore(run_root=root)
    for shard_id, rows in assigned.groupby("selection_shard_id", sort=True):
        lease = store.begin(shard_id=int(shard_id), protected_state_sha256=protected)
        assert lease.temporary_dir is not None
        statuses = pd.DataFrame(
            [
                {
                    "fit_id": fit_id,
                    "status": "complete",
                }
                for fit_id in rows.fit_id
            ]
        )
        metrics = statuses.assign(
            balanced_accuracy=1.0,
            macro_f1=1.0,
        )
        predictions = pd.DataFrame(
            [
                {
                    "fit_id": fit_id,
                    "observation_uid": f"observation-{fit_id}",
                }
                for fit_id in rows.fit_id
            ]
        )
        statuses.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
        metrics.to_csv(
            lease.temporary_dir / "selection_unit_metrics.csv", index=False
        )
        predictions.to_parquet(
            lease.temporary_dir / "validation_predictions.parquet", index=False
        )
        descriptor = {
            "fit_count": len(rows),
            "fit_id_sha256": sha256_value(sorted(rows.fit_id)),
        }
        (lease.temporary_dir / "shard_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        store.commit(lease)
    return manifest, shards


def test_selection_collection_rehashes_and_reconciles_every_planned_fit(
    tmp_path: Path,
) -> None:
    protected = "a" * 64
    root = tmp_path / "selection"
    manifest, shards = _write_shards(root, protected)
    evidence = collect_selection_evidence(
        selection_run_root=root,
        fit_manifest=manifest,
        selection_shard_manifest=shards,
        protected_state_sha256=protected,
        shard_target_fits=2,
    )
    assert set(evidence.fit_status.fit_id) == {"fit-0", "fit-1", "fit-2"}
    assert set(evidence.selection_unit_metrics.fit_id) == {
        "fit-0",
        "fit-1",
        "fit-2",
    }
    assert evidence.shard_validation.valid.all()
    predictions = load_selection_predictions(
        selection_run_root=root,
        fit_manifest=manifest,
        fit_ids={"fit-0", "fit-2"},
        shard_target_fits=2,
    )
    assert set(predictions.fit_id) == {"fit-0", "fit-2"}


def test_selection_collection_refuses_corrupt_completed_shard(tmp_path: Path) -> None:
    protected = "b" * 64
    root = tmp_path / "selection"
    manifest, shards = _write_shards(root, protected)
    corrupt = root / "shards" / "shard-000000" / "fit_status.csv"
    corrupt.write_text("fit_id,status\nfit-0,fit_failure\n")
    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        collect_selection_evidence(
            selection_run_root=root,
            fit_manifest=manifest,
            selection_shard_manifest=shards,
            protected_state_sha256=protected,
            shard_target_fits=2,
        )


def test_selection_prediction_request_must_be_fully_planned(tmp_path: Path) -> None:
    protected = "c" * 64
    root = tmp_path / "selection"
    manifest, _ = _write_shards(root, protected)
    with pytest.raises(ValueError, match="not all planned"):
        load_selection_predictions(
            selection_run_root=root,
            fit_manifest=manifest,
            fit_ids={"fit-0", "unknown-fit"},
            shard_target_fits=2,
        )
