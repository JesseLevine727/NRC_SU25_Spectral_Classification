from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p03_reconcile import (
    _canonicalize_terminal_fit_ledger,
    collect_outer_evidence,
)
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from tests.test_p03_collect import _write_shards


def test_terminal_ledger_canonicalizes_mixed_csv_seed_scalars(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        [
            {"fit_id": "fit-integer", "seed": "20260805"},
            {"fit_id": "fit-text", "seed": "deterministic"},
        ]
    )
    ledger = pd.DataFrame(
        [
            {"fit_id": "fit-integer", "status": "complete", "seed": 20260805},
            {
                "fit_id": "fit-text",
                "status": "complete",
                "seed": "deterministic",
            },
        ]
    )
    canonical = _canonicalize_terminal_fit_ledger(ledger, manifest)
    assert canonical.seed.tolist() == ["20260805", "deterministic"]
    canonical.to_parquet(tmp_path / "terminal_fit_ledger.parquet", index=False)


def test_complete_selection_and_outer_ledgers_reconcile_exactly(tmp_path: Path) -> None:
    protected = "d" * 64
    run_root = tmp_path / "p03-run"
    selection_manifest, selection_shards = _write_shards(
        run_root / "selection", protected
    )
    test_uids = ["test-1", "test-2"]
    selection_manifest["test_uid_sha256"] = sha256_value(test_uids)
    final_row = pd.DataFrame(
        [
            {
                "fit_id": "final-fit",
                "stage": "final_selected_refit",
                "accounting": "new_fit",
                "outer_run_id": "outer-1",
                "test_uid_sha256": sha256_value(test_uids),
            }
        ]
    )
    selection_manifest["outer_run_id"] = "outer-1"
    fit_manifest = pd.concat([selection_manifest, final_row], ignore_index=True)
    expected_runs = pd.DataFrame(
        [
            {
                "outer_run_id": "outer-1",
                "experiment_id": "EXP-C01-T1",
                "execution_status": "planned",
            }
        ]
    )
    outer = P03ShardStore(run_root=run_root / "outer")
    lease = outer.begin(shard_id=0, protected_state_sha256=protected)
    assert lease.temporary_dir is not None
    status = pd.DataFrame(
        [{"fit_id": "final-fit", "status": "complete"}]
    )
    predictions = pd.DataFrame(
        [
            {
                "outer_run_id": "outer-1",
                "procedure_id": "C-PCA-LDA",
                "observation_uid": uid,
            }
            for uid in test_uids
        ]
    )
    status.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
    predictions.to_parquet(
        lease.temporary_dir / "final_predictions.parquet", index=False
    )
    descriptor = {
        "outer_index": 0,
        "outer_run_id": "outer-1",
        "experiment_id": "EXP-C01-T1",
        "terminal_fit_count": 1,
        "terminal_fit_id_sha256": sha256_value(["final-fit"]),
        "final_prediction_rows": 2,
    }
    (lease.temporary_dir / "outer_descriptor.json").write_bytes(
        canonical_json_bytes(descriptor, pretty=True)
    )
    outer.commit(lease)
    evidence = collect_outer_evidence(
        p03_run_root=run_root,
        fit_manifest=fit_manifest,
        expected_run_registry=expected_runs,
        selection_shard_manifest=selection_shards,
        protected_state_sha256=protected,
        shard_target_fits=2,
    )
    assert set(evidence.fit_status.fit_id) == {
        "fit-0",
        "fit-1",
        "fit-2",
        "final-fit",
    }
    assert len(evidence.final_predictions) == 2
    assert evidence.outer_validation.valid.all()
