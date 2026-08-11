from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

from atlas_sers.evaluation.p03_runtime import P03Dataset
from atlas_sers.evaluation.p03_worker import (
    SelectionShardResult,
    execute_metadata_control_selection_rows,
    execute_selection_rows,
    execute_selection_shard,
    execute_source_covariance_selection_rows,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.governance.p03_store import P03ShardStore


def _inputs() -> tuple[
    P03Dataset,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, object],
]:
    rng = np.random.default_rng(20260809)
    records: list[dict[str, object]] = []
    rows: list[np.ndarray] = []
    assignments: list[dict[str, object]] = []
    index = 0
    for fold in range(4):
        for label, center in (("a", -1.0), ("b", 1.0)):
            master = f"{label}-m{fold}"
            uid = f"row-{index}"
            records.append(
                {
                    "observation_uid": uid,
                    "master_sample_id": master,
                    "target_analyte": label,
                    "instrument": f"unit-{fold % 2}",
                    "station": "cwa",
                }
            )
            rows.append(rng.normal(center, 0.05, 8))
            assignments.append(
                {
                    "outer_repeat": 1,
                    "station": "cwa",
                    "master_sample_id": master,
                    "outer_fold": fold,
                }
            )
            index += 1
    manifest = pd.DataFrame(records)
    dataset = P03Dataset.from_frozen_representation(
        intensity=np.vstack(rows),
        representation_uids=manifest.observation_uid.to_numpy(),
        metadata=manifest,
    )
    test = manifest[manifest.master_sample_id.isin({"a-m0", "b-m0"})]
    validation = manifest[manifest.master_sample_id.isin({"a-m1", "b-m1"})]
    fit = manifest[manifest.master_sample_id.isin({"a-m2", "b-m2", "a-m3", "b-m3"})]
    parameters = {"metric": "cosine"}
    parameter_hash = sha256_value(parameters)
    candidate_registry = pd.DataFrame(
        [
            {
                "candidate_id": "C-SPECTRAL-MATCH-000",
                "model_id": "C-SPECTRAL-MATCH",
                "parameters_json": json.dumps(parameters, separators=(",", ":")),
                "hyperparameter_sha256": parameter_hash,
            }
        ]
    )
    fit_rows = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C01-T1",
                "task_id": "T1-CWA",
                "outer_run_id": "outer-1",
                "domain": "cwa:within",
                "station": "cwa",
                "held_instrument": "not_applicable",
                "outer_repeat": 1,
                "outer_fold": 0,
                "selection_mode": "inner_master_cv",
                "stage": "inner_selection",
                "selection_unit_id": "outer_fold_as_inner:1",
                "model_id": "C-SPECTRAL-MATCH",
                "candidate_id": "C-SPECTRAL-MATCH-000",
                "hyperparameter_sha256": parameter_hash,
                "seed": "deterministic",
                "fit_rows": len(fit),
                "fit_masters": fit.master_sample_id.nunique(),
                "fit_uid_sha256": sha256_value(sorted(fit.observation_uid)),
                "validation_rows": len(validation),
                "validation_masters": validation.master_sample_id.nunique(),
                "validation_uid_sha256": sha256_value(sorted(validation.observation_uid)),
                "test_rows": len(test),
                "test_masters": test.master_sample_id.nunique(),
                "test_uid_sha256": sha256_value(sorted(test.observation_uid)),
                "accounting": "new_fit",
                "fit_id": "fit-1",
            }
        ]
    )
    tables = {"master_split_registry.csv": pd.DataFrame(assignments)}
    contract: dict[str, object] = {"class_vocabulary": {"cwa": ["a", "b"]}}
    return dataset, manifest, fit_rows, candidate_registry, tables, contract


def test_selection_worker_refuses_before_any_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    calls = 0

    def forbidden_fit(**_: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("atlas_sers.evaluation.p03_worker.run_candidate_fit", forbidden_fit)
    with pytest.raises(PermissionError, match="disabled"):
        execute_selection_rows(
            dataset=dataset,
            fit_rows=rows,
            candidate_registry=candidates,
            manifest=manifest,
            p02_tables=tables,
            p03_contract=contract,
            scientific_fitting_authorized=False,
        )
    assert calls == 0


def test_selection_worker_produces_complete_terminal_evidence() -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    result = execute_selection_rows(
        dataset=dataset,
        fit_rows=rows,
        candidate_registry=candidates,
        manifest=manifest,
        p02_tables=tables,
        p03_contract=contract,
        scientific_fitting_authorized=True,
    )
    assert result.fit_status.status.tolist() == ["complete"]
    assert len(result.validation_predictions) == 2
    assert result.validation_predictions.candidate_id.eq(
        "C-SPECTRAL-MATCH-000"
    ).all()
    assert result.validation_predictions.model_id.eq("C-SPECTRAL-MATCH").all()
    assert result.validation_predictions.seed.eq("deterministic").all()
    assert result.selection_unit_metrics.balanced_accuracy.tolist() == [1.0]
    assert result.selection_unit_metrics.macro_f1.tolist() == [1.0]


def test_source_covariance_worker_refuses_unresolved_method_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    rows.loc[:, "stage"] = "inner_source_coral_selection"
    contract["coral"] = {"status": "requires_versioned_method_resolution_before_fitting"}
    calls = 0

    def forbidden_worker(**_: object) -> SelectionShardResult:
        nonlocal calls
        calls += 1
        raise AssertionError("generic worker must not run")

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_worker.execute_selection_rows", forbidden_worker
    )
    with pytest.raises(PermissionError, match="remains unresolved"):
        execute_source_covariance_selection_rows(
            dataset=dataset,
            fit_rows=rows,
            coral_candidate_registry=candidates,
            manifest=manifest,
            p02_tables=tables,
            p03_contract=contract,
            scientific_fitting_authorized=True,
        )
    assert calls == 0


def test_source_covariance_worker_routes_only_authorized_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    rows.loc[:, "stage"] = "inner_source_coral_selection"
    contract["coral"] = {
        "status": "resolved_source_to_source_covariance_augmentation_v1"
    }
    observed: dict[str, object] = {}
    expected = SelectionShardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def capture_worker(**kwargs: object) -> SelectionShardResult:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_worker.execute_selection_rows", capture_worker
    )
    actual = execute_source_covariance_selection_rows(
        dataset=dataset,
        fit_rows=rows,
        coral_candidate_registry=candidates,
        manifest=manifest,
        p02_tables=tables,
        p03_contract=contract,
        scientific_fitting_authorized=True,
    )
    assert actual is expected
    assert observed["allowed_stages"] == {"inner_source_coral_selection"}
    assert observed["candidate_registry"] is candidates


def test_metadata_control_worker_refuses_unresolved_scope_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    rows.loc[:, "stage"] = "metadata_inner_selection"
    contract["negative_controls"] = {
        "status": "requires_versioned_scope_resolution_before_fitting"
    }
    calls = 0

    def forbidden_worker(**_: object) -> SelectionShardResult:
        nonlocal calls
        calls += 1
        raise AssertionError("generic worker must not run")

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_worker.execute_selection_rows", forbidden_worker
    )
    with pytest.raises(PermissionError, match="scope remains unresolved"):
        execute_metadata_control_selection_rows(
            dataset=dataset,
            fit_rows=rows,
            control_registry=pd.DataFrame(),
            manifest=manifest,
            p02_tables=tables,
            p03_contract=contract,
            scientific_fitting_authorized=True,
        )
    assert calls == 0


def test_metadata_control_worker_normalizes_only_its_frozen_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest, rows, _, tables, contract = _inputs()
    rows.loc[:, "stage"] = "metadata_inner_selection"
    contract["negative_controls"] = {"status": "resolved_p03_negative_controls_v1"}
    control_registry = pd.DataFrame(
        [
            {
                "control_candidate_id": f"CTRL-META-{index:02d}",
                "control_type": "acquisition_metadata_only",
                "model_id": "C-METADATA-LOGREG",
                "parameters_json": "{}",
                "configuration_sha256": "a" * 64,
                "declared_candidate_order": index,
                "complexity_rank": index,
                "seed_count": 1,
            }
            for index in range(30)
        ]
        + [
            {
                "control_candidate_id": "CTRL-PRIOR-1",
                "control_type": "station_or_target_prior",
                "model_id": "C-PRIOR",
                "parameters_json": "{}",
                "configuration_sha256": "b" * 64,
                "declared_candidate_order": 0,
                "complexity_rank": 0,
                "seed_count": 1,
            },
        ]
    )
    observed: dict[str, object] = {}
    expected = SelectionShardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def capture_worker(**kwargs: object) -> SelectionShardResult:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_worker.execute_selection_rows", capture_worker
    )
    actual = execute_metadata_control_selection_rows(
        dataset=dataset,
        fit_rows=rows,
        control_registry=control_registry,
        manifest=manifest,
        p02_tables=tables,
        p03_contract=contract,
        scientific_fitting_authorized=True,
    )
    assert actual is expected
    normalized = observed["candidate_registry"]
    assert isinstance(normalized, pd.DataFrame)
    assert normalized.candidate_id.tolist() == [f"CTRL-META-{index:02d}" for index in range(30)]
    assert normalized.hyperparameter_sha256.eq("a" * 64).all()
    assert observed["allowed_stages"] == {"metadata_inner_selection"}


def test_selection_shard_commits_atomically_and_verified_skips(tmp_path: Path) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    store = P03ShardStore(run_root=tmp_path / "run")
    arguments = {
        "store": store,
        "shard_id": 3,
        "protected_state_sha256": "a" * 64,
        "dataset": dataset,
        "fit_rows": rows,
        "candidate_registry": candidates,
        "manifest": manifest,
        "p02_tables": tables,
        "p03_contract": contract,
        "scientific_fitting_authorized": True,
    }
    final, action = execute_selection_shard(**arguments)
    assert action == "new"
    assert (final / "fit_status.csv").is_file()
    assert (final / "selection_unit_metrics.csv").is_file()
    assert (final / "validation_predictions.parquet").is_file()
    descriptor = json.loads((final / "shard_descriptor.json").read_text())
    assert descriptor["native_math_threads_per_worker"] == 1
    repeated, action = execute_selection_shard(**arguments)
    assert repeated == final
    assert action == "verified_skip"


def test_selection_shard_rejects_unfrozen_native_thread_count(tmp_path: Path) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    with pytest.raises(ValueError, match="exactly one native math thread"):
        execute_selection_shard(
            store=P03ShardStore(run_root=tmp_path / "run"),
            shard_id=0,
            protected_state_sha256="a" * 64,
            dataset=dataset,
            fit_rows=rows,
            candidate_registry=candidates,
            manifest=manifest,
            p02_tables=tables,
            p03_contract=contract,
            scientific_fitting_authorized=True,
            native_thread_limit=2,
        )


def test_selection_shard_actually_limits_native_thread_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, manifest, rows, candidates, tables, contract = _inputs()
    observed: list[int] = []

    def capture_limits(**kwargs: object) -> SelectionShardResult:
        observed.extend(int(pool["num_threads"]) for pool in threadpool_info())
        return execute_selection_rows(**kwargs)

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_worker.execute_selection_rows", capture_limits
    )
    execute_selection_shard(
        store=P03ShardStore(run_root=tmp_path / "run"),
        shard_id=0,
        protected_state_sha256="a" * 64,
        dataset=dataset,
        fit_rows=rows,
        candidate_registry=candidates,
        manifest=manifest,
        p02_tables=tables,
        p03_contract=contract,
        scientific_fitting_authorized=True,
    )

    assert observed
    assert max(observed) == 1
