from __future__ import annotations

import pandas as pd
import pytest

from atlas_sers.evaluation.p03_roles import resolve_fit_roles
from atlas_sers.governance.canonical import sha256_value


def _hash(frame: pd.DataFrame) -> str:
    return sha256_value(sorted(frame.observation_uid.astype(str)))


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station in ("cwa", "pills", "surfaces"):
        for master, label, instrument in (
            (1, "4_ANPP", "unit-1"),
            (2, "4_ANPP", "unit-2"),
            (3, "benzyl_fentanyl", "unit-1"),
            (4, "benzyl_fentanyl", "unit-2"),
        ):
            rows.append(
                {
                    "observation_uid": f"{station}-{master}",
                    "master_sample_id": master,
                    "target_analyte": label,
                    "instrument": instrument,
                    "station": station,
                }
            )
    return pd.DataFrame(rows)


def _master_splits() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station in ("cwa", "pills", "surfaces"):
        for master in range(1, 5):
            rows.append(
                {
                    "outer_repeat": 1,
                    "station": station,
                    "master_sample_id": master,
                    "outer_fold": master - 1,
                }
            )
    return pd.DataFrame(rows)


def _row(
    *,
    experiment_id: str,
    stage: str,
    selection_unit_id: str,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    **extra: object,
) -> pd.Series:
    return pd.Series(
        {
            "experiment_id": experiment_id,
            "task_id": "T1-CWA",
            "station": "cwa",
            "domain": "cwa:within",
            "held_instrument": "not_applicable",
            "outer_repeat": 1,
            "outer_fold": 0,
            "stage": stage,
            "selection_unit_id": selection_unit_id,
            "accounting": "new_fit",
            "fit_uid_sha256": _hash(fit),
            "validation_uid_sha256": _hash(validation),
            "test_uid_sha256": _hash(test),
            "fit_rows": len(fit),
            "fit_masters": fit.master_sample_id.astype(str).nunique(),
            "validation_rows": len(validation),
            "validation_masters": validation.master_sample_id.astype(str).nunique(),
            "test_rows": len(test),
            "test_masters": test.master_sample_id.astype(str).nunique(),
            **extra,
        }
    )


def test_resolves_t1_inner_roles_from_frozen_master_assignments() -> None:
    manifest = _manifest()
    cwa = manifest[manifest.station == "cwa"]
    test = cwa[cwa.master_sample_id == 1]
    validation = cwa[cwa.master_sample_id == 2]
    fit = cwa[cwa.master_sample_id.isin({3, 4})]
    row = _row(
        experiment_id="EXP-C03-T1",
        stage="inner_selection",
        selection_unit_id="outer_fold_as_inner:1",
        fit=fit,
        validation=validation,
        test=test,
    )
    roles = resolve_fit_roles(
        row,
        manifest=manifest,
        p02_tables={"master_split_registry.csv": _master_splits()},
    )
    assert roles.fit_uids == fit.observation_uid.tolist()
    assert roles.validation_uids == validation.observation_uid.tolist()
    assert roles.test_uids == test.observation_uid.tolist()


def _t3_tables(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = manifest[manifest.station == "cwa"].copy()
    source["partition_id"] = "partition-1"
    source["domain"] = "cwa:held"
    source["held_instrument"] = "held"
    source["outer_repeat"] = 1
    source["outer_fold"] = 0
    source["role"] = "train_source"
    test = source.iloc[[0]].copy()
    test["observation_uid"] = "held-test"
    test["master_sample_id"] = 9
    test["instrument"] = "held"
    test["role"] = "test_target"
    partitions = pd.concat([source, test], ignore_index=True)
    inner = pd.DataFrame(
        {
            "partition_id": ["partition-1", "partition-1"],
            "inner_fold": [0, 1],
            "master_sample_id": [1, 2],
        }
    )
    return partitions, inner


def test_resolves_t3_pseudo_domain_and_removes_same_master_views() -> None:
    manifest = _manifest()
    partitions, inner = _t3_tables(manifest)
    source = partitions[partitions.role == "train_source"]
    test = partitions[partitions.role == "test_target"]
    validation = source[source.instrument == "unit-1"]
    validation_masters = set(validation.master_sample_id)
    fit = source[
        (source.instrument != "unit-1")
        & (~source.master_sample_id.isin(validation_masters))
    ]
    row = _row(
        experiment_id="EXP-C09-T3",
        stage="inner_selection",
        selection_unit_id="pseudo:unit-1",
        fit=fit,
        validation=validation,
        test=test,
        domain="cwa:held",
        held_instrument="held",
    )
    roles = resolve_fit_roles(
        row,
        manifest=manifest,
        p02_tables={
            "t3_partition_registry.csv": partitions,
            "inner_master_split_registry.csv": inner,
        },
    )
    assert set(roles.fit.master_sample_id).isdisjoint(roles.validation.master_sample_id)
    assert roles.fit_uids == fit.observation_uid.tolist()


def test_resolves_t3_master_cv_and_fails_closed_on_hash_drift() -> None:
    manifest = _manifest()
    partitions, inner = _t3_tables(manifest)
    source = partitions[partitions.role == "train_source"]
    test = partitions[partitions.role == "test_target"]
    validation = source[source.master_sample_id == 1]
    fit = source[source.master_sample_id != 1]
    row = _row(
        experiment_id="EXP-C09-T3",
        stage="inner_selection",
        selection_unit_id="master_cv:0",
        fit=fit,
        validation=validation,
        test=test,
        domain="cwa:held",
        held_instrument="held",
    )
    tables = {
        "t3_partition_registry.csv": partitions,
        "inner_master_split_registry.csv": inner,
    }
    resolve_fit_roles(row, manifest=manifest, p02_tables=tables)
    row["fit_uid_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable P03 plan"):
        resolve_fit_roles(row, manifest=manifest, p02_tables=tables)


def test_resolves_t2_training_station_without_target_station_access() -> None:
    manifest = _manifest()
    manifest.loc[manifest.station == "surfaces", "master_sample_id"] += 10
    source = manifest[manifest.station == "pills"]
    test = manifest[manifest.station == "surfaces"]
    validation = source[source.master_sample_id == 1]
    fit = source[source.master_sample_id != 1]
    row = _row(
        experiment_id="EXP-C11-T2",
        stage="training_station_inner_selection",
        selection_unit_id="repeat:1:fold:0",
        fit=fit,
        validation=validation,
        test=test,
        task_id="T2-PS",
        station="pills",
        domain="pills_to_surfaces",
    )
    roles = resolve_fit_roles(
        row,
        manifest=manifest,
        p02_tables={"master_split_registry.csv": _master_splits()},
    )
    assert roles.fit.station.eq("pills").all()
    assert roles.validation.station.eq("pills").all()
    assert roles.test.station.eq("surfaces").all()
