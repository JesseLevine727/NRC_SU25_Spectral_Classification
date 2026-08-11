"""Reconstruct P03 fit/validation/test roles from immutable P02 records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import sha256_value


def _uid_hash(frame: pd.DataFrame) -> str:
    return sha256_value(sorted(frame.observation_uid.astype(str)))


@dataclass(frozen=True)
class ResolvedRoles:
    fit: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def fit_uids(self) -> list[str]:
        return self.fit.observation_uid.astype(str).tolist()

    @property
    def validation_uids(self) -> list[str]:
        return self.validation.observation_uid.astype(str).tolist()

    @property
    def test_uids(self) -> list[str]:
        return self.test.observation_uid.astype(str).tolist()


def _t1_roles(
    row: pd.Series, manifest: pd.DataFrame, master_splits: pd.DataFrame
) -> ResolvedRoles:
    repeat = int(row.outer_repeat)
    outer_fold = int(row.outer_fold)
    station = str(row.station)
    assignments = master_splits[
        (master_splits.outer_repeat == repeat) & (master_splits.station == station)
    ]
    if assignments.master_sample_id.nunique() == 0:
        raise ValueError("T1 outer assignment cell is absent from P02.")
    test_masters = set(
        assignments.loc[assignments.outer_fold == outer_fold, "master_sample_id"].astype(str)
    )
    station_rows = manifest[manifest.station == station]
    test = station_rows[station_rows.master_sample_id.astype(str).isin(test_masters)]
    outer_train = station_rows[~station_rows.master_sample_id.astype(str).isin(test_masters)]
    if row.stage == "inner_selection":
        prefix = "outer_fold_as_inner:"
        if not str(row.selection_unit_id).startswith(prefix):
            raise ValueError("T1 inner-selection unit is malformed.")
        inner_fold = int(str(row.selection_unit_id).removeprefix(prefix))
        validation_masters = set(
            assignments.loc[assignments.outer_fold == inner_fold, "master_sample_id"].astype(str)
        )
        validation = outer_train[
            outer_train.master_sample_id.astype(str).isin(validation_masters)
        ]
        fit = outer_train[
            ~outer_train.master_sample_id.astype(str).isin(validation_masters)
        ]
        return ResolvedRoles(fit, validation, test)
    return ResolvedRoles(outer_train, outer_train.iloc[0:0], test)


def _t3_cell(row: pd.Series, partitions: pd.DataFrame) -> pd.DataFrame:
    cell = partitions[
        (partitions.domain == row.domain)
        & (partitions.outer_repeat == int(row.outer_repeat))
        & (partitions.outer_fold == int(row.outer_fold))
    ]
    if cell.partition_id.nunique() != 1:
        raise ValueError("T3 row does not resolve to exactly one P02 partition.")
    return cell


def _master_cv_roles(
    source: pd.DataFrame,
    *,
    partition_id: str,
    selection_unit_id: str,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_fold = int(selection_unit_id.rsplit(":", 1)[1])
    inner = assignments[
        (assignments.partition_id == partition_id)
        & (assignments.inner_fold == inner_fold)
    ]
    if inner.empty:
        raise ValueError("T3 master-CV selection unit is absent from P02.")
    validation_masters = set(inner.master_sample_id.astype(str))
    validation = source[source.master_sample_id.astype(str).isin(validation_masters)]
    fit = source[~source.master_sample_id.astype(str).isin(validation_masters)]
    return fit, validation


def _t3_roles(
    row: pd.Series,
    partitions: pd.DataFrame,
    inner_master: pd.DataFrame,
) -> ResolvedRoles:
    cell = _t3_cell(row, partitions)
    partition_id = str(cell.partition_id.iloc[0])
    source = cell[cell.role == "train_source"]
    test = cell[cell.role == "test_target"]
    unit = str(row.selection_unit_id)
    if row.stage in {
        "inner_selection",
        "inner_source_coral_selection",
        "metadata_inner_selection",
    }:
        if unit.startswith("pseudo:"):
            pseudo = unit.removeprefix("pseudo:")
            validation = source[source.instrument == pseudo]
            validation_masters = set(validation.master_sample_id.astype(str))
            fit = source[
                (source.instrument != pseudo)
                & (~source.master_sample_id.astype(str).isin(validation_masters))
            ]
        elif unit.startswith("master_cv:"):
            fit, validation = _master_cv_roles(
                source,
                partition_id=partition_id,
                selection_unit_id=unit,
                assignments=inner_master,
            )
        else:
            raise ValueError("T3 selection unit is neither pseudo-domain nor master CV.")
        return ResolvedRoles(fit, validation, test)
    if row.stage in {"calibration_crossfit", "metadata_calibration_crossfit"}:
        fit, validation = _master_cv_roles(
            source,
            partition_id=partition_id,
            selection_unit_id=unit,
            assignments=inner_master,
        )
        return ResolvedRoles(fit, validation, test)
    if row.accounting == "cache_reuse":
        return ResolvedRoles(source.iloc[0:0], source.iloc[0:0], test)
    return ResolvedRoles(source, source.iloc[0:0], test)


def _t2_roles(
    row: pd.Series, manifest: pd.DataFrame, master_splits: pd.DataFrame
) -> ResolvedRoles:
    shared = {"4_ANPP", "benzyl_fentanyl"}
    if row.task_id == "T2-PS":
        source_station, target_station = "pills", "surfaces"
    elif row.task_id == "T2-SP":
        source_station, target_station = "surfaces", "pills"
    else:
        raise ValueError("Unknown T2 direction.")
    source = manifest[
        (manifest.station == source_station) & manifest.target_analyte.isin(shared)
    ]
    test = manifest[(manifest.station == target_station) & manifest.target_analyte.isin(shared)]
    if row.stage == "training_station_inner_selection":
        parts = str(row.selection_unit_id).split(":")
        if len(parts) != 4 or parts[0] != "repeat" or parts[2] != "fold":
            raise ValueError("T2 inner-selection unit is malformed.")
        repeat, fold = int(parts[1]), int(parts[3])
        assignments = master_splits[
            (master_splits.station == source_station)
            & (master_splits.outer_repeat == repeat)
            & (master_splits.outer_fold == fold)
        ]
        validation_masters = set(assignments.master_sample_id.astype(str))
        validation = source[source.master_sample_id.astype(str).isin(validation_masters)]
        fit = source[~source.master_sample_id.astype(str).isin(validation_masters)]
        return ResolvedRoles(fit, validation, test)
    if row.accounting == "cache_reuse":
        return ResolvedRoles(source.iloc[0:0], source.iloc[0:0], test)
    return ResolvedRoles(source, source.iloc[0:0], test)


def resolve_fit_roles(
    fit_row: pd.Series | dict[str, Any],
    *,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
) -> ResolvedRoles:
    """Resolve a fit-manifest row and prove its role hashes unchanged."""

    row = fit_row if isinstance(fit_row, pd.Series) else pd.Series(fit_row)
    experiment_id = str(row.experiment_id)
    if experiment_id.endswith("-T1"):
        roles = _t1_roles(row, manifest, p02_tables["master_split_registry.csv"])
    elif experiment_id in {
        "EXP-C09-T3",
        "EXP-C10-T3",
        "EXP-C12-CORAL",
        "EXP-C09-CONTROL-PERM",
        "EXP-C09-CONTROL-META",
        "EXP-C09-CONTROL-PRIOR",
    }:
        roles = _t3_roles(
            row,
            p02_tables["t3_partition_registry.csv"],
            p02_tables["inner_master_split_registry.csv"],
        )
    elif experiment_id == "EXP-C11-T2":
        roles = _t2_roles(row, manifest, p02_tables["master_split_registry.csv"])
    else:
        raise ValueError(f"Fit row references an unrecognized experiment: {experiment_id}")
    observed = {
        "fit_uid_sha256": _uid_hash(roles.fit),
        "validation_uid_sha256": _uid_hash(roles.validation),
        "test_uid_sha256": _uid_hash(roles.test),
    }
    for field, digest in observed.items():
        if str(row[field]) != digest:
            raise ValueError(f"Resolved {field} differs from the immutable P03 plan.")
    counts = {
        "fit_rows": len(roles.fit),
        "fit_masters": roles.fit.master_sample_id.astype(str).nunique(),
        "validation_rows": len(roles.validation),
        "validation_masters": roles.validation.master_sample_id.astype(str).nunique(),
        "test_rows": len(roles.test),
        "test_masters": roles.test.master_sample_id.astype(str).nunique(),
    }
    for field, count in counts.items():
        if int(row[field]) != count:
            raise ValueError(f"Resolved {field} differs from the immutable P03 plan.")
    uid_sets = {
        "fit": set(roles.fit.observation_uid.astype(str)),
        "validation": set(roles.validation.observation_uid.astype(str)),
        "test": set(roles.test.observation_uid.astype(str)),
    }
    master_sets = {
        "fit": set(roles.fit.master_sample_id.astype(str)),
        "validation": set(roles.validation.master_sample_id.astype(str)),
        "test": set(roles.test.master_sample_id.astype(str)),
    }
    for left, right in (("fit", "validation"), ("fit", "test"), ("validation", "test")):
        if uid_sets[left] & uid_sets[right]:
            raise ValueError(f"Resolved {left} and {right} observation roles overlap.")
        if master_sets[left] & master_sets[right]:
            raise ValueError(f"Resolved {left} and {right} master roles overlap.")
    if experiment_id in {
        "EXP-C09-T3",
        "EXP-C10-T3",
        "EXP-C12-CORAL",
        "EXP-C09-CONTROL-PERM",
        "EXP-C09-CONTROL-META",
        "EXP-C09-CONTROL-PRIOR",
    }:
        if str(row.held_instrument) in set(roles.fit.instrument.astype(str)):
            raise ValueError("Held T3 instrument entered the resolved fitting role.")
        if str(row.held_instrument) in set(roles.validation.instrument.astype(str)):
            raise ValueError("Held T3 instrument entered the resolved validation role.")
    return roles
