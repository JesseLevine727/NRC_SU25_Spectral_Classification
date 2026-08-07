"""Pure deterministic builders for the ATLAS P02 evaluation freeze."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from atlas_sers.governance.canonical import sha256_value

REQUIRED_COLUMNS = {
    "observation_uid",
    "master_sample_id",
    "station",
    "instrument",
    "sensor_family",
    "target_analyte",
    "first_difference_noise_mad",
    "intensity_range",
    "spike_fraction_proxy",
    "baseline_energy_fraction_proxy",
    "baseline_span_fraction_proxy",
    "negative_fraction",
}

NOISE_FEATURES = (
    "first_difference_noise_mad_over_intensity_range",
    "spike_fraction_proxy",
)
BASELINE_FEATURES = (
    "baseline_energy_fraction_proxy",
    "baseline_span_fraction_proxy",
    "negative_fraction",
)


def instrument_family(instrument: str) -> str:
    """Derive an acquisition platform family without consulting sensor metadata."""

    value = str(instrument).strip()
    if "-" not in value:
        raise ValueError(f"Instrument has no acquisition-unit suffix: {value!r}")
    family, suffix = value.rsplit("-", 1)
    if not family or not suffix:
        raise ValueError(f"Instrument cannot be mapped to a platform family: {value!r}")
    return family


def _normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"P02 manifest is missing required columns: {missing}")
    frame = manifest.copy()
    for column in (
        "observation_uid",
        "master_sample_id",
        "station",
        "instrument",
        "sensor_family",
        "target_analyte",
    ):
        frame[column] = frame[column].astype(str)
    if frame.observation_uid.duplicated().any():
        raise ValueError("P02 requires unique observation_uid values.")
    frame["instrument_family"] = frame.instrument.map(instrument_family)
    if frame[["master_sample_id", "station", "target_analyte"]].isna().any(axis=None):
        raise ValueError("P02 grouping and target fields must be complete.")
    conflicts = (
        frame.groupby("master_sample_id")[["station", "target_analyte"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError("A physical master maps to multiple stations or targets.")
    return frame.sort_values("observation_uid", kind="stable").reset_index(drop=True)


def _set_sha256(values: pd.Series | list[str] | set[str]) -> str:
    return sha256_value(sorted(str(value) for value in values))


def _records_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].astype(str).sort_values(columns, kind="stable")
    return sha256_value(ordered.to_dict(orient="records"))


def _domains(split_contract: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in split_contract["primary_domain_eligibility"]["domains"]:
        station, held = value.split(":", 1)
        rows.append(
            {
                "domain": value,
                "station": station,
                "held_instrument": held,
                "scope": "primary",
            }
        )
    for declaration in split_contract["exploratory_low_support_domains"]:
        value = declaration["domain"]
        station, held = value.split(":", 1)
        rows.append(
            {"domain": value, "station": station, "held_instrument": held, "scope": "exploratory"}
        )
    return rows


def build_master_splits(
    manifest: pd.DataFrame, split_contract: dict[str, Any]
) -> pd.DataFrame:
    masters = (
        manifest[["station", "master_sample_id", "target_analyte"]]
        .drop_duplicates()
        .sort_values(["station", "master_sample_id"], kind="stable")
    )
    rows: list[dict[str, Any]] = []
    seeds = split_contract["outer_repeat_seeds"]
    folds = int(split_contract["outer_folds_per_station"])
    for repeat_index, seed in enumerate(seeds, start=1):
        for station, station_masters in masters.groupby("station", sort=True):
            station_masters = station_masters.reset_index(drop=True)
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=int(seed),
            )
            assignments = np.full(len(station_masters), -1, dtype=int)
            for fold, (_, test_indices) in enumerate(
                splitter.split(
                    station_masters,
                    station_masters.target_analyte,
                    station_masters.master_sample_id,
                )
            ):
                assignments[test_indices] = fold
            if (assignments < 0).any():
                raise RuntimeError("An outer master did not receive a fold.")
            for index, master in station_masters.iterrows():
                rows.append(
                    {
                        "split_id": f"OS-r{repeat_index:02d}-{station}-f{assignments[index]:02d}",
                        "outer_repeat": repeat_index,
                        "outer_seed": int(seed),
                        "station": station,
                        "master_sample_id": master.master_sample_id,
                        "target_analyte": master.target_analyte,
                        "outer_fold": int(assignments[index]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["outer_repeat", "station", "outer_fold", "master_sample_id"], kind="stable"
    ).reset_index(drop=True)


def build_domain_registry(
    manifest: pd.DataFrame, split_contract: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in _domains(split_contract):
        selected = manifest[
            (manifest.station == domain["station"])
            & (manifest.instrument == domain["held_instrument"])
        ]
        rows.append(
            {
                **domain,
                "instrument_family": instrument_family(domain["held_instrument"]),
                "rows": len(selected),
                "masters": selected.master_sample_id.nunique(),
                "classes": selected.target_analyte.nunique(),
                "pooled_class_support": "|".join(
                    f"{name}:{count}"
                    for name, count in sorted(
                        selected.groupby("target_analyte").master_sample_id.nunique().items()
                    )
                ),
                "eligibility_status": (
                    "primary_supported"
                    if domain["scope"] == "primary"
                    else "exploratory_low_support"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "domain"], kind="stable").reset_index(drop=True)


def build_t3_partitions(
    manifest: pd.DataFrame,
    master_splits: pd.DataFrame,
    split_contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for domain in _domains(split_contract):
        station_rows = manifest[manifest.station == domain["station"]].copy()
        station_splits = master_splits[master_splits.station == domain["station"]]
        for repeat, repeat_splits in station_splits.groupby("outer_repeat", sort=True):
            fold_by_master = repeat_splits.set_index("master_sample_id").outer_fold
            seed = int(repeat_splits.outer_seed.iloc[0])
            attached = station_rows.copy()
            attached["master_outer_fold"] = attached.master_sample_id.map(fold_by_master)
            if attached.master_outer_fold.isna().any():
                raise RuntimeError("T3 derivation encountered an unassigned master.")
            for outer_fold in range(int(split_contract["outer_folds_per_station"])):
                is_test = attached.master_outer_fold.eq(outer_fold)
                is_held = attached.instrument.eq(domain["held_instrument"])
                role = np.select(
                    [~is_test & ~is_held, is_test & is_held, ~is_test & is_held],
                    ["train_source", "test_target", "excluded_train_target"],
                    default="excluded_test_source",
                )
                reason = np.select(
                    [~is_test & ~is_held, is_test & is_held, ~is_test & is_held],
                    [
                        "eligible_source_training_row",
                        "held_instrument_outer_test_row",
                        "held_instrument_forbidden_from_zero_shot_fit",
                    ],
                    default="outer_test_master_nonheld_view_preserved_not_evaluated",
                )
                piece = attached[
                    [
                        "observation_uid",
                        "master_sample_id",
                        "target_analyte",
                        "instrument",
                        "instrument_family",
                    ]
                ].copy()
                partition_id = (
                    f"T3-r{int(repeat):02d}-{domain['domain']}-f{outer_fold:02d}"
                )
                piece.insert(0, "partition_id", partition_id)
                piece.insert(1, "domain", domain["domain"])
                piece.insert(2, "domain_scope", domain["scope"])
                piece.insert(3, "station", domain["station"])
                piece.insert(4, "held_instrument", domain["held_instrument"])
                piece.insert(5, "outer_repeat", int(repeat))
                piece.insert(6, "outer_seed", seed)
                piece.insert(7, "outer_fold", outer_fold)
                piece["role"] = role
                piece["reason_code"] = reason
                rows.append(piece)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["domain_scope", "domain", "outer_repeat", "outer_fold", "observation_uid"],
        kind="stable",
    ).reset_index(drop=True)


def build_inner_selection(
    t3: pd.DataFrame,
    p02_contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    cell_columns = [
        "partition_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_seed",
        "outer_fold",
    ]
    primary = t3[t3.domain_scope == "primary"]
    for keys, cell in primary.groupby(cell_columns, sort=True):
        metadata = dict(zip(cell_columns, keys, strict=True))
        source = cell[cell.role == "train_source"].copy()
        required_classes = set(cell.target_analyte)
        source_instruments = sorted(source.instrument.unique())
        supported = 0
        for pseudo in source_instruments:
            validation = source[source.instrument == pseudo]
            fitting = source[source.instrument != pseudo]
            validation_classes = set(validation.target_analyte)
            fitting_classes = set(fitting.target_analyte)
            validation_masters = set(validation.master_sample_id)
            fitting_masters = set(fitting.master_sample_id)
            reasons: list[str] = []
            if validation_classes != required_classes:
                reasons.append("pseudo_validation_missing_station_class")
            if fitting_classes != required_classes:
                reasons.append("remaining_source_fit_missing_station_class")
            if validation_masters & fitting_masters:
                # Same physical sample across instruments is expected. It cannot be in both
                # roles, so all views of pseudo-validation masters are removed from fitting.
                fitting = fitting[~fitting.master_sample_id.isin(validation_masters)]
                fitting_masters = set(fitting.master_sample_id)
                fitting_classes = set(fitting.target_analyte)
                if fitting_classes != required_classes:
                    reasons.append("master_disjoint_fit_missing_station_class")
            status = not reasons
            supported += int(status)
            selection_rows.append(
                {
                    **metadata,
                    "pseudo_instrument": pseudo,
                    "pseudo_instrument_family": instrument_family(pseudo),
                    "supported": status,
                    "reason_code": "supported" if status else "|".join(sorted(set(reasons))),
                    "fit_rows": len(fitting),
                    "fit_masters": len(fitting_masters),
                    "validation_rows": len(validation),
                    "validation_masters": len(validation_masters),
                    "fit_observation_set_sha256": _set_sha256(fitting.observation_uid),
                    "validation_observation_set_sha256": _set_sha256(validation.observation_uid),
                    "master_disjoint": not bool(fitting_masters & validation_masters),
                }
            )
        source_masters = (
            source[["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .sort_values("master_sample_id", kind="stable")
            .reset_index(drop=True)
        )
        n_inner = int(p02_contract["inner_master_folds"])
        minimum_class = int(source_masters.groupby("target_analyte").size().min())
        if minimum_class >= n_inner:
            inner = StratifiedGroupKFold(
                n_splits=n_inner,
                shuffle=True,
                random_state=int(metadata["outer_seed"]) + int(metadata["outer_fold"]) + 1103,
            )
            assignments = np.full(len(source_masters), -1, dtype=int)
            for fold, (_, indices) in enumerate(
                inner.split(
                    source_masters,
                    source_masters.target_analyte,
                    source_masters.master_sample_id,
                )
            ):
                assignments[indices] = fold
            for index, master in source_masters.iterrows():
                fallback_rows.append(
                    {
                        **metadata,
                        "inner_fold": int(assignments[index]),
                        "master_sample_id": master.master_sample_id,
                        "target_analyte": master.target_analyte,
                        "selection_mode": "pseudo_domain" if supported >= 2 else "master_cv",
                    }
                )
        elif supported < 2:
            selection_rows.append(
                {
                    **metadata,
                    "pseudo_instrument": "not_applicable",
                    "pseudo_instrument_family": "not_applicable",
                    "supported": False,
                    "reason_code": "master_cv_fallback_also_unsupported",
                    "fit_rows": len(source),
                    "fit_masters": len(source_masters),
                    "validation_rows": 0,
                    "validation_masters": 0,
                    "fit_observation_set_sha256": _set_sha256(source.observation_uid),
                    "validation_observation_set_sha256": _set_sha256([]),
                    "master_disjoint": True,
                }
            )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold", "pseudo_instrument"], kind="stable"
    ).reset_index(drop=True)
    fallback = pd.DataFrame(fallback_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold", "inner_fold", "master_sample_id"],
        kind="stable",
    ).reset_index(drop=True)
    return selection, fallback


def build_family_support(
    t3: pd.DataFrame,
    policy_contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = sorted(
        int(value)
        for value in policy_contract["family_aware_policy"]["support_rule"][
            "minimum_masters_per_class"
        ]["candidate_values"]
    )
    minimum_units = int(
        policy_contract["family_aware_policy"]["support_rule"][
            "minimum_distinct_supported_source_units"
        ]
    )
    support_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    cell_columns = [
        "partition_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_seed",
        "outer_fold",
    ]
    primary = t3[t3.domain_scope == "primary"]
    for keys, cell in primary.groupby(cell_columns, sort=True):
        metadata = dict(zip(cell_columns, keys, strict=True))
        source = cell[cell.role == "train_source"]
        required_classes = sorted(cell.target_analyte.unique())
        held_family = instrument_family(metadata["held_instrument"])
        families = sorted(set(source.instrument_family) | {held_family})
        family_states: dict[str, dict[str, Any]] = {}
        for family in families:
            family_source = source[source.instrument_family == family]
            units = sorted(family_source.instrument.unique())
            supported_by_threshold: dict[int, list[str]] = {}
            for threshold in thresholds:
                supported_units: list[str] = []
                for unit in units:
                    counts = (
                        family_source[family_source.instrument == unit]
                        .groupby("target_analyte")
                        .master_sample_id.nunique()
                    )
                    if all(int(counts.get(target, 0)) >= threshold for target in required_classes):
                        supported_units.append(unit)
                supported_by_threshold[threshold] = supported_units
            viable = [
                threshold
                for threshold, units_at_threshold in supported_by_threshold.items()
                if len(units_at_threshold) >= minimum_units
            ]
            resolved = max(viable) if viable else None
            reference = resolved if resolved is not None else thresholds[0]
            supported_units = supported_by_threshold[reference]
            family_states[family] = {
                "known": bool(units),
                "supported": resolved is not None,
                "resolved_threshold": resolved,
                "units": units,
                "supported_units": supported_units,
            }
            support_rows.append(
                {
                    **metadata,
                    "instrument_family": family,
                    "held_instrument_family": held_family,
                    "family_known_in_source": bool(units),
                    "family_supported": resolved is not None,
                    "resolved_minimum_masters_per_class": (
                        resolved if resolved is not None else ""
                    ),
                    "source_units": "|".join(units) if units else "none",
                    "source_unit_count": len(units),
                    "supported_source_units": (
                        "|".join(supported_units) if supported_units else "none"
                    ),
                    "supported_source_unit_count": len(supported_units),
                    "threshold_support_counts": "|".join(
                        f"{threshold}:{len(supported_by_threshold[threshold])}"
                        for threshold in thresholds
                    ),
                    "support_uses_outcomes": False,
                }
            )
        held_state = family_states[held_family]
        if not held_state["known"]:
            family_status = "unknown_family"
            fallback = "unknown_family_to_PP-U-MIN"
        elif not held_state["supported"]:
            family_status = "known_unsupported_family"
            fallback = "known_unsupported_family_to_PP-U-MIN"
        else:
            family_status = "known_supported_family"
            fallback = "none"
        qc_source = source
        split_hash = _records_sha256(
            cell, ["observation_uid", "master_sample_id", "instrument", "role"]
        )
        estimator_hash = sha256_value(
            {
                "partition_id": metadata["partition_id"],
                "selection": "leave_one_supported_source_instrument_out",
                "objective": [
                    "mean_balanced_accuracy_desc",
                    "worst_balanced_accuracy_desc",
                    "macro_f1_desc",
                    "complexity_asc",
                    "declared_order_asc",
                ],
            }
        )
        family_hash = sha256_value(
            {
                "partition_id": metadata["partition_id"],
                "held_family": held_family,
                "state": held_state,
                "thresholds": thresholds,
                "minimum_units": minimum_units,
            }
        )
        qc_hash = sha256_value(
            {
                "partition_id": metadata["partition_id"],
                "source_observation_uids": sorted(qc_source.observation_uid),
                "features": [*NOISE_FEATURES, *BASELINE_FEATURES],
                "quantiles": policy_contract["qc_adaptive_policy"]["gate_library"][
                    "source_quantile_candidates"
                ],
            }
        )
        role_rows.append(
            {
                **metadata,
                "held_instrument_family": held_family,
                "family_status": family_status,
                "family_known": held_state["known"],
                "family_supported": held_state["supported"],
                "family_minimum_masters_per_class": (
                    held_state["resolved_threshold"]
                    if held_state["resolved_threshold"] is not None
                    else ""
                ),
                "family_source_unit_count": len(held_state["units"]),
                "family_supported_source_unit_count": len(held_state["supported_units"]),
                "family_fallback_reason": fallback,
                "family_permitted_target_access": "family_metadata_only",
                "qc_source_role": "train_source",
                "qc_source_rows": len(qc_source),
                "qc_source_masters": qc_source.master_sample_id.nunique(),
                "qc_permitted_features": "|".join([*NOISE_FEATURES, *BASELINE_FEATURES]),
                "qc_source_quantiles": "0.50|0.75|0.90",
                "qc_numeric_cutpoints_status": "future_fold_local_source_training",
                "qc_permitted_target_access": "row_local_qc_only",
                "universal_primary_action": "R_MIN_400_1800",
                "universal_sensitivity_actions": "R_SG_400_1800|R_ARPLS_400_1800",
                "split_state_sha256": split_hash,
                "estimator_selection_sha256": estimator_hash,
                "family_policy_sha256": family_hash,
                "qc_gate_state_sha256": qc_hash,
                "test_outcomes_used": False,
            }
        )
    support = pd.DataFrame(support_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold", "instrument_family"], kind="stable"
    ).reset_index(drop=True)
    roles = pd.DataFrame(role_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    return support, roles


def build_qc_gate_library(p02_contract: dict[str, Any]) -> pd.DataFrame:
    quantiles = p02_contract["qc_gate_enumeration"]["source_quantiles"]
    priorities = p02_contract["qc_gate_enumeration"]["priority_orders"]
    rows: list[dict[str, Any]] = [
        {
            "gate_candidate_id": "QC-000-MIN",
            "gate_kind": "baseline",
            "noise_feature": "none",
            "noise_quantile": "none",
            "baseline_feature": "none",
            "baseline_quantile": "none",
            "priority_order": "none",
            "noise_action": "R_MIN_400_1800",
            "baseline_action": "R_MIN_400_1800",
            "default_action": "R_MIN_400_1800",
            "numeric_cutpoints_status": "future_fold_local_source_training",
        }
    ]
    counter = 1
    for feature in [*NOISE_FEATURES, *BASELINE_FEATURES]:
        for quantile in quantiles:
            is_noise = feature in NOISE_FEATURES
            rows.append(
                {
                    "gate_candidate_id": f"QC-{counter:03d}-SINGLE",
                    "gate_kind": "single_trigger",
                    "noise_feature": feature if is_noise else "none",
                    "noise_quantile": quantile if is_noise else "none",
                    "baseline_feature": feature if not is_noise else "none",
                    "baseline_quantile": quantile if not is_noise else "none",
                    "priority_order": "single",
                    "noise_action": "R_SG_400_1800" if is_noise else "R_MIN_400_1800",
                    "baseline_action": (
                        "R_ARPLS_400_1800" if not is_noise else "R_MIN_400_1800"
                    ),
                    "default_action": "R_MIN_400_1800",
                    "numeric_cutpoints_status": "future_fold_local_source_training",
                }
            )
            counter += 1
    for noise in NOISE_FEATURES:
        for baseline in BASELINE_FEATURES:
            for noise_quantile in quantiles:
                for baseline_quantile in quantiles:
                    for priority in priorities:
                        rows.append(
                            {
                                "gate_candidate_id": f"QC-{counter:03d}-DUAL",
                                "gate_kind": "dual_trigger",
                                "noise_feature": noise,
                                "noise_quantile": noise_quantile,
                                "baseline_feature": baseline,
                                "baseline_quantile": baseline_quantile,
                                "priority_order": priority,
                                "noise_action": "R_SG_400_1800",
                                "baseline_action": "R_ARPLS_400_1800",
                                "default_action": "R_MIN_400_1800",
                                "numeric_cutpoints_status": "future_fold_local_source_training",
                            }
                        )
                        counter += 1
    return pd.DataFrame(rows)


def _draw_seed(metadata: dict[str, Any], regime: str, amount: int) -> int:
    return int(
        sha256_value(
            {
                "domain": metadata["domain"],
                "repeat": int(metadata["outer_repeat"]),
                "fold": int(metadata["outer_fold"]),
                "seed": int(metadata["outer_seed"]),
                "regime": regime,
                "amount": amount,
            }
        )[:8],
        16,
    )


def _balanced_total_draw(
    candidates: pd.DataFrame,
    source_class_counts: Counter[str],
    amount: int,
    seed: int,
) -> tuple[list[str], str]:
    random = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {}
    for target, group in candidates.groupby("target_analyte", sort=True):
        values = sorted(group.master_sample_id.unique())
        random.shuffle(values)
        capacity = max(0, int(source_class_counts[target]) - 1)
        by_class[target] = values[:capacity]
    selected: list[str] = []
    while len(selected) < amount:
        progressed = False
        for target in sorted(by_class):
            if by_class[target] and len(selected) < amount:
                selected.append(by_class[target].pop())
                progressed = True
        if not progressed:
            return [], "insufficient_calibration_masters_with_source_class_retention"
    return sorted(selected), "supported"


def _few_shot_draw(
    candidates: pd.DataFrame,
    source_class_counts: Counter[str],
    required_classes: list[str],
    k: int,
    seed: int,
) -> tuple[list[str], str]:
    random = np.random.default_rng(seed)
    selected: list[str] = []
    for target in required_classes:
        values = sorted(
            candidates[candidates.target_analyte == target].master_sample_id.unique()
        )
        random.shuffle(values)
        capacity = min(len(values), max(0, int(source_class_counts[target]) - 1))
        if capacity < k:
            return [], f"class_{target}_cannot_support_k{k}_and_retain_source_master"
        selected.extend(values[:k])
    return sorted(selected), "supported"


def build_target_access(
    t3: pd.DataFrame,
    p02_contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    cell_columns = [
        "partition_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_seed",
        "outer_fold",
    ]
    primary = t3[t3.domain_scope == "primary"]
    draw_contract = p02_contract["target_access_draws"]
    scenarios = [("zero_shot", 0, "none")]
    scenarios.extend(
        ("unlabeled_target_adaptation", int(value), "unlabeled_adaptation_master")
        for value in draw_contract["unlabeled_target_adaptation_total_masters"]
    )
    scenarios.extend(
        ("paired_calibration", int(value), "paired_calibration_master")
        for value in draw_contract["paired_calibration_total_masters"]
    )
    scenarios.extend(
        ("supervised_few_shot", int(value), "labelled_calibration_master")
        for value in draw_contract["supervised_few_shot_masters_per_class"]
    )
    for keys, cell in primary.groupby(cell_columns, sort=True):
        metadata = dict(zip(cell_columns, keys, strict=True))
        master_rows = (
            cell[["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .sort_values("master_sample_id", kind="stable")
        )
        test_masters = set(
            cell[cell.role.isin(["test_target", "excluded_test_source"])].master_sample_id
        )
        target_view = set(cell[cell.instrument == metadata["held_instrument"]].master_sample_id)
        source_view = set(cell[cell.instrument != metadata["held_instrument"]].master_sample_id)
        train_masters = set(master_rows.master_sample_id) - test_masters
        candidates = master_rows[
            master_rows.master_sample_id.isin(train_masters & target_view & source_view)
        ]
        source_counts = Counter(
            master_rows[master_rows.master_sample_id.isin(train_masters & source_view)]
            .groupby("target_analyte")
            .master_sample_id.nunique()
            .to_dict()
        )
        required_classes = sorted(master_rows.target_analyte.unique())
        evaluation = test_masters & target_view
        for regime, amount, calibration_role in scenarios:
            seed = _draw_seed(metadata, regime, amount)
            if regime == "zero_shot":
                selected: list[str] = []
                reason = "supported"
            elif regime == "supervised_few_shot":
                selected, reason = _few_shot_draw(
                    candidates, source_counts, required_classes, amount, seed
                )
            else:
                selected, reason = _balanced_total_draw(candidates, source_counts, amount, seed)
            if not evaluation:
                selected = []
                reason = "no_outer_test_held_instrument_view"
            supported = reason == "supported"
            selected_set = set(selected) if supported else set()
            scenario_id = (
                f"{metadata['partition_id']}::{regime}::"
                + (f"k{amount}" if regime == "supervised_few_shot" else f"n{amount}")
            )
            scenario_rows.append(
                {
                    **metadata,
                    "scenario_id": scenario_id,
                    "information_regime": regime,
                    "requested_masters": amount,
                    "amount_unit": (
                        "per_class" if regime == "supervised_few_shot" else "total"
                    ),
                    "supported": supported,
                    "reason_code": reason,
                    "draw_seed": seed,
                    "candidate_masters": len(candidates),
                    "selected_calibration_or_adaptation_masters": len(selected_set),
                    "evaluation_masters": len(evaluation),
                    "target_labels_accessible": regime == "supervised_few_shot",
                    "pair_ids_accessible": regime == "paired_calibration",
                    "target_batch_intensities_accessible": regime != "zero_shot",
                    "preprocessing_reselection_authorized": False,
                    "assignment_sha256": sha256_value(
                        {
                            "scenario": scenario_id,
                            "selected": sorted(selected_set),
                            "evaluation": sorted(evaluation),
                        }
                    ),
                }
            )
            for master in master_rows.itertuples(index=False):
                master_id = master.master_sample_id
                if master_id in test_masters:
                    if master_id in evaluation:
                        role = "evaluation_only"
                        assignment_reason = "unchanged_outer_test_master_with_target_view"
                    else:
                        role = "excluded_evaluation_no_target_view"
                        assignment_reason = "outer_test_master_has_no_held_instrument_view"
                elif master_id in selected_set:
                    role = calibration_role
                    assignment_reason = "deterministic_supported_target_access_draw"
                elif master_id in source_view:
                    role = "source_training_master"
                    assignment_reason = "outer_training_master_not_selected_for_target_access"
                else:
                    role = "excluded_no_source_training_view"
                    assignment_reason = "outer_training_master_has_no_nonheld_instrument_view"
                assignment_rows.append(
                    {
                        **metadata,
                        "scenario_id": scenario_id,
                        "information_regime": regime,
                        "master_sample_id": master_id,
                        "target_analyte": master.target_analyte,
                        "target_access_role": role,
                        "reason_code": assignment_reason,
                        "scenario_supported": supported,
                    }
                )
    scenarios_frame = pd.DataFrame(scenario_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold", "information_regime", "requested_masters"],
        kind="stable",
    ).reset_index(drop=True)
    assignments = pd.DataFrame(assignment_rows).sort_values(
        ["scenario_id", "master_sample_id"], kind="stable"
    ).reset_index(drop=True)
    return scenarios_frame, assignments


def build_open_set_partitions(
    manifest: pd.DataFrame,
    master_splits: pd.DataFrame,
    split_contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for task in split_contract["tasks"]["T4_OPEN"]["held_tasks"]:
        station, held_target = task.split(":", 1)
        station_rows = manifest[manifest.station == station].copy()
        station_splits = master_splits[master_splits.station == station]
        for repeat, repeat_splits in station_splits.groupby("outer_repeat", sort=True):
            fold_by_master = repeat_splits.set_index("master_sample_id").outer_fold
            seed = int(repeat_splits.outer_seed.iloc[0])
            attached = station_rows.copy()
            attached["master_outer_fold"] = attached.master_sample_id.map(fold_by_master)
            for outer_fold in range(int(split_contract["outer_folds_per_station"])):
                is_test = attached.master_outer_fold.eq(outer_fold)
                is_unknown = attached.target_analyte.eq(held_target)
                role = np.select(
                    [~is_test & ~is_unknown, is_test & is_unknown, is_test & ~is_unknown],
                    ["train_known", "test_unknown", "test_known"],
                    default="excluded_train_unknown",
                )
                reason = np.select(
                    [~is_test & ~is_unknown, is_test & is_unknown, is_test & ~is_unknown],
                    [
                        "known_outer_training_row",
                        "held_chemical_outer_test_row",
                        "known_outer_test_row",
                    ],
                    default="held_chemical_forbidden_from_all_development_roles",
                )
                piece = attached[
                    ["observation_uid", "master_sample_id", "target_analyte", "instrument"]
                ].copy()
                open_partition_id = f"OPEN-r{int(repeat):02d}-{task}-f{outer_fold:02d}"
                piece.insert(0, "open_partition_id", open_partition_id)
                piece.insert(1, "open_task", task)
                piece.insert(2, "station", station)
                piece.insert(3, "held_target", held_target)
                piece.insert(4, "outer_repeat", int(repeat))
                piece.insert(5, "outer_seed", seed)
                piece.insert(6, "outer_fold", outer_fold)
                piece["role"] = role
                piece["reason_code"] = reason
                rows.append(piece)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["open_task", "outer_repeat", "outer_fold", "observation_uid"], kind="stable"
    ).reset_index(drop=True)


def build_leakage_audit(
    master_splits: pd.DataFrame,
    t3: pd.DataFrame,
    preprocessing_roles: pd.DataFrame,
    target_scenarios: pd.DataFrame,
    target_assignments: pd.DataFrame,
    open_set: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    cell_columns = ["partition_id", "domain", "outer_repeat", "outer_fold"]
    primary = t3[t3.domain_scope == "primary"]
    for keys, cell in primary.groupby(cell_columns, sort=True):
        metadata = dict(zip(cell_columns, keys, strict=True))
        held = str(cell.held_instrument.iloc[0])
        required_classes = set(cell.target_analyte)
        train = cell[cell.role == "train_source"]
        test = cell[cell.role == "test_target"]
        train_masters = set(train.master_sample_id)
        outer_test_masters = set(
            cell[cell.role.isin(["test_target", "excluded_test_source"])].master_sample_id
        )
        checks = {
            "partition_assignment_complete_unique": cell.observation_uid.nunique() == len(cell),
            "held_instrument_absent_from_train_source": not train.instrument.eq(held).any(),
            "outer_test_master_absent_from_train_source": not bool(
                train_masters & outer_test_masters
            ),
            "train_source_contains_all_station_classes": set(train.target_analyte)
            == required_classes,
            "test_target_nonempty": not test.empty,
        }
        for check_id, status in checks.items():
            audit_rows.append(
                {
                    **metadata,
                    "check_id": check_id,
                    "severity": "fatal",
                    "status": "pass" if status else "fail",
                    "detail": "boolean invariant",
                }
            )
        fold_has_all = set(test.target_analyte) == required_classes
        audit_rows.append(
            {
                **metadata,
                "check_id": "held_instrument_fold_contains_all_station_classes",
                "severity": "informational",
                "status": "pass" if fold_has_all else "recorded",
                "detail": "not used to construct folds; repeat-pooled inference is definitive",
            }
        )
        if not fold_has_all:
            unsupported.append(
                {
                    "cell_type": "sparse_held_instrument_fold",
                    "cell_id": metadata["partition_id"],
                    "domain": metadata["domain"],
                    "outer_repeat": metadata["outer_repeat"],
                    "outer_fold": metadata["outer_fold"],
                    "reason_code": (
                        "held_instrument_fold_missing_class_"
                        "pooled_repeat_remains_supported"
                    ),
                    "fatal": False,
                }
            )
    for domain, group in primary.groupby("domain", sort=True):
        required_classes = set(group.target_analyte)
        for repeat, repeated in group.groupby("outer_repeat", sort=True):
            pooled = repeated[repeated.role == "test_target"]
            status = set(pooled.target_analyte) == required_classes
            audit_rows.append(
                {
                    "partition_id": f"{domain}::repeat-{int(repeat):02d}",
                    "domain": domain,
                    "outer_repeat": int(repeat),
                    "outer_fold": "pooled",
                    "check_id": "repeat_pooled_test_contains_all_station_classes",
                    "severity": "fatal",
                    "status": "pass" if status else "fail",
                    "detail": "definitive domain-repeat inference unit",
                }
            )
    for row in preprocessing_roles.itertuples(index=False):
        hashes = {
            row.split_state_sha256,
            row.estimator_selection_sha256,
            row.family_policy_sha256,
            row.qc_gate_state_sha256,
        }
        audit_rows.append(
            {
                "partition_id": row.partition_id,
                "domain": row.domain,
                "outer_repeat": row.outer_repeat,
                "outer_fold": row.outer_fold,
                "check_id": "split_estimator_family_qc_hashes_distinct",
                "severity": "fatal",
                "status": "pass" if len(hashes) == 4 else "fail",
                "detail": "independent protected selection states",
            }
        )
        if not row.family_supported:
            unsupported.append(
                {
                    "cell_type": "family_policy_fallback",
                    "cell_id": row.partition_id,
                    "domain": row.domain,
                    "outer_repeat": row.outer_repeat,
                    "outer_fold": row.outer_fold,
                    "reason_code": row.family_fallback_reason,
                    "fatal": False,
                }
            )
    for scenario in target_scenarios.itertuples(index=False):
        assignments = target_assignments[
            target_assignments.scenario_id == scenario.scenario_id
        ]
        selected = set(
            assignments[
                assignments.target_access_role.isin(
                    [
                        "unlabeled_adaptation_master",
                        "paired_calibration_master",
                        "labelled_calibration_master",
                    ]
                )
            ].master_sample_id
        )
        evaluation = set(
            assignments[assignments.target_access_role == "evaluation_only"].master_sample_id
        )
        status = not bool(selected & evaluation) and assignments.master_sample_id.is_unique
        audit_rows.append(
            {
                "partition_id": scenario.partition_id,
                "domain": scenario.domain,
                "outer_repeat": scenario.outer_repeat,
                "outer_fold": scenario.outer_fold,
                "check_id": (
                    f"target_access_disjoint::{scenario.information_regime}::"
                    f"{scenario.requested_masters}"
                ),
                "severity": "fatal",
                "status": "pass" if status else "fail",
                "detail": scenario.scenario_id,
            }
        )
        if not scenario.supported:
            unsupported.append(
                {
                    "cell_type": "target_access_scenario",
                    "cell_id": scenario.scenario_id,
                    "domain": scenario.domain,
                    "outer_repeat": scenario.outer_repeat,
                    "outer_fold": scenario.outer_fold,
                    "reason_code": scenario.reason_code,
                    "fatal": False,
                }
            )
    for open_id, cell in open_set.groupby("open_partition_id", sort=True):
        held = str(cell.held_target.iloc[0])
        status = not cell[cell.role == "train_known"].target_analyte.eq(held).any()
        audit_rows.append(
            {
                "partition_id": open_id,
                "domain": str(cell.open_task.iloc[0]),
                "outer_repeat": int(cell.outer_repeat.iloc[0]),
                "outer_fold": int(cell.outer_fold.iloc[0]),
                "check_id": "held_chemical_absent_from_development",
                "severity": "fatal",
                "status": "pass" if status else "fail",
                "detail": "training selection calibration and threshold roles prohibited",
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["domain", "outer_repeat", "outer_fold", "check_id"], kind="stable"
    ).reset_index(drop=True)
    unsupported_frame = pd.DataFrame(unsupported).sort_values(
        ["cell_type", "domain", "outer_repeat", "outer_fold", "cell_id"], kind="stable"
    ).reset_index(drop=True)
    return audit, unsupported_frame


def build_p02_tables(
    manifest: pd.DataFrame,
    split_contract: dict[str, Any],
    policy_contract: dict[str, Any],
    p02_contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Materialize every protected P02 registry without fitting a predictive model."""

    normalized = _normalize_manifest(manifest)
    master_splits = build_master_splits(normalized, split_contract)
    domain_registry = build_domain_registry(normalized, split_contract)
    t3 = build_t3_partitions(normalized, master_splits, split_contract)
    inner_selection, inner_master = build_inner_selection(t3, p02_contract)
    family_support, preprocessing_roles = build_family_support(t3, policy_contract)
    qc_gates = build_qc_gate_library(p02_contract)
    target_scenarios, target_assignments = build_target_access(t3, p02_contract)
    open_set = build_open_set_partitions(normalized, master_splits, split_contract)
    leakage, unsupported = build_leakage_audit(
        master_splits,
        t3,
        preprocessing_roles,
        target_scenarios,
        target_assignments,
        open_set,
    )
    return {
        "master_split_registry.csv": master_splits,
        "t3_partition_registry.csv": t3,
        "inner_selection_registry.csv": inner_selection,
        "inner_master_split_registry.csv": inner_master,
        "domain_registry.csv": domain_registry,
        "family_support_registry.csv": family_support,
        "preprocessing_policy_roles.csv": preprocessing_roles,
        "qc_gate_candidate_registry.csv": qc_gates,
        "target_access_scenario_registry.csv": target_scenarios,
        "target_access_assignment_registry.csv": target_assignments,
        "open_set_partition_registry.csv": open_set,
        "unsupported_cells.csv": unsupported,
        "leakage_audit.csv": leakage,
    }


def validate_p02_tables(
    tables: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    split_contract: dict[str, Any],
    p02_contract: dict[str, Any],
) -> dict[str, bool]:
    """Run fatal, reconstruction-oriented checks over materialized registries."""

    normalized = _normalize_manifest(manifest)
    master_splits = tables["master_split_registry.csv"]
    t3 = tables["t3_partition_registry.csv"]
    domain_registry = tables["domain_registry.csv"]
    roles = tables["preprocessing_policy_roles.csv"]
    target_scenarios = tables["target_access_scenario_registry.csv"]
    target_assignments = tables["target_access_assignment_registry.csv"]
    open_set = tables["open_set_partition_registry.csv"]
    leakage = tables["leakage_audit.csv"]
    expected_repeats = len(split_contract["outer_repeat_seeds"])
    expected_folds = int(split_contract["outer_folds_per_station"])
    expected_master_rows = normalized.master_sample_id.nunique() * expected_repeats
    primary_domains = domain_registry[domain_registry.scope == "primary"]
    exploratory_domains = domain_registry[domain_registry.scope == "exploratory"]
    reconstructed_t3 = build_t3_partitions(normalized, master_splits, split_contract)
    exact_columns = [
        "partition_id",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "role",
        "reason_code",
    ]
    allowed_preprocessing_columns = set(roles.columns)
    forbidden_qc_columns = {
        "sensor_family",
        "target_analyte",
        "master_sample_id",
        "model_confidence",
        "test_outcome",
    }
    assignments_unique = not target_assignments.duplicated(
        ["scenario_id", "master_sample_id"]
    ).any()
    scenario_counts_match = all(
        len(target_assignments[target_assignments.scenario_id == row.scenario_id])
        == normalized[normalized.station == row.station].master_sample_id.nunique()
        for row in target_scenarios.itertuples(index=False)
    )
    held_unknown_absent = all(
        not group[group.role == "train_known"].target_analyte.eq(group.held_target.iloc[0]).any()
        for _, group in open_set.groupby("open_partition_id", sort=False)
    )
    hashes_distinct = all(
        len(
            {
                row.split_state_sha256,
                row.estimator_selection_sha256,
                row.family_policy_sha256,
                row.qc_gate_state_sha256,
            }
        )
        == 4
        for row in roles.itertuples(index=False)
    )
    checks = {
        "source_population_exact": len(normalized)
        == int(p02_contract["expected_population"]["rows"])
        and normalized.master_sample_id.nunique()
        == int(p02_contract["expected_population"]["physical_masters"]),
        "master_assignment_count_exact": len(master_splits) == expected_master_rows,
        "master_once_per_repeat": not master_splits.duplicated(
            ["outer_repeat", "master_sample_id"]
        ).any(),
        "four_folds_per_station_repeat": all(
            group.outer_fold.nunique() == expected_folds
            for _, group in master_splits.groupby(["outer_repeat", "station"])
        ),
        "outer_training_classes_complete": all(
            set(group[group.outer_fold != fold].target_analyte) == set(group.target_analyte)
            for _, group in master_splits.groupby(["outer_repeat", "station"])
            for fold in range(expected_folds)
        ),
        "thirteen_primary_domains_exact": len(primary_domains) == 13,
        "four_exploratory_domains_exact": len(exploratory_domains) == 4,
        "domain_registry_matches_contract": set(primary_domains.domain)
        == set(split_contract["primary_domain_eligibility"]["domains"]),
        "t3_reconstructs_exactly_from_metadata": t3[exact_columns].equals(
            reconstructed_t3[exact_columns]
        ),
        "t3_partition_assignments_complete_unique": all(
            len(group) == group.observation_uid.nunique()
            for _, group in t3.groupby("partition_id")
        ),
        "held_instrument_never_train_source": not t3[
            t3.role == "train_source"
        ].instrument.eq(t3[t3.role == "train_source"].held_instrument).any(),
        "fatal_leakage_audit_passes": leakage[leakage.severity == "fatal"].status.eq("pass").all(),
        "preprocessing_role_count_exact": len(roles)
        == 13 * expected_repeats * expected_folds,
        "family_and_sensor_semantics_separate": all(
            instrument_family(value) == family
            for value, family in normalized[["instrument", "instrument_family"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ),
        "qc_roles_exclude_forbidden_identity_and_outcome_fields": not bool(
            forbidden_qc_columns & allowed_preprocessing_columns
        ),
        "qc_gate_library_complete": len(tables["qc_gate_candidate_registry.csv"])
        == int(p02_contract["qc_gate_enumeration"]["expected_candidates"]),
        "qc_cutpoints_remain_unresolved": tables["qc_gate_candidate_registry.csv"]
        .numeric_cutpoints_status.eq("future_fold_local_source_training")
        .all(),
        "selection_state_hashes_are_distinct": hashes_distinct,
        "target_access_assignment_unique": assignments_unique,
        "target_access_assignment_complete": scenario_counts_match,
        "target_access_and_evaluation_disjoint": all(
            not (
                set(
                    group[
                        group.target_access_role.isin(
                            [
                                "unlabeled_adaptation_master",
                                "paired_calibration_master",
                                "labelled_calibration_master",
                            ]
                        )
                    ].master_sample_id
                )
                & set(group[group.target_access_role == "evaluation_only"].master_sample_id)
            )
            for _, group in target_assignments.groupby("scenario_id")
        ),
        "held_chemical_absent_from_development": held_unknown_absent,
        "test_outcomes_never_used": roles.test_outcomes_used.eq(False).all(),  # noqa: E712
    }
    return {name: bool(value) for name, value in checks.items()}
