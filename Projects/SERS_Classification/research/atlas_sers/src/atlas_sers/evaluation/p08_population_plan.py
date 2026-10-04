"""P08 filtered-population metadata plan builder (no scientific execution).

This module re-derives outer master folds plus source, selection and calibration
metadata roles for a supplied filtered population. It is a pure in-memory
metadata builder: it never fits a model, selects a recipe, computes a score or
produces a prediction.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from atlas_sers.governance.canonical import sha256_value
from atlas_sers.splits import p02

_EXPECTED_ALGORITHM = "StratifiedGroupKFold with shuffle true and repeat seed"
_REGISTERED_SELECTION_FALLBACK = (
    "three-fold stratified master-grouped inner CV"
)
_REGISTERED_VALIDATION_UNIT = "one source acquisition instrument"
_PSEUDO_SUPPORT_FIELDS = (
    "validation_unit",
    "validation_requires_all_station_classes",
    "remaining_training_requires_all_station_classes",
    "minimum_supported_pseudo_domains",
    "fallback",
)
_T3_ROLES = frozenset({
    "train_source",
    "test_target",
    "excluded_train_target",
    "excluded_test_source",
})
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "station",
    "instrument",
    "target_analyte",
)
_DOMAIN_COLUMNS = (
    "population_id",
    "population_sha256",
    "domain",
    "station",
    "instrument",
    "instrument_family",
    "observed_rows",
    "observed_masters",
    "observed_classes",
    "required_classes",
    "required_masters",
    "eligible",
    "reason_code",
)
_CONTEXT_COLUMNS = (
    "population_id",
    "population_sha256",
    "context_id",
    "domain",
    "station",
    "held_instrument",
    "instrument_family",
    "outer_repeat",
    "outer_seed",
    "outer_fold",
    "domain_eligible",
    "source_has_all_station_classes",
    "source_rows",
    "source_masters",
    "source_classes",
    "test_rows",
    "test_masters",
    "test_classes",
    "sparse_test_fold",
    "empty_test_fold",
    "selection_mode",
    "selection_supported",
    "calibration_supported",
    "metadata_ready",
    "reason_code",
    "source_observation_set_sha256",
    "test_observation_set_sha256",
    "source_master_set_sha256",
    "test_master_set_sha256",
)
_PSEUDO_COLUMNS = (
    "population_id",
    "population_sha256",
    "context_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_seed",
    "outer_fold",
    "pseudo_instrument",
    "pseudo_instrument_family",
    "supported",
    "reason_code",
    "fit_rows",
    "fit_masters",
    "fit_classes",
    "validation_rows",
    "validation_masters",
    "validation_classes",
    "fit_observation_set_sha256",
    "validation_observation_set_sha256",
    "master_disjoint",
)
_ASSIGN_COLUMNS = (
    "population_id",
    "population_sha256",
    "context_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_seed",
    "outer_fold",
    "inner_fold",
    "master_sample_id",
    "target_analyte",
    "selection_mode",
)
_UNIT_COLUMNS = (
    "population_id",
    "population_sha256",
    "context_id",
    "purpose",
    "unit_id",
    "support",
    "reason_code",
    "fit_rows",
    "fit_masters",
    "fit_classes",
    "validation_rows",
    "validation_masters",
    "validation_classes",
    "fit_observation_set_sha256",
    "validation_observation_set_sha256",
    "master_disjoint",
    "held_instrument_absent",
)
_ROLE_COLUMNS = (
    "population_id",
    "population_sha256",
    "context_id",
    "purpose",
    "unit_id",
    "role",
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
)
_ROLE_IDENTITY = (
    "context_id",
    "purpose",
    "unit_id",
    "role",
    "observation_uid",
)


@dataclass(frozen=True)
class PopulationPlanTables:
    master_splits: pd.DataFrame
    domain_registry: pd.DataFrame
    t3_partitions: pd.DataFrame
    context_registry: pd.DataFrame
    inner_selection_registry: pd.DataFrame
    inner_master_split_registry: pd.DataFrame
    unit_registry: pd.DataFrame
    role_registry: pd.DataFrame
    validation_report: dict[str, Any]


def _hash_obj(obj: Any) -> str:
    return sha256_value(obj)


def _hash_set(values: Iterable[Any]) -> str:
    return sha256_value(sorted(set(values)))


def _hash_table(frame: pd.DataFrame) -> str:
    return sha256_value(
        {
            "columns": [str(column) for column in frame.columns],
            "records": frame.to_dict(orient="records"),
        }
    )


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows, columns=list(columns))
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _require_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_population_id(value: Any) -> None:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ValueError("population_id must be a nonempty safe token")


def _require_digest(value: Any) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError("population_sha256 must be 64 lowercase hex characters")


def _validate_split(split_contract: dict) -> tuple[int, int, int]:
    algorithm = split_contract.get("canonical_algorithm")
    if algorithm is not None and algorithm != _EXPECTED_ALGORITHM:
        raise ValueError(f"unsupported canonical_algorithm: {algorithm!r}")
    if split_contract.get("stratification_label") not in (None, "target_analyte"):
        raise ValueError("stratification_label must be target_analyte")
    if split_contract.get("group_label") not in (None, "master_sample_id"):
        raise ValueError("group_label must be master_sample_id")
    seeds = split_contract.get("outer_repeat_seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("outer_repeat_seeds must be a nonempty sequence")
    validated = [
        _require_int(seed, "outer_repeat_seed", 0) for seed in seeds
    ]
    if len(set(validated)) != len(validated):
        raise ValueError("outer_repeat_seeds must be distinct")
    outer_folds = _require_int(
        split_contract.get("outer_folds_per_station"),
        "outer_folds_per_station",
        2,
    )
    eligibility = split_contract.get("primary_domain_eligibility")
    if not isinstance(eligibility, dict) or not isinstance(
        eligibility.get("requirements"), dict
    ):
        raise ValueError("primary_domain_eligibility.requirements is required")
    requirements = eligibility["requirements"]
    required_classes = _require_int(
        requirements.get("test_classes"), "test_classes", 1
    )
    required_masters = _require_int(
        requirements.get("pooled_test_masters_minimum"),
        "pooled_test_masters_minimum",
        1,
    )
    return outer_folds, required_classes, required_masters


def _validate_pseudo_support(p02_contract: dict) -> None:
    rule = p02_contract.get("pseudo_domain_support")
    if rule is None:
        return
    if not isinstance(rule, dict):
        raise TypeError(
            "pseudo_domain_support must be a mapping when supplied"
        )
    expected = set(_PSEUDO_SUPPORT_FIELDS)
    missing = sorted(expected - set(rule))
    if missing:
        raise ValueError(
            f"pseudo_domain_support is missing fields: {missing}"
        )
    unknown = sorted(set(rule) - expected)
    if unknown:
        raise ValueError(
            f"pseudo_domain_support has unknown fields: {unknown}"
        )
    validation_unit = rule["validation_unit"]
    if (
        not isinstance(validation_unit, str)
        or validation_unit != _REGISTERED_VALIDATION_UNIT
    ):
        raise ValueError(
            "pseudo_domain_support.validation_unit must equal "
            f"{_REGISTERED_VALIDATION_UNIT!r}"
        )
    for name in (
        "validation_requires_all_station_classes",
        "remaining_training_requires_all_station_classes",
    ):
        if rule[name] is not True:
            raise ValueError(
                f"pseudo_domain_support.{name} must be true"
            )
    minimum = rule["minimum_supported_pseudo_domains"]
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError(
            "pseudo_domain_support.minimum_supported_pseudo_domains "
            "must be an int"
        )
    if minimum != 2:
        raise ValueError(
            "pseudo_domain_support.minimum_supported_pseudo_domains "
            "must be 2"
        )
    fallback = rule["fallback"]
    if (
        not isinstance(fallback, str)
        or fallback != _REGISTERED_SELECTION_FALLBACK
    ):
        raise ValueError(
            "pseudo_domain_support.fallback must equal "
            f"{_REGISTERED_SELECTION_FALLBACK!r}"
        )


def _parse_domains(split_contract: dict) -> list[tuple[str, str, str]]:
    domains = split_contract["primary_domain_eligibility"].get("domains")
    if not isinstance(domains, (list, tuple)) or not domains:
        raise ValueError(
            "primary_domain_eligibility.domains must be a nonempty sequence"
        )
    parsed: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw in domains:
        if not isinstance(raw, str) or raw.count(":") != 1:
            raise ValueError(f"malformed domain token: {raw!r}")
        station, instrument = raw.split(":")
        if not station.strip() or not instrument.strip():
            raise ValueError(f"malformed domain token: {raw!r}")
        if raw in seen:
            raise ValueError(f"duplicate domain token: {raw!r}")
        seen.add(raw)
        parsed.append((raw, station, instrument))
    return parsed


def _normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.columns.has_duplicates:
        duplicated = sorted(set(manifest.columns[manifest.columns.duplicated()]))
        raise ValueError(f"manifest has duplicate column labels: {duplicated}")
    missing = [name for name in _REQUIRED_COLUMNS if name not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    frame = manifest.loc[:, list(_REQUIRED_COLUMNS)].copy()
    for name in _REQUIRED_COLUMNS:
        values = frame[name]
        nonblank = values.map(
            lambda item: isinstance(item, str) and item.strip() != ""
        )
        if values.isna().any() or not nonblank.all():
            raise ValueError(f"{name} must contain nonblank string values")
    if frame["observation_uid"].duplicated().any():
        raise ValueError("observation_uid values must be unique")
    labels = frame[["master_sample_id", "station", "target_analyte"]]
    labels = labels.drop_duplicates()
    if labels["master_sample_id"].duplicated().any():
        raise ValueError(
            "each master_sample_id must map to exactly one station and target"
        )
    frame["instrument_family"] = frame["instrument"].map(
        p02.instrument_family
    )
    return frame.sort_values(
        [
            "station",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "observation_uid",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _unit_record(
    *,
    provenance: dict[str, Any],
    context_id: str,
    purpose: str,
    unit_id: str,
    support: bool,
    reason: str,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    held_instrument: str,
) -> dict[str, Any]:
    fit_masters = set(fit["master_sample_id"])
    validation_masters = set(validation["master_sample_id"])
    return {
        **provenance,
        "context_id": context_id,
        "purpose": purpose,
        "unit_id": unit_id,
        "support": bool(support),
        "reason_code": reason,
        "fit_rows": int(len(fit)),
        "fit_masters": len(fit_masters),
        "fit_classes": int(fit["target_analyte"].nunique()),
        "validation_rows": int(len(validation)),
        "validation_masters": len(validation_masters),
        "validation_classes": int(validation["target_analyte"].nunique()),
        "fit_observation_set_sha256": _hash_set(fit["observation_uid"]),
        "validation_observation_set_sha256": _hash_set(
            validation["observation_uid"]
        ),
        "master_disjoint": not bool(fit_masters & validation_masters),
        "held_instrument_absent": bool(
            held_instrument not in set(fit["instrument"])
            and held_instrument not in set(validation["instrument"])
        ),
    }


def _role_records(
    *,
    provenance: dict[str, Any],
    context_id: str,
    purpose: str,
    unit_id: str,
    role: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    return [
        {
            **provenance,
            "context_id": context_id,
            "purpose": purpose,
            "unit_id": unit_id,
            "role": role,
            "observation_uid": uid,
            "master_sample_id": mid,
            "target_analyte": target,
            "instrument": inst,
        }
        for uid, mid, target, inst in zip(
            frame["observation_uid"],
            frame["master_sample_id"],
            frame["target_analyte"],
            frame["instrument"],
            strict=True,
        )
    ]


def _reconstruct_inner(
    *,
    provenance: dict[str, Any],
    context_id: str,
    source: pd.DataFrame,
    station_classes: set[str],
    n_inner: int,
    outer_seed: int,
    outer_fold: int,
    held_instrument: str,
) -> dict[str, Any]:
    source = source.reset_index(drop=True)
    pseudo_rows: list[dict[str, Any]] = []
    pseudo_specs: list[dict[str, Any]] = []
    if source.empty:
        pseudo_rows.append({
            **provenance,
            "context_id": context_id,
            "pseudo_instrument": "not_applicable",
            "pseudo_instrument_family": "not_applicable",
            "supported": False,
            "reason_code": "no_source_rows",
            "fit_rows": 0,
            "fit_masters": 0,
            "fit_classes": 0,
            "validation_rows": 0,
            "validation_masters": 0,
            "validation_classes": 0,
            "fit_observation_set_sha256": _hash_set([]),
            "validation_observation_set_sha256": _hash_set([]),
            "master_disjoint": True,
        })
    else:
        for instrument in sorted(source["instrument"].unique()):
            validation = source[source["instrument"] == instrument]
            fitting = source[source["instrument"] != instrument]
            validation_masters = set(validation["master_sample_id"])
            reasons: list[str] = []
            if set(validation["target_analyte"]) != station_classes:
                reasons.append("pseudo_validation_missing_station_class")
            if set(fitting["target_analyte"]) != station_classes:
                reasons.append(
                    "remaining_source_fit_missing_station_class"
                )
            overlap = validation_masters & set(fitting["master_sample_id"])
            if overlap:
                fitting = fitting[
                    ~fitting["master_sample_id"].isin(validation_masters)
                ]
                if set(fitting["target_analyte"]) != station_classes:
                    reasons.append(
                        "master_disjoint_fit_missing_station_class"
                    )
            support = not reasons
            reason = (
                "supported" if support else "|".join(sorted(set(reasons)))
            )
            pseudo_rows.append({
                **provenance,
                "context_id": context_id,
                "pseudo_instrument": instrument,
                "pseudo_instrument_family": p02.instrument_family(instrument),
                "supported": support,
                "reason_code": reason,
                "fit_rows": int(len(fitting)),
                "fit_masters": len(set(fitting["master_sample_id"])),
                "fit_classes": int(fitting["target_analyte"].nunique()),
                "validation_rows": int(len(validation)),
                "validation_masters": len(validation_masters),
                "validation_classes": int(
                    validation["target_analyte"].nunique()
                ),
                "fit_observation_set_sha256": _hash_set(
                    fitting["observation_uid"]
                ),
                "validation_observation_set_sha256": _hash_set(
                    validation["observation_uid"]
                ),
                "master_disjoint": not bool(
                    set(fitting["master_sample_id"]) & validation_masters
                ),
            })
            pseudo_specs.append({
                "unit_id": f"pseudo:{instrument}",
                "support": support,
                "reason": reason,
                "fit": fitting,
                "validation": validation,
            })

    source_has_all_classes = bool(
        station_classes
        and set(source["target_analyte"]) == station_classes
    )
    master_cv_reason = ""
    inner_rows: list[dict[str, Any]] = []
    calibration_specs: list[dict[str, Any]] = []
    fold_masters: dict[int, set[str]] = {}
    assignments_supported = False
    if source.empty:
        master_cv_reason = "no_source_rows"
    elif not source_has_all_classes:
        master_cv_reason = "missing_station_class"
    else:
        source_masters = (
            source[["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .sort_values(
                ["master_sample_id", "target_analyte"], kind="stable"
            )
            .reset_index(drop=True)
        )
        class_counts = source_masters.groupby("target_analyte").size()
        if int(class_counts.min()) < n_inner:
            master_cv_reason = "insufficient_class_masters"
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_inner,
                shuffle=True,
                random_state=int(outer_seed) + int(outer_fold) + 1103,
            )
            assignments = pd.Series(-1, index=source_masters.index, dtype=int)
            for fold, (_, test_indices) in enumerate(
                splitter.split(
                    source_masters,
                    source_masters["target_analyte"],
                    source_masters["master_sample_id"],
                )
            ):
                assignments.iloc[test_indices] = fold
            if (assignments < 0).any():
                raise RuntimeError(
                    "an inner source master did not receive a fold"
                )
            assignments_supported = True
            for index, master in source_masters.iterrows():
                inner_fold = int(assignments.iloc[index])
                fold_masters.setdefault(inner_fold, set()).add(
                    master["master_sample_id"]
                )
                inner_rows.append({
                    **provenance,
                    "context_id": context_id,
                    "inner_fold": inner_fold,
                    "master_sample_id": master["master_sample_id"],
                    "target_analyte": master["target_analyte"],
                    "selection_mode": "pending",
                })
            for inner_fold in range(n_inner):
                validation_masters = fold_masters.get(inner_fold, set())
                validation = source[
                    source["master_sample_id"].isin(validation_masters)
                ]
                fitting = source[
                    ~source["master_sample_id"].isin(validation_masters)
                ]
                reasons = []
                if set(validation["target_analyte"]) != station_classes:
                    reasons.append("inner_validation_missing_station_class")
                if set(fitting["target_analyte"]) != station_classes:
                    reasons.append("inner_fit_missing_station_class")
                support = not reasons
                calibration_specs.append({
                    "unit_id": f"inner_fold:{inner_fold}",
                    "support": support,
                    "reason": (
                        "supported"
                        if support
                        else "|".join(sorted(set(reasons)))
                    ),
                    "fit": fitting,
                    "validation": validation,
                })

    calibration_supported = bool(
        assignments_supported
        and len(calibration_specs) == n_inner
        and all(spec["support"] for spec in calibration_specs)
    )
    supported_pseudo = [spec for spec in pseudo_specs if spec["support"]]
    if len(supported_pseudo) >= 2:
        selection_mode = "pseudo_domain"
        selection_specs = supported_pseudo
    elif assignments_supported and calibration_supported:
        selection_mode = "master_cv"
        selection_specs = calibration_specs
    else:
        selection_mode = "unsupported"
        selection_specs = []
    selection_supported = selection_mode != "unsupported"
    for row in inner_rows:
        row["selection_mode"] = selection_mode

    def _units(
        purpose: str, specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            _unit_record(
                provenance=provenance,
                context_id=context_id,
                purpose=purpose,
                unit_id=spec["unit_id"],
                support=spec["support"],
                reason=spec["reason"],
                fit=spec["fit"],
                validation=spec["validation"],
                held_instrument=held_instrument,
            )
            for spec in specs
        ]

    roles: list[dict[str, Any]] = []
    for purpose, specs in (
        ("selection", selection_specs),
        ("calibration", calibration_specs),
    ):
        for spec in specs:
            roles.extend(_role_records(
                provenance=provenance,
                context_id=context_id,
                purpose=purpose,
                unit_id=spec["unit_id"],
                role="fit",
                frame=spec["fit"],
            ))
            roles.extend(_role_records(
                provenance=provenance,
                context_id=context_id,
                purpose=purpose,
                unit_id=spec["unit_id"],
                role="validation",
                frame=spec["validation"],
            ))
    return {
        "pseudo_rows": pseudo_rows,
        "inner_rows": inner_rows,
        "selection_units": _units("selection", selection_specs),
        "calibration_units": _units("calibration", calibration_specs),
        "roles": roles,
        "selection_mode": selection_mode,
        "selection_supported": selection_supported,
        "calibration_supported": calibration_supported,
        "master_cv_reason": master_cv_reason,
    }


def build_population_plan(
    *,
    manifest: pd.DataFrame,
    population_id: str,
    population_sha256: str,
    split_contract: dict,
    p02_contract: dict,
) -> PopulationPlanTables:
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    if not isinstance(split_contract, dict) or not isinstance(
        p02_contract, dict
    ):
        raise TypeError("split_contract and p02_contract must be dicts")
    _require_population_id(population_id)
    _require_digest(population_sha256)
    _validate_pseudo_support(p02_contract)
    split_sha = _hash_obj(split_contract)
    p02_sha = _hash_obj(p02_contract)
    outer_folds, required_classes, required_masters = _validate_split(
        split_contract
    )
    n_inner = _require_int(
        p02_contract.get("inner_master_folds"), "inner_master_folds", 2
    )
    domains = _parse_domains(split_contract)
    normalized = _normalize_manifest(manifest)
    station_set = set(normalized["station"])
    for _domain, station, _instrument in domains:
        if station not in station_set:
            raise ValueError(
                f"domain station absent from manifest: {station!r}"
            )
    metadata_sha = _hash_table(normalized)
    provenance = {
        "population_id": population_id,
        "population_sha256": population_sha256,
    }

    raw_splits = p02.build_master_splits(normalized, split_contract)
    master_splits = raw_splits.copy()
    master_splits["population_id"] = population_id
    master_splits["population_sha256"] = population_sha256
    master_splits = master_splits.sort_values(
        ["outer_repeat", "station", "outer_fold", "master_sample_id"],
        kind="stable",
    ).reset_index(drop=True)

    fold_lookup: dict[int, pd.Series] = {}
    seed_lookup: dict[int, int] = {}
    for repeat, group in master_splits.groupby("outer_repeat", sort=True):
        fold_lookup[int(repeat)] = group.set_index("master_sample_id")[
            "outer_fold"
        ]
        seed_lookup[int(repeat)] = int(group["outer_seed"].iloc[0])

    domain_rows: list[dict[str, Any]] = []
    eligible_by_domain: dict[str, bool] = {}
    reason_by_domain: dict[str, str] = {}
    for domain, station, instrument in domains:
        subset = normalized[
            (normalized["station"] == station)
            & (normalized["instrument"] == instrument)
        ]
        observed_rows = int(len(subset))
        observed_masters = int(subset["master_sample_id"].nunique())
        observed_classes = int(subset["target_analyte"].nunique())
        reasons: list[str] = []
        if observed_rows == 0:
            reasons.append("no_observations")
        if observed_classes != required_classes:
            reasons.append("class_support_insufficient")
        if observed_masters < required_masters:
            reasons.append("master_support_insufficient")
        reason = "eligible" if not reasons else "|".join(sorted(reasons))
        eligible_by_domain[domain] = reason == "eligible"
        reason_by_domain[domain] = reason
        domain_rows.append({
            **provenance,
            "domain": domain,
            "station": station,
            "instrument": instrument,
            "instrument_family": p02.instrument_family(instrument),
            "observed_rows": observed_rows,
            "observed_masters": observed_masters,
            "observed_classes": observed_classes,
            "required_classes": required_classes,
            "required_masters": required_masters,
            "eligible": eligible_by_domain[domain],
            "reason_code": reason,
        })
    domain_registry = _frame(domain_rows, _DOMAIN_COLUMNS).sort_values(
        ["domain"], kind="stable"
    ).reset_index(drop=True)

    station_classes = {
        station: set(group["target_analyte"])
        for station, group in normalized.groupby("station")
    }
    station_observation_uids = {
        station: set(group["observation_uid"])
        for station, group in normalized.groupby("station")
    }
    context_data: list[tuple[str, dict[str, Any]]] = []
    context_lookup: dict[tuple[str, int, int], str] = {}
    context_source_uids: dict[str, set[str]] = {}
    context_test_uids: dict[str, set[str]] = {}
    context_station_uids: dict[str, set[str]] = {}
    for domain, station, instrument in domains:
        station_rows = normalized[normalized["station"] == station].copy()
        classes = station_classes.get(station, set())
        family = p02.instrument_family(instrument)
        for repeat in sorted(fold_lookup):
            station_rows["_fold"] = station_rows["master_sample_id"].map(
                fold_lookup[repeat]
            )
            if station_rows["_fold"].isna().any():
                raise RuntimeError(
                    "outer split derivation left an unassigned master"
                )
            for fold in range(outer_folds):
                context_id = "P08POPCTX-" + _hash_obj({
                    "population_id": population_id,
                    "population_sha256": population_sha256,
                    "metadata_sha256": metadata_sha,
                    "split_contract_sha256": split_sha,
                    "p02_contract_sha256": p02_sha,
                    "domain": domain,
                    "outer_repeat": repeat,
                    "outer_fold": fold,
                })[:24]
                is_test = station_rows["_fold"] == fold
                is_held = station_rows["instrument"] == instrument
                source = station_rows[~is_test & ~is_held]
                test = station_rows[is_test & is_held]
                if set(source["master_sample_id"]) & set(
                    test["master_sample_id"]
                ):
                    raise ValueError(
                        f"outer source/test masters overlap for {context_id}"
                    )
                if instrument in set(source["instrument"]):
                    raise ValueError(
                        f"held instrument present in outer source: {context_id}"
                    )
                source_classes = set(source["target_analyte"])
                test_classes = set(test["target_analyte"])
                record = {
                    **provenance,
                    "context_id": context_id,
                    "domain": domain,
                    "station": station,
                    "held_instrument": instrument,
                    "instrument_family": family,
                    "outer_repeat": repeat,
                    "outer_seed": seed_lookup[repeat],
                    "outer_fold": fold,
                    "domain_eligible": eligible_by_domain[domain],
                    "source_has_all_station_classes": bool(
                        classes and source_classes >= classes
                    ),
                    "source_rows": int(len(source)),
                    "source_masters": int(source["master_sample_id"].nunique()),
                    "source_classes": int(
                        source["target_analyte"].nunique()
                    ),
                    "test_rows": int(len(test)),
                    "test_masters": int(test["master_sample_id"].nunique()),
                    "test_classes": int(test["target_analyte"].nunique()),
                    "sparse_test_fold": bool(
                        len(test) and test_classes != classes
                    ),
                    "empty_test_fold": len(test) == 0,
                    "source_observation_set_sha256": _hash_set(
                        source["observation_uid"]
                    ),
                    "test_observation_set_sha256": _hash_set(
                        test["observation_uid"]
                    ),
                    "source_master_set_sha256": _hash_set(
                        source["master_sample_id"]
                    ),
                    "test_master_set_sha256": _hash_set(
                        test["master_sample_id"]
                    ),
                }
                context_data.append((context_id, {
                    "record": record,
                    "source": source,
                    "test": test,
                    "classes": classes,
                    "seed": seed_lookup[repeat],
                    "fold": fold,
                    "domain": domain,
                    "station": station,
                    "held_instrument": instrument,
                }))
                context_lookup[(domain, repeat, fold)] = context_id
                context_source_uids[context_id] = set(
                    source["observation_uid"]
                )
                context_test_uids[context_id] = set(test["observation_uid"])
                context_station_uids[context_id] = set(
                    station_observation_uids[station]
                )

    planned_ids = [context_id for context_id, _ in context_data]
    if len(set(planned_ids)) != len(planned_ids):
        raise ValueError("planned context IDs are not unique")

    pseudo_records: list[dict[str, Any]] = []
    inner_records: list[dict[str, Any]] = []
    selection_units: list[dict[str, Any]] = []
    calibration_units: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    for context_id, data in context_data:
        record = data["record"]
        eligible = eligible_by_domain[data["domain"]]
        inner: dict[str, Any] | None = None
        if eligible:
            inner = _reconstruct_inner(
                provenance={
                    **provenance,
                    "domain": data["domain"],
                    "station": data["station"],
                    "held_instrument": data["held_instrument"],
                    "outer_repeat": record["outer_repeat"],
                    "outer_seed": record["outer_seed"],
                    "outer_fold": record["outer_fold"],
                },
                context_id=context_id,
                source=data["source"],
                station_classes=data["classes"],
                n_inner=n_inner,
                outer_seed=data["seed"],
                outer_fold=data["fold"],
                held_instrument=data["held_instrument"],
            )
            record["selection_mode"] = inner["selection_mode"]
            record["selection_supported"] = inner["selection_supported"]
            record["calibration_supported"] = inner["calibration_supported"]
            pseudo_records.extend(inner["pseudo_rows"])
            inner_records.extend(inner["inner_rows"])
            selection_units.extend(inner["selection_units"])
            calibration_units.extend(inner["calibration_units"])
            role_records.extend(inner["roles"])
        else:
            record["selection_mode"] = "not_applicable"
            record["selection_supported"] = False
            record["calibration_supported"] = False
        role_records.extend(_role_records(
            provenance=provenance,
            context_id=context_id,
            purpose="outer",
            unit_id="outer",
            role="source",
            frame=data["source"],
        ))
        role_records.extend(_role_records(
            provenance=provenance,
            context_id=context_id,
            purpose="outer",
            unit_id="outer",
            role="test",
            frame=data["test"],
        ))
        reasons: list[str] = []
        if not eligible:
            reasons.append(reason_by_domain[data["domain"]])
        if record["source_rows"] == 0:
            reasons.append("empty_source")
        if not record["source_has_all_station_classes"]:
            reasons.append("source_missing_station_class")
        if record["empty_test_fold"]:
            reasons.append("empty_test_fold")
        if record["sparse_test_fold"]:
            reasons.append("sparse_test_fold")
        if not record["selection_supported"]:
            reasons.append("selection_unsupported")
        if not record["calibration_supported"]:
            reasons.append("calibration_unsupported")
        if inner is not None and inner["master_cv_reason"]:
            reasons.append(inner["master_cv_reason"])
        if not reasons:
            reasons.append("metadata_ready")
        record["reason_code"] = "|".join(sorted(set(reasons)))
        record["metadata_ready"] = bool(
            eligible
            and record["source_has_all_station_classes"]
            and not record["empty_test_fold"]
            and record["selection_supported"]
            and record["calibration_supported"]
        )

    for unit in selection_units + calibration_units:
        if not unit["master_disjoint"]:
            raise ValueError(
                f"unit {unit['context_id']}:{unit['unit_id']} masters overlap"
            )
        if not unit["held_instrument_absent"]:
            raise ValueError(
                f"unit {unit['context_id']}:{unit['unit_id']} "
                "contains held instrument"
            )
    for row in role_records:
        if row["purpose"] in ("selection", "calibration"):
            if row["observation_uid"] not in context_source_uids[
                row["context_id"]
            ]:
                raise ValueError(
                    "selection/calibration role leaves its context source"
                )

    context_registry = _frame(
        [data["record"] for _, data in context_data], _CONTEXT_COLUMNS
    ).sort_values(
        ["domain", "outer_repeat", "outer_fold"], kind="stable"
    ).reset_index(drop=True)

    t3_contract = copy.deepcopy(split_contract)
    t3_contract["exploratory_low_support_domains"] = []
    t3_partitions = p02.build_t3_partitions(
        normalized.copy(), master_splits.copy(), t3_contract
    ).copy()
    t3_partitions["population_id"] = population_id
    t3_partitions["population_sha256"] = population_sha256
    t3_context_ids: list[str] = []
    for domain, repeat, fold in zip(
        t3_partitions["domain"],
        t3_partitions["outer_repeat"],
        t3_partitions["outer_fold"],
        strict=True,
    ):
        key = (domain, int(repeat), int(fold))
        mapped = context_lookup.get(key)
        if mapped is None:
            raise KeyError(f"t3 partition has no planned context for {key!r}")
        t3_context_ids.append(mapped)
    t3_partitions["context_id"] = t3_context_ids
    if set(t3_context_ids) != set(planned_ids):
        raise ValueError(
            "t3 partitions must cover every planned context"
        )
    for (domain, repeat, fold), group in t3_partitions.groupby(
        ["domain", "outer_repeat", "outer_fold"], sort=False
    ):
        key = (domain, int(repeat), int(fold))
        mapped = context_lookup.get(key)
        if mapped is None:
            raise KeyError(f"t3 partition has no planned context for {key!r}")
        roles = group["role"]
        unknown_roles = set(roles) - _T3_ROLES
        if unknown_roles:
            raise ValueError(
                f"t3 context {mapped} has unknown role values: "
                f"{sorted(unknown_roles)!r}"
            )
        uids = list(group["observation_uid"])
        if len(uids) != len(set(uids)):
            raise ValueError(
                f"t3 context {mapped} maps an observation more than once"
            )
        if set(uids) != context_station_uids[mapped]:
            raise ValueError(
                f"t3 context {mapped} does not equal all station observations"
            )
        train = set(
            group.loc[roles.eq("train_source"), "observation_uid"]
        )
        target = set(
            group.loc[roles.eq("test_target"), "observation_uid"]
        )
        if train & target:
            raise ValueError(
                f"t3 context {mapped} train_source/test_target overlap"
            )
        if train != context_source_uids[mapped]:
            raise ValueError(
                f"t3 context {mapped} train_source disagrees with source"
            )
        if target != context_test_uids[mapped]:
            raise ValueError(
                f"t3 context {mapped} test_target disagrees with test"
            )
    t3_partitions = t3_partitions.sort_values(
        [
            "domain_scope",
            "domain",
            "outer_repeat",
            "outer_fold",
            "observation_uid",
        ],
        kind="stable",
    ).reset_index(drop=True)

    inner_selection_registry = _frame(
        pseudo_records, _PSEUDO_COLUMNS
    ).sort_values(
        ["domain", "outer_repeat", "outer_fold", "pseudo_instrument"],
        kind="stable",
    ).reset_index(drop=True)
    inner_master_split_registry = _frame(
        inner_records, _ASSIGN_COLUMNS
    ).sort_values(
        [
            "domain",
            "outer_repeat",
            "outer_fold",
            "inner_fold",
            "master_sample_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    unit_registry = _frame(
        selection_units + calibration_units, _UNIT_COLUMNS
    ).sort_values(
        ["context_id", "purpose", "unit_id"], kind="stable"
    ).reset_index(drop=True)
    role_registry = _frame(role_records, _ROLE_COLUMNS)
    duplicated_roles = role_registry.duplicated(subset=list(_ROLE_IDENTITY))
    if duplicated_roles.any():
        raise ValueError("role_registry contains duplicate role identity rows")
    role_registry = role_registry.sort_values(
        ["context_id", "purpose", "unit_id", "role", "observation_uid"],
        kind="stable",
    ).reset_index(drop=True)

    table_hashes = {
        "master_splits": _hash_table(master_splits),
        "domain_registry": _hash_table(domain_registry),
        "t3_partitions": _hash_table(t3_partitions),
        "context_registry": _hash_table(context_registry),
        "inner_selection_registry": _hash_table(inner_selection_registry),
        "inner_master_split_registry": _hash_table(
            inner_master_split_registry
        ),
        "unit_registry": _hash_table(unit_registry),
        "role_registry": _hash_table(role_registry),
    }
    plan_sha256 = _hash_obj({
        "population_id": population_id,
        "population_sha256": population_sha256,
        "metadata_canonical_sha256": metadata_sha,
        "split_contract_sha256": split_sha,
        "p02_contract_sha256": p02_sha,
        "table_hashes": table_hashes,
    })

    def _count(mask: pd.Series) -> int:
        return int(mask.sum()) if len(mask) else 0

    reason_counts: dict[str, int] = {}
    for row in context_registry.to_dict(orient="records"):
        reason_counts[row["reason_code"]] = (
            reason_counts.get(row["reason_code"], 0) + 1
        )
    validation_report = {
        "status": "metadata_plan_only",
        "execution_authorized": False,
        "scientific_operations": 0,
        "exact_fit_counts_enumerated": False,
        "population_id": population_id,
        "population_sha256": population_sha256,
        "metadata_canonical_sha256": metadata_sha,
        "split_contract_sha256": split_sha,
        "p02_contract_sha256": p02_sha,
        "plan_id": "P08PLAN-" + plan_sha256[:16],
        "plan_sha256": plan_sha256,
        "table_hashes": table_hashes,
        "counts": {
            "domain_count": int(len(domain_registry)),
            "eligible_domain_count": _count(domain_registry["eligible"]),
            "context_count": int(len(context_registry)),
            "metadata_ready_context_count": _count(
                context_registry["metadata_ready"]
            ),
            "selection_supported_context_count": _count(
                context_registry["selection_supported"]
            ),
            "calibration_supported_context_count": _count(
                context_registry["calibration_supported"]
            ),
            "sparse_test_context_count": _count(
                context_registry["sparse_test_fold"]
            ),
            "empty_test_context_count": _count(
                context_registry["empty_test_fold"]
            ),
            "source_unit_count": _count(
                context_registry["source_rows"] > 0
            ),
            "pseudo_candidate_count": int(len(inner_selection_registry)),
            "pseudo_supported_count": _count(
                inner_selection_registry["supported"]
            ),
            "selection_unit_count": _count(
                unit_registry["purpose"] == "selection"
            ),
            "calibration_unit_count": _count(
                unit_registry["purpose"] == "calibration"
            ),
            "role_row_count": int(len(role_registry)),
        },
        "diagnostics": {
            "context_reason_counts": dict(sorted(reason_counts.items()))
        },
        "denied_operations": [
            "model_fitting",
            "recipe_selection",
            "metric_computation",
            "preprocessing_fit",
            "prediction",
            "benchmark_completion_claim",
        ],
    }
    return PopulationPlanTables(
        master_splits=master_splits,
        domain_registry=domain_registry,
        t3_partitions=t3_partitions,
        context_registry=context_registry,
        inner_selection_registry=inner_selection_registry,
        inner_master_split_registry=inner_master_split_registry,
        unit_registry=unit_registry,
        role_registry=role_registry,
        validation_report=validation_report,
    )
