"""Outcome-blind expansion and validation of the locked P13 execution plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.governance.canonical import sha256_file, sha256_value

PROTOCOL_VERSION = "nato-sers-p13-v1-locked"
ELIGIBLE_TIERS = {"confirmatory", "exploratory_low_support"}
POLICY_REPRESENTATIONS = {
    "PP-U-MIN": "R_MIN_400_1800",
    "PP-U-SG": "R_SG_400_1800",
    "PP-U-ARPLS": "R_ARPLS_400_1800",
}
PROCEDURE_FAMILIES = {
    "C-RBF-SVM": "C-RBF-SVM",
    "C-PCA-LDA": "C-PCA-LDA",
    "C-PLS-DA": "C-PLS-DA",
    "C-LOGREG-EN": "C-LOGREG-EN",
    "C-RANDOM-FOREST": "C-RANDOM-FOREST",
    "C-EXTRA-TREES": "C-EXTRA-TREES",
}
PROCEDURE_ORDER = ("C-SELECTED", *PROCEDURE_FAMILIES)


@dataclass(frozen=True)
class P13PlanTables:
    context_registry: pd.DataFrame
    role_registry: pd.DataFrame
    procedure_registry: pd.DataFrame
    fit_manifest: pd.DataFrame
    expected_endpoint_registry: pd.DataFrame
    shard_manifest: pd.DataFrame
    input_hashes: dict[str, str]
    validation_report: dict[str, Any]


def _uid_hash(values: list[str] | pd.Series | np.ndarray) -> str:
    return sha256_value(sorted(str(value) for value in values))


def _json_values(values: list[str] | pd.Series | np.ndarray) -> str:
    return json.dumps(sorted(str(value) for value in values), separators=(",", ":"))


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}-{sha256_value(payload)[:24]}"


def _sort_winner(frame: pd.DataFrame) -> pd.Series:
    supported = frame[frame.complete_support.astype(bool)].copy()
    if supported.empty:
        raise ValueError("No complete P03 source-only candidate is available.")
    return supported.sort_values(
        [
            "mean_balanced_accuracy",
            "worst_balanced_accuracy",
            "mean_macro_f1",
            "complexity_rank",
            "declared_candidate_order",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).iloc[0]


def _candidate_lookup(candidate_registry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {
        "candidate_id",
        "model_id",
        "parameters_json",
        "hyperparameter_sha256",
        "technical_seeds",
        "seed_count",
        "declared_candidate_order",
    }
    if missing := sorted(required - set(candidate_registry)):
        raise ValueError(f"P03 candidate registry is missing fields: {missing}")
    if candidate_registry.candidate_id.astype(str).duplicated().any():
        raise ValueError("P03 candidate IDs are not unique.")
    return {
        str(row.candidate_id): row._asdict()
        for row in candidate_registry.itertuples(index=False)
    }


def _p03_resolution(
    *,
    selected_specs: pd.DataFrame,
    selection_trace: pd.DataFrame,
    candidate_registry: pd.DataFrame,
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Resolve primary and fixed-family candidates only from frozen P03 evidence."""

    candidates = _candidate_lookup(candidate_registry)
    c09_specs = selected_specs[
        selected_specs.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    c09_trace = selection_trace[
        selection_trace.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    if len(c09_specs) != 260 or c09_specs.status.astype(str).ne("complete").any():
        raise ValueError("The frozen P03 C09 selection mapping is incomplete.")
    resolutions: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    keys = ["domain", "outer_repeat", "outer_fold"]
    for key_values, group in c09_trace.groupby(keys, sort=True):
        domain, repeat, fold = key_values
        context_specs = c09_specs[
            c09_specs.domain.astype(str).eq(str(domain))
            & c09_specs.outer_repeat.eq(int(repeat))
            & c09_specs.outer_fold.eq(int(fold))
        ]
        if len(context_specs) != 1:
            raise ValueError("P03 C09 selection context is not unique.")
        selected_candidate = str(context_specs.selected_candidate_id.iloc[0])
        resolutions[(str(domain), int(repeat), int(fold), "C-SELECTED")] = {
            **candidates[selected_candidate],
            "selection_source": "P03_EXP-C09_source_only",
            "selection_state_sha256": str(
                context_specs.selection_state_sha256.iloc[0]
            ),
        }
        for procedure_id, family in PROCEDURE_FAMILIES.items():
            family_trace = group[group.model_id.astype(str).eq(family)]
            winner = _sort_winner(family_trace)
            candidate_id = str(winner.candidate_id)
            resolutions[(str(domain), int(repeat), int(fold), procedure_id)] = {
                **candidates[candidate_id],
                "selection_source": "P03_EXP-C09_family_source_only",
                "selection_state_sha256": sha256_value(
                    {
                        "p03_domain": str(domain),
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "model_id": family,
                        "candidate_id": candidate_id,
                    }
                ),
            }

    c10 = selected_specs[
        selected_specs.experiment_id.astype(str).eq("EXP-C10-T3")
    ]
    for row in c10.itertuples(index=False):
        key = (
            str(row.domain),
            int(row.outer_repeat),
            int(row.outer_fold),
            str(row.selected_model_id),
        )
        derived = resolutions.get(key)
        if derived is None or str(derived["candidate_id"]) != str(
            row.selected_candidate_id
        ):
            raise ValueError("Derived P13 family candidate differs from frozen P03 C10.")
    return resolutions


def _seeds(record: dict[str, Any]) -> list[str]:
    values = str(record["technical_seeds"]).split("|")
    if len(values) != int(record["seed_count"]):
        raise ValueError("Candidate seed count differs from its seed registry.")
    return values


def _role_record(role_context_id: str, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    uids = frame.observation_uid.astype(str).tolist()
    masters = frame.master_sample_id.astype(str).tolist()
    return {
        "role_context_id": role_context_id,
        "role_name": name,
        "observation_count": len(uids),
        "master_count": len(set(masters)),
        "instrument_count": frame.instrument.astype(str).nunique(),
        "class_count": frame.target_analyte.astype(str).nunique(),
        "observation_uid_sha256": _uid_hash(uids),
        "master_id_sha256": _uid_hash(sorted(set(masters))),
        "observation_uids_json": _json_values(uids),
    }


def _outer_roles(
    *,
    manifest: pd.DataFrame,
    master_splits: pd.DataFrame,
    inner_splits: pd.DataFrame,
    domain: pd.Series,
    repeat: int,
    fold: int,
) -> tuple[str, list[dict[str, Any]], int, str]:
    station = str(domain.station)
    substrate = str(domain.substrate_family)
    held = str(domain.held_instrument)
    domain_name = f"{station}:{held}"
    split = master_splits[
        master_splits.station.astype(str).eq(station)
        & master_splits.outer_repeat.eq(repeat)
    ]
    test_masters = set(split.loc[split.outer_fold.eq(fold), "master_sample_id"])
    station_substrate = manifest[
        manifest.station.astype(str).eq(station)
        & manifest.sensor_family.astype(str).eq(substrate)
    ]
    source = station_substrate[
        ~station_substrate.instrument.astype(str).eq(held)
    ]
    held_rows = station_substrate[
        station_substrate.instrument.astype(str).eq(held)
        & station_substrate.master_sample_id.isin(test_masters)
    ]
    source_train = source[~source.master_sample_id.isin(test_masters)]
    held_master_ids = set(held_rows.master_sample_id)
    source_test = source[source.master_sample_id.isin(held_master_ids)]
    paired_masters = held_master_ids & set(source_test.master_sample_id)
    matched_held = held_rows[held_rows.master_sample_id.isin(paired_masters)]
    source_test = source_test[source_test.master_sample_id.isin(paired_masters)]
    role_context_id = _stable_id(
        "P13ROLE",
        {
            "domain_id": str(domain.domain_id),
            "outer_repeat": repeat,
            "outer_fold": fold,
        },
    )
    roles = [
        _role_record(role_context_id, "outer_source_fit", source_train),
        _role_record(role_context_id, "outer_held_test", held_rows),
        _role_record(role_context_id, "outer_matched_held_test", matched_held),
        _role_record(role_context_id, "outer_matched_source_test", source_test),
    ]

    inner = inner_splits[
        inner_splits.domain.astype(str).eq(domain_name)
        & inner_splits.outer_repeat.eq(repeat)
        & inner_splits.outer_fold.eq(fold)
    ]
    if inner.empty:
        calibration_source = "derived_non_test_outer_folds_exploratory"
        inner_assignments = split[~split.outer_fold.eq(fold)][
            ["master_sample_id", "outer_fold"]
        ].rename(columns={"outer_fold": "inner_fold"})
        fold_map = {
            original: replacement
            for replacement, original in enumerate(
                sorted(inner_assignments.inner_fold.unique())
            )
        }
        inner_assignments["inner_fold"] = inner_assignments.inner_fold.map(fold_map)
    else:
        calibration_source = "P02_inner_master_split_registry"
        inner_assignments = inner[["master_sample_id", "inner_fold"]].drop_duplicates()
    training_masters = set(source_train.master_sample_id)
    inner_assignments = inner_assignments[
        inner_assignments.master_sample_id.isin(training_masters)
    ]
    folds = sorted(int(value) for value in inner_assignments.inner_fold.unique())
    for inner_fold in folds:
        validation_masters = set(
            inner_assignments.loc[
                inner_assignments.inner_fold.eq(inner_fold), "master_sample_id"
            ]
        )
        inner_validation = source_train[
            source_train.master_sample_id.isin(validation_masters)
        ]
        inner_fit = source_train[
            ~source_train.master_sample_id.isin(validation_masters)
        ]
        roles.extend(
            [
                _role_record(
                    role_context_id,
                    f"calibration_fit_{inner_fold}",
                    inner_fit,
                ),
                _role_record(
                    role_context_id,
                    f"calibration_validation_{inner_fold}",
                    inner_validation,
                ),
            ]
        )
    return role_context_id, roles, len(folds), calibration_source


def _role_map(role_registry: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if role_registry.duplicated(["role_context_id", "role_name"]).any():
        raise ValueError("P13 role registry contains duplicate roles.")
    return {
        (str(row.role_context_id), str(row.role_name)): row
        for _, row in role_registry.iterrows()
    }


def build_p13_plan(
    *,
    manifest: pd.DataFrame,
    master_splits: pd.DataFrame,
    inner_splits: pd.DataFrame,
    domain_support: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    selected_specs: pd.DataFrame,
    selection_trace: pd.DataFrame,
    input_paths: dict[str, Path],
) -> P13PlanTables:
    """Expand every P13 role and fit without loading spectral intensities."""

    eligible = domain_support[domain_support.support_tier.isin(ELIGIBLE_TIERS)].copy()
    if len(manifest) != 598 or manifest.master_sample_id.nunique() != 69:
        raise ValueError("P13 requires the immutable 598-spectrum/69-master population.")
    if eligible.support_tier.value_counts().to_dict() != {
        "confirmatory": 13,
        "exploratory_low_support": 3,
    }:
        raise ValueError("P13 eligible-domain support differs from the locked registry.")
    resolutions = _p03_resolution(
        selected_specs=selected_specs,
        selection_trace=selection_trace,
        candidate_registry=candidate_registry,
    )
    candidates = _candidate_lookup(candidate_registry)
    first_family_candidate = {
        family: candidate_registry[
            candidate_registry.model_id.astype(str).eq(family)
        ]
        .sort_values("declared_candidate_order", kind="stable")
        .iloc[0]
        for family in PROCEDURE_FAMILIES.values()
    }

    contexts: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    for domain in eligible.sort_values("domain_id", kind="stable").itertuples(index=False):
        domain_series = pd.Series(domain._asdict())
        for repeat in range(1, 6):
            for fold in range(4):
                role_context_id, role_rows, inner_fold_count, calibration_source = (
                    _outer_roles(
                        manifest=manifest,
                        master_splits=master_splits,
                        inner_splits=inner_splits,
                        domain=domain_series,
                        repeat=repeat,
                        fold=fold,
                    )
                )
                roles.extend(role_rows)
                for policy_id, representation_id in POLICY_REPRESENTATIONS.items():
                    context = {
                        "domain_id": str(domain.domain_id),
                        "station": str(domain.station),
                        "substrate_family": str(domain.substrate_family),
                        "held_instrument": str(domain.held_instrument),
                        "support_tier": str(domain.support_tier),
                        "outer_repeat": repeat,
                        "outer_fold": fold,
                        "policy_id": policy_id,
                        "representation_id": representation_id,
                        "role_context_id": role_context_id,
                        "calibration_fold_count": inner_fold_count,
                        "calibration_role_source": calibration_source,
                    }
                    context["context_id"] = _stable_id("P13CTX", context)
                    contexts.append(context)
    context_registry = pd.DataFrame(contexts)
    role_registry = pd.DataFrame(roles).drop_duplicates().sort_values(
        ["role_context_id", "role_name"], kind="stable"
    )
    role_lookup = _role_map(role_registry)

    procedures: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    for context in context_registry.itertuples(index=False):
        p03_domain = f"{context.station}:{context.held_instrument}"
        for procedure_order, procedure_id in enumerate(PROCEDURE_ORDER):
            resolution = resolutions.get(
                (
                    p03_domain,
                    int(context.outer_repeat),
                    int(context.outer_fold),
                    procedure_id,
                )
            )
            if resolution is None and procedure_id != "C-SELECTED":
                family = PROCEDURE_FAMILIES[procedure_id]
                row = first_family_candidate[family]
                resolution = {
                    **candidates[str(row.candidate_id)],
                    "selection_source": "exploratory_first_declared_family_candidate",
                    "selection_state_sha256": sha256_value(
                        {
                            "rule": "first_declared_family_candidate",
                            "model_id": family,
                            "candidate_id": str(row.candidate_id),
                        }
                    ),
                }
            planned_status = "planned" if resolution is not None else "unavailable"
            reason_code = (
                None
                if resolution is not None
                else "no_frozen_source_only_selection_for_exploratory_domain"
            )
            procedure_record = {
                "context_id": str(context.context_id),
                "procedure_id": procedure_id,
                "procedure_order": procedure_order,
                "planned_status": planned_status,
                "reason_code": reason_code,
                "candidate_id": None,
                "model_id": None,
                "parameters_json": None,
                "hyperparameter_sha256": None,
                "selection_source": None,
                "selection_state_sha256": None,
                "technical_seeds": None,
                "seed_count": 0,
            }
            if resolution is not None:
                procedure_record.update(
                    {
                        field: resolution[field]
                        for field in (
                            "candidate_id",
                            "model_id",
                            "parameters_json",
                            "hyperparameter_sha256",
                            "selection_source",
                            "selection_state_sha256",
                            "technical_seeds",
                            "seed_count",
                        )
                    }
                )
            procedures.append(procedure_record)
            endpoint = {
                "endpoint_id": _stable_id(
                    "P13END",
                    {
                        "domain_id": str(context.domain_id),
                        "policy_id": str(context.policy_id),
                        "procedure_id": procedure_id,
                        "outer_repeat": int(context.outer_repeat),
                    },
                ),
                "domain_id": str(context.domain_id),
                "station": str(context.station),
                "substrate_family": str(context.substrate_family),
                "held_instrument": str(context.held_instrument),
                "support_tier": str(context.support_tier),
                "policy_id": str(context.policy_id),
                "procedure_id": procedure_id,
                "outer_repeat": int(context.outer_repeat),
                "expected_fold_count": 4,
                "planned_status": planned_status,
                "reason_code": reason_code,
            }
            endpoints.append(endpoint)
            if resolution is None:
                continue
            for seed in _seeds(resolution):
                for inner_fold in range(int(context.calibration_fold_count)):
                    fit_role = role_lookup[
                        (str(context.role_context_id), f"calibration_fit_{inner_fold}")
                    ]
                    validation_role = role_lookup[
                        (
                            str(context.role_context_id),
                            f"calibration_validation_{inner_fold}",
                        )
                    ]
                    fit_record = {
                        "context_id": str(context.context_id),
                        "domain_id": str(context.domain_id),
                        "policy_id": str(context.policy_id),
                        "representation_id": str(context.representation_id),
                        "procedure_id": procedure_id,
                        "candidate_id": str(resolution["candidate_id"]),
                        "model_id": str(resolution["model_id"]),
                        "hyperparameter_sha256": str(
                            resolution["hyperparameter_sha256"]
                        ),
                        "stage": "calibration_crossfit",
                        "inner_fold": inner_fold,
                        "seed": seed,
                        "fit_uid_sha256": str(fit_role.observation_uid_sha256),
                        "validation_uid_sha256": str(
                            validation_role.observation_uid_sha256
                        ),
                        "test_uid_sha256": None,
                    }
                    fit_record["fit_id"] = _stable_id("P13FIT", fit_record)
                    fits.append(fit_record)
                outer_fit = role_lookup[
                    (str(context.role_context_id), "outer_source_fit")
                ]
                held_test = role_lookup[
                    (str(context.role_context_id), "outer_held_test")
                ]
                source_test = role_lookup[
                    (str(context.role_context_id), "outer_matched_source_test")
                ]
                final_record = {
                    "context_id": str(context.context_id),
                    "domain_id": str(context.domain_id),
                    "policy_id": str(context.policy_id),
                    "representation_id": str(context.representation_id),
                    "procedure_id": procedure_id,
                    "candidate_id": str(resolution["candidate_id"]),
                    "model_id": str(resolution["model_id"]),
                    "hyperparameter_sha256": str(resolution["hyperparameter_sha256"]),
                    "stage": "outer_final",
                    "inner_fold": -1,
                    "seed": seed,
                    "fit_uid_sha256": str(outer_fit.observation_uid_sha256),
                    "validation_uid_sha256": None,
                    "test_uid_sha256": sha256_value(
                        {
                            "held": str(held_test.observation_uid_sha256),
                            "matched_source": str(source_test.observation_uid_sha256),
                        }
                    ),
                }
                final_record["fit_id"] = _stable_id("P13FIT", final_record)
                fits.append(final_record)

    procedure_registry = pd.DataFrame(procedures).sort_values(
        ["context_id", "procedure_order"], kind="stable"
    )
    fit_manifest = pd.DataFrame(fits).sort_values(
        ["domain_id", "policy_id", "context_id", "procedure_id", "stage", "seed"],
        kind="stable",
    )
    expected_endpoint_registry = pd.DataFrame(endpoints).drop_duplicates().sort_values(
        ["domain_id", "policy_id", "procedure_id", "outer_repeat"],
        kind="stable",
    )
    shards: list[dict[str, Any]] = []
    for keys, group in context_registry.groupby(
        ["domain_id", "policy_id", "outer_repeat"], sort=True
    ):
        domain_id, policy_id, repeat = keys
        context_ids = set(group.context_id.astype(str))
        shard_fits = fit_manifest[fit_manifest.context_id.astype(str).isin(context_ids)]
        shard_endpoints = expected_endpoint_registry[
            expected_endpoint_registry.domain_id.astype(str).eq(str(domain_id))
            & expected_endpoint_registry.policy_id.astype(str).eq(str(policy_id))
            & expected_endpoint_registry.outer_repeat.eq(int(repeat))
        ]
        record = {
            "domain_id": str(domain_id),
            "policy_id": str(policy_id),
            "outer_repeat": int(repeat),
            "context_count": len(group),
            "fit_count": len(shard_fits),
            "endpoint_count": len(shard_endpoints),
            "context_id_sha256": _uid_hash(group.context_id.astype(str).tolist()),
            "fit_id_sha256": _uid_hash(shard_fits.fit_id.astype(str).tolist()),
            "status": "planned",
        }
        record["shard_id"] = _stable_id("P13SHARD", record)
        shards.append(record)
    shard_manifest = pd.DataFrame(shards).sort_values(
        ["domain_id", "policy_id", "outer_repeat"], kind="stable"
    )
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    validation = validate_p13_plan(
        manifest=manifest,
        context_registry=context_registry,
        role_registry=role_registry,
        procedure_registry=procedure_registry,
        fit_manifest=fit_manifest,
        expected_endpoint_registry=expected_endpoint_registry,
        shard_manifest=shard_manifest,
    )
    return P13PlanTables(
        context_registry=context_registry.reset_index(drop=True),
        role_registry=role_registry.reset_index(drop=True),
        procedure_registry=procedure_registry.reset_index(drop=True),
        fit_manifest=fit_manifest.reset_index(drop=True),
        expected_endpoint_registry=expected_endpoint_registry.reset_index(drop=True),
        shard_manifest=shard_manifest.reset_index(drop=True),
        input_hashes=input_hashes,
        validation_report=validation,
    )


def validate_p13_plan(
    *,
    manifest: pd.DataFrame,
    context_registry: pd.DataFrame,
    role_registry: pd.DataFrame,
    procedure_registry: pd.DataFrame,
    fit_manifest: pd.DataFrame,
    expected_endpoint_registry: pd.DataFrame,
    shard_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed on leakage, incompleteness, or unexpected plan cardinality."""

    checks: dict[str, bool] = {}
    checks["context_count"] = len(context_registry) == 16 * 5 * 4 * 3
    checks["domain_tiers"] = (
        context_registry.groupby("support_tier").domain_id.nunique().to_dict()
        == {
            "confirmatory": 13,
            "exploratory_low_support": 3,
        }
    )
    checks["procedure_count"] = len(procedure_registry) == len(context_registry) * 7
    checks["endpoint_count"] = len(expected_endpoint_registry) == 16 * 3 * 7 * 5
    checks["shard_count"] = len(shard_manifest) == 16 * 3 * 5
    checks["unique_ids"] = bool(
        context_registry.context_id.is_unique
        and fit_manifest.fit_id.is_unique
        and expected_endpoint_registry.endpoint_id.is_unique
        and shard_manifest.shard_id.is_unique
    )
    unavailable = procedure_registry[procedure_registry.planned_status.eq("unavailable")]
    checks["declared_exploratory_unavailability"] = bool(
        len(unavailable) == 3 * 5 * 4
        and set(unavailable.procedure_id) == {"C-SELECTED"}
        and set(
            context_registry[
                context_registry.context_id.isin(unavailable.context_id)
            ].held_instrument
        )
        == {"Agilent-3"}
    )
    unavailable_endpoints = expected_endpoint_registry[
        expected_endpoint_registry.planned_status.eq("unavailable")
    ]
    checks["unavailable_endpoints_retained"] = bool(
        len(unavailable_endpoints) == 3 * 5
        and set(unavailable_endpoints.procedure_id) == {"C-SELECTED"}
    )
    metadata = manifest.set_index(manifest.observation_uid.astype(str), drop=False)
    role_rows = {
        (str(row.role_context_id), str(row.role_name)): row
        for row in role_registry.itertuples(index=False)
    }
    leakage_free = True
    calibration_leakage_free = True
    role_complete = True
    for context in context_registry.itertuples(index=False):
        if context.policy_id != "PP-U-MIN":
            continue
        names = [
            "outer_source_fit",
            "outer_held_test",
            "outer_matched_held_test",
            "outer_matched_source_test",
        ]
        records = [role_rows[(str(context.role_context_id), name)] for name in names]
        frames = []
        for record in records:
            uids = json.loads(str(record.observation_uids_json))
            frame = metadata.loc[uids]
            frames.append(frame)
            role_complete &= bool(
                len(frame) == int(record.observation_count)
                and _uid_hash(uids) == str(record.observation_uid_sha256)
            )
        source_fit, held, matched_held, matched_source = frames
        leakage_free &= bool(
            set(source_fit.master_sample_id).isdisjoint(set(held.master_sample_id))
            and source_fit.instrument.astype(str).ne(str(context.held_instrument)).all()
            and held.instrument.astype(str).eq(str(context.held_instrument)).all()
            and matched_source.instrument.astype(str).ne(str(context.held_instrument)).all()
            and set(matched_held.master_sample_id) == set(matched_source.master_sample_id)
        )
        calibration_validation_masters: set[str] = set()
        held_masters = set(held.master_sample_id.astype(str))
        for inner_fold in range(int(context.calibration_fold_count)):
            inner_fit_record = role_rows[
                (str(context.role_context_id), f"calibration_fit_{inner_fold}")
            ]
            inner_validation_record = role_rows[
                (
                    str(context.role_context_id),
                    f"calibration_validation_{inner_fold}",
                )
            ]
            inner_fit_uids = json.loads(str(inner_fit_record.observation_uids_json))
            inner_validation_uids = json.loads(
                str(inner_validation_record.observation_uids_json)
            )
            inner_fit = metadata.loc[inner_fit_uids]
            inner_validation = metadata.loc[inner_validation_uids]
            validation_masters = set(inner_validation.master_sample_id.astype(str))
            calibration_validation_masters |= validation_masters
            calibration_leakage_free &= bool(
                set(inner_fit.master_sample_id.astype(str)).isdisjoint(
                    validation_masters
                )
                and set(inner_fit.master_sample_id.astype(str)).isdisjoint(held_masters)
                and validation_masters.isdisjoint(held_masters)
                and inner_fit.instrument.astype(str)
                .ne(str(context.held_instrument))
                .all()
                and inner_validation.instrument.astype(str)
                .ne(str(context.held_instrument))
                .all()
            )
        calibration_leakage_free &= calibration_validation_masters == set(
            source_fit.master_sample_id.astype(str)
        )
    checks["role_hash_and_membership_complete"] = role_complete
    checks["outer_master_and_instrument_leakage_free"] = leakage_free
    checks["calibration_master_and_instrument_leakage_free"] = (
        calibration_leakage_free
    )
    checks["fit_manifest_nonempty"] = not fit_manifest.empty
    checks["every_fit_maps_to_context"] = set(fit_manifest.context_id) <= set(
        context_registry.context_id
    )
    checks["all_shards_have_four_contexts"] = shard_manifest.context_count.eq(4).all()
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": "nato-sers-p13-plan-validation-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "checks": checks,
        "counts": {
            "contexts": len(context_registry),
            "roles": len(role_registry),
            "procedures": len(procedure_registry),
            "fits": len(fit_manifest),
            "endpoints": len(expected_endpoint_registry),
            "shards": len(shard_manifest),
            "unavailable_procedures": len(unavailable),
        },
    }
