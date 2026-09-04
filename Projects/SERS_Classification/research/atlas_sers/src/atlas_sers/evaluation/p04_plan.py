"""Outcome-blind expansion of all compact-D0 development and T3 work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import sha256_value


@dataclass(frozen=True)
class P04PlanTables:
    candidate_registry: pd.DataFrame
    context_registry: pd.DataFrame
    role_registry: pd.DataFrame
    fit_manifest: pd.DataFrame
    expected_endpoint_registry: pd.DataFrame
    shard_manifest: pd.DataFrame
    validation_report: dict[str, Any]


def _set_hash(values: pd.Series) -> str:
    return sha256_value(sorted(values.astype(str).tolist()))


def _candidate_registry(contract: dict[str, Any]) -> pd.DataFrame:
    declared = contract["optimization"]["candidate_order"]
    rows: list[dict[str, Any]] = []
    order = 0
    for learning_rate in contract["optimization"]["learning_rates"]:
        for weight_decay in contract["optimization"]["weight_decays"]:
            learning_rate_text = f"{learning_rate:.5f}".rstrip("0")
            weight_decay_text = f"{weight_decay:.5f}".rstrip("0")
            candidate_id = f"LR{learning_rate_text}_WD{weight_decay_text}"
            if candidate_id not in declared:
                raise ValueError(f"Deep candidate {candidate_id} is absent from declared order.")
            specification = {
                "optimizer": "AdamW",
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "batch_size": int(contract["optimization"]["batch_size"]),
            }
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": "D0-ERM",
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "batch_size": int(contract["optimization"]["batch_size"]),
                    "complexity_rank": 0,
                    "declared_candidate_order": order,
                    "hyperparameter_sha256": sha256_value(specification),
                }
            )
            order += 1
    frame = pd.DataFrame(rows)
    if frame.candidate_id.tolist() != declared:
        raise ValueError("Generated deep candidate order differs from the locked contract.")
    return frame


def _context_id(values: dict[str, Any]) -> str:
    identity = {
        key: values[key]
        for key in (
            "experiment_id",
            "domain",
            "station",
            "held_instrument",
            "outer_repeat",
            "outer_fold",
        )
    }
    return f"P04CTX-{sha256_value(identity)[:24]}"


def _append_roles(
    rows: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    role_name: str,
    unit_id: str,
    frame: pd.DataFrame,
) -> str:
    role_identity = {
        "context_id": context["context_id"],
        "role": role_name,
        "unit": unit_id,
    }
    role_id = f"P04ROLE-{sha256_value(role_identity)[:24]}"
    for record in frame[
        ["observation_uid", "master_sample_id", "target_analyte", "instrument"]
    ].itertuples(index=False):
        rows.append(
            {
                "context_id": context["context_id"],
                "role_id": role_id,
                "role": role_name,
                "selection_unit_id": unit_id,
                "observation_uid": str(record.observation_uid),
                "master_sample_id": str(record.master_sample_id),
                "target_analyte": str(record.target_analyte),
                "instrument": str(record.instrument),
            }
        )
    return role_id


def _t1_contexts(
    manifest: pd.DataFrame, master_splits: pd.DataFrame
) -> list[
    tuple[
        dict[str, Any],
        pd.DataFrame,
        pd.DataFrame,
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ]
]:
    records = []
    for repeat in sorted(master_splits.outer_repeat.unique()):
        assignments = master_splits[master_splits.outer_repeat.eq(repeat)]
        for station in sorted(manifest.station.unique()):
            station_rows = manifest[manifest.station.eq(station)]
            station_assignments = assignments[assignments.station.eq(station)]
            mapping = dict(
                zip(
                    station_assignments.master_sample_id.astype(str),
                    station_assignments.outer_fold.astype(int),
                    strict=True,
                )
            )
            assigned = station_rows.master_sample_id.astype(str).map(mapping)
            if assigned.isna().any():
                raise ValueError("A T1 row lacks its frozen master-fold assignment.")
            for outer_fold in range(4):
                test = station_rows[assigned.eq(outer_fold)]
                train = station_rows[~assigned.eq(outer_fold)]
                units = []
                for inner_fold in sorted(set(range(4)) - {outer_fold}):
                    validation = train[assigned.loc[train.index].eq(inner_fold)]
                    fitting = train[~assigned.loc[train.index].eq(inner_fold)]
                    units.append((f"outer_fold_as_inner:{inner_fold}", fitting, validation))
                context = {
                    "experiment_id": "EXP-N00-DEV",
                    "task_id": f"T1-{station.upper() if station != 'surfaces' else 'SURF'}",
                    "domain": f"{station}:within",
                    "station": station,
                    "held_instrument": "not_applicable",
                    "outer_repeat": int(repeat),
                    "outer_fold": outer_fold,
                    "selection_mode": "inner_master_cv",
                    "phase_gate": "development",
                }
                context["context_id"] = _context_id(context)
                records.append((context, train, test, units))
    return records


def _t3_contexts(
    t3: pd.DataFrame,
    selection: pd.DataFrame,
    fallback: pd.DataFrame,
) -> list[
    tuple[
        dict[str, Any],
        pd.DataFrame,
        pd.DataFrame,
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ]
]:
    records = []
    primary = t3[t3.domain_scope.eq("primary")]
    for partition_id, cell in primary.groupby("partition_id", sort=True):
        first = cell.iloc[0]
        source = cell[cell.role.eq("train_source")]
        test = cell[cell.role.eq("test_target")]
        supported = selection[
            selection.partition_id.astype(str).eq(str(partition_id)) & selection.supported
        ]
        units: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        if len(supported) >= 2:
            for row in supported.itertuples(index=False):
                validation = source[source.instrument.eq(row.pseudo_instrument)]
                validation_masters = set(validation.master_sample_id.astype(str))
                fitting = source[
                    ~source.instrument.eq(row.pseudo_instrument)
                    & ~source.master_sample_id.astype(str).isin(validation_masters)
                ]
                if _set_hash(fitting.observation_uid) != row.fit_observation_set_sha256:
                    raise ValueError(
                        f"P04 reconstructed a different P02 fit set for {partition_id}."
                    )
                if _set_hash(validation.observation_uid) != row.validation_observation_set_sha256:
                    raise ValueError(
                        f"P04 reconstructed a different P02 validation set for {partition_id}."
                    )
                units.append((f"pseudo:{row.pseudo_instrument}", fitting, validation))
            selection_mode = "pseudo_domain"
        else:
            assignments = fallback[
                fallback.partition_id.astype(str).eq(str(partition_id))
            ]
            if assignments.empty or assignments.inner_fold.nunique() != 3:
                raise ValueError(f"P04 cannot reconstruct the P02 fallback for {partition_id}.")
            for inner_fold in sorted(assignments.inner_fold.unique()):
                validation_masters = set(
                    assignments.loc[
                        assignments.inner_fold.eq(inner_fold), "master_sample_id"
                    ].astype(str)
                )
                validation = source[
                    source.master_sample_id.astype(str).isin(validation_masters)
                ]
                fitting = source[
                    ~source.master_sample_id.astype(str).isin(validation_masters)
                ]
                units.append((f"master_cv:{int(inner_fold)}", fitting, validation))
            selection_mode = "master_cv"
        context = {
            "experiment_id": "EXP-N00-T3",
            "task_id": "T3-ZS",
            "domain": str(first.domain),
            "station": str(first.station),
            "held_instrument": str(first.held_instrument),
            "outer_repeat": int(first.outer_repeat),
            "outer_fold": int(first.outer_fold),
            "selection_mode": selection_mode,
            "phase_gate": "held_evaluation",
            "partition_id": str(partition_id),
        }
        context["context_id"] = _context_id(context)
        records.append((context, source, test, units))
    return records


def build_p04_plan(
    *,
    manifest: pd.DataFrame,
    master_splits: pd.DataFrame,
    t3_partitions: pd.DataFrame,
    inner_selection: pd.DataFrame,
    inner_fallback: pd.DataFrame,
    contract: dict[str, Any],
) -> P04PlanTables:
    candidates = _candidate_registry(contract)
    contexts: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    context_payloads = _t1_contexts(manifest, master_splits) + _t3_contexts(
        t3_partitions, inner_selection, inner_fallback
    )
    seeds = [int(value) for value in contract["optimization"]["training_seeds"]]
    for shard_index, (context, outer_fit, outer_test, units) in enumerate(context_payloads):
        outer_fit_masters = set(outer_fit.master_sample_id.astype(str))
        outer_test_masters = set(outer_test.master_sample_id.astype(str))
        if outer_fit_masters & outer_test_masters:
            raise ValueError("P04 outer fitting and test masters overlap.")
        if context["experiment_id"] == "EXP-N00-T3" and (
            outer_fit.instrument.astype(str).eq(context["held_instrument"]).any()
        ):
            raise ValueError("P04 T3 fitting rows contain the held instrument.")
        context = {
            **context,
            "shard_index": shard_index,
            "outer_fit_rows": len(outer_fit),
            "outer_fit_masters": len(outer_fit_masters),
            "outer_test_rows": len(outer_test),
            "outer_test_masters": len(outer_test_masters),
            "selection_unit_count": len(units),
            "outer_fit_uid_sha256": _set_hash(outer_fit.observation_uid),
            "outer_test_uid_sha256": _set_hash(outer_test.observation_uid),
        }
        contexts.append(context)
        outer_fit_role = _append_roles(
            roles,
            context=context,
            role_name="outer_fit",
            unit_id="outer_fit",
            frame=outer_fit,
        )
        outer_test_role = _append_roles(
            roles,
            context=context,
            role_name="outer_test",
            unit_id="outer_test",
            frame=outer_test,
        )
        for unit_id, fitting, validation in units:
            if set(fitting.master_sample_id.astype(str)) & set(
                validation.master_sample_id.astype(str)
            ):
                raise ValueError("P04 inner fitting and validation masters overlap.")
            fit_role = _append_roles(
                roles,
                context=context,
                role_name="selection_fit",
                unit_id=unit_id,
                frame=fitting,
            )
            validation_role = _append_roles(
                roles,
                context=context,
                role_name="selection_validation",
                unit_id=unit_id,
                frame=validation,
            )
            for candidate in candidates.itertuples(index=False):
                for seed in seeds:
                    identity = {
                        "context_id": context["context_id"],
                        "stage": "inner_selection",
                        "candidate_id": candidate.candidate_id,
                        "seed": seed,
                        "selection_unit_id": unit_id,
                    }
                    fits.append(
                        {
                            **identity,
                            "fit_id": f"P04FIT-{sha256_value(identity)[:24]}",
                            "experiment_id": context["experiment_id"],
                            "fit_role_id": fit_role,
                            "validation_role_id": validation_role,
                            "test_role_id": "not_accessed",
                            "learning_rate": candidate.learning_rate,
                            "weight_decay": candidate.weight_decay,
                            "batch_size": candidate.batch_size,
                            "hyperparameter_sha256": candidate.hyperparameter_sha256,
                            "execution_condition": "unconditional",
                        }
                    )
        for seed in seeds:
            identity = {
                "context_id": context["context_id"],
                "stage": "final_selected_refit",
                "candidate_id": "selected_after_inner",
                "seed": seed,
                "selection_unit_id": "outer_fit",
            }
            fits.append(
                {
                    **identity,
                    "fit_id": f"P04FIT-{sha256_value(identity)[:24]}",
                    "experiment_id": context["experiment_id"],
                    "fit_role_id": outer_fit_role,
                    "validation_role_id": "source_inner_evidence_only",
                    "test_role_id": outer_test_role,
                    "learning_rate": None,
                    "weight_decay": None,
                    "batch_size": int(contract["optimization"]["batch_size"]),
                    "hyperparameter_sha256": "selected_after_inner",
                    "execution_condition": "selected_candidate_has_complete_support",
                }
            )
        endpoints.append(
            {
                "context_id": context["context_id"],
                "experiment_id": context["experiment_id"],
                "task_id": context["task_id"],
                "domain": context["domain"],
                "station": context["station"],
                "held_instrument": context["held_instrument"],
                "outer_repeat": context["outer_repeat"],
                "outer_fold": context["outer_fold"],
                "expected_test_rows": len(outer_test),
                "expected_test_uid_sha256": _set_hash(outer_test.observation_uid),
                "expected_seed_count": len(seeds),
            }
        )
    context_frame = pd.DataFrame(contexts).sort_values("shard_index").reset_index(drop=True)
    role_frame = pd.DataFrame(roles).sort_values(
        ["context_id", "role_id", "observation_uid"], kind="stable"
    ).reset_index(drop=True)
    fit_frame = pd.DataFrame(fits).sort_values(
        ["context_id", "stage", "candidate_id", "selection_unit_id", "seed"],
        kind="stable",
    ).reset_index(drop=True)
    endpoint_frame = pd.DataFrame(endpoints).sort_values(
        ["experiment_id", "domain", "outer_repeat", "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    shard_frame = (
        fit_frame.groupby(["context_id", "experiment_id"], as_index=False)
        .size()
        .rename(columns={"size": "planned_fit_count"})
        .merge(
            context_frame[["context_id", "shard_index", "phase_gate"]],
            on="context_id",
            validate="one_to_one",
        )
        .sort_values("shard_index")
        .reset_index(drop=True)
    )
    checks = {
        "population_598": len(manifest) == 598,
        "masters_69": manifest.master_sample_id.nunique() == 69,
        "candidate_grid_six": len(candidates) == 6,
        "development_contexts_60": context_frame.experiment_id.eq("EXP-N00-DEV").sum()
        == 60,
        "held_contexts_260": context_frame.experiment_id.eq("EXP-N00-T3").sum() == 260,
        "all_context_ids_unique": context_frame.context_id.is_unique,
        "all_fit_ids_unique": fit_frame.fit_id.is_unique,
        "all_roles_nonempty": not role_frame.empty,
        "all_endpoints_have_rows": endpoint_frame.expected_test_rows.gt(0).all(),
        "all_t3_sources_exclude_held": True,
        "all_outer_master_sets_disjoint": True,
        "every_context_has_three_final_seeds": bool(
            fit_frame[fit_frame.stage.eq("final_selected_refit")]
            .groupby("context_id")
            .size()
            .eq(3)
            .all()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "contexts": len(context_frame),
            "development_contexts": int(
                context_frame.experiment_id.eq("EXP-N00-DEV").sum()
            ),
            "held_contexts": int(context_frame.experiment_id.eq("EXP-N00-T3").sum()),
            "role_rows": len(role_frame),
            "selection_fits": int(fit_frame.stage.eq("inner_selection").sum()),
            "conditional_final_fits": int(
                fit_frame.stage.eq("final_selected_refit").sum()
            ),
            "total_planned_fits": len(fit_frame),
            "development_planned_fits": int(
                fit_frame.experiment_id.eq("EXP-N00-DEV").sum()
            ),
            "held_planned_fits": int(fit_frame.experiment_id.eq("EXP-N00-T3").sum()),
            "shards": len(shard_frame),
        },
    }
    return P04PlanTables(
        candidates,
        context_frame,
        role_frame,
        fit_frame,
        endpoint_frame,
        shard_frame,
        report,
    )
