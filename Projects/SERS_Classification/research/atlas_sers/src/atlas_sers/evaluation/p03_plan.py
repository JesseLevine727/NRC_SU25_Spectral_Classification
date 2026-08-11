"""Outcome-blind expansion of the registered P03 classical benchmark."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import sha256_value

PRIMARY_T3_EXPERIMENTS = ("EXP-C09-T3", "EXP-C10-T3", "EXP-C12-CORAL")
T1_EXPERIMENT_MODELS = {
    "EXP-C00-T1": "C-PRIOR",
    "EXP-C01-T1": "C-SPECTRAL-MATCH",
    "EXP-C02-T1": "C-NEAREST-CENTROID",
    "EXP-C03-T1": "C-PCA-LDA",
    "EXP-C04-T1": "C-PLS-DA",
    "EXP-C05-T1": "C-LOGREG-EN",
    "EXP-C06-T1": "C-RBF-SVM",
    "EXP-C07-T1": "C-RANDOM-FOREST",
    "EXP-C08-T1": "C-EXTRA-TREES",
}
FIXED_SUITE = ("C-PCA-LDA", "C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
STOCHASTIC_MODELS = {"C-RANDOM-FOREST", "C-EXTRA-TREES"}
CONTROL_EXPERIMENTS = {
    "EXP-C09-CONTROL-PERM",
    "EXP-C09-CONTROL-META",
    "EXP-C09-CONTROL-PRIOR",
}
SELECTION_FIT_STAGES = {
    "inner_selection",
    "training_station_inner_selection",
    "inner_source_coral_selection",
    "metadata_inner_selection",
}
SELECTION_KIND_BY_STAGE = {
    "inner_selection": "standard",
    "training_station_inner_selection": "standard",
    "inner_source_coral_selection": "source_covariance",
    "metadata_inner_selection": "metadata_control",
}


def _uid_hash(values: Iterable[object]) -> str:
    return sha256_value(sorted(str(value) for value in values))


def _canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _product(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(parameters[k] for k in keys))
    ]


def build_candidate_registry(
    hyperparameters: dict[str, Any], p03_contract: dict[str, Any]
) -> pd.DataFrame:
    """Expand the frozen classical grids in stable declared order."""

    hp = hyperparameters["classical"]
    specs: dict[str, list[dict[str, Any]]] = {
        "C-PRIOR": _product(p03_contract["non_registry_candidates"]["C-PRIOR"]),
        "C-SPECTRAL-MATCH": _product(p03_contract["non_registry_candidates"]["C-SPECTRAL-MATCH"]),
        "C-NEAREST-CENTROID": _product(hp["nearest_centroid"]),
        "C-PCA-LDA": _product(
            {"pca_components": hp["pca_lda"]["pca_components"], "lda": hp["pca_lda"]["lda"]}
        ),
        "C-PLS-DA": _product(
            {"components": hp["pls_da"]["components"], "head": [hp["pls_da"]["head"]]}
        ),
        "C-LOGREG-EN": _product(hp["elastic_net_logistic"]),
        "C-RBF-SVM": _product(
            {
                "C": hp["rbf_svm"]["C"],
                "gamma": hp["rbf_svm"]["gamma"],
                "class_weight": [hp["rbf_svm"]["class_weight"]],
            }
        ),
        "C-RANDOM-FOREST": _product(
            {
                "n_estimators": [hp["random_forest"]["n_estimators"]],
                "max_features": hp["random_forest"]["max_features"],
                "min_samples_leaf": hp["random_forest"]["min_samples_leaf"],
                "class_weight": [hp["random_forest"]["class_weight"]],
            }
        ),
        "C-EXTRA-TREES": _product(
            {
                "n_estimators": [hp["extra_trees"]["n_estimators"]],
                "max_features": hp["extra_trees"]["max_features"],
                "min_samples_leaf": hp["extra_trees"]["min_samples_leaf"],
                "class_weight": [hp["extra_trees"]["class_weight"]],
                "bootstrap": [hp["extra_trees"]["bootstrap"]],
            }
        ),
    }
    rows: list[dict[str, Any]] = []
    declared = 0
    seeds = p03_contract["stochastic_seeds"]
    for family_order, model_id in enumerate(p03_contract["candidate_order"]):
        for family_candidate_order, parameters in enumerate(specs[model_id]):
            parameters_json = _canonical_parameters(parameters)
            rows.append(
                {
                    "candidate_id": f"{model_id}-{family_candidate_order:03d}",
                    "model_id": model_id,
                    "family_order": family_order,
                    "family_candidate_order": family_candidate_order,
                    "declared_candidate_order": declared,
                    "parameters_json": parameters_json,
                    "hyperparameter_sha256": sha256_value(parameters),
                    "complexity_rank": family_candidate_order,
                    "stochastic": model_id in STOCHASTIC_MODELS,
                    "technical_seeds": "|".join(str(seed) for seed in seeds)
                    if model_id in STOCHASTIC_MODELS
                    else "deterministic",
                    "seed_count": len(seeds) if model_id in STOCHASTIC_MODELS else 1,
                }
            )
            declared += 1
    return pd.DataFrame(rows)


def build_coral_candidate_registry(
    candidate_registry: pd.DataFrame, p03_contract: dict[str, Any]
) -> pd.DataFrame:
    """Derive the fixed C12 base candidates plus source-only method state."""

    method = p03_contract["coral"]["proposed_method"]
    allowed = set(str(value) for value in method["base_models"])
    base = candidate_registry[candidate_registry.model_id.isin(allowed)].copy()
    rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(base.itertuples(index=False)):
        parameters = {
            "base_model_id": str(candidate.model_id),
            "base_parameters": json.loads(str(candidate.parameters_json)),
            "rank_cap": int(method["rank_cap"]),
            "ridge_fraction": float(
                method["ridge_fraction_of_mean_feature_variance"]
            ),
        }
        rows.append(
            {
                "candidate_id": f"CORAL-{candidate.candidate_id}",
                "model_id": "C-SOURCE-CORAL",
                "base_candidate_id": candidate.candidate_id,
                "base_model_id": candidate.model_id,
                "family_order": candidate.family_order,
                "family_candidate_order": candidate.family_candidate_order,
                "declared_candidate_order": order,
                "parameters_json": _canonical_parameters(parameters),
                "hyperparameter_sha256": sha256_value(parameters),
                "complexity_rank": candidate.complexity_rank,
                "stochastic": False,
                "technical_seeds": "deterministic",
                "seed_count": 1,
                "method_id": method["method_id"],
                "method_status": p03_contract["coral"]["status"],
            }
        )
    if set(base.model_id) != allowed:
        raise ValueError("C12 base-model registry is incomplete.")
    return pd.DataFrame(rows)


def build_control_registry(
    candidate_registry: pd.DataFrame, p03_contract: dict[str, Any]
) -> pd.DataFrame:
    """Expand the proposed controls without using a predictive outcome."""

    controls = p03_contract["negative_controls"]
    design = controls["proposed_design"]
    rows: list[dict[str, Any]] = []
    declared = 0
    permutation = design["permutation"]
    if len(permutation["seeds"]) != int(permutation["count"]):
        raise ValueError("The proposed permutation count and seed registry disagree.")
    if len(set(int(seed) for seed in permutation["seeds"])) != int(
        permutation["count"]
    ):
        raise ValueError("The proposed permutation seeds are not unique.")
    for replicate, seed in enumerate(permutation["seeds"]):
        parameters = {
            "permutation_seed": int(seed),
            "unit": permutation["unit"],
            "exchangeability_block": permutation["exchangeability_block"],
            "candidate_policy": permutation["candidate_policy"],
            "technical_seed_policy": permutation["technical_seed_policy"],
        }
        rows.append(
            {
                "control_candidate_id": f"CTRL-PERM-{replicate:02d}",
                "control_type": "master_label_permutation",
                "model_id": "C-PERMUTED-SELECTED",
                "base_candidate_id": "frozen_real_C09_selected",
                "control_replicate": replicate,
                "declared_control_order": declared,
                "declared_candidate_order": replicate,
                "complexity_rank": 0,
                "seed_count": 1,
                "parameters_json": _canonical_parameters(parameters),
                "configuration_sha256": sha256_value(parameters),
                "method_id": design["method_id"],
                "method_status": controls["status"],
            }
        )
        declared += 1
    metadata = design["metadata_only"]
    metadata_candidates = candidate_registry[
        candidate_registry.model_id == metadata["base_model_id"]
    ]
    if len(metadata_candidates) != 30:
        raise ValueError("Metadata-only control must inherit exactly 30 logistic candidates.")
    for candidate in metadata_candidates.itertuples(index=False):
        parameters = {
            "base_parameters": json.loads(str(candidate.parameters_json)),
            "categorical_features": metadata["categorical_features"],
            "numeric_features": metadata["numeric_features"],
            "source_fitted_operations": metadata["source_fitted_operations"],
        }
        rows.append(
            {
                "control_candidate_id": f"CTRL-META-{candidate.candidate_id}",
                "control_type": "acquisition_metadata_only",
                "model_id": metadata["model_id"],
                "base_candidate_id": candidate.candidate_id,
                "control_replicate": "not_applicable",
                "declared_control_order": declared,
                "declared_candidate_order": candidate.family_candidate_order,
                "complexity_rank": candidate.complexity_rank,
                "seed_count": 1,
                "parameters_json": _canonical_parameters(parameters),
                "configuration_sha256": sha256_value(parameters),
                "method_id": design["method_id"],
                "method_status": controls["status"],
            }
        )
        declared += 1
    for prior in design["priors"]["candidates"]:
        parameters = {"prior": str(prior)}
        rows.append(
            {
                "control_candidate_id": f"CTRL-PRIOR-{str(prior).upper()}",
                "control_type": "station_or_target_prior",
                "model_id": "C-PRIOR",
                "base_candidate_id": f"C-PRIOR-{str(prior)}",
                "control_replicate": "not_applicable",
                "declared_control_order": declared,
                "declared_candidate_order": declared,
                "complexity_rank": declared,
                "seed_count": 1,
                "parameters_json": _canonical_parameters(parameters),
                "configuration_sha256": sha256_value(parameters),
                "method_id": design["method_id"],
                "method_status": controls["status"],
            }
        )
        declared += 1
    frame = pd.DataFrame(rows)
    if len(frame) != 52 or not frame.control_candidate_id.is_unique:
        raise ValueError("The proposed P03 control registry is incomplete.")
    return frame


def _role_summary(frame: pd.DataFrame) -> dict[str, Any]:
    cached = frame.attrs.get("p03_role_summary")
    manager_id = id(frame._mgr)  # noqa: SLF001 - object-local cache guard
    if cached is not None and cached.get("manager_id") == manager_id:
        return cached
    uids = frozenset(str(value) for value in frame.observation_uid)
    masters = frozenset(str(value) for value in frame.master_sample_id)
    summary = {
        "rows": len(frame),
        "masters": len(masters),
        "uid_sha256": sha256_value(sorted(uids)),
        "uid_set": uids,
        "master_set": masters,
        "manager_id": manager_id,
    }
    frame.attrs["p03_role_summary"] = summary
    return summary


def _outer_run_id(record: dict[str, Any]) -> str:
    identity = {key: record[key] for key in sorted(record) if key != "outer_run_id"}
    return f"P03OUTER-{sha256_value(identity)[:24]}"


def _fit_id(record: dict[str, Any]) -> str:
    identity = {key: record[key] for key in sorted(record) if key != "fit_id"}
    return f"P03FIT-{sha256_value(identity)[:24]}"


def _fit_record(
    *,
    outer: dict[str, Any],
    stage: str,
    model_id: str,
    candidate_id: str,
    hyperparameter_sha256: str,
    seed: int | str,
    unit_id: str,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    accounting: str = "new_fit",
    condition: str = "always",
) -> dict[str, Any]:
    fit_summary = _role_summary(fit)
    validation_summary = _role_summary(validation)
    test_summary = _role_summary(test)
    record = {
        "experiment_id": outer["experiment_id"],
        "task_id": outer["task_id"],
        "outer_run_id": outer["outer_run_id"],
        "domain": outer["domain"],
        "station": outer["station"],
        "held_instrument": outer["held_instrument"],
        "outer_repeat": outer["outer_repeat"],
        "outer_fold": outer["outer_fold"],
        "selection_mode": outer["selection_mode"],
        "stage": stage,
        "selection_unit_id": unit_id,
        "model_id": model_id,
        "candidate_id": candidate_id,
        "hyperparameter_sha256": hyperparameter_sha256,
        "seed": seed,
        "fit_rows": fit_summary["rows"],
        "fit_masters": fit_summary["masters"],
        "fit_uid_sha256": fit_summary["uid_sha256"],
        "validation_rows": validation_summary["rows"],
        "validation_masters": validation_summary["masters"],
        "validation_uid_sha256": validation_summary["uid_sha256"],
        "test_rows": test_summary["rows"],
        "test_masters": test_summary["masters"],
        "test_uid_sha256": test_summary["uid_sha256"],
        "fit_test_disjoint": not bool(fit_summary["uid_set"] & test_summary["uid_set"]),
        "fit_validation_master_disjoint": not bool(
            fit_summary["master_set"] & validation_summary["master_set"]
        ),
        "accounting": accounting,
        "condition": condition,
    }
    record["fit_id"] = _fit_id(record)
    return record


def _control_fit_record(
    *,
    control_type: str,
    control_candidate_id: str,
    control_replicate: int | str,
    source_outer_run_id: str,
    label_policy: str,
    feature_space: str,
    **fit_arguments: Any,
) -> dict[str, Any]:
    record = _fit_record(**fit_arguments)
    record.update(
        {
            "control_type": control_type,
            "control_candidate_id": control_candidate_id,
            "control_replicate": control_replicate,
            "source_outer_run_id": source_outer_run_id,
            "label_policy": label_policy,
            "feature_space": feature_space,
        }
    )
    return record


def _candidate_seeds(candidate: pd.Series, p03_contract: dict[str, Any]) -> list[int | str]:
    return list(p03_contract["stochastic_seeds"]) if candidate.stochastic else ["deterministic"]


def _conditional_selected_seeds(
    p03_contract: dict[str, Any],
) -> list[tuple[int | str, str]]:
    return [
        ("deterministic", "selected_model_is_deterministic"),
        *[
            (int(seed), "selected_model_is_stochastic")
            for seed in p03_contract["stochastic_seeds"]
        ],
    ]


def assign_selection_shards(fits: pd.DataFrame, *, target: int) -> pd.DataFrame:
    """Assign bounded shards without mixing incompatible candidate registries."""

    if target < 1:
        raise ValueError("Selection shard target must be positive.")
    selected = fits[fits.stage.isin(SELECTION_FIT_STAGES)].copy()
    selected["selection_kind"] = selected.stage.map(SELECTION_KIND_BY_STAGE)
    if selected.selection_kind.isna().any():
        raise RuntimeError("A P03 selection stage has no execution kind.")
    frames: list[pd.DataFrame] = []
    offset = 0
    for (kind, stage), group in selected.groupby(
        ["selection_kind", "stage"], sort=True
    ):
        frame = group.copy().reset_index(drop=True)
        frame["selection_kind"] = kind
        frame["stage"] = stage
        frame["selection_shard_id"] = frame.index // target + offset
        offset = int(frame.selection_shard_id.max()) + 1
        frames.append(frame)
    if not frames:
        return selected.assign(selection_shard_id=pd.Series(dtype=int))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["selection_shard_id", "fit_id"], kind="stable"
    ).reset_index(drop=True)


def _t1_cells(
    manifest: pd.DataFrame, masters: pd.DataFrame
) -> Iterable[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    assignments = masters.set_index(["outer_repeat", "station", "master_sample_id"])["outer_fold"]
    for experiment_id, model_id in T1_EXPERIMENT_MODELS.items():
        for (repeat, station), station_masters in masters.groupby(
            ["outer_repeat", "station"], sort=True
        ):
            station_rows = manifest[manifest.station == station]
            for fold in sorted(station_masters.outer_fold.unique()):
                test_master_ids = set(
                    station_masters.loc[station_masters.outer_fold == fold, "master_sample_id"]
                )
                test = station_rows[station_rows.master_sample_id.isin(test_master_ids)]
                train = station_rows[~station_rows.master_sample_id.isin(test_master_ids)]
                outer = {
                    "experiment_id": experiment_id,
                    "task_id": f"T1-{str(station).upper() if station != 'surfaces' else 'SURF'}",
                    "scope": "P",
                    "domain": f"{station}:within",
                    "station": station,
                    "held_instrument": "not_applicable",
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "selection_mode": "inner_master_cv",
                    "model_id": model_id,
                    "execution_status": "planned",
                }
                outer["outer_run_id"] = _outer_run_id(outer)
                yield outer, train, test, assignments


def _selection_units(
    cell: pd.DataFrame,
    selection: pd.DataFrame,
    fallback: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    partition_id = str(cell.partition_id.iloc[0])
    source = cell[cell.role == "train_source"]
    supported = selection[(selection.partition_id == partition_id) & selection.supported]
    units: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    if len(supported) >= 2:
        for row in supported.itertuples(index=False):
            validation = source[source.instrument == row.pseudo_instrument]
            validation_masters = set(validation.master_sample_id)
            fit = source[
                (source.instrument != row.pseudo_instrument)
                & (~source.master_sample_id.isin(validation_masters))
            ]
            if _uid_hash(fit.observation_uid) != row.fit_observation_set_sha256:
                raise ValueError(f"P02 pseudo-domain fit hash mismatch for {partition_id}")
            if _uid_hash(validation.observation_uid) != row.validation_observation_set_sha256:
                raise ValueError(f"P02 pseudo-domain validation hash mismatch for {partition_id}")
            units.append((f"pseudo:{row.pseudo_instrument}", fit, validation))
        return units
    assignments = fallback[fallback.partition_id == partition_id]
    if assignments.empty or assignments.inner_fold.nunique() != 3:
        raise ValueError(f"P02 master-CV fallback is incomplete for {partition_id}")
    for inner_fold in sorted(assignments.inner_fold.unique()):
        validation_masters = set(
            assignments.loc[assignments.inner_fold == inner_fold, "master_sample_id"]
        )
        validation = source[source.master_sample_id.isin(validation_masters)]
        fit = source[~source.master_sample_id.isin(validation_masters)]
        units.append((f"master_cv:{inner_fold}", fit, validation))
    return units


def _registered_master_cv_units(
    source: pd.DataFrame, fallback: pd.DataFrame, partition_id: str
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    assignments = fallback[fallback.partition_id == partition_id]
    if assignments.empty or assignments.inner_fold.nunique() != 3:
        raise ValueError(f"P02 calibration master folds are incomplete for {partition_id}")
    units: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for inner_fold in sorted(assignments.inner_fold.unique()):
        validation_masters = set(
            assignments.loc[assignments.inner_fold == inner_fold, "master_sample_id"]
        )
        validation = source[source.master_sample_id.isin(validation_masters)]
        fit = source[~source.master_sample_id.isin(validation_masters)]
        units.append((f"calibration_master_cv:{inner_fold}", fit, validation))
    return units


def build_p03_plan_tables(
    *,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    hyperparameters: dict[str, Any],
    p03_contract: dict[str, Any],
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Build the complete outcome-blind P03 run and fit expansion."""

    candidates = build_candidate_registry(hyperparameters, p03_contract)
    coral_candidates = build_coral_candidate_registry(candidates, p03_contract)
    control_candidates = build_control_registry(candidates, p03_contract)
    masters = p02_tables["master_split_registry.csv"]
    t3 = p02_tables["t3_partition_registry.csv"]
    selection = p02_tables["inner_selection_registry.csv"]
    fallback = p02_tables["inner_master_split_registry.csv"]
    domains = p02_tables["domain_registry.csv"]
    outer_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for outer, train, test, assignment_lookup in _t1_cells(manifest, masters):
        outer_rows.append(outer)
        family = candidates[candidates.model_id == outer["model_id"]]
        if outer["model_id"] == "C-PRIOR":
            for candidate in family.itertuples(index=False):
                fit_rows.append(
                    _fit_record(
                        outer=outer,
                        stage="final_fixed",
                        model_id=candidate.model_id,
                        candidate_id=candidate.candidate_id,
                        hyperparameter_sha256=candidate.hyperparameter_sha256,
                        seed="deterministic",
                        unit_id="outer_train",
                        fit=train,
                        validation=train.iloc[0:0],
                        test=test,
                    )
                )
            continue
        non_test_folds = sorted(set(range(4)) - {int(outer["outer_fold"])})
        inner_units: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
        for inner_fold in non_test_folds:
            validation_master_ids = {
                master
                for (repeat, station, master), assigned in assignment_lookup.items()
                if repeat == outer["outer_repeat"]
                and station == outer["station"]
                and assigned == inner_fold
            }
            validation = train[train.master_sample_id.isin(validation_master_ids)]
            fitting = train[~train.master_sample_id.isin(validation_master_ids)]
            inner_units.append((inner_fold, fitting, validation))
        for candidate in family.itertuples(index=False):
            for inner_fold, fitting, validation in inner_units:
                for seed in _candidate_seeds(pd.Series(candidate._asdict()), p03_contract):
                    fit_rows.append(
                        _fit_record(
                            outer=outer,
                            stage="inner_selection",
                            model_id=candidate.model_id,
                            candidate_id=candidate.candidate_id,
                            hyperparameter_sha256=candidate.hyperparameter_sha256,
                            seed=seed,
                            unit_id=f"outer_fold_as_inner:{inner_fold}",
                            fit=fitting,
                            validation=validation,
                            test=test,
                        )
                    )
        final_seed_count = (
            len(p03_contract["stochastic_seeds"]) if outer["model_id"] in STOCHASTIC_MODELS else 1
        )
        for seed_index in range(final_seed_count):
            seed = (
                p03_contract["stochastic_seeds"][seed_index]
                if final_seed_count > 1
                else "deterministic"
            )
            fit_rows.append(
                _fit_record(
                    outer=outer,
                    stage="final_selected_refit",
                    model_id=outer["model_id"],
                    candidate_id="selected_after_inner",
                    hyperparameter_sha256=sha256_value({"selection_dependent": True}),
                    seed=seed,
                    unit_id="outer_train",
                    fit=train,
                    validation=train.iloc[0:0],
                    test=test,
                    condition="selected_candidate_supports_fit",
                )
            )

    primary_t3 = t3[t3.domain_scope == "primary"]
    for partition_id, cell in primary_t3.groupby("partition_id", sort=True):
        first = cell.iloc[0]
        source = cell[cell.role == "train_source"]
        test = cell[cell.role == "test_target"]
        units = _selection_units(cell, selection, fallback)
        calibration_units = _registered_master_cv_units(source, fallback, str(partition_id))
        selection_mode = "pseudo_domain" if units[0][0].startswith("pseudo:") else "master_cv"
        base = {
            "task_id": "T3-ZS",
            "scope": "P",
            "domain": first.domain,
            "station": first.station,
            "held_instrument": first.held_instrument,
            "outer_repeat": int(first.outer_repeat),
            "outer_fold": int(first.outer_fold),
            "selection_mode": selection_mode,
            "execution_status": "planned",
        }
        c09 = {**base, "experiment_id": "EXP-C09-T3", "model_id": "C-SELECTED"}
        c09["outer_run_id"] = _outer_run_id(c09)
        outer_rows.append(c09)
        for candidate in candidates.itertuples(index=False):
            for unit_id, fitting, validation in units:
                for seed in _candidate_seeds(pd.Series(candidate._asdict()), p03_contract):
                    fit_rows.append(
                        _fit_record(
                            outer=c09,
                            stage="inner_selection",
                            model_id=candidate.model_id,
                            candidate_id=candidate.candidate_id,
                            hyperparameter_sha256=candidate.hyperparameter_sha256,
                            seed=seed,
                            unit_id=unit_id,
                            fit=fitting,
                            validation=validation,
                            test=test,
                        )
                    )
        for seed, condition in _conditional_selected_seeds(p03_contract):
            fit_rows.append(
                _fit_record(
                    outer=c09,
                    stage="final_selected_refit",
                    model_id="C-SELECTED",
                    candidate_id="selected_after_inner",
                    hyperparameter_sha256=sha256_value({"selection_dependent": True}),
                    seed=seed,
                    unit_id="train_source",
                    fit=source,
                    validation=source.iloc[0:0],
                    test=test,
                    accounting="conditional_fit",
                    condition=condition,
                )
            )
        for unit_id, fitting, validation in calibration_units:
            for seed, condition in _conditional_selected_seeds(p03_contract):
                fit_rows.append(
                    _fit_record(
                        outer=c09,
                        stage="calibration_crossfit",
                        model_id="C-SELECTED",
                        candidate_id="selected_after_inner",
                        hyperparameter_sha256=sha256_value({"calibration": "temperature"}),
                        seed=seed,
                        unit_id=unit_id,
                        fit=fitting,
                        validation=validation,
                        test=test,
                        accounting="cache_reuse"
                        if selection_mode == "master_cv"
                        else "conditional_fit",
                        condition=condition,
                    )
                )

        permutation_controls = control_candidates[
            control_candidates.control_type == "master_label_permutation"
        ]
        for control in permutation_controls.itertuples(index=False):
            permutation_outer = {
                **base,
                "experiment_id": "EXP-C09-CONTROL-PERM",
                "model_id": "C-PERMUTED-SELECTED",
                "scope": "S",
                "source_outer_run_id": c09["outer_run_id"],
                "control_type": control.control_type,
                "control_candidate_id": control.control_candidate_id,
                "control_replicate": control.control_replicate,
            }
            permutation_outer["outer_run_id"] = _outer_run_id(permutation_outer)
            outer_rows.append(permutation_outer)
            for seed, condition in _conditional_selected_seeds(p03_contract):
                fit_rows.append(
                    _control_fit_record(
                        control_type=control.control_type,
                        control_candidate_id=control.control_candidate_id,
                        control_replicate=control.control_replicate,
                        source_outer_run_id=c09["outer_run_id"],
                        label_policy="source_master_labels_permuted_test_labels_real",
                        feature_space="R_MIN_400_1800",
                        outer=permutation_outer,
                        stage="permutation_selected_refit",
                        model_id="C-PERMUTED-SELECTED",
                        candidate_id="frozen_real_C09_selected",
                        hyperparameter_sha256=sha256_value(
                            {
                                "selection_dependency": c09["outer_run_id"],
                                "control_configuration_sha256": control.configuration_sha256,
                            }
                        ),
                        seed=seed,
                        unit_id="train_source",
                        fit=source,
                        validation=source.iloc[0:0],
                        test=test,
                        accounting="conditional_fit",
                        condition=condition,
                    )
                )

        metadata_outer = {
            **base,
            "experiment_id": "EXP-C09-CONTROL-META",
            "model_id": "C-METADATA-LOGREG",
            "scope": "S",
            "source_outer_run_id": c09["outer_run_id"],
            "control_type": "acquisition_metadata_only",
            "control_candidate_id": "selected_after_inner",
            "control_replicate": "not_applicable",
        }
        metadata_outer["outer_run_id"] = _outer_run_id(metadata_outer)
        outer_rows.append(metadata_outer)
        metadata_controls = control_candidates[
            control_candidates.control_type == "acquisition_metadata_only"
        ]
        for control in metadata_controls.itertuples(index=False):
            for unit_id, fitting, validation in units:
                fit_rows.append(
                    _control_fit_record(
                        control_type=control.control_type,
                        control_candidate_id=control.control_candidate_id,
                        control_replicate="not_applicable",
                        source_outer_run_id=c09["outer_run_id"],
                        label_policy="real",
                        feature_space="frozen_acquisition_metadata_allowlist",
                        outer=metadata_outer,
                        stage="metadata_inner_selection",
                        model_id="C-METADATA-LOGREG",
                        candidate_id=control.control_candidate_id,
                        hyperparameter_sha256=control.configuration_sha256,
                        seed="deterministic",
                        unit_id=unit_id,
                        fit=fitting,
                        validation=validation,
                        test=test,
                        condition=str(p03_contract["negative_controls"]["status"]),
                    )
                )
        fit_rows.append(
            _control_fit_record(
                control_type="acquisition_metadata_only",
                control_candidate_id="selected_after_inner",
                control_replicate="not_applicable",
                source_outer_run_id=c09["outer_run_id"],
                label_policy="real",
                feature_space="frozen_acquisition_metadata_allowlist",
                outer=metadata_outer,
                stage="metadata_final_refit",
                model_id="C-METADATA-LOGREG",
                candidate_id="selected_after_inner",
                hyperparameter_sha256=sha256_value(
                    {"metadata_selection_dependency": metadata_outer["outer_run_id"]}
                ),
                seed="deterministic",
                unit_id="train_source",
                fit=source,
                validation=source.iloc[0:0],
                test=test,
                condition=str(p03_contract["negative_controls"]["status"]),
            )
        )
        for unit_id, fitting, validation in calibration_units:
            fit_rows.append(
                _control_fit_record(
                    control_type="acquisition_metadata_only",
                    control_candidate_id="selected_after_inner",
                    control_replicate="not_applicable",
                    source_outer_run_id=c09["outer_run_id"],
                    label_policy="real",
                    feature_space="frozen_acquisition_metadata_allowlist",
                    outer=metadata_outer,
                    stage="metadata_calibration_crossfit",
                    model_id="C-METADATA-LOGREG",
                    candidate_id="selected_after_inner",
                    hyperparameter_sha256=sha256_value(
                        {"metadata_calibration_dependency": metadata_outer["outer_run_id"]}
                    ),
                    seed="deterministic",
                    unit_id=unit_id,
                    fit=fitting,
                    validation=validation,
                    test=test,
                    accounting="cache_reuse" if selection_mode == "master_cv" else "new_fit",
                    condition=str(p03_contract["negative_controls"]["status"]),
                )
            )
        prior_controls = control_candidates[
            control_candidates.control_type == "station_or_target_prior"
        ]
        for control in prior_controls.itertuples(index=False):
            prior_outer = {
                **base,
                "experiment_id": "EXP-C09-CONTROL-PRIOR",
                "model_id": "C-PRIOR",
                "scope": "S",
                "source_outer_run_id": c09["outer_run_id"],
                "control_type": control.control_type,
                "control_candidate_id": control.control_candidate_id,
                "control_replicate": "not_applicable",
            }
            prior_outer["outer_run_id"] = _outer_run_id(prior_outer)
            outer_rows.append(prior_outer)
            fit_rows.append(
                _control_fit_record(
                    control_type=control.control_type,
                    control_candidate_id=control.control_candidate_id,
                    control_replicate="not_applicable",
                    source_outer_run_id=c09["outer_run_id"],
                    label_policy="real",
                    feature_space="none_source_label_prior_only",
                    outer=prior_outer,
                    stage="prior_control_final",
                    model_id="C-PRIOR",
                    candidate_id=control.control_candidate_id,
                    hyperparameter_sha256=control.configuration_sha256,
                    seed="deterministic",
                    unit_id="train_source",
                    fit=source,
                    validation=source.iloc[0:0],
                    test=test,
                    condition=str(p03_contract["negative_controls"]["status"]),
                )
            )
        for model_id in FIXED_SUITE:
            c10 = {**base, "experiment_id": "EXP-C10-T3", "model_id": model_id, "scope": "S"}
            c10["outer_run_id"] = _outer_run_id(c10)
            outer_rows.append(c10)
            fit_rows.append(
                _fit_record(
                    outer=c10,
                    stage="reuse_C09_inner_selection",
                    model_id=model_id,
                    candidate_id="family_selected_from_C09_cache",
                    hyperparameter_sha256=sha256_value({"reuse": "C09"}),
                    seed="cache",
                    unit_id=str(partition_id),
                    fit=source.iloc[0:0],
                    validation=source.iloc[0:0],
                    test=test,
                    accounting="cache_reuse",
                )
            )
            seeds = (
                p03_contract["stochastic_seeds"]
                if model_id in STOCHASTIC_MODELS
                else ["deterministic"]
            )
            for seed in seeds:
                fit_rows.append(
                    _fit_record(
                        outer=c10,
                        stage="final_family_refit",
                        model_id=model_id,
                        candidate_id="family_selected_from_C09_cache",
                        hyperparameter_sha256=sha256_value({"selection_dependent": model_id}),
                        seed=seed,
                        unit_id="train_source",
                        fit=source,
                        validation=source.iloc[0:0],
                        test=test,
                    )
                )
            for unit_id, fitting, validation in calibration_units:
                for seed in seeds:
                    fit_rows.append(
                        _fit_record(
                            outer=c10,
                            stage="calibration_crossfit",
                            model_id=model_id,
                            candidate_id="family_selected_from_C09_cache",
                            hyperparameter_sha256=sha256_value(
                                {"calibration": "temperature", "family": model_id}
                            ),
                            seed=seed,
                            unit_id=unit_id,
                            fit=fitting,
                            validation=validation,
                            test=test,
                            accounting="cache_reuse"
                            if selection_mode == "master_cv"
                            else "new_fit",
                        )
                    )
        c12 = {**base, "experiment_id": "EXP-C12-CORAL", "model_id": "C-SOURCE-CORAL", "scope": "S"}
        c12["outer_run_id"] = _outer_run_id(c12)
        outer_rows.append(c12)
        for candidate in coral_candidates.itertuples(index=False):
            for unit_id, fitting, validation in units:
                fit_rows.append(
                    _fit_record(
                        outer=c12,
                        stage="inner_source_coral_selection",
                        model_id="C-SOURCE-CORAL",
                        candidate_id=candidate.candidate_id,
                        hyperparameter_sha256=candidate.hyperparameter_sha256,
                        seed="deterministic",
                        unit_id=unit_id,
                        fit=fitting,
                        validation=validation,
                        test=test,
                        condition=str(p03_contract["coral"]["status"]),
                    )
                )
        fit_rows.append(
            _fit_record(
                outer=c12,
                stage="final_source_coral_refit",
                model_id="C-SOURCE-CORAL",
                candidate_id="selected_after_inner",
                hyperparameter_sha256=sha256_value({"selection_dependent": True, "coral": True}),
                seed="deterministic",
                unit_id="train_source",
                fit=source,
                validation=source.iloc[0:0],
                test=test,
                condition=str(p03_contract["coral"]["status"]),
            )
        )
        for unit_id, fitting, validation in calibration_units:
            fit_rows.append(
                _fit_record(
                    outer=c12,
                    stage="calibration_crossfit",
                    model_id="C-SOURCE-CORAL",
                    candidate_id="selected_after_inner",
                    hyperparameter_sha256=sha256_value(
                        {"calibration": "temperature", "coral": True}
                    ),
                    seed="deterministic",
                    unit_id=unit_id,
                    fit=fitting,
                    validation=validation,
                    test=test,
                    accounting="cache_reuse" if selection_mode == "master_cv" else "new_fit",
                    condition=str(p03_contract["coral"]["status"]),
                )
            )

    # T2 uses only immutable station and analyte fields; all five P02 source-station
    # split repeats are development folds, while target-station outcomes remain unseen.
    shared_targets = {"4_ANPP", "benzyl_fentanyl"}
    for task_id, source_station, target_station in (
        ("T2-PS", "pills", "surfaces"),
        ("T2-SP", "surfaces", "pills"),
    ):
        train = manifest[
            (manifest.station == source_station) & manifest.target_analyte.isin(shared_targets)
        ]
        test = manifest[
            (manifest.station == target_station) & manifest.target_analyte.isin(shared_targets)
        ]
        outer = {
            "experiment_id": "EXP-C11-T2",
            "task_id": task_id,
            "scope": "S",
            "domain": f"{source_station}_to_{target_station}",
            "station": source_station,
            "held_instrument": "target_station",
            "outer_repeat": -1,
            "outer_fold": -1,
            "selection_mode": "five_repeat_training_station_master_cv",
            "model_id": "C-CANDIDATE-SUITE",
            "execution_status": "planned",
        }
        outer["outer_run_id"] = _outer_run_id(outer)
        outer_rows.append(outer)
        source_assignments = masters[masters.station == source_station]
        inner_units = []
        for (repeat, fold), assigned in source_assignments.groupby(
            ["outer_repeat", "outer_fold"], sort=True
        ):
            validation_masters = set(assigned.master_sample_id) & set(train.master_sample_id)
            validation = train[train.master_sample_id.isin(validation_masters)]
            fitting = train[~train.master_sample_id.isin(validation_masters)]
            inner_units.append((repeat, fold, fitting, validation))
        for candidate in candidates.itertuples(index=False):
            for repeat, fold, fitting, validation in inner_units:
                for seed in _candidate_seeds(pd.Series(candidate._asdict()), p03_contract):
                    fit_rows.append(
                        _fit_record(
                            outer=outer,
                            stage="training_station_inner_selection",
                            model_id=candidate.model_id,
                            candidate_id=candidate.candidate_id,
                            hyperparameter_sha256=candidate.hyperparameter_sha256,
                            seed=seed,
                            unit_id=f"repeat:{repeat}:fold:{fold}",
                            fit=fitting,
                            validation=validation,
                            test=test,
                        )
                    )
        for seed, condition in _conditional_selected_seeds(p03_contract):
            fit_rows.append(
                _fit_record(
                    outer=outer,
                    stage="final_selected_refit",
                    model_id="C-CANDIDATE-SUITE",
                    candidate_id="selected_after_inner",
                    hyperparameter_sha256=sha256_value({"selection_dependent": True}),
                    seed=seed,
                    unit_id="all_training_station",
                    fit=train,
                    validation=train.iloc[0:0],
                    test=test,
                    accounting="conditional_fit",
                    condition=condition,
                )
            )
        fit_rows.append(
            _fit_record(
                outer=outer,
                stage="calibration_crossfit_reuse",
                model_id="C-CANDIDATE-SUITE",
                candidate_id="selected_after_inner",
                hyperparameter_sha256=sha256_value({"calibration": "temperature"}),
                seed="cache",
                unit_id="repeat:1:all_folds",
                fit=train.iloc[0:0],
                validation=train.iloc[0:0],
                test=test,
                accounting="cache_reuse",
            )
        )

    # Exploratory domains remain visible but are not authorized by a P03 experiment.
    exploratory = domains[domains.scope == "exploratory"]
    for domain in exploratory.itertuples(index=False):
        for repeat in range(5):
            for fold in range(4):
                outer = {
                    "experiment_id": "UNREGISTERED-T3-LOW",
                    "task_id": "T3-LOW",
                    "scope": "E",
                    "domain": domain.domain,
                    "station": domain.station,
                    "held_instrument": domain.held_instrument,
                    "outer_repeat": repeat,
                    "outer_fold": fold,
                    "selection_mode": "not_authorized",
                    "model_id": "C-SELECTED",
                    "execution_status": "manifest_only_exploratory",
                }
                outer["outer_run_id"] = _outer_run_id(outer)
                outer_rows.append(outer)

    outer_frame = (
        pd.DataFrame(outer_rows)
        .sort_values(
            ["experiment_id", "domain", "outer_repeat", "outer_fold", "model_id"], kind="stable"
        )
        .reset_index(drop=True)
    )
    fit_frame = (
        pd.DataFrame(fit_rows)
        .sort_values(
            [
                "experiment_id",
                "domain",
                "outer_repeat",
                "outer_fold",
                "stage",
                "model_id",
                "candidate_id",
                "selection_unit_id",
                "seed",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    fit_frame["fit_test_disjoint"] = fit_frame.fit_test_disjoint.astype(bool)
    fit_frame["fit_validation_master_disjoint"] = fit_frame.fit_validation_master_disjoint.astype(
        bool
    )
    return {
        "candidate_registry.csv": candidates,
        "coral_candidate_registry.csv": coral_candidates,
        "control_registry.csv": control_candidates,
        "expected_run_registry.csv": outer_frame,
        "fit_manifest.csv": fit_frame,
    }


def summarize_compute(
    tables: dict[str, pd.DataFrame | dict[str, Any]], p03_contract: dict[str, Any]
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    fits = tables["fit_manifest.csv"]
    assert isinstance(fits, pd.DataFrame)
    bounds = p03_contract["resource_estimation"]["cpu_seconds_per_fit_bounds"]
    direct = fits[fits.accounting == "new_fit"].copy()
    conditional = fits[fits.accounting == "conditional_fit"].copy()
    # Each selected outer model activates either the single deterministic slot
    # or all three stochastic slots. Resource planning uses the larger branch;
    # every inactive slot still receives excluded_by_protocol at execution.
    conditional_upper = conditional[
        conditional.condition == "selected_model_is_stochastic"
    ].copy()
    counted = pd.concat([direct, conditional_upper], ignore_index=True)
    counted["resource_model_id"] = counted.model_id.where(
        counted.model_id != "C-SELECTED", "C-RBF-SVM"
    ).where(counted.model_id != "C-CANDIDATE-SUITE", "C-RBF-SVM")
    counted.loc[
        counted.model_id == "C-METADATA-LOGREG", "resource_model_id"
    ] = "C-LOGREG-EN"
    counted.loc[
        counted.model_id == "C-PERMUTED-SELECTED", "resource_model_id"
    ] = "C-RBF-SVM"
    counted.loc[
        counted.condition == "selected_model_is_stochastic", "resource_model_id"
    ] = "C-RANDOM-FOREST"
    counted.loc[counted.model_id == "C-SOURCE-CORAL", "resource_model_id"] = "C-SOURCE-CORAL"
    counted["cpu_seconds_low"] = counted.resource_model_id.map(lambda value: bounds[value][0])
    counted["cpu_seconds_high"] = counted.resource_model_id.map(lambda value: bounds[value][1])
    summary = (
        counted.groupby(["experiment_id", "task_id", "model_id"], dropna=False)
        .agg(
            fit_count=("fit_id", "size"),
            cpu_seconds_low=("cpu_seconds_low", "sum"),
            cpu_seconds_high=("cpu_seconds_high", "sum"),
        )
        .reset_index()
    )
    summary["cpu_hours_low"] = summary.cpu_seconds_low / 3600
    summary["cpu_hours_high"] = summary.cpu_seconds_high / 3600
    cpu = summary.groupby("experiment_id", as_index=False).agg(
        fit_count=("fit_count", "sum"),
        estimated_cpu_hours_low=("cpu_hours_low", "sum"),
        estimated_cpu_hours_high=("cpu_hours_high", "sum"),
    )
    gpu = pd.DataFrame(
        [{"phase": "P03", "estimated_gpu_hours_low": 0.0, "estimated_gpu_hours_high": 0.0}]
    )
    final_prediction_records = int(
        (fits.drop_duplicates(["outer_run_id", "test_uid_sha256"])["test_rows"]).sum()
    )
    selection_prediction_records = int(
        direct.loc[
            direct.stage.isin(SELECTION_FIT_STAGES),
            "validation_rows",
        ].sum()
    )
    calibration_prediction_records = int(
        counted.loc[
            counted.stage.isin({"calibration_crossfit", "metadata_calibration_crossfit"}),
            "validation_rows",
        ].sum()
    )
    planned_predictions = (
        final_prediction_records
        + selection_prediction_records
        + calibration_prediction_records
    )
    record_bytes = int(p03_contract["resource_estimation"]["prediction_record_bytes"])
    overhead = float(p03_contract["resource_estimation"]["checkpoint_overhead_fraction"])
    estimated_bytes = int(planned_predictions * record_bytes * (1 + overhead))
    target = int(p03_contract["resource_estimation"]["shard_target_fits"])
    counted = counted.reset_index(drop=True)
    counted["shard_id"] = counted.index // target
    shards = counted.groupby("shard_id", as_index=False).agg(
        fit_count=("fit_id", "size"),
        first_fit_id=("fit_id", "first"),
        last_fit_id=("fit_id", "last"),
        cpu_seconds_low=("cpu_seconds_low", "sum"),
        cpu_seconds_high=("cpu_seconds_high", "sum"),
    )
    selection_counted = assign_selection_shards(counted, target=target)
    selection_shards = selection_counted.groupby("selection_shard_id", as_index=False).agg(
        selection_kind=("selection_kind", "first"),
        stage_count=("stage", "nunique"),
        fit_count=("fit_id", "size"),
        first_fit_id=("fit_id", "first"),
        last_fit_id=("fit_id", "last"),
        fit_id_sha256=("fit_id", lambda values: _uid_hash(values)),
        cpu_seconds_low=("cpu_seconds_low", "sum"),
        cpu_seconds_high=("cpu_seconds_high", "sum"),
    )
    if selection_shards.stage_count.ne(1).any():
        raise RuntimeError("A selection shard mixes stages despite kind partitioning.")
    budget_low = int(p03_contract["planning"]["registered_fit_budget_low"])
    budget_high = int(p03_contract["planning"]["registered_fit_budget_high"])
    if budget_low < 1 or budget_high < budget_low:
        raise ValueError("P03 registered fit-budget bounds are invalid.")
    fit_count = len(counted)
    coral_unresolved = (
        p03_contract["coral"]["status"]
        == "requires_versioned_method_resolution_before_fitting"
    )
    controls_unresolved = (
        p03_contract["negative_controls"]["status"]
        == "requires_versioned_scope_resolution_before_fitting"
    )
    protocol_not_authorized = not bool(p03_contract["planning"]["model_fitting_authorized"])
    blocking_reasons = [
        reason
        for reason, active in (
            ("protocol_model_fitting_not_authorized", protocol_not_authorized),
            ("literal_grid_exceeds_registered_compute_budget", fit_count > budget_high),
            ("source_only_CORAL_method_not_fully_resolved", coral_unresolved),
            ("negative_control_scope_not_fully_resolved", controls_unresolved),
        )
        if active
    ]
    budget = {
        "schema_version": "p03-compute-budget-gate-v1",
        "status": "fail" if blocking_reasons else "pass",
        "planned_new_fit_count": fit_count,
        "conditional_task_row_count": len(conditional),
        "maximum_activated_conditional_fit_count": len(conditional_upper),
        "registered_fit_estimate_low": budget_low,
        "registered_fit_estimate_high": budget_high,
        "excess_over_registered_high": max(0, fit_count - budget_high),
        "scientific_fitting_authorized": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
    return {
        "fit_count_by_phase_model_task.csv": summary,
        "estimated_cpu_hours.csv": cpu,
        "estimated_gpu_hours.csv": gpu,
        "estimated_disk_bytes.json": {
            "schema_version": "p03-disk-estimate-v1",
            "selection_validation_prediction_records": selection_prediction_records,
            "calibration_validation_prediction_records_upper_bound": (
                calibration_prediction_records
            ),
            "final_prediction_records_upper_bound": final_prediction_records,
            "planned_prediction_records_upper_bound": planned_predictions,
            "bytes_per_record": record_bytes,
            "checkpoint_overhead_fraction": overhead,
            "estimated_disk_bytes": estimated_bytes,
        },
        "shard_manifest.csv": shards,
        "selection_shard_manifest.csv": selection_shards,
        "budget_gate.json": budget,
    }
