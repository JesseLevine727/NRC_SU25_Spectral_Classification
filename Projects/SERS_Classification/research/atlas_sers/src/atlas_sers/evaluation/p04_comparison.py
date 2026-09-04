"""Fair D0/classical comparison on identical frozen P02 T3 test rows."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import instrument_balanced_master_probabilities
from atlas_sers.evaluation.p04_results import endpoint_metrics
from atlas_sers.governance.canonical import sha256_value

CLASSICAL_MODELS = (
    "C-SELECTED",
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def normalize_classical_predictions(
    predictions: pd.DataFrame, p04_contexts: pd.DataFrame
) -> pd.DataFrame:
    """Map P03 C09/C10 rows to P04 endpoint identities without changing scores."""

    c09 = predictions[predictions.experiment_id.astype(str).eq("EXP-C09-T3")].copy()
    c09["comparison_model_id"] = "C-SELECTED"
    c10 = predictions[
        predictions.experiment_id.astype(str).eq("EXP-C10-T3")
        & predictions.model_id.astype(str).isin(CLASSICAL_MODELS[1:])
    ].copy()
    c10["comparison_model_id"] = c10.model_id.astype(str)
    frame = pd.concat([c09, c10], ignore_index=True)
    context_lookup = p04_contexts[["context_id", "domain", "outer_repeat", "outer_fold"]].copy()
    context_lookup = context_lookup[p04_contexts.experiment_id.astype(str).eq("EXP-N00-T3")]
    frame = frame.merge(
        context_lookup,
        on=["domain", "outer_repeat", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    if frame.context_id.isna().any():
        raise ValueError("A P03 classical prediction has no matching P04 T3 context.")
    frame["experiment_id"] = "EXP-N00-T3"
    frame["candidate_id"] = frame.comparison_model_id
    frame["class_vocabulary"] = frame.class_vocabulary.map(
        lambda value: json.dumps(_as_list(value), separators=(",", ":"))
    )
    probability_matrix = np.asarray([_as_list(value) for value in frame.probabilities], dtype=float)
    if probability_matrix.shape != (len(frame), 3):
        raise ValueError("P03 classical probabilities are not station-local three-class rows.")
    for index in range(3):
        frame[f"probability_{index}"] = probability_matrix[:, index]
    return frame


def _uid_sets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in frame.groupby(["comparison_model_id", "context_id"], sort=True):
        rows.append(
            {
                "comparison_model_id": keys[0],
                "context_id": keys[1],
                "rows": len(cell),
                "uid_sha256": sha256_value(sorted(cell.observation_uid.astype(str).tolist())),
            }
        )
    return pd.DataFrame(rows)


def compare_endpoint_metrics(
    *,
    d0_ensemble: pd.DataFrame,
    classical: pd.DataFrame,
    expected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d0_spectrum, d0_master = endpoint_metrics(d0_ensemble)
    d0_metrics = pd.concat([d0_spectrum, d0_master], ignore_index=True)
    classical_metrics = []
    for model, rows in classical.groupby("comparison_model_id", sort=True):
        spectrum, master = endpoint_metrics(rows)
        metric = pd.concat([spectrum, master], ignore_index=True)
        metric["comparison_model_id"] = model
        classical_metrics.append(metric)
    classical_metric = pd.concat(classical_metrics, ignore_index=True)
    d0_sets = (
        d0_ensemble.groupby("context_id")
        .observation_uid.apply(lambda values: sha256_value(sorted(values.astype(str))))
        .rename("d0_uid_sha256")
        .reset_index()
    )
    classical_sets = _uid_sets(classical)
    support = (
        expected[expected.experiment_id.eq("EXP-N00-T3")][
            [
                "context_id",
                "domain",
                "station",
                "held_instrument",
                "outer_repeat",
                "outer_fold",
                "expected_test_rows",
                "expected_test_uid_sha256",
            ]
        ]
        .assign(join_key=1)
        .merge(
            pd.DataFrame({"comparison_model_id": CLASSICAL_MODELS, "join_key": 1}),
            on="join_key",
        )
        .drop(columns="join_key")
        .merge(d0_sets, on="context_id", how="left", validate="many_to_one")
        .merge(
            classical_sets,
            on=["comparison_model_id", "context_id"],
            how="left",
            validate="one_to_one",
        )
    )
    support["d0_complete"] = support.d0_uid_sha256.eq(support.expected_test_uid_sha256)
    support["classical_complete"] = support.uid_sha256.eq(
        support.expected_test_uid_sha256
    ) & support.rows.eq(support.expected_test_rows)
    support["common_complete"] = support.d0_complete & support.classical_complete
    paired = d0_metrics[
        [
            "context_id",
            "aggregation_id",
            "balanced_accuracy",
            "macro_f1",
            "negative_log_likelihood",
            "ece",
        ]
    ].rename(
        columns={
            "balanced_accuracy": "d0_balanced_accuracy",
            "macro_f1": "d0_macro_f1",
            "negative_log_likelihood": "d0_negative_log_likelihood",
            "ece": "d0_ece",
        }
    )
    paired = support.merge(paired, on="context_id", how="left", validate="many_to_many")
    paired = paired.merge(
        classical_metric[
            [
                "comparison_model_id",
                "context_id",
                "aggregation_id",
                "balanced_accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "ece",
            ]
        ].rename(
            columns={
                "balanced_accuracy": "classical_balanced_accuracy",
                "macro_f1": "classical_macro_f1",
                "negative_log_likelihood": "classical_negative_log_likelihood",
                "ece": "classical_ece",
            }
        ),
        on=["comparison_model_id", "context_id", "aggregation_id"],
        how="left",
        validate="one_to_one",
    )
    for metric in (
        "balanced_accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "ece",
    ):
        paired[f"delta_d0_minus_classical_{metric}"] = (
            paired[f"d0_{metric}"] - paired[f"classical_{metric}"]
        )
    summaries = []
    for keys, cell in paired.groupby(["comparison_model_id", "aggregation_id"], sort=True):
        common = cell[cell.common_complete]
        domain = common.groupby("domain", as_index=False).agg(
            d0=("d0_balanced_accuracy", "mean"),
            classical=("classical_balanced_accuracy", "mean"),
        )
        summaries.append(
            {
                "comparison_model_id": keys[0],
                "aggregation_id": keys[1],
                "planned_endpoints": len(cell),
                "common_complete_endpoints": len(common),
                "common_coverage": len(common) / len(cell),
                "d0_mean_ba_common": float(common.d0_balanced_accuracy.mean()),
                "classical_mean_ba_common": float(common.classical_balanced_accuracy.mean()),
                "mean_paired_delta_d0_minus_classical": float(
                    common.delta_d0_minus_classical_balanced_accuracy.mean()
                ),
                "d0_worst_domain_ba_common": float(domain.d0.min()),
                "classical_worst_domain_ba_common": float(domain.classical.min()),
                "d0_failure_sensitive_mean_ba": float(cell.d0_balanced_accuracy.fillna(0).mean()),
                "classical_failure_sensitive_mean_ba": float(
                    cell.classical_balanced_accuracy.fillna(0).mean()
                ),
                "mean_nll_delta_d0_minus_classical": float(
                    common.delta_d0_minus_classical_negative_log_likelihood.mean()
                ),
                "mean_ece_delta_d0_minus_classical": float(
                    common.delta_d0_minus_classical_ece.mean()
                ),
            }
        )
    return paired, pd.DataFrame(summaries), support


def _master_correctness_from_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for keys, cell in frame.groupby(["domain", "outer_repeat", "master_sample_id"], sort=True):
        classes = _as_list(cell.class_vocabulary.iloc[0])
        probability_columns = ["probability_0", "probability_1", "probability_2"]
        master = instrument_balanced_master_probabilities(
            probabilities=cell[probability_columns].to_numpy(),
            true_labels=cell.true_label.astype(str).to_numpy(),
            master_ids=cell.master_sample_id.astype(str).to_numpy(),
            instruments=cell.instrument.astype(str).to_numpy(),
            class_vocabulary=classes,
        )
        if len(master) != 1:
            raise ValueError("A P04 comparison master group did not resolve to one master.")
        rows.append(
            {
                "model_id": model,
                "domain": keys[0],
                "outer_repeat": int(keys[1]),
                "master_sample_id": str(keys[2]),
                "station": str(cell.station.iloc[0]),
                "true_label": str(master.true_label.iloc[0]),
                "correct": float(master.predicted_label.iloc[0] == master.true_label.iloc[0]),
                "spectrum_count": len(cell),
            }
        )
    return pd.DataFrame(rows)


def master_clustered_paired_bootstrap(
    *,
    d0_ensemble: pd.DataFrame,
    classical: pd.DataFrame,
    support: pd.DataFrame,
    draws: int = 5000,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Bootstrap physical masters, preserving every repeated prediction as correlated."""

    records = []
    rng = np.random.default_rng(seed)
    for model in CLASSICAL_MODELS:
        eligible_contexts = set(
            support[
                support.comparison_model_id.eq(model) & support.common_complete
            ].context_id.astype(str)
        )
        d0 = d0_ensemble[d0_ensemble.context_id.astype(str).isin(eligible_contexts)].copy()
        comparator = classical[
            classical.comparison_model_id.eq(model)
            & classical.context_id.astype(str).isin(eligible_contexts)
        ].copy()
        row_keys = ["context_id", "outer_repeat", "observation_uid"]
        aligned = d0.merge(
            comparator[
                row_keys
                + [
                    "predicted_label",
                    "true_label",
                ]
            ],
            on=row_keys,
            suffixes=("_d0", "_classical"),
            validate="one_to_one",
        )
        if (
            len(aligned) != len(d0)
            or not aligned.true_label_d0.eq(aligned.true_label_classical).all()
        ):
            raise ValueError("P04/classical common row pairing failed.")
        spectrum = aligned.assign(
            d0_correct=aligned.predicted_label_d0.eq(aligned.true_label_d0).astype(float),
            classical_correct=aligned.predicted_label_classical.eq(
                aligned.true_label_classical
            ).astype(float),
        )[
            [
                "domain",
                "station",
                "master_sample_id",
                "true_label_d0",
                "outer_repeat",
                "observation_uid",
                "d0_correct",
                "classical_correct",
            ]
        ].rename(
            columns={
                "true_label_d0": "true_label",
            }
        )
        spectrum = spectrum.groupby(
            [
                "domain",
                "station",
                "master_sample_id",
                "true_label",
                "observation_uid",
            ],
            as_index=False,
        ).agg(
            d0_correct=("d0_correct", "mean"),
            classical_correct=("classical_correct", "mean"),
        )
        spectrum["base_weight"] = 1.0
        d0_master = _master_correctness_from_rows(d0, "D0-ERM").rename(
            columns={"correct": "d0_correct"}
        )
        classical_master = _master_correctness_from_rows(comparator, model).rename(
            columns={"correct": "classical_correct"}
        )
        master = d0_master.merge(
            classical_master[
                [
                    "domain",
                    "outer_repeat",
                    "master_sample_id",
                    "classical_correct",
                ]
            ],
            on=["domain", "outer_repeat", "master_sample_id"],
            validate="one_to_one",
        )
        master = master.groupby(
            ["domain", "station", "master_sample_id", "true_label"], as_index=False
        ).agg(
            d0_correct=("d0_correct", "mean"),
            classical_correct=("classical_correct", "mean"),
        )
        master["base_weight"] = 1.0
        for aggregation_id, units in (("M01", spectrum), ("M06", master)):
            master_labels = units[["station", "true_label", "master_sample_id"]].drop_duplicates()
            strata = [
                group.master_sample_id.astype(str).to_numpy()
                for _, group in master_labels.groupby(["station", "true_label"], sort=True)
            ]
            point_domain = _weighted_domain_delta(units, None)
            evaluate_draw = _prepare_domain_delta(units)
            bootstrap_values: dict[str, list[float]] = {
                domain: [] for domain in ["__overall__", *sorted(point_domain)]
            }
            attempts = 0
            while len(bootstrap_values["__overall__"]) < draws:
                attempts += 1
                if attempts > draws * 10:
                    raise RuntimeError("Master-clustered bootstrap could not retain enough draws.")
                weights: dict[str, int] = {}
                for members in strata:
                    sampled = rng.choice(members, size=len(members), replace=True)
                    members_drawn, counts = np.unique(sampled, return_counts=True)
                    weights.update(
                        {
                            str(key): int(value)
                            for key, value in zip(members_drawn, counts, strict=True)
                        }
                    )
                domain_values = evaluate_draw(weights)
                if set(domain_values) != set(point_domain):
                    continue
                for domain, value in domain_values.items():
                    bootstrap_values[domain].append(value)
                bootstrap_values["__overall__"].append(float(np.mean(list(domain_values.values()))))
            point_overall = float(np.mean(list(point_domain.values())))
            for domain, values in bootstrap_values.items():
                array = np.asarray(values, dtype=float)
                point = point_overall if domain == "__overall__" else point_domain[domain]
                records.append(
                    {
                        "comparison_model_id": model,
                        "aggregation_id": aggregation_id,
                        "domain": domain,
                        "estimate_d0_minus_classical_ba": point,
                        "lower_95": float(np.quantile(array, 0.025)),
                        "upper_95": float(np.quantile(array, 0.975)),
                        "probability_delta_above_zero": float(np.mean(array > 0)),
                        "bootstrap_draws": len(array),
                        "bootstrap_seed": seed,
                        "independent_physical_masters": (
                            master_labels.master_sample_id.nunique()
                            if domain == "__overall__"
                            else units.loc[units.domain.eq(domain), "master_sample_id"].nunique()
                        ),
                    }
                )
    return pd.DataFrame(records)


def _prepare_domain_delta(
    units: pd.DataFrame,
) -> Callable[[dict[str, int]], dict[str, float]]:
    """Cache cluster totals; evaluate the same weighted statistic without groupby per draw."""

    masters = pd.Index(sorted(units.master_sample_id.astype(str).unique()))
    master_codes = masters.get_indexer(units.master_sample_id.astype(str))
    group_codes, group_keys = pd.MultiIndex.from_frame(
        units[["domain", "true_label"]]
    ).factorize(sort=True)
    shape = (len(group_keys), len(masters))
    denominator = np.zeros(shape, dtype=float)
    d0_numerator = np.zeros(shape, dtype=float)
    classical_numerator = np.zeros(shape, dtype=float)
    base = units.base_weight.to_numpy(dtype=float)
    np.add.at(denominator, (group_codes, master_codes), base)
    np.add.at(d0_numerator, (group_codes, master_codes), base * units.d0_correct.to_numpy())
    np.add.at(
        classical_numerator,
        (group_codes, master_codes),
        base * units.classical_correct.to_numpy(),
    )
    domain_groups = {
        str(domain): np.flatnonzero(group_keys.get_level_values(0) == domain)
        for domain in sorted(units.domain.unique())
    }

    def evaluate(master_weights: dict[str, int]) -> dict[str, float]:
        weights = np.fromiter(
            (master_weights.get(str(master), 0) for master in masters),
            dtype=float,
            count=len(masters),
        )
        totals = denominator @ weights
        supported = totals > 0
        d0_scores = np.divide(
            d0_numerator @ weights, totals, out=np.zeros(len(totals)), where=supported
        )
        classical_scores = np.divide(
            classical_numerator @ weights, totals, out=np.zeros(len(totals)), where=supported
        )
        return {
            domain: float(d0_scores[indices].mean() - classical_scores[indices].mean())
            for domain, indices in domain_groups.items()
            if len(indices) == 3 and supported[indices].all()
        }

    return evaluate


def _weighted_domain_delta(
    units: pd.DataFrame, master_weights: dict[str, int] | None
) -> dict[str, float]:
    frame = units.copy()
    if master_weights is None:
        frame["bootstrap_weight"] = 1.0
    else:
        frame["bootstrap_weight"] = (
            frame.master_sample_id.astype(str).map(master_weights).fillna(0.0)
        )
    frame["weight"] = frame.base_weight * frame.bootstrap_weight
    rows = []
    for keys, cell in frame.groupby(["domain", "true_label"], sort=True):
        denominator = float(cell.weight.sum())
        if denominator <= 0:
            continue
        rows.append(
            {
                "domain": keys[0],
                "d0": float(np.average(cell.d0_correct, weights=cell.weight)),
                "classical": float(np.average(cell.classical_correct, weights=cell.weight)),
            }
        )
    class_scores = pd.DataFrame(rows)
    if class_scores.empty:
        return {}
    counts = class_scores.groupby("domain").size()
    class_scores = class_scores[class_scores.domain.isin(counts[counts.eq(3)].index)]
    domain = class_scores.groupby("domain", as_index=False).agg(
        d0=("d0", "mean"), classical=("classical", "mean")
    )
    return {
        str(row.domain): float(row.d0 - row.classical) for row in domain.itertuples(index=False)
    }
