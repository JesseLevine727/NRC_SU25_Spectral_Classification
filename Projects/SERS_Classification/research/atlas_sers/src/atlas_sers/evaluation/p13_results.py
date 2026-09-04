"""Aggregate locked P13 predictions into metrics, intervals, and bounded claims."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from atlas_sers.evaluation.classical import classification_metrics
from atlas_sers.governance.canonical import sha256_value

BOOTSTRAP_RESAMPLES = 10_000
TAU = 0.60
DELTA = 0.10
CHANCE_BA = 1 / 3


@dataclass(frozen=True)
class P13AggregateTables:
    master_view_predictions: pd.DataFrame
    domain_metrics: pd.DataFrame
    interval_table: pd.DataFrame
    domain_claims: pd.DataFrame
    class_cell_claims: pd.DataFrame
    substrate_claims: pd.DataFrame
    preprocessing_sensitivity: pd.DataFrame
    procedure_comparison: pd.DataFrame
    crossover_effects: pd.DataFrame
    field_log_results: pd.DataFrame
    failure_table: pd.DataFrame


def _probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    vocabularies = frame.class_vocabulary.astype(str).unique()
    if len(vocabularies) != 1:
        raise ValueError("Prediction group has multiple class vocabularies.")
    vocabulary = tuple(str(value) for value in json.loads(vocabularies[0]))
    values = np.vstack(
        [np.asarray(json.loads(value), dtype=float) for value in frame.probabilities]
    )
    if values.shape != (len(frame), len(vocabulary)) or not np.isfinite(values).all():
        raise ValueError("Prediction probability matrix is invalid.")
    return values, vocabulary


def build_master_view_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average outer repeats, then technical repeats, at the frozen view unit."""

    keys = [
        "domain_id",
        "station",
        "substrate_family",
        "held_instrument",
        "support_tier",
        "policy_id",
        "procedure_id",
        "prediction_role",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "true_label",
        "class_vocabulary",
    ]
    records: list[dict[str, Any]] = []
    for group_key, group in predictions.groupby(keys, sort=True, dropna=False):
        probabilities, vocabulary = _probabilities(group)
        record = dict(zip(keys, group_key, strict=True))
        record["probabilities_array"] = probabilities.mean(axis=0)
        record["outer_repeat_predictions"] = group.outer_repeat.nunique()
        record["class_vocabulary_tuple"] = vocabulary
        record["candidate_ids"] = tuple(sorted(set(group.candidate_id.astype(str))))
        record["model_ids"] = tuple(sorted(set(group.model_id.astype(str))))
        records.append(record)
    observation = pd.DataFrame(records)
    view_keys = [
        "domain_id",
        "station",
        "substrate_family",
        "held_instrument",
        "support_tier",
        "policy_id",
        "procedure_id",
        "prediction_role",
        "master_sample_id",
        "instrument",
        "true_label",
        "class_vocabulary",
    ]
    views: list[dict[str, Any]] = []
    for group_key, group in observation.groupby(view_keys, sort=True, dropna=False):
        record = dict(zip(view_keys, group_key, strict=True))
        probability = np.mean(np.stack(group.probabilities_array), axis=0)
        vocabulary = tuple(group.class_vocabulary_tuple.iloc[0])
        record["probabilities"] = json.dumps(
            probability.tolist(), separators=(",", ":")
        )
        record["predicted_label"] = vocabulary[int(np.argmax(probability))]
        record["technical_repeat_count"] = len(group)
        record["outer_repeat_predictions_min"] = int(
            group.outer_repeat_predictions.min()
        )
        record["candidate_ids_json"] = json.dumps(
            sorted(set().union(*group.candidate_ids)), separators=(",", ":")
        )
        record["model_ids_json"] = json.dumps(
            sorted(set().union(*group.model_ids)), separators=(",", ":")
        )
        views.append(record)
    return pd.DataFrame(views).sort_values(view_keys, kind="stable").reset_index(
        drop=True
    )


def _master_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (master, label, vocabulary_json), group in frame.groupby(
        ["master_sample_id", "true_label", "class_vocabulary"], sort=True
    ):
        probabilities, vocabulary = _probabilities(group)
        probability = probabilities.mean(axis=0)
        records.append(
            {
                "master_sample_id": str(master),
                "true_label": str(label),
                "class_vocabulary": str(vocabulary_json),
                "predicted_label": vocabulary[int(np.argmax(probability))],
                "probabilities_array": probability,
                "instrument_views": group.instrument.astype(str).nunique(),
            }
        )
    return pd.DataFrame(records)


def _fixed_balanced_accuracy(frame: pd.DataFrame, vocabulary: tuple[str, ...]) -> float:
    recalls = []
    for label in vocabulary:
        rows = frame[frame.true_label.astype(str).eq(label)]
        if rows.empty:
            return float("nan")
        recalls.append(float(rows.predicted_label.astype(str).eq(label).mean()))
    return float(np.mean(recalls))


def _class_arrays(
    frame: pd.DataFrame, vocabulary: tuple[str, ...], value: str
) -> list[np.ndarray]:
    return [
        frame.loc[frame.true_label.astype(str).eq(label), value].to_numpy(dtype=float)
        for label in vocabulary
    ]


def _bootstrap_class_means(
    arrays: list[np.ndarray], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_means = []
    for values in arrays:
        if len(values) == 0:
            return np.full(resamples, np.nan)
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        class_means.append(values[indices].mean(axis=1))
    return np.mean(np.stack(class_means), axis=0)


def _jackknife_class_means(arrays: list[np.ndarray]) -> np.ndarray:
    values: list[float] = []
    for class_index, members in enumerate(arrays):
        if len(members) < 2:
            continue
        for member_index in range(len(members)):
            reduced = [item.copy() for item in arrays]
            reduced[class_index] = np.delete(reduced[class_index], member_index)
            values.append(float(np.mean([item.mean() for item in reduced])))
    return np.asarray(values, dtype=float)


def _jackknife_values(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.asarray([], dtype=float)
    total = values.sum()
    return (total - values) / (len(values) - 1)


def _interval(
    observed: float,
    bootstrap: np.ndarray,
    jackknife: np.ndarray,
) -> dict[str, Any]:
    finite = bootstrap[np.isfinite(bootstrap)]
    if len(finite) != BOOTSTRAP_RESAMPLES:
        return {
            "estimate": observed,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "interval_method": "unavailable",
            "bootstrap_resamples": len(finite),
        }
    percentile = np.quantile(finite, [0.025, 0.975])
    method = "percentile"
    lower, upper = float(percentile[0]), float(percentile[1])
    if len(jackknife) >= 3 and np.isfinite(jackknife).all():
        proportion = np.clip(np.mean(finite < observed), 1e-6, 1 - 1e-6)
        z0 = float(norm.ppf(proportion))
        center = float(jackknife.mean())
        difference = center - jackknife
        denominator = 6 * float(np.sum(difference**2) ** 1.5)
        acceleration = float(np.sum(difference**3) / denominator) if denominator else np.nan
        if np.isfinite(acceleration):
            adjusted = []
            stable = True
            for alpha in (0.025, 0.975):
                z_alpha = float(norm.ppf(alpha))
                divisor = 1 - acceleration * (z0 + z_alpha)
                if divisor == 0:
                    stable = False
                    break
                probability = float(norm.cdf(z0 + (z0 + z_alpha) / divisor))
                if not 0 <= probability <= 1:
                    stable = False
                    break
                adjusted.append(probability)
            if stable and adjusted[0] < adjusted[1]:
                lower, upper = [float(value) for value in np.quantile(finite, adjusted)]
                method = "BCa"
    return {
        "estimate": observed,
        "lower_95": lower,
        "upper_95": upper,
        "interval_method": method,
        "bootstrap_resamples": len(finite),
    }


def _metric_record(
    group: pd.DataFrame,
    *,
    endpoint_status: str,
    endpoint_reason: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = {
        field: group[field].iloc[0]
        for field in (
            "domain_id",
            "station",
            "substrate_family",
            "held_instrument",
            "support_tier",
            "policy_id",
            "procedure_id",
            "candidate_ids_json",
            "model_ids_json",
        )
    }
    held = _master_predictions(group[group.prediction_role.eq("held_test")])
    vocabulary = tuple(json.loads(str(held.class_vocabulary.iloc[0])))
    held_probabilities = np.stack(held.probabilities_array)
    held_metrics = classification_metrics(
        held.true_label.astype(str).to_numpy(),
        held.predicted_label.astype(str).to_numpy(),
        class_vocabulary=vocabulary,
        probabilities=held_probabilities,
    )
    source_views = group[group.prediction_role.eq("matched_source_test")]
    source = _master_predictions(source_views) if not source_views.empty else pd.DataFrame()
    matched_held = held[
        held.master_sample_id.astype(str).isin(source.master_sample_id.astype(str))
    ] if not source.empty else pd.DataFrame()
    source_ba = (
        _fixed_balanced_accuracy(source, vocabulary) if not source.empty else np.nan
    )
    matched_held_ba = (
        _fixed_balanced_accuracy(matched_held, vocabulary)
        if not matched_held.empty
        else np.nan
    )
    loss = source_ba - matched_held_ba
    held["correct"] = held.predicted_label.astype(str).eq(held.true_label).astype(float)
    seed_base = int(
        sha256_value(
            {key: str(value) for key, value in identity.items() if key != "candidate_id"}
        )[:8],
        16,
    )
    held_arrays = _class_arrays(held, vocabulary, "correct")
    held_bootstrap = _bootstrap_class_means(held_arrays, seed=seed_base)
    held_interval = _interval(
        float(held_metrics["balanced_accuracy"]),
        held_bootstrap,
        _jackknife_class_means(held_arrays),
    )
    intervals = [
        {
            **identity,
            "metric_id": "P13-M01",
            "metric_name": "held_balanced_accuracy",
            **held_interval,
            "one_sided_threshold": TAU,
            "opposite_tail_probability": float(np.mean(held_bootstrap < TAU)),
        }
    ]
    loss_interval: dict[str, Any]
    if not source.empty:
        paired = source.merge(
            matched_held[["master_sample_id", "true_label", "predicted_label"]],
            on=["master_sample_id", "true_label"],
            suffixes=("_source", "_held"),
            validate="one_to_one",
        )
        paired["correctness_difference"] = (
            paired.predicted_label_source.astype(str).eq(paired.true_label).astype(float)
            - paired.predicted_label_held.astype(str).eq(paired.true_label).astype(float)
        )
        loss_arrays = _class_arrays(paired, vocabulary, "correctness_difference")
        loss_bootstrap = _bootstrap_class_means(loss_arrays, seed=seed_base + 1)
        loss_interval = _interval(
            float(loss), loss_bootstrap, _jackknife_class_means(loss_arrays)
        )
        intervals.append(
            {
                **identity,
                "metric_id": "P13-M02",
                "metric_name": "matched_source_minus_held_balanced_accuracy",
                **loss_interval,
                "one_sided_threshold": DELTA,
                "opposite_tail_probability": float(np.mean(loss_bootstrap > DELTA)),
            }
        )
    else:
        loss_interval = _interval(np.nan, np.asarray([]), np.asarray([]))
        intervals.append(
            {
                **identity,
                "metric_id": "P13-M02",
                "metric_name": "matched_source_minus_held_balanced_accuracy",
                **loss_interval,
                "one_sided_threshold": DELTA,
                "opposite_tail_probability": np.nan,
            }
        )
    record = {
        **identity,
        "endpoint_status": endpoint_status,
        "endpoint_reason": endpoint_reason,
        "held_masters": len(held),
        "held_balanced_accuracy": held_metrics["balanced_accuracy"],
        "held_macro_f1": held_metrics["macro_f1"],
        "held_negative_log_likelihood": held_metrics["negative_log_likelihood"],
        "held_brier_score": held_metrics["brier_score"],
        "held_ece": held_metrics["ece"],
        "held_per_class_recall_json": json.dumps(
            held_metrics["per_class_recall"], sort_keys=True, separators=(",", ":")
        ),
        "matched_masters": len(source),
        "matched_source_balanced_accuracy": source_ba,
        "matched_held_balanced_accuracy": matched_held_ba,
        "source_minus_held_balanced_accuracy": loss,
        "held_lcb95": held_interval["lower_95"],
        "held_ucb95": held_interval["upper_95"],
        "loss_lcb95": loss_interval["lower_95"],
        "loss_ucb95": loss_interval["upper_95"],
    }
    return record, intervals


def build_domain_results(
    *,
    master_views: pd.DataFrame,
    fold_endpoints: pd.DataFrame,
    domain_support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score eligible endpoints and retain every unsupported or failed domain."""

    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    endpoint_keys = ["domain_id", "policy_id", "procedure_id"]
    for key, endpoint in fold_endpoints.groupby(endpoint_keys, sort=True):
        domain_id, policy_id, procedure_id = key
        terminal = endpoint.status.astype(str).eq("unavailable_terminal_failure").any()
        group = master_views[
            master_views.domain_id.astype(str).eq(str(domain_id))
            & master_views.policy_id.astype(str).eq(str(policy_id))
            & master_views.procedure_id.astype(str).eq(str(procedure_id))
        ]
        if terminal or group[group.prediction_role.eq("held_test")].empty:
            first = endpoint.iloc[0]
            metric_rows.append(
                {
                    "domain_id": str(domain_id),
                    "station": str(first.station),
                    "substrate_family": str(first.substrate_family),
                    "held_instrument": str(first.held_instrument),
                    "support_tier": str(first.support_tier),
                    "policy_id": str(policy_id),
                    "procedure_id": str(procedure_id),
                    "candidate_ids_json": json.dumps(
                        sorted(set(endpoint.candidate_id.dropna().astype(str))),
                        separators=(",", ":"),
                    ),
                    "model_ids_json": json.dumps(
                        sorted(set(endpoint.model_id.dropna().astype(str))),
                        separators=(",", ":"),
                    ),
                    "endpoint_status": "unavailable_terminal_failure",
                    "endpoint_reason": "|".join(
                        sorted(
                            set(
                                endpoint.loc[
                                    endpoint.reason_code.notna(), "reason_code"
                                ].astype(str)
                            )
                        )
                    ),
                }
            )
            continue
        record, intervals = _metric_record(
            group,
            endpoint_status="complete",
            endpoint_reason=None,
        )
        metric_rows.append(record)
        interval_rows.extend(intervals)
    metrics = pd.DataFrame(metric_rows)
    intervals = pd.DataFrame(interval_rows)

    def procedure_state(row: pd.Series) -> str:
        if str(row.endpoint_status) != "complete":
            return "unavailable_terminal_failure"
        if row.held_lcb95 >= TAU and row.loss_ucb95 <= DELTA:
            return "supports_portability"
        if row.held_ucb95 < TAU or row.loss_lcb95 > DELTA:
            return "inferior_portability"
        return "inconclusive"

    metrics["bounded_state"] = metrics.apply(procedure_state, axis=1)
    claims = domain_support.copy()
    primary = metrics[
        metrics.policy_id.astype(str).eq("PP-U-MIN")
        & metrics.procedure_id.astype(str).eq("C-SELECTED")
    ][
        [
            "domain_id",
            "endpoint_status",
            "matched_masters",
            "held_balanced_accuracy",
            "source_minus_held_balanced_accuracy",
            "held_lcb95",
            "held_ucb95",
            "loss_lcb95",
            "loss_ucb95",
        ]
    ]
    claims = claims.merge(primary, on="domain_id", how="left", validate="one_to_one")

    def state(row: pd.Series) -> str:
        if str(row.support_tier) == "unsupported_by_design":
            return "unsupported_by_design"
        if row.endpoint_status != "complete":
            return "unavailable_terminal_failure"
        if row.held_lcb95 >= TAU and row.loss_ucb95 <= DELTA:
            return "supports_portability"
        if row.held_ucb95 < TAU or row.loss_lcb95 > DELTA:
            return "inferior_portability"
        return "inconclusive"

    claims["completion_state"] = claims.apply(state, axis=1)
    confirmatory = claims[claims.support_tier.astype(str).eq("confirmatory")].copy()
    holm_rows = confirmatory.copy()
    if not holm_rows.empty:
        pvalues = intervals[
            intervals.policy_id.astype(str).eq("PP-U-MIN")
            & intervals.procedure_id.astype(str).eq("C-SELECTED")
        ].pivot(index="domain_id", columns="metric_id", values="opposite_tail_probability")
        holm_rows = holm_rows.merge(pvalues, on="domain_id", how="left")
        holm_rows["iut_p_value"] = holm_rows[["P13-M01", "P13-M02"]].max(axis=1)
        holm_rows["iut_p_value"] = holm_rows.iut_p_value.fillna(1.0)
        ordered = holm_rows.sort_values("iut_p_value", kind="stable").copy()
        total = len(ordered)
        adjusted = np.maximum.accumulate(
            np.minimum(1.0, ordered.iut_p_value.to_numpy() * np.arange(total, 0, -1))
        )
        ordered["holm_adjusted_iut_p_value"] = adjusted
        claims = claims.merge(
            ordered[["domain_id", "iut_p_value", "holm_adjusted_iut_p_value"]],
            on="domain_id",
            how="left",
        )
    substrate_rows: list[dict[str, Any]] = []
    for substrate in sorted(claims.substrate_family.astype(str).unique()):
        cells = claims[
            claims.substrate_family.astype(str).eq(substrate)
            & claims.support_tier.astype(str).eq("confirmatory")
        ]
        states = cells.completion_state.astype(str).tolist()
        if not states:
            result = "no_confirmatory_domains"
        elif all(value == "supports_portability" for value in states):
            result = "supports_portability"
        elif any(value == "inferior_portability" for value in states):
            result = "inferior_portability"
        elif any(value == "unavailable_terminal_failure" for value in states):
            result = "unavailable_terminal_failure"
        else:
            result = "inconclusive"
        substrate_rows.append(
            {
                "substrate_family": substrate,
                "confirmatory_domains": len(cells),
                "supporting_domains": states.count("supports_portability"),
                "inferior_domains": states.count("inferior_portability"),
                "inconclusive_domains": states.count("inconclusive"),
                "unavailable_domains": states.count("unavailable_terminal_failure"),
                "intersection_union_state": result,
            }
        )
    return metrics, intervals, claims, pd.DataFrame(substrate_rows)


def build_preprocessing_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    complete = metrics[metrics.endpoint_status.astype(str).eq("complete")]
    keys = [
        "domain_id",
        "station",
        "substrate_family",
        "held_instrument",
        "support_tier",
        "procedure_id",
    ]
    primary = complete[complete.policy_id.astype(str).eq("PP-U-MIN")]
    rows: list[pd.DataFrame] = []
    for policy in ("PP-U-SG", "PP-U-ARPLS"):
        sensitivity = complete[complete.policy_id.astype(str).eq(policy)]
        merged = primary.merge(
            sensitivity,
            on=keys,
            suffixes=("_min", "_sensitivity"),
            validate="one_to_one",
        )
        merged["sensitivity_policy_id"] = policy
        for metric in (
            "held_balanced_accuracy",
            "source_minus_held_balanced_accuracy",
            "held_macro_f1",
        ):
            merged[f"delta_{metric}"] = (
                merged[f"{metric}_sensitivity"] - merged[f"{metric}_min"]
            )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_class_cell_claims(
    *,
    master_views: pd.DataFrame,
    fold_endpoints: pd.DataFrame,
    domain_support: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the locked Holm correction to primary analyte-level cell claims."""

    primary_views = master_views[
        master_views.policy_id.astype(str).eq("PP-U-MIN")
        & master_views.procedure_id.astype(str).eq("C-SELECTED")
    ]
    primary_endpoints = fold_endpoints[
        fold_endpoints.policy_id.astype(str).eq("PP-U-MIN")
        & fold_endpoints.procedure_id.astype(str).eq("C-SELECTED")
    ]
    records: list[dict[str, Any]] = []
    for domain in domain_support.itertuples(index=False):
        labels = [
            item.split(":", maxsplit=1)[0]
            for item in str(domain.held_class_support).split("|")
        ]
        endpoint = primary_endpoints[
            primary_endpoints.domain_id.astype(str).eq(str(domain.domain_id))
        ]
        terminal = endpoint.empty or endpoint.status.astype(str).eq(
            "unavailable_terminal_failure"
        ).any()
        views = primary_views[
            primary_views.domain_id.astype(str).eq(str(domain.domain_id))
        ]
        held = (
            _master_predictions(views[views.prediction_role.eq("held_test")])
            if not views.empty
            else pd.DataFrame()
        )
        source_views = views[views.prediction_role.eq("matched_source_test")]
        source = _master_predictions(source_views) if not source_views.empty else pd.DataFrame()
        for label in labels:
            record: dict[str, Any] = {
                "domain_id": str(domain.domain_id),
                "station": str(domain.station),
                "substrate_family": str(domain.substrate_family),
                "held_instrument": str(domain.held_instrument),
                "target_analyte": label,
                "support_tier": str(domain.support_tier),
                "held_recall": np.nan,
                "held_recall_lcb95": np.nan,
                "held_recall_ucb95": np.nan,
                "class_source_minus_held_recall": np.nan,
                "class_loss_lcb95": np.nan,
                "class_loss_ucb95": np.nan,
                "iut_p_value": np.nan,
                "holm_adjusted_iut_p_value": np.nan,
            }
            if str(domain.support_tier) == "unsupported_by_design":
                record["completion_state"] = "unsupported_by_design"
                records.append(record)
                continue
            if terminal or held.empty or source.empty:
                record["completion_state"] = "unavailable_terminal_failure"
                records.append(record)
                continue
            held_class = held[held.true_label.astype(str).eq(label)].copy()
            source_class = source[source.true_label.astype(str).eq(label)].copy()
            paired = source_class.merge(
                held_class[["master_sample_id", "true_label", "predicted_label"]],
                on=["master_sample_id", "true_label"],
                suffixes=("_source", "_held"),
                validate="one_to_one",
            )
            if held_class.empty or paired.empty:
                record["completion_state"] = "unavailable_terminal_failure"
                records.append(record)
                continue
            held_values = held_class.predicted_label.astype(str).eq(label).to_numpy(
                dtype=float
            )
            loss_values = (
                paired.predicted_label_source.astype(str).eq(label).to_numpy(dtype=float)
                - paired.predicted_label_held.astype(str).eq(label).to_numpy(dtype=float)
            )
            seed = int(
                sha256_value(
                    {"domain_id": str(domain.domain_id), "target_analyte": label}
                )[:8],
                16,
            )
            held_interval = _mean_interval(held_values, seed=seed)
            loss_interval = _mean_interval(loss_values, seed=seed + 1)
            rng_held = np.random.default_rng(seed)
            held_bootstrap = held_values[
                rng_held.integers(
                    0,
                    len(held_values),
                    size=(BOOTSTRAP_RESAMPLES, len(held_values)),
                )
            ].mean(axis=1)
            rng_loss = np.random.default_rng(seed + 1)
            loss_bootstrap = loss_values[
                rng_loss.integers(
                    0,
                    len(loss_values),
                    size=(BOOTSTRAP_RESAMPLES, len(loss_values)),
                )
            ].mean(axis=1)
            iut_p = max(
                float(np.mean(held_bootstrap < TAU)),
                float(np.mean(loss_bootstrap > DELTA)),
            )
            if held_interval["lower_95"] >= TAU and loss_interval["upper_95"] <= DELTA:
                state = "supports_portability"
            elif held_interval["upper_95"] < TAU or loss_interval["lower_95"] > DELTA:
                state = "inferior_portability"
            else:
                state = "inconclusive"
            record.update(
                {
                    "held_masters": len(held_class),
                    "matched_masters": len(paired),
                    "held_recall": held_interval["estimate"],
                    "held_recall_lcb95": held_interval["lower_95"],
                    "held_recall_ucb95": held_interval["upper_95"],
                    "held_interval_method": held_interval["interval_method"],
                    "class_source_minus_held_recall": loss_interval["estimate"],
                    "class_loss_lcb95": loss_interval["lower_95"],
                    "class_loss_ucb95": loss_interval["upper_95"],
                    "class_loss_interval_method": loss_interval["interval_method"],
                    "iut_p_value": iut_p,
                    "completion_state": state,
                }
            )
            records.append(record)
    result = pd.DataFrame(records)
    confirmatory = result[
        result.support_tier.astype(str).eq("confirmatory")
    ].copy()
    if not confirmatory.empty:
        confirmatory["multiplicity_p_value"] = confirmatory.iut_p_value.fillna(1.0)
        ordered = confirmatory.sort_values("multiplicity_p_value", kind="stable")
        total = len(ordered)
        adjusted = np.maximum.accumulate(
            np.minimum(
                1.0,
                ordered.multiplicity_p_value.to_numpy()
                * np.arange(total, 0, -1),
            )
        )
        result.loc[ordered.index, "holm_adjusted_iut_p_value"] = adjusted
    result["holm_positive_claim"] = (
        result.completion_state.astype(str).eq("supports_portability")
        & result.holm_adjusted_iut_p_value.le(0.05)
    )
    return result.sort_values(
        ["domain_id", "target_analyte"], kind="stable"
    ).reset_index(drop=True)


def build_procedure_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics.policy_id.astype(str).eq("PP-U-MIN")
        & metrics.support_tier.astype(str).eq("confirmatory")
    ]
    procedures = sorted(primary.procedure_id.astype(str).unique())
    complete_sets = {
        procedure: set(
            primary[
                primary.procedure_id.astype(str).eq(procedure)
                & primary.endpoint_status.astype(str).eq("complete")
            ].domain_id.astype(str)
        )
        for procedure in procedures
    }
    common = set.intersection(*complete_sets.values()) if complete_sets else set()
    rows = []
    declared_domains = primary.domain_id.astype(str).nunique()
    for procedure in procedures:
        group = primary[primary.procedure_id.astype(str).eq(procedure)]
        successful = group[group.endpoint_status.astype(str).eq("complete")]
        common_group = successful[successful.domain_id.astype(str).isin(common)]
        chance_values = group.held_balanced_accuracy.fillna(CHANCE_BA)
        rows.append(
            {
                "procedure_id": procedure,
                "declared_confirmatory_domains": declared_domains,
                "successful_domains": len(successful),
                "common_successful_domains": len(common_group),
                "mean_held_ba_successful": successful.held_balanced_accuracy.mean(),
                "mean_held_ba_common": common_group.held_balanced_accuracy.mean(),
                "mean_held_ba_chance_imputed": chance_values.mean(),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and "C-SELECTED" in set(result.procedure_id):
        selected = result.set_index("procedure_id").loc["C-SELECTED"]
        result["common_minus_selected"] = (
            result.mean_held_ba_common - selected.mean_held_ba_common
        )
        result["chance_imputed_minus_selected"] = (
            result.mean_held_ba_chance_imputed
            - selected.mean_held_ba_chance_imputed
        )
        result["direction_consistent_under_chance_sensitivity"] = (
            np.sign(result.common_minus_selected.fillna(0))
            == np.sign(result.chance_imputed_minus_selected.fillna(0))
        )
        result["positive_comparison_claim_allowed"] = (
            result.common_minus_selected.gt(0)
            & result.chance_imputed_minus_selected.gt(0)
        )
    return result


def _mean_interval(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return _interval(np.nan, np.asarray([]), np.asarray([]))
    rng = np.random.default_rng(seed)
    bootstrap = finite[
        rng.integers(0, len(finite), size=(BOOTSTRAP_RESAMPLES, len(finite)))
    ].mean(axis=1)
    return _interval(float(finite.mean()), bootstrap, _jackknife_values(finite))


def _view_probability(frame: pd.DataFrame, true_label: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    vocabulary = tuple(json.loads(str(frame.class_vocabulary.iloc[0])))
    if true_label not in vocabulary:
        return pd.Series(dtype=float)
    index = vocabulary.index(true_label)
    return frame.probabilities.map(lambda value: float(json.loads(str(value))[index]))


def _representation_view_lookup(
    manifest: pd.DataFrame, intensity: np.ndarray
) -> dict[tuple[str, str, str, str], np.ndarray]:
    frame = manifest.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    result: dict[tuple[str, str, str, str], np.ndarray] = {}
    for key, group in frame.groupby(
        ["station", "master_sample_id", "sensor_family", "instrument"], sort=True
    ):
        result[tuple(str(value) for value in key)] = intensity[
            group.row_index.to_numpy(dtype=int)
        ].mean(axis=0)
    return result


def build_crossover_effects(
    *,
    master_views: pd.DataFrame,
    crossover_support: pd.DataFrame,
    manifest: pd.DataFrame,
    min_intensity: np.ndarray,
) -> pd.DataFrame:
    """Calculate frozen two-substrate by two-instrument interaction contrasts."""

    prediction_views = master_views[
        master_views.policy_id.astype(str).eq("PP-U-MIN")
        & master_views.prediction_role.astype(str).eq("held_test")
    ]
    representation = _representation_view_lookup(manifest, min_intensity)
    records: list[dict[str, Any]] = []
    for block in crossover_support.itertuples(index=False):
        base = {
            "crossover_block_id": str(block.crossover_block_id),
            "station": str(block.station),
            "target_analyte": str(block.target_analyte),
            "substrate_a": str(block.substrate_a),
            "substrate_b": str(block.substrate_b),
            "instrument_a": str(block.instrument_a),
            "instrument_b": str(block.instrument_b),
            "physical_masters": int(block.physical_masters),
            "support_tier": str(block.support_tier),
        }
        block_manifest = manifest[
            manifest.station.astype(str).eq(str(block.station))
            & manifest.target_analyte.astype(str).eq(str(block.target_analyte))
        ]
        required = {
            (str(block.substrate_a), str(block.instrument_a)),
            (str(block.substrate_a), str(block.instrument_b)),
            (str(block.substrate_b), str(block.instrument_a)),
            (str(block.substrate_b), str(block.instrument_b)),
        }
        master_cells = {
            str(master): set(
                zip(
                    frame.sensor_family.astype(str),
                    frame.instrument.astype(str),
                    strict=False,
                )
            )
            for master, frame in block_manifest.groupby("master_sample_id", sort=True)
        }
        masters = [master for master, cells in master_cells.items() if required <= cells]
        distance_differences = []
        for master in masters:
            vector_aa = representation[
                (str(block.station), master, str(block.substrate_a), str(block.instrument_a))
            ]
            vector_ab = representation[
                (str(block.station), master, str(block.substrate_a), str(block.instrument_b))
            ]
            vector_ba = representation[
                (str(block.station), master, str(block.substrate_b), str(block.instrument_a))
            ]
            vector_bb = representation[
                (str(block.station), master, str(block.substrate_b), str(block.instrument_b))
            ]

            def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                return 1 - float(left @ right) / denominator if denominator else np.nan

            distance_differences.append(
                cosine_distance(vector_ba, vector_bb)
                - cosine_distance(vector_aa, vector_ab)
            )
        if len(distance_differences) == 1:
            distance_interval = {
                "estimate": float(distance_differences[0]),
                "lower_95": np.nan,
                "upper_95": np.nan,
                "interval_method": "descriptive_singleton",
            }
        else:
            distance_interval = _mean_interval(
                np.asarray(distance_differences),
                seed=int(sha256_value({**base, "metric": "distance"})[:8], 16),
            )
        for procedure_id in sorted(prediction_views.procedure_id.astype(str).unique()):
            cells: dict[tuple[str, str], pd.DataFrame] = {}
            for substrate, instrument in required:
                cells[(substrate, instrument)] = prediction_views[
                    prediction_views.station.astype(str).eq(str(block.station))
                    & prediction_views.substrate_family.astype(str).eq(substrate)
                    & prediction_views.instrument.astype(str).eq(instrument)
                    & prediction_views.true_label.astype(str).eq(str(block.target_analyte))
                    & prediction_views.procedure_id.astype(str).eq(procedure_id)
                    & prediction_views.master_sample_id.astype(str).isin(masters)
                ].copy()
            available = all(
                set(cell.master_sample_id.astype(str)) == set(masters)
                and cell.outer_repeat_predictions_min.eq(5).all()
                for cell in cells.values()
            ) and bool(masters)
            record = {
                **base,
                "procedure_id": procedure_id,
                "predictive_status": "complete" if available else "unavailable_by_design",
                "predictive_masters": len(masters) if available else 0,
                "correctness_interaction": np.nan,
                "correctness_interaction_lower_95": np.nan,
                "correctness_interaction_upper_95": np.nan,
                "correctness_interaction_interval_method": "unavailable_by_design",
                "true_probability_interaction": np.nan,
                "true_probability_interaction_lower_95": np.nan,
                "true_probability_interaction_upper_95": np.nan,
                "true_probability_interaction_interval_method": (
                    "unavailable_by_design"
                ),
                "representation_distance_difference": distance_interval["estimate"],
                "representation_distance_lower_95": distance_interval["lower_95"],
                "representation_distance_upper_95": distance_interval["upper_95"],
                "representation_distance_interval_method": distance_interval[
                    "interval_method"
                ],
            }
            if available:
                master_order = sorted(masters)
                indexed = {
                    key: value.set_index(value.master_sample_id.astype(str)).loc[
                        master_order
                    ]
                    for key, value in cells.items()
                }
                sa, sb = str(block.substrate_a), str(block.substrate_b)
                ia, ib = str(block.instrument_a), str(block.instrument_b)
                correctness = {
                    key: frame.predicted_label.astype(str)
                    .eq(str(block.target_analyte))
                    .astype(float)
                    for key, frame in indexed.items()
                }
                probability = {
                    key: _view_probability(frame, str(block.target_analyte))
                    for key, frame in indexed.items()
                }
                correctness_did = (
                    correctness[(sb, ib)]
                    - correctness[(sa, ib)]
                    - correctness[(sb, ia)]
                    + correctness[(sa, ia)]
                ).to_numpy()
                probability_did = (
                    probability[(sb, ib)].to_numpy()
                    - probability[(sa, ib)].to_numpy()
                    - probability[(sb, ia)].to_numpy()
                    + probability[(sa, ia)].to_numpy()
                )
                seed = int(
                    sha256_value({**base, "procedure_id": procedure_id})[:8], 16
                )
                if len(correctness_did) == 1:
                    correct_interval = {
                        "estimate": float(correctness_did[0]),
                        "lower_95": np.nan,
                        "upper_95": np.nan,
                        "interval_method": "descriptive_singleton",
                    }
                    probability_interval = {
                        "estimate": float(probability_did[0]),
                        "lower_95": np.nan,
                        "upper_95": np.nan,
                        "interval_method": "descriptive_singleton",
                    }
                else:
                    correct_interval = _mean_interval(correctness_did, seed=seed)
                    probability_interval = _mean_interval(
                        probability_did, seed=seed + 1
                    )
                for prefix, interval in (
                    ("correctness_interaction", correct_interval),
                    ("true_probability_interaction", probability_interval),
                ):
                    record[prefix] = interval["estimate"]
                    record[f"{prefix}_lower_95"] = interval["lower_95"]
                    record[f"{prefix}_upper_95"] = interval["upper_95"]
                    record[f"{prefix}_interval_method"] = interval["interval_method"]
            records.append(record)
    return pd.DataFrame(records)


def _field_view_status(values: pd.Series) -> str:
    observed = sorted(set(values.dropna().astype(str).str.strip()) - {""})
    if not observed:
        return "missing"
    if len(observed) > 1:
        return "conflict"
    return observed[0]


def build_field_log_results(
    *,
    master_views: pd.DataFrame,
    manifest: pd.DataFrame,
    domain_support: pd.DataFrame,
) -> pd.DataFrame:
    prediction = master_views[
        master_views.policy_id.astype(str).eq("PP-U-MIN")
        & master_views.prediction_role.astype(str).eq("held_test")
        & master_views.outer_repeat_predictions_min.eq(5)
    ].copy()
    eligible = domain_support[
        domain_support.support_tier.astype(str).isin(
            ["confirmatory", "exploratory_low_support"]
        )
    ][["station", "substrate_family", "held_instrument"]].rename(
        columns={"held_instrument": "instrument"}
    )
    eligible_manifest = manifest.merge(
        eligible,
        left_on=["station", "sensor_family", "instrument"],
        right_on=["station", "substrate_family", "instrument"],
        how="inner",
        validate="many_to_one",
    )
    logs = (
        eligible_manifest.groupby(
            ["station", "master_sample_id", "substrate_family", "instrument"],
            as_index=False,
        )
        .agg(
            target_analyte=("target_analyte", "first"),
            field_log_status=("target_detected_log", _field_view_status),
        )
    )
    procedures = sorted(master_views.procedure_id.astype(str).unique())
    records: list[dict[str, Any]] = []
    for substrate, group in logs.groupby("substrate_family", sort=True):
        for endpoint_type, mask, success_code in (
            (
                "nonblank_detection",
                group.target_analyte.astype(str).ne("blank"),
                "Y",
            ),
            (
                "blank_specificity",
                group.target_analyte.astype(str).eq("blank"),
                "N",
            ),
        ):
            subset = group[mask].copy()
            if subset.empty:
                continue
            definite = subset.field_log_status.isin(["Y", "N"])
            missing = subset.field_log_status.eq("missing")
            ambiguous = subset.field_log_status.isin(["M", "conflict"])
            success = subset.loc[definite, "field_log_status"].eq(success_code)
            total_for_bounds = int(definite.sum() + missing.sum())
            successes = int(success.sum())
            complete_case = float(success.mean()) if len(success) else np.nan
            worst = successes / total_for_bounds if total_for_bounds else np.nan
            best = (
                (successes + int(missing.sum())) / total_for_bounds
                if total_for_bounds
                else np.nan
            )
            for procedure in procedures:
                model = prediction[
                    prediction.procedure_id.astype(str).eq(procedure)
                    & prediction.substrate_family.astype(str).eq(str(substrate))
                ][
                    [
                        "station",
                        "master_sample_id",
                        "substrate_family",
                        "instrument",
                        "true_label",
                        "predicted_label",
                    ]
                ].rename(columns={"true_label": "target_analyte"})
                evaluated = subset.merge(
                    model,
                    on=[
                        "station",
                        "master_sample_id",
                        "substrate_family",
                        "instrument",
                        "target_analyte",
                    ],
                    how="inner",
                    validate="one_to_one",
                )
                evaluated = evaluated[
                    evaluated.field_log_status.isin(["Y", "N"])
                ].copy()
                field_success = evaluated.field_log_status.eq(success_code)
                model_correct = evaluated.predicted_label.astype(str).eq(
                    evaluated.target_analyte.astype(str)
                )
                agreement = (
                    float(field_success.eq(model_correct).mean())
                    if len(evaluated)
                    else np.nan
                )
                records.append(
                    {
                        "substrate_family": str(substrate),
                        "procedure_id": str(procedure),
                        "endpoint_type": endpoint_type,
                        "eligible_master_views": len(subset),
                        "definite_views": int(definite.sum()),
                        "successful_views": successes,
                        "missing_views": int(missing.sum()),
                        "ambiguous_or_conflicting_views": int(ambiguous.sum()),
                        "complete_case_estimate": complete_case,
                        "worst_case_missing_bound": worst,
                        "best_case_missing_bound": best,
                        "model_prediction_views": len(
                            subset.merge(
                                model,
                                on=[
                                    "station",
                                    "master_sample_id",
                                    "substrate_family",
                                    "instrument",
                                    "target_analyte",
                                ],
                                how="inner",
                            )
                        ),
                        "model_field_agreement_views": len(evaluated),
                        "model_field_agreement": agreement,
                    }
                )
    return pd.DataFrame(records)


def build_failure_table(
    *, fit_status: pd.DataFrame, fold_endpoints: pd.DataFrame
) -> pd.DataFrame:
    fit_failures = (
        fit_status[~fit_status.status.astype(str).eq("complete")]
        .groupby(
            ["domain_id", "policy_id", "procedure_id", "stage", "status", "reason_code"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    endpoint_failures = (
        fold_endpoints[
            fold_endpoints.status.astype(str).eq("unavailable_terminal_failure")
        ]
        .groupby(
            ["domain_id", "policy_id", "procedure_id", "status", "reason_code"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    fit_failures["record_type"] = "fit"
    endpoint_failures["record_type"] = "fold_endpoint"
    endpoint_failures["stage"] = "outer_endpoint"
    return pd.concat([fit_failures, endpoint_failures], ignore_index=True)


def build_p13_aggregate_tables(
    *,
    predictions: pd.DataFrame,
    fit_status: pd.DataFrame,
    fold_endpoints: pd.DataFrame,
    domain_support: pd.DataFrame,
    crossover_support: pd.DataFrame,
    manifest: pd.DataFrame,
    min_intensity: np.ndarray,
) -> P13AggregateTables:
    master_views = build_master_view_predictions(predictions)
    metrics, intervals, claims, substrate_claims = build_domain_results(
        master_views=master_views,
        fold_endpoints=fold_endpoints,
        domain_support=domain_support,
    )
    return P13AggregateTables(
        master_view_predictions=master_views,
        domain_metrics=metrics,
        interval_table=intervals,
        domain_claims=claims,
        class_cell_claims=build_class_cell_claims(
            master_views=master_views,
            fold_endpoints=fold_endpoints,
            domain_support=domain_support,
        ),
        substrate_claims=substrate_claims,
        preprocessing_sensitivity=build_preprocessing_sensitivity(metrics),
        procedure_comparison=build_procedure_comparison(metrics),
        crossover_effects=build_crossover_effects(
            master_views=master_views,
            crossover_support=crossover_support,
            manifest=manifest,
            min_intensity=min_intensity,
        ),
        field_log_results=build_field_log_results(
            master_views=master_views,
            manifest=manifest,
            domain_support=domain_support,
        ),
        failure_table=build_failure_table(
            fit_status=fit_status, fold_endpoints=fold_endpoints
        ),
    )
