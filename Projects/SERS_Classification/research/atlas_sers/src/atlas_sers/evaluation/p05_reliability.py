"""Strict, anonymous P05 reliability-diagram diagnostics.

This module is pure and in-memory: it never reads a file, touches torch, fits,
infers, calibrates or selects. Callers authenticate the frozen P05 three-seed
mean calibrated ensemble predictions together with the allowlisted public
context metrics before handing them in. Only pooled station-level reliability
bins and summary policy text are emitted; private identifiers such as
``observation_uid``, ``master_sample_id``, ``context_id``, ``true_label``,
``predicted_label``, per-row probabilities, class vocabularies and source paths
are never copied. Context identity is replaced by the same anonymous 1-based
``point_index`` derived from the sorted private context IDs that
``p05_public_metrics`` uses.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import (
    expected_calibration_error,
    instrument_balanced_master_probabilities,
)
from atlas_sers.evaluation.p05_public_metrics import (
    P05_PUBLIC_AGGREGATIONS,
    P05_PUBLIC_MODELS,
    P05_PUBLIC_STATIONS,
    PHASE_BY_EXPERIMENT,
)

__all__ = ["P05ReliabilityError", "build_reliability"]

RELIABILITY_BINS = 10
ECE_ATOL = 1e-12
PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
_ENSEMBLE_REQUIRED = (
    "context_id",
    "experiment_id",
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "true_label",
    "predicted_label",
    "class_vocabulary",
    *PROBABILITY_COLUMNS,
)
_STRATEGY_REQUIRED = (
    "point_index",
    "station",
    "phase",
    "domain",
    "held_instrument",
    "model_id",
    "aggregation_id",
    "ece",
    "observations",
    "physical_masters",
    "observed_class_count",
)
_RELIABILITY_BIN_COLUMNS = (
    "station",
    "phase",
    "model_id",
    "aggregation_id",
    "bin_index",
    "count",
    "mean_confidence",
    "observed_accuracy",
    "signed_gap",
    "bin_weight",
)
_RELIABILITY_SUMMARY_COLUMNS = (
    "station",
    "phase",
    "model_id",
    "aggregation_id",
    "total_appearances",
    "contributing_contexts",
    "pooled_reliability_ece",
    "mean_context_ece",
    "pooled_vs_mean_context_policy",
    "independence_policy",
    "endpoint_policy",
    "diagnostic_policy",
)
POOLED_VS_MEAN_CONTEXT_POLICY = (
    "pooled descriptive reliability diagram differs from mean per-context ECE"
)
INDEPENDENCE_POLICY = (
    "repeated split appearances are not independent samples; no confidence interval "
    "or significance test is computed"
)
ENDPOINT_POLICY = (
    "pooled reliability ECE is a descriptive diagnostic, not the registered primary endpoint"
)
DIAGNOSTIC_POLICY = (
    "reliability curves are diagnostics only; no calibration, training, model "
    "selection, or confidence interval is performed"
)


class P05ReliabilityError(ValueError):
    """Raised when the ensemble predictions or public context metrics are invalid."""


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise P05ReliabilityError(code)
    return value


def _instrument(value: Any, code: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise P05ReliabilityError(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA:
        raise P05ReliabilityError(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
        raise P05ReliabilityError(code)
    if isinstance(value, str):
        if value and value == value.strip():
            try:
                number = int(value)
            except ValueError:
                raise P05ReliabilityError(code) from None
            if str(number) == value:
                return number
    raise P05ReliabilityError(code)


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(float(value)))
    return False


def _probability(value: Any, code: str) -> float:
    if isinstance(value, (bool, np.bool_)) or _missing(value):
        raise P05ReliabilityError(code)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise P05ReliabilityError(code) from None
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise P05ReliabilityError(code)
    return number


def _classes(value: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        raise P05ReliabilityError("class_vocabulary_invalid") from None
    if isinstance(parsed, (str, bytes)):
        raise P05ReliabilityError("class_vocabulary_invalid")
    try:
        items = list(parsed)
    except TypeError:
        raise P05ReliabilityError("class_vocabulary_invalid") from None
    classes = tuple(_text(item, "class_vocabulary_invalid") for item in items)
    if len(classes) != 3 or len(set(classes)) != 3 or classes != tuple(sorted(classes)):
        raise P05ReliabilityError("class_vocabulary_invalid")
    return classes


def _validate_ensemble(ensemble: Any) -> pd.DataFrame:
    if not isinstance(ensemble, pd.DataFrame):
        raise P05ReliabilityError("ensemble_predictions_malformed")
    if ensemble.empty:
        raise P05ReliabilityError("ensemble_predictions_empty")
    missing = [column for column in _ENSEMBLE_REQUIRED if column not in ensemble.columns]
    if missing:
        raise P05ReliabilityError("ensemble_columns_missing:" + ",".join(missing))
    frame = ensemble.copy()
    for column in (
        "context_id",
        "experiment_id",
        "domain",
        "model_id",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "true_label",
        "predicted_label",
    ):
        for value in frame[column]:
            _text(value, f"ensemble_{column}_invalid")
    for value in frame["station"]:
        if not isinstance(value, str) or value not in P05_PUBLIC_STATIONS:
            raise P05ReliabilityError("ensemble_station_invalid")
    for value in frame["held_instrument"]:
        _instrument(value, "ensemble_held_instrument_invalid")
    if not frame["model_id"].isin(P05_PUBLIC_MODELS).all():
        raise P05ReliabilityError("ensemble_model_unknown")
    phase = frame["experiment_id"].map(PHASE_BY_EXPERIMENT)
    if phase.isna().any():
        raise P05ReliabilityError("ensemble_phase_unknown")
    frame["phase"] = phase
    if any(
        isinstance(value, (bool, np.bool_))
        for column in PROBABILITY_COLUMNS
        for value in frame[column]
    ):
        raise P05ReliabilityError("ensemble_probability_invalid")
    try:
        values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise P05ReliabilityError("ensemble_probability_invalid") from None
    if values.shape != (len(frame), len(PROBABILITY_COLUMNS)):
        raise P05ReliabilityError("ensemble_probability_shape")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise P05ReliabilityError("ensemble_probability_invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise P05ReliabilityError("ensemble_probability_not_normalized")
    classes_list = [_classes(value) for value in frame["class_vocabulary"]]
    frame["_classes"] = classes_list
    predicted = np.argmax(values, axis=1)
    truth = frame["true_label"].astype(str).tolist()
    declared = frame["predicted_label"].astype(str).tolist()
    for position in range(len(frame)):
        class_index = {label: index for index, label in enumerate(classes_list[position])}
        if truth[position] not in class_index:
            raise P05ReliabilityError("ensemble_true_label_outside_vocabulary")
        if declared[position] not in class_index:
            raise P05ReliabilityError("ensemble_predicted_label_outside_vocabulary")
        if class_index[declared[position]] != int(predicted[position]):
            raise P05ReliabilityError("ensemble_predicted_label_mismatch")
    if frame.duplicated(["context_id", "model_id", "observation_uid"]).any():
        raise P05ReliabilityError("ensemble_duplicate_appearance")
    if not frame.groupby("master_sample_id").true_label.nunique().eq(1).all():
        raise P05ReliabilityError("ensemble_master_label_conflict")
    for _, cell in frame.groupby("context_id", sort=False):
        if len(set(cell["_classes"])) != 1:
            raise P05ReliabilityError("ensemble_context_vocabulary_conflict")
        if set(cell["model_id"]) != set(P05_PUBLIC_MODELS):
            raise P05ReliabilityError("ensemble_context_model_incomplete")
        for column in ("station", "phase", "domain", "held_instrument"):
            if cell[column].nunique(dropna=False) != 1:
                raise P05ReliabilityError("ensemble_context_metadata_conflict")
        coordinates = None
        for model in P05_PUBLIC_MODELS:
            current = (
                cell.loc[cell.model_id.eq(model), [
                    "observation_uid", "master_sample_id", "instrument", "true_label"
                ]]
                .sort_values("observation_uid", kind="stable")
                .reset_index(drop=True)
            )
            if coordinates is not None and not current.equals(coordinates):
                raise P05ReliabilityError("ensemble_model_coordinates_mismatch")
            coordinates = current
    return frame


def _validate_strategy_contexts(strategy_contexts: Any) -> pd.DataFrame:
    if not isinstance(strategy_contexts, pd.DataFrame):
        raise P05ReliabilityError("strategy_contexts_malformed")
    if strategy_contexts.empty:
        raise P05ReliabilityError("strategy_contexts_empty")
    missing = [column for column in _STRATEGY_REQUIRED if column not in strategy_contexts.columns]
    if missing:
        raise P05ReliabilityError("strategy_columns_missing:" + ",".join(missing))
    frame = strategy_contexts.copy()
    for value in frame["domain"]:
        _text(value, "strategy_domain_invalid")
    for value in frame["station"]:
        if not isinstance(value, str) or value not in P05_PUBLIC_STATIONS:
            raise P05ReliabilityError("strategy_station_invalid")
    for value in frame["held_instrument"]:
        _instrument(value, "strategy_held_instrument_invalid")
    if not frame["model_id"].isin(P05_PUBLIC_MODELS).all():
        raise P05ReliabilityError("strategy_model_unknown")
    if not frame["aggregation_id"].isin(P05_PUBLIC_AGGREGATIONS).all():
        raise P05ReliabilityError("strategy_aggregation_unknown")
    if not frame["phase"].isin(tuple(PHASE_BY_EXPERIMENT.values())).all():
        raise P05ReliabilityError("strategy_phase_unknown")
    frame["point_index"] = [
        _integer(value, "strategy_point_index_invalid") for value in frame["point_index"]
    ]
    if (frame["point_index"] <= 0).any():
        raise P05ReliabilityError("strategy_point_index_invalid")
    for column in ("observations", "physical_masters", "observed_class_count"):
        frame[column] = [_integer(value, f"strategy_{column}_invalid") for value in frame[column]]
    frame["ece"] = [_probability(value, "strategy_ece_invalid") for value in frame["ece"]]
    if (frame["observations"] <= 0).any() or (frame["physical_masters"] <= 0).any():
        raise P05ReliabilityError("strategy_count_nonpositive")
    if (frame["physical_masters"] > frame["observations"]).any():
        raise P05ReliabilityError("strategy_master_exceeds_observations")
    if not frame["observed_class_count"].between(1, 3).all():
        raise P05ReliabilityError("strategy_observed_class_count_invalid")
    m06 = frame["aggregation_id"].eq("M06")
    if (
        frame.loc[m06, "observations"].to_numpy()
        != frame.loc[m06, "physical_masters"].to_numpy()
    ).any():
        raise P05ReliabilityError("strategy_m06_count_mismatch")
    if frame.duplicated(["point_index", "model_id", "aggregation_id"]).any():
        raise P05ReliabilityError("strategy_duplicate_key")
    expected = {
        (model, aggregation)
        for model in P05_PUBLIC_MODELS
        for aggregation in P05_PUBLIC_AGGREGATIONS
    }
    for _, cell in frame.groupby("point_index", sort=False):
        if set(zip(cell["model_id"], cell["aggregation_id"], strict=True)) != expected:
            raise P05ReliabilityError("strategy_context_incomplete")
        for column in ("station", "phase", "domain", "held_instrument"):
            if cell[column].nunique(dropna=False) != 1:
                raise P05ReliabilityError("strategy_context_metadata_conflict")
        for _, aggregation_cell in cell.groupby("aggregation_id", sort=False):
            if aggregation_cell["observed_class_count"].nunique() != 1:
                raise P05ReliabilityError("strategy_class_count_conflict")
            if (
                aggregation_cell["observations"].nunique() != 1
                or aggregation_cell["physical_masters"].nunique() != 1
            ):
                raise P05ReliabilityError("strategy_count_conflict")
    return frame


def _context_index(ensemble: pd.DataFrame) -> dict[str, int]:
    identifiers = sorted({str(value) for value in ensemble["context_id"].tolist()})
    if not identifiers:
        raise P05ReliabilityError("contexts_empty")
    return {identifier: position for position, identifier in enumerate(identifiers, start=1)}


def _strategy_lookup(strategy_contexts: pd.DataFrame, index: dict[str, int]) -> dict[Any, Any]:
    observed = {int(value) for value in strategy_contexts["point_index"].tolist()}
    if observed != set(index.values()):
        raise P05ReliabilityError("strategy_context_index_mismatch")
    expected_rows = len(index) * len(P05_PUBLIC_MODELS) * len(P05_PUBLIC_AGGREGATIONS)
    if len(strategy_contexts) != expected_rows:
        raise P05ReliabilityError("strategy_row_count_mismatch")
    lookup: dict[Any, Any] = {}
    for row in strategy_contexts.itertuples(index=False):
        lookup[(int(row.point_index), str(row.model_id), str(row.aggregation_id))] = row
    return lookup


def _validate_alignment(
    ensemble: pd.DataFrame, lookup: dict[Any, Any], index: dict[str, int]
) -> None:
    metadata = ensemble.groupby("context_id", sort=False).agg(
        station=("station", "first"),
        phase=("phase", "first"),
        domain=("domain", "first"),
        held_instrument=("held_instrument", "first"),
    )
    for context_id, point_index in index.items():
        row_meta = metadata.loc[context_id]
        for model_id in P05_PUBLIC_MODELS:
            for aggregation_id in P05_PUBLIC_AGGREGATIONS:
                strategy = lookup.get((point_index, model_id, aggregation_id))
                if strategy is None:
                    raise P05ReliabilityError("strategy_key_missing")
                for column in ("station", "phase", "domain", "held_instrument"):
                    if str(getattr(strategy, column)) != str(row_meta[column]):
                        raise P05ReliabilityError("strategy_metadata_mismatch")


def _check_context(
    lookup: dict[Any, Any],
    point_index: int,
    model_id: str,
    aggregation_id: str,
    observations: int,
    masters: int,
    classes: int,
    ece: float,
) -> None:
    row = lookup.get((point_index, model_id, aggregation_id))
    if row is None:
        raise P05ReliabilityError("strategy_key_missing")
    if int(row.observations) != int(observations):
        raise P05ReliabilityError("reliability_observation_count_mismatch")
    if int(row.physical_masters) != int(masters):
        raise P05ReliabilityError("reliability_master_count_mismatch")
    if int(row.observed_class_count) != int(classes):
        raise P05ReliabilityError("reliability_class_count_mismatch")
    if not np.isclose(float(row.ece), float(ece), rtol=0.0, atol=ECE_ATOL):
        raise P05ReliabilityError("reliability_ece_mismatch")


def _context_records(
    ensemble: pd.DataFrame, lookup: dict[Any, Any], index: dict[str, int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context_id, cell in ensemble.groupby("context_id", sort=True):
        point_index = index[str(context_id)]
        classes = cell["_classes"].iloc[0]
        class_index = {label: position for position, label in enumerate(classes)}
        station = str(cell["station"].iloc[0])
        phase = str(cell["phase"].iloc[0])
        for model_id, model_cell in cell.groupby("model_id", sort=True):
            values = model_cell[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            truth = model_cell["true_label"].astype(str).to_numpy()
            indices = np.asarray([class_index[label] for label in truth], dtype=int)
            # Match classification_metrics: normalize each row before ECE.
            # Retain the original values for the existing M06 helper, which
            # performs its own row normalization before instrument averaging.
            spectrum_values = values / values.sum(axis=1, keepdims=True)
            spectrum_ece = expected_calibration_error(
                spectrum_values, indices, bins=RELIABILITY_BINS
            )
            _check_context(
                lookup,
                point_index,
                str(model_id),
                "M01",
                len(model_cell),
                model_cell["master_sample_id"].astype(str).nunique(),
                len(set(truth.tolist())),
                spectrum_ece,
            )
            records.append(
                {
                    "station": station,
                    "phase": phase,
                    "model_id": str(model_id),
                    "aggregation_id": "M01",
                    "values": spectrum_values,
                    "indices": indices,
                    "context_ece": spectrum_ece,
                }
            )
            master = instrument_balanced_master_probabilities(
                probabilities=values,
                true_labels=truth,
                master_ids=model_cell["master_sample_id"].astype(str).to_numpy(),
                instruments=model_cell["instrument"].astype(str).to_numpy(),
                class_vocabulary=list(classes),
            )
            master_values = np.asarray(master["probabilities"].tolist(), dtype=float)
            master_values = master_values / master_values.sum(axis=1, keepdims=True)
            master_truth = master["true_label"].astype(str).to_numpy()
            master_indices = np.asarray([class_index[label] for label in master_truth], dtype=int)
            master_ece = expected_calibration_error(
                master_values, master_indices, bins=RELIABILITY_BINS
            )
            _check_context(
                lookup,
                point_index,
                str(model_id),
                "M06",
                len(master),
                len(master),
                len(set(master_truth.tolist())),
                master_ece,
            )
            records.append(
                {
                    "station": station,
                    "phase": phase,
                    "model_id": str(model_id),
                    "aggregation_id": "M06",
                    "values": master_values,
                    "indices": master_indices,
                    "context_ece": master_ece,
                }
            )
    return records


def _reliability_bins(
    values: np.ndarray, indices: np.ndarray
) -> tuple[list[dict[str, Any]], float]:
    total = len(values)
    if total == 0:
        raise P05ReliabilityError("reliability_group_empty")
    predicted = values.argmax(axis=1)
    confidence = values.max(axis=1)
    order = np.argsort(confidence, kind="stable")
    records: list[dict[str, Any]] = []
    for bin_index, members in enumerate(
        np.array_split(order, min(RELIABILITY_BINS, total)), start=1
    ):
        count = int(len(members))
        mean_confidence = float(np.mean(confidence[members]))
        observed_accuracy = float(np.mean(predicted[members] == indices[members]))
        signed_gap = observed_accuracy - mean_confidence
        records.append(
            {
                "bin_index": bin_index,
                "count": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "signed_gap": signed_gap,
                "bin_weight": count / total,
            }
        )
    ece = float(sum(record["bin_weight"] * abs(record["signed_gap"]) for record in records))
    return records, ece


def _build_tables(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["station"],
            record["phase"],
            record["model_id"],
            record["aggregation_id"],
        )
        groups.setdefault(key, []).append(record)
    bin_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        station, phase, model_id, aggregation_id = key
        members = groups[key]
        values = np.concatenate([member["values"] for member in members], axis=0)
        indices = np.concatenate([member["indices"] for member in members], axis=0)
        bins, pooled_ece = _reliability_bins(values, indices)
        for record in bins:
            bin_rows.append(
                {
                    "station": station,
                    "phase": phase,
                    "model_id": model_id,
                    "aggregation_id": aggregation_id,
                    **record,
                }
            )
        context_ece = [member["context_ece"] for member in members]
        summary_rows.append(
            {
                "station": station,
                "phase": phase,
                "model_id": model_id,
                "aggregation_id": aggregation_id,
                "total_appearances": int(len(values)),
                "contributing_contexts": len(members),
                "pooled_reliability_ece": pooled_ece,
                "mean_context_ece": float(np.mean(context_ece)),
                "pooled_vs_mean_context_policy": POOLED_VS_MEAN_CONTEXT_POLICY,
                "independence_policy": INDEPENDENCE_POLICY,
                "endpoint_policy": ENDPOINT_POLICY,
                "diagnostic_policy": DIAGNOSTIC_POLICY,
            }
        )
    return bin_rows, summary_rows


def build_reliability(*, ensemble_predictions, strategy_contexts) -> dict[str, pd.DataFrame]:
    """Build anonymous pooled P05 reliability bins and summary policy tables."""
    ensemble = _validate_ensemble(ensemble_predictions)
    contexts = _validate_strategy_contexts(strategy_contexts)
    index = _context_index(ensemble)
    lookup = _strategy_lookup(contexts, index)
    _validate_alignment(ensemble, lookup, index)
    records = _context_records(ensemble, lookup, index)
    bin_rows, summary_rows = _build_tables(records)
    return {
        "reliability_bins": pd.DataFrame(bin_rows, columns=list(_RELIABILITY_BIN_COLUMNS)),
        "reliability_summary": pd.DataFrame(
            summary_rows, columns=list(_RELIABILITY_SUMMARY_COLUMNS)
        ),
    }
