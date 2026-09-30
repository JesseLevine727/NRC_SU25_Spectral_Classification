"""Deterministic frozen-panel descriptive metrics for P06/P11 (T032)."""

from __future__ import annotations

import json
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import classification_metrics

ENDPOINTS = ("M01", "M06")
MODEL_IDS = (
    "D0-M",
    "D3",
    "P05-SELECTED",
    "D0-ERM",
    "C-SELECTED",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
    "C-RBF-SVM",
)
STATIONS = ("cwa", "pills", "surfaces")
SCOPES = ("full_support", "primary_common")
SELECTED_MODELS = ("P05-SELECTED", "C-SELECTED")
TEXT_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "model_id",
    "predicted_label",
)
PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
REQUIRED_COLUMNS = tuple(
    dict.fromkeys(TEXT_COLUMNS + ("class_vocabulary",) + PROBABILITY_COLUMNS + ("correct",))
)
SORT_COLUMNS = ("context_id", "unit_id", "model_id")
IDENTITY_COLUMNS = ("master_sample_id", "true_label", "instrument", "domain")
METRIC_COLUMNS = (
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece_equal_mass",
)

DOMAIN_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "station",
    "domain",
    "instrument",
    "contexts",
    "unit_appearances",
    "physical_masters",
    *METRIC_COLUMNS,
]
SUMMARY_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "domains",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
    *METRIC_COLUMNS,
]
CONFUSION_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "station",
    "true_chemical",
    "predicted_chemical",
    "count",
    "true_appearances",
    "row_fraction",
]
CLASS_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "station",
    "chemical",
    "true_appearances",
    "physical_masters",
    "contributing_contexts",
    "contributing_domains",
    "pooled_recall",
    "mean_context_recall",
    "mean_domain_recall",
]
BIN_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "station",
    "bin_index",
    "lower",
    "upper",
    "count",
    "mean_confidence",
    "observed_accuracy",
]
RELIABILITY_COLUMNS = [
    "scope",
    "aggregation_id",
    "model_id",
    "station",
    "contexts",
    "unit_appearances",
    "pooled_ece_equal_width",
    "mean_context_ece_equal_mass",
]


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64).mean())


def _union(sets: Any) -> set[str]:
    output: set[str] = set()
    for item in sets:
        output.update(item)
    return output


def _validate_vocabulary(frame: pd.DataFrame) -> None:
    parsed: list[tuple[str, ...]] = []
    for value in frame["class_vocabulary"].to_numpy():
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                _fail("vocabulary_json")
        elif isinstance(value, np.ndarray):
            if value.ndim != 1:
                _fail("vocabulary_value")
            decoded = value.tolist()
        elif isinstance(value, (list, tuple)):
            decoded = list(value)
        else:
            _fail("vocabulary_type")
        if (
            not isinstance(decoded, list)
            or len(decoded) != 3
            or any(
                not isinstance(item, str) or item == "" or item != item.strip() for item in decoded
            )
            or list(decoded) != sorted(decoded)
            or len(set(decoded)) != 3
        ):
            _fail("vocabulary_value")
        parsed.append(tuple(decoded))
    by_station: dict[str, tuple[str, ...]] = {}
    for station, vocabulary in zip(frame["station"].to_numpy(), parsed, strict=True):
        existing = by_station.setdefault(station, vocabulary)
        if existing != vocabulary:
            _fail("station_vocabulary")
    frame["_vocabulary"] = parsed
    true_labels = frame["true_label"].to_numpy()
    for vocabulary, label in zip(parsed, true_labels, strict=True):
        if label not in vocabulary:
            _fail("true_label_vocabulary")


def _validate_probabilities(frame: pd.DataFrame) -> None:
    numeric = np.empty((len(frame), len(PROBABILITY_COLUMNS)), dtype=np.float64)
    for position, column in enumerate(PROBABILITY_COLUMNS):
        values = frame[column].to_numpy()
        dtype = values.dtype
        if dtype == np.dtype("O"):
            for value in values:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    _fail("probability_type")
            column_values = np.asarray(values, dtype=np.float64)
        else:
            if np.issubdtype(dtype, np.bool_):
                _fail("probability_boolean")
            if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
                _fail("probability_type")
            column_values = values.astype(np.float64, copy=False)
        if not np.isfinite(column_values).all():
            _fail("probability_finite")
        if ((column_values < 0.0) | (column_values > 1.0)).any():
            _fail("probability_range")
        numeric[:, position] = column_values
    if not np.allclose(numeric.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        _fail("probability_sum")
    vocabulary = frame["_vocabulary"].to_numpy()
    predicted = np.asarray(
        [vocabulary[row][int(index)] for row, index in enumerate(numeric.argmax(axis=1))]
    )
    if not np.array_equal(frame["predicted_label"].to_numpy(), predicted):
        _fail("prediction_argmax")


def _validate_correct(frame: pd.DataFrame) -> None:
    values = frame["correct"].to_numpy()
    dtype = values.dtype
    if dtype == np.dtype("O"):
        numeric = np.empty(len(frame), dtype=np.int64)
        for position, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                numeric[position] = int(value)
            elif isinstance(value, (int, np.integer)):
                number = int(value)
                if number not in (0, 1):
                    _fail("correct_value")
                numeric[position] = number
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                if not np.isfinite(number):
                    _fail("correct_finite")
                if number != np.floor(number):
                    _fail("correct_type")
                number = int(number)
                if number not in (0, 1):
                    _fail("correct_value")
                numeric[position] = number
            else:
                _fail("correct_type")
    elif np.issubdtype(dtype, np.bool_):
        numeric = values.astype(np.int64)
    elif np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating):
        column_values = values.astype(np.float64, copy=False)
        if not np.isfinite(column_values).all():
            _fail("correct_finite")
        if not np.array_equal(column_values, np.floor(column_values)):
            _fail("correct_type")
        numeric = column_values.astype(np.int64)
        if not np.isin(numeric, (0, 1)).all():
            _fail("correct_value")
    else:
        _fail("correct_type")
    agreement = (frame["true_label"].to_numpy() == frame["predicted_label"].to_numpy()).astype(
        np.int64
    )
    if not np.array_equal(agreement, numeric):
        _fail("correct_agreement")


def _validate_panel(panel: Any, endpoint: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        _fail("panel_frame")
    if panel.columns.duplicated().any():
        _fail("duplicate_columns")
    if not set(REQUIRED_COLUMNS).issubset(set(panel.columns)):
        _fail("missing_columns")
    frame = panel.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in TEXT_COLUMNS:
        for value in frame[column].to_numpy():
            if not isinstance(value, str) or value == "" or value != value.strip():
                _fail("text_value")
    if set(frame["model_id"]) != set(MODEL_IDS):
        _fail("model_coverage")
    if not set(frame["station"]).issubset(set(STATIONS)):
        _fail("station_value")
    _validate_vocabulary(frame)
    _validate_probabilities(frame)
    _validate_correct(frame)
    if frame.duplicated(["model_id", "context_id", "unit_id"]).any():
        _fail("identity_duplicate")
    if endpoint == "M06" and frame.duplicated(["model_id", "context_id", "master_sample_id"]).any():
        _fail("m06_master_duplicate")
    return frame.sort_values(list(SORT_COLUMNS), kind="stable").reset_index(drop=True)


def _validate_global(frame: pd.DataFrame) -> None:
    if (
        frame.groupby("context_id")[["domain", "station", "instrument"]]
        .nunique()
        .gt(1)
        .to_numpy()
        .any()
    ):
        _fail("context_identity")
    if (
        frame.groupby("master_sample_id")[["station", "true_label"]]
        .nunique()
        .gt(1)
        .to_numpy()
        .any()
    ):
        _fail("master_identity")
    if frame.groupby("unit_id")[list(IDENTITY_COLUMNS)].nunique().gt(1).to_numpy().any():
        _fail("unit_identity")
    if frame.groupby("domain")[["station", "instrument"]].nunique().gt(1).to_numpy().any():
        _fail("domain_identity")


def _cross_metadata(
    frames: dict[str, pd.DataFrame], key: str, columns: list[str], code: str
) -> None:
    left = frames["M01"].groupby(key)[list(columns)].first().sort_index()
    right = frames["M06"].groupby(key)[list(columns)].first().sort_index()
    shared = left.index.intersection(right.index)
    if not left.loc[shared].equals(right.loc[shared]):
        _fail(code)


def _cross_vocabulary(frames: dict[str, pd.DataFrame]) -> None:
    by_endpoint = {
        endpoint: dict(
            zip(
                frames[endpoint]["station"].to_numpy(),
                frames[endpoint]["_vocabulary"].to_numpy(),
                strict=True,
            )
        )
        for endpoint in ENDPOINTS
    }
    left = by_endpoint["M01"]
    right = by_endpoint["M06"]
    for station in set(left) & set(right):
        if left[station] != right[station]:
            _fail("cross_vocabulary")


def _validate_cross(frames: dict[str, pd.DataFrame]) -> None:
    for model in MODEL_IDS:
        contexts = {
            endpoint: set(
                frames[endpoint].loc[frames[endpoint]["model_id"].eq(model), "context_id"]
            )
            for endpoint in ENDPOINTS
        }
        if contexts["M01"] != contexts["M06"]:
            _fail("cross_context")
        for context_id in contexts["M01"]:
            masters = {
                endpoint: set(
                    frames[endpoint].loc[
                        frames[endpoint]["model_id"].eq(model)
                        & frames[endpoint]["context_id"].eq(context_id),
                        "master_sample_id",
                    ]
                )
                for endpoint in ENDPOINTS
            }
            if masters["M01"] != masters["M06"]:
                _fail("cross_master")
    _cross_metadata(frames, "master_sample_id", ["station", "true_label"], "cross_master_metadata")
    _cross_metadata(
        frames, "context_id", ["domain", "station", "instrument"], "cross_context_metadata"
    )
    _cross_vocabulary(frames)


def _validate_common_context(frame: pd.DataFrame, first: str, second: str, context_id: str) -> None:
    left = frame[frame["model_id"].eq(first) & frame["context_id"].eq(context_id)]
    right = frame[frame["model_id"].eq(second) & frame["context_id"].eq(context_id)]
    if set(left["unit_id"]) != set(right["unit_id"]):
        _fail("common_unit_mismatch")
    columns = list(IDENTITY_COLUMNS)
    left_frame = left.set_index("unit_id")[columns].sort_index()
    right_frame = right.set_index("unit_id")[columns].sort_index()
    if not left_frame.equals(right_frame):
        _fail("common_identity_mismatch")


def _scope_selection(frame: pd.DataFrame, scope: str) -> dict[str, list[str]]:
    if scope == "full_support":
        return {
            model: sorted(frame.loc[frame["model_id"].eq(model), "context_id"].unique().tolist())
            for model in MODEL_IDS
        }
    first, second = SELECTED_MODELS
    left = set(frame.loc[frame["model_id"].eq(first), "context_id"])
    right = set(frame.loc[frame["model_id"].eq(second), "context_id"])
    common = sorted(left & right)
    for context_id in common:
        _validate_common_context(frame, first, second, context_id)
    return {model: common for model in SELECTED_MODELS}


def _pooled_metrics(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict[str, Any]:
    return classification_metrics(
        frame["true_label"].to_numpy(),
        frame["predicted_label"].to_numpy(),
        class_vocabulary=list(classes),
        probabilities=frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
    )


def _context_records(
    scope: str, endpoint: str, model: str, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context_id in sorted(frame["context_id"].unique().tolist()):
        group = frame[frame["context_id"].eq(context_id)]
        classes = tuple(group["_vocabulary"].iloc[0])
        metrics = _pooled_metrics(group, classes)
        records.append(
            {
                "scope": scope,
                "aggregation_id": endpoint,
                "model_id": model,
                "context_id": context_id,
                "domain": group["domain"].iloc[0],
                "station": group["station"].iloc[0],
                "instrument": group["instrument"].iloc[0],
                "unit_appearances": int(len(group)),
                "masters": set(group["master_sample_id"]),
                "units": set(group["unit_id"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "negative_log_likelihood": float(metrics["negative_log_likelihood"]),
                "brier_score": float(metrics["brier_score"]),
                "ece_equal_mass": float(metrics["ece"]),
                "support": metrics["support"],
                "recalls": metrics["per_class_recall"],
            }
        )
    return records


def _finalize(
    rows: list[dict[str, Any]],
    columns: list[str],
    integer_columns: list[str],
    sort_keys: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    for column in integer_columns:
        frame[column] = frame[column].astype(np.int64)
    return frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)


def build_metrics(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aggregate two authenticated frozen endpoints into six descriptive tables."""

    if not isinstance(panels, dict) or set(panels) != set(ENDPOINTS):
        _fail("panel_keys")
    frames = {endpoint: _validate_panel(panels[endpoint], endpoint) for endpoint in ENDPOINTS}
    for endpoint in ENDPOINTS:
        _validate_global(frames[endpoint])
    _validate_cross(frames)

    domain_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []

    for scope in SCOPES:
        for endpoint in ENDPOINTS:
            frame = frames[endpoint]
            selection = _scope_selection(frame, scope)
            for model in selection:
                context_ids = selection[model]
                sub = frame[frame["model_id"].eq(model) & frame["context_id"].isin(context_ids)]
                if sub.empty:
                    continue
                records = _context_records(scope, endpoint, model, sub)
                by_domain: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
                for record in records:
                    by_domain.setdefault(
                        (record["station"], record["domain"], record["instrument"]), []
                    ).append(record)

                for (station, domain, instrument), group in sorted(by_domain.items()):
                    domain_rows.append(
                        {
                            "scope": scope,
                            "aggregation_id": endpoint,
                            "model_id": model,
                            "station": station,
                            "domain": domain,
                            "instrument": instrument,
                            "contexts": len(group),
                            "unit_appearances": sum(r["unit_appearances"] for r in group),
                            "physical_masters": len(_union(r["masters"] for r in group)),
                            **{name: _mean([r[name] for r in group]) for name in METRIC_COLUMNS},
                        }
                    )

                summary_rows.append(
                    {
                        "scope": scope,
                        "aggregation_id": endpoint,
                        "model_id": model,
                        "domains": len(by_domain),
                        "contexts": len(records),
                        "unit_appearances": sum(r["unit_appearances"] for r in records),
                        "physical_masters": len(_union(r["masters"] for r in records)),
                        "distinct_units": len(_union(r["units"] for r in records)),
                        **{
                            name: _mean(
                                [_mean([r[name] for r in group]) for group in by_domain.values()]
                            )
                            for name in METRIC_COLUMNS
                        },
                    }
                )

                for station, station_frame in sub.groupby("station", sort=True):
                    classes = tuple(station_frame["_vocabulary"].iloc[0])
                    pooled = _pooled_metrics(station_frame, classes)
                    matrix = pooled["confusion_matrix"]
                    support = pooled["support"]
                    for row_index, true_chemical in enumerate(classes):
                        for column_index, predicted_chemical in enumerate(classes):
                            count = int(matrix[row_index][column_index])
                            appearances = int(support[true_chemical])
                            confusion_rows.append(
                                {
                                    "scope": scope,
                                    "aggregation_id": endpoint,
                                    "model_id": model,
                                    "station": station,
                                    "true_chemical": true_chemical,
                                    "predicted_chemical": predicted_chemical,
                                    "count": count,
                                    "true_appearances": appearances,
                                    "row_fraction": (
                                        count / appearances if appearances else float("nan")
                                    ),
                                }
                            )

                    station_records = [r for r in records if r["station"] == station]
                    for class_index, chemical in enumerate(classes):
                        appearances = int(support[chemical])
                        pooled_recall = (
                            matrix[class_index][class_index] / appearances
                            if appearances
                            else float("nan")
                        )
                        contributing = [r for r in station_records if r["support"][chemical] > 0]
                        if contributing:
                            mean_context_recall = _mean(
                                [r["recalls"][chemical] for r in contributing]
                            )
                            domains: dict[str, list[float]] = {}
                            for record in contributing:
                                domains.setdefault(record["domain"], []).append(
                                    record["recalls"][chemical]
                                )
                            mean_domain_recall = _mean(
                                [_mean(values) for values in domains.values()]
                            )
                            contributing_domains = len(domains)
                        else:
                            mean_context_recall = float("nan")
                            mean_domain_recall = float("nan")
                            contributing_domains = 0
                        class_rows.append(
                            {
                                "scope": scope,
                                "aggregation_id": endpoint,
                                "model_id": model,
                                "station": station,
                                "chemical": chemical,
                                "true_appearances": appearances,
                                "physical_masters": len(
                                    set(
                                        station_frame.loc[
                                            station_frame["true_label"].eq(chemical),
                                            "master_sample_id",
                                        ]
                                    )
                                ),
                                "contributing_contexts": len(contributing),
                                "contributing_domains": contributing_domains,
                                "pooled_recall": pooled_recall,
                                "mean_context_recall": mean_context_recall,
                                "mean_domain_recall": mean_domain_recall,
                            }
                        )

                    probabilities = station_frame[list(PROBABILITY_COLUMNS)].to_numpy(
                        dtype=np.float64
                    )
                    confidence = probabilities.max(axis=1)
                    correct = station_frame["correct"].to_numpy(dtype=np.float64)
                    index = np.clip(
                        np.searchsorted(
                            np.arange(11, dtype=float) / 10.0,
                            confidence,
                            side="right",
                        )
                        - 1,
                        0,
                        9,
                    ).astype(int)
                    total = len(confidence)
                    pooled_width = 0.0
                    for bin_index in range(10):
                        mask = index == bin_index
                        count = int(mask.sum())
                        if count:
                            mean_confidence = float(confidence[mask].mean())
                            observed_accuracy = float(correct[mask].mean())
                            pooled_width += count / total * abs(observed_accuracy - mean_confidence)
                        else:
                            mean_confidence = float("nan")
                            observed_accuracy = float("nan")
                        bin_rows.append(
                            {
                                "scope": scope,
                                "aggregation_id": endpoint,
                                "model_id": model,
                                "station": station,
                                "bin_index": bin_index,
                                "lower": bin_index / 10.0,
                                "upper": (bin_index + 1) / 10.0,
                                "count": count,
                                "mean_confidence": mean_confidence,
                                "observed_accuracy": observed_accuracy,
                            }
                        )
                    reliability_rows.append(
                        {
                            "scope": scope,
                            "aggregation_id": endpoint,
                            "model_id": model,
                            "station": station,
                            "contexts": len(station_records),
                            "unit_appearances": int(len(station_frame)),
                            "pooled_ece_equal_width": float(pooled_width),
                            "mean_context_ece_equal_mass": _mean(
                                [r["ece_equal_mass"] for r in station_records]
                            ),
                        }
                    )

    return {
        "domain_metrics": _finalize(
            domain_rows,
            DOMAIN_COLUMNS,
            ["contexts", "unit_appearances", "physical_masters"],
            ["scope", "aggregation_id", "model_id", "station", "domain", "instrument"],
        ),
        "model_summary": _finalize(
            summary_rows,
            SUMMARY_COLUMNS,
            ["domains", "contexts", "unit_appearances", "physical_masters", "distinct_units"],
            ["scope", "aggregation_id", "model_id"],
        ),
        "confusion": _finalize(
            confusion_rows,
            CONFUSION_COLUMNS,
            ["count", "true_appearances"],
            [
                "scope",
                "aggregation_id",
                "model_id",
                "station",
                "true_chemical",
                "predicted_chemical",
            ],
        ),
        "class_sensitivity": _finalize(
            class_rows,
            CLASS_COLUMNS,
            [
                "true_appearances",
                "physical_masters",
                "contributing_contexts",
                "contributing_domains",
            ],
            ["scope", "aggregation_id", "model_id", "station", "chemical"],
        ),
        "reliability_bins": _finalize(
            bin_rows,
            BIN_COLUMNS,
            ["bin_index", "count"],
            ["scope", "aggregation_id", "model_id", "station", "bin_index"],
        ),
        "reliability_summary": _finalize(
            reliability_rows,
            RELIABILITY_COLUMNS,
            ["contexts", "unit_appearances"],
            ["scope", "aggregation_id", "model_id", "station"],
        ),
    }
