"""P08 registered prediction units and descriptive summaries (private draft).

This is a pure in-memory results adapter. The caller authenticates the file,
job and input hashes plus the source/calibration lineage *before* calling
:func:`build_units`; the identity, coverage and probability checks below only
confirm that the supplied frozen tables are mutually consistent and do not by
themselves establish that authority. Nothing here loads data, fits, selects,
recalibrates, resamples, clips or averages technical seeds. Inputs are never
mutated.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import (
    classification_metrics,
    instrument_balanced_master_probabilities,
)
from atlas_sers.governance.canonical import sha256_value

__all__ = [
    "AGGREGATIONS",
    "ESTIMANDS",
    "MODELS",
    "POLICIES",
    "P08UnitsError",
    "build_units",
    "summarize_units",
]

POLICIES = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")
MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED")
ESTIMANDS = ("equal_context", "pooled_four_fold")
AGGREGATIONS = ("M01", "M06")

PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")

_PROBABILITY_TOLERANCE = 1e-12
_POOLED_CONTEXT_PREFIX = "pool-"
_FOLD_COUNT = 4
_RELIABILITY_BINS = 10
_SCOPE_POOLED = "pooled_repeated_appearances"
_SCOPE_EQUAL_CONTEXT = "equal_context_fixed_bins"
_METRIC_COLUMNS = (
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece",
)

_REGISTERED_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "outer_fold",
    "observation_uid",
    "master_sample_id",
    "true_label",
    "class_vocabulary",
)

_PREDICTION_COLUMNS = (
    "context_id",
    "policy_id",
    "model_id",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "station",
    "true_label",
    "class_vocabulary",
    "probability_0",
    "probability_1",
    "probability_2",
)

_PANEL_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "model_id",
    "class_vocabulary",
    "probability_0",
    "probability_1",
    "probability_2",
    "predicted_label",
    "correct",
    "outer_repeat",
    "outer_fold",
    "outer_fold_count",
)

_SUMMARY_REQUIRED = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "model_id",
    "class_vocabulary",
    "probability_0",
    "probability_1",
    "probability_2",
    "predicted_label",
    "correct",
)


class P08UnitsError(ValueError):
    """Raised when frozen P08 prediction tables are inconsistent."""


def _require_frame(frame, code: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise P08UnitsError(code)
    if frame.columns.duplicated().any():
        raise P08UnitsError("duplicate_columns")


def _require_columns(frame: pd.DataFrame, columns, code: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise P08UnitsError(f"{code}:{','.join(missing)}")


def _strict_identifier(value, code: str) -> str:
    if not isinstance(value, str):
        raise P08UnitsError(code)
    if not value or value != value.strip():
        raise P08UnitsError(code)
    return value


def _integral_value(value, code: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise P08UnitsError(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
        raise P08UnitsError(code)
    raise P08UnitsError(code)


def _probability(value, code: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise P08UnitsError(code)
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise P08UnitsError(code)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise P08UnitsError(code) from None
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise P08UnitsError(code)
    return number


def _boolean_flag(value, code: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    raise P08UnitsError(code)


def _normalize_vocabulary(value) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise P08UnitsError("vocabulary_invalid")
        items = value.tolist()
    elif isinstance(value, str):
        try:
            items = json.loads(value)
        except (TypeError, ValueError):
            raise P08UnitsError("vocabulary_invalid") from None
        if not isinstance(items, list):
            raise P08UnitsError("vocabulary_invalid")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise P08UnitsError("vocabulary_invalid")
    if len(items) != 3:
        raise P08UnitsError("vocabulary_invalid")
    if not all(
        isinstance(item, str) and item == item.strip() and item for item in items
    ):
        raise P08UnitsError("vocabulary_invalid")
    if len(set(items)) != 3:
        raise P08UnitsError("vocabulary_invalid")
    if list(items) != sorted(items):
        raise P08UnitsError("vocabulary_invalid")
    return tuple(items)


def _require_one_to_one(frame: pd.DataFrame, key, values, code: str) -> None:
    counts = frame.groupby(key, dropna=False, sort=False)[list(values)].nunique(
        dropna=False
    )
    if (counts.to_numpy() > 1).any():
        raise P08UnitsError(code)


def _require_fold_disjoint(frame: pd.DataFrame, value: str, code: str) -> None:
    counts = frame.groupby(["domain", "outer_repeat", value], dropna=False, sort=False)[
        "outer_fold"
    ].nunique(dropna=False)
    if (counts > 1).any():
        raise P08UnitsError(code)


def _validate_registered_structure(registered: pd.DataFrame) -> None:
    _require_one_to_one(
        registered, "domain", ("station", "instrument"), "registered_domain_mapping"
    )
    _require_one_to_one(
        registered,
        "context_id",
        ("domain", "station", "instrument", "outer_repeat", "outer_fold"),
        "registered_context_mapping",
    )
    _require_one_to_one(
        registered,
        "master_sample_id",
        ("station", "true_label"),
        "registered_master_mapping",
    )
    _require_one_to_one(
        registered,
        "observation_uid",
        ("master_sample_id", "instrument", "station", "true_label"),
        "registered_observation_mapping",
    )
    vocabulary_key = registered.class_vocabulary.map(lambda value: "\x1f".join(value))
    _require_one_to_one(
        registered.assign(_vocabulary_key=vocabulary_key),
        "station",
        ("_vocabulary_key",),
        "registered_station_vocabulary",
    )
    fold_counts = registered.groupby(
        ["domain", "outer_repeat"], dropna=False, sort=False
    )["outer_fold"].nunique(dropna=False)
    if not fold_counts.eq(_FOLD_COUNT).all():
        raise P08UnitsError("registered_fold_count")
    context_per_fold = registered.groupby(
        ["domain", "outer_repeat", "outer_fold"], dropna=False, sort=False
    )["context_id"].nunique(dropna=False)
    if not context_per_fold.eq(1).all():
        raise P08UnitsError("registered_context_per_fold")
    _require_fold_disjoint(
        registered, "observation_uid", "registered_observation_overlap_folds"
    )
    _require_fold_disjoint(
        registered, "master_sample_id", "registered_master_overlap_folds"
    )


def _parse_registered(frame) -> pd.DataFrame:
    _require_frame(frame, "registered_not_frame")
    _require_columns(frame, _REGISTERED_COLUMNS, "registered_columns_missing")
    for forbidden in ("model_id", "policy_id"):
        if forbidden in frame.columns:
            raise P08UnitsError("registered_has_model_or_policy_column")
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        context_id = _strict_identifier(row.context_id, "registered_context_invalid")
        domain = _strict_identifier(row.domain, "registered_domain_invalid")
        station = _strict_identifier(row.station, "registered_station_invalid")
        instrument = _strict_identifier(row.instrument, "registered_instrument_invalid")
        observation_uid = _strict_identifier(
            row.observation_uid, "registered_observation_invalid"
        )
        master_sample_id = _strict_identifier(
            row.master_sample_id, "registered_master_invalid"
        )
        true_label = _strict_identifier(row.true_label, "registered_label_invalid")
        outer_repeat = _integral_value(row.outer_repeat, "registered_repeat_invalid")
        outer_fold = _integral_value(row.outer_fold, "registered_fold_invalid")
        vocabulary = _normalize_vocabulary(row.class_vocabulary)
        if true_label not in vocabulary:
            raise P08UnitsError("registered_label_not_in_vocabulary")
        key = (context_id, observation_uid)
        if key in seen:
            raise P08UnitsError("registered_duplicate_observation")
        seen.add(key)
        records.append(
            {
                "context_id": context_id,
                "domain": domain,
                "station": station,
                "instrument": instrument,
                "outer_repeat": outer_repeat,
                "outer_fold": outer_fold,
                "observation_uid": observation_uid,
                "master_sample_id": master_sample_id,
                "true_label": true_label,
                "class_vocabulary": vocabulary,
            }
        )
    registered = pd.DataFrame(records, columns=list(_REGISTERED_COLUMNS))
    if registered.empty:
        raise P08UnitsError("registered_empty")
    _validate_registered_structure(registered)
    return registered


def _validate_policy_model_coverage(
    predictions: pd.DataFrame, registered: pd.DataFrame
) -> None:
    expected_pairs = {(policy, model) for policy in POLICIES for model in MODELS}
    observed_pairs = set(
        zip(
            predictions.policy_id.astype(str),
            predictions.model_id.astype(str),
            strict=True,
        )
    )
    if observed_pairs != expected_pairs:
        raise P08UnitsError("predictions_policy_model_mismatch")
    full_keys = set(
        zip(
            registered.context_id.astype(str),
            registered.observation_uid.astype(str),
            strict=True,
        )
    )
    for _, cell in predictions.groupby(["policy_id", "model_id"], sort=True):
        keys = list(
            zip(
                cell.context_id.astype(str),
                cell.observation_uid.astype(str),
                strict=True,
            )
        )
        if len(set(keys)) != len(keys):
            raise P08UnitsError("predictions_duplicate_observation")
        if set(keys) != full_keys:
            raise P08UnitsError("predictions_registered_set_mismatch")


def _parse_predictions(frame, registered: pd.DataFrame) -> pd.DataFrame:
    _require_frame(frame, "predictions_not_frame")
    _require_columns(frame, _PREDICTION_COLUMNS, "predictions_columns_missing")
    reference = {
        (row.context_id, row.observation_uid): row
        for row in registered.itertuples(index=False)
    }
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        context_id = _strict_identifier(row.context_id, "predictions_context_invalid")
        policy_id = _strict_identifier(row.policy_id, "predictions_policy_invalid")
        model_id = _strict_identifier(row.model_id, "predictions_model_invalid")
        observation_uid = _strict_identifier(
            row.observation_uid, "predictions_observation_invalid"
        )
        master_sample_id = _strict_identifier(
            row.master_sample_id, "predictions_master_invalid"
        )
        instrument = _strict_identifier(
            row.instrument, "predictions_instrument_invalid"
        )
        station = _strict_identifier(row.station, "predictions_station_invalid")
        true_label = _strict_identifier(row.true_label, "predictions_label_invalid")
        vocabulary = _normalize_vocabulary(row.class_vocabulary)
        if policy_id not in POLICIES:
            raise P08UnitsError("predictions_unknown_policy")
        if model_id not in MODELS:
            raise P08UnitsError("predictions_unknown_model")
        reference_row = reference.get((context_id, observation_uid))
        if reference_row is None:
            raise P08UnitsError("predictions_unknown_observation")
        if (
            master_sample_id != reference_row.master_sample_id
            or instrument != reference_row.instrument
            or station != reference_row.station
            or true_label != reference_row.true_label
            or vocabulary != reference_row.class_vocabulary
        ):
            raise P08UnitsError("predictions_identity_mismatch")
        p0 = _probability(row.probability_0, "predictions_probability_invalid")
        p1 = _probability(row.probability_1, "predictions_probability_invalid")
        p2 = _probability(row.probability_2, "predictions_probability_invalid")
        if abs((p0 + p1 + p2) - 1.0) > _PROBABILITY_TOLERANCE:
            raise P08UnitsError("predictions_probability_not_normalised")
        records.append(
            {
                "context_id": context_id,
                "policy_id": policy_id,
                "model_id": model_id,
                "observation_uid": observation_uid,
                "master_sample_id": master_sample_id,
                "instrument": instrument,
                "station": station,
                "true_label": true_label,
                "class_vocabulary": vocabulary,
                "probability_0": p0,
                "probability_1": p1,
                "probability_2": p2,
            }
        )
    predictions = pd.DataFrame(records, columns=list(_PREDICTION_COLUMNS))
    _validate_policy_model_coverage(predictions, registered)
    return predictions


def _finalize_panel(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(_PANEL_COLUMNS))
    for column in PROBABILITY_COLUMNS:
        frame[column] = frame[column].astype(float)
    frame["correct"] = frame["correct"].astype(bool)
    for column in (
        "context_id",
        "domain",
        "station",
        "instrument",
        "master_sample_id",
        "unit_id",
        "true_label",
        "model_id",
        "predicted_label",
    ):
        frame[column] = frame[column].astype(str)
    frame["outer_repeat"] = frame["outer_repeat"].astype(int)
    frame["outer_fold_count"] = frame["outer_fold_count"].astype(int)
    if frame.duplicated(["model_id", "context_id", "unit_id"]).any():
        raise P08UnitsError("panel_duplicate_unit")
    return frame.sort_values(
        ["model_id", "context_id", "unit_id"], kind="stable"
    ).reset_index(drop=True)


def _m01_frame(tagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = tagged.groupby(["model_id", "policy_id", "panel_context_id"], sort=True)
    for (model_id, _policy_id, panel_context), cell in grouped:
        vocabulary = cell.class_vocabulary.iloc[0]
        probabilities = cell[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        predicted = [
            vocabulary[int(index)] for index in np.argmax(probabilities, axis=1)
        ]
        for position, row in enumerate(cell.itertuples(index=False)):
            true_label = str(row.true_label)
            predicted_label = str(predicted[position])
            rows.append(
                {
                    "context_id": str(panel_context),
                    "domain": str(row.domain),
                    "station": str(row.station),
                    "instrument": str(row.instrument),
                    "master_sample_id": str(row.master_sample_id),
                    "unit_id": str(row.observation_uid),
                    "true_label": true_label,
                    "model_id": str(model_id),
                    "class_vocabulary": tuple(vocabulary),
                    "probability_0": float(row.probability_0),
                    "probability_1": float(row.probability_1),
                    "probability_2": float(row.probability_2),
                    "predicted_label": predicted_label,
                    "correct": bool(true_label == predicted_label),
                    "outer_repeat": row.outer_repeat,
                    "outer_fold": row.outer_fold,
                    "outer_fold_count": row.outer_fold_count,
                }
            )
    return _finalize_panel(rows)


def _m06_frame(tagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = tagged.groupby(["model_id", "policy_id", "panel_context_id"], sort=True)
    for (model_id, _policy_id, panel_context), cell in grouped:
        vocabulary = cell.class_vocabulary.iloc[0]
        master = instrument_balanced_master_probabilities(
            probabilities=cell[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
            true_labels=cell.true_label.astype(str).to_numpy(),
            master_ids=cell.master_sample_id.astype(str).to_numpy(),
            instruments=cell.instrument.astype(str).to_numpy(),
            class_vocabulary=list(vocabulary),
        )
        first = cell.iloc[0]
        domain = str(first.domain)
        station = str(first.station)
        instrument = str(first.instrument)
        for row in master.itertuples(index=False):
            probabilities = [float(value) for value in row.probabilities]
            master_id = str(row.master_sample_id)
            true_label = str(row.true_label)
            predicted_label = str(row.predicted_label)
            rows.append(
                {
                    "context_id": str(panel_context),
                    "domain": domain,
                    "station": station,
                    "instrument": instrument,
                    "master_sample_id": master_id,
                    "unit_id": sha256_value([domain, master_id]),
                    "true_label": true_label,
                    "model_id": str(model_id),
                    "class_vocabulary": tuple(vocabulary),
                    "probability_0": probabilities[0],
                    "probability_1": probabilities[1],
                    "probability_2": probabilities[2],
                    "predicted_label": predicted_label,
                    "correct": bool(true_label == predicted_label),
                    "outer_repeat": first.outer_repeat,
                    "outer_fold": first.outer_fold,
                    "outer_fold_count": first.outer_fold_count,
                }
            )
    return _finalize_panel(rows)


def _pooled_context_id(domain: str, repeat) -> str:
    return _POOLED_CONTEXT_PREFIX + sha256_value([str(domain), str(int(repeat))])


def build_units(
    predictions: pd.DataFrame,
    registered_test_rows: pd.DataFrame,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Build M01/M06 units for both estimands from complete frozen panels.

    Every policy/model pair must cover exactly the registered observation set:
    missing, extra or duplicated cells/rows are rejected instead of intersected
    or imputed. Authority over the frozen inputs is authenticated by the caller
    before this adapter runs; these checks are consistency checks only.
    """

    registered = _parse_registered(registered_test_rows)
    predictions = _parse_predictions(predictions, registered)
    tagged = predictions.merge(
        registered[
            ["context_id", "observation_uid", "domain", "outer_repeat", "outer_fold"]
        ],
        on=["context_id", "observation_uid"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if tagged["domain"].isna().any():
        raise P08UnitsError("predictions_unregistered_observation")
    tagged["outer_fold_count"] = _FOLD_COUNT

    equal = tagged.copy()
    equal["panel_context_id"] = equal["context_id"]

    pooled = tagged.copy()
    pooled["panel_context_id"] = [
        _pooled_context_id(domain, repeat)
        for domain, repeat in zip(pooled["domain"], pooled["outer_repeat"], strict=True)
    ]
    pooled["outer_fold"] = pd.NA

    panels: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for estimand, tagged_estimand in (
        ("equal_context", equal),
        ("pooled_four_fold", pooled),
    ):
        panels[estimand] = {}
        for policy in POLICIES:
            policy_rows = tagged_estimand[tagged_estimand.policy_id.eq(policy)]
            panels[estimand][policy] = {
                "M01": _m01_frame(policy_rows),
                "M06": _m06_frame(policy_rows),
            }
    return panels


def _validated_summary(units: pd.DataFrame) -> pd.DataFrame:
    _require_frame(units, "units_not_frame")
    _require_columns(units, _SUMMARY_REQUIRED, "units_columns_missing")
    if units.empty:
        raise P08UnitsError("units_empty")
    frame = units.copy()
    for column in (
        "context_id",
        "domain",
        "station",
        "instrument",
        "master_sample_id",
        "unit_id",
    ):
        frame[column] = [
            _strict_identifier(value, "units_identity_invalid")
            for value in frame[column]
        ]
    frame["true_label"] = [
        _strict_identifier(value, "units_label_invalid")
        for value in frame["true_label"]
    ]
    frame["predicted_label"] = [
        _strict_identifier(value, "units_prediction_invalid")
        for value in frame["predicted_label"]
    ]
    frame["model_id"] = [
        _strict_identifier(value, "units_model_invalid") for value in frame["model_id"]
    ]
    frame["class_vocabulary"] = [
        _normalize_vocabulary(value) for value in frame["class_vocabulary"]
    ]
    probabilities = np.empty((len(frame), len(PROBABILITY_COLUMNS)), dtype=float)
    for column_index, column in enumerate(PROBABILITY_COLUMNS):
        for row_index, value in enumerate(frame[column]):
            probabilities[row_index, column_index] = _probability(
                value, "units_probability_invalid"
            )
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE
    ):
        raise P08UnitsError("units_probability_not_normalised")
    argmax_indices = probabilities.argmax(axis=1)
    for row_index, (true_label, predicted_label, vocabulary) in enumerate(
        zip(
            frame.true_label, frame.predicted_label, frame.class_vocabulary, strict=True
        )
    ):
        if true_label not in vocabulary or predicted_label not in vocabulary:
            raise P08UnitsError("units_label_outside_vocabulary")
        if predicted_label != vocabulary[argmax_indices[row_index]]:
            raise P08UnitsError("units_prediction_mismatch")
    correct_flags = np.asarray(
        [_boolean_flag(value, "units_correct_invalid") for value in frame["correct"]],
        dtype=bool,
    )
    truth = frame.true_label.eq(frame.predicted_label).to_numpy(dtype=bool)
    if not np.array_equal(correct_flags, truth):
        raise P08UnitsError("units_correct_invalid")
    if frame.duplicated(["model_id", "context_id", "unit_id"]).any():
        raise P08UnitsError("units_duplicate_unit")
    _require_one_to_one(
        frame,
        "context_id",
        ("domain", "station", "instrument"),
        "units_context_mapping",
    )
    _require_one_to_one(
        frame, "master_sample_id", ("station", "true_label"), "units_master_mapping"
    )
    _require_one_to_one(
        frame,
        "unit_id",
        ("master_sample_id", "station", "instrument", "true_label", "domain"),
        "units_unit_mapping",
    )
    vocabulary_key = frame.class_vocabulary.map(lambda value: "\x1f".join(value))
    _require_one_to_one(
        frame.assign(_vocabulary_key=vocabulary_key),
        "station",
        ("_vocabulary_key",),
        "units_station_vocabulary",
    )
    frame["correct_flag"] = truth
    frame["confidence"] = probabilities.max(axis=1)
    frame["bin_index"] = np.minimum(
        (frame["confidence"].to_numpy(dtype=float) * 10.0).astype(int),
        _RELIABILITY_BINS - 1,
    )
    return frame


def _context_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, context), cell in frame.groupby(["model_id", "context_id"], sort=True):
        vocabulary = cell.class_vocabulary.iloc[0]
        metrics = classification_metrics(
            cell.true_label.to_numpy(),
            cell.predicted_label.to_numpy(),
            class_vocabulary=list(vocabulary),
            probabilities=cell[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        )
        first = cell.iloc[0]
        rows.append(
            {
                "model_id": str(model),
                "context_id": str(context),
                "domain": str(first.domain),
                "station": str(first.station),
                "instrument": str(first.instrument),
                "unit_appearances": len(cell),
                "physical_masters": int(cell.master_sample_id.astype(str).nunique()),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "negative_log_likelihood": float(metrics["negative_log_likelihood"]),
                "brier_score": float(metrics["brier_score"]),
                "ece": float(metrics["ece"]),
            }
        )
    columns = [
        "model_id",
        "context_id",
        "domain",
        "station",
        "instrument",
        "unit_appearances",
        "physical_masters",
        *_METRIC_COLUMNS,
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["model_id", "context_id"], kind="stable")
        .reset_index(drop=True)
    )


def _domain_metrics(context_metrics: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    aggregated = context_metrics.groupby(["model_id", "domain"], as_index=False).agg(
        contexts=("context_id", "nunique"),
        **{column: (column, "mean") for column in _METRIC_COLUMNS},
    )
    counts = frame.groupby(["model_id", "domain"], as_index=False).agg(
        station=("station", "first"),
        instrument=("instrument", "first"),
        unit_appearances=("unit_id", "size"),
        physical_masters=("master_sample_id", "nunique"),
        distinct_units=("unit_id", "nunique"),
    )
    merged = aggregated.merge(
        counts, on=["model_id", "domain"], how="left", validate="one_to_one"
    )
    columns = [
        "model_id",
        "domain",
        "station",
        "instrument",
        "contexts",
        "unit_appearances",
        "physical_masters",
        "distinct_units",
        *_METRIC_COLUMNS,
    ]
    return (
        merged[columns]
        .sort_values(["model_id", "domain"], kind="stable")
        .reset_index(drop=True)
    )


def _model_summary(domain_metrics: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    summary = domain_metrics.groupby("model_id", as_index=False).agg(
        domains=("domain", "nunique"),
        **{column: (column, "mean") for column in _METRIC_COLUMNS},
    )
    counts = frame.groupby("model_id", as_index=False).agg(
        contexts=("context_id", "nunique"),
        unit_appearances=("unit_id", "size"),
        physical_masters=("master_sample_id", "nunique"),
        distinct_units=("unit_id", "nunique"),
    )
    summary = summary.merge(counts, on="model_id", how="left", validate="one_to_one")
    columns = [
        "model_id",
        "contexts",
        "domains",
        "unit_appearances",
        "physical_masters",
        "distinct_units",
        *_METRIC_COLUMNS,
    ]
    return (
        summary[columns].sort_values("model_id", kind="stable").reset_index(drop=True)
    )


def _class_recall(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, context), cell in frame.groupby(["model_id", "context_id"], sort=True):
        vocabulary = cell.class_vocabulary.iloc[0]
        labels = cell.true_label.to_numpy()
        predicted = cell.predicted_label.to_numpy()
        for klass in vocabulary:
            support = int(np.sum(labels == klass))
            correct = int(np.sum((labels == klass) & (predicted == klass)))
            recall = float(correct / support) if support else float("nan")
            rows.append(
                {
                    "model_id": str(model),
                    "context_id": str(context),
                    "domain": str(cell.domain.iloc[0]),
                    "station": str(cell.station.iloc[0]),
                    "class_label": str(klass),
                    "correct": correct,
                    "support": support,
                    "recall": recall,
                }
            )
    columns = [
        "model_id",
        "context_id",
        "domain",
        "station",
        "class_label",
        "correct",
        "support",
        "recall",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["model_id", "context_id", "class_label"], kind="stable")
        .reset_index(drop=True)
    )


def _confusion(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (station, model), cell in frame.groupby(["station", "model_id"], sort=True):
        vocabulary = cell.class_vocabulary.iloc[0]
        labels = cell.true_label.to_numpy()
        predicted = cell.predicted_label.to_numpy()
        for true_label in vocabulary:
            true_appearances = int(np.sum(labels == true_label))
            for predicted_label in vocabulary:
                count = int(
                    np.sum((labels == true_label) & (predicted == predicted_label))
                )
                rows.append(
                    {
                        "scope": _SCOPE_POOLED,
                        "station": str(station),
                        "model_id": str(model),
                        "true_label": str(true_label),
                        "predicted_label": str(predicted_label),
                        "count": count,
                        "true_appearances": true_appearances,
                        "row_fraction": (
                            float(count / true_appearances)
                            if true_appearances
                            else float("nan")
                        ),
                    }
                )
    columns = [
        "scope",
        "station",
        "model_id",
        "true_label",
        "predicted_label",
        "count",
        "true_appearances",
        "row_fraction",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["station", "model_id", "true_label", "predicted_label"], kind="stable"
        )
        .reset_index(drop=True)
    )


def _reliability_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (station, model), cell in frame.groupby(["station", "model_id"], sort=True):
        for index in range(_RELIABILITY_BINS):
            members = cell[cell.bin_index.eq(index)]
            record = {
                "scope": _SCOPE_POOLED,
                "station": str(station),
                "model_id": str(model),
                "bin_index": index,
                "bin_lower": index / _RELIABILITY_BINS,
                "bin_upper": (index + 1) / _RELIABILITY_BINS,
            }
            if len(members) == 0:
                record.update(
                    {
                        "count": 0,
                        "sum_confidence": 0.0,
                        "sum_correct": 0.0,
                        "mean_confidence": float("nan"),
                        "accuracy": float("nan"),
                    }
                )
            else:
                confidence = members.confidence.to_numpy(dtype=float)
                correct = members.correct_flag.to_numpy(dtype=bool)
                record.update(
                    {
                        "count": len(members),
                        "sum_confidence": float(confidence.sum()),
                        "sum_correct": float(correct.sum()),
                        "mean_confidence": float(confidence.mean()),
                        "accuracy": float(correct.mean()),
                    }
                )
            rows.append(record)
    columns = [
        "scope",
        "station",
        "model_id",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "sum_confidence",
        "sum_correct",
        "mean_confidence",
        "accuracy",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["station", "model_id", "bin_index"], kind="stable")
        .reset_index(drop=True)
    )


def _equal_context_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    context_rows: list[dict] = []
    for (model, context), cell in frame.groupby(["model_id", "context_id"], sort=True):
        total = len(cell)
        ece = 0.0
        for index in range(_RELIABILITY_BINS):
            members = cell[cell.bin_index.eq(index)]
            if len(members) == 0:
                continue
            accuracy = float(members.correct_flag.to_numpy(dtype=bool).mean())
            mean_confidence = float(members.confidence.to_numpy(dtype=float).mean())
            ece += len(members) / total * abs(accuracy - mean_confidence)
        context_rows.append(
            {
                "model_id": str(model),
                "context_id": str(context),
                "context_ece": float(ece),
            }
        )
    per_context = pd.DataFrame(
        context_rows, columns=["model_id", "context_id", "context_ece"]
    )
    summary = per_context.groupby("model_id", as_index=False).agg(
        contexts=("context_id", "nunique"),
        equal_context_ece=("context_ece", "mean"),
    )
    summary["scope"] = _SCOPE_EQUAL_CONTEXT
    columns = ["scope", "model_id", "contexts", "equal_context_ece"]
    return (
        summary[columns].sort_values("model_id", kind="stable").reset_index(drop=True)
    )


def summarize_units(units: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize one endpoint/policy/estimand panel without model selection.

    Balanced accuracy, macro F1 and the calibration fields are inherited
    unchanged from
    :func:`atlas_sers.evaluation.classical.classification_metrics`; the helper
    computes macro F1 over the fixed three-class vocabulary while balanced
    accuracy excludes true classes that are absent from a context (zero
    support). The ``ece`` field returned by ``classification_metrics`` is
    equal-mass-bin ECE, whereas the reliability tables produced here use
    fixed-width bins; the two must not be conflated. Reliability is reported
    both as a pooled repeated-appearance table and as an explicit equal-context
    mean so the two scopes are never conflated.
    """

    frame = _validated_summary(units)
    context_metrics = _context_metrics(frame)
    domain_metrics = _domain_metrics(context_metrics, frame)
    model_summary = _model_summary(domain_metrics, frame)
    return {
        "context_metrics": context_metrics,
        "domain_metrics": domain_metrics,
        "model_summary": model_summary,
        "class_recall": _class_recall(frame),
        "confusion": _confusion(frame),
        "reliability_bins": _reliability_bins(frame),
        "equal_context_reliability": _equal_context_reliability(frame),
    }
