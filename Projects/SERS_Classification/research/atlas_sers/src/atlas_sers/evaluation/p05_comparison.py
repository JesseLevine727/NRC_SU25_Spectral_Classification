"""Pure in-memory comparison of P05 refits against frozen P04/P03 references.

The caller is responsible for authenticating every new frozen prediction before
handing it here; this module never reads a file, fits, selects, imputes or
authorizes any read. Only registered ``held_evaluation`` contexts whose parent
experiment is ``EXP-N00-T3`` are compared. The P05 evaluation family is reported
as ``P05-CORE-T3`` while the parent coordinate is retained as
``source_context_experiment_id``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p04_comparison import (
    CLASSICAL_MODELS,
    normalize_classical_predictions,
)
from atlas_sers.evaluation.p04_results import endpoint_metrics
from atlas_sers.governance.canonical import sha256_value

__all__ = ["P05ComparisonError", "compare_predictions"]

P05_MODELS = ("D0-M", "P05-SELECTED", "D3")
HISTORICAL_MODEL = "D0-ERM"
REFERENCE_MODELS = (*CLASSICAL_MODELS, HISTORICAL_MODEL)
ALL_MODELS = (*P05_MODELS, *REFERENCE_MODELS)
SELECTED_PAIRS = (("P05-SELECTED", "D0-M"), ("D3", "D0-M"))
PAIRS = (
    tuple((model, reference) for model in P05_MODELS for reference in REFERENCE_MODELS)
    + SELECTED_PAIRS
)
P05_EXPERIMENT = "P05-CORE-T3"
SOURCE_HELD_EXPERIMENT = "EXP-N00-T3"
HELD_PHASE = "held_evaluation"
PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
AGGREGATIONS = ("M01", "M06")
CONTEXT_METADATA = (
    "context_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_fold",
)
PREDICTION_COLUMNS = (
    "context_id",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "true_label",
    "class_vocabulary",
    *PROBABILITY_COLUMNS,
)
STANDARD_COLUMNS = (
    *CONTEXT_METADATA,
    "experiment_id",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "true_label",
    "class_vocabulary",
    "candidate_id",
    *PROBABILITY_COLUMNS,
    "predicted_label",
    "model_id",
    "comparison_model_id",
    "model_family",
)
METRIC_COLUMNS = (
    "observations",
    "physical_masters",
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece",
)
DELTA_METRICS = ("balanced_accuracy", "macro_f1", "negative_log_likelihood", "ece")


class P05ComparisonError(ValueError):
    """Raised when P05 comparison inputs are missing or inconsistent."""


def _family(model: str) -> str:
    if model in P05_MODELS:
        return "p05"
    if model == HISTORICAL_MODEL:
        return "historical"
    return "classical"


def _require(frame: pd.DataFrame, columns, code: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise P05ComparisonError(f"{code}:{','.join(missing)}")


def _classes(value) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, np.ndarray):
        parsed = value.tolist()
    else:
        parsed = value
    if isinstance(parsed, (str, bytes)) or not isinstance(parsed, Sequence):
        raise P05ComparisonError("class_vocabulary_invalid")
    classes = tuple(parsed)
    if len(classes) != 3 or any(
        not isinstance(item, str) or not item or item != item.strip() for item in classes
    ):
        raise P05ComparisonError("class_vocabulary_invalid")
    if len(set(classes)) != 3:
        raise P05ComparisonError("class_vocabulary_invalid")
    if classes != tuple(sorted(classes)):
        raise P05ComparisonError("class_vocabulary_invalid")
    return classes


def _integral(value) -> int:
    if isinstance(value, bool) or value is None:
        raise P05ComparisonError("context_coordinate_invalid")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number) or number != int(number):
            raise P05ComparisonError("context_coordinate_invalid")
        return int(number)
    if isinstance(value, str):
        text = value
        if not text or text != text.strip():
            raise P05ComparisonError("context_coordinate_invalid")
        try:
            number = int(text)
        except ValueError:
            raise P05ComparisonError("context_coordinate_invalid") from None
        if str(number) != text:
            raise P05ComparisonError("context_coordinate_invalid")
        return number
    raise P05ComparisonError("context_coordinate_invalid")


def _metadata_agrees(provided, expected, column: str) -> bool:
    if provided is None or (not isinstance(provided, (list, np.ndarray)) and pd.isna(provided)):
        return False
    if column in ("outer_repeat", "outer_fold"):
        return _integral(provided) == _integral(expected)
    return str(provided) == str(expected)


def _validate_probabilities(frame: pd.DataFrame, code: str) -> None:
    values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if values.shape != (len(frame), len(PROBABILITY_COLUMNS)):
        raise P05ComparisonError(f"{code}_shape")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise P05ComparisonError(f"{code}_range")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise P05ComparisonError(f"{code}_normalization")


def _predicted_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for row in frame.itertuples(index=False):
        classes = _classes(row.class_vocabulary)
        probabilities = np.asarray(
            [getattr(row, column) for column in PROBABILITY_COLUMNS], dtype=float
        )
        labels.append(classes[int(probabilities.argmax())])
    return labels


def _empty_standard() -> pd.DataFrame:
    return pd.DataFrame(columns=list(STANDARD_COLUMNS))


def _identity(context, *, model, comparison=None, reference=None) -> dict:
    record = {
        "context_id": str(context.context_id),
        "experiment_id": P05_EXPERIMENT,
        "source_context_experiment_id": SOURCE_HELD_EXPERIMENT,
        "domain": str(context.domain),
        "station": str(context.station),
        "held_instrument": str(context.held_instrument),
        "outer_repeat": context.outer_repeat,
        "outer_fold": context.outer_fold,
        "model_id": model,
        "comparison_model_id": comparison or model,
        "model_family": _family(model),
    }
    if reference is not None:
        record["reference_model_id"] = reference
        record["reference_family"] = _family(reference)
    return record


def _held_contexts(contexts: pd.DataFrame) -> pd.DataFrame:
    frame = contexts.copy()
    _require(
        frame,
        (
            *CONTEXT_METADATA,
            "phase_gate",
            "experiment_id",
            "outer_test_uid_sha256",
        ),
        "context_columns_missing",
    )
    source = frame.experiment_id.astype(str)
    phase_gate = frame.phase_gate.astype(str)
    held = frame[source.eq(SOURCE_HELD_EXPERIMENT) & phase_gate.eq(HELD_PHASE)].copy()
    if held.empty:
        raise P05ComparisonError("held_contexts_empty")
    held["context_id"] = held.context_id.astype(str)
    if held.context_id.duplicated().any():
        raise P05ComparisonError("held_context_duplicate")
    if held.outer_test_uid_sha256.isna().any():
        raise P05ComparisonError("held_context_hash_missing")
    for column in ("context_id", "domain", "station", "held_instrument"):
        values = held[column]
        if values.isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise P05ComparisonError("held_context_metadata_malformed")
    held["outer_repeat"] = held.outer_repeat.map(_integral)
    held["outer_fold"] = held.outer_fold.map(_integral)
    return held


def _attach_context(frame: pd.DataFrame, context_meta: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    lookup = context_meta.set_index("context_id")
    for column in CONTEXT_METADATA:
        if column == "context_id" or column not in frame.columns:
            continue
        provided = frame[["context_id", column]].drop_duplicates()
        for row in provided.itertuples(index=False):
            context_id = str(row.context_id)
            if context_id not in lookup.index:
                continue
            if not _metadata_agrees(getattr(row, column), lookup.at[context_id, column], column):
                raise P05ComparisonError(f"context_metadata_conflict:{column}")
    drop = [
        column for column in CONTEXT_METADATA if column in frame.columns and column != "context_id"
    ]
    frame = frame.drop(columns=drop)
    frame = frame.merge(context_meta, on="context_id", how="left", validate="many_to_one")
    if frame[list(CONTEXT_METADATA[1:])].isna().any().any():
        raise P05ComparisonError("context_metadata_missing")
    frame["experiment_id"] = P05_EXPERIMENT
    return frame


def _prepare_p05(p05_ensemble: pd.DataFrame, held_ids: set[str]) -> pd.DataFrame:
    frame = p05_ensemble.copy()
    _require(
        frame,
        (*PREDICTION_COLUMNS, "experiment_id", "source_context_experiment_id", "model_id"),
        "p05_columns_missing",
    )
    frame = frame[frame.experiment_id.astype(str).eq(P05_EXPERIMENT)].copy()
    if frame.empty:
        raise P05ComparisonError("p05_ensemble_empty")
    if not frame.source_context_experiment_id.eq(SOURCE_HELD_EXPERIMENT).all():
        raise P05ComparisonError("p05_parent_experiment_mismatch")
    for column in (
        "context_id",
        "model_id",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "true_label",
    ):
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in frame[column]
        ):
            raise P05ComparisonError("p05_identity_malformed")
    frame["context_id"] = frame.context_id.astype(str)
    if not frame.context_id.isin(held_ids).all():
        raise P05ComparisonError("p05_unregistered_context")
    if not frame.model_id.astype(str).isin(P05_MODELS).all():
        raise P05ComparisonError("p05_unknown_model")
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = frame.model_id
    frame["observation_uid"] = frame.observation_uid.astype(str)
    _validate_probabilities(frame, "p05_probability_invalid")
    if frame.duplicated(["context_id", "model_id", "observation_uid"]).any():
        raise P05ComparisonError("p05_duplicate_uid")
    return frame


def _prepare_historical(
    p04_ensemble: pd.DataFrame, held_ids: set[str], context_meta: pd.DataFrame
) -> pd.DataFrame:
    if p04_ensemble is None or p04_ensemble.empty:
        return _empty_standard()
    frame = p04_ensemble.copy()
    _require(
        frame,
        (*PREDICTION_COLUMNS, "experiment_id", "model_id"),
        "historical_columns_missing",
    )
    frame = frame[frame.experiment_id.astype(str).eq(SOURCE_HELD_EXPERIMENT)].copy()
    if frame.empty:
        return _empty_standard()
    if not frame.model_id.astype(str).eq(HISTORICAL_MODEL).all():
        raise P05ComparisonError("historical_model_unexpected")
    frame["context_id"] = frame.context_id.astype(str)
    if not frame.context_id.isin(held_ids).all():
        raise P05ComparisonError("historical_unregistered_context")
    frame = _attach_context(frame, context_meta)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = HISTORICAL_MODEL
    frame["comparison_model_id"] = HISTORICAL_MODEL
    frame["model_id"] = HISTORICAL_MODEL
    frame["model_family"] = "historical"
    return frame


def _prepare_classical(
    p03_predictions, held: pd.DataFrame, held_ids: set[str], context_meta: pd.DataFrame
) -> pd.DataFrame:
    if p03_predictions is None or p03_predictions.empty:
        return _empty_standard()
    frame = p03_predictions.copy()
    _require(
        frame,
        ("experiment_id", "model_id", "domain", "outer_repeat", "outer_fold"),
        "classical_columns_missing",
    )
    experiment = frame.experiment_id.astype(str)
    model = frame.model_id.astype(str)
    selected = experiment.eq("EXP-C09-T3") | (
        experiment.eq("EXP-C10-T3") & model.isin(CLASSICAL_MODELS[1:])
    )
    frame = frame[selected].copy()
    if frame.empty:
        return _empty_standard()
    frame["outer_repeat"] = frame.outer_repeat.map(_integral)
    frame["outer_fold"] = frame.outer_fold.map(_integral)
    classical_contexts = held[["context_id", "domain", "outer_repeat", "outer_fold"]].copy()
    classical_contexts["experiment_id"] = SOURCE_HELD_EXPERIMENT
    frame = normalize_classical_predictions(frame, classical_contexts)
    if frame.empty:
        return _empty_standard()
    frame["context_id"] = frame.context_id.astype(str)
    if not frame.context_id.isin(held_ids).all():
        raise P05ComparisonError("classical_unregistered_context")
    frame = _attach_context(frame, context_meta)
    frame["comparison_model_id"] = frame.comparison_model_id.astype(str)
    frame["model_id"] = frame.comparison_model_id
    frame["model_family"] = "classical"
    return frame


def _establish_truth(p05: pd.DataFrame, held: pd.DataFrame) -> dict[str, dict]:
    hashes = {
        str(row.context_id): str(row.outer_test_uid_sha256) for row in held.itertuples(index=False)
    }
    grouped = {str(context_id): cell for context_id, cell in p05.groupby("context_id", sort=True)}
    if set(grouped) != set(hashes):
        raise P05ComparisonError("p05_held_context_incomplete")
    truth: dict[str, dict] = {}
    for context_id, cell in grouped.items():
        vocabularies = {
            json.dumps(list(_classes(value)), separators=(",", ":"))
            for value in cell.class_vocabulary
        }
        if len(vocabularies) != 1:
            raise P05ComparisonError("p05_class_order_disagreement")
        classes = _classes(cell.class_vocabulary.iloc[0])
        true_labels = {str(value) for value in cell.true_label}
        if not true_labels <= set(classes):
            raise P05ComparisonError("p05_true_label_outside_class_vocabulary")
        per_model = {}
        for model in P05_MODELS:
            rows = cell[cell.model_id.astype(str).eq(model)]
            if rows.empty:
                raise P05ComparisonError("p05_missing_model")
            per_model[model] = frozenset(rows.observation_uid.astype(str))
        uids = per_model[P05_MODELS[0]]
        if any(per_model[model] != uids for model in P05_MODELS[1:]):
            raise P05ComparisonError("p05_uid_set_disagreement")
        if sha256_value(sorted(uids)) != hashes[context_id]:
            raise P05ComparisonError("p05_uid_hash_mismatch")
        identity = (
            cell.assign(
                observation_uid=cell.observation_uid.astype(str),
                master_sample_id=cell.master_sample_id.astype(str),
                true_label=cell.true_label.astype(str),
                instrument=cell.instrument.astype(str),
            )
            .groupby("observation_uid", as_index=False)
            .agg(
                master_sample_id=("master_sample_id", "nunique"),
                true_label=("true_label", "nunique"),
                instrument=("instrument", "nunique"),
            )
        )
        if (identity[["master_sample_id", "true_label", "instrument"]] > 1).to_numpy().any():
            raise P05ComparisonError("p05_metadata_disagreement")
        base = cell[cell.model_id.astype(str).eq(P05_MODELS[0])].drop_duplicates("observation_uid")
        frame = base.set_index(base.observation_uid.astype(str))[
            ["master_sample_id", "true_label", "instrument"]
        ].astype(str)
        truth[context_id] = {
            "classes": classes,
            "uids": uids,
            "frame": frame,
            "complete": True,
        }
    return truth


def _standard_p05(p05: pd.DataFrame) -> pd.DataFrame:
    frame = p05.copy()
    frame["comparison_model_id"] = frame.model_id.astype(str)
    frame["model_family"] = "p05"
    frame["predicted_label"] = _predicted_labels(frame)
    return frame[list(STANDARD_COLUMNS)].reset_index(drop=True)


def _validate_reference(frame: pd.DataFrame, truth: dict[str, dict], code: str) -> pd.DataFrame:
    _validate_probabilities(frame, f"{code}_probability_invalid")
    if frame.duplicated(["comparison_model_id", "context_id", "observation_uid"]).any():
        raise P05ComparisonError(f"{code}_duplicate_uid")
    for context_id, cell in frame.groupby("context_id", sort=True):
        context_id = str(context_id)
        if context_id not in truth:
            raise P05ComparisonError(f"{code}_context_without_truth")
        entry = truth[context_id]
        for value in cell.class_vocabulary:
            if _classes(value) != entry["classes"]:
                raise P05ComparisonError(f"{code}_class_order_mismatch")
        index = entry["frame"]
        for row in cell.itertuples(index=False):
            uid = str(row.observation_uid)
            if uid not in index.index:
                raise P05ComparisonError(f"{code}_unexpected_uid")
            expected = index.loc[uid]
            if (
                str(row.master_sample_id) != str(expected.master_sample_id)
                or str(row.true_label) != str(expected.true_label)
                or str(row.instrument) != str(expected.instrument)
            ):
                raise P05ComparisonError(f"{code}_metadata_mismatch")
    frame["predicted_label"] = _predicted_labels(frame)
    return frame


def _observed_sets(frame: pd.DataFrame) -> dict[str, frozenset]:
    if frame.empty:
        return {}
    return {
        str(context_id): frozenset(cell.observation_uid.astype(str))
        for context_id, cell in frame.groupby("context_id", sort=True)
    }


def _coverage(
    held: pd.DataFrame, truth: dict[str, dict], observed: dict[str, dict]
) -> pd.DataFrame:
    rows = []
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        expected_hash = str(context.outer_test_uid_sha256)
        entry = truth.get(context_id)
        expected_rows = len(entry["uids"]) if entry else np.nan
        for model in ALL_MODELS:
            uids = sorted(observed.get(model, {}).get(context_id, frozenset()))
            observed_hash = sha256_value(uids)
            complete = bool(
                entry
                and entry["complete"]
                and len(uids) == len(entry["uids"])
                and frozenset(uids) == entry["uids"]
                and observed_hash == expected_hash
            )
            rows.append(
                {
                    **_identity(context, model=model),
                    "expected_test_rows": expected_rows,
                    "observed_test_rows": len(uids),
                    "expected_test_uid_sha256": expected_hash,
                    "observed_test_uid_sha256": observed_hash,
                    "complete": complete,
                    "reason_code": ("complete" if complete else "incomplete_or_missing_reference"),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_table(
    held: pd.DataFrame, model_frames: dict[str, pd.DataFrame], coverage: pd.DataFrame
) -> pd.DataFrame:
    complete_lookup = {
        (str(row.model_id), str(row.context_id)): bool(row.complete)
        for row in coverage.itertuples(index=False)
    }
    rows = []
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        for model in ALL_MODELS:
            if not complete_lookup.get((model, context_id), False):
                continue
            subset = model_frames[model]
            subset = subset[subset.context_id.astype(str).eq(context_id)]
            spectrum, master = endpoint_metrics(subset)
            for aggregation, scored in (("M01", spectrum), ("M06", master)):
                if len(scored) != 1:
                    raise P05ComparisonError("endpoint_group_ambiguity")
                metric = scored.iloc[0]
                record = {
                    **_identity(context, model=model),
                    "aggregation_id": aggregation,
                    "complete": True,
                }
                for column in METRIC_COLUMNS:
                    record[column] = metric[column]
                record["test_uid_sha256"] = metric["test_uid_sha256"]
                rows.append(record)
    return pd.DataFrame(rows)


def _paired_table(
    held: pd.DataFrame, endpoint: pd.DataFrame, coverage: pd.DataFrame
) -> pd.DataFrame:
    endpoint_lookup = {
        (str(row.model_id), str(row.context_id), str(row.aggregation_id)): row
        for row in endpoint.itertuples(index=False)
    }
    complete_lookup = {
        (str(row.model_id), str(row.context_id)): bool(row.complete)
        for row in coverage.itertuples(index=False)
    }
    rows = []
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        for aggregation in AGGREGATIONS:
            for model, reference in PAIRS:
                common = bool(
                    complete_lookup.get((model, context_id), False)
                    and complete_lookup.get((reference, context_id), False)
                )
                model_complete = complete_lookup.get((model, context_id), False)
                reference_complete = complete_lookup.get((reference, context_id), False)
                record = {
                    **_identity(context, model=model, comparison=reference, reference=reference),
                    "aggregation_id": aggregation,
                    "model_complete": model_complete,
                    "reference_complete": reference_complete,
                    "common_complete": common,
                }
                for column in METRIC_COLUMNS:
                    record[f"model_{column}"] = np.nan
                    record[f"reference_{column}"] = np.nan
                for column in DELTA_METRICS:
                    record[f"delta_{column}"] = np.nan
                if model_complete:
                    model_row = endpoint_lookup[(model, context_id, aggregation)]
                    for column in METRIC_COLUMNS:
                        record[f"model_{column}"] = getattr(model_row, column)
                if reference_complete:
                    reference_row = endpoint_lookup[(reference, context_id, aggregation)]
                    for column in METRIC_COLUMNS:
                        record[f"reference_{column}"] = getattr(reference_row, column)
                if common:
                    for column in DELTA_METRICS:
                        record[f"delta_{column}"] = float(
                            getattr(model_row, column) - getattr(reference_row, column)
                        )
                rows.append(record)
    return pd.DataFrame(rows)


def _summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["station", "model_id", "reference_model_id", "aggregation_id"]
    for keys, cell in paired.groupby(groups, sort=True, dropna=False):
        station, model_id, reference_model_id, aggregation_id = keys
        common = cell[cell.common_complete]
        planned = len(cell)
        complete_count = len(common)
        planned_domains = int(cell.domain.nunique())
        contributing_common_domains = int(common.domain.nunique())
        if common.empty:
            model_mean = reference_mean = np.nan
            model_domain_mean = reference_domain_mean = np.nan
            delta_ba = delta_f1 = delta_nll = delta_ece = np.nan
            model_worst = reference_worst = np.nan
        else:
            model_mean = float(common.model_balanced_accuracy.mean())
            reference_mean = float(common.reference_balanced_accuracy.mean())
            delta_ba = float(common.delta_balanced_accuracy.mean())
            delta_f1 = float(common.delta_macro_f1.mean())
            delta_nll = float(common.delta_negative_log_likelihood.mean())
            delta_ece = float(common.delta_ece.mean())
            model_domain = common.groupby("domain", as_index=False).model_balanced_accuracy.mean()
            reference_domain = common.groupby(
                "domain", as_index=False
            ).reference_balanced_accuracy.mean()
            model_domain_mean = float(model_domain.model_balanced_accuracy.mean())
            reference_domain_mean = float(reference_domain.reference_balanced_accuracy.mean())
            model_worst = float(model_domain.model_balanced_accuracy.min())
            reference_worst = float(reference_domain.reference_balanced_accuracy.min())
        rows.append(
            {
                "station": station,
                "model_id": model_id,
                "reference_model_id": reference_model_id,
                "aggregation_id": aggregation_id,
                "planned_contexts": planned,
                "common_complete_contexts": complete_count,
                "common_coverage": complete_count / planned if planned else np.nan,
                "planned_domains": planned_domains,
                "contributing_common_domains": contributing_common_domains,
                "model_mean_balanced_accuracy_equal_contexts": model_mean,
                "reference_mean_balanced_accuracy_equal_contexts": reference_mean,
                "model_mean_balanced_accuracy_equal_domains": model_domain_mean,
                "reference_mean_balanced_accuracy_equal_domains": reference_domain_mean,
                "mean_delta_balanced_accuracy": delta_ba,
                "mean_delta_macro_f1": delta_f1,
                "mean_delta_negative_log_likelihood": delta_nll,
                "mean_delta_ece": delta_ece,
                "model_worst_domain_balanced_accuracy_common": model_worst,
                "reference_worst_domain_balanced_accuracy_common": reference_worst,
                "model_failure_sensitive_mean_balanced_accuracy_missing_as_zero": float(
                    cell.model_balanced_accuracy.fillna(0.0).mean()
                ),
                "reference_failure_sensitive_mean_balanced_accuracy_missing_as_zero": float(
                    cell.reference_balanced_accuracy.fillna(0.0).mean()
                ),
                "failure_sensitive_missing_policy": (
                    "missing assigned zero; not imputed valid scores"
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_predictions(
    *,
    p05_ensemble: pd.DataFrame,
    p04_ensemble: pd.DataFrame,
    p03_predictions: pd.DataFrame,
    contexts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare three P05 strategies against four classical and one historical model.

    Callers must authenticate every new frozen prediction before invoking this
    function. Reference tables may be partial or absent; partial references are
    reported as incomplete and never scored as complete.
    """

    held = _held_contexts(contexts)
    held_ids = set(held.context_id.astype(str))
    context_meta = held[list(CONTEXT_METADATA)].copy()

    p05 = _prepare_p05(p05_ensemble, held_ids)
    truth = _establish_truth(p05, held)
    p05_standard = _standard_p05(_attach_context(p05, context_meta))

    historical = _prepare_historical(p04_ensemble, held_ids, context_meta)
    if not historical.empty:
        historical = _validate_reference(historical, truth, "historical")
    classical = _prepare_classical(p03_predictions, held, held_ids, context_meta)
    if not classical.empty:
        classical = _validate_reference(classical, truth, "classical")

    model_frames = {model: _empty_standard() for model in ALL_MODELS}
    for model in P05_MODELS:
        model_frames[model] = p05_standard[p05_standard.model_id.eq(model)].reset_index(drop=True)
    for source in (historical, classical):
        if source.empty:
            continue
        for model, cell in source.groupby("comparison_model_id", sort=True):
            model_frames[str(model)] = cell[list(STANDARD_COLUMNS)].reset_index(drop=True)

    observed = {model: _observed_sets(model_frames[model]) for model in ALL_MODELS}
    coverage = _coverage(held, truth, observed)
    endpoint = _endpoint_table(held, model_frames, coverage)
    paired = _paired_table(held, endpoint, coverage)
    summary = _summary(paired)
    return {
        "endpoint_metrics": endpoint,
        "paired_metrics": paired,
        "coverage": coverage,
        "summary": summary,
    }
