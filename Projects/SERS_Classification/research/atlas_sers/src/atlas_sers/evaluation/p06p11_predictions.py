"""Pure in-memory P06/P11 adapter over frozen P05/P04/P03 predictions.

The adapter is read-free: it never loads, samples, fits, imputes, rescales or
authenticates any input. Callers must hand it already-frozen prediction tables
and an already-authenticated frozen endpoint-metric table.

Preparation deliberately reuses the established
``atlas_sers.evaluation.p05_comparison`` helpers -- including its private
``_held_contexts``, ``_prepare_p05``, ``_establish_truth``, ``_attach_context``,
``_standard_p05``, ``_prepare_historical``, ``_prepare_classical``,
``_validate_reference``, ``_coverage`` and ``_observed_sets`` -- so that the
probability, class-order and truth policies cannot drift away from the
historical comparison module. Regression tests bind that historical API: if
those helpers change, this adapter must follow them rather than reimplementing
a parallel policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_comparison as _p05c
from atlas_sers.evaluation.classical import instrument_balanced_master_probabilities
from atlas_sers.governance.canonical import sha256_value

__all__ = [
    "PredictionAuditError",
    "prepare_panel",
    "pair_units",
    "audit_point_estimates",
]

AGGREGATIONS = ("M01", "M06")
PROBABILITY_COLUMNS = tuple(_p05c.PROBABILITY_COLUMNS)
ALL_MODELS = tuple(_p05c.ALL_MODELS)
ALL_MODELS_SET = frozenset(ALL_MODELS)
PAIR_SET = frozenset(_p05c.PAIRS)

UNIT_COLUMNS = (
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
UNIT_REQUIRED = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "model_id",
    "class_vocabulary",
    "correct",
)
PAIR_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "correct_model",
    "correct_reference",
)
AUDIT_COLUMNS = (
    "model_id",
    "context_id",
    "aggregation_id",
    "balanced_accuracy",
    "reference_balanced_accuracy",
    "absolute_error",
)
FROZEN_ENDPOINT_COLUMNS = ("model_id", "context_id", "aggregation_id", "balanced_accuracy")
AUDIT_ATOL = 1e-12
MASTER_UNIT_PREFIX = "master-"
UNIT_DUPLICATE_CODE = "panel_duplicate_unit"

_STANDARD_REQUIRED = (
    "context_id",
    "domain",
    "station",
    "master_sample_id",
    "instrument",
    "true_label",
    "class_vocabulary",
    "predicted_label",
    *PROBABILITY_COLUMNS,
)
_M01_REQUIRED = (*_STANDARD_REQUIRED, "observation_uid")


class PredictionAuditError(ValueError):
    """Raised when a frozen P06/P11 prediction panel is inconsistent."""


def _require_unique_columns(frame: pd.DataFrame, code: str) -> None:
    if frame.columns.duplicated().any():
        raise PredictionAuditError(code)


def _require_columns(frame: pd.DataFrame, columns, code: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PredictionAuditError(f"{code}:{','.join(missing)}")


def _check_input_columns(frame) -> None:
    if frame is None:
        return
    if not isinstance(frame, pd.DataFrame):
        raise PredictionAuditError("panel_input_not_frame")
    _require_unique_columns(frame, "panel_duplicate_columns")


def _require_panel(panel) -> None:
    if not isinstance(panel, dict):
        raise PredictionAuditError("panel_not_mapping")
    for aggregation in AGGREGATIONS:
        if aggregation not in panel or not isinstance(panel[aggregation], pd.DataFrame):
            raise PredictionAuditError("panel_missing_aggregation")
    if "coverage" not in panel or not isinstance(panel["coverage"], pd.DataFrame):
        raise PredictionAuditError("panel_missing_coverage")
    complete = _complete_lookup(panel["coverage"])
    expected = {key for key, flag in complete.items() if flag}
    for aggregation in AGGREGATIONS:
        units = panel[aggregation]
        _validate_units(units)
        observed = set(zip(units["model_id"], units["context_id"], strict=True))
        for key in observed:
            if key not in complete:
                raise PredictionAuditError("unexpected_unit_context")
            if not complete[key]:
                raise PredictionAuditError("unit_on_incomplete_context")
        if expected - observed:
            raise PredictionAuditError("coverage_unit_disagreement")


def _complete_lookup(coverage: pd.DataFrame) -> dict[tuple[str, str], bool]:
    if not isinstance(coverage, pd.DataFrame):
        raise PredictionAuditError("coverage_not_frame")
    _require_unique_columns(coverage, "coverage_duplicate_columns")
    _require_columns(coverage, ("model_id", "context_id", "complete"), "coverage_columns_missing")
    lookup: dict[tuple[str, str], bool] = {}
    for row in coverage.itertuples(index=False):
        model = _strict_identifier(row.model_id, "coverage_model_invalid")
        context = _strict_identifier(row.context_id, "coverage_context_invalid")
        if model not in ALL_MODELS_SET:
            raise PredictionAuditError("coverage_unknown_model")
        flag = row.complete
        if not isinstance(flag, (bool, np.bool_)):
            raise PredictionAuditError("coverage_flag_not_bool")
        key = (model, context)
        if key in lookup:
            raise PredictionAuditError("coverage_duplicate_key")
        lookup[key] = bool(flag)
    return lookup


def _held_instrument_lookup(held: pd.DataFrame) -> dict[str, str]:
    return {str(row.context_id): str(row.held_instrument) for row in held.itertuples(index=False)}


def _coerce_correct(value, code: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise PredictionAuditError(code)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number in (0.0, 1.0):
            return bool(number)
        raise PredictionAuditError(code)
    raise PredictionAuditError(code)


_UNIT_IDENTITY_FIELDS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
)
_UNIT_DEPENDENCIES = (
    (("unit_id",), ("master_sample_id", "station", "instrument", "true_label")),
    (("master_sample_id",), ("station", "true_label")),
    (("context_id",), ("domain", "station", "instrument")),
    (("domain",), ("station", "instrument")),
)


def _strict_identifier(value, code: str) -> str:
    if not isinstance(value, str):
        raise PredictionAuditError(code)
    if not value or value != value.strip():
        raise PredictionAuditError(code)
    return value


def _validate_units(units: pd.DataFrame) -> None:
    _require_unique_columns(units, "panel_duplicate_columns")
    _require_columns(units, UNIT_REQUIRED, "panel_columns_missing")
    if units.empty:
        return
    working = units.loc[:, list(UNIT_REQUIRED)].copy()
    for row in working.itertuples(index=False):
        for field in _UNIT_IDENTITY_FIELDS:
            _strict_identifier(getattr(row, field), "unit_identity_invalid")
        model = _strict_identifier(row.model_id, "unit_model_invalid")
        if model not in ALL_MODELS_SET:
            raise PredictionAuditError("unit_unknown_model")
        _coerce_correct(row.correct, "unit_correctness_invalid")
    if working.duplicated(["model_id", "context_id", "unit_id"]).any():
        raise PredictionAuditError("panel_duplicate_unit")
    for keys, dependents in _UNIT_DEPENDENCIES:
        counts = working.groupby(list(keys), dropna=False)[list(dependents)].nunique(dropna=False)
        if (counts > 1).to_numpy().any():
            raise PredictionAuditError("unit_identity_mismatch")
    vocabularies = [_p05c._classes(value) for value in working["class_vocabulary"]]
    working["class_vocabulary"] = [
        value if isinstance(value, str) else tuple(value) for value in vocabularies
    ]
    grouped = working.groupby("context_id", dropna=False)["class_vocabulary"].nunique(dropna=False)
    if (grouped > 1).any():
        raise PredictionAuditError("unit_vocabulary_mismatch")


def _finite(value, code: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise PredictionAuditError(code)
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise PredictionAuditError(code)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PredictionAuditError(code) from None
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PredictionAuditError(code)
    return number


def _complete_contexts(complete: dict[tuple[str, str], bool], model: str) -> set[str]:
    return {
        context for (row_model, context), flag in complete.items() if row_model == model and flag
    }


def _grouped_units(units: pd.DataFrame, model: str) -> dict[str, pd.DataFrame]:
    subset = units[units["model_id"].astype(str).eq(model)]
    return {str(context): cell for context, cell in subset.groupby("context_id", sort=True)}


def _index_units(cell: pd.DataFrame) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in cell.itertuples(index=False):
        unit_id = str(row.unit_id)
        if unit_id in index:
            raise PredictionAuditError("duplicate_unit_id")
        index[unit_id] = {
            "context_id": str(row.context_id),
            "domain": str(row.domain),
            "station": str(row.station),
            "instrument": str(row.instrument),
            "master_sample_id": str(row.master_sample_id),
            "unit_id": unit_id,
            "true_label": str(row.true_label),
            "class_vocabulary": _p05c._classes(row.class_vocabulary),
            "correct": _coerce_correct(row.correct, "unit_correctness_invalid"),
        }
    return index


def _require_matching_identity(model_row: dict, reference_row: dict) -> None:
    for field in (
        "context_id",
        "domain",
        "station",
        "instrument",
        "master_sample_id",
        "unit_id",
        "true_label",
        "class_vocabulary",
    ):
        if model_row[field] != reference_row[field]:
            raise PredictionAuditError("unit_identity_mismatch")


def _model_units_m01(
    model: str,
    frame: pd.DataFrame,
    complete_lookup: dict[tuple[str, str], bool],
    instrument_lookup: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    if frame.empty:
        return rows
    _require_unique_columns(frame, "panel_duplicate_columns")
    _require_columns(frame, _M01_REQUIRED, "panel_columns_missing")
    for context_id, cell in frame.groupby("context_id", sort=True):
        context_id = str(context_id)
        if not complete_lookup.get((model, context_id), False):
            continue
        expected = instrument_lookup.get(context_id)
        if expected is None:
            raise PredictionAuditError("panel_context_not_held")
        classes = _p05c._classes(cell.class_vocabulary.iloc[0])
        domain = str(cell.domain.iloc[0])
        station = str(cell.station.iloc[0])
        for row in cell.itertuples(index=False):
            instrument = str(row.instrument)
            if instrument != expected:
                raise PredictionAuditError("panel_instrument_mismatch")
            true_label = str(row.true_label)
            predicted_label = str(row.predicted_label)
            rows.append(
                {
                    "context_id": context_id,
                    "domain": domain,
                    "station": station,
                    "instrument": instrument,
                    "master_sample_id": str(row.master_sample_id),
                    "unit_id": str(row.observation_uid),
                    "true_label": true_label,
                    "model_id": model,
                    "class_vocabulary": classes,
                    "probability_0": float(row.probability_0),
                    "probability_1": float(row.probability_1),
                    "probability_2": float(row.probability_2),
                    "predicted_label": predicted_label,
                    "correct": bool(true_label == predicted_label),
                }
            )
    return rows


def _model_units_m06(
    model: str,
    frame: pd.DataFrame,
    complete_lookup: dict[tuple[str, str], bool],
    instrument_lookup: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    if frame.empty:
        return rows
    _require_unique_columns(frame, "panel_duplicate_columns")
    _require_columns(frame, _STANDARD_REQUIRED, "panel_columns_missing")
    for context_id, cell in frame.groupby("context_id", sort=True):
        context_id = str(context_id)
        if not complete_lookup.get((model, context_id), False):
            continue
        expected = instrument_lookup.get(context_id)
        if expected is None:
            raise PredictionAuditError("panel_context_not_held")
        instruments = cell.instrument.astype(str)
        if not instruments.eq(expected).all():
            raise PredictionAuditError("panel_instrument_mismatch")
        classes = _p05c._classes(cell.class_vocabulary.iloc[0])
        domain = str(cell.domain.iloc[0])
        station = str(cell.station.iloc[0])
        master = instrument_balanced_master_probabilities(
            probabilities=cell[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
            true_labels=cell.true_label.astype(str).to_numpy(),
            master_ids=cell.master_sample_id.astype(str).to_numpy(),
            instruments=instruments.to_numpy(),
            class_vocabulary=classes,
        )
        for row in master.itertuples(index=False):
            master_id = str(row.master_sample_id)
            probability = [float(value) for value in row.probabilities]
            true_label = str(row.true_label)
            predicted_label = str(row.predicted_label)
            rows.append(
                {
                    "context_id": context_id,
                    "domain": domain,
                    "station": station,
                    "instrument": expected,
                    "master_sample_id": master_id,
                    "unit_id": MASTER_UNIT_PREFIX + sha256_value([domain, master_id]),
                    "true_label": true_label,
                    "model_id": model,
                    "class_vocabulary": classes,
                    "probability_0": probability[0],
                    "probability_1": probability[1],
                    "probability_2": probability[2],
                    "predicted_label": predicted_label,
                    "correct": bool(true_label == predicted_label),
                }
            )
    return rows


def _finalize_units(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(UNIT_COLUMNS))
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
    if frame.duplicated(["model_id", "context_id", "unit_id"]).any():
        raise PredictionAuditError(UNIT_DUPLICATE_CODE)
    frame = frame.sort_values(["model_id", "context_id", "unit_id"], kind="stable").reset_index(
        drop=True
    )
    return frame


def prepare_panel(
    *,
    p05_ensemble: pd.DataFrame,
    p04_ensemble: pd.DataFrame,
    p03_predictions: pd.DataFrame,
    contexts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Convert complete model/context groups into M01 and M06 prediction units.

    Incomplete groups stay documented in the returned coverage table and are
    never scored from a small row intersection. The returned mapping always
    contains the keys ``"M01"``, ``"M06"`` and ``"coverage"``.
    """

    for frame in (p05_ensemble, p04_ensemble, p03_predictions, contexts):
        _check_input_columns(frame)

    held = _p05c._held_contexts(contexts)
    held_ids = set(held.context_id.astype(str))
    context_meta = held[list(_p05c.CONTEXT_METADATA)].copy()

    p05 = _p05c._prepare_p05(p05_ensemble, held_ids)
    truth = _p05c._establish_truth(p05, held)
    p05_standard = _p05c._standard_p05(_p05c._attach_context(p05, context_meta))

    historical = _p05c._prepare_historical(p04_ensemble, held_ids, context_meta)
    if not historical.empty:
        historical = _p05c._validate_reference(historical, truth, "historical")
    classical = _p05c._prepare_classical(p03_predictions, held, held_ids, context_meta)
    if not classical.empty:
        classical = _p05c._validate_reference(classical, truth, "classical")

    model_frames = {model: _p05c._empty_standard() for model in ALL_MODELS}
    for model in _p05c.P05_MODELS:
        model_frames[model] = p05_standard[p05_standard.model_id.eq(model)].reset_index(drop=True)
    for source in (historical, classical):
        if source.empty:
            continue
        for model, cell in source.groupby("comparison_model_id", sort=True):
            model_frames[str(model)] = cell[list(_p05c.STANDARD_COLUMNS)].reset_index(drop=True)

    observed = {model: _p05c._observed_sets(model_frames[model]) for model in ALL_MODELS}
    coverage = _p05c._coverage(held, truth, observed)
    complete_lookup = _complete_lookup(coverage)
    instrument_lookup = _held_instrument_lookup(held)

    m01_rows: list[dict] = []
    m06_rows: list[dict] = []
    for model in ALL_MODELS:
        frame = model_frames[model]
        m01_rows.extend(_model_units_m01(model, frame, complete_lookup, instrument_lookup))
        m06_rows.extend(_model_units_m06(model, frame, complete_lookup, instrument_lookup))

    return {
        "M01": _finalize_units(m01_rows),
        "M06": _finalize_units(m06_rows),
        "coverage": coverage.reset_index(drop=True),
    }


def pair_units(
    panel: dict[str, pd.DataFrame],
    *,
    model_id: str,
    reference_model_id: str,
    aggregation_id: str,
) -> pd.DataFrame:
    """Pair one registered model/reference over whole common complete contexts."""

    _require_panel(panel)
    if aggregation_id not in AGGREGATIONS:
        raise PredictionAuditError("unknown_aggregation")
    model_id = str(model_id)
    reference_model_id = str(reference_model_id)
    if (model_id, reference_model_id) not in PAIR_SET:
        raise PredictionAuditError("unknown_pair")
    units = panel[aggregation_id]
    _require_unique_columns(units, "panel_duplicate_columns")
    _require_columns(units, UNIT_REQUIRED, "panel_columns_missing")
    complete = _complete_lookup(panel["coverage"])
    for row in units.itertuples(index=False):
        row_model = str(row.model_id)
        row_context = str(row.context_id)
        if row_model not in ALL_MODELS_SET:
            raise PredictionAuditError("unexpected_unit_model")
        key = (row_model, row_context)
        if key not in complete:
            raise PredictionAuditError("unexpected_unit_context")
        if not complete[key]:
            raise PredictionAuditError("unit_on_incomplete_context")
    common = sorted(
        _complete_contexts(complete, model_id) & _complete_contexts(complete, reference_model_id)
    )
    if not common:
        raise PredictionAuditError("no_common_complete_contexts")
    model_cells = _grouped_units(units, model_id)
    reference_cells = _grouped_units(units, reference_model_id)
    rows: list[dict] = []
    for context_id in common:
        model_cell = model_cells.get(context_id)
        reference_cell = reference_cells.get(context_id)
        if model_cell is None or reference_cell is None:
            raise PredictionAuditError("coverage_unit_disagreement")
        model_units = _index_units(model_cell)
        reference_units = _index_units(reference_cell)
        if set(model_units) != set(reference_units):
            raise PredictionAuditError("unit_set_mismatch")
        for unit_id in sorted(model_units):
            model_row = model_units[unit_id]
            reference_row = reference_units[unit_id]
            _require_matching_identity(model_row, reference_row)
            rows.append(
                {
                    "context_id": model_row["context_id"],
                    "domain": model_row["domain"],
                    "station": model_row["station"],
                    "instrument": model_row["instrument"],
                    "master_sample_id": model_row["master_sample_id"],
                    "unit_id": unit_id,
                    "true_label": model_row["true_label"],
                    "correct_model": model_row["correct"],
                    "correct_reference": reference_row["correct"],
                }
            )
    paired = pd.DataFrame(rows, columns=list(PAIR_COLUMNS))
    for column in (
        "context_id",
        "domain",
        "station",
        "instrument",
        "master_sample_id",
        "unit_id",
        "true_label",
    ):
        paired[column] = paired[column].astype(str)
    paired["correct_model"] = paired["correct_model"].astype(bool)
    paired["correct_reference"] = paired["correct_reference"].astype(bool)
    paired = paired.sort_values(["context_id", "unit_id"], kind="stable").reset_index(drop=True)
    return paired


def audit_point_estimates(
    panel: dict[str, pd.DataFrame],
    frozen_endpoint_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute balanced accuracy from unit correctness without any stochastic draws.

    Each prepared model/context/aggregation key is independently rebuilt by
    grouping correctness by original true class and averaging class means. The
    value is compared against the authenticated frozen endpoint table within
    ``AUDIT_ATOL``. Missing, extra, duplicated or nonfinite frozen rows are
    rejected rather than silently tolerated.
    """

    _require_panel(panel)
    frozen = frozen_endpoint_metrics
    if not isinstance(frozen, pd.DataFrame):
        raise PredictionAuditError("frozen_endpoint_not_frame")
    _require_unique_columns(frozen, "frozen_duplicate_columns")
    _require_columns(frozen, FROZEN_ENDPOINT_COLUMNS, "frozen_columns_missing")
    frozen_index: dict[tuple[str, str, str], float] = {}
    for row in frozen.itertuples(index=False):
        model = _strict_identifier(row.model_id, "frozen_model_invalid")
        context = _strict_identifier(row.context_id, "frozen_context_invalid")
        aggregation = _strict_identifier(row.aggregation_id, "frozen_aggregation_invalid")
        if model not in ALL_MODELS_SET:
            raise PredictionAuditError("frozen_unknown_model")
        if aggregation not in AGGREGATIONS:
            raise PredictionAuditError("frozen_unknown_aggregation")
        key = (model, context, aggregation)
        if key in frozen_index:
            raise PredictionAuditError("frozen_duplicate_key")
        frozen_index[key] = _finite(row.balanced_accuracy, "frozen_balanced_accuracy_invalid")
    rows: list[dict] = []
    observed_keys: set[tuple[str, str, str]] = set()
    for aggregation in AGGREGATIONS:
        units = panel[aggregation]
        _require_unique_columns(units, "panel_duplicate_columns")
        _require_columns(
            units, ("model_id", "context_id", "true_label", "correct"), "panel_columns_missing"
        )
        for (model, context), cell in units.groupby(["model_id", "context_id"], sort=True):
            key = (str(model), str(context), aggregation)
            if key in observed_keys:
                raise PredictionAuditError("panel_duplicate_key")
            observed_keys.add(key)
            per_class: dict[str, list[float]] = {}
            labels = cell.true_label.astype(str).tolist()
            correctness = [
                _coerce_correct(value, "panel_correctness_invalid") for value in cell.correct
            ]
            for label, value in zip(labels, correctness, strict=True):
                per_class.setdefault(label, []).append(1.0 if value else 0.0)
            recomputed = float(np.mean([np.mean(values) for values in per_class.values()]))
            if not np.isfinite(recomputed):
                raise PredictionAuditError("panel_balanced_accuracy_nonfinite")
            if key not in frozen_index:
                raise PredictionAuditError("frozen_missing_key")
            reference = frozen_index[key]
            error = abs(recomputed - reference)
            if error > AUDIT_ATOL:
                raise PredictionAuditError("frozen_point_mismatch")
            rows.append(
                {
                    "model_id": key[0],
                    "context_id": key[1],
                    "aggregation_id": aggregation,
                    "balanced_accuracy": recomputed,
                    "reference_balanced_accuracy": reference,
                    "absolute_error": error,
                }
            )
    if set(frozen_index) != observed_keys:
        if set(frozen_index) - observed_keys:
            raise PredictionAuditError("frozen_extra_key")
        raise PredictionAuditError("frozen_missing_key")
    audit = pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))
    audit = audit.sort_values(
        ["model_id", "context_id", "aggregation_id"], kind="stable"
    ).reset_index(drop=True)
    return audit
