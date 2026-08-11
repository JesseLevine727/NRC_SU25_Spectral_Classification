"""Selection, aggregation, calibration, and metrics for P03 classical runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from atlas_sers.governance.canonical import sha256_value

PROBABILITY_CLIP = (1e-7, 1 - 1e-7)


def softmax(scores: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Multiclass score matrix must be two-dimensional.")
    if not np.isfinite(values).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Scores and positive temperature must be finite.")
    shifted = values / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    class_vocabulary: tuple[str, ...]
    observations: int
    masters: int
    fit_observation_uid_sha256: str
    fit_master_uid_sha256: str
    optimizer_success: bool
    optimizer_objective: float

    @property
    def state_sha256(self) -> str:
        return sha256_value(asdict(self))


def fit_temperature(
    scores: np.ndarray,
    true_labels: np.ndarray,
    *,
    class_vocabulary: list[str] | tuple[str, ...] | np.ndarray,
    observation_uids: np.ndarray,
    master_ids: np.ndarray,
) -> TemperatureCalibration:
    """Fit one scalar temperature to authorized cross-fitted development scores."""

    matrix = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(true_labels).astype(str)
    classes = tuple(str(value) for value in class_vocabulary)
    observations = np.asarray(observation_uids).astype(str)
    masters = np.asarray(master_ids).astype(str)
    if not (len(matrix) == len(labels) == len(observations) == len(masters)):
        raise ValueError("Calibration score and provenance lengths differ.")
    if matrix.shape[1] != len(classes):
        raise ValueError("Calibration class vocabulary does not match score columns.")
    class_index = {label: index for index, label in enumerate(classes)}
    if any(label not in class_index for label in labels):
        raise ValueError("Calibration contains a label outside the class vocabulary.")
    indices = np.asarray([class_index[label] for label in labels], dtype=int)

    def objective(log_temperature: float) -> float:
        probabilities = softmax(matrix, temperature=float(np.exp(log_temperature)))
        true_probability = np.clip(
            probabilities[np.arange(len(indices)), indices], *PROBABILITY_CLIP
        )
        return float(-np.log(true_probability).mean())

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    temperature = float(np.exp(result.x))
    return TemperatureCalibration(
        temperature=temperature,
        class_vocabulary=classes,
        observations=len(observations),
        masters=len(set(masters)),
        fit_observation_uid_sha256=sha256_value(sorted(observations)),
        fit_master_uid_sha256=sha256_value(sorted(set(masters))),
        optimizer_success=bool(result.success),
        optimizer_objective=float(result.fun),
    )


def apply_temperature(scores: np.ndarray, calibration: TemperatureCalibration) -> np.ndarray:
    return softmax(scores, temperature=calibration.temperature)


def instrument_balanced_master_probabilities(
    *,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    master_ids: np.ndarray,
    instruments: np.ndarray,
    class_vocabulary: list[str] | tuple[str, ...] | np.ndarray,
) -> pd.DataFrame:
    """Average rows within instrument view, then weight instrument views equally."""

    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(true_labels).astype(str)
    masters = np.asarray(master_ids).astype(str)
    units = np.asarray(instruments).astype(str)
    classes = [str(value) for value in class_vocabulary]
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("Probability matrix and class vocabulary disagree.")
    if not (len(values) == len(labels) == len(masters) == len(units)):
        raise ValueError("Prediction and aggregation metadata lengths differ.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    normalized = values / values.sum(axis=1, keepdims=True)
    probability_columns = [f"probability_{index}" for index in range(len(classes))]
    frame = pd.DataFrame(normalized, columns=probability_columns)
    frame["true_label"] = labels
    frame["master_sample_id"] = masters
    frame["instrument"] = units
    label_count = frame.groupby("master_sample_id").true_label.nunique()
    if not label_count.eq(1).all():
        raise ValueError("A physical master has multiple true labels.")
    view = (
        frame.groupby(["master_sample_id", "instrument", "true_label"], as_index=False)[
            probability_columns
        ]
        .mean()
        .sort_values(["master_sample_id", "instrument"], kind="stable")
    )
    master = (
        view.groupby(["master_sample_id", "true_label"], as_index=False)[probability_columns]
        .mean()
        .sort_values("master_sample_id", kind="stable")
        .reset_index(drop=True)
    )
    probability_matrix = master[probability_columns].to_numpy()
    master["predicted_label"] = np.asarray(classes)[np.argmax(probability_matrix, axis=1)]
    master["probabilities"] = [row.tolist() for row in probability_matrix]
    master["instrument_views"] = master.master_sample_id.map(
        view.groupby("master_sample_id").instrument.nunique()
    )
    return master.drop(columns=probability_columns)


def expected_calibration_error(
    probabilities: np.ndarray, true_indices: np.ndarray, *, bins: int = 10
) -> float:
    """Equal-mass-bin ECE with deterministic confidence ordering."""

    values = np.asarray(probabilities, dtype=np.float64)
    indices = np.asarray(true_indices, dtype=int)
    if len(values) == 0:
        raise ValueError("ECE requires at least one prediction.")
    predicted = values.argmax(axis=1)
    confidence = values.max(axis=1)
    order = np.argsort(confidence, kind="stable")
    total = len(values)
    ece = 0.0
    for members in np.array_split(order, min(bins, total)):
        accuracy = np.mean(predicted[members] == indices[members])
        mean_confidence = np.mean(confidence[members])
        ece += len(members) / total * abs(float(accuracy - mean_confidence))
    return float(ece)


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    class_vocabulary: list[str] | tuple[str, ...] | np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(true_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    classes = np.asarray(class_vocabulary).astype(str)
    if len(labels) != len(predicted):
        raise ValueError("True and predicted label lengths differ.")
    recalls = recall_score(labels, predicted, labels=classes, average=None, zero_division=0)
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "per_class_recall": {
            label: float(value) for label, value in zip(classes, recalls, strict=True)
        },
        "confusion_matrix": confusion_matrix(labels, predicted, labels=classes).tolist(),
        "support": {label: int(np.sum(labels == label)) for label in classes},
    }
    if probabilities is None:
        result.update({"negative_log_likelihood": None, "brier_score": None, "ece": None})
        return result
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(labels), len(classes)):
        raise ValueError("Probability matrix has the wrong shape.")
    class_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_index[label] for label in labels], dtype=int)
    values = values / values.sum(axis=1, keepdims=True)
    clipped = np.clip(values[np.arange(len(values)), indices], *PROBABILITY_CLIP)
    one_hot = np.eye(len(classes))[indices]
    result.update(
        {
            "negative_log_likelihood": float(-np.log(clipped).mean()),
            "brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
            "ece": expected_calibration_error(values, indices),
        }
    )
    return result


def select_lexicographic_candidate(
    unit_metrics: pd.DataFrame, candidate_registry: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the frozen pseudo-domain/master-CV objective exactly."""

    required = {"candidate_id", "selection_unit_id", "balanced_accuracy", "macro_f1"}
    if not required <= set(unit_metrics):
        raise ValueError(f"Selection metrics miss fields: {sorted(required - set(unit_metrics))}")
    if unit_metrics.empty:
        raise ValueError("Selection metrics are empty.")
    records = unit_metrics.copy()
    if "seed" not in records:
        records["seed"] = "deterministic"
    if "status" not in records:
        records["status"] = "complete"
    if records.duplicated(["candidate_id", "selection_unit_id", "seed"]).any():
        raise ValueError("Selection metrics contain a duplicate candidate/unit/seed record.")
    if "seed_count" not in candidate_registry:
        candidate_registry = candidate_registry.assign(seed_count=1)
    registry_fields = [
        "candidate_id",
        "model_id",
        "complexity_rank",
        "declared_candidate_order",
        "seed_count",
    ]
    missing_registry = set(registry_fields) - set(candidate_registry)
    if missing_registry:
        raise ValueError(f"Candidate registry misses fields: {sorted(missing_registry)}")
    observed_candidates = set(records.candidate_id.astype(str))
    declared = candidate_registry[
        candidate_registry.candidate_id.astype(str).isin(observed_candidates)
    ][registry_fields].copy()
    if len(declared) != len(observed_candidates):
        raise ValueError("Selection metrics reference an undeclared candidate.")
    expected_units = unit_metrics.selection_unit_id.nunique()
    successful = records[
        records.status.eq("complete")
        & records.balanced_accuracy.notna()
        & records.macro_f1.notna()
    ].copy()
    if not successful.empty and not np.isfinite(
        successful[["balanced_accuracy", "macro_f1"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("A completed selection metric is nonfinite.")
    per_unit = (
        successful.groupby(["candidate_id", "selection_unit_id"], as_index=False)
        .agg(
            successful_seeds=("seed", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
    )
    summarized = (
        per_unit.groupby("candidate_id", as_index=False)
        .agg(
            selection_units=("selection_unit_id", "nunique"),
            minimum_successful_seeds=("successful_seeds", "min"),
            maximum_successful_seeds=("successful_seeds", "max"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
            worst_balanced_accuracy=("mean_balanced_accuracy", "min"),
            mean_macro_f1=("mean_macro_f1", "mean"),
        )
    )
    accounting = (
        records.groupby("candidate_id", as_index=False)
        .agg(
            terminal_rows=("status", "size"),
            completed_rows=("status", lambda values: int(values.eq("complete").sum())),
            failure_rows=("status", lambda values: int((~values.eq("complete")).sum())),
            terminal_statuses=(
                "status",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
        )
    )
    aggregation = (
        declared.merge(accounting, on="candidate_id", how="left", validate="one_to_one")
        .merge(summarized, on="candidate_id", how="left", validate="one_to_one")
        .fillna(
            {
                "terminal_rows": 0,
                "completed_rows": 0,
                "failure_rows": 0,
                "terminal_statuses": "absent",
                "selection_units": 0,
                "minimum_successful_seeds": 0,
                "maximum_successful_seeds": 0,
            }
        )
    )
    aggregation["expected_rows"] = aggregation.seed_count.astype(int) * expected_units
    aggregation["complete_support"] = (
        aggregation.terminal_rows.eq(aggregation.expected_rows)
        & aggregation.completed_rows.eq(aggregation.expected_rows)
        & aggregation.failure_rows.eq(0)
        & aggregation.selection_units.eq(expected_units)
        & aggregation.minimum_successful_seeds.eq(aggregation.seed_count)
        & aggregation.maximum_successful_seeds.eq(aggregation.seed_count)
    )
    supported = aggregation[aggregation.complete_support].copy()
    if supported.empty:
        raise ValueError("No candidate has complete selection-unit support.")
    supported = supported.sort_values(
        [
            "mean_balanced_accuracy",
            "worst_balanced_accuracy",
            "mean_macro_f1",
            "complexity_rank",
            "declared_candidate_order",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    winner = supported.iloc[0]
    aggregation["selected"] = aggregation.candidate_id.eq(winner.candidate_id)
    return winner, aggregation.sort_values("declared_candidate_order", kind="stable")
