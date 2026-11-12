#!/usr/bin/env python3
"""Shared estimators and metrics for the NATO SERS open-world study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


PROTOCOL_VERSION = "sers-open-world-shift-v1"
BASELINE_REGISTRY_VERSION = "sers-open-world-baselines-v1"

sha256_file = classical.sha256_file
stable_seed = classical.stable_seed
json_clean = classical.json_clean
write_json = classical.write_json
align_scores = classical.align_scores
fit_temperature = classical.fit_temperature
probabilities_from_scores = classical.probabilities_from_scores
classification_metrics = classical.classification_metrics
expected_calibration_error = classical.expected_calibration_error


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected open-world protocol")
    if value.get("status_before_execution") != (
        "predeclared_partitions_before_open_set_outcomes"
    ):
        raise ValueError("Open-world protocol was not predeclared")
    return value


def load_baseline_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("registry_version") != BASELINE_REGISTRY_VERSION:
        raise ValueError("Unexpected baseline registry")
    if value.get("status_before_execution") != (
        "predeclared_before_true_unknown_outcomes"
    ):
        raise ValueError("Baseline registry was not predeclared")
    return value


@dataclass(frozen=True)
class Candidate:
    order: int
    candidate_id: str
    model_family: str
    representation: str
    parameters: dict[str, Any]

    def row(self) -> dict[str, Any]:
        return {
            "candidate_order": self.order,
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "representation": self.representation,
            "parameters_json": json.dumps(
                json_clean(self.parameters), sort_keys=True
            ),
        }


def candidates_from_registry(registry: dict[str, Any]) -> list[Candidate]:
    return [
        Candidate(
            order=index,
            candidate_id=str(record["candidate_id"]),
            model_family=str(record["model_family"]),
            representation=str(record["representation"]),
            parameters=dict(record["parameters"]),
        )
        for index, record in enumerate(registry["candidate_registry"])
    ]


def build_estimator(candidate: Candidate, seed: int) -> BaseEstimator:
    parameters = candidate.parameters
    if candidate.model_family == "pca_logistic":
        return Pipeline(
            [
                (
                    "pca",
                    PCA(
                        n_components=int(parameters["pca_components"]),
                        whiten=bool(parameters["pca_whiten"]),
                        random_state=seed,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        class_weight=str(parameters["class_weight"]),
                        max_iter=int(parameters["maximum_iterations"]),
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate.model_family == "rbf_svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=float(parameters["C"]),
                        gamma=parameters["gamma"],
                        kernel="rbf",
                        class_weight=str(parameters["class_weight"]),
                        probability=False,
                        decision_function_shape="ovr",
                        random_state=seed,
                        cache_size=1024,
                    ),
                ),
            ]
        )
    if candidate.model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_features=parameters["max_features"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            criterion=str(parameters["criterion"]),
            bootstrap=bool(parameters["bootstrap"]),
            class_weight=str(parameters["class_weight"]),
            random_state=seed,
            n_jobs=int(parameters["estimator_jobs"]),
        )
    raise ValueError(f"Unsupported baseline family: {candidate.model_family}")


def estimator_classes(estimator: BaseEstimator) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_, dtype=str)
    if isinstance(estimator, Pipeline):
        return np.asarray(estimator.steps[-1][1].classes_, dtype=str)
    raise ValueError("Estimator does not expose classes")


def estimator_scores(estimator: BaseEstimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
        return np.log(np.clip(probabilities, 1.0e-12, 1.0))
    scores = np.asarray(estimator.decision_function(x), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def uniform_master_weights(groups: np.ndarray) -> np.ndarray:
    values, counts = np.unique(np.asarray(groups, dtype=str), return_counts=True)
    count_by_group = dict(zip(values, counts))
    weights = np.asarray(
        [1.0 / count_by_group[str(group)] for group in groups], dtype=float
    )
    return weights / weights.mean()


@dataclass
class FittedBaseline:
    candidate: Candidate
    estimator: BaseEstimator
    score_scaler: StandardScaler
    score_pca: PCA
    class_centers: dict[str, np.ndarray]
    precision: np.ndarray

    @property
    def classes(self) -> np.ndarray:
        return estimator_classes(self.estimator)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return estimator_scores(self.estimator, x)

    def mahalanobis(self, x: np.ndarray) -> np.ndarray:
        embedding = self.score_pca.transform(self.score_scaler.transform(x))
        distances: list[np.ndarray] = []
        for label in self.classes:
            delta = embedding - self.class_centers[str(label)]
            distances.append(
                np.einsum("ij,jk,ik->i", delta, self.precision, delta)
            )
        return np.maximum(np.min(np.column_stack(distances), axis=1), 0.0)


def fit_baseline(
    candidate: Candidate,
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    registry: dict[str, Any],
) -> FittedBaseline:
    estimator = build_estimator(candidate, seed)
    fit_kwargs: dict[str, Any] = {}
    if candidate.model_family == "random_forest":
        weighting = str(candidate.parameters.get("master_weighting", "uniform_rows"))
        if weighting == "inverse_master_frequency":
            fit_kwargs["sample_weight"] = uniform_master_weights(groups)
    estimator.fit(x, labels, **fit_kwargs)

    spec = registry["mahalanobis"]
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    components = min(
        int(spec["pca_components"]), scaled.shape[1], max(1, len(x) - 1)
    )
    pca = PCA(
        n_components=components,
        whiten=bool(spec["pca_whiten"]),
        random_state=seed,
    ).fit(scaled)
    embedding = pca.transform(scaled)
    centers = {
        str(label): embedding[labels == label].mean(axis=0)
        for label in np.unique(labels)
    }
    residuals = np.vstack(
        [
            embedding[labels == label] - centers[str(label)]
            for label in np.unique(labels)
        ]
    )
    precision = LedoitWolf(assume_centered=True).fit(residuals).precision_
    return FittedBaseline(
        candidate=candidate,
        estimator=estimator,
        score_scaler=scaler,
        score_pca=pca,
        class_centers=centers,
        precision=precision,
    )


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1.0e-12, 1.0)
    return -np.sum(values * np.log(values), axis=1)


def anomaly_scores(
    fitted: FittedBaseline,
    x: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    logits = fitted.logits(x)
    probabilities = probabilities_from_scores(logits, temperature)
    return logits, probabilities, {
        "one_minus_calibrated_max_probability": 1.0
        - probabilities.max(axis=1),
        "predictive_entropy": predictive_entropy(probabilities),
        "minimum_class_mahalanobis_distance": fitted.mahalanobis(x),
    }


def fpr_at_95_tpr(
    unknown: np.ndarray, scores: np.ndarray
) -> float:
    if len(np.unique(unknown)) != 2:
        return np.nan
    fpr, tpr, _ = roc_curve(unknown.astype(int), scores)
    eligible = fpr[tpr >= 0.95]
    return float(np.min(eligible)) if len(eligible) else 1.0


def oscr_auc(
    known_correct: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    known_correct = np.asarray(known_correct, dtype=bool)
    known_scores = np.asarray(known_scores, dtype=float)
    unknown_scores = np.asarray(unknown_scores, dtype=float)
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return np.nan
    values = np.unique(np.concatenate([known_scores, unknown_scores]))
    thresholds = np.concatenate(([-np.inf], values, [np.inf]))
    ccr = np.asarray(
        [
            np.mean(known_correct & (known_scores <= threshold))
            for threshold in thresholds
        ],
        dtype=float,
    )
    fpr = np.asarray(
        [
            np.mean(unknown_scores <= threshold)
            for threshold in thresholds
        ],
        dtype=float,
    )
    # Threshold order traces the OSCR curve. Repeated FPR values are vertical
    # segments and must retain zero horizontal area.
    return float(np.trapezoid(ccr, fpr))


def master_level_frame(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
    scores: np.ndarray,
    unknown: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    frame = manifest.reset_index(drop=True)
    frame = frame.assign(_unknown_status=np.asarray(unknown, dtype=bool))
    for (master, unknown_status), indices in frame.groupby(
        ["master_sample_id", "_unknown_status"], sort=True
    ).groups.items():
        index = np.asarray(list(indices), dtype=int)
        unknown_values = np.asarray(unknown, dtype=bool)[index]
        if not np.all(unknown_values == bool(unknown_status)):
            raise ValueError("Master-condition aggregation is inconsistent")
        true_labels = np.asarray(labels, dtype=str)[index]
        if len(np.unique(true_labels)) != 1:
            raise ValueError("A master maps to multiple analytes")
        mean_probability = np.asarray(probabilities)[index].mean(axis=0)
        predicted = str(probability_classes[np.argmax(mean_probability)])
        records.append(
            {
                "master_sample_id": str(master),
                "target_analyte": str(true_labels[0]),
                "predicted_analyte": predicted,
                "correct": predicted == str(true_labels[0]),
                "unknown": bool(unknown_status),
                "anomaly_score": float(np.asarray(scores)[index].mean()),
                "rows": len(index),
            }
        )
    return pd.DataFrame(records)


def open_set_metrics_from_master(master: pd.DataFrame) -> dict[str, Any]:
    unknown = master["unknown"].astype(bool).to_numpy()
    scores = master["anomaly_score"].to_numpy(float)
    known = ~unknown
    result: dict[str, Any] = {
        "n_known_masters": int(known.sum()),
        "n_unknown_masters": int(unknown.sum()),
        "master_unknown_auroc": np.nan,
        "master_unknown_auprc": np.nan,
        "master_fpr_at_95_tpr": np.nan,
        "master_oscr_auc": np.nan,
        "master_known_balanced_accuracy": np.nan,
        "master_known_macro_f1": np.nan,
        "master_known_accuracy": np.nan,
    }
    if known.any():
        result["master_known_balanced_accuracy"] = float(
            balanced_accuracy_score(
                master.loc[known, "target_analyte"],
                master.loc[known, "predicted_analyte"],
            )
        )
        result["master_known_macro_f1"] = float(
            f1_score(
                master.loc[known, "target_analyte"],
                master.loc[known, "predicted_analyte"],
                average="macro",
            )
        )
        result["master_known_accuracy"] = float(
            accuracy_score(
                master.loc[known, "target_analyte"],
                master.loc[known, "predicted_analyte"],
            )
        )
    if known.any() and unknown.any():
        result["master_unknown_auroc"] = float(roc_auc_score(unknown, scores))
        result["master_unknown_auprc"] = float(
            average_precision_score(unknown, scores)
        )
        result["master_fpr_at_95_tpr"] = fpr_at_95_tpr(unknown, scores)
        result["master_oscr_auc"] = oscr_auc(
            master.loc[known, "correct"].to_numpy(bool),
            master.loc[known, "anomaly_score"].to_numpy(float),
            master.loc[unknown, "anomaly_score"].to_numpy(float),
        )
    return result


def open_set_metrics(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
    scores: np.ndarray,
    unknown: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    unknown = np.asarray(unknown, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    known = ~unknown
    result: dict[str, Any] = {
        "n_known_rows": int(known.sum()),
        "n_unknown_rows": int(unknown.sum()),
        "row_unknown_auroc": np.nan,
        "row_unknown_auprc": np.nan,
        "row_fpr_at_95_tpr": np.nan,
        "row_oscr_auc": np.nan,
    }
    if known.any() and unknown.any():
        result.update(
            {
                "row_unknown_auroc": float(roc_auc_score(unknown, scores)),
                "row_unknown_auprc": float(
                    average_precision_score(unknown, scores)
                ),
                "row_fpr_at_95_tpr": fpr_at_95_tpr(unknown, scores),
                "row_oscr_auc": oscr_auc(
                    np.asarray(predictions, dtype=str)[known]
                    == np.asarray(labels, dtype=str)[known],
                    scores[known],
                    scores[unknown],
                ),
            }
        )
    master = master_level_frame(
        manifest,
        labels,
        predictions,
        probabilities,
        probability_classes,
        scores,
        unknown,
    )
    result.update(open_set_metrics_from_master(master))
    return result, master


def threshold_from_known(
    known_scores: np.ndarray, retained_coverage: float
) -> float:
    if not 0.0 < retained_coverage <= 1.0:
        raise ValueError("Coverage must be in (0, 1]")
    return float(
        np.quantile(
            np.asarray(known_scores, dtype=float),
            retained_coverage,
            method="higher",
        )
    )


def threshold_rows(
    context: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    unknown: np.ndarray,
    thresholds: dict[float, float],
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    scores = np.asarray(scores, dtype=float)
    unknown = np.asarray(unknown, dtype=bool)
    rows: list[dict[str, Any]] = []
    for coverage, threshold in thresholds.items():
        accepted = scores <= threshold
        known = ~unknown
        known_accepted = accepted & known
        rows.append(
            {
                **context,
                "requested_known_coverage": float(coverage),
                "threshold": float(threshold),
                "n_rows": len(labels),
                "n_accepted": int(accepted.sum()),
                "acceptance_rate": float(accepted.mean()),
                "known_acceptance_rate": (
                    float(accepted[known].mean()) if known.any() else np.nan
                ),
                "unknown_acceptance_rate": (
                    float(accepted[unknown].mean()) if unknown.any() else np.nan
                ),
                "known_accuracy_accepted": (
                    float(
                        np.mean(
                            labels[known_accepted]
                            == predictions[known_accepted]
                        )
                    )
                    if known_accepted.any()
                    else np.nan
                ),
                "known_selective_risk": (
                    float(
                        1.0
                        - np.mean(
                            labels[known_accepted]
                            == predictions[known_accepted]
                        )
                    )
                    if known_accepted.any()
                    else np.nan
                ),
            }
        )
    return rows


def aligned_probabilities(
    logits: np.ndarray,
    source_classes: Iterable[str],
    target_classes: Iterable[str],
    temperature: float,
) -> np.ndarray:
    aligned = align_scores(logits, source_classes, target_classes)
    return probabilities_from_scores(aligned, temperature)
