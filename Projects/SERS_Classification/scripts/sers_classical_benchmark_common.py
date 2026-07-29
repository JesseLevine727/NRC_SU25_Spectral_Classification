#!/usr/bin/env python3
"""Shared, side-effect-free machinery for the NATO SERS classical benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sers_baseline_common as baseline


PROTOCOL_VERSION = "sers-classical-benchmark-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**31 - 1
    )


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected classical benchmark protocol")
    if protocol.get("status_before_execution") != "predeclared":
        raise ValueError("Classical benchmark protocol was not predeclared")
    return protocol


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS regression against one-hot targets with an argmax classifier."""

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 1000,
        tol: float = 1.0e-6,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSDA":
        self.classes_ = np.unique(np.asarray(y, dtype=str))
        class_index = {label: index for index, label in enumerate(self.classes_)}
        targets = np.zeros((len(y), len(self.classes_)), dtype=float)
        targets[
            np.arange(len(y)),
            [class_index[str(label)] for label in y],
        ] = 1.0
        components = min(
            int(self.n_components),
            x.shape[1],
            max(1, x.shape[0] - 1),
        )
        self.model_ = PLSRegression(
            n_components=components,
            scale=False,
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        )
        self.model_.fit(x, targets)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.model_.predict(x), dtype=float)
        if scores.ndim == 1:
            scores = scores[:, None]
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.decision_function(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


@dataclass(frozen=True)
class Candidate:
    order: int
    candidate_id: str
    representation: str
    model_family: str
    parameters: dict[str, Any]

    def row(self) -> dict[str, Any]:
        return {
            "candidate_order": self.order,
            "candidate_id": self.candidate_id,
            "representation": self.representation,
            "model_family": self.model_family,
            "parameters_json": json.dumps(
                json_clean(self.parameters), sort_keys=True
            ),
        }


def candidate_grid(protocol: dict[str, Any]) -> list[Candidate]:
    grid = protocol["candidate_grid"]
    candidates: list[Candidate] = []

    def add(
        representation: str, family: str, parameters: dict[str, Any]
    ) -> None:
        order = len(candidates)
        payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
        candidates.append(
            Candidate(
                order=order,
                candidate_id=f"{representation}__{family}__{suffix}",
                representation=representation,
                model_family=family,
                parameters=parameters,
            )
        )

    for representation in protocol["immutable_input"]["representations"]:
        spec = grid["pca_logistic"]
        for components in spec["pca_components"]:
            for c_value in spec["C"]:
                add(
                    representation,
                    "pca_logistic",
                    {
                        "pca_components": components,
                        "pca_whiten": spec["pca_whiten"],
                        "C": c_value,
                    },
                )
        spec = grid["elastic_net_logistic"]
        for c_value in spec["C"]:
            for ratio in spec["l1_ratio"]:
                add(
                    representation,
                    "elastic_net_logistic",
                    {"C": c_value, "l1_ratio": ratio},
                )
        spec = grid["shrinkage_lda"]
        for shrinkage in spec["shrinkage"]:
            add(
                representation,
                "shrinkage_lda",
                {"shrinkage": shrinkage},
            )
        spec = grid["pls_da"]
        for components in spec["components"]:
            add(
                representation,
                "pls_da",
                {"components": components},
            )
        spec = grid["linear_svm"]
        for c_value in spec["C"]:
            add(representation, "linear_svm", {"C": c_value})
        spec = grid["rbf_svm"]
        for c_value in spec["C"]:
            for gamma in spec["gamma"]:
                add(
                    representation,
                    "rbf_svm",
                    {"C": c_value, "gamma": gamma},
                )
    return candidates


def build_estimator(
    candidate: Candidate,
    protocol: dict[str, Any],
    seed: int,
) -> BaseEstimator:
    family = candidate.model_family
    parameters = candidate.parameters
    spec = protocol["candidate_grid"][family]
    if family == "pca_logistic":
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
                        class_weight=spec["class_weight"],
                        max_iter=int(spec["maximum_iterations"]),
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "elastic_net_logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        l1_ratio=float(parameters["l1_ratio"]),
                        class_weight=spec["class_weight"],
                        max_iter=int(spec["maximum_iterations"]),
                        solver=str(spec["solver"]),
                        random_state=seed,
                        tol=1.0e-4,
                    ),
                ),
            ]
        )
    if family == "shrinkage_lda":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver=str(spec["solver"]),
                        shrinkage=parameters["shrinkage"],
                    ),
                ),
            ]
        )
    if family == "pls_da":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    PLSDA(
                        n_components=int(parameters["components"]),
                        max_iter=int(spec["maximum_iterations"]),
                        tol=float(spec["tolerance"]),
                    ),
                ),
            ]
        )
    if family in {"linear_svm", "rbf_svm"}:
        kwargs: dict[str, Any] = {
            "kernel": "linear" if family == "linear_svm" else "rbf",
            "C": float(parameters["C"]),
            "class_weight": spec["class_weight"],
            "decision_function_shape": "ovr",
            "probability": False,
            "random_state": seed,
            "cache_size": 1024,
        }
        if family == "rbf_svm":
            kwargs["gamma"] = parameters["gamma"]
        return Pipeline(
            [("scale", StandardScaler()), ("classifier", SVC(**kwargs))]
        )
    raise ValueError(f"Unknown model family: {family}")


def fit_estimator(
    candidate: Candidate,
    protocol: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> BaseEstimator:
    estimator = build_estimator(candidate, protocol, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(x, y)
    return estimator


def estimator_classes(estimator: BaseEstimator) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(getattr(estimator, "classes_"), dtype=str)
    if isinstance(estimator, Pipeline):
        return np.asarray(estimator.steps[-1][1].classes_, dtype=str)
    raise ValueError("Estimator has no classes_")


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -700.0, 700.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def estimator_scores(estimator: BaseEstimator, x: np.ndarray) -> np.ndarray:
    classes = estimator_classes(estimator)
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
        return np.log(np.clip(probabilities, 1.0e-12, 1.0))
    scores = np.asarray(estimator.decision_function(x), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    if scores.shape[1] != len(classes):
        raise ValueError("Score columns do not match estimator classes")
    return scores


def align_scores(
    scores: np.ndarray,
    source_classes: Iterable[str],
    target_classes: Iterable[str],
) -> np.ndarray:
    source = [str(value) for value in source_classes]
    target = [str(value) for value in target_classes]
    if source == target:
        return np.asarray(scores, dtype=float)
    output = np.full((len(scores), len(target)), -1.0e12, dtype=float)
    target_index = {label: index for index, label in enumerate(target)}
    for index, label in enumerate(source):
        if label in target_index:
            output[:, target_index[label]] = scores[:, index]
    return output


def probabilities_from_scores(
    scores: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    return softmax(np.asarray(scores, dtype=float) / max(float(temperature), 1e-8))


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    class_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_index.get(str(label), -1) for label in labels])
    valid = indices >= 0
    if not np.any(valid):
        return 1.0

    def objective(log_temperature: float) -> float:
        probabilities = probabilities_from_scores(
            scores[valid], math.exp(log_temperature)
        )
        return float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[np.arange(valid.sum()), indices[valid]],
                        1.0e-12,
                        1.0,
                    )
                )
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(bounds[0]), math.log(bounds[1])),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    return float(math.exp(result.x)) if result.success else 1.0


def expected_calibration_error(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = np.asarray(labels, dtype=str) == np.asarray(predictions, dtype=str)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (
                confidence <= edges[index + 1]
            )
        else:
            mask = (confidence >= edges[index]) & (
                confidence < edges[index + 1]
            )
        if np.any(mask):
            value += (
                float(mask.sum())
                / total
                * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            )
    return float(value)


def classification_metrics(
    train_classes: Iterable[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
) -> dict[str, Any]:
    train_class_set = set(str(value) for value in train_classes)
    labels = np.asarray(labels, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    supported = np.asarray([label in train_class_set for label in labels])
    supported_labels = labels[supported]
    supported_predictions = predictions[supported]
    result: dict[str, Any] = {
        "n_test": int(len(labels)),
        "n_supported": int(supported.sum()),
        "n_unsupported": int((~supported).sum()),
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "accuracy": np.nan,
        "negative_log_likelihood": np.nan,
        "brier_multiclass": np.nan,
        "expected_calibration_error_10": np.nan,
        "supported_mask": supported,
    }
    if not np.any(supported):
        return result
    result["balanced_accuracy"] = float(
        balanced_accuracy_score(supported_labels, supported_predictions)
    )
    result["macro_f1"] = float(
        f1_score(supported_labels, supported_predictions, average="macro")
    )
    result["accuracy"] = float(
        accuracy_score(supported_labels, supported_predictions)
    )
    class_index = {
        str(label): index for index, label in enumerate(probability_classes)
    }
    target_index = np.asarray(
        [class_index[str(label)] for label in supported_labels], dtype=int
    )
    supported_probabilities = probabilities[supported]
    result["negative_log_likelihood"] = float(
        -np.mean(
            np.log(
                np.clip(
                    supported_probabilities[
                        np.arange(len(target_index)), target_index
                    ],
                    1.0e-12,
                    1.0,
                )
            )
        )
    )
    one_hot = np.zeros_like(supported_probabilities)
    one_hot[np.arange(len(target_index)), target_index] = 1.0
    result["brier_multiclass"] = float(
        np.mean(np.sum((supported_probabilities - one_hot) ** 2, axis=1))
    )
    result["expected_calibration_error_10"] = expected_calibration_error(
        supported_labels,
        supported_predictions,
        supported_probabilities,
        bins=10,
    )
    return result


def per_class_rows(
    context: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    supported_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    for label in sorted(np.unique(labels[supported_mask])):
        mask = supported_mask & (labels == label)
        rows.append(
            {
                **context,
                "target_analyte": label,
                "support": int(mask.sum()),
                "correct": int(np.sum(predictions[mask] == label)),
                "recall": float(np.mean(predictions[mask] == label)),
            }
        )
    return rows


def selective_rows(
    context: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    supported_mask: np.ndarray,
    coverages: Iterable[float],
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=str)[supported_mask]
    predictions = np.asarray(predictions, dtype=str)[supported_mask]
    confidence = np.asarray(confidence, dtype=float)[supported_mask]
    order = np.argsort(-confidence, kind="stable")
    rows: list[dict[str, Any]] = []
    if len(order) == 0:
        for requested in coverages:
            rows.append(
                {
                    **context,
                    "requested_coverage": float(requested),
                    "n_retained": 0,
                    "realized_coverage": 0.0,
                    "accuracy": np.nan,
                    "selective_risk": np.nan,
                    "confidence_threshold": np.nan,
                }
            )
        return rows
    for requested in coverages:
        retained = max(1, int(math.ceil(float(requested) * len(order))))
        keep = order[:retained]
        rows.append(
            {
                **context,
                "requested_coverage": float(requested),
                "n_retained": int(retained),
                "realized_coverage": float(retained / max(len(order), 1)),
                "accuracy": float(np.mean(labels[keep] == predictions[keep])),
                "selective_risk": float(
                    1.0 - np.mean(labels[keep] == predictions[keep])
                ),
                "confidence_threshold": float(np.min(confidence[keep])),
            }
        )
    return rows


def ood_metrics(
    in_distribution_score: np.ndarray,
    out_distribution_score: np.ndarray,
) -> dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(len(in_distribution_score), dtype=int),
            np.ones(len(out_distribution_score), dtype=int),
        ]
    )
    scores = np.concatenate(
        [in_distribution_score, out_distribution_score]
    )
    return {
        "ood_auroc": float(roc_auc_score(labels, scores)),
        "ood_auprc": float(average_precision_score(labels, scores)),
    }


def candidate_lookup(
    candidates: Iterable[Candidate],
) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def subset_mask(manifest: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "strict_core":
        return np.ones(len(manifest), dtype=bool)
    if subset == "quality_pass":
        return manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    if subset == "field_quality_stress":
        return manifest["field_quality_stress"].astype(bool).to_numpy()
    raise ValueError(f"Unknown subset: {subset}")
