#!/usr/bin/env python3
"""Side-effect-free helpers for the NATO SERS random-forest addendum."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


PROTOCOL_VERSION = "sers-random-forest-addendum-v1"

# Reuse the already validated metric/calibration implementations.
sha256_file = classical.sha256_file
stable_seed = classical.stable_seed
json_clean = classical.json_clean
write_json = classical.write_json
classification_metrics = classical.classification_metrics
per_class_rows = classical.per_class_rows
selective_rows = classical.selective_rows
ood_metrics = classical.ood_metrics
fit_temperature = classical.fit_temperature
probabilities_from_scores = classical.probabilities_from_scores
align_scores = classical.align_scores
subset_mask = classical.subset_mask


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected random-forest protocol")
    if protocol.get("status_before_execution") != (
        "predeclared_before_random_forest_execution"
    ):
        raise ValueError("Random-forest protocol was not predeclared")
    return protocol


@dataclass(frozen=True)
class Candidate:
    order: int
    candidate_id: str
    representation: str
    max_features: str | float
    max_depth: int | None
    min_samples_leaf: int
    master_weighting: str

    def parameters(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "master_weighting": self.master_weighting,
        }

    def row(self) -> dict[str, Any]:
        return {
            "candidate_order": self.order,
            "candidate_id": self.candidate_id,
            "representation": self.representation,
            "model_family": "random_forest",
            "parameters_json": json.dumps(
                json_clean(self.parameters()), sort_keys=True
            ),
        }


def candidate_grid(protocol: dict[str, Any]) -> list[Candidate]:
    spec = protocol["random_forest"]
    candidates: list[Candidate] = []
    for representation in protocol["immutable_input"]["representations"]:
        for max_features in spec["max_features"]:
            for max_depth in spec["max_depth"]:
                for min_leaf in spec["min_samples_leaf"]:
                    for weighting in spec["master_weighting"]:
                        parameters = {
                            "max_features": max_features,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "master_weighting": weighting,
                        }
                        payload = json.dumps(
                            parameters, sort_keys=True, separators=(",", ":")
                        )
                        suffix = hashlib.sha256(
                            payload.encode("utf-8")
                        ).hexdigest()[:10]
                        order = len(candidates)
                        candidates.append(
                            Candidate(
                                order=order,
                                candidate_id=(
                                    f"{representation}__random_forest__{suffix}"
                                ),
                                representation=representation,
                                max_features=max_features,
                                max_depth=max_depth,
                                min_samples_leaf=int(min_leaf),
                                master_weighting=str(weighting),
                            )
                        )
    return candidates


def candidate_lookup(
    candidates: Iterable[Candidate],
) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def master_sample_weights(groups: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "uniform_rows":
        return None
    if mode != "inverse_master_frequency":
        raise ValueError(f"Unknown master weighting: {mode}")
    values, counts = np.unique(np.asarray(groups, dtype=str), return_counts=True)
    count_by_group = dict(zip(values, counts))
    weights = np.asarray(
        [1.0 / count_by_group[str(group)] for group in groups], dtype=float
    )
    return weights / weights.mean()


def fit_estimator(
    candidate: Candidate,
    protocol: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> RandomForestClassifier:
    spec = protocol["random_forest"]
    estimator = RandomForestClassifier(
        n_estimators=int(spec["n_estimators"]),
        criterion=str(spec["criterion"]),
        bootstrap=bool(spec["bootstrap"]),
        class_weight=str(spec["class_weight"]),
        max_features=candidate.max_features,
        max_depth=candidate.max_depth,
        min_samples_leaf=candidate.min_samples_leaf,
        random_state=int(seed),
        n_jobs=int(spec["estimator_jobs"]),
    )
    estimator.fit(
        x,
        y,
        sample_weight=master_sample_weights(groups, candidate.master_weighting),
    )
    return estimator


def estimator_scores(
    estimator: RandomForestClassifier, x: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
    return np.log(np.clip(probabilities, 1.0e-12, 1.0))
