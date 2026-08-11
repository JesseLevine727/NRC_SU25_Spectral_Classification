"""Outcome-blind negative-control primitives for the P03 T3 benchmark."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.classical import FitAudit, UnsupportedCandidate
from atlas_sers.splits.p02 import instrument_family


@dataclass(frozen=True)
class MasterLabelPermutation:
    labels: np.ndarray
    mapping_sha256: str
    label_sha256: str
    masters: int
    fixed_points: int


def permute_master_labels(
    metadata: pd.DataFrame, *, seed: int
) -> MasterLabelPermutation:
    """Permute source labels by physical master within each station block."""

    required = {"observation_uid", "master_sample_id", "station", "target_analyte"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Permutation metadata misses fields: {missing}")
    frame = metadata.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("Permutation source role is empty.")
    if frame.observation_uid.astype(str).duplicated().any():
        raise ValueError("Permutation source role contains duplicate observation UIDs.")
    consistency = frame.groupby("master_sample_id", dropna=False).agg(
        stations=("station", "nunique"), labels=("target_analyte", "nunique")
    )
    if consistency.gt(1).any(axis=None):
        raise ValueError("A physical master has multiple station or target labels.")
    rng = np.random.default_rng(int(seed))
    mapping: dict[str, str] = {}
    fixed_points = 0
    for station, block in frame.groupby("station", sort=True, dropna=False):
        del station
        master_table = (
            block[["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .astype(str)
            .sort_values("master_sample_id", kind="stable")
            .reset_index(drop=True)
        )
        masters = master_table.master_sample_id.to_numpy()
        original = master_table.target_analyte.to_numpy()
        permuted = original[rng.permutation(len(original))]
        mapping.update(dict(zip(masters, permuted, strict=True)))
        fixed_points += int(np.sum(original == permuted))
        if sorted(original.tolist()) != sorted(permuted.tolist()):
            raise RuntimeError("Master-label permutation changed class support.")
    labels = frame.master_sample_id.astype(str).map(mapping).to_numpy(dtype=str)
    if pd.isna(labels).any():
        raise RuntimeError("Master-label permutation did not cover every source row.")
    mapping_records = [
        {"master_sample_id": master, "permuted_label": mapping[master]}
        for master in sorted(mapping)
    ]
    label_records = [
        {"observation_uid": str(uid), "fit_label": str(label)}
        for uid, label in zip(frame.observation_uid, labels, strict=True)
    ]
    return MasterLabelPermutation(
        labels=labels,
        mapping_sha256=sha256_value(mapping_records),
        label_sha256=sha256_value(label_records),
        masters=len(mapping),
        fixed_points=fixed_points,
    )


def metadata_feature_frame(
    metadata: pd.DataFrame,
    *,
    categorical_features: list[str] | tuple[str, ...],
    numeric_features: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Materialize only the frozen row-local acquisition metadata allowlist."""

    frame = metadata.copy()
    if "instrument_family" in categorical_features and "instrument_family" not in frame:
        if "instrument" not in frame:
            raise ValueError("instrument_family cannot be derived without instrument.")
        frame["instrument_family"] = frame.instrument.astype(str).map(instrument_family)
    requested = [*categorical_features, *numeric_features]
    if len(requested) != len(set(requested)):
        raise ValueError("Metadata-only allowlist contains a duplicate feature.")
    missing = sorted(set(requested) - set(frame))
    if missing:
        raise ValueError(f"Metadata-only source misses frozen features: {missing}")
    features = frame[requested].copy()
    for column in categorical_features:
        values = features[column].astype("object")
        features[column] = values.where(pd.notna(values), "__MISSING__").astype(str)
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


class MetadataOnlyClassifier:
    """Elastic-net classifier that cannot accept a spectral intensity matrix."""

    input_kind = "acquisition_metadata"

    def __init__(
        self,
        *,
        base_parameters: dict[str, Any],
        categorical_features: list[str] | tuple[str, ...],
        numeric_features: list[str] | tuple[str, ...],
        random_state: int | None = None,
    ) -> None:
        self.base_parameters = dict(base_parameters)
        self.categorical_features = tuple(str(value) for value in categorical_features)
        self.numeric_features = tuple(str(value) for value in numeric_features)
        self.random_state = random_state
        self.fit_audit: FitAudit | None = None

    def _features(self, metadata: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(metadata, pd.DataFrame):
            raise UnsupportedCandidate("metadata_control_requires_dataframe_input")
        return metadata_feature_frame(
            metadata,
            categorical_features=self.categorical_features,
            numeric_features=self.numeric_features,
        )

    def fit(
        self,
        metadata: pd.DataFrame,
        y: np.ndarray,
        *,
        observation_uids: np.ndarray,
        master_ids: np.ndarray,
    ) -> MetadataOnlyClassifier:
        features = self._features(metadata)
        labels = np.asarray(y).astype(str)
        observations = np.asarray(observation_uids).astype(str)
        masters = np.asarray(master_ids).astype(str)
        if not (len(features) == len(labels) == len(observations) == len(masters)):
            raise UnsupportedCandidate("metadata_control_fit_length_mismatch")
        if len(np.unique(labels)) < 2:
            raise UnsupportedCandidate("fit_role_contains_fewer_than_two_classes")
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("numeric", numeric, list(self.numeric_features)),
                ("categorical", categorical, list(self.categorical_features)),
            ],
            remainder="drop",
        )
        classifier = LogisticRegression(
            C=float(self.base_parameters["C"]),
            class_weight="balanced",
            l1_ratio=float(self.base_parameters["l1_ratio"]),
            max_iter=5000,
            random_state=self.random_state,
            solver="saga",
        )
        self.pipeline_ = Pipeline([("metadata", transformer), ("classifier", classifier)])
        self.pipeline_.fit(features, labels)
        self.classes_ = self.pipeline_.classes_
        self.fit_audit = FitAudit(
            observation_uid_sha256=sha256_value(sorted(observations.tolist())),
            master_uid_sha256=sha256_value(sorted(set(masters.tolist()))),
            domain_uid_sha256=None,
            observations=len(observations),
            masters=len(set(masters.tolist())),
        )
        self.source_state_sha256_ = sha256_value(
            {
                "observation_uid_sha256": self.fit_audit.observation_uid_sha256,
                "categorical_features": self.categorical_features,
                "numeric_features": self.numeric_features,
                "classes": self.classes_.astype(str).tolist(),
                "serialized_pipeline_sha256": sha256_value(
                    list(pickle.dumps(self.pipeline_, protocol=pickle.HIGHEST_PROTOCOL))
                ),
            }
        )
        return self

    def scores(self, metadata: pd.DataFrame) -> np.ndarray:
        features = self._features(metadata)
        scores = np.asarray(self.pipeline_.decision_function(features), dtype=np.float64)
        if scores.ndim == 1:
            return np.column_stack([-scores, scores])
        return scores

    def probabilities(self, metadata: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline_.predict_proba(self._features(metadata)), dtype=np.float64)

    def predict(self, metadata: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline_.predict(self._features(metadata))).astype(str)

    def serialized_size_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


def build_metadata_only_classifier(
    parameters: dict[str, Any], *, seed: int | None = None
) -> MetadataOnlyClassifier:
    required = {"base_parameters", "categorical_features", "numeric_features"}
    missing = sorted(required - set(parameters))
    if missing:
        raise UnsupportedCandidate(f"metadata_control_parameters_missing:{missing}")
    return MetadataOnlyClassifier(
        base_parameters=dict(parameters["base_parameters"]),
        categorical_features=list(parameters["categorical_features"]),
        numeric_features=list(parameters["numeric_features"]),
        random_state=seed,
    )


def metadata_control_candidate_registry(control_registry: pd.DataFrame) -> pd.DataFrame:
    """Normalize only the frozen metadata-control rows to the P03 selection schema."""

    required = {
        "control_candidate_id",
        "control_type",
        "model_id",
        "parameters_json",
        "configuration_sha256",
        "declared_candidate_order",
        "complexity_rank",
        "seed_count",
    }
    missing = sorted(required - set(control_registry))
    if missing:
        raise ValueError(f"Control registry misses metadata-selection fields: {missing}")
    candidates = control_registry[
        control_registry.control_type.astype(str).eq("acquisition_metadata_only")
    ].copy()
    candidates = candidates.rename(
        columns={
            "control_candidate_id": "candidate_id",
            "configuration_sha256": "hyperparameter_sha256",
        }
    )
    if len(candidates) != 30 or not candidates.candidate_id.astype(str).is_unique:
        raise ValueError("Metadata-only candidate registry is not the frozen 30-row grid.")
    return candidates
