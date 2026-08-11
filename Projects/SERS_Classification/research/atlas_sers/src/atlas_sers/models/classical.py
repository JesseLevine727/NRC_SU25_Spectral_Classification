"""Classical estimators used by the frozen P03 candidate registry."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from atlas_sers.governance.canonical import sha256_value

STOCHASTIC_MODELS = {"C-RANDOM-FOREST", "C-EXTRA-TREES"}


class UnsupportedCandidate(ValueError):
    """Raised when a frozen candidate is invalid for the authorized fit role."""


def _as_float_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise UnsupportedCandidate("fit_matrix_must_be_nonempty_2d")
    if not np.isfinite(matrix).all():
        raise UnsupportedCandidate("fit_matrix_contains_nonfinite_value")
    return matrix


def _class_support(labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    if len(classes) < 2:
        raise UnsupportedCandidate("fit_role_contains_fewer_than_two_classes")
    return classes


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=1, keepdims=True)


class PriorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, prior: str = "empirical") -> None:
        self.prior = prior

    def fit(self, x: np.ndarray, y: np.ndarray) -> PriorClassifier:
        del x
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise UnsupportedCandidate("fit_role_contains_fewer_than_two_classes")
        if self.prior == "empirical":
            probabilities = counts / counts.sum()
        elif self.prior == "uniform":
            probabilities = np.full(len(labels), 1 / len(labels))
        else:
            raise UnsupportedCandidate(f"unknown_prior:{self.prior}")
        self.classes_ = labels
        self.class_probabilities_ = probabilities
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self.class_probabilities_, (len(x), 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


class MasterPrototypeClassifier(BaseEstimator, ClassifierMixin):
    """Class prototypes that weight physical training masters equally."""

    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, master_ids: np.ndarray
    ) -> MasterPrototypeClassifier:
        matrix = _as_float_matrix(x)
        labels = np.asarray(y)
        masters = np.asarray(master_ids)
        if len(masters) != len(matrix):
            raise UnsupportedCandidate("master_id_length_mismatch")
        self.classes_ = _class_support(labels)
        master_vectors: list[np.ndarray] = []
        master_labels: list[Any] = []
        for master in sorted(np.unique(masters), key=str):
            index = masters == master
            unique_labels = np.unique(labels[index])
            if len(unique_labels) != 1:
                raise UnsupportedCandidate("physical_master_has_multiple_labels")
            master_vectors.append(matrix[index].mean(axis=0))
            master_labels.append(unique_labels[0])
        vectors = np.vstack(master_vectors)
        master_labels_array = np.asarray(master_labels)
        self.prototypes_ = np.vstack(
            [vectors[master_labels_array == label].mean(axis=0) for label in self.classes_]
        )
        if self.metric not in {"cosine", "pearson", "spectral_angle"}:
            raise UnsupportedCandidate(f"unknown_spectral_metric:{self.metric}")
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(x)
        prototypes = self.prototypes_
        if self.metric == "pearson":
            matrix = matrix - matrix.mean(axis=1, keepdims=True)
            prototypes = prototypes - prototypes.mean(axis=1, keepdims=True)
        numerator = matrix @ prototypes.T
        denominator = (
            np.linalg.norm(matrix, axis=1, keepdims=True)
            * np.linalg.norm(prototypes, axis=1, keepdims=True).T
        )
        cosine = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        cosine = np.clip(cosine, -1, 1)
        if self.metric == "spectral_angle":
            return -np.arccos(cosine)
        return cosine

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _stable_softmax(self.decision_function(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


class CosineNearestCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self, shrink_threshold: float | None = None) -> None:
        self.shrink_threshold = shrink_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> CosineNearestCentroid:
        matrix = _as_float_matrix(x)
        labels = np.asarray(y)
        self.classes_ = _class_support(labels)
        centroids = np.vstack([matrix[labels == label].mean(axis=0) for label in self.classes_])
        if self.shrink_threshold is not None:
            if self.shrink_threshold < 0:
                raise UnsupportedCandidate("negative_centroid_shrinkage")
            global_centroid = matrix.mean(axis=0)
            delta = centroids - global_centroid
            delta = np.sign(delta) * np.maximum(np.abs(delta) - self.shrink_threshold, 0)
            centroids = global_centroid + delta
        self.centroids_ = centroids
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(x)
        numerator = matrix @ self.centroids_.T
        denominator = (
            np.linalg.norm(matrix, axis=1, keepdims=True)
            * np.linalg.norm(self.centroids_, axis=1, keepdims=True).T
        )
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _stable_softmax(self.decision_function(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int, random_state: int | None = None) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> PLSDAClassifier:
        matrix = _as_float_matrix(x)
        labels = np.asarray(y)
        self.classes_ = _class_support(labels)
        maximum = min(matrix.shape[0] - 1, matrix.shape[1])
        if self.n_components > maximum:
            raise UnsupportedCandidate(
                f"pls_components_exceed_rank:requested={self.n_components}:maximum={maximum}"
            )
        encoder = LabelEncoder().fit(labels)
        encoded = encoder.transform(labels)
        one_hot = np.eye(len(encoder.classes_))[encoded]
        self.pls_ = PLSRegression(n_components=self.n_components, scale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.pls_.fit(matrix, one_hot)
        scores = self.pls_.transform(matrix)
        self.head_ = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=self.random_state,
            solver="lbfgs",
        ).fit(scores, labels)
        self.classes_ = self.head_.classes_
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = self.head_.decision_function(self.pls_.transform(_as_float_matrix(x)))
        if scores.ndim == 1:
            return np.column_stack([-scores, scores])
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.head_.predict_proba(self.pls_.transform(_as_float_matrix(x)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.head_.predict(self.pls_.transform(_as_float_matrix(x)))


@dataclass(frozen=True)
class FitAudit:
    observation_uid_sha256: str
    master_uid_sha256: str
    domain_uid_sha256: str | None
    observations: int
    masters: int


@dataclass(frozen=True)
class LowRankCovarianceState:
    mean: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    ridge: float
    observations: int

    @property
    def rank(self) -> int:
        return int(len(self.eigenvalues))

    def inverse_sqrt(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        base = 1.0 / np.sqrt(self.ridge)
        projected = matrix @ self.eigenvectors
        adjustment = 1.0 / np.sqrt(self.eigenvalues + self.ridge) - base
        return matrix * base + (projected * adjustment) @ self.eigenvectors.T

    def sqrt(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        base = np.sqrt(self.ridge)
        projected = matrix @ self.eigenvectors
        adjustment = np.sqrt(self.eigenvalues + self.ridge) - base
        return matrix * base + (projected * adjustment) @ self.eigenvectors.T


def _low_rank_covariance_state(
    values: np.ndarray, *, rank_cap: int, ridge_fraction: float
) -> LowRankCovarianceState:
    matrix = _as_float_matrix(values)
    if rank_cap < 2:
        raise UnsupportedCandidate("source_covariance_rank_cap_below_two")
    if not np.isfinite(ridge_fraction) or ridge_fraction <= 0:
        raise UnsupportedCandidate("source_covariance_ridge_must_be_positive")
    if len(matrix) < 3:
        raise UnsupportedCandidate("source_domain_has_fewer_than_three_master_views")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, vectors_t = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular * singular / (len(matrix) - 1)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * eigenvalues.max(initial=0)
    positive = int(np.sum(eigenvalues > tolerance))
    rank = min(rank_cap, len(matrix) - 1, matrix.shape[1], positive)
    if rank < 2:
        raise UnsupportedCandidate("source_domain_covariance_rank_below_two")
    total_variance = float(eigenvalues.sum())
    ridge = ridge_fraction * total_variance / matrix.shape[1]
    if not np.isfinite(ridge) or ridge <= 0:
        raise UnsupportedCandidate("source_domain_covariance_ridge_is_nonpositive")
    return LowRankCovarianceState(
        mean=mean,
        eigenvectors=vectors_t[:rank].T,
        eigenvalues=eigenvalues[:rank],
        ridge=ridge,
        observations=len(matrix),
    )


class SourceCovarianceAugmentedClassifier(BaseEstimator, ClassifierMixin):
    """CORAL-inspired source-to-source augmentation with raw unseen-row inference."""

    def __init__(
        self,
        *,
        base_model_id: str,
        base_parameters: dict[str, Any],
        rank_cap: int = 20,
        ridge_fraction: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.base_model_id = base_model_id
        self.base_parameters = base_parameters
        self.rank_cap = rank_cap
        self.ridge_fraction = ridge_fraction
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        master_ids: np.ndarray,
        domain_ids: np.ndarray,
    ) -> SourceCovarianceAugmentedClassifier:
        matrix = _as_float_matrix(x)
        labels = np.asarray(y).astype(str)
        masters = np.asarray(master_ids).astype(str)
        domains = np.asarray(domain_ids).astype(str)
        if not (len(matrix) == len(labels) == len(masters) == len(domains)):
            raise UnsupportedCandidate("source_covariance_metadata_length_mismatch")
        _class_support(labels)
        unique_domains = sorted(set(domains))
        if len(unique_domains) < 2:
            raise UnsupportedCandidate("source_covariance_requires_two_source_domains")
        master_labels: dict[str, str] = {}
        master_domain_views: dict[str, dict[str, np.ndarray]] = {}
        domain_views: dict[str, list[np.ndarray]] = {domain: [] for domain in unique_domains}
        for master in sorted(set(masters)):
            master_mask = masters == master
            unique_labels = set(labels[master_mask])
            if len(unique_labels) != 1:
                raise UnsupportedCandidate("physical_master_has_multiple_labels")
            master_labels[master] = next(iter(unique_labels))
            views: dict[str, np.ndarray] = {}
            for domain in sorted(set(domains[master_mask])):
                view = matrix[master_mask & (domains == domain)].mean(axis=0)
                views[domain] = view
                domain_views[domain].append(view)
            master_domain_views[master] = views
        states = {
            domain: _low_rank_covariance_state(
                np.vstack(domain_views[domain]),
                rank_cap=self.rank_cap,
                ridge_fraction=self.ridge_fraction,
            )
            for domain in unique_domains
        }
        augmented: list[np.ndarray] = []
        augmented_labels: list[str] = []
        augmented_masters: list[str] = []
        augmented_uids: list[str] = []
        for master in sorted(master_domain_views):
            views = master_domain_views[master]
            raw_master = matrix[masters == master].mean(axis=0)
            augmented.append(raw_master)
            augmented_labels.append(master_labels[master])
            augmented_masters.append(master)
            augmented_uids.append(f"source_covaug:{master}:raw")
            for destination in unique_domains:
                destination_state = states[destination]
                transformed = []
                for origin, view in sorted(views.items()):
                    origin_state = states[origin]
                    whitened = origin_state.inverse_sqrt(
                        (view - origin_state.mean)[np.newaxis, :]
                    )
                    transformed.append(
                        destination_state.sqrt(whitened)[0] + destination_state.mean
                    )
                augmented.append(np.vstack(transformed).mean(axis=0))
                augmented_labels.append(master_labels[master])
                augmented_masters.append(master)
                augmented_uids.append(f"source_covaug:{master}:to:{destination}")
        augmented_matrix = np.vstack(augmented)
        self.base_model_ = build_classical_estimator(
            self.base_model_id,
            self.base_parameters,
            fit_samples=len(augmented_matrix),
            features=augmented_matrix.shape[1],
            seed=self.random_state,
        )
        self.base_model_.fit(
            augmented_matrix,
            np.asarray(augmented_labels),
            observation_uids=np.asarray(augmented_uids),
            master_ids=np.asarray(augmented_masters),
            domain_ids=np.asarray(["source_covaug"] * len(augmented_matrix)),
        )
        self.classes_ = self.base_model_.classes_
        self.source_domain_states_ = states
        self.source_domain_order_ = tuple(unique_domains)
        self.source_domain_state_sha256_ = sha256_value(
            {
                domain: {
                    "rank": state.rank,
                    "ridge": state.ridge,
                    "observations": state.observations,
                    "mean_sha256": sha256_value(state.mean.tolist()),
                    "eigenvalue_sha256": sha256_value(state.eigenvalues.tolist()),
                }
                for domain, state in states.items()
            }
        )
        self.augmented_observations_ = len(augmented_matrix)
        self.augmented_masters_ = len(master_domain_views)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        # The unseen row is evaluated directly. No target mean, covariance,
        # batch statistic, or transform exists in this estimator API.
        return self.base_model_.scores(_as_float_matrix(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


class AuditedClassifier:
    """Thin provenance wrapper shared by every P03 estimator."""

    def __init__(
        self,
        estimator: Any,
        *,
        requires_master_ids: bool = False,
        requires_domain_ids: bool = False,
        requires_external_calibration: bool = False,
    ) -> None:
        self.estimator = estimator
        self.requires_master_ids = requires_master_ids
        self.requires_domain_ids = requires_domain_ids
        self.requires_external_calibration = requires_external_calibration
        self.fit_audit: FitAudit | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimator.classes_)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        observation_uids: np.ndarray,
        master_ids: np.ndarray,
        domain_ids: np.ndarray | None = None,
    ) -> AuditedClassifier:
        matrix = _as_float_matrix(x)
        labels = np.asarray(y)
        observations = np.asarray(observation_uids)
        masters = np.asarray(master_ids)
        domains = None if domain_ids is None else np.asarray(domain_ids)
        if not (len(matrix) == len(labels) == len(observations) == len(masters)):
            raise UnsupportedCandidate("fit_metadata_length_mismatch")
        _class_support(labels)
        if self.requires_domain_ids:
            if domains is None or len(domains) != len(matrix):
                raise UnsupportedCandidate("fit_domain_id_length_mismatch")
            self.estimator.fit(
                matrix,
                labels,
                master_ids=masters,
                domain_ids=domains,
            )
        elif self.requires_master_ids:
            self.estimator.fit(matrix, labels, master_ids=masters)
        else:
            self.estimator.fit(matrix, labels)
        self.fit_audit = FitAudit(
            observation_uid_sha256=sha256_value(sorted(str(value) for value in observations)),
            master_uid_sha256=sha256_value(sorted(set(str(value) for value in masters))),
            domain_uid_sha256=sha256_value(sorted(set(str(value) for value in domains)))
            if domains is not None
            else None,
            observations=len(observations),
            masters=len(set(str(value) for value in masters)),
        )
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(x)
        if hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(matrix), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            return scores
        if isinstance(self.estimator, NearestCentroid):
            difference = matrix[:, np.newaxis, :] - self.estimator.centroids_[np.newaxis, :, :]
            return -np.sum(difference * difference, axis=2)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        return np.log(np.clip(probabilities, 1e-7, 1 - 1e-7))

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        if self.requires_external_calibration:
            raise UnsupportedCandidate("rbf_svm_requires_external_cross_fitted_calibration")
        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)
        else:
            probabilities = _stable_softmax(self.scores(x))
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.scores(x), axis=1)]

    def serialized_size_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


def _rank_checked_pca_components(value: int | str, *, samples: int, features: int) -> int | float:
    if value == "0.95_variance":
        return 0.95
    components = int(value)
    maximum = min(samples, features)
    if components > maximum:
        raise UnsupportedCandidate(
            f"pca_components_exceed_rank:requested={components}:maximum={maximum}"
        )
    return components


def build_classical_estimator(
    model_id: str,
    parameters: dict[str, Any],
    *,
    fit_samples: int,
    features: int,
    seed: int | None = None,
) -> AuditedClassifier:
    """Construct one exact frozen candidate without touching evaluation data."""

    if model_id == "C-PRIOR":
        return AuditedClassifier(PriorClassifier(prior=str(parameters["prior"])))
    if model_id == "C-SPECTRAL-MATCH":
        return AuditedClassifier(
            MasterPrototypeClassifier(metric=str(parameters["metric"])),
            requires_master_ids=True,
        )
    if model_id == "C-NEAREST-CENTROID":
        metric = str(parameters["metric"])
        shrinkage = parameters["shrink_threshold"]
        if metric == "euclidean":
            estimator = NearestCentroid(metric="euclidean", shrink_threshold=shrinkage)
        elif metric == "cosine":
            estimator = CosineNearestCentroid(shrink_threshold=shrinkage)
        else:
            raise UnsupportedCandidate(f"unknown_centroid_metric:{metric}")
        return AuditedClassifier(estimator)
    if model_id == "C-PCA-LDA":
        components = _rank_checked_pca_components(
            parameters["pca_components"], samples=fit_samples, features=features
        )
        lda_name = str(parameters["lda"])
        if lda_name == "svd":
            lda = LinearDiscriminantAnalysis(solver="svd")
        elif lda_name == "lsqr_auto_shrinkage":
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        else:
            raise UnsupportedCandidate(f"unknown_lda_configuration:{lda_name}")
        return AuditedClassifier(Pipeline([("pca", PCA(n_components=components)), ("lda", lda)]))
    if model_id == "C-PLS-DA":
        return AuditedClassifier(PLSDAClassifier(int(parameters["components"]), seed))
    if model_id == "C-LOGREG-EN":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        class_weight="balanced",
                        l1_ratio=float(parameters["l1_ratio"]),
                        max_iter=5000,
                        random_state=seed,
                        solver="saga",
                    ),
                ),
            ]
        )
        return AuditedClassifier(estimator)
    if model_id == "C-RBF-SVM":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=float(parameters["C"]),
                        gamma=parameters["gamma"],
                        kernel="rbf",
                        class_weight="balanced",
                        decision_function_shape="ovr",
                        random_state=seed,
                    ),
                ),
            ]
        )
        return AuditedClassifier(estimator, requires_external_calibration=True)
    if model_id in STOCHASTIC_MODELS:
        common = {
            "n_estimators": int(parameters["n_estimators"]),
            "max_features": parameters["max_features"],
            "min_samples_leaf": int(parameters["min_samples_leaf"]),
            "class_weight": parameters["class_weight"],
            "n_jobs": 1,
            "random_state": seed,
        }
        if model_id == "C-RANDOM-FOREST":
            estimator = RandomForestClassifier(**common)
        else:
            estimator = ExtraTreesClassifier(
                **common,
                bootstrap=bool(parameters["bootstrap"]),
            )
        return AuditedClassifier(estimator)
    if model_id == "C-SOURCE-CORAL":
        base_model_id = str(parameters["base_model_id"])
        if base_model_id not in {"C-PCA-LDA", "C-RBF-SVM"}:
            raise UnsupportedCandidate(f"unsupported_source_covariance_base:{base_model_id}")
        estimator = SourceCovarianceAugmentedClassifier(
            base_model_id=base_model_id,
            base_parameters=dict(parameters["base_parameters"]),
            rank_cap=int(parameters.get("rank_cap", 20)),
            ridge_fraction=float(parameters.get("ridge_fraction", 1e-3)),
            random_state=seed,
        )
        return AuditedClassifier(
            estimator,
            requires_master_ids=True,
            requires_domain_ids=True,
            requires_external_calibration=base_model_id == "C-RBF-SVM",
        )
    raise UnsupportedCandidate(f"unknown_model_id:{model_id}")
