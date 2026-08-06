"""Reproducible, descriptive P01 geometry and clustering analyses."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from atlas_sers.preprocessing.representations import RepresentationBundle


def _pca_features(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    maximum = min(50, len(matrix) - 1, matrix.shape[1])
    model = PCA(n_components=maximum, svd_solver="full", random_state=seed)
    scores = model.fit_transform(matrix)
    keep = min(
        max(int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.95) + 1), 2),
        maximum,
    )
    return scores[:, :keep], scores[:, :2], model


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    nonnoise = labels != -1
    unique = np.unique(labels[nonnoise])
    if nonnoise.sum() < 3 or len(unique) < 2:
        return None
    return float(silhouette_score(features[nonnoise], labels[nonnoise]))


def _associations(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    representation_id: str,
    level: str,
    method: str,
) -> list[dict[str, Any]]:
    fields = (
        (
            "target_analyte",
            "instrument",
            "sensor_family",
            "station",
            "master_sample_id",
            "quality_group",
        )
        if level == "spectrum"
        else ("target_analyte", "station")
    )
    rows: list[dict[str, Any]] = []
    for field in fields:
        if field not in metadata:
            continue
        values = metadata[field].fillna("<missing>").astype(str).to_numpy()
        rows.append(
            {
                "representation_id": representation_id,
                "level": level,
                "method": method,
                "metadata_field": field,
                "n_units": len(labels),
                "cluster_count_including_noise": len(np.unique(labels)),
                "noise_fraction": float(np.mean(labels == -1)),
                "adjusted_rand_index": float(adjusted_rand_score(values, labels)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(values, labels)
                ),
            }
        )
    return rows


def _distance_stability(left: np.ndarray, right: np.ndarray) -> float:
    left_distance = np.linalg.norm(left[:, None, :] - left[None, :, :], axis=2)
    right_distance = np.linalg.norm(right[:, None, :] - right[None, :, :], axis=2)
    upper = np.triu_indices(len(left), k=1)
    return float(spearmanr(left_distance[upper], right_distance[upper]).statistic)


def _analyze_level(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    *,
    representation_id: str,
    level: str,
    contract: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    settings = contract["exploration"]
    seeds = [int(value) for value in contract["analysis_seeds"]]
    seed = seeds[0]
    features, pca2, pca_model = _pca_features(matrix, seed)
    clustering_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in settings["kmeans_k"]:
        if int(k) >= len(matrix):
            continue
        model = KMeans(
            n_clusters=int(k),
            n_init=int(settings["kmeans_n_init"]),
            random_state=seed,
        )
        labels = model.fit_predict(features)
        labels_by_k[int(k)] = labels
        clustering_rows.append(
            {
                "representation_id": representation_id,
                "level": level,
                "method": "kmeans",
                "parameter_1": f"k={k}",
                "parameter_2": f"n_init={settings['kmeans_n_init']}",
                "cluster_count": int(k),
                "noise_fraction": 0.0,
                "silhouette": float(silhouette_score(features, labels)),
                "inertia": float(model.inertia_),
                "persistence": np.nan,
            }
        )
    best_k = int(
        max(
            (row for row in clustering_rows if row["method"] == "kmeans"),
            key=lambda row: (row["silhouette"], -row["cluster_count"]),
        )["cluster_count"]
    )
    kmeans_labels = labels_by_k[best_k]
    kmeans_seed_labels = [
        KMeans(
            n_clusters=best_k,
            n_init=int(settings["kmeans_n_init"]),
            random_state=value,
        ).fit_predict(features)
        for value in seeds
    ]
    kmeans_stability = [
        adjusted_rand_score(left, right) for left, right in combinations(kmeans_seed_labels, 2)
    ]

    primary_hdbscan_labels: np.ndarray | None = None
    hdbscan_grid_labels: list[np.ndarray] = []
    for minimum_cluster_size in settings["hdbscan_min_cluster_size"]:
        if int(minimum_cluster_size) > len(matrix):
            continue
        for minimum_samples in settings["hdbscan_min_samples"]:
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(minimum_cluster_size),
                min_samples=int(minimum_samples),
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = model.fit_predict(features)
            hdbscan_grid_labels.append(labels)
            if int(minimum_cluster_size) == 12 and int(minimum_samples) == 5:
                primary_hdbscan_labels = labels
            clustering_rows.append(
                {
                    "representation_id": representation_id,
                    "level": level,
                    "method": "hdbscan",
                    "parameter_1": f"min_cluster_size={minimum_cluster_size}",
                    "parameter_2": f"min_samples={minimum_samples}",
                    "cluster_count": int(len(set(labels)) - (-1 in labels)),
                    "noise_fraction": float(np.mean(labels == -1)),
                    "silhouette": _safe_silhouette(features, labels),
                    "inertia": np.nan,
                    "persistence": float(np.mean(model.cluster_persistence_))
                    if len(model.cluster_persistence_)
                    else np.nan,
                }
            )
    if primary_hdbscan_labels is None:
        raise RuntimeError("The registered primary HDBSCAN setting was not executed.")

    neighbors = min(int(settings["umap_neighbors"]), len(matrix) - 1)
    perplexity = min(float(settings["tsne_perplexity"]), (len(matrix) - 1) / 3)
    umap_embeddings = [
        umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=float(settings["umap_min_dist"]),
            metric=str(settings["umap_metric"]),
            random_state=value,
            transform_seed=value,
            n_jobs=1,
        ).fit_transform(features)
        for value in seeds[:3]
    ]
    tsne_embeddings = [
        TSNE(
            n_components=2,
            perplexity=perplexity,
            init=str(settings["tsne_init"]),
            learning_rate=settings["tsne_learning_rate"],
            random_state=value,
            method="barnes_hut",
        ).fit_transform(features)
        for value in seeds[:3]
    ]
    umap2 = umap_embeddings[0]
    tsne2 = tsne_embeddings[0]
    embedding = metadata.copy().reset_index(drop=True)
    embedding.insert(0, "level", level)
    embedding.insert(0, "representation_id", representation_id)
    for prefix, values in (("pca", pca2), ("umap", umap2), ("tsne", tsne2)):
        embedding[f"{prefix}_1"] = values[:, 0]
        embedding[f"{prefix}_2"] = values[:, 1]
    embedding["kmeans_cluster"] = kmeans_labels
    embedding["hdbscan_cluster"] = primary_hdbscan_labels

    association_rows: list[dict[str, Any]] = []
    association_rows.extend(
        _associations(
            metadata,
            kmeans_labels,
            representation_id=representation_id,
            level=level,
            method=f"kmeans_k{best_k}",
        )
    )
    association_rows.extend(
        _associations(
            metadata,
            primary_hdbscan_labels,
            representation_id=representation_id,
            level=level,
            method="hdbscan_mcs12_ms5",
        )
    )
    stability_rows = [
        {
            "representation_id": representation_id,
            "level": level,
            "method": "kmeans",
            "comparison": "five_seed_pairwise_ari",
            "median_stability": float(np.median(kmeans_stability)),
            "minimum_stability": float(np.min(kmeans_stability)),
        },
        {
            "representation_id": representation_id,
            "level": level,
            "method": "hdbscan",
            "comparison": "parameter_grid_pairwise_ari",
            "median_stability": float(
                np.median(
                    [
                        adjusted_rand_score(left, right)
                        for left, right in combinations(hdbscan_grid_labels, 2)
                    ]
                )
            ),
            "minimum_stability": float(
                np.min(
                    [
                        adjusted_rand_score(left, right)
                        for left, right in combinations(hdbscan_grid_labels, 2)
                    ]
                )
            ),
        },
        {
            "representation_id": representation_id,
            "level": level,
            "method": "umap",
            "comparison": "three_seed_pairwise_distance_spearman",
            "median_stability": float(
                np.median(
                    [
                        _distance_stability(left, right)
                        for left, right in combinations(umap_embeddings, 2)
                    ]
                )
            ),
            "minimum_stability": float(
                np.min(
                    [
                        _distance_stability(left, right)
                        for left, right in combinations(umap_embeddings, 2)
                    ]
                )
            ),
        },
        {
            "representation_id": representation_id,
            "level": level,
            "method": "tsne",
            "comparison": "three_seed_pairwise_distance_spearman",
            "median_stability": float(
                np.median(
                    [
                        _distance_stability(left, right)
                        for left, right in combinations(tsne_embeddings, 2)
                    ]
                )
            ),
            "minimum_stability": float(
                np.min(
                    [
                        _distance_stability(left, right)
                        for left, right in combinations(tsne_embeddings, 2)
                    ]
                )
            ),
        },
    ]
    diagnostics = {
        "representation_id": representation_id,
        "level": level,
        "n_units": len(matrix),
        "pca_components_to_95pct": features.shape[1],
        "pc1_explained_variance": float(pca_model.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca_model.explained_variance_ratio_[1]),
        "umap_trustworthiness": float(
            trustworthiness(
                features,
                umap2,
                n_neighbors=min(neighbors, max(1, (len(matrix) - 1) // 2)),
            )
        ),
        "tsne_trustworthiness": float(
            trustworthiness(
                features,
                tsne2,
                n_neighbors=min(10, max(1, (len(matrix) - 1) // 2)),
            )
        ),
        "selected_descriptive_k": best_k,
    }
    return (
        embedding,
        clustering_rows,
        association_rows,
        stability_rows,
        diagnostics,
    )


def _master_means(matrix: np.ndarray, manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    work = manifest.reset_index(drop=True).copy()
    work["row_index"] = np.arange(len(work))
    rows: list[dict[str, Any]] = []
    means: list[np.ndarray] = []
    for master, group in work.groupby("master_sample_id", sort=True):
        indices = group.row_index.to_numpy(int)
        targets = sorted(group.target_analyte.astype(str).unique())
        stations = sorted(group.station.astype(str).unique())
        if len(targets) != 1 or len(stations) != 1:
            raise ValueError(f"Master {master} has inconsistent chemistry or station metadata.")
        rows.append(
            {
                "observation_uid": "",
                "master_sample_id": master,
                "target_analyte": targets[0],
                "station": stations[0],
                "n_spectra": len(group),
                "quality_group": "master_aggregate",
            }
        )
        means.append(matrix[indices].mean(axis=0))
    return np.asarray(means), pd.DataFrame(rows)


def analyze_structure(
    raw: np.ndarray,
    bundles: dict[str, RepresentationBundle],
    manifest: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Analyze spectrum- and master-level geometry for raw plus eight frozen views."""

    metadata = manifest[
        [
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "sensor_family",
            "station",
        ]
    ].copy()
    metadata["quality_group"] = np.where(
        manifest["tier_notes_clear_sers"].astype(bool), "notes_clear", "notes_flagged"
    )
    matrices: dict[str, np.ndarray] = {
        "R_NATIVE_COMMON_400_1849": StandardScaler().fit_transform(raw)
    }
    matrices.update({key: value.intensity.astype(float) for key, value in bundles.items()})
    embeddings: list[pd.DataFrame] = []
    cluster_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    for identifier, matrix in matrices.items():
        outputs = _analyze_level(
            matrix,
            metadata,
            representation_id=identifier,
            level="spectrum",
            contract=contract,
        )
        embeddings.append(outputs[0])
        cluster_rows.extend(outputs[1])
        association_rows.extend(outputs[2])
        stability_rows.extend(outputs[3])
        pca_rows.append(outputs[4])
        master_matrix, master_metadata = _master_means(matrix, manifest)
        master_outputs = _analyze_level(
            master_matrix,
            master_metadata,
            representation_id=identifier,
            level="master",
            contract=contract,
        )
        embeddings.append(master_outputs[0])
        cluster_rows.extend(master_outputs[1])
        association_rows.extend(master_outputs[2])
        stability_rows.extend(master_outputs[3])
        pca_rows.append(master_outputs[4])
    return {
        "embedding_rows.csv": pd.concat(embeddings, ignore_index=True),
        "clustering_runs.csv": pd.DataFrame(cluster_rows),
        "cluster_metadata_association.csv": pd.DataFrame(association_rows),
        "cluster_stability.csv": pd.DataFrame(stability_rows),
        "pca_diagnostics.csv": pd.DataFrame(pca_rows),
    }
