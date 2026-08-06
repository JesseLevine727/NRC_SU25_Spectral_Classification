from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_sers.exploration.structure import _analyze_level


def _contract() -> dict:
    return {
        "analysis_seeds": [17, 18, 19],
        "exploration": {
            "kmeans_k": [2, 3, 4],
            "kmeans_n_init": 10,
            "hdbscan_min_cluster_size": [8, 12],
            "hdbscan_min_samples": [3, 5],
            "umap_neighbors": 8,
            "umap_min_dist": 0.1,
            "umap_metric": "cosine",
            "tsne_perplexity": 10.0,
            "tsne_init": "pca",
            "tsne_learning_rate": "auto",
        },
    }


def test_descriptive_geometry_is_complete_and_reproducible() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(36, 20))
    metadata = pd.DataFrame(
        {
            "observation_uid": [f"OBS-{index}" for index in range(36)],
            "master_sample_id": [index // 2 for index in range(36)],
            "target_analyte": [f"target-{index % 3}" for index in range(36)],
            "instrument": [f"instrument-{index % 2}" for index in range(36)],
            "sensor_family": [f"sensor-{index % 2}" for index in range(36)],
            "station": [f"station-{index % 2}" for index in range(36)],
            "quality_group": ["clear" if index % 3 else "flagged" for index in range(36)],
        }
    )
    first = _analyze_level(
        matrix,
        metadata,
        representation_id="synthetic",
        level="spectrum",
        contract=_contract(),
    )
    second = _analyze_level(
        matrix,
        metadata,
        representation_id="synthetic",
        level="spectrum",
        contract=_contract(),
    )
    assert first[0].shape == second[0].shape == (36, 17)
    assert np.array_equal(
        first[0][["pca_1", "pca_2", "umap_1", "umap_2", "tsne_1", "tsne_2"]],
        second[0][["pca_1", "pca_2", "umap_1", "umap_2", "tsne_1", "tsne_2"]],
    )
    for first_rows, second_rows in zip(first[1:4], second[1:4], strict=True):
        pd.testing.assert_frame_equal(
            pd.DataFrame(first_rows),
            pd.DataFrame(second_rows),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    for key in first[4]:
        if isinstance(first[4][key], float):
            assert np.isclose(first[4][key], second[4][key], rtol=1e-12, atol=1e-12)
        else:
            assert first[4][key] == second[4][key]
