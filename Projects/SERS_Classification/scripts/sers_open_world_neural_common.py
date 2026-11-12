#!/usr/bin/env python3
"""Frozen neural controls and common outputs for the SERS open-world study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import run_sers_standard_vae_final as standard_final
import run_sers_structured_vae_selection as structured_selection
import sers_baseline_common as baseline
import sers_contrastive_common as contrastive
import sers_open_world_common as open_common
import sers_structured_vae_common as structured
import sers_vae_common as standard


REGISTRY_VERSION = "sers-open-world-neural-controls-v1"


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unexpected neural-control registry")
    if value.get("status_before_execution") != (
        "predeclared_before_neural_true_unknown_outcomes"
    ):
        raise ValueError("Neural-control registry was not predeclared")
    return value


def control_record(
    registry: dict[str, Any], method: str
) -> dict[str, Any]:
    matches = [
        record
        for record in registry["controls"]
        if record["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one neural control for {method}")
    return dict(matches[0])


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update(b"utf8-string-array-v1")
        for value in array.reshape(-1):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def spectral_composite_per_row(
    target: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    difference = reconstruction - target
    absolute = np.abs(difference)
    smooth_l1 = np.where(
        absolute < 1.0, 0.5 * np.square(difference), absolute - 0.5
    ).mean(axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    reconstruction_norm = np.linalg.norm(reconstruction, axis=1)
    denominator = np.maximum(target_norm * reconstruction_norm, 1.0e-12)
    cosine = np.sum(target * reconstruction, axis=1) / denominator
    angle = np.arccos(np.clip(cosine, -1.0 + 1.0e-7, 1.0 - 1.0e-7))
    angle = angle / np.pi
    derivative = np.mean(
        np.abs(np.diff(reconstruction, axis=1) - np.diff(target, axis=1)),
        axis=1,
    )
    return smooth_l1 + 0.1 * angle + 0.1 * derivative


@dataclass
class EmbeddingDensity:
    scaler: StandardScaler
    pca: PCA
    classes: np.ndarray
    centers: dict[str, np.ndarray]
    pooled_precision: np.ndarray
    class_precisions: dict[str, np.ndarray]
    class_log_determinants: dict[str, float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(values))

    def mahalanobis(self, values: np.ndarray) -> np.ndarray:
        embedding = self.transform(values)
        distances = []
        for label in self.classes:
            delta = embedding - self.centers[str(label)]
            distances.append(
                np.einsum(
                    "ij,jk,ik->i",
                    delta,
                    self.pooled_precision,
                    delta,
                )
            )
        return np.maximum(np.min(np.column_stack(distances), axis=1), 0.0)

    def negative_log_class_density(self, values: np.ndarray) -> np.ndarray:
        embedding = self.transform(values)
        dimension_constant = embedding.shape[1] * np.log(2.0 * np.pi)
        negative_log_likelihoods = []
        for label in self.classes:
            delta = embedding - self.centers[str(label)]
            quadratic = np.einsum(
                "ij,jk,ik->i",
                delta,
                self.class_precisions[str(label)],
                delta,
            )
            negative_log_likelihoods.append(
                0.5
                * (
                    quadratic
                    + self.class_log_determinants[str(label)]
                    + dimension_constant
                )
            )
        return np.min(np.column_stack(negative_log_likelihoods), axis=1)


def fit_embedding_density(
    values: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    seed: int,
    registry: dict[str, Any],
) -> EmbeddingDensity:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    spec = registry["embedding_density"]
    components = min(
        int(spec["pca_components"]),
        scaled.shape[1],
        max(1, len(scaled) - 1),
    )
    pca = PCA(
        n_components=components,
        whiten=bool(spec["pca_whiten"]),
        random_state=seed,
    ).fit(scaled)
    embedding = pca.transform(scaled)
    centers = {
        str(label): embedding[labels == label].mean(axis=0)
        for label in classes
    }
    residuals = np.vstack(
        [
            embedding[labels == label] - centers[str(label)]
            for label in classes
        ]
    )
    def stable_covariance_parts(
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return a reproducible inverse/logdet for near-singular embeddings."""
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        floor = max(scale * 1.0e-10, 1.0e-12)
        eigenvalues = np.maximum(eigenvalues, floor)
        precision = (
            eigenvectors
            @ np.diag(1.0 / eigenvalues)
            @ eigenvectors.T
        )
        return precision, float(np.sum(np.log(eigenvalues)))

    pooled_covariance = LedoitWolf(
        assume_centered=True
    ).fit(residuals).covariance_
    pooled_precision, _ = stable_covariance_parts(pooled_covariance)
    class_precisions: dict[str, np.ndarray] = {}
    class_log_determinants: dict[str, float] = {}
    for label in classes:
        class_values = embedding[labels == label]
        covariance = LedoitWolf().fit(class_values).covariance_
        precision, log_determinant = stable_covariance_parts(covariance)
        class_precisions[str(label)] = precision
        class_log_determinants[str(label)] = log_determinant
    return EmbeddingDensity(
        scaler=scaler,
        pca=pca,
        classes=classes,
        centers=centers,
        pooled_precision=pooled_precision,
        class_precisions=class_precisions,
        class_log_determinants=class_log_determinants,
    )


@dataclass
class HeadOutput:
    logits: np.ndarray
    classes: np.ndarray
    embedding: np.ndarray
    reconstruction_error: np.ndarray | None


@dataclass
class NeuralFit:
    method: str
    representation: str
    model: torch.nn.Module
    train_values: np.ndarray
    train_manifest: pd.DataFrame
    classes: np.ndarray
    train_embeddings: dict[str, np.ndarray]
    probes: dict[str, Any]
    densities: dict[str, EmbeddingDensity]
    heads: tuple[str, ...]
    applicable_scores: tuple[str, ...]
    primary_score: str
    device: torch.device
    instrument_mapping: dict[str, int] | None = None
    sensor_mapping: dict[str, int] | None = None

    def _domain_indices(
        self, manifest: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.instrument_mapping is None or self.sensor_mapping is None:
            raise ValueError("Structured model mappings are unavailable")
        instrument = (
            manifest["instrument"]
            .astype(str)
            .map(self.instrument_mapping)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        sensor = (
            manifest["sensor_family"]
            .astype(str)
            .map(self.sensor_mapping)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        return instrument, sensor

    def output(
        self,
        values: np.ndarray,
        manifest: pd.DataFrame,
        head: str,
    ) -> HeadOutput:
        values = np.asarray(values, dtype=np.float32)
        if self.method == "siamese_triplet":
            embedding = baseline.embed_siamese(
                self.model, values, self.device
            )
            train_embedding = self.train_embeddings[head]
            prototypes = np.vstack(
                [
                    train_embedding[
                        self.train_manifest["target_analyte"]
                        .astype(str)
                        .to_numpy()
                        == label
                    ].mean(axis=0)
                    for label in self.classes
                ]
            )
            logits = -np.sum(
                (
                    embedding[:, None, :]
                    - prototypes[None, :, :]
                )
                ** 2,
                axis=2,
            )
            return HeadOutput(logits, self.classes, embedding, None)
        if self.method == "supervised_contrastive":
            embedding, logits = contrastive.embed(
                self.model, values, self.device
            )
            return HeadOutput(logits, self.classes, embedding, None)
        if self.method == "standard_vae":
            reconstruction, embedding, _ = standard.vae_outputs(
                self.model, values, self.device
            )
            probe = self.probes[head]
            probabilities = np.asarray(
                probe.predict_proba(embedding), dtype=float
            )
            logits = np.log(np.clip(probabilities, 1.0e-12, 1.0))
            classes = open_common.estimator_classes(probe)
            error = spectral_composite_per_row(values, reconstruction)
            return HeadOutput(logits, classes, embedding, error)
        if self.method == "structured_vae":
            instrument, sensor = self._domain_indices(manifest)
            outputs = structured.outputs(
                self.model, values, instrument, sensor, self.device
            )
            key = "chemical_mu" if head == "chemical" else "union_mu"
            embedding = outputs[key]
            probe = self.probes[head]
            probabilities = np.asarray(
                probe.predict_proba(embedding), dtype=float
            )
            logits = np.log(np.clip(probabilities, 1.0e-12, 1.0))
            classes = open_common.estimator_classes(probe)
            error = spectral_composite_per_row(
                values, outputs["reconstruction"]
            )
            return HeadOutput(logits, classes, embedding, error)
        raise ValueError(f"Unsupported neural method: {self.method}")

    def scores(
        self,
        output: HeadOutput,
        head: str,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        logits = open_common.align_scores(
            output.logits, output.classes, self.classes
        )
        probabilities = open_common.probabilities_from_scores(
            logits, temperature
        )
        scores: dict[str, np.ndarray] = {
            "one_minus_calibrated_max_probability": (
                1.0 - probabilities.max(axis=1)
            ),
            "predictive_entropy": open_common.predictive_entropy(
                probabilities
            ),
            "minimum_class_mahalanobis_distance": self.densities[
                head
            ].mahalanobis(output.embedding),
        }
        if "minimum_prototype_squared_distance" in self.applicable_scores:
            scores["minimum_prototype_squared_distance"] = -np.max(
                logits, axis=1
            )
        if "energy" in self.applicable_scores:
            scores["energy"] = -logsumexp(logits, axis=1)
        if "negative_log_class_latent_density" in self.applicable_scores:
            scores[
                "negative_log_class_latent_density"
            ] = self.densities[head].negative_log_class_density(
                output.embedding
            )
        if "spectral_composite_reconstruction_error" in self.applicable_scores:
            if output.reconstruction_error is None:
                raise ValueError("Reconstruction score requested without decoder")
            scores[
                "spectral_composite_reconstruction_error"
            ] = output.reconstruction_error
        missing = set(self.applicable_scores) - set(scores)
        if missing:
            raise ValueError(f"Missing neural anomaly scores: {sorted(missing)}")
        return logits, probabilities, {
            name: scores[name] for name in self.applicable_scores
        }


def method_spec(
    registry: dict[str, Any], method: str
) -> dict[str, Any]:
    return control_record(registry, method)


def _train_model(
    method: str,
    values: np.ndarray,
    manifest: pd.DataFrame,
    validation_values: np.ndarray,
    validation_manifest: pd.DataFrame,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    if method == "siamese_triplet":
        trained = baseline.train_siamese(
            values,
            manifest["target_analyte"].astype(str),
            manifest["instrument"].astype(str),
            manifest["observation_uid"].astype(str),
            seed,
            device,
            epochs=epochs,
            margin=0.2,
            circular_shift=False,
        )
        return trained.model, trained.history, {
            "state_sha256": trained.state_sha256,
            "parameter_count": trained.parameter_count,
        }
    if method == "supervised_contrastive":
        protocol = contrastive.load_protocol(
            Path("configs/sers_supervised_contrastive_v1.json")
        )
        spec = contrastive.ModelSpec(
            name="open_world_frozen_successor",
            representation="derivative_1",
            architecture="legacy",
            embedding_dimension=64,
            classification_weight=1.0,
            supervised_contrastive_weight=0.5,
            pair_margin_weight=0.25,
            contrastive_temperature=0.1,
            pair_margin=0.2,
            domain_aware_positives=True,
            hard_negative_mining=True,
            domain_balanced_batches=True,
        )
        trained = contrastive.train(
            values,
            manifest,
            None,
            None,
            spec,
            protocol,
            seed,
            device,
            fixed_epochs=epochs,
        )
        return trained.model, trained.history, {
            "state_sha256": trained.state_sha256,
            "parameter_count": trained.parameters["total_parameters"],
        }
    if method == "standard_vae":
        selected = json.loads(
            Path(
                "Workspace/sers_standard_vae/standard_vae_v1/"
                "selected_configuration.json"
            ).read_text()
        )
        config = standard_final.config_from_selected(selected)
        model, history, state_sha256 = standard_final.train_vae_fixed(
            values,
            manifest["observation_uid"].astype(str),
            config,
            epochs,
            seed,
            device,
        )
        return model, history, {
            "state_sha256": state_sha256,
            "parameter_count": baseline.model_parameter_count(model),
        }
    if method == "structured_vae":
        closure = json.loads(
            Path(
                "Workspace/sers_structured_vae/structured_vae_v1/"
                "inner_selection_closure.json"
            ).read_text()
        )
        config = structured_selection.config_from_record(
            closure["selected_configuration"]
        )
        if config.maximum_epoch != epochs:
            raise ValueError("Structured epoch count differs from frozen config")
        target_mapping = {
            label: index
            for index, label in enumerate(
                sorted(manifest["target_analyte"].astype(str).unique())
            )
        }
        all_manifest = pd.concat(
            [manifest, validation_manifest], ignore_index=True
        )
        instrument_mapping = {
            label: index
            for index, label in enumerate(
                sorted(all_manifest["instrument"].astype(str).unique())
            )
        }
        sensor_mapping = {
            label: index
            for index, label in enumerate(
                sorted(all_manifest["sensor_family"].astype(str).unique())
            )
        }
        history, states, _ = structured.train_registered_checkpoints(
            values,
            manifest,
            validation_values,
            validation_manifest,
            target_mapping,
            instrument_mapping,
            sensor_mapping,
            config,
            seed,
            [epochs],
            device,
        )
        state = states[epochs]
        model = structured.build_model_from_state(
            values.shape[1],
            config,
            len(target_mapping),
            len(instrument_mapping),
            len(sensor_mapping),
            state,
            device,
        )
        return model, history, {
            "state_sha256": baseline.state_dict_sha256(state),
            "parameter_count": baseline.model_parameter_count(model),
            "target_mapping": target_mapping,
            "instrument_mapping": instrument_mapping,
            "sensor_mapping": sensor_mapping,
        }
    raise ValueError(f"Unsupported neural method: {method}")


def _build_model_from_payload(
    method: str,
    input_length: int,
    classes: np.ndarray,
    payload: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    state = payload["state_dict"]
    if baseline.state_dict_sha256(state) != payload["state_sha256"]:
        raise ValueError("Cached neural state hash mismatch")
    if method == "siamese_triplet":
        model: torch.nn.Module = baseline.SiameseEncoder(input_length)
    elif method == "supervised_contrastive":
        model = contrastive.ContrastiveClassifier(
            input_length,
            len(classes),
            "legacy",
            64,
        )
    elif method == "standard_vae":
        model = standard.ConvVariationalAutoencoder(
            input_length, (8, 16), 64
        )
    elif method == "structured_vae":
        closure = json.loads(
            Path(
                "Workspace/sers_structured_vae/structured_vae_v1/"
                "inner_selection_closure.json"
            ).read_text()
        )
        config = structured_selection.config_from_record(
            closure["selected_configuration"]
        )
        metadata = payload["method_metadata"]
        model = structured.PartitionedSERSVAE(
            input_length,
            config.chemical_dimension,
            config.nuisance_dimension,
            len(metadata["target_mapping"]),
            len(metadata["instrument_mapping"]),
            len(metadata["sensor_mapping"]),
            config.condition_decoder,
        )
    else:
        raise ValueError(f"Unsupported neural method: {method}")
    model.load_state_dict(state)
    return model.to(device)


def load_or_train(
    method: str,
    registry: dict[str, Any],
    values: np.ndarray,
    manifest: pd.DataFrame,
    validation_values: np.ndarray,
    validation_manifest: pd.DataFrame,
    seed: int,
    run_identifier: str,
    cache_dir: Path,
    device: torch.device,
) -> tuple[NeuralFit, pd.DataFrame, dict[str, Any]]:
    spec = method_spec(registry, method)
    epochs = int(spec["epochs"])
    checkpoint = cache_dir / f"{run_identifier}.pt"
    history_path = cache_dir / f"{run_identifier}.csv"
    classes = np.asarray(
        sorted(manifest["target_analyte"].astype(str).unique()), dtype=str
    )
    expected = {
        "registry_version": REGISTRY_VERSION,
        "method": method,
        "run_identifier": run_identifier,
        "representation": spec["representation"],
        "epochs": epochs,
        "seed": int(seed),
        "train_uids_sha256": array_sha256(
            manifest["observation_uid"].astype(str).to_numpy()
        ),
        "train_values_sha256": array_sha256(
            np.asarray(values, dtype=np.float32)
        ),
        "validation_uids_sha256": array_sha256(
            validation_manifest["observation_uid"].astype(str).to_numpy()
        ),
        "classes": classes.tolist(),
    }
    if checkpoint.exists() and history_path.exists():
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if payload["metadata"] != expected:
            raise ValueError(f"Stale neural cache: {run_identifier}")
        model = _build_model_from_payload(
            method, values.shape[1], classes, payload, device
        )
        history = pd.read_csv(history_path)
        method_metadata = dict(payload.get("method_metadata", {}))
        cache_status = "loaded"
    else:
        model, history, method_metadata = _train_model(
            method,
            values,
            manifest,
            validation_values,
            validation_manifest,
            seed,
            epochs,
            device,
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        state_sha256 = baseline.state_dict_sha256(state)
        if method_metadata["state_sha256"] != state_sha256:
            raise ValueError("Training-returned neural state hash mismatch")
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": expected,
                "state_dict": state,
                "state_sha256": state_sha256,
                "method_metadata": method_metadata,
            },
            checkpoint,
        )
        history.to_csv(history_path, index=False)
        cache_status = "trained"

    train_embeddings: dict[str, np.ndarray] = {}
    probes: dict[str, Any] = {}
    heads: tuple[str, ...]
    instrument_mapping = method_metadata.get("instrument_mapping")
    sensor_mapping = method_metadata.get("sensor_mapping")
    if method == "siamese_triplet":
        heads = ("default",)
        train_embeddings["default"] = baseline.embed_siamese(
            model, values, device
        )
    elif method == "supervised_contrastive":
        heads = ("default",)
        embedding, _ = contrastive.embed(model, values, device)
        train_embeddings["default"] = embedding
    elif method == "standard_vae":
        heads = ("default",)
        _, embedding, _ = standard.vae_outputs(model, values, device)
        train_embeddings["default"] = embedding
        probes["default"] = baseline.fit_latent_probe_model(
            embedding, manifest["target_analyte"], seed
        )
    elif method == "structured_vae":
        heads = tuple(spec["prediction_heads"])
        if instrument_mapping is None or sensor_mapping is None:
            raise ValueError("Structured cache lacks domain mappings")
        instrument = (
            manifest["instrument"]
            .astype(str)
            .map(instrument_mapping)
            .to_numpy(dtype=np.int64)
        )
        sensor = (
            manifest["sensor_family"]
            .astype(str)
            .map(sensor_mapping)
            .to_numpy(dtype=np.int64)
        )
        output = structured.outputs(
            model, values, instrument, sensor, device
        )
        for head, key in (
            ("chemical", "chemical_mu"),
            ("union", "union_mu"),
        ):
            train_embeddings[head] = output[key]
            probes[head] = baseline.fit_latent_probe_model(
                output[key], manifest["target_analyte"], seed
            )
    else:
        raise ValueError(f"Unsupported neural method: {method}")

    labels = manifest["target_analyte"].astype(str).to_numpy()
    densities = {
        head: fit_embedding_density(
            train_embeddings[head],
            labels,
            classes,
            open_common.stable_seed(seed, method, head, "density"),
            registry,
        )
        for head in heads
    }
    fit = NeuralFit(
        method=method,
        representation=str(spec["representation"]),
        model=model,
        train_values=values,
        train_manifest=manifest.reset_index(drop=True),
        classes=classes,
        train_embeddings=train_embeddings,
        probes=probes,
        densities=densities,
        heads=heads,
        applicable_scores=tuple(spec["applicable_scores"]),
        primary_score=str(spec["primary_score"]),
        device=device,
        instrument_mapping=instrument_mapping,
        sensor_mapping=sensor_mapping,
    )
    registry_row = {
        **expected,
        "state_sha256": method_metadata["state_sha256"],
        "parameter_count": int(method_metadata["parameter_count"]),
        "cache_status": cache_status,
        "checkpoint": str(checkpoint),
        "history": str(history_path),
    }
    return fit, history, registry_row
