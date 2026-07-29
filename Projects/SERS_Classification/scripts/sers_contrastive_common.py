#!/usr/bin/env python3
"""Deterministic models and losses for supervised-contrastive SERS learning."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

import sers_baseline_common as baseline


PROTOCOL_VERSION = "sers-supervised-contrastive-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected supervised-contrastive protocol")
    if protocol.get("status_before_execution") != (
        "predeclared_before_classical_outer_results"
    ):
        raise ValueError("Supervised-contrastive protocol was not predeclared")
    return protocol


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**31 - 1
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    representation: str
    architecture: str = "legacy"
    embedding_dimension: int = 64
    classification_weight: float = 1.0
    supervised_contrastive_weight: float = 0.0
    pair_margin_weight: float = 0.0
    contrastive_temperature: float = 0.1
    pair_margin: float = 0.2
    domain_aware_positives: bool = False
    hard_negative_mining: bool = False
    domain_balanced_batches: bool = False

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self.name}__{self.representation}__{self.architecture}{self.embedding_dimension}__{suffix}"


class CompactEncoder(nn.Module):
    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dimension),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.features(values)), p=2, dim=1)


class ContrastiveClassifier(nn.Module):
    def __init__(
        self,
        input_length: int,
        class_count: int,
        architecture: str,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        if architecture == "legacy":
            self.encoder: nn.Module = baseline.ConvEncoder(
                input_length,
                channels=(16, 32),
                bottleneck_dimension=embedding_dimension,
                normalize_output=True,
            )
        elif architecture == "compact":
            self.encoder = CompactEncoder(embedding_dimension)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.classifier = nn.Linear(embedding_dimension, class_count)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        return embedding, self.classifier(embedding)


def parameter_counts(model: ContrastiveClassifier) -> dict[str, int]:
    return {
        "encoder_parameters": int(
            sum(parameter.numel() for parameter in model.encoder.parameters())
        ),
        "classifier_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.classifier.parameters()
            )
        ),
        "total_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def edge_shift(values: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return values.copy()
    output = np.empty_like(values)
    if shift > 0:
        output[:shift] = values[0]
        output[shift:] = values[:-shift]
    else:
        count = -shift
        output[-count:] = values[-1]
        output[:-count] = values[count:]
    return output


def augment(
    values: np.ndarray,
    rng: np.random.Generator,
    protocol: dict[str, Any],
) -> np.ndarray:
    config = protocol["fixed_training"]["augmentation"]
    low, high = (float(value) for value in config["scale_range"])
    output = np.asarray(values, dtype=np.float32) * float(rng.uniform(low, high))
    output = output + rng.normal(
        0.0,
        float(config["gaussian_noise_standard_deviation"]),
        size=len(output),
    ).astype(np.float32)
    maximum = int(config["edge_filled_shift_maximum_cm1"])
    return edge_shift(output, int(rng.integers(-maximum, maximum + 1)))


class StructuredRelationships:
    def __init__(self, manifest: pd.DataFrame) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.labels = self.manifest["target_analyte"].astype(str).to_numpy()
        self.instruments = self.manifest["instrument"].astype(str).to_numpy()
        self.sensors = self.manifest["sensor_family"].astype(str).to_numpy()
        self.masters = self.manifest["master_sample_id"].astype(str).to_numpy()
        self.uids = self.manifest["observation_uid"].astype(str).to_numpy()
        self.indices = np.arange(len(self.manifest), dtype=int)
        self.by_label = {
            label: np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self.by_instrument = {
            domain: np.flatnonzero(self.instruments == domain)
            for domain in np.unique(self.instruments)
        }
        self.by_sensor = {
            domain: np.flatnonzero(self.sensors == domain)
            for domain in np.unique(self.sensors)
        }
        self.by_master = {
            master: np.flatnonzero(self.masters == master)
            for master in np.unique(self.masters)
        }

    def positive(
        self,
        anchor: int,
        rng: np.random.Generator,
        domain_aware: bool,
    ) -> int:
        label = self.labels[anchor]
        candidates = self.by_label[label]
        candidates = candidates[candidates != anchor]
        if not len(candidates):
            return anchor
        if domain_aware:
            same_master = self.by_master[self.masters[anchor]]
            same_master = same_master[
                (same_master != anchor)
                & (self.labels[same_master] == label)
                & (self.instruments[same_master] != self.instruments[anchor])
            ]
            if len(same_master):
                return int(rng.choice(same_master))
            cross_instrument = candidates[
                self.instruments[candidates] != self.instruments[anchor]
            ]
            if len(cross_instrument):
                return int(rng.choice(cross_instrument))
        return int(rng.choice(candidates))

    def negative(
        self,
        anchor: int,
        rng: np.random.Generator,
        domain_aware: bool,
    ) -> int:
        label = self.labels[anchor]
        if domain_aware:
            same_instrument = self.by_instrument[self.instruments[anchor]]
            same_instrument = same_instrument[
                self.labels[same_instrument] != label
            ]
            if len(same_instrument):
                return int(rng.choice(same_instrument))
            same_sensor = self.by_sensor[self.sensors[anchor]]
            same_sensor = same_sensor[self.labels[same_sensor] != label]
            if len(same_sensor):
                return int(rng.choice(same_sensor))
        candidates = self.indices[self.labels != label]
        return int(rng.choice(candidates))

    def anchor_order(
        self,
        rng: np.random.Generator,
        domain_balanced: bool,
    ) -> np.ndarray:
        if not domain_balanced:
            return rng.permutation(self.indices)
        cells: dict[tuple[str, str], np.ndarray] = {}
        for index, (label, instrument) in enumerate(
            zip(self.labels, self.instruments)
        ):
            key = (label, instrument)
            cells[key] = np.append(
                cells.get(key, np.asarray([], dtype=int)), index
            )
        labels = list(sorted(np.unique(self.labels)))
        label_cells = {
            label: [cell for cell in cells if cell[0] == label]
            for label in labels
        }
        row_queues: dict[tuple[str, str], list[int]] = {}
        cell_queues: dict[str, list[tuple[str, str]]] = {}

        def refill_rows(cell: tuple[str, str]) -> None:
            values = cells[cell].copy()
            rng.shuffle(values)
            row_queues[cell] = values.astype(int).tolist()

        def refill_cells(label: str) -> None:
            values = list(label_cells[label])
            rng.shuffle(values)
            cell_queues[label] = values

        for cell in cells:
            refill_rows(cell)
        for label in labels:
            refill_cells(label)
        ordered: list[int] = []
        while len(ordered) < len(self.indices):
            label_cycle = list(labels)
            rng.shuffle(label_cycle)
            for label in label_cycle:
                if len(ordered) >= len(self.indices):
                    break
                if not cell_queues[label]:
                    refill_cells(label)
                cell = cell_queues[label].pop()
                if not row_queues[cell]:
                    refill_rows(cell)
                ordered.append(row_queues[cell].pop())
        return np.asarray(ordered, dtype=int)


def structured_batch(
    values: np.ndarray,
    relationships: StructuredRelationships,
    anchor_indices: np.ndarray,
    spec: ModelSpec,
    protocol: dict[str, Any],
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    positives: list[int] = []
    negatives: list[int] = []
    for anchor in anchor_indices:
        rng = np.random.default_rng(
            stable_seed(
                PROTOCOL_VERSION,
                seed,
                epoch,
                relationships.uids[anchor],
                "relationship",
            )
        )
        positives.append(
            relationships.positive(anchor, rng, spec.domain_aware_positives)
        )
        negatives.append(
            relationships.negative(anchor, rng, spec.hard_negative_mining)
        )
    selected = np.concatenate(
        [
            np.asarray(anchor_indices, dtype=int),
            np.asarray(positives, dtype=int),
            np.asarray(negatives, dtype=int),
        ]
    )
    rows: list[np.ndarray] = []
    for role_index, selected_index in enumerate(selected):
        rng = np.random.default_rng(
            stable_seed(
                PROTOCOL_VERSION,
                seed,
                epoch,
                relationships.uids[selected_index],
                role_index,
                "augmentation",
            )
        )
        rows.append(augment(values[selected_index], rng, protocol))
    class_names = sorted(np.unique(relationships.labels))
    class_index = {label: index for index, label in enumerate(class_names)}
    targets = np.asarray(
        [class_index[relationships.labels[index]] for index in selected],
        dtype=np.int64,
    )
    return (
        torch.from_numpy(np.asarray(rows, dtype=np.float32)).unsqueeze(1),
        torch.from_numpy(targets),
        len(anchor_indices),
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    similarity = embeddings @ embeddings.T / float(temperature)
    maximum = similarity.detach().max(dim=1, keepdim=True).values
    logits = similarity - maximum
    self_mask = ~torch.eye(
        len(embeddings), dtype=torch.bool, device=embeddings.device
    )
    positive_mask = (targets[:, None] == targets[None, :]) & self_mask
    exp_logits = torch.exp(logits) * self_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    )
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    mean_log_probability = (
        (positive_mask * log_probability).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -mean_log_probability[valid].mean()


def embed(
    model: ContrastiveClassifier,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1)
            batch_embedding, batch_logits = model(batch.to(device))
            embeddings.append(batch_embedding.cpu().numpy())
            logits.append(batch_logits.cpu().numpy())
    return np.vstack(embeddings), np.vstack(logits)


def prototype_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    prototypes = np.vstack(
        [train_embeddings[train_labels == label].mean(axis=0) for label in classes]
    )
    prototypes = prototypes / np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1.0e-12
    )
    return -np.sum(
        (test_embeddings[:, None, :] - prototypes[None, :, :]) ** 2, axis=2
    )


def prediction_scores(
    model: ContrastiveClassifier,
    spec: ModelSpec,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_embedding, _ = embed(model, train_values, device)
    test_embedding, logits = embed(model, test_values, device)
    classes = np.asarray(sorted(np.unique(train_labels)), dtype=str)
    scores = (
        logits
        if spec.classification_weight > 0
        else prototype_scores(
            train_embedding, train_labels, test_embedding, classes
        )
    )
    return scores, classes, train_embedding, test_embedding


def effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    probability = variance / max(float(variance.sum()), 1.0e-12)
    probability = probability[probability > 1.0e-12]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def pair_geometry(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
) -> dict[str, float]:
    labels = manifest["target_analyte"].astype(str).to_numpy()
    masters = manifest["master_sample_id"].astype(str).to_numpy()
    instruments = manifest["instrument"].astype(str).to_numpy()
    distance = np.sqrt(
        np.maximum(
            np.sum(
                (embeddings[:, None, :] - embeddings[None, :, :]) ** 2,
                axis=2,
            ),
            0.0,
        )
    )
    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)
    same_analyte = upper & (labels[:, None] == labels[None, :])
    different_analyte = upper & (labels[:, None] != labels[None, :])
    same_master_cross_instrument = (
        upper
        & (masters[:, None] == masters[None, :])
        & (instruments[:, None] != instruments[None, :])
    )
    same_mean = float(distance[same_analyte].mean())
    different_mean = float(distance[different_analyte].mean())
    return {
        "same_analyte_distance": same_mean,
        "different_analyte_distance": different_mean,
        "different_minus_same_margin": different_mean - same_mean,
        "same_master_cross_instrument_distance": (
            float(distance[same_master_cross_instrument].mean())
            if np.any(same_master_cross_instrument)
            else np.nan
        ),
        "embedding_effective_rank": effective_rank(embeddings),
    }


def evaluate_validation(
    model: ContrastiveClassifier,
    spec: ModelSpec,
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_values: np.ndarray,
    validation_manifest: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    train_labels = train_manifest["target_analyte"].astype(str).to_numpy()
    validation_labels = (
        validation_manifest["target_analyte"].astype(str).to_numpy()
    )
    scores, classes, _, validation_embedding = prediction_scores(
        model,
        spec,
        train_values,
        train_labels,
        validation_values,
        device,
    )
    predictions = classes[np.argmax(scores, axis=1)]
    geometry = pair_geometry(validation_embedding, validation_manifest)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(validation_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(validation_labels, predictions, average="macro")
        ),
        "predicted_class_count": int(len(np.unique(predictions))),
        **geometry,
    }


@dataclass
class TrainedModel:
    model: ContrastiveClassifier
    spec: ModelSpec
    history: pd.DataFrame
    best_epoch: int
    run_seed: int
    state_sha256: str
    parameters: dict[str, int]
    classes: np.ndarray


def train(
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_values: np.ndarray | None,
    validation_manifest: pd.DataFrame | None,
    spec: ModelSpec,
    protocol: dict[str, Any],
    seed: int,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> TrainedModel:
    baseline.configure_determinism(seed)
    class_names = np.asarray(
        sorted(train_manifest["target_analyte"].astype(str).unique()), dtype=str
    )
    model = ContrastiveClassifier(
        train_values.shape[1],
        len(class_names),
        spec.architecture,
        spec.embedding_dimension,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["fixed_training"]["learning_rate"]),
        weight_decay=float(protocol["fixed_training"]["weight_decay"]),
    )
    relationships = StructuredRelationships(train_manifest)
    batch_size = int(protocol["fixed_training"]["batch_anchors"])
    maximum_epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else int(protocol["fixed_training"]["maximum_epochs"])
    )
    minimum_epochs = int(protocol["fixed_training"]["minimum_epochs"])
    patience = int(protocol["fixed_training"]["early_stopping_patience"])
    gradient_clip = float(protocol["fixed_training"]["gradient_clip_norm"])
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)
    best_epoch = maximum_epochs
    stale_epochs = 0
    for epoch in range(maximum_epochs):
        model.train()
        rng = np.random.default_rng(
            stable_seed(PROTOCOL_VERSION, seed, epoch, "anchor_order")
        )
        anchors = relationships.anchor_order(
            rng, spec.domain_balanced_batches
        )
        total_loss = 0.0
        total_ce = 0.0
        total_contrastive = 0.0
        total_margin = 0.0
        seen = 0
        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start : start + batch_size]
            inputs, targets, anchor_count = structured_batch(
                train_values,
                relationships,
                batch_anchors,
                spec,
                protocol,
                seed,
                epoch,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            embeddings, logits = model(inputs)
            ce = F.cross_entropy(logits, targets)
            contrastive = supervised_contrastive_loss(
                embeddings, targets, spec.contrastive_temperature
            )
            margin = F.triplet_margin_loss(
                embeddings[:anchor_count],
                embeddings[anchor_count : 2 * anchor_count],
                embeddings[2 * anchor_count : 3 * anchor_count],
                margin=spec.pair_margin,
                p=2,
            )
            loss = (
                spec.classification_weight * ce
                + spec.supervised_contrastive_weight * contrastive
                + spec.pair_margin_weight * margin
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            count = len(inputs)
            total_loss += float(loss.detach().cpu()) * count
            total_ce += float(ce.detach().cpu()) * count
            total_contrastive += float(contrastive.detach().cpu()) * count
            total_margin += float(margin.detach().cpu()) * count
            seen += count
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(seen, 1),
            "train_cross_entropy": total_ce / max(seen, 1),
            "train_supervised_contrastive": total_contrastive / max(seen, 1),
            "train_pair_margin": total_margin / max(seen, 1),
        }
        should_validate = (
            validation_values is not None
            and validation_manifest is not None
            and (
                epoch + 1 == minimum_epochs
                or (epoch + 1 >= minimum_epochs and (epoch + 1) % 5 == 0)
            )
        )
        if should_validate:
            metrics = evaluate_validation(
                model,
                spec,
                train_values,
                train_manifest,
                validation_values,
                validation_manifest,
                device,
            )
            row.update({f"validation_{key}": value for key, value in metrics.items()})
            key = (
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
                metrics["different_minus_same_margin"],
            )
            collapse_ok = (
                metrics["different_minus_same_margin"] > 0
                and metrics["embedding_effective_rank"]
                >= float(
                    protocol["nested_selection"]["collapse_gates"][
                        "embedding_effective_rank_minimum"
                    ]
                )
                and metrics["predicted_class_count"] >= 2
            )
            if collapse_ok and key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 5
        history.append(row)
        if (
            fixed_epochs is None
            and epoch + 1 >= minimum_epochs
            and stale_epochs >= patience
            and best_state is not None
        ):
            break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum_epochs
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return TrainedModel(
        model=model,
        spec=spec,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        run_seed=seed,
        state_sha256=baseline.state_dict_sha256(state),
        parameters=parameter_counts(model),
        classes=class_names,
    )


def domain_probe(
    train_embeddings: np.ndarray,
    train_domains: Sequence[str],
    test_embeddings: np.ndarray,
    test_domains: Sequence[str],
    seed: int,
) -> float:
    train_domains = np.asarray(train_domains, dtype=str)
    test_domains = np.asarray(test_domains, dtype=str)
    supported = np.isin(test_domains, np.unique(train_domains))
    if len(np.unique(train_domains)) < 2 or not np.any(supported):
        return np.nan
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed,
    )
    model.fit(train_embeddings, train_domains)
    predictions = model.predict(test_embeddings[supported])
    return float(
        recall_score(
            test_domains[supported],
            predictions,
            labels=np.unique(test_domains[supported]),
            average="macro",
            zero_division=0,
        )
    )


def leave_one_group_out_probe(
    embeddings: np.ndarray,
    targets: Sequence[str],
    groups: Sequence[str],
    seed: int,
) -> dict[str, float | int]:
    """Probe one embedding space without treating replicate rows as independent."""
    targets = np.asarray(targets, dtype=str)
    groups = np.asarray(groups, dtype=str)
    predictions = np.full(len(targets), "", dtype=object)
    supported = np.zeros(len(targets), dtype=bool)
    unique_groups = np.unique(groups)
    for heldout_group in unique_groups:
        train_mask = groups != heldout_group
        test_mask = groups == heldout_group
        train_classes = np.unique(targets[train_mask])
        if len(train_classes) < 2:
            continue
        row_supported = test_mask & np.isin(targets, train_classes)
        if not np.any(row_supported):
            continue
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=stable_seed(seed, heldout_group),
        )
        model.fit(embeddings[train_mask], targets[train_mask])
        predictions[row_supported] = model.predict(
            embeddings[row_supported]
        )
        supported[row_supported] = True
    return {
        "balanced_accuracy": (
            float(
                recall_score(
                    targets[supported],
                    predictions[supported].astype(str),
                    labels=np.unique(targets[supported]),
                    average="macro",
                    zero_division=0,
                )
            )
            if np.any(supported)
            else np.nan
        ),
        "n_supported": int(supported.sum()),
        "supported_fraction": float(supported.mean()),
        "n_groups": int(len(unique_groups)),
    }


@dataclass
class MahalanobisModel:
    classes: np.ndarray
    means: np.ndarray
    precisions: np.ndarray


def fit_class_mahalanobis(
    embeddings: np.ndarray, labels: np.ndarray
) -> MahalanobisModel:
    classes = np.asarray(sorted(np.unique(labels)), dtype=str)
    means: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    for label in classes:
        values = embeddings[labels == label]
        means.append(values.mean(axis=0))
        if len(values) > 1:
            precisions.append(LedoitWolf().fit(values).precision_)
        else:
            precisions.append(np.eye(embeddings.shape[1]))
    return MahalanobisModel(
        classes=classes,
        means=np.asarray(means),
        precisions=np.asarray(precisions),
    )


def mahalanobis_scores(
    model: MahalanobisModel, embeddings: np.ndarray
) -> np.ndarray:
    distances = []
    for mean, precision in zip(model.means, model.precisions):
        centered = embeddings - mean
        distances.append(
            np.einsum("ni,ij,nj->n", centered, precision, centered)
        )
    return np.min(np.column_stack(distances), axis=1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    return -(
        maximum[:, 0]
        + np.log(np.exp(logits - maximum).sum(axis=1))
    )


def integrated_gradients(
    model: ContrastiveClassifier,
    values: np.ndarray,
    target_indices: np.ndarray,
    device: torch.device,
    steps: int = 16,
    batch_size: int = 32,
) -> np.ndarray:
    """Integrated gradients from a zero baseline for classifier logits."""
    model.eval()
    outputs: list[np.ndarray] = []
    alphas = torch.linspace(
        1.0 / steps, 1.0, steps, device=device, dtype=torch.float32
    )
    for start in range(0, len(values), batch_size):
        batch_values = torch.from_numpy(
            np.asarray(values[start : start + batch_size], dtype=np.float32)
        ).to(device)
        batch_targets = torch.from_numpy(
            np.asarray(
                target_indices[start : start + batch_size], dtype=np.int64
            )
        ).to(device)
        gradient_sum = torch.zeros_like(batch_values)
        for alpha in alphas:
            scaled = (batch_values * alpha).detach().requires_grad_(True)
            _, logits = model(scaled.unsqueeze(1))
            selected = logits.gather(1, batch_targets[:, None]).sum()
            gradients = torch.autograd.grad(
                selected, scaled, retain_graph=False, create_graph=False
            )[0]
            gradient_sum += gradients.detach()
        attribution = batch_values * gradient_sum / float(steps)
        outputs.append(attribution.detach().cpu().numpy())
    return np.vstack(outputs)


def attribution_peak_stability(
    attributions: np.ndarray,
    manifest: pd.DataFrame,
    top_k: int = 30,
    tolerance_indices: int = 2,
) -> dict[str, float]:
    importance = np.abs(np.asarray(attributions, dtype=float))
    masters = manifest["master_sample_id"].astype(str).to_numpy()
    labels = manifest["target_analyte"].astype(str).to_numpy()
    instruments = manifest["instrument"].astype(str).to_numpy()
    peak_sets: list[set[int]] = []
    for row in importance:
        top = np.argpartition(row, -min(top_k, len(row)))[-top_k:]
        expanded: set[int] = set()
        for index in top:
            expanded.update(
                range(
                    max(0, int(index) - tolerance_indices),
                    min(len(row), int(index) + tolerance_indices + 1),
                )
            )
        peak_sets.append(expanded)
    same_master: list[float] = []
    same_analyte_different_master: list[float] = []
    for first in range(len(peak_sets)):
        for second in range(first + 1, len(peak_sets)):
            union = peak_sets[first] | peak_sets[second]
            overlap = (
                len(peak_sets[first] & peak_sets[second]) / max(len(union), 1)
            )
            if (
                masters[first] == masters[second]
                and instruments[first] != instruments[second]
            ):
                same_master.append(overlap)
            elif (
                labels[first] == labels[second]
                and masters[first] != masters[second]
            ):
                same_analyte_different_master.append(overlap)
    return {
        "same_master_cross_instrument_attribution_jaccard": (
            float(np.mean(same_master)) if same_master else np.nan
        ),
        "same_analyte_different_master_attribution_jaccard": (
            float(np.mean(same_analyte_different_master))
            if same_analyte_different_master
            else np.nan
        ),
        "n_same_master_attribution_pairs": len(same_master),
        "n_same_analyte_different_master_attribution_pairs": len(
            same_analyte_different_master
        ),
    }
