#!/usr/bin/env python3
"""Compact residual CNN models for the frozen SERS open-world experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

import sers_baseline_common as baseline
import sers_contrastive_common as contrastive


REGISTRY_VERSION = "sers-open-world-cnn-v1"


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unexpected CNN registry")
    if value.get("status_before_execution") != (
        "predeclared_after_controls_before_cnn_outcomes"
    ):
        raise ValueError("CNN registry was not predeclared")
    return value


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**31 - 1
    )


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int, stride: int):
        super().__init__()
        self.convolution_1 = nn.Conv1d(
            input_channels, channels, 7, stride=stride, padding=3, bias=False
        )
        self.normalization_1 = nn.GroupNorm(
            max(1, min(8, channels // 4)), channels
        )
        self.convolution_2 = nn.Conv1d(
            channels, channels, 7, padding=3, bias=False
        )
        self.normalization_2 = nn.GroupNorm(
            max(1, min(8, channels // 4)), channels
        )
        self.skip = (
            nn.Identity()
            if input_channels == channels and stride == 1
            else nn.Conv1d(
                input_channels, channels, 1, stride=stride, bias=False
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = F.gelu(self.normalization_1(self.convolution_1(values)))
        values = self.normalization_2(self.convolution_2(values))
        return F.gelu(values + residual)


class ResidualCNN(nn.Module):
    def __init__(self, class_count: int, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 24, 11, stride=2, padding=5, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(24, 24, 1),
            ResidualBlock(24, 48, 2),
            ResidualBlock(48, 96, 2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, class_count)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(self.pool(self.blocks(self.stem(values))))
        return embedding, self.classifier(embedding)


@dataclass(frozen=True)
class Variant:
    order: int
    name: str
    representation: str


def variants(registry: dict[str, Any]) -> list[Variant]:
    return [
        Variant(index, row["variant"], row["representation"])
        for index, row in enumerate(registry["candidate_variants"])
    ]


def edge_shift_batch(values: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(values)
    for index, shift_value in enumerate(shifts.tolist()):
        shift = int(shift_value)
        if shift == 0:
            output[index] = values[index]
        elif shift > 0:
            output[index, :, :shift] = values[index, :, :1]
            output[index, :, shift:] = values[index, :, :-shift]
        else:
            count = -shift
            output[index, :, -count:] = values[index, :, -1:]
            output[index, :, :-count] = values[index, :, count:]
    return output


def augment(
    values: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    count = len(values)
    scale = 0.95 + 0.10 * torch.rand(
        (count, 1, 1), generator=generator, device=values.device
    )
    noise = 0.01 * torch.randn(
        values.shape, generator=generator, device=values.device
    )
    shifts = torch.randint(
        -2, 3, (count,), generator=generator, device=values.device
    )
    return edge_shift_batch(values * scale + noise, shifts)


def conditional_coral(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for target in torch.unique(targets):
        mask = targets == target
        target_domains = torch.unique(domains[mask])
        if len(target_domains) < 2:
            continue
        parts = [embedding[mask & (domains == domain)] for domain in target_domains]
        parts = [part for part in parts if len(part) >= 2]
        for first, second in zip(parts[:-1], parts[1:]):
            first_centered = first - first.mean(0, keepdim=True)
            second_centered = second - second.mean(0, keepdim=True)
            first_cov = first_centered.T @ first_centered / max(len(first) - 1, 1)
            second_cov = second_centered.T @ second_centered / max(len(second) - 1, 1)
            losses.append(
                (first.mean(0) - second.mean(0)).pow(2).mean()
                + (first_cov - second_cov).pow(2).mean()
            )
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


def rbf_mmd(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([first, second], dim=0)
    squared = torch.cdist(combined, combined).pow(2)
    scale = torch.median(squared.detach()[squared.detach() > 0])
    scale = scale.clamp_min(1.0e-3)
    kernels = sum(
        torch.exp(-squared / (factor * scale))
        for factor in (0.5, 1.0, 2.0)
    )
    n_first = len(first)
    return (
        kernels[:n_first, :n_first].mean()
        + kernels[n_first:, n_first:].mean()
        - 2.0 * kernels[:n_first, n_first:].mean()
    )


def conditional_mmd(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for target in torch.unique(targets):
        mask = targets == target
        parts = [
            embedding[mask & (domains == domain)]
            for domain in torch.unique(domains[mask])
        ]
        parts = [part for part in parts if len(part) >= 2]
        for first, second in zip(parts[:-1], parts[1:]):
            losses.append(rbf_mmd(first, second))
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


def synthetic_outliers(
    values: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.randperm(
        len(values), generator=generator, device=values.device
    )
    # Rotate until most pairs differ; remaining same-class mixtures still
    # receive severe baseline/noise/spike corruption.
    for _ in range(3):
        same = targets[permutation] == targets
        if not same.any():
            break
        permutation[same] = permutation[same].roll(1)
    mixed = 0.5 * values + 0.5 * values[permutation]
    axis = torch.linspace(-1.0, 1.0, values.shape[-1], device=values.device)
    curvature = 0.15 * (axis.pow(2) - axis.pow(2).mean())
    noisy = mixed + curvature.reshape(1, 1, -1)
    noisy = noisy + 0.05 * torch.randn(
        noisy.shape, generator=generator, device=noisy.device
    )
    spike_position = torch.randint(
        0,
        values.shape[-1],
        (len(values),),
        generator=generator,
        device=values.device,
    )
    noisy[
        torch.arange(len(values), device=values.device),
        0,
        spike_position,
    ] += 0.5
    return noisy


@dataclass
class Trained:
    model: ResidualCNN
    classes: np.ndarray
    history: pd.DataFrame
    best_epoch: int
    state_sha256: str
    parameter_count: int


def train(
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_values: np.ndarray | None,
    validation_manifest: pd.DataFrame | None,
    variant: Variant,
    registry: dict[str, Any],
    seed: int,
    device: torch.device,
    fixed_epochs: int | None = None,
    augmentation_enabled: bool = True,
) -> Trained:
    baseline.configure_determinism(seed)
    classes = np.asarray(
        sorted(train_manifest["target_analyte"].astype(str).unique()), dtype=str
    )
    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.asarray(
        [
            class_index[label]
            for label in train_manifest["target_analyte"].astype(str)
        ],
        dtype=np.int64,
    )
    domains_text = (
        train_manifest["instrument"].astype(str)
        + "::"
        + train_manifest["sensor_family"].astype(str)
    )
    domain_names = sorted(domains_text.unique())
    domain_index = {label: index for index, label in enumerate(domain_names)}
    domains = np.asarray([domain_index[x] for x in domains_text], dtype=np.int64)
    values = torch.from_numpy(
        np.asarray(train_values, dtype=np.float32)
    ).unsqueeze(1).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    domain_tensor = torch.from_numpy(domains).to(device)
    model = ResidualCNN(len(classes), float(registry["architecture"]["dropout"])).to(device)
    config = registry["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum = int(fixed_epochs or config["maximum_epochs"])
    minimum = int(config["minimum_epochs"])
    patience = int(config["early_stopping_patience"])
    batch_size = int(config["batch_size"])
    group_weights = torch.ones(
        len(domain_names), device=device
    ) / max(len(domain_names), 1)
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-np.inf, -np.inf)
    best_epoch = maximum
    stale = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(maximum):
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(seed, epoch, "cnn"))
        order = torch.randperm(len(values), generator=generator, device=device)
        model.train()
        loss_total = 0.0
        count_total = 0
        for start in range(0, len(values), batch_size):
            index = order[start : start + batch_size]
            batch = (
                augment(values[index], generator)
                if augmentation_enabled
                else values[index]
            )
            batch_target = target_tensor[index]
            batch_domain = domain_tensor[index]
            embedding, logits = model(batch)
            per_row = F.cross_entropy(logits, batch_target, reduction="none")
            loss = per_row.mean()
            if variant.name == "supervised_contrastive_joint":
                normalized = F.normalize(embedding, dim=1)
                loss = loss + 0.2 * contrastive.supervised_contrastive_loss(
                    normalized, batch_target, 0.1
                )
            elif variant.name == "synthetic_objectosphere":
                outlier = synthetic_outliers(
                    values[index], batch_target, generator
                )
                out_embedding, out_logits = model(outlier)
                uniform = -F.log_softmax(out_logits, dim=1).mean()
                known_norm = embedding.norm(dim=1)
                out_norm = out_embedding.norm(dim=1)
                objectosphere = (
                    F.relu(1.0 - known_norm).pow(2).mean()
                    + F.relu(out_norm - 0.2).pow(2).mean()
                )
                loss = loss + 0.1 * uniform + 0.05 * objectosphere
            elif variant.name == "conditional_coral":
                loss = loss + 0.1 * conditional_coral(
                    embedding, batch_target, batch_domain
                )
            elif variant.name == "conditional_mmd":
                loss = loss + 0.1 * conditional_mmd(
                    embedding, batch_target, batch_domain
                )
            elif variant.name == "group_dro":
                group_losses = torch.zeros_like(group_weights)
                present = torch.zeros_like(group_weights)
                for domain in torch.unique(batch_domain):
                    mask = batch_domain == domain
                    group_losses[domain] = per_row[mask].mean()
                    present[domain] = 1.0
                with torch.no_grad():
                    group_weights *= torch.exp(
                        0.05 * group_losses.detach() * present
                    )
                    group_weights /= group_weights.sum()
                active = present > 0
                normalized_weight = group_weights[active]
                normalized_weight /= normalized_weight.sum()
                loss = torch.sum(
                    normalized_weight * group_losses[active]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(index)
            count_total += len(index)
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": loss_total / max(count_total, 1),
        }
        if (
            fixed_epochs is None
            and validation_values is not None
            and validation_manifest is not None
            and epoch + 1 >= minimum
            and (epoch + 1) % 5 == 0
        ):
            _, logits = deterministic_outputs(
                model, validation_values, device
            )
            prediction = classes[np.argmax(logits, axis=1)]
            truth = validation_manifest["target_analyte"].astype(str).to_numpy()
            ba = balanced_accuracy_score(truth, prediction)
            macro = f1_score(truth, prediction, average="macro")
            row.update(
                validation_balanced_accuracy=ba,
                validation_macro_f1=macro,
            )
            if (ba, macro) > best_key:
                best_key = (ba, macro)
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
        rows.append(row)
        if fixed_epochs is None and best_state is not None and stale >= patience:
            break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return Trained(
        model=model,
        classes=classes,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        state_sha256=baseline.state_dict_sha256(state),
        parameter_count=baseline.model_parameter_count(model),
    )


def deterministic_outputs(
    model: ResidualCNN,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings, logits = [], []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1).to(device)
            embedding, score = model(batch)
            embeddings.append(embedding.cpu().numpy())
            logits.append(score.cpu().numpy())
    return np.vstack(embeddings), np.vstack(logits)


def mc_dropout_outputs(
    model: ResidualCNN,
    values: np.ndarray,
    device: torch.device,
    passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    return mc_dropout_prefix_outputs(model, values, device, (passes,))[passes]


def mc_dropout_prefix_outputs(
    model: ResidualCNN,
    values: np.ndarray,
    device: torch.device,
    pass_counts: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    requested = tuple(sorted(set(int(value) for value in pass_counts)))
    if not requested or requested[0] < 2:
        raise ValueError("MC-dropout pass counts must be at least two")
    probabilities = []
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    with torch.no_grad():
        batch = torch.from_numpy(
            np.asarray(values, dtype=np.float32)
        ).unsqueeze(1).to(device)
        for _ in range(requested[-1]):
            _, logits = model(batch)
            probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
    stacked = np.stack(probabilities)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for count in requested:
        prefix = stacked[:count]
        mean_probability = prefix.mean(axis=0)
        predictive = -np.sum(
            np.clip(mean_probability, 1.0e-12, 1.0)
            * np.log(np.clip(mean_probability, 1.0e-12, 1.0)),
            axis=1,
        )
        expected = np.mean(
            -np.sum(
                np.clip(prefix, 1.0e-12, 1.0)
                * np.log(np.clip(prefix, 1.0e-12, 1.0)),
                axis=2,
            ),
            axis=0,
        )
        result[count] = (mean_probability, predictive - expected)
    model.eval()
    return result
