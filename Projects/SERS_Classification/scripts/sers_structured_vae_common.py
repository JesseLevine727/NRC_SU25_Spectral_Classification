#!/usr/bin/env python3
"""Deterministic partitioned VAE models and training for NATO SERS."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import sers_baseline_common as baseline
import sers_vae_adequacy_common as adequacy
import sers_vae_common as standard


PROTOCOL_VERSION = "sers-structured-vae-v1"


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class StructuredConfig:
    chemical_dimension: int = 48
    nuisance_dimension: int = 16
    chemical_supervision_weight: float = 0.0
    instrument_supervision_weight: float = 0.0
    sensor_supervision_weight: float = 0.0
    condition_decoder: bool = False
    instrument_adversary_weight: float = 0.0
    sensor_adversary_weight: float = 0.0
    same_master_consistency_weight: float = 0.0
    cross_reconstruction_weight: float = 0.0
    dependence_weight: float = 0.0
    beta_target: float = 0.25
    maximum_epoch: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    batch_size: int = 64
    gradient_clip_norm: float = 5.0
    kl_normalization_divisor: int = 1401

    @property
    def total_latent_dimension(self) -> int:
        return self.chemical_dimension + self.nuisance_dimension

    @property
    def identifier(self) -> str:
        fields = [
            f"zc{self.chemical_dimension}",
            f"zn{self.nuisance_dimension}",
            f"chem{token(self.chemical_supervision_weight)}",
            f"ni{token(self.instrument_supervision_weight)}",
            f"ns{token(self.sensor_supervision_weight)}",
            f"cond{int(self.condition_decoder)}",
            f"ai{token(self.instrument_adversary_weight)}",
            f"as{token(self.sensor_adversary_weight)}",
            f"pair{token(self.same_master_consistency_weight)}",
            f"xrec{token(self.cross_reconstruction_weight)}",
            f"dep{token(self.dependence_weight)}",
            f"e{self.maximum_epoch}",
        ]
        return "__".join(fields)

    def record(self) -> dict[str, Any]:
        return {"identifier": self.identifier, **asdict(self)}


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any, values: torch.Tensor, strength: float
    ) -> torch.Tensor:
        context.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(
        context: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -context.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReversalFunction.apply(values, float(strength))


class ConditionedConvDecoder(nn.Module):
    """Frozen decoder topology with a fixed-size optional domain condition."""

    def __init__(
        self,
        output_length: int,
        feature_length: int,
        latent_dimension: int,
        instrument_count: int,
        sensor_count: int,
    ):
        super().__init__()
        self.output_length = int(output_length)
        self.feature_length = int(feature_length)
        self.latent_dimension = int(latent_dimension)
        self.instrument_count = int(instrument_count)
        self.sensor_count = int(sensor_count)
        decoder_input = (
            self.latent_dimension
            + self.instrument_count
            + self.sensor_count
        )
        self.expansion = nn.Sequential(
            nn.Linear(decoder_input, 16 * self.feature_length),
            nn.ReLU(),
        )
        self.convolution_2 = nn.Conv1d(16, 8, kernel_size=5, padding=2)
        self.convolution_1 = nn.Conv1d(8, 1, kernel_size=7, padding=3)

    @staticmethod
    def _one_hot(
        indices: torch.Tensor, classes: int, enabled: bool
    ) -> torch.Tensor:
        result = torch.zeros(
            len(indices), classes, dtype=torch.float32, device=indices.device
        )
        if not enabled:
            return result
        valid = (indices >= 0) & (indices < classes)
        if valid.any():
            result[valid] = F.one_hot(
                indices[valid].long(), num_classes=classes
            ).float()
        return result

    def forward(
        self,
        latent: torch.Tensor,
        instrument_index: torch.Tensor,
        sensor_index: torch.Tensor,
        condition_enabled: bool,
    ) -> torch.Tensor:
        condition = torch.cat(
            [
                self._one_hot(
                    instrument_index,
                    self.instrument_count,
                    condition_enabled,
                ),
                self._one_hot(
                    sensor_index, self.sensor_count, condition_enabled
                ),
            ],
            dim=1,
        )
        values = self.expansion(torch.cat([latent, condition], dim=1)).reshape(
            len(latent), 16, self.feature_length
        )
        values = F.interpolate(values, scale_factor=2.0, mode="nearest")
        values = F.relu(self.convolution_2(values))
        values = F.interpolate(
            values, size=self.output_length, mode="nearest"
        )
        return torch.sigmoid(self.convolution_1(values))


class PartitionedSERSVAE(nn.Module):
    """Capacity-frozen convolutional VAE with chemical/nuisance partitions."""

    def __init__(
        self,
        input_length: int,
        chemical_dimension: int,
        nuisance_dimension: int,
        target_count: int,
        instrument_count: int,
        sensor_count: int,
        condition_decoder: bool,
    ):
        super().__init__()
        if chemical_dimension + nuisance_dimension != 64:
            raise ValueError("Structured protocol requires total z64")
        self.input_length = int(input_length)
        self.chemical_dimension = int(chemical_dimension)
        self.nuisance_dimension = int(nuisance_dimension)
        self.latent_dimension = 64
        self.target_count = int(target_count)
        self.instrument_count = int(instrument_count)
        self.sensor_count = int(sensor_count)
        self.condition_decoder = bool(condition_decoder)
        encoder = baseline.ConvEncoder(
            self.input_length, (8, 16), 64, normalize_output=False
        )
        self.features = encoder.features
        self.feature_length = encoder.feature_length
        flattened = 16 * self.feature_length
        self.chemical_mu_head = nn.Linear(
            flattened, self.chemical_dimension
        )
        self.chemical_log_variance_head = nn.Linear(
            flattened, self.chemical_dimension
        )
        self.nuisance_mu_head = nn.Linear(
            flattened, self.nuisance_dimension
        )
        self.nuisance_log_variance_head = nn.Linear(
            flattened, self.nuisance_dimension
        )
        self.decoder = ConditionedConvDecoder(
            self.input_length,
            self.feature_length,
            64,
            self.instrument_count,
            self.sensor_count,
        )
        self.chemical_classifier = nn.Linear(
            self.chemical_dimension, self.target_count
        )
        self.nuisance_instrument_classifier = nn.Linear(
            self.nuisance_dimension, self.instrument_count
        )
        self.nuisance_sensor_classifier = nn.Linear(
            self.nuisance_dimension, self.sensor_count
        )
        adversary_width = 64
        self.chemical_instrument_adversary = nn.Sequential(
            nn.Linear(
                self.chemical_dimension + self.target_count,
                adversary_width,
            ),
            nn.ReLU(),
            nn.Linear(adversary_width, self.instrument_count),
        )
        self.chemical_sensor_adversary = nn.Sequential(
            nn.Linear(
                self.chemical_dimension + self.target_count,
                adversary_width,
            ),
            nn.ReLU(),
            nn.Linear(adversary_width, self.sensor_count),
        )

    def encode_parts(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.features(values), start_dim=1)
        return (
            self.chemical_mu_head(features),
            torch.clamp(
                self.chemical_log_variance_head(features),
                min=-12.0,
                max=8.0,
            ),
            self.nuisance_mu_head(features),
            torch.clamp(
                self.nuisance_log_variance_head(features),
                min=-12.0,
                max=8.0,
            ),
        )

    def encode(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chem_mu, chem_logvar, nuisance_mu, nuisance_logvar = (
            self.encode_parts(values)
        )
        return (
            torch.cat([chem_mu, nuisance_mu], dim=1),
            torch.cat([chem_logvar, nuisance_logvar], dim=1),
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_variance: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        epsilon = torch.randn_like(mu) if epsilon is None else epsilon
        return mu + torch.exp(0.5 * log_variance) * epsilon

    def decode_parts(
        self,
        chemical: torch.Tensor,
        nuisance: torch.Tensor,
        instrument_index: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            torch.cat([chemical, nuisance], dim=1),
            instrument_index,
            sensor_index,
            self.condition_decoder,
        )

    def adversary_logits(
        self,
        chemical: torch.Tensor,
        target_index: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = F.one_hot(
            target_index.long(), num_classes=self.target_count
        ).float()
        reversed_chemical = gradient_reverse(chemical, strength)
        features = torch.cat([reversed_chemical, target], dim=1)
        return (
            self.chemical_instrument_adversary(features),
            self.chemical_sensor_adversary(features),
        )

    def forward(
        self,
        values: torch.Tensor,
        instrument_index: torch.Tensor,
        sensor_index: torch.Tensor,
        *,
        sample: bool = True,
        epsilon: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        chem_mu, chem_logvar, nuisance_mu, nuisance_logvar = (
            self.encode_parts(values)
        )
        union_mu = torch.cat([chem_mu, nuisance_mu], dim=1)
        union_logvar = torch.cat([chem_logvar, nuisance_logvar], dim=1)
        union = (
            self.reparameterize(union_mu, union_logvar, epsilon)
            if sample
            else union_mu
        )
        chemical = union[:, : self.chemical_dimension]
        nuisance = union[:, self.chemical_dimension :]
        return {
            "reconstruction": self.decode_parts(
                chemical,
                nuisance,
                instrument_index,
                sensor_index,
            ),
            "chemical_mu": chem_mu,
            "chemical_log_variance": chem_logvar,
            "nuisance_mu": nuisance_mu,
            "nuisance_log_variance": nuisance_logvar,
            "chemical": chemical,
            "nuisance": nuisance,
            "union_mu": union_mu,
            "union_log_variance": union_logvar,
            "union": union,
        }


def class_balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    lookup = {
        int(label): len(labels) / (len(unique) * int(count))
        for label, count in zip(unique, counts)
    }
    weights = np.asarray([lookup[int(label)] for label in labels], dtype=np.float32)
    return weights / max(float(weights.mean()), 1.0e-12)


def cell_balanced_weights(
    targets: np.ndarray, domains: np.ndarray
) -> np.ndarray:
    targets = np.asarray(targets, dtype=int)
    domains = np.asarray(domains, dtype=int)
    cells = np.stack([targets, domains], axis=1)
    unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    weights = len(cells) / (len(unique) * counts[inverse])
    weights = weights.astype(np.float32)
    return weights / max(float(weights.mean()), 1.0e-12)


class StructuredDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        observation_uids: Sequence[str],
        master_ids: Sequence[str],
        targets: np.ndarray,
        instruments: np.ndarray,
        sensors: np.ndarray,
        run_seed: int,
    ):
        self.values = np.asarray(values, dtype=np.float32)
        self.observation_uids = np.asarray(observation_uids, dtype=str)
        self.master_ids = np.asarray(master_ids, dtype=str)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.instruments = np.asarray(instruments, dtype=np.int64)
        self.sensors = np.asarray(sensors, dtype=np.int64)
        self.run_seed = int(run_seed)
        self.epoch = 0
        self.chemical_weights = class_balanced_weights(self.targets)
        self.instrument_weights = cell_balanced_weights(
            self.targets, self.instruments
        )
        self.sensor_weights = cell_balanced_weights(
            self.targets, self.sensors
        )
        self.partner_candidates: list[np.ndarray] = []
        for index in range(len(self.values)):
            selection = np.flatnonzero(
                (self.master_ids == self.master_ids[index])
                & (self.instruments != self.instruments[index])
            )
            self.partner_candidates.append(selection)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def partner_index(self, index: int) -> tuple[int, bool]:
        candidates = self.partner_candidates[index]
        if len(candidates) == 0:
            return index, False
        seed = baseline.stable_seed(
            PROTOCOL_VERSION,
            self.run_seed,
            self.epoch,
            self.observation_uids[index],
            "real_partner",
        )
        return int(candidates[seed % len(candidates)]), True

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        partner, valid = self.partner_index(index)
        return (
            torch.from_numpy(self.values[index]).unsqueeze(0),
            torch.tensor(self.targets[index], dtype=torch.long),
            torch.tensor(self.instruments[index], dtype=torch.long),
            torch.tensor(self.sensors[index], dtype=torch.long),
            torch.tensor(self.chemical_weights[index], dtype=torch.float32),
            torch.tensor(self.instrument_weights[index], dtype=torch.float32),
            torch.tensor(self.sensor_weights[index], dtype=torch.float32),
            torch.from_numpy(self.values[partner]).unsqueeze(0),
            torch.tensor(self.targets[partner], dtype=torch.long),
            torch.tensor(self.instruments[partner], dtype=torch.long),
            torch.tensor(self.sensors[partner], dtype=torch.long),
            torch.tensor(valid, dtype=torch.bool),
        )


def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    values = F.cross_entropy(logits, labels, reduction="none")
    return torch.mean(values * weights)


def cross_covariance_penalty(
    chemical: torch.Tensor, nuisance: torch.Tensor
) -> torch.Tensor:
    if len(chemical) <= 1:
        return chemical.new_tensor(0.0)
    chemical = chemical - chemical.mean(dim=0, keepdim=True)
    nuisance = nuisance - nuisance.mean(dim=0, keepdim=True)
    chemical = chemical / torch.clamp(
        chemical.std(dim=0, unbiased=False, keepdim=True), min=1.0e-4
    )
    nuisance = nuisance / torch.clamp(
        nuisance.std(dim=0, unbiased=False, keepdim=True), min=1.0e-4
    )
    covariance = chemical.T @ nuisance / float(len(chemical))
    return torch.mean(covariance.pow(2))


def adversary_strength(epoch_one_based: int) -> float:
    if epoch_one_based <= 100:
        return 0.0
    if epoch_one_based >= 200:
        return 1.0
    return float(epoch_one_based - 100) / 100.0


def objective(
    model: PartitionedSERSVAE,
    batch: tuple[torch.Tensor, ...],
    config: StructuredConfig,
    beta: float,
    adversary_scale: float,
    device: torch.device,
    sample: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    (
        values,
        targets,
        instruments,
        sensors,
        chemical_weights,
        instrument_weights,
        sensor_weights,
        partner_values,
        _partner_targets,
        partner_instruments,
        partner_sensors,
        pair_valid,
    ) = (item.to(device) for item in batch)
    result = model(
        values,
        instruments,
        sensors,
        sample=sample,
    )
    reconstruction = adequacy.reconstruction_loss(
        result["reconstruction"], values, "spectral_composite"
    )
    chemical_kl = standard.kl_per_observation(
        result["chemical_mu"], result["chemical_log_variance"]
    ).mean()
    nuisance_kl = standard.kl_per_observation(
        result["nuisance_mu"], result["nuisance_log_variance"]
    ).mean()
    kl = (chemical_kl + nuisance_kl) / float(
        config.kl_normalization_divisor
    )
    chemical_logits = model.chemical_classifier(result["chemical"])
    instrument_logits = model.nuisance_instrument_classifier(
        result["nuisance"]
    )
    sensor_logits = model.nuisance_sensor_classifier(result["nuisance"])
    adversarial_instrument_logits, adversarial_sensor_logits = (
        model.adversary_logits(
            result["chemical"], targets, adversary_scale
        )
    )
    chemical_loss = weighted_cross_entropy(
        chemical_logits, targets, chemical_weights
    )
    instrument_loss = weighted_cross_entropy(
        instrument_logits, instruments, instrument_weights
    )
    sensor_loss = weighted_cross_entropy(
        sensor_logits, sensors, sensor_weights
    )
    adversarial_instrument_loss = weighted_cross_entropy(
        adversarial_instrument_logits, instruments, instrument_weights
    )
    adversarial_sensor_loss = weighted_cross_entropy(
        adversarial_sensor_logits, sensors, sensor_weights
    )
    dependence = cross_covariance_penalty(
        result["chemical_mu"], result["nuisance_mu"]
    )
    pair_consistency = values.new_tensor(0.0)
    cross_reconstruction = values.new_tensor(0.0)
    if (
        config.same_master_consistency_weight > 0
        or config.cross_reconstruction_weight > 0
    ):
        partner = model(
            partner_values,
            partner_instruments,
            partner_sensors,
            sample=sample,
        )
        if pair_valid.any():
            pair_consistency = (
                1.0
                - F.cosine_similarity(
                    result["chemical_mu"][pair_valid],
                    partner["chemical_mu"][pair_valid],
                    dim=1,
                )
            ).mean()
            swapped = model.decode_parts(
                result["chemical_mu"][pair_valid],
                partner["nuisance_mu"][pair_valid],
                partner_instruments[pair_valid],
                partner_sensors[pair_valid],
            )
            cross_reconstruction = adequacy.reconstruction_loss(
                swapped,
                partner_values[pair_valid],
                "spectral_composite",
            )
    total = (
        reconstruction
        + float(beta) * kl
        + config.chemical_supervision_weight * chemical_loss
        + config.instrument_supervision_weight * instrument_loss
        + config.sensor_supervision_weight * sensor_loss
        + config.instrument_adversary_weight
        * adversarial_instrument_loss
        + config.sensor_adversary_weight * adversarial_sensor_loss
        + config.same_master_consistency_weight * pair_consistency
        + config.cross_reconstruction_weight * cross_reconstruction
        + config.dependence_weight * dependence
    )
    components = {
        "total": total,
        "reconstruction": reconstruction,
        "chemical_kl_unnormalized": chemical_kl,
        "nuisance_kl_unnormalized": nuisance_kl,
        "chemical_supervision": chemical_loss,
        "instrument_supervision": instrument_loss,
        "sensor_supervision": sensor_loss,
        "instrument_adversary": adversarial_instrument_loss,
        "sensor_adversary": adversarial_sensor_loss,
        "pair_consistency": pair_consistency,
        "cross_reconstruction": cross_reconstruction,
        "dependence": dependence,
    }
    return total, components


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_registered_checkpoints(
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_values: np.ndarray,
    validation_manifest: pd.DataFrame,
    target_mapping: dict[str, int],
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
    config: StructuredConfig,
    run_seed: int,
    checkpoints: Sequence[int],
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, Any]],
]:
    baseline.configure_determinism(run_seed)
    model = PartitionedSERSVAE(
        train_values.shape[1],
        config.chemical_dimension,
        config.nuisance_dimension,
        len(target_mapping),
        len(instrument_mapping),
        len(sensor_mapping),
        config.condition_decoder,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def make_dataset(
        values: np.ndarray, manifest: pd.DataFrame, seed: int
    ) -> StructuredDataset:
        return StructuredDataset(
            values,
            manifest["observation_uid"].astype(str),
            manifest["master_sample_id"].astype(str),
            manifest["target_analyte"].astype(str).map(target_mapping).to_numpy(),
            manifest["instrument"].astype(str).map(instrument_mapping).to_numpy(),
            manifest["sensor_family"].astype(str).map(sensor_mapping).to_numpy(),
            seed,
        )

    train_dataset = make_dataset(train_values, train_manifest, run_seed)
    validation_dataset = make_dataset(
        validation_values,
        validation_manifest,
        baseline.stable_seed(run_seed, "validation"),
    )
    validation_dataset.set_epoch(config.maximum_epoch - 1)
    generator = torch.Generator()
    generator.manual_seed(baseline.stable_seed(run_seed, "loader"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    names = [
        "total",
        "reconstruction",
        "chemical_kl_unnormalized",
        "nuisance_kl_unnormalized",
        "chemical_supervision",
        "instrument_supervision",
        "sensor_supervision",
        "instrument_adversary",
        "sensor_adversary",
        "pair_consistency",
        "cross_reconstruction",
        "dependence",
    ]
    wanted = {int(value) for value in checkpoints}
    states: dict[int, dict[str, torch.Tensor]] = {}
    optimizer_states: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    parameter_count = baseline.model_parameter_count(model)
    for epoch_zero in range(config.maximum_epoch):
        epoch = epoch_zero + 1
        train_dataset.set_epoch(epoch_zero)
        beta = adequacy.beta_for_epoch(epoch, config.beta_target)
        adversary_scale = adversary_strength(epoch)
        model.train()
        train_sums = {name: 0.0 for name in names}
        train_count = 0
        gradient_sum = 0.0
        gradient_steps = 0
        for batch in train_loader:
            total, components = objective(
                model,
                batch,
                config,
                beta,
                adversary_scale,
                device,
                sample=True,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            count = len(batch[0])
            for name in names:
                train_sums[name] += (
                    float(components[name].detach().cpu()) * count
                )
            train_count += count
            gradient_sum += float(gradient.detach().cpu())
            gradient_steps += 1
        model.eval()
        validation_sums = {name: 0.0 for name in names}
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                _, components = objective(
                    model,
                    batch,
                    config,
                    config.beta_target,
                    1.0,
                    device,
                    sample=False,
                )
                count = len(batch[0])
                for name in names:
                    validation_sums[name] += (
                        float(components[name].cpu()) * count
                    )
                validation_count += count
        record: dict[str, Any] = {
            "epoch": epoch,
            "beta": beta,
            "adversary_scale": adversary_scale,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": gradient_sum / max(gradient_steps, 1),
            "parameter_count": parameter_count,
        }
        for name in names:
            record[f"train_{name}"] = train_sums[name] / max(
                train_count, 1
            )
            record[f"validation_{name}"] = validation_sums[name] / max(
                validation_count, 1
            )
        records.append(record)
        if epoch in wanted:
            states[epoch] = cpu_state_dict(model)
            if epoch in {100, config.maximum_epoch}:
                optimizer_states[epoch] = copy.deepcopy(
                    {"optimizer": optimizer.state_dict()}
                )
    missing = wanted - set(states)
    if missing:
        raise RuntimeError(f"Missing checkpoints: {sorted(missing)}")
    return pd.DataFrame(records), states, optimizer_states


def build_model_from_state(
    input_length: int,
    config: StructuredConfig,
    target_count: int,
    instrument_count: int,
    sensor_count: int,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> PartitionedSERSVAE:
    model = PartitionedSERSVAE(
        input_length,
        config.chemical_dimension,
        config.nuisance_dimension,
        target_count,
        instrument_count,
        sensor_count,
        config.condition_decoder,
    )
    model.load_state_dict(state)
    return model.to(device)


def outputs(
    model: PartitionedSERSVAE,
    values: np.ndarray,
    instrument_indices: np.ndarray,
    sensor_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "reconstruction": [],
        "chemical_mu": [],
        "chemical_log_variance": [],
        "nuisance_mu": [],
        "nuisance_log_variance": [],
        "union_mu": [],
        "union_log_variance": [],
    }
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = start + batch_size
            batch = torch.from_numpy(
                np.asarray(values[start:stop], dtype=np.float32)
            ).unsqueeze(1).to(device)
            instrument = torch.from_numpy(
                np.array(
                    instrument_indices[start:stop],
                    dtype=np.int64,
                    copy=True,
                )
            ).to(device)
            sensor = torch.from_numpy(
                np.array(
                    sensor_indices[start:stop],
                    dtype=np.int64,
                    copy=True,
                )
            ).to(device)
            result = model(
                batch, instrument, sensor, sample=False
            )
            for name in collected:
                values_out = result[name]
                if name == "reconstruction":
                    values_out = values_out.squeeze(1)
                collected[name].append(values_out.cpu().numpy())
    return {name: np.vstack(parts) for name, parts in collected.items()}


def partition_dependence(
    chemical: np.ndarray, nuisance: np.ndarray, ridge: float = 1.0e-3
) -> dict[str, float]:
    chemical = np.asarray(chemical, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    chemical -= chemical.mean(axis=0, keepdims=True)
    nuisance -= nuisance.mean(axis=0, keepdims=True)
    chemical /= np.maximum(chemical.std(axis=0, keepdims=True), 1.0e-6)
    nuisance /= np.maximum(nuisance.std(axis=0, keepdims=True), 1.0e-6)
    covariance = chemical.T @ nuisance / max(len(chemical), 1)
    frobenius_mean_square = float(np.mean(np.square(covariance)))
    chemical_cov = chemical.T @ chemical / max(len(chemical), 1)
    nuisance_cov = nuisance.T @ nuisance / max(len(nuisance), 1)

    def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(
            matrix + ridge * np.eye(matrix.shape[0])
        )
        return (
            vectors
            @ np.diag(1.0 / np.sqrt(np.maximum(values, ridge)))
            @ vectors.T
        )

    whitened = (
        inverse_sqrt(chemical_cov)
        @ covariance
        @ inverse_sqrt(nuisance_cov)
    )
    maximum_canonical = float(
        np.clip(np.linalg.svd(whitened, compute_uv=False)[0], 0.0, 1.0)
    )
    return {
        "partition_cross_covariance_mean_square": frobenius_mean_square,
        "partition_maximum_canonical_correlation": maximum_canonical,
    }
