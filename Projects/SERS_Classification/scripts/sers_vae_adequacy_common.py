#!/usr/bin/env python3
"""Models and deterministic training utilities for SERS VAE adequacy v1."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sers_baseline_common as baseline
import sers_vae_common as standard


PROTOCOL_VERSION = "sers-vae-adequacy-v1"


@dataclass(frozen=True)
class AdequacyConfig:
    architecture: str = "base_maxpool"
    latent_dimension: int = 64
    reconstruction_loss: str = "spectral_composite"
    beta_target: float = 1.0
    optimizer_policy: str = "constant_lr"
    maximum_epoch: int = 300
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    batch_size: int = 64
    gradient_clip_norm: float = 5.0
    kl_normalization_divisor: int = 1401
    plateau_factor: float = 0.5
    plateau_patience: int = 20
    plateau_threshold: float = 1.0e-5
    minimum_learning_rate: float = 1.0e-5

    @property
    def identifier(self) -> str:
        beta = str(self.beta_target).replace(".", "p")
        return (
            f"{self.architecture}__z{self.latent_dimension}__"
            f"{self.reconstruction_loss}__beta{beta}__"
            f"{self.optimizer_policy}__e{self.maximum_epoch}"
        )

    def record(self) -> dict[str, Any]:
        return {"identifier": self.identifier, **asdict(self)}


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.convolution_1 = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2, dilation=1
        )
        self.convolution_2 = nn.Conv1d(
            channels, channels, kernel_size=5, padding=4, dilation=2
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.relu(self.convolution_1(values))
        values = self.convolution_2(values)
        return F.relu(values + residual)


class ResidualMultiscaleVAE(nn.Module):
    """Two-pool VAE with parameter-light dilated residual feature refinement."""

    def __init__(self, input_length: int, latent_dimension: int):
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dimension = int(latent_dimension)
        self.encoder_1 = nn.Conv1d(1, 8, kernel_size=7, padding=3)
        self.residual_1 = ResidualDilatedBlock(8)
        self.encoder_2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.residual_2 = ResidualDilatedBlock(16)
        with torch.no_grad():
            example = torch.zeros(1, 1, self.input_length)
            feature_length = self._features(example).shape[-1]
        self.feature_length = int(feature_length)
        flattened = 16 * self.feature_length
        self.mu_head = nn.Linear(flattened, self.latent_dimension)
        self.log_variance_head = nn.Linear(flattened, self.latent_dimension)
        self.expansion = nn.Linear(self.latent_dimension, flattened)
        self.decoder_residual_2 = ResidualDilatedBlock(16)
        self.decoder_2 = nn.Conv1d(16, 8, kernel_size=5, padding=2)
        self.decoder_residual_1 = ResidualDilatedBlock(8)
        self.decoder_1 = nn.Conv1d(8, 1, kernel_size=7, padding=3)

    def _features(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.encoder_1(values))
        values = self.residual_1(values)
        values = F.max_pool1d(values, 2)
        values = F.relu(self.encoder_2(values))
        values = self.residual_2(values)
        return F.max_pool1d(values, 2)

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(values).flatten(1)
        return (
            self.mu_head(features),
            torch.clamp(self.log_variance_head(features), min=-12.0, max=8.0),
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_variance: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        epsilon = torch.randn_like(mu) if epsilon is None else epsilon
        return mu + torch.exp(0.5 * log_variance) * epsilon

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.expansion(latent)).reshape(
            len(latent), 16, self.feature_length
        )
        values = self.decoder_residual_2(values)
        values = F.interpolate(values, scale_factor=2.0, mode="nearest")
        values = F.relu(self.decoder_2(values))
        values = self.decoder_residual_1(values)
        values = F.interpolate(values, size=self.input_length, mode="nearest")
        return torch.sigmoid(self.decoder_1(values))

    def forward(
        self,
        values: torch.Tensor,
        *,
        sample: bool = True,
        epsilon: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(values)
        latent = (
            self.reparameterize(mu, log_variance, epsilon)
            if sample
            else mu
        )
        return self.decode(latent), mu, log_variance, latent

    @property
    def decoder(self):
        return self.decode


class SinglePoolPeakVAE(nn.Module):
    """One-pool VAE retaining 700 spectral positions at constant dense width."""

    def __init__(self, input_length: int, latent_dimension: int):
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dimension = int(latent_dimension)
        self.encoder_1 = nn.Conv1d(1, 8, kernel_size=7, padding=3)
        self.encoder_2 = nn.Conv1d(8, 8, kernel_size=5, padding=2)
        with torch.no_grad():
            feature_length = self._features(
                torch.zeros(1, 1, self.input_length)
            ).shape[-1]
        self.feature_length = int(feature_length)
        flattened = 8 * self.feature_length
        self.mu_head = nn.Linear(flattened, self.latent_dimension)
        self.log_variance_head = nn.Linear(flattened, self.latent_dimension)
        self.expansion = nn.Linear(self.latent_dimension, flattened)
        self.decoder_2 = nn.Conv1d(8, 8, kernel_size=5, padding=2)
        self.decoder_1 = nn.Conv1d(8, 1, kernel_size=7, padding=3)

    def _features(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.encoder_1(values))
        values = F.max_pool1d(values, 2)
        return F.relu(self.encoder_2(values))

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(values).flatten(1)
        return (
            self.mu_head(features),
            torch.clamp(self.log_variance_head(features), min=-12.0, max=8.0),
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_variance: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        epsilon = torch.randn_like(mu) if epsilon is None else epsilon
        return mu + torch.exp(0.5 * log_variance) * epsilon

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.expansion(latent)).reshape(
            len(latent), 8, self.feature_length
        )
        values = F.relu(self.decoder_2(values))
        values = F.interpolate(values, size=self.input_length, mode="nearest")
        return torch.sigmoid(self.decoder_1(values))

    def forward(
        self,
        values: torch.Tensor,
        *,
        sample: bool = True,
        epsilon: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(values)
        latent = (
            self.reparameterize(mu, log_variance, epsilon)
            if sample
            else mu
        )
        return self.decode(latent), mu, log_variance, latent

    @property
    def decoder(self):
        return self.decode


def build_model(
    input_length: int, architecture: str, latent_dimension: int
) -> nn.Module:
    if architecture == "base_maxpool":
        return standard.ConvVariationalAutoencoder(
            input_length, (8, 16), latent_dimension
        )
    if architecture == "residual_multiscale":
        return ResidualMultiscaleVAE(input_length, latent_dimension)
    if architecture == "single_pool_peak":
        return SinglePoolPeakVAE(input_length, latent_dimension)
    raise ValueError(f"Unknown architecture: {architecture}")


def beta_for_epoch(epoch_one_based: int, beta_target: float) -> float:
    """Four fixed 25-epoch cycles followed by a fixed target beta."""
    if epoch_one_based > 100:
        return float(beta_target)
    position = float((epoch_one_based - 1) % 25) / 25.0
    return float(beta_target) * min(1.0, position / 0.5)


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, name: str
) -> torch.Tensor:
    if name == "spectral_composite":
        return baseline.reconstruction_loss(prediction, target, name)
    if name != "peak_multiscale":
        raise ValueError(f"Unknown adequacy loss: {name}")
    smooth = F.smooth_l1_loss(prediction, target)
    angle = baseline.spectral_angle_loss(prediction, target)
    first = F.l1_loss(
        torch.diff(prediction, dim=-1), torch.diff(target, dim=-1)
    )
    second = F.l1_loss(
        torch.diff(prediction, n=2, dim=-1),
        torch.diff(target, n=2, dim=-1),
    )
    pooled = []
    for width in (5, 15):
        pooled.append(
            F.l1_loss(
                F.avg_pool1d(
                    prediction, kernel_size=width, stride=1, padding=width // 2
                ),
                F.avg_pool1d(
                    target, kernel_size=width, stride=1, padding=width // 2
                ),
            )
        )
    return (
        smooth
        + 0.1 * angle
        + 0.25 * first
        + 0.1 * second
        + 0.1 * torch.stack(pooled).mean()
    )


def loss_components(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_variance: torch.Tensor,
    config: AdequacyConfig,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_term = reconstruction_loss(
        reconstruction, target, config.reconstruction_loss
    )
    kl_unnormalized = standard.kl_per_observation(
        mu, log_variance
    ).mean()
    # Preserve the exact arithmetic grouping used by standard-VAE v1.
    kl_normalized = kl_unnormalized / float(config.kl_normalization_divisor)
    total = reconstruction_term + float(beta) * kl_normalized
    return total, reconstruction_term, kl_unnormalized


def _validation_epsilons(
    row_count: int, latent_dimension: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        row_count, latent_dimension, generator=generator, dtype=torch.float32
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_registered_checkpoints(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    validation_values: np.ndarray,
    validation_uids: Sequence[str],
    config: AdequacyConfig,
    run_seed: int,
    metric_checkpoints: Sequence[int],
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, Any]],
]:
    """Train through every registered checkpoint without diagnostic stopping."""
    baseline.configure_determinism(run_seed)
    model = build_model(
        train_values.shape[1], config.architecture, config.latent_dimension
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            threshold_mode="abs",
            min_lr=config.minimum_learning_rate,
        )
        if config.optimizer_policy == "plateau_lr"
        else None
    )
    if config.optimizer_policy not in {
        "constant_lr",
        "plateau_lr",
        "step_lr_300",
    }:
        raise ValueError(f"Unknown optimizer policy: {config.optimizer_policy}")
    train_dataset = baseline.ReconstructionDataset(
        train_values,
        train_uids,
        "clean",
        run_seed,
        config.maximum_epoch,
    )
    validation_dataset = baseline.ReconstructionDataset(
        validation_values,
        validation_uids,
        "clean",
        baseline.stable_seed(run_seed, "validation"),
        config.maximum_epoch,
        fixed_epoch=config.maximum_epoch - 1,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(baseline.stable_seed(run_seed, "loader"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_epsilon = _validation_epsilons(
        len(validation_values),
        config.latent_dimension,
        baseline.stable_seed(run_seed, "validation_epsilon"),
    )
    wanted = set(int(value) for value in metric_checkpoints)
    states: dict[int, dict[str, torch.Tensor]] = {}
    optimizer_states: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    parameter_count = baseline.model_parameter_count(model)
    for epoch_zero in range(config.maximum_epoch):
        epoch = epoch_zero + 1
        if config.optimizer_policy == "step_lr_300" and epoch == 301:
            for group in optimizer.param_groups:
                group["lr"] = max(
                    float(group["lr"]) * 0.1, config.minimum_learning_rate
                )
        beta = beta_for_epoch(epoch, config.beta_target)
        train_dataset.set_epoch(epoch_zero)
        model.train()
        train_totals = np.zeros(3, dtype=float)
        train_count = 0
        gradient_norm_sum = 0.0
        gradient_steps = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, mu, log_variance, _ = model(inputs, sample=True)
            total, reconstruction_term, kl_unnormalized = loss_components(
                reconstruction, targets, mu, log_variance, config, beta
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            count = len(inputs)
            train_totals += np.asarray(
                [
                    float(total.detach().cpu()),
                    float(reconstruction_term.detach().cpu()),
                    float(kl_unnormalized.detach().cpu()),
                ]
            ) * count
            train_count += count
            gradient_norm_sum += float(gradient_norm.detach().cpu())
            gradient_steps += 1

        model.eval()
        validation_totals = np.zeros(3, dtype=float)
        validation_count = 0
        offset = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                count = len(inputs)
                epsilon = validation_epsilon[offset : offset + count].to(device)
                offset += count
                inputs = inputs.to(device)
                targets = targets.to(device)
                reconstruction, mu, log_variance, _ = model(
                    inputs, sample=True, epsilon=epsilon
                )
                total, reconstruction_term, kl_unnormalized = loss_components(
                    reconstruction,
                    targets,
                    mu,
                    log_variance,
                    config,
                    config.beta_target,
                )
                validation_totals += np.asarray(
                    [
                        float(total.cpu()),
                        float(reconstruction_term.cpu()),
                        float(kl_unnormalized.cpu()),
                    ]
                ) * count
                validation_count += count
        train_mean = train_totals / max(train_count, 1)
        validation_mean = validation_totals / max(validation_count, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        records.append(
            {
                "epoch": epoch,
                "beta": beta,
                "learning_rate": current_lr,
                "train_loss": train_mean[0],
                "train_reconstruction_loss": train_mean[1],
                "train_kl_unnormalized": train_mean[2],
                "train_kl_normalized": (
                    train_mean[2] / config.kl_normalization_divisor
                ),
                "validation_loss": validation_mean[0],
                "validation_reconstruction_loss": validation_mean[1],
                "validation_kl_unnormalized": validation_mean[2],
                "validation_kl_normalized": (
                    validation_mean[2] / config.kl_normalization_divisor
                ),
                "mean_unclipped_gradient_norm": (
                    gradient_norm_sum / max(gradient_steps, 1)
                ),
                "parameter_count": parameter_count,
            }
        )
        if scheduler is not None and epoch > 100:
            scheduler.step(float(validation_mean[0]))
        if epoch in wanted:
            states[epoch] = cpu_state_dict(model)
            if epoch in {100, config.maximum_epoch}:
                optimizer_states[epoch] = copy.deepcopy(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": (
                            scheduler.state_dict() if scheduler else None
                        ),
                    }
                )
    missing = wanted - set(states)
    if missing:
        raise RuntimeError(f"Missing registered checkpoints: {sorted(missing)}")
    return pd.DataFrame(records), states, optimizer_states


def outputs(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    reconstructions: list[np.ndarray] = []
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1).to(device)
            reconstruction, mu, log_variance, _ = model(batch, sample=False)
            reconstructions.append(reconstruction.squeeze(1).cpu().numpy())
            means.append(mu.cpu().numpy())
            log_variances.append(log_variance.cpu().numpy())
    return (
        np.vstack(reconstructions),
        np.vstack(means),
        np.vstack(log_variances),
    )


def sample_reconstruction_variability(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    seed: int,
    draws: int = 8,
) -> float:
    rng = np.random.default_rng(seed)
    all_standard_deviations: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 128):
            batch_values = np.asarray(
                values[start : start + 128], dtype=np.float32
            )
            batch = torch.from_numpy(batch_values).unsqueeze(1).to(device)
            mu, log_variance = model.encode(batch)
            decoded: list[np.ndarray] = []
            for _ in range(draws):
                epsilon = torch.from_numpy(
                    rng.standard_normal(mu.shape).astype(np.float32)
                ).to(device)
                latent = model.reparameterize(mu, log_variance, epsilon)
                reconstruction = (
                    model.decoder(latent)
                    if not callable(getattr(model, "decoder", None))
                    else model.decoder(latent)
                )
                decoded.append(reconstruction.squeeze(1).cpu().numpy())
            all_standard_deviations.append(
                np.std(np.stack(decoded), axis=0, ddof=0)
            )
    return float(np.mean(np.vstack(all_standard_deviations)))
