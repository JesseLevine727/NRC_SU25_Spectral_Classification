#!/usr/bin/env python3
"""Shared deterministic standard-VAE implementation for SERS protocol v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import sers_baseline_common as baseline


PROTOCOL_VERSION = "sers-standard-vae-v1"


@dataclass(frozen=True)
class VAETrainingConfig:
    channels: tuple[int, int] = (8, 16)
    latent_dimension: int = 64
    reconstruction_loss: str = "spectral_composite"
    kl_schedule: str = "constant"
    beta: float = 1.0
    kl_normalization_divisor: int = 1401
    optimizer: str = "Adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    batch_size: int = 64
    maximum_epochs: int = 100
    minimum_epochs: int = 30
    early_stopping_patience: int = 15
    early_stopping_minimum_delta: float = 1.0e-5
    gradient_clip_norm: float = 5.0

    @property
    def identifier(self) -> str:
        channels = "x".join(str(value) for value in self.channels)
        return (
            f"c{channels}_z{self.latent_dimension}_"
            f"{self.reconstruction_loss}_beta1_{self.kl_schedule}"
        )


class ConvVariationalAutoencoder(nn.Module):
    """Capacity-matched convolutional VAE with a diagonal Gaussian posterior."""

    def __init__(
        self,
        input_length: int,
        channels: Sequence[int],
        latent_dimension: int,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError("Exactly two convolutional channel widths required")
        self.input_length = int(input_length)
        self.channels = tuple(int(value) for value in channels)
        self.latent_dimension = int(latent_dimension)
        feature_encoder = baseline.ConvEncoder(
            self.input_length,
            self.channels,
            self.latent_dimension,
            normalize_output=False,
        )
        self.features = feature_encoder.features
        self.feature_length = feature_encoder.feature_length
        flattened = self.channels[1] * self.feature_length
        self.mu_head = nn.Linear(flattened, self.latent_dimension)
        self.log_variance_head = nn.Linear(flattened, self.latent_dimension)
        self.decoder = baseline.ConvDecoder(
            self.input_length,
            self.channels,
            self.feature_length,
            self.latent_dimension,
        )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.features(values), start_dim=1)
        mu = self.mu_head(features)
        log_variance = torch.clamp(
            self.log_variance_head(features), min=-12.0, max=8.0
        )
        return mu, log_variance

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_variance: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * log_variance) * epsilon

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
        return self.decoder(latent), mu, log_variance, latent


def beta_for_epoch(schedule: str, epoch_zero_based: int, maximum_epochs: int) -> float:
    if schedule == "constant":
        return 1.0
    if schedule == "linear_warmup_20":
        return min(1.0, float(epoch_zero_based + 1) / 20.0)
    if schedule == "cyclical_4":
        cycle_length = maximum_epochs / 4.0
        position = (float(epoch_zero_based) % cycle_length) / cycle_length
        return min(1.0, position / 0.5)
    raise ValueError(f"Unknown KL schedule: {schedule}")


def kl_per_observation(
    mu: torch.Tensor, log_variance: torch.Tensor
) -> torch.Tensor:
    return -0.5 * torch.sum(
        1.0 + log_variance - mu.pow(2) - log_variance.exp(),
        dim=1,
    )


def loss_components(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_variance: torch.Tensor,
    config: VAETrainingConfig,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_term = baseline.reconstruction_loss(
        reconstruction,
        target,
        config.reconstruction_loss,
    )
    kl_unnormalized = kl_per_observation(mu, log_variance).mean()
    kl_normalized = kl_unnormalized / float(config.kl_normalization_divisor)
    total = reconstruction_term + float(beta) * kl_normalized
    return total, reconstruction_term, kl_unnormalized


@dataclass
class TrainedVAE:
    model: ConvVariationalAutoencoder
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float
    state_sha256: str
    parameter_count: int
    run_seed: int
    config: VAETrainingConfig


def _fixed_validation_epsilons(
    row_count: int,
    latent_dimension: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        row_count,
        latent_dimension,
        generator=generator,
        dtype=torch.float32,
    )


def train_vae(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    validation_values: np.ndarray,
    validation_uids: Sequence[str],
    config: VAETrainingConfig,
    run_seed: int,
    device: torch.device,
) -> TrainedVAE:
    baseline.configure_determinism(run_seed)
    model = ConvVariationalAutoencoder(
        train_values.shape[1],
        config.channels,
        config.latent_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_dataset = baseline.ReconstructionDataset(
        train_values,
        train_uids,
        "clean",
        run_seed,
        config.maximum_epochs,
    )
    validation_dataset = baseline.ReconstructionDataset(
        validation_values,
        validation_uids,
        "clean",
        baseline.stable_seed(run_seed, "validation"),
        config.maximum_epochs,
        fixed_epoch=config.maximum_epochs - 1,
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
    validation_epsilon = _fixed_validation_epsilons(
        len(validation_values),
        config.latent_dimension,
        baseline.stable_seed(run_seed, "validation_epsilon"),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []

    for epoch in range(config.maximum_epochs):
        beta = beta_for_epoch(config.kl_schedule, epoch, config.maximum_epochs)
        model.train()
        train_totals = np.zeros(3, dtype=float)
        train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, mu, log_variance, _ = model(inputs, sample=True)
            total, reconstruction_term, kl_unnormalized = loss_components(
                reconstruction,
                targets,
                mu,
                log_variance,
                config,
                beta,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            count = len(inputs)
            train_totals += (
                np.asarray(
                    [
                        float(total.detach().cpu()),
                        float(reconstruction_term.detach().cpu()),
                        float(kl_unnormalized.detach().cpu()),
                    ]
                )
                * count
            )
            train_count += count

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
                    inputs,
                    sample=True,
                    epsilon=epsilon,
                )
                # Selection always monitors the final beta=1 objective, even
                # while the optimization schedule is warming up or cycling.
                total, reconstruction_term, kl_unnormalized = loss_components(
                    reconstruction,
                    targets,
                    mu,
                    log_variance,
                    config,
                    config.beta,
                )
                validation_totals += (
                    np.asarray(
                        [
                            float(total.cpu()),
                            float(reconstruction_term.cpu()),
                            float(kl_unnormalized.cpu()),
                        ]
                    )
                    * count
                )
                validation_count += count

        train_means = train_totals / max(train_count, 1)
        validation_means = validation_totals / max(validation_count, 1)
        records.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "train_loss": train_means[0],
                "train_reconstruction_loss": train_means[1],
                "train_kl_unnormalized": train_means[2],
                "train_kl_normalized": (
                    train_means[2] / config.kl_normalization_divisor
                ),
                "validation_loss": validation_means[0],
                "validation_reconstruction_loss": validation_means[1],
                "validation_kl_unnormalized": validation_means[2],
                "validation_kl_normalized": (
                    validation_means[2] / config.kl_normalization_divisor
                ),
            }
        )
        validation_loss = float(validation_means[0])
        if (
            validation_loss
            < best_validation - config.early_stopping_minimum_delta
        ):
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch + 1 >= config.minimum_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError("No VAE state was selected")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainedVAE(
        model=model,
        history=pd.DataFrame(records),
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation),
        state_sha256=baseline.state_dict_sha256(best_state),
        parameter_count=baseline.model_parameter_count(model),
        run_seed=run_seed,
        config=config,
    )


def vae_outputs(
    model: ConvVariationalAutoencoder,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior-mean reconstruction, mu, and log variance."""
    model.eval()
    reconstructions: list[np.ndarray] = []
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1)
            reconstruction, mu, log_variance, _ = model(
                batch.to(device), sample=False
            )
            reconstructions.append(reconstruction.squeeze(1).cpu().numpy())
            means.append(mu.cpu().numpy())
            log_variances.append(log_variance.cpu().numpy())
    return (
        np.vstack(reconstructions),
        np.vstack(means),
        np.vstack(log_variances),
    )


def posterior_sample_reconstruction_variability(
    model: ConvVariationalAutoencoder,
    values: np.ndarray,
    device: torch.device,
    seed: int,
    samples: int = 8,
) -> float:
    """Mean spectral standard deviation across deterministic posterior draws."""
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    variability: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 128):
            batch_values = np.asarray(values[start : start + 128], dtype=np.float32)
            batch = torch.from_numpy(batch_values).unsqueeze(1).to(device)
            mu, log_variance = model.encode(batch)
            draws: list[np.ndarray] = []
            for _ in range(samples):
                epsilon = torch.randn(
                    mu.shape,
                    generator=generator,
                    dtype=mu.dtype,
                    device="cpu",
                ).to(device)
                latent = model.reparameterize(mu, log_variance, epsilon)
                draws.append(model.decoder(latent).squeeze(1).cpu().numpy())
            variability.append(np.std(np.stack(draws, axis=0), axis=0))
    return float(np.mean(np.concatenate(variability, axis=0)))


def variational_metrics(
    mu: np.ndarray,
    log_variance: np.ndarray,
    *,
    normalization_divisor: int,
    sample_reconstruction_variability: float,
) -> dict[str, float | int]:
    mu = np.asarray(mu, dtype=float)
    log_variance = np.asarray(log_variance, dtype=float)
    per_cell_kl = -0.5 * (
        1.0 + log_variance - np.square(mu) - np.exp(log_variance)
    )
    per_dimension = per_cell_kl.mean(axis=0)
    unnormalized = float(per_cell_kl.sum(axis=1).mean())
    return {
        "vae_kl_unnormalized_per_observation": unnormalized,
        "vae_kl_normalized": unnormalized / float(normalization_divisor),
        "vae_mean_kl_per_latent_dimension": float(per_dimension.mean()),
        "vae_median_kl_per_latent_dimension": float(np.median(per_dimension)),
        "vae_max_kl_per_latent_dimension": float(per_dimension.max()),
        "vae_active_units_var_mu_gt_0_01": int(
            np.sum(np.var(mu, axis=0) > 0.01)
        ),
        "vae_dimensions_mean_kl_gt_0_01": int(
            np.sum(per_dimension > 0.01)
        ),
        "vae_posterior_mean_absolute_mu": float(np.mean(np.abs(mu))),
        "vae_posterior_mean_log_variance": float(np.mean(log_variance)),
        "vae_posterior_sample_reconstruction_variability": float(
            sample_reconstruction_variability
        ),
    }


def config_record(config: VAETrainingConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "channels": list(config.channels),
        "identifier": config.identifier,
    }
