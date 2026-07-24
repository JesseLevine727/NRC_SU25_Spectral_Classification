#!/usr/bin/env python3
"""Shared deterministic machinery for SERS representation baselines.

This module deliberately contains no experiment-selection side effects. The
orchestrator supplies explicit split masks, configurations, and seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# This must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

import freeze_nato_sers_preprocessing as preprocessing_v1


PROTOCOL_VERSION = "sers-representation-baselines-v1"
AUTHORIZED_REPRESENTATIONS = (
    "minimal_minmax",
    "arpls_minmax",
    "derivative_1",
)
INTENSITY_REPRESENTATIONS = ("minimal_minmax", "arpls_minmax")
CORRUPTION_NAMES = (
    "scale_offset",
    "smooth_baseline",
    "gaussian_noise",
    "isolated_spikes",
    "edge_filled_shift",
    "gaussian_broadening",
    "composite",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def configure_determinism(seed: int) -> None:
    """Make a run order-independent and fail on nondeterministic torch ops."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def verify_hash_catalog(bundle_dir: Path) -> dict[str, str]:
    catalog_path = bundle_dir / "artifact_hashes.json"
    catalog = json.loads(catalog_path.read_text())
    for relative_path, expected in catalog.items():
        actual = sha256_file(bundle_dir / relative_path)
        if actual != expected:
            raise ValueError(
                f"Immutable bundle hash mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )
    return catalog


@dataclass(frozen=True)
class SpectralDataset:
    name: str
    axis_cm1: np.ndarray
    observation_uid: np.ndarray
    representations: dict[str, np.ndarray]
    manifest: pd.DataFrame

    def __post_init__(self) -> None:
        n_rows = len(self.observation_uid)
        if len(self.manifest) != n_rows:
            raise ValueError("Manifest and observation ID lengths differ")
        if not np.array_equal(
            self.manifest["observation_uid"].astype(str).to_numpy(),
            self.observation_uid.astype(str),
        ):
            raise ValueError("Manifest and array observation order differ")
        for name, values in self.representations.items():
            if values.shape != (n_rows, len(self.axis_cm1)):
                raise ValueError(f"Unexpected shape for {name}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {name}")


def load_nato_dataset(bundle_dir: Path) -> SpectralDataset:
    verify_hash_catalog(bundle_dir)
    version = json.loads((bundle_dir / "dataset_version.json").read_text())
    if version["dataset_version"] != "nato-sers-preprocessing-v2":
        raise ValueError("Expected nato-sers-preprocessing-v2")
    if version["selected_final_representations"] != list(
        AUTHORIZED_REPRESENTATIONS
    ):
        raise ValueError("Frozen NATO representation selection changed")
    archive = np.load(bundle_dir / "final_model_inputs_core.npz")
    manifest = pd.read_csv(bundle_dir / "core_preprocessing_manifest.csv")
    axis = archive["axis_cm1"].astype(np.float32)
    if not np.array_equal(axis, np.arange(400, 1801, dtype=np.float32)):
        raise ValueError("Unexpected NATO axis")
    representations = {
        name: archive[name].astype(np.float32)
        for name in AUTHORIZED_REPRESENTATIONS
    }
    for name in INTENSITY_REPRESENTATIONS:
        values = representations[name]
        if not np.allclose(values.min(axis=1), 0.0, atol=2.0e-6):
            raise ValueError(f"{name} rows do not begin at zero")
        if not np.allclose(values.max(axis=1), 1.0, atol=2.0e-6):
            raise ValueError(f"{name} rows do not end at one")
    return SpectralDataset(
        name="NATO-L598",
        axis_cm1=axis,
        observation_uid=archive["observation_uid"].astype(str),
        representations=representations,
        manifest=manifest,
    )


def minmax_rows(values: np.ndarray) -> np.ndarray:
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return ((values - low) / np.maximum(high - low, 1.0e-12)).astype(
        np.float32
    )


def load_poster_dataset(source_csv: Path) -> SpectralDataset:
    frame = pd.read_csv(source_csv)
    frame = frame.copy()
    frame["source_row_index"] = np.arange(len(frame), dtype=int)
    frame["source_label"] = frame["Label"].astype(str)
    frame["source_substrate"] = frame["Substrate"].astype(str)
    frame["Label"] = frame["Label"].replace({"bt": "benzenethiol"})
    frame["Substrate"] = frame["Substrate"].replace(
        {"AgNP": "Ag", "AuNP": "Au"}
    )
    supported = (
        frame.groupby("Label")["Substrate"]
        .nunique()
        .loc[lambda values: values >= 2]
        .index
    )
    frame = frame[frame["Label"].isin(supported)].reset_index(drop=True)
    if len(frame) != 275:
        raise ValueError(f"Expected 275 poster chemical-only rows, got {len(frame)}")

    spectral_columns: list[tuple[float, str]] = []
    for column in frame.columns:
        try:
            spectral_columns.append((float(column), column))
        except (TypeError, ValueError):
            continue
    spectral_columns.sort()
    native_axis = np.asarray([item[0] for item in spectral_columns], dtype=float)
    native_values = frame[[item[1] for item in spectral_columns]].to_numpy(
        dtype=float
    )
    axis = np.arange(400, 1801, dtype=np.float32)
    common = np.vstack(
        [np.interp(axis, native_axis, row) for row in native_values]
    )

    config = preprocessing_v1.PreprocessingConfig()
    spike_mask, _ = preprocessing_v1.detect_spikes(common, config)
    despiked = preprocessing_v1.repair_masked_points(common, spike_mask)
    arpls_baseline = preprocessing_v1.arpls_baseline_matrix(despiked, config)
    minimal = minmax_rows(despiked)
    arpls = minmax_rows(despiked - arpls_baseline)
    derivative = preprocessing_v1.l2_rows(
        savgol_filter(
            preprocessing_v1.snv(despiked),
            config.derivative_window_points,
            config.derivative_polynomial_order,
            deriv=1,
            axis=1,
        )
    ).astype(np.float32)

    manifest = pd.DataFrame(
        {
            "observation_uid": [
                f"poster_row_{index:04d}"
                for index in frame["source_row_index"].astype(int)
            ],
            "source_row_index": frame["source_row_index"].astype(int),
            "target_analyte": frame["Label"].astype(str),
            "substrate_family": frame["Substrate"].astype(str),
            "source_substrate": frame["source_substrate"].astype(str),
            "source_label": frame["source_label"].astype(str),
            "master_sample_id": pd.NA,
            "include_sers_qc_pass": True,
            "field_quality_stress": False,
        }
    )
    return SpectralDataset(
        name="Poster-275",
        axis_cm1=axis,
        observation_uid=manifest["observation_uid"].to_numpy(dtype=str),
        representations={
            "minimal_minmax": minimal,
            "arpls_minmax": arpls,
            "derivative_1": derivative,
        },
        manifest=manifest,
    )


def load_poster_historical_derivative(
    source_csv: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(source_csv)
    frame = frame.copy()
    frame["source_row_index"] = np.arange(len(frame), dtype=int)
    frame["source_substrate"] = frame["Substrate"].astype(str)
    frame["Label"] = frame["Label"].replace({"bt": "benzenethiol"})
    frame["Substrate"] = frame["Substrate"].replace(
        {"AgNP": "Ag", "AuNP": "Au"}
    )
    supported = (
        frame.groupby("Label")["Substrate"]
        .nunique()
        .loc[lambda values: values >= 2]
        .index
    )
    frame = frame[frame["Label"].isin(supported)].reset_index(drop=True)
    columns: list[tuple[float, str]] = []
    for column in frame.columns:
        try:
            wavenumber = float(column)
        except (TypeError, ValueError):
            continue
        if 330 <= wavenumber <= 1800:
            columns.append((wavenumber, column))
    columns.sort()
    axis = np.asarray([item[0] for item in columns], dtype=np.float32)
    raw = frame[[item[1] for item in columns]].to_numpy(dtype=float)
    derivative = preprocessing_v1.l2_rows(
        savgol_filter(
            preprocessing_v1.snv(raw),
            17,
            3,
            deriv=1,
            axis=1,
        )
    ).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "observation_uid": [
                f"poster_row_{index:04d}"
                for index in frame["source_row_index"].astype(int)
            ],
            "target_analyte": frame["Label"].astype(str),
            "substrate_family": frame["Substrate"].astype(str),
            "source_substrate": frame["source_substrate"].astype(str),
        }
    )
    return axis, derivative, metadata


def edge_shift(values: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.empty_like(values)
    if amount > 0:
        shifted[:amount] = values[0]
        shifted[amount:] = values[:-amount]
    elif amount < 0:
        shifted[amount:] = values[-1]
        shifted[:amount] = values[-amount:]
    else:
        shifted[:] = values
    return shifted


def _smooth_baseline(
    length: int, rng: np.random.Generator, severity: float
) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, length)
    slope = rng.uniform(-0.6, 0.6)
    curve = 0.6 * coordinate**2 + slope * coordinate
    curve -= curve.min()
    curve /= max(float(np.ptp(curve)), 1.0e-12)
    return 0.25 * severity * curve


def apply_corruption(
    clean: np.ndarray,
    corruption: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Corrupt one [0,1] intensity representation without renormalizing it."""
    work = np.asarray(clean, dtype=np.float64).copy()
    severity = float(max(severity, 0.0))
    operations = (
        (
            "scale_offset",
            "smooth_baseline",
            "gaussian_noise",
            "isolated_spikes",
            "edge_filled_shift",
            "gaussian_broadening",
        )
        if corruption == "composite"
        else (corruption,)
    )
    for operation in operations:
        if operation == "scale_offset":
            factor = rng.uniform(1.0 - 0.3 * severity, 1.0 + 0.3 * severity)
            offset = rng.uniform(-0.1 * severity, 0.1 * severity)
            work = factor * work + offset
        elif operation == "smooth_baseline":
            work = work + _smooth_baseline(len(work), rng, severity)
        elif operation == "gaussian_noise":
            work = work + rng.normal(0.0, 0.03 * severity, size=len(work))
        elif operation == "isolated_spikes":
            positions = rng.choice(
                np.arange(5, len(work) - 5), size=2, replace=False
            )
            work[positions] += 0.5 * severity
        elif operation == "edge_filled_shift":
            maximum = max(1, int(round(3 * severity)))
            amount = int(rng.integers(1, maximum + 1))
            amount *= -1 if rng.random() < 0.5 else 1
            work = edge_shift(work, amount)
        elif operation == "gaussian_broadening":
            work = gaussian_filter1d(
                work, sigma=max(0.05, severity), mode="nearest"
            )
        else:
            raise ValueError(f"Unknown corruption: {operation}")
    return np.clip(work, 0.0, 1.0).astype(np.float32)


def corruption_for_curriculum(
    curriculum: str,
    clean: np.ndarray,
    uid: str,
    base_seed: int,
    epoch: int,
    maximum_epochs: int,
) -> tuple[np.ndarray, str, float]:
    rng = np.random.default_rng(
        stable_seed(PROTOCOL_VERSION, base_seed, epoch, uid, curriculum)
    )
    if curriculum == "clean":
        return clean.astype(np.float32), "clean", 0.0
    if curriculum == "gaussian_only":
        corruption = "gaussian_noise"
        severity = 1.0
    elif curriculum in {"mixed_uniform", "mixed_progressive"}:
        corruption = str(rng.choice(CORRUPTION_NAMES))
        severity = (
            min(1.0, 0.25 + 0.75 * epoch / max(maximum_epochs - 1, 1))
            if curriculum == "mixed_progressive"
            else 1.0
        )
    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    return apply_corruption(clean, corruption, severity, rng), corruption, severity


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        clean: np.ndarray,
        observation_uids: Sequence[str],
        curriculum: str,
        base_seed: int,
        maximum_epochs: int,
        fixed_epoch: int | None = None,
    ):
        self.clean = np.asarray(clean, dtype=np.float32)
        self.observation_uids = np.asarray(observation_uids, dtype=str)
        self.curriculum = curriculum
        self.base_seed = int(base_seed)
        self.maximum_epochs = int(maximum_epochs)
        self.fixed_epoch = fixed_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch = self.fixed_epoch if self.fixed_epoch is not None else self.epoch
        corrupted, _, _ = corruption_for_curriculum(
            self.curriculum,
            self.clean[index],
            self.observation_uids[index],
            self.base_seed,
            int(epoch),
            self.maximum_epochs,
        )
        return (
            torch.from_numpy(corrupted).unsqueeze(0),
            torch.from_numpy(self.clean[index]).unsqueeze(0),
        )


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_length: int,
        channels: Sequence[int],
        bottleneck_dimension: int,
        normalize_output: bool = False,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError("Exactly two convolutional channel widths required")
        self.input_length = int(input_length)
        self.channels = tuple(int(value) for value in channels)
        self.bottleneck_dimension = int(bottleneck_dimension)
        self.normalize_output = bool(normalize_output)
        self.features = nn.Sequential(
            nn.Conv1d(1, self.channels[0], kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(
                self.channels[0], self.channels[1], kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            example = torch.zeros(1, 1, self.input_length)
            feature_shape = self.features(example).shape
        self.feature_length = int(feature_shape[-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.channels[1] * self.feature_length,
                self.bottleneck_dimension,
            ),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        latent = self.projection(self.features(values))
        return F.normalize(latent, dim=1) if self.normalize_output else latent


class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_length: int,
        channels: Sequence[int],
        feature_length: int,
        bottleneck_dimension: int,
    ):
        super().__init__()
        self.output_length = int(output_length)
        self.channels = tuple(int(value) for value in channels)
        self.feature_length = int(feature_length)
        self.expansion = nn.Sequential(
            nn.Linear(
                int(bottleneck_dimension),
                self.channels[1] * self.feature_length,
            ),
            nn.ReLU(),
        )
        self.convolution_2 = nn.Conv1d(
            self.channels[1], self.channels[0], kernel_size=5, padding=2
        )
        self.convolution_1 = nn.Conv1d(
            self.channels[0], 1, kernel_size=7, padding=3
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.expansion(latent).reshape(
            len(latent), self.channels[1], self.feature_length
        )
        # CUDA's linear 1-D interpolation backward pass is nondeterministic in
        # the pinned environment. Nearest upsampling is deterministic; the
        # following learned convolution performs the smooth reconstruction.
        values = F.interpolate(values, scale_factor=2.0, mode="nearest")
        values = F.relu(self.convolution_2(values))
        values = F.interpolate(
            values,
            size=self.output_length,
            mode="nearest",
        )
        return torch.sigmoid(self.convolution_1(values))


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        input_length: int,
        channels: Sequence[int],
        bottleneck_dimension: int,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            input_length,
            channels,
            bottleneck_dimension,
            normalize_output=False,
        )
        self.decoder = ConvDecoder(
            input_length,
            channels,
            self.encoder.feature_length,
            bottleneck_dimension,
        )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        return self.decoder(latent), latent


class SiameseEncoder(ConvEncoder):
    def __init__(self, input_length: int):
        super().__init__(
            input_length,
            channels=(16, 32),
            bottleneck_dimension=64,
            normalize_output=True,
        )


def augment_siamese_vector(
    values: np.ndarray,
    rng: np.random.Generator,
    noise_standard_deviation: float = 0.01,
    maximum_shift_indices: int = 2,
    circular_shift: bool = False,
) -> np.ndarray:
    augmented = np.asarray(values, dtype=np.float32) + rng.normal(
        0.0, noise_standard_deviation, size=len(values)
    ).astype(np.float32)
    shift = int(
        rng.integers(-maximum_shift_indices, maximum_shift_indices + 1)
    )
    return (
        np.roll(augmented, shift)
        if circular_shift
        else edge_shift(augmented, shift)
    ).astype(np.float32)


class DeterministicTripletDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        labels: Sequence[str],
        domains: Sequence[str],
        observation_uids: Sequence[str],
        base_seed: int,
        circular_shift: bool = False,
    ):
        self.values = np.asarray(values, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=str)
        self.domains = np.asarray(domains, dtype=str)
        self.observation_uids = np.asarray(observation_uids, dtype=str)
        self.base_seed = int(base_seed)
        self.circular_shift = bool(circular_shift)
        self.epoch = 0
        self.by_label = {
            label: np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self.by_domain = {
            domain: np.flatnonzero(self.domains == domain)
            for domain in np.unique(self.domains)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(
            stable_seed(
                PROTOCOL_VERSION,
                self.base_seed,
                self.epoch,
                self.observation_uids[index],
                "triplet",
            )
        )
        label = self.labels[index]
        domain = self.domains[index]
        positive_candidates = self.by_label[label]
        cross_domain = positive_candidates[
            self.domains[positive_candidates] != domain
        ]
        if len(cross_domain):
            positive_index = int(rng.choice(cross_domain))
        else:
            other = positive_candidates[positive_candidates != index]
            positive_index = int(
                rng.choice(other if len(other) else positive_candidates)
            )

        same_domain = self.by_domain[domain]
        same_domain_negative = same_domain[
            self.labels[same_domain] != label
        ]
        if len(same_domain_negative):
            negative_index = int(rng.choice(same_domain_negative))
        else:
            other_labels = sorted(set(self.by_label) - {label})
            negative_label = str(rng.choice(other_labels))
            negative_index = int(rng.choice(self.by_label[negative_label]))

        vectors: list[torch.Tensor] = []
        for role, selected_index in (
            ("anchor", index),
            ("positive", positive_index),
            ("negative", negative_index),
        ):
            role_rng = np.random.default_rng(
                stable_seed(
                    self.base_seed,
                    self.epoch,
                    self.observation_uids[index],
                    role,
                )
            )
            augmented = augment_siamese_vector(
                self.values[selected_index],
                role_rng,
                circular_shift=self.circular_shift,
            )
            vectors.append(torch.from_numpy(augmented).unsqueeze(0))
        return vectors[0], vectors[1], vectors[2]


@dataclass
class TrainedSiamese:
    model: SiameseEncoder
    history: pd.DataFrame
    state_sha256: str
    parameter_count: int
    run_seed: int
    epochs: int
    circular_shift: bool


def train_siamese(
    train_values: np.ndarray,
    train_labels: Sequence[str],
    train_domains: Sequence[str],
    train_uids: Sequence[str],
    run_seed: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1.0e-3,
    margin: float = 0.2,
    circular_shift: bool = False,
) -> TrainedSiamese:
    configure_determinism(run_seed)
    dataset = DeterministicTripletDataset(
        train_values,
        train_labels,
        train_domains,
        train_uids,
        run_seed,
        circular_shift=circular_shift,
    )
    generator = torch.Generator()
    generator.manual_seed(stable_seed(run_seed, "siamese_loader"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = SiameseEncoder(train_values.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        total = 0.0
        count = 0
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            loss = F.triplet_margin_loss(
                model(anchor),
                model(positive),
                model(negative),
                margin=margin,
                p=2,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(anchor)
            count += len(anchor)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(count, 1),
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return TrainedSiamese(
        model=model,
        history=pd.DataFrame(history),
        state_sha256=state_dict_sha256(state),
        parameter_count=model_parameter_count(model),
        run_seed=run_seed,
        epochs=epochs,
        circular_shift=circular_shift,
    )


def embed_siamese(
    model: SiameseEncoder,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1)
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.vstack(outputs)


def nearest_prototype_predict(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    test_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_labels_array = np.asarray(train_labels, dtype=str)
    prototype_labels = sorted(np.unique(train_labels_array))
    prototypes = np.vstack(
        [
            train_embeddings[train_labels_array == label].mean(axis=0)
            for label in prototype_labels
        ]
    )
    distances = np.linalg.norm(
        test_embeddings[:, None, :] - prototypes[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    prediction = np.asarray(
        [prototype_labels[index] for index in nearest], dtype=str
    )
    confidence = -distances[np.arange(len(distances)), nearest]
    return prediction, confidence, prototypes


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def spectral_angle_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    cosine = F.cosine_similarity(prediction_flat, target_flat, dim=1)
    cosine = torch.clamp(cosine, -1.0 + 1.0e-7, 1.0 - 1.0e-7)
    return torch.acos(cosine).mean() / torch.pi


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "spectral_composite":
        smooth_l1 = F.smooth_l1_loss(prediction, target)
        angle = spectral_angle_loss(prediction, target)
        derivative = F.l1_loss(
            torch.diff(prediction, dim=-1),
            torch.diff(target, dim=-1),
        )
        return smooth_l1 + 0.1 * angle + 0.1 * derivative
    raise ValueError(f"Unknown reconstruction loss: {loss_name}")


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    channels: tuple[int, int]
    bottleneck_dimension: int
    loss_name: str
    curriculum: str = "clean"
    optimizer: str = "Adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    batch_size: int = 64
    maximum_epochs: int = 100
    minimum_epochs: int = 20
    early_stopping_patience: int = 12
    early_stopping_minimum_delta: float = 1.0e-5
    gradient_clip_norm: float = 5.0

    @property
    def identifier(self) -> str:
        channel_text = "x".join(str(value) for value in self.channels)
        return (
            f"c{channel_text}_z{self.bottleneck_dimension}_"
            f"{self.loss_name}_{self.curriculum}"
        )


@dataclass
class TrainedAutoencoder:
    model: ConvAutoencoder
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float
    state_sha256: str
    parameter_count: int
    run_seed: int
    config: AutoencoderTrainingConfig


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train_autoencoder(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    validation_values: np.ndarray,
    validation_uids: Sequence[str],
    config: AutoencoderTrainingConfig,
    run_seed: int,
    device: torch.device,
) -> TrainedAutoencoder:
    configure_determinism(run_seed)
    model = ConvAutoencoder(
        train_values.shape[1],
        config.channels,
        config.bottleneck_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_dataset = ReconstructionDataset(
        train_values,
        train_uids,
        config.curriculum,
        run_seed,
        config.maximum_epochs,
    )
    validation_dataset = ReconstructionDataset(
        validation_values,
        validation_uids,
        config.curriculum,
        stable_seed(run_seed, "validation"),
        config.maximum_epochs,
        fixed_epoch=config.maximum_epochs - 1,
    )
    generator = torch.Generator()
    generator.manual_seed(stable_seed(run_seed, "loader"))
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

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []

    for epoch in range(config.maximum_epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, _ = model(inputs)
            loss = reconstruction_loss(reconstruction, targets, config.loss_name)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(inputs)
            train_count += len(inputs)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                reconstruction, _ = model(inputs)
                loss = reconstruction_loss(
                    reconstruction, targets, config.loss_name
                )
                validation_total += float(loss.detach().cpu()) * len(inputs)
                validation_count += len(inputs)
        train_mean = train_total / max(train_count, 1)
        validation_mean = validation_total / max(validation_count, 1)
        records.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_mean,
                "validation_loss": validation_mean,
            }
        )

        if (
            validation_mean
            < best_validation - config.early_stopping_minimum_delta
        ):
            best_validation = validation_mean
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
        raise RuntimeError("No autoencoder state was selected")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainedAutoencoder(
        model=model,
        history=pd.DataFrame(records),
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation),
        state_sha256=state_dict_sha256(best_state),
        parameter_count=model_parameter_count(model),
        run_seed=run_seed,
        config=config,
    )


def autoencoder_outputs(
    model: ConvAutoencoder,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reconstructions: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.asarray(values[start : start + batch_size], dtype=np.float32)
            ).unsqueeze(1)
            reconstruction, latent = model(batch.to(device))
            reconstructions.append(reconstruction.squeeze(1).cpu().numpy())
            latents.append(latent.cpu().numpy())
    return np.vstack(reconstructions), np.vstack(latents)


def supported_balanced_accuracy(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> float:
    labels = np.unique(true_labels)
    return float(
        np.mean(
            [
                np.mean(predicted_labels[true_labels == label] == label)
                for label in labels
            ]
        )
    )


def classification_summary(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, float]:
    true_values = np.asarray(true_labels, dtype=str)
    predicted_values = np.asarray(predicted_labels, dtype=str)
    supported = sorted(np.unique(true_values))
    union = sorted(np.unique(np.concatenate([true_values, predicted_values])))
    return {
        "balanced_accuracy": supported_balanced_accuracy(
            true_values, predicted_values
        ),
        "macro_f1_supported": float(
            f1_score(
                true_values,
                predicted_values,
                labels=supported,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_union": float(
            f1_score(
                true_values,
                predicted_values,
                labels=union,
                average="macro",
                zero_division=0,
            )
        ),
    }


def per_class_classification(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> pd.DataFrame:
    true_values = np.asarray(true_labels, dtype=str)
    predicted_values = np.asarray(predicted_labels, dtype=str)
    labels = sorted(np.unique(np.concatenate([true_values, predicted_values])))
    matrix = confusion_matrix(true_values, predicted_values, labels=labels)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        rows.append(
            {
                "class_label": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "recall": true_positive / support if support else np.nan,
                "precision": (
                    true_positive / predicted_count
                    if predicted_count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_classical_predict(
    model_name: str,
    train_values: np.ndarray,
    train_labels: Sequence[str],
    test_values: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    train_labels_array = np.asarray(train_labels, dtype=str)
    if model_name == "nearest_centroid":
        model: Any = NearestCentroid(metric="euclidean")
    elif model_name == "pca_logistic":
        components = min(32, len(train_values) - 1, train_values.shape[1])
        model = make_pipeline(
            PCA(
                n_components=components,
                whiten=True,
                svd_solver="randomized",
                random_state=random_seed,
            ),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_seed,
            ),
        )
    elif model_name == "linear_svm":
        model = make_pipeline(
            StandardScaler(with_mean=False),
            SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced",
                probability=False,
                random_state=random_seed,
            ),
        )
    else:
        raise ValueError(f"Unknown classical model: {model_name}")
    model.fit(train_values, train_labels_array)
    prediction = model.predict(test_values).astype(str)
    confidence: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(test_values), axis=1)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(test_values)
        confidence = (
            np.abs(decision)
            if np.ndim(decision) == 1
            else np.max(decision, axis=1)
        )
    return prediction, confidence


def pca_project_train_test(
    train_values: np.ndarray,
    test_values: np.ndarray,
    random_seed: int,
    maximum_components: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    components = min(
        maximum_components, len(train_values) - 1, train_values.shape[1]
    )
    model = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=random_seed,
    )
    return model.fit_transform(train_values), model.transform(test_values)


def fit_latent_probe_predict(
    train_latent: np.ndarray,
    train_labels: Sequence[str],
    test_latent: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    model.fit(train_latent, np.asarray(train_labels, dtype=str))
    prediction = model.predict(test_latent).astype(str)
    confidence = np.max(model.predict_proba(test_latent), axis=1)
    return prediction, confidence


def fit_latent_probe_model(
    train_latent: np.ndarray,
    train_labels: Sequence[str],
    random_seed: int,
) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    model.fit(train_latent, np.asarray(train_labels, dtype=str))
    return model


def target_domain_cell_weights(
    targets: Sequence[str], domains: Sequence[str]
) -> np.ndarray:
    cells = pd.Series(
        list(zip(np.asarray(targets, dtype=str), np.asarray(domains, dtype=str)))
    )
    counts = cells.value_counts()
    weights = np.asarray([1.0 / counts[cell] for cell in cells], dtype=float)
    return weights / np.mean(weights)


def cell_balanced_accuracy(
    targets: Sequence[str],
    domains: Sequence[str],
    predictions: Sequence[str],
) -> float:
    frame = pd.DataFrame(
        {
            "target": np.asarray(targets, dtype=str),
            "domain": np.asarray(domains, dtype=str),
            "correct": np.asarray(predictions, dtype=str)
            == np.asarray(domains, dtype=str),
        }
    )
    return float(frame.groupby(["target", "domain"])["correct"].mean().mean())


def target_adjusted_domain_probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: Sequence[str],
    test_targets: Sequence[str],
    train_domains: Sequence[str],
    test_domains: Sequence[str],
    random_seed: int,
) -> dict[str, float]:
    train_targets_array = np.asarray(train_targets, dtype=str)
    test_targets_array = np.asarray(test_targets, dtype=str)
    train_domains_array = np.asarray(train_domains, dtype=str)
    test_domains_array = np.asarray(test_domains, dtype=str)
    weights = target_domain_cell_weights(
        train_targets_array, train_domains_array
    )
    spectral_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=random_seed),
    )
    spectral_model.fit(
        train_features, train_domains_array, logisticregression__sample_weight=weights
    )
    spectral_prediction = spectral_model.predict(test_features)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_target_only = encoder.fit_transform(
        train_targets_array.reshape(-1, 1)
    )
    test_target_only = encoder.transform(test_targets_array.reshape(-1, 1))
    null_model = LogisticRegression(max_iter=3000, random_state=random_seed)
    null_model.fit(
        train_target_only, train_domains_array, sample_weight=weights
    )
    null_prediction = null_model.predict(test_target_only)
    spectral_score = cell_balanced_accuracy(
        test_targets_array, test_domains_array, spectral_prediction
    )
    null_score = cell_balanced_accuracy(
        test_targets_array, test_domains_array, null_prediction
    )
    return {
        "cell_balanced_accuracy": spectral_score,
        "target_only_null_cell_balanced_accuracy": null_score,
        "increment_over_target_only": spectral_score - null_score,
    }


def correlation_distance_rows(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered, axis=1
    )
    correlation = np.sum(left_centered * right_centered, axis=1) / np.maximum(
        denominator, 1.0e-12
    )
    return 1.0 - correlation


def geometry_metrics(
    features: np.ndarray,
    manifest: pd.DataFrame,
    selection: np.ndarray,
    domain_column: str = "instrument",
) -> dict[str, float]:
    selected_indices = np.flatnonzero(selection)
    same_master: list[float] = []
    different_target: list[float] = []
    for local_left in range(len(selected_indices)):
        left = selected_indices[local_left]
        for local_right in range(local_left + 1, len(selected_indices)):
            right = selected_indices[local_right]
            if (
                str(manifest.iloc[left][domain_column])
                == str(manifest.iloc[right][domain_column])
            ):
                continue
            distance = float(
                correlation_distance_rows(
                    features[left : left + 1], features[right : right + 1]
                )[0]
            )
            if (
                str(manifest.iloc[left]["master_sample_id"])
                == str(manifest.iloc[right]["master_sample_id"])
            ):
                same_master.append(distance)
            if (
                str(manifest.iloc[left]["target_analyte"])
                != str(manifest.iloc[right]["target_analyte"])
            ):
                different_target.append(distance)
    same_mean = float(np.mean(same_master)) if same_master else np.nan
    different_mean = (
        float(np.mean(different_target)) if different_target else np.nan
    )
    return {
        "same_master_cross_domain_mean_distance": same_mean,
        "different_target_cross_domain_mean_distance": different_mean,
        "cross_domain_separation_margin": different_mean - same_mean,
        "same_master_pair_count": len(same_master),
        "different_target_pair_count": len(different_target),
    }


def peak_table(values: np.ndarray) -> pd.DataFrame:
    peaks, properties = find_peaks(values, prominence=0.05, distance=5)
    widths = (
        peak_widths(values, peaks, rel_height=0.5)[0]
        if len(peaks)
        else np.asarray([], dtype=float)
    )
    return pd.DataFrame(
        {
            "position": peaks.astype(int),
            "prominence": properties.get(
                "prominences", np.asarray([], dtype=float)
            ),
            "width": widths,
        }
    )


def repeatable_peak_positions(
    reference_values: np.ndarray, manifest: pd.DataFrame
) -> list[set[int]]:
    tables = [peak_table(row) for row in reference_values]
    repeatable: list[set[int]] = [set() for _ in tables]
    for _, indices in manifest.groupby("master_sample_id").groups.items():
        group = list(indices)
        instruments = {
            index: str(manifest.iloc[index]["instrument"]) for index in group
        }
        if len(group) < 2 or len(set(instruments.values())) < 2:
            continue
        required = max(1, int(np.ceil(0.5 * (len(group) - 1))))
        for index in group:
            for peak in tables[index].itertuples(index=False):
                if float(peak.prominence) < 0.15:
                    continue
                supporting: list[int] = []
                for other in group:
                    if other == index:
                        continue
                    positions = tables[other]["position"].to_numpy(dtype=int)
                    if len(positions) and np.min(
                        np.abs(positions - int(peak.position))
                    ) <= 3:
                        supporting.append(other)
                if (
                    len(supporting) >= required
                    and any(
                        instruments[other] != instruments[index]
                        for other in supporting
                    )
                ):
                    repeatable[index].add(int(peak.position))
    return repeatable


def reconstruction_metrics(
    reference_values: np.ndarray,
    reconstructed_values: np.ndarray,
    observation_uids: Sequence[str],
    repeatable_positions: Sequence[set[int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (reference, reconstructed) in enumerate(
        zip(reference_values, reconstructed_values)
    ):
        reference = np.asarray(reference, dtype=float)
        reconstructed = np.asarray(reconstructed, dtype=float)
        denominator = max(
            float(np.linalg.norm(reference) * np.linalg.norm(reconstructed)),
            1.0e-12,
        )
        cosine = float(np.dot(reference, reconstructed) / denominator)
        correlation = (
            float(np.corrcoef(reference, reconstructed)[0, 1])
            if np.std(reference) > 0 and np.std(reconstructed) > 0
            else np.nan
        )
        reference_peaks = peak_table(reference)
        reconstructed_peaks = peak_table(reconstructed)
        available = list(range(len(reconstructed_peaks)))
        repeatable = (
            repeatable_positions[index]
            if repeatable_positions is not None
            else {
                int(position)
                for position in reference_peaks.loc[
                    reference_peaks["prominence"] >= 0.15, "position"
                ]
            }
        )
        shifts: list[float] = []
        width_changes: list[float] = []
        prominence_changes: list[float] = []
        repeatable_matches = 0
        for peak in reference_peaks.itertuples(index=False):
            if not available:
                continue
            positions = reconstructed_peaks.iloc[available][
                "position"
            ].to_numpy(dtype=int)
            differences = np.abs(positions - int(peak.position))
            best_local = int(np.argmin(differences))
            if differences[best_local] > 5:
                continue
            match_index = available.pop(best_local)
            match = reconstructed_peaks.iloc[match_index]
            shifts.append(float(differences[best_local]))
            width_changes.append(
                abs(float(match["width"]) - float(peak.width))
                / max(float(peak.width), 1.0e-12)
            )
            prominence_changes.append(
                abs(float(match["prominence"]) - float(peak.prominence))
            )
            if int(peak.position) in repeatable:
                repeatable_matches += 1
        rows.append(
            {
                "observation_uid": str(observation_uids[index]),
                "mse": float(np.mean((reference - reconstructed) ** 2)),
                "smooth_l1": float(
                    np.mean(
                        np.where(
                            np.abs(reference - reconstructed) < 1,
                            0.5 * (reference - reconstructed) ** 2,
                            np.abs(reference - reconstructed) - 0.5,
                        )
                    )
                ),
                "spectral_angle": float(
                    np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
                ),
                "pearson_correlation": correlation,
                "first_derivative_mae": float(
                    np.mean(
                        np.abs(np.diff(reference) - np.diff(reconstructed))
                    )
                ),
                "reference_peak_count": len(reference_peaks),
                "matched_peak_count": len(shifts),
                "repeatable_reference_peak_count": len(repeatable),
                "repeatable_matched_peak_count": repeatable_matches,
                "median_peak_shift_cm1": (
                    float(np.median(shifts)) if shifts else np.nan
                ),
                "median_absolute_relative_peak_width_change": (
                    float(np.median(width_changes))
                    if width_changes
                    else np.nan
                ),
                "median_absolute_peak_prominence_change": (
                    float(np.median(prominence_changes))
                    if prominence_changes
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_reconstruction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    repeatable_count = float(frame["repeatable_reference_peak_count"].sum())
    return {
        "reconstruction_mse": float(frame["mse"].mean()),
        "reconstruction_smooth_l1": float(frame["smooth_l1"].mean()),
        "reconstruction_spectral_angle": float(frame["spectral_angle"].mean()),
        "reconstruction_median_row_correlation": float(
            frame["pearson_correlation"].median()
        ),
        "reconstruction_first_derivative_mae": float(
            frame["first_derivative_mae"].mean()
        ),
        "repeatable_peak_recall": (
            float(
                frame["repeatable_matched_peak_count"].sum()
                / repeatable_count
            )
            if repeatable_count
            else np.nan
        ),
        "median_peak_shift_cm1": float(frame["median_peak_shift_cm1"].median()),
        "median_absolute_relative_peak_width_change": float(
            frame["median_absolute_relative_peak_width_change"].median()
        ),
        "median_absolute_peak_prominence_change": float(
            frame["median_absolute_peak_prominence_change"].median()
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state,
            "state_sha256": state_dict_sha256(state),
            "metadata": metadata,
        },
        path,
    )


def autoencoder_config_record(
    config: AutoencoderTrainingConfig,
) -> dict[str, Any]:
    record = asdict(config)
    record["identifier"] = config.identifier
    return record
