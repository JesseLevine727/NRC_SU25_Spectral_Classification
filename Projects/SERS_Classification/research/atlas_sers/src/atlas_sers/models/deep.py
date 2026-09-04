"""Locked compact one-dimensional residual model for P04/P05."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class ResidualSpectralBlock(nn.Module):
    """One pre-activation dilated spectral convolution with an identity skip."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.normalization = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.convolution(self.activation(self.normalization(values)))


class SpectralTransition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class DeterministicAdaptiveAveragePool1d(nn.Module):
    """Adaptive mean pooling expressed with deterministic tensor reductions."""

    def __init__(self, output_bins: int) -> None:
        super().__init__()
        if output_bins < 1:
            raise ValueError("Adaptive pooling needs a positive bin count.")
        self.output_bins = output_bins

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        length = values.shape[-1]
        pooled = []
        for index in range(self.output_bins):
            start = math.floor(index * length / self.output_bins)
            stop = math.ceil((index + 1) * length / self.output_bins)
            pooled.append(values[..., start:stop].mean(dim=-1))
        return torch.stack(pooled, dim=-1)


class CompactResidualEncoder(nn.Module):
    """D0 encoder preserving ordered spectral bins through adaptive pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 24, 11, stride=1, padding=5),
            nn.GroupNorm(8, 24),
            nn.GELU(),
        )
        self.stage_1 = nn.Sequential(
            ResidualSpectralBlock(24, kernel_size=7, dilation=1),
            ResidualSpectralBlock(24, kernel_size=7, dilation=2),
        )
        self.transition_1 = SpectralTransition(24, 48)
        self.stage_2 = nn.Sequential(
            ResidualSpectralBlock(48, kernel_size=7, dilation=1),
            ResidualSpectralBlock(48, kernel_size=7, dilation=2),
        )
        self.transition_2 = SpectralTransition(48, 64)
        self.stage_3 = nn.Sequential(
            ResidualSpectralBlock(64, kernel_size=5, dilation=1),
            ResidualSpectralBlock(64, kernel_size=5, dilation=2),
        )
        self.ordered_pool = DeterministicAdaptiveAveragePool1d(16)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values)
        values = self.stage_1(values)
        values = self.transition_1(values)
        values = self.stage_2(values)
        values = self.transition_2(values)
        values = self.stage_3(values)
        return self.projection(self.ordered_pool(values))


class CompactSERSClassifier(nn.Module):
    """Station-local D0 classifier with the shared compact encoder contract."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("A station-local classifier needs at least two classes.")
        self.encoder = CompactResidualEncoder()
        self.classifier = nn.Linear(64, class_count)

    def forward(
        self, values: torch.Tensor, *, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


@dataclass(frozen=True)
class ArchitectureAudit:
    input_shape: tuple[int, int, int]
    stem_shape: tuple[int, int, int]
    transition_1_shape: tuple[int, int, int]
    transition_2_shape: tuple[int, int, int]
    pooled_shape: tuple[int, int, int]
    embedding_shape: tuple[int, int]
    logits_shape: tuple[int, int]
    trainable_parameters: int
    batch_normalization_modules: int


def architecture_audit(*, class_count: int = 3, batch_size: int = 2) -> ArchitectureAudit:
    model = CompactSERSClassifier(class_count)
    values = torch.zeros(batch_size, 1, 1401)
    with torch.no_grad():
        stem = model.encoder.stem(values)
        first = model.encoder.transition_1(model.encoder.stage_1(stem))
        second = model.encoder.transition_2(model.encoder.stage_2(first))
        pooled = model.encoder.ordered_pool(model.encoder.stage_3(second))
        embedding = model.encoder.projection(pooled)
        logits = model.classifier(embedding)
    return ArchitectureAudit(
        input_shape=tuple(values.shape),
        stem_shape=tuple(stem.shape),
        transition_1_shape=tuple(first.shape),
        transition_2_shape=tuple(second.shape),
        pooled_shape=tuple(pooled.shape),
        embedding_shape=tuple(embedding.shape),
        logits_shape=tuple(logits.shape),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        batch_normalization_modules=sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in model.modules()
        ),
    )
