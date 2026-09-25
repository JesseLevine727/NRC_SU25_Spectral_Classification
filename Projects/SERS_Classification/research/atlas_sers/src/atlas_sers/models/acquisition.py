"""Acquisition adapter around the unchanged P04 compact classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .deep import CompactSERSClassifier

EMBEDDING_SIZE = 64
PROJECTION_SIZE = 64
L2_EPSILON = 1e-8
BASE_PARAMETERS = 208691
PROJECTION_PARAMETERS = 4160
PROJECTION_MODEL_PARAMETERS = 212851
MAXIMUM_PARAMETERS_EXCLUSIVE = 250000


class AcquisitionClassifier(nn.Module):
    """Frozen compact backbone plus an optional normalized 64->64 head."""

    def __init__(self, class_count: int = 3, *, use_projection: bool = False) -> None:
        if isinstance(class_count, bool) or not isinstance(class_count, int):
            raise TypeError("class_count must be an integer")
        if class_count < 2:
            raise ValueError("A station-local classifier needs at least two classes.")
        if not isinstance(use_projection, bool):
            raise TypeError("use_projection must be a bool")
        super().__init__()
        self.class_count = int(class_count)
        self.use_projection = bool(use_projection)
        self.backbone = CompactSERSClassifier(class_count)
        if self.use_projection:
            self.projection = nn.Linear(EMBEDDING_SIZE, PROJECTION_SIZE, bias=True)
        else:
            self.projection = None
        self.assert_no_batch_normalization()
        if self.trainable_parameter_count() >= MAXIMUM_PARAMETERS_EXCLUSIVE:
            raise ValueError("AcquisitionClassifier exceeds the parameter ceiling")

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits, embedding = self.backbone(values, return_embedding=True)
        if self.projection is None:
            return logits, embedding, None
        projection = F.normalize(self.projection(embedding), dim=-1, eps=L2_EPSILON)
        return logits, embedding, projection

    def backbone_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())

    def projection_parameter_count(self) -> int:
        if self.projection is None:
            return 0
        return sum(parameter.numel() for parameter in self.projection.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def batch_normalization_modules(self) -> int:
        return sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in self.modules()
        )

    def assert_no_batch_normalization(self) -> None:
        if self.batch_normalization_modules() != 0:
            raise RuntimeError("AcquisitionClassifier forbids batch normalization")


@dataclass(frozen=True)
class AcquisitionAudit:
    class_count: int
    use_projection: bool
    backbone_parameters: int
    projection_parameters: int
    total_parameters: int
    trainable_parameters: int
    batch_normalization_modules: int
    embedding_size: int
    projection_size: int | None
    logits_size: int
    enforcing_parameter_ceiling: bool


def acquisition_audit(
    *, class_count: int = 3, use_projection: bool = False, batch_size: int = 2
) -> AcquisitionAudit:
    model = AcquisitionClassifier(class_count, use_projection=use_projection)
    values = torch.zeros(batch_size, 1, 1401)
    with torch.no_grad():
        logits, embedding, projection = model(values)
    total = sum(parameter.numel() for parameter in model.parameters())
    return AcquisitionAudit(
        class_count=class_count,
        use_projection=use_projection,
        backbone_parameters=model.backbone_parameter_count(),
        projection_parameters=model.projection_parameter_count(),
        total_parameters=total,
        trainable_parameters=model.trainable_parameter_count(),
        batch_normalization_modules=model.batch_normalization_modules(),
        embedding_size=int(embedding.shape[1]),
        projection_size=None if projection is None else int(projection.shape[1]),
        logits_size=int(logits.shape[1]),
        enforcing_parameter_ceiling=total < MAXIMUM_PARAMETERS_EXCLUSIVE,
    )


__all__ = [
    "AcquisitionAudit",
    "AcquisitionClassifier",
    "BASE_PARAMETERS",
    "EMBEDDING_SIZE",
    "L2_EPSILON",
    "MAXIMUM_PARAMETERS_EXCLUSIVE",
    "PROJECTION_MODEL_PARAMETERS",
    "PROJECTION_PARAMETERS",
    "PROJECTION_SIZE",
    "acquisition_audit",
]
