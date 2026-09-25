"""P05 weighted objectives and acquisition-consistency terms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import torch
from torch.nn import functional as F

from .p05_sampling import Observation, validate_rows

UNKNOWN_FAMILIES = frozenset(
    {"", "na", "n/a", "none", "unknown", "not_applicable", "unspecified"}
)

_TINY = 1e-12
_POSITIVE_WEIGHT_CEILING = 2.5


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
    available: bool
    reason: str | None = None
    counts: dict[str, int] = field(default_factory=dict)


def _require_observation(row: object) -> Observation:
    if not isinstance(row, Observation):
        raise TypeError("expected an Observation")
    return row


def normalized_family(substrate: str) -> str:
    """Casefolded, stripped substrate family label."""

    if not isinstance(substrate, str):
        raise TypeError("substrate must be a string")
    return substrate.strip().casefold()


def is_known_family(substrate: str) -> bool:
    return normalized_family(substrate) not in UNKNOWN_FAMILIES


def positive_weight_between(first: Observation, second: Observation) -> float:
    """Protocol positive weight for an ordered anchor/positive pair."""

    first = _require_observation(first)
    second = _require_observation(second)
    if first.uid == second.uid:
        raise ValueError("positive pairs require two distinct rows")
    same_master = first.master == second.master
    same_instrument = first.instrument == second.instrument
    if same_master and same_instrument:
        raise ValueError(
            "same-master/same-instrument positives are not supported by the sampler"
        )
    if same_master:
        weight = 2.0
    elif not same_instrument:
        weight = 1.5
    else:
        weight = 1.0
    family_a = normalized_family(first.substrate)
    family_b = normalized_family(second.substrate)
    if (
        family_a not in UNKNOWN_FAMILIES
        and family_b not in UNKNOWN_FAMILIES
        and family_a != family_b
    ):
        weight *= 1.25
    return min(weight, _POSITIVE_WEIGHT_CEILING)


def _require_float_matrix(
    tensor: object, name: str, *, minimum_columns: int = 1
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if tensor.shape[1] < minimum_columns:
        raise ValueError(f"{name} must contain at least {minimum_columns} column(s)")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _require_positive_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _clean_labels(
    labels: object, row_count: int, class_count: int, device: torch.device
) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        tensor = labels
    else:
        tensor = torch.as_tensor(labels)
    if tensor.dim() != 1 or tensor.shape[0] != row_count:
        raise ValueError("labels must be one-dimensional with one entry per row")
    if (
        tensor.dtype == torch.bool
        or tensor.dtype.is_floating_point
        or tensor.dtype.is_complex
    ):
        raise TypeError("labels must be integer class indices")
    tensor = tensor.to(device=device, dtype=torch.long)
    if int(tensor.min()) < 0 or int(tensor.max()) >= class_count:
        raise ValueError("labels must lie in [0, class_count)")
    return tensor


def _row_weights(
    weights: object, row_count: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if isinstance(weights, torch.Tensor):
        tensor = weights
    else:
        tensor = torch.as_tensor(weights)
    if tensor.dim() != 1 or tensor.shape[0] != row_count:
        raise ValueError("weights must be one-dimensional with one entry per row")
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float64)
    tensor = tensor.to(dtype=dtype, device=device)
    if not torch.isfinite(tensor).all():
        raise ValueError("weights must be finite")
    if not bool((tensor > 0).all()):
        raise ValueError("weights must be strictly positive")
    total = tensor.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise ValueError("weights must have a finite positive sum")
    return tensor


def _validate_objective_rows(
    rows: object, expected_length: int
) -> list[Observation]:
    materialized = list(rows)
    if len(materialized) != expected_length:
        raise ValueError("rows must have one entry per tensor row")
    observations = [_require_observation(row) for row in materialized]
    validate_rows(observations)
    seen_views: set[tuple[str, str]] = set()
    for observation in observations:
        view = (observation.master, observation.instrument)
        if view in seen_views:
            raise ValueError(
                "same-master/same-instrument duplicates are not valid objective inputs"
            )
        seen_views.add(view)
    return observations


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return (reference * 0.0).sum()


def weighted_cross_entropy(
    logits: torch.Tensor, labels: object, weights: object
) -> torch.Tensor:
    """Weighted sum of unreduced per-row cross-entropy.

    Row weights must be finite and strictly positive; they are normalized to
    sum to one before the weighted sum.
    """

    logits = _require_float_matrix(logits, "logits", minimum_columns=2)
    row_count, class_count = logits.shape
    labels = _clean_labels(labels, row_count, class_count, logits.device)
    weights = _row_weights(
        weights, row_count, dtype=logits.dtype, device=logits.device
    )
    normalized = weights / weights.sum()
    log_probabilities = F.log_softmax(logits, dim=-1)
    picked = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    return -(normalized * picked).sum()


def supervised_contrastive(
    projections: torch.Tensor,
    rows: object,
    weights: object,
    *,
    temperature: float = 0.1,
    epsilon: float = 1e-8,
) -> LossResult:
    """Supervised contrastive loss with protocol positive weighting."""

    projections = _require_float_matrix(projections, "projections")
    _require_positive_scalar("temperature", temperature)
    _require_positive_scalar("epsilon", epsilon)
    row_count = projections.shape[0]
    observations = _validate_objective_rows(rows, row_count)
    weight_tensor = _row_weights(
        weights, row_count, dtype=projections.dtype, device=projections.device
    )

    device = projections.device
    dtype = projections.dtype
    eye = torch.eye(row_count, dtype=torch.bool, device=device)
    positives = torch.zeros((row_count, row_count), dtype=torch.bool, device=device)
    positive_weights = torch.zeros((row_count, row_count), dtype=dtype, device=device)
    for anchor in range(row_count):
        for other in range(row_count):
            if anchor == other:
                continue
            if observations[anchor].target == observations[other].target:
                positives[anchor, other] = True
                positive_weights[anchor, other] = positive_weight_between(
                    observations[anchor], observations[other]
                )

    positive_pairs = int(positives.sum())
    negative_pairs = int((~positives & ~eye).sum())
    eligible = positives.any(dim=1)
    eligible_anchors = int(eligible.sum())
    counts = {
        "eligible_anchors": eligible_anchors,
        "zero_positive_anchors": row_count - eligible_anchors,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }
    if positive_pairs == 0:
        return LossResult(_differentiable_zero(projections), False, "no_positives", counts)
    if negative_pairs == 0:
        return LossResult(_differentiable_zero(projections), False, "no_negatives", counts)

    embeddings = F.normalize(projections, dim=-1, eps=epsilon)
    similarity = embeddings @ embeddings.transpose(0, 1) / temperature
    masked = similarity.masked_fill(eye, float("-inf"))
    log_denominator = torch.logsumexp(masked, dim=1)
    log_probability = similarity - log_denominator.unsqueeze(1)
    safe_log_probability = torch.where(
        positives, log_probability, torch.zeros_like(log_probability)
    )
    positive_mass = positive_weights.sum(dim=1, keepdim=True)
    normalized_positive = positive_weights / positive_mass.clamp_min(_TINY)
    per_anchor = -(normalized_positive * safe_log_probability).sum(dim=1)
    anchor_weights = weight_tensor * eligible.to(dtype)
    loss = (anchor_weights * per_anchor).sum() / anchor_weights.sum()
    return LossResult(loss, True, None, counts)


def paired_consistency(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    rows: object,
    *,
    epsilon: float = 1e-8,
) -> LossResult:
    """Symmetric-KL plus cosine consistency over same-master instrument pairs."""

    logits = _require_float_matrix(logits, "logits", minimum_columns=2)
    embeddings = _require_float_matrix(embeddings, "embeddings")
    _require_positive_scalar("epsilon", epsilon)
    if logits.shape[0] != embeddings.shape[0]:
        raise ValueError("logits and embeddings must agree on row count")
    if logits.device != embeddings.device:
        raise ValueError("logits and embeddings must share a device")
    if logits.dtype != embeddings.dtype:
        raise ValueError("logits and embeddings must share a dtype")
    observations = _validate_objective_rows(rows, logits.shape[0])

    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, observation in enumerate(observations):
        grouped.setdefault(observation.master, []).append(
            (observation.instrument, index)
        )

    eligible_masters = 0
    pair_count = 0
    master_losses: list[torch.Tensor] = []
    for master in sorted(grouped):
        views = sorted(grouped[master], key=lambda item: item[0])
        if len(views) < 2:
            continue
        eligible_masters += 1
        losses: list[torch.Tensor] = []
        for (_, first_index), (_, second_index) in combinations(views, 2):
            first_log_probability = F.log_softmax(logits[first_index], dim=-1)
            second_log_probability = F.log_softmax(logits[second_index], dim=-1)
            first_probability = first_log_probability.exp()
            second_probability = second_log_probability.exp()
            forward_kl = (
                first_probability * (first_log_probability - second_log_probability)
            ).sum()
            reverse_kl = (
                second_probability * (second_log_probability - first_log_probability)
            ).sum()
            symmetric_kl = 0.5 * (forward_kl + reverse_kl)
            first_embedding = F.normalize(
                embeddings[first_index], dim=-1, eps=epsilon
            )
            second_embedding = F.normalize(
                embeddings[second_index], dim=-1, eps=epsilon
            )
            cosine = (first_embedding * second_embedding).sum()
            losses.append(0.5 * symmetric_kl + 0.5 * (1.0 - cosine))
        pair_count += len(losses)
        master_losses.append(torch.stack(losses).mean())

    counts = {"eligible_masters": eligible_masters, "pairs": pair_count}
    if eligible_masters == 0:
        zero = _differentiable_zero(logits) + _differentiable_zero(embeddings)
        return LossResult(zero, False, "no_pairs", counts)
    loss = torch.stack(master_losses).mean()
    return LossResult(loss, True, None, counts)


__all__ = [
    "LossResult",
    "UNKNOWN_FAMILIES",
    "is_known_family",
    "normalized_family",
    "paired_consistency",
    "positive_weight_between",
    "supervised_contrastive",
    "weighted_cross_entropy",
]
