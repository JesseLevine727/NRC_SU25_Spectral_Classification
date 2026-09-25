"""Bounded training-only numerical smoke kernel for the P05 core ladder.

This module owns the single in-memory fit used by the numerical smoke.  It
validates a source-fitting-only role, reuses the frozen P05 sampler and the
frozen objective implementations together with the unchanged P04 augmentation
and metric helpers, and returns an honest terminal record.  It performs no
file I/O, no plotting, no calibration, no model selection, no public export
and no retries.  Held validation/test arrays never enter the kernel.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from atlas_sers.governance.canonical import canonical_json_bytes
from atlas_sers.models.acquisition import (
    BASE_PARAMETERS,
    MAXIMUM_PARAMETERS_EXCLUSIVE,
    PROJECTION_MODEL_PARAMETERS,
    AcquisitionClassifier,
)

from .p04_runtime import (
    _augment,
    _metric_values,
    _noise_quantiles,
    _seed_from,
    _state_hash,
)
from .p05_objectives import (
    paired_consistency,
    supervised_contrastive,
    weighted_cross_entropy,
)
from .p05_sampling import Observation, sample_master_views, validate_rows

RECIPE_SPECIFICATIONS: dict[str, tuple[float, float, bool]] = {
    "D0-M": (0.0, 0.0, False),
    "D1": (0.3, 0.0, True),
    "D2": (0.0, 0.3, False),
    "D3": (0.3, 0.3, True),
}

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
MAXIMUM_EPOCHS = 8
MAXIMUM_BATCHES_PER_EPOCH = 4
MAXIMUM_OPTIMIZER_STEPS = 32
DEFAULT_FIT_SECONDS = 120.0
MAXIMUM_FIT_SECONDS = 120.0
DEFAULT_CUDA_BYTES = 4294967296
EVALUATION_CHUNK = 64
MINMAX_TOLERANCE = 1e-3

TERMINAL_COMPLETE = "complete"
TERMINAL_NUMERICAL = "numerical_failure"
TERMINAL_RESOURCE = "resource_failure"
TERMINAL_FIT = "fit_failure"

_NOISE_COLUMNS = ("observation_uid", "first_difference_noise_mad", "intensity_range")
_CUBLAS_ENV_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
_ALLOWED_CUBLAS_WORKSPACE_CONFIG = frozenset({":4096:8"})


@dataclass
class SmokeFitResult:
    """Terminal record of one bounded smoke fit, success or retained failure."""

    status: str
    reason_code: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    parameter_count: int = 0
    initial_state_digest: str | None = None
    final_state_digest: str | None = None
    initial_backbone_digest: str | None = None
    final_backbone_digest: str | None = None
    initial_head_digest: str | None = None
    final_head_digest: str | None = None
    state_dict: dict[str, torch.Tensor] | None = None
    augmentation_digest: str | None = None
    sampling_digest: str | None = None
    pair_digest: str | None = None
    optimizer_steps: int = 0
    elapsed_seconds: float = 0.0
    peak_cuda_bytes: int = 0
    finite_gradient_batches: int = 0
    nonzero_gradient_elements: int = 0
    supcon_support: dict[str, int] = field(default_factory=dict)
    paired_support: dict[str, int] = field(default_factory=dict)
    traceback_digest: str | None = None


class _StopFit(Exception):
    """Internal control-flow signal carrying a sanitized terminal reason."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _require_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _safe_class(exc: BaseException) -> str:
    cleaned = "".join(ch for ch in type(exc).__name__ if ch.isalnum())
    return cleaned or "Exception"


def _traceback_digest(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return hashlib.sha256(canonical_json_bytes([lines])).hexdigest()


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().to("cpu").clone() for name, value in model.state_dict().items()}


def _state_digest(state: dict[str, torch.Tensor], prefix: str) -> str | None:
    subset = {name: tensor for name, tensor in state.items() if name.startswith(prefix)}
    if not subset:
        return None
    return _state_hash(subset)


def _grad_norm(gradients: list[torch.Tensor]) -> float:
    total = 0.0
    for gradient in gradients:
        total += float(gradient.detach().norm() ** 2)
    return math.sqrt(total)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _eligible_pairs(rows: list[Observation]) -> list[tuple[str, str]]:
    """Sorted same-chemical positive UID pairs excluding same-master/instrument.

    This is the SupCon positive set for one sampled batch, computed from the
    sampler output alone so the pair digest is recipe-independent.
    """

    pairs: set[tuple[str, str]] = set()
    for index in range(len(rows)):
        for other in range(index + 1, len(rows)):
            first = rows[index]
            second = rows[other]
            if first.target != second.target:
                continue
            if first.master == second.master and first.instrument == second.instrument:
                continue
            pairs.add(tuple(sorted((first.uid, second.uid))))
    return sorted(pairs)


def _prepare_inputs(
    values: object,
    observations: object,
    noise_metadata: object,
    *,
    epochs: object,
    batches_per_epoch: object,
) -> tuple[np.ndarray, list[Observation], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(epochs, int) or isinstance(epochs, bool):
        raise TypeError("epochs must be an integer")
    if not 1 <= epochs <= MAXIMUM_EPOCHS:
        raise ValueError("epochs must lie in [1, 8]")
    if not isinstance(batches_per_epoch, int) or isinstance(batches_per_epoch, bool):
        raise TypeError("batches_per_epoch must be an integer")
    if not 1 <= batches_per_epoch <= MAXIMUM_BATCHES_PER_EPOCH:
        raise ValueError("batches_per_epoch must lie in [1, 4]")
    if epochs * batches_per_epoch > MAXIMUM_OPTIMIZER_STEPS:
        raise ValueError("schedule may not exceed 32 optimizer steps")

    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy array")
    if values.dtype != np.float32:
        raise ValueError("values must have dtype float32")
    matrix = np.ascontiguousarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != 1401:
        raise ValueError("values must be a (N, 1401) matrix")
    if matrix.shape[0] < 1:
        raise ValueError("values must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("values must be finite")
    row_min = matrix.min(axis=1)
    row_max = matrix.max(axis=1)
    if not np.all(np.abs(row_min) <= MINMAX_TOLERANCE):
        raise ValueError("each row must have minimum near 0")
    if not np.all(np.abs(row_max - 1.0) <= MINMAX_TOLERANCE):
        raise ValueError("each row must have maximum near 1")

    rows = list(observations)
    if len(rows) != matrix.shape[0]:
        raise ValueError("observations must align one-to-one with values rows")
    validate_rows(rows)
    ordered = [row.uid for row in rows]
    if ordered != sorted(ordered):
        raise ValueError("observations must be supplied in canonical sorted-UID order")
    targets = {row.target for row in rows}
    if len(targets) != 3:
        raise ValueError("fitting roles must contain exactly three chemicals")

    if not isinstance(noise_metadata, pd.DataFrame):
        raise TypeError("noise_metadata must be a pandas frame")
    missing = [name for name in _NOISE_COLUMNS if name not in noise_metadata.columns]
    if missing:
        raise ValueError(f"noise_metadata misses required columns: {missing}")
    if len(noise_metadata) != len(rows):
        raise ValueError("noise_metadata must have one row per observation")
    frame = noise_metadata.reset_index(drop=True)
    if frame.observation_uid.astype(str).tolist() != ordered:
        raise ValueError("noise_metadata UID order must match sorted observations")
    noise = frame.first_difference_noise_mad.to_numpy(dtype=float)
    span = frame.intensity_range.to_numpy(dtype=float)
    if not (np.isfinite(noise).all() and bool((noise >= 0).all())):
        raise ValueError("noise metadata must be finite and nonnegative")
    if not (np.isfinite(span).all() and bool((span > 0).all())):
        raise ValueError("intensity ranges must be finite and positive")
    noise_levels = _noise_quantiles(frame)

    classes = tuple(sorted(targets))
    lookup = {label: index for index, label in enumerate(classes)}
    labels = np.asarray([lookup[row.target] for row in rows], dtype=np.int64)
    if (np.bincount(labels, minlength=len(classes)) == 0).any():
        raise ValueError("fitting roles must exercise all three chemicals")
    uids = np.asarray(ordered)
    return matrix, rows, classes, labels, noise_levels, uids


def _evaluate(
    model: torch.nn.Module,
    values: torch.Tensor,
    labels: np.ndarray,
    classes: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, float], float, float]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    embedding_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, values.shape[0], EVALUATION_CHUNK):
            batch = values[start : start + EVALUATION_CHUNK].to(device)
            logits, embedding, _ = model(batch)
            logits_chunks.append(logits.detach().to("cpu"))
            embedding_chunks.append(embedding.detach().to("cpu"))
    logits = torch.cat(logits_chunks, dim=0).numpy().astype(np.float64)
    embeddings = torch.cat(embedding_chunks, dim=0).numpy().astype(np.float64)
    if not (np.isfinite(logits).all() and np.isfinite(embeddings).all()):
        raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_evaluation")
    metrics = _metric_values(labels, logits, classes)
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_metric")
    variance = float(embeddings.var(axis=0).mean())
    norm_mean = float(np.linalg.norm(embeddings, axis=1).mean())
    if not (math.isfinite(variance) and math.isfinite(norm_mean)):
        raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_embedding_summary")
    return metrics, variance, norm_mean


def train_smoke_fit(
    *,
    values: np.ndarray,
    observations: list[Observation],
    noise_metadata: pd.DataFrame,
    role_id: str,
    recipe: str,
    seed: int,
    device: str | torch.device,
    epochs: int = MAXIMUM_EPOCHS,
    batches_per_epoch: int = MAXIMUM_BATCHES_PER_EPOCH,
    maximum_fit_seconds: float = DEFAULT_FIT_SECONDS,
    global_deadline: float | None = None,
    maximum_cuda_allocated_bytes: int = DEFAULT_CUDA_BYTES,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> SmokeFitResult:
    started = time.perf_counter()

    if not isinstance(role_id, str):
        raise TypeError("role_id must be a string")
    if not role_id.strip() or role_id != role_id.strip():
        raise ValueError("role_id must be a non-empty, unpadded string")
    if recipe not in RECIPE_SPECIFICATIONS:
        raise ValueError("recipe must be one of D0-M, D1, D2, D3")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    maximum_fit_seconds = _require_finite_number("maximum_fit_seconds", maximum_fit_seconds)
    if not 0.0 < maximum_fit_seconds <= MAXIMUM_FIT_SECONDS:
        raise ValueError("maximum_fit_seconds must lie in (0, 120]")
    if global_deadline is not None:
        global_deadline = _require_finite_number("global_deadline", global_deadline)
    if not isinstance(maximum_cuda_allocated_bytes, int) or isinstance(
        maximum_cuda_allocated_bytes, bool
    ):
        raise TypeError("maximum_cuda_allocated_bytes must be an integer")
    if maximum_cuda_allocated_bytes <= 0:
        raise ValueError("maximum_cuda_allocated_bytes must be positive")
    if maximum_cuda_allocated_bytes > DEFAULT_CUDA_BYTES:
        raise ValueError("maximum_cuda_allocated_bytes may not exceed 4 GiB")
    if on_epoch is not None and not callable(on_epoch):
        raise TypeError("on_epoch must be callable or None")

    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        raise ValueError("device must be a valid 'cpu' or 'cuda' device") from None
    if torch_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    lambda_supcon, lambda_pair, use_projection = RECIPE_SPECIFICATIONS[recipe]
    matrix, rows, classes, labels, noise_levels, uids = _prepare_inputs(
        values,
        observations,
        noise_metadata,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
    )

    evaluation_tensor = torch.from_numpy(np.ascontiguousarray(matrix[:, None, :]))
    sampling_hasher = hashlib.sha256()
    augmentation_hasher = hashlib.sha256()
    pair_hasher = hashlib.sha256()

    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    finite_gradient_batches = 0
    nonzero_gradient_elements = 0
    parameter_count = 0
    initial_state: dict[str, torch.Tensor] | None = None
    terminal_state: dict[str, torch.Tensor] | None = None
    supcon_support: dict[str, int] = {
        "enabled": int(lambda_supcon > 0.0),
        "available_batches": 0,
        "eligible_anchors": 0,
        "zero_positive_anchors": 0,
    }
    paired_support: dict[str, int] = {
        "enabled": int(lambda_pair > 0.0),
        "available_batches": 0,
        "eligible_masters": 0,
        "pairs": 0,
    }

    def _peak_cuda_bytes() -> int:
        if torch_device.type != "cuda":
            return 0
        try:
            if not torch.cuda.is_available():
                return 0
            return int(torch.cuda.max_memory_allocated(torch_device))
        except Exception:
            return 0

    def _finalize(
        status: str, reason_code: str | None, traceback_digest: str | None = None
    ) -> SmokeFitResult:
        state = terminal_state if terminal_state is not None else initial_state
        if state is None:
            state_dict = None
            final_state_digest = None
            final_backbone_digest = None
            final_head_digest = None
        else:
            state_dict = {name: value.clone() for name, value in state.items()}
            final_state_digest = _state_hash(state)
            final_backbone_digest = _state_digest(state, "backbone.")
            final_head_digest = _state_digest(state, "projection.")
        return SmokeFitResult(
            status=status,
            reason_code=reason_code,
            history=list(history),
            parameter_count=parameter_count,
            initial_state_digest=None if initial_state is None else _state_hash(initial_state),
            final_state_digest=final_state_digest,
            initial_backbone_digest=(
                None if initial_state is None else _state_digest(initial_state, "backbone.")
            ),
            final_backbone_digest=final_backbone_digest,
            initial_head_digest=(
                None if initial_state is None else _state_digest(initial_state, "projection.")
            ),
            final_head_digest=final_head_digest,
            state_dict=state_dict,
            augmentation_digest=augmentation_hasher.hexdigest(),
            sampling_digest=sampling_hasher.hexdigest(),
            pair_digest=pair_hasher.hexdigest(),
            optimizer_steps=optimizer_steps,
            elapsed_seconds=float(time.perf_counter() - started),
            peak_cuda_bytes=_peak_cuda_bytes(),
            finite_gradient_batches=finite_gradient_batches,
            nonzero_gradient_elements=nonzero_gradient_elements,
            supcon_support=dict(supcon_support),
            paired_support=dict(paired_support),
            traceback_digest=traceback_digest,
        )

    phase = "setup"
    try:
        _configure_determinism()
        if torch_device.type == "cuda":
            if not torch.cuda.is_available():
                raise _StopFit(TERMINAL_RESOURCE, "cuda_unavailable")
            workspace = os.environ.get(_CUBLAS_ENV_VARIABLE, "")
            if workspace not in _ALLOWED_CUBLAS_WORKSPACE_CONFIG:
                raise _StopFit(TERMINAL_RESOURCE, "cublas_workspace_config_invalid")
            torch.cuda.reset_peak_memory_stats(torch_device)

        if time.perf_counter() - started > maximum_fit_seconds:
            raise _StopFit(TERMINAL_RESOURCE, "deadline_exceeded")
        if global_deadline is not None and time.perf_counter() > float(global_deadline):
            raise _StopFit(TERMINAL_RESOURCE, "global_deadline_exceeded")

        torch.manual_seed(_seed_from("initialization", role_id, seed))
        model = AcquisitionClassifier(class_count=len(classes), use_projection=use_projection)
        model.to(torch_device)
        torch.manual_seed(_seed_from("training", role_id, seed))

        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        expected = PROJECTION_MODEL_PARAMETERS if use_projection else BASE_PARAMETERS
        if parameter_count != expected:
            raise _StopFit(TERMINAL_FIT, "unexpected_parameter_count")
        if parameter_count >= MAXIMUM_PARAMETERS_EXCLUSIVE:
            raise _StopFit(TERMINAL_FIT, "parameter_ceiling_exceeded")
        if model.batch_normalization_modules() != 0:
            raise _StopFit(TERMINAL_FIT, "batch_normalization_present")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        initial_state = _clone_state(model)
        terminal_state = initial_state

        def _sync() -> None:
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

        def _check_deadline() -> None:
            now = time.perf_counter()
            if now - started > maximum_fit_seconds:
                raise _StopFit(TERMINAL_RESOURCE, "deadline_exceeded")
            if global_deadline is not None and now > float(global_deadline):
                raise _StopFit(TERMINAL_RESOURCE, "global_deadline_exceeded")

        def _check_memory() -> None:
            if torch_device.type != "cuda":
                return
            _sync()
            current = int(torch.cuda.memory_allocated(torch_device))
            peak = int(torch.cuda.max_memory_allocated(torch_device))
            if current > maximum_cuda_allocated_bytes or peak > maximum_cuda_allocated_bytes:
                raise _StopFit(TERMINAL_RESOURCE, "cuda_memory_exceeded")

        phase = "fit"
        for epoch in range(1, epochs + 1):
            ce_values: list[float] = []
            total_values: list[float] = []
            supcon_values: list[float] = []
            paired_values: list[float] = []
            gradient_norms: list[float] = []
            backbone_norms: list[float] = []
            head_norms: list[float] = []
            clipped: list[float] = []
            epoch_supcon = {"available": 0, "eligible": 0, "zero_positive": 0}
            epoch_paired = {"available": 0, "masters": 0, "pairs": 0}

            for ordinal in range(batches_per_epoch):
                _check_deadline()
                _check_memory()

                batch = sample_master_views(
                    rows, role_id=role_id, seed=seed, epoch=epoch, batch_ordinal=ordinal
                )
                indices = np.asarray(batch.indices, dtype=int)
                batch_rows = [rows[index] for index in indices]
                batch_values = matrix[indices]
                batch_uids = uids[indices]
                batch_labels = labels[indices]

                sampling_hasher.update(
                    canonical_json_bytes([role_id, epoch, ordinal, batch.draw_sha256])
                )
                pair_hasher.update(
                    canonical_json_bytes([role_id, epoch, ordinal, _eligible_pairs(batch_rows)])
                )
                augmentation_hasher.update(canonical_json_bytes([role_id, epoch, ordinal]))

                augmentation_rng = np.random.default_rng(
                    _seed_from("augment", role_id, seed, epoch, ordinal)
                )
                augmented = _augment(
                    batch_values,
                    batch_uids,
                    rng=augmentation_rng,
                    noise_levels=noise_levels,
                    digest=augmentation_hasher,
                )
                batch_tensor = torch.from_numpy(np.ascontiguousarray(augmented[:, None, :])).to(
                    torch_device
                )

                torch.manual_seed(_seed_from("batch", role_id, seed, epoch, ordinal))
                model.train()
                optimizer.zero_grad(set_to_none=True)

                logits, embedding, projection = model(batch_tensor)
                ce_loss = weighted_cross_entropy(logits, batch_labels, batch.weights)
                total_loss = ce_loss

                supcon_result = None
                if lambda_supcon > 0.0:
                    supcon_result = supervised_contrastive(projection, batch_rows, batch.weights)
                    total_loss = total_loss + lambda_supcon * supcon_result.loss
                paired_result = None
                if lambda_pair > 0.0:
                    paired_result = paired_consistency(logits, embedding, batch_rows)
                    total_loss = total_loss + lambda_pair * paired_result.loss

                _sync()
                if not bool(torch.isfinite(ce_loss).all()):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_loss")
                if supcon_result is not None and not bool(torch.isfinite(supcon_result.loss).all()):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_loss")
                if paired_result is not None and not bool(torch.isfinite(paired_result.loss).all()):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_loss")
                if not bool(torch.isfinite(total_loss).all()):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_loss")

                total_loss.backward()
                _sync()

                backbone_parameters = [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.grad is not None
                ]
                head_parameters = (
                    []
                    if model.projection is None
                    else [
                        parameter
                        for parameter in model.projection.parameters()
                        if parameter.grad is not None
                    ]
                )
                gradients = [parameter.grad for parameter in backbone_parameters]
                gradients += [parameter.grad for parameter in head_parameters]
                if not gradients:
                    raise _StopFit(TERMINAL_NUMERICAL, "missing_gradients")
                if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_gradient")
                nonzero = int(sum(int((gradient != 0).sum().item()) for gradient in gradients))
                if nonzero == 0:
                    raise _StopFit(TERMINAL_NUMERICAL, "zero_gradient")

                backbone_norm = _grad_norm([parameter.grad for parameter in backbone_parameters])
                head_norm = _grad_norm([parameter.grad for parameter in head_parameters])
                total_norm = math.sqrt(backbone_norm**2 + head_norm**2)
                if not math.isfinite(total_norm):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_gradient_norm")

                finite_gradient_batches += 1
                nonzero_gradient_elements += nonzero
                gradient_norms.append(total_norm)
                backbone_norms.append(backbone_norm)
                head_norms.append(head_norm)
                clipped.append(1.0 if total_norm > GRADIENT_CLIP_NORM else 0.0)

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

                optimizer_steps += 1
                for parameter in model.parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_parameter")

                terminal_state = _clone_state(model)

                ce_values.append(float(ce_loss.detach()))
                total_values.append(float(total_loss.detach()))
                if supcon_result is not None:
                    supcon_values.append(float(supcon_result.loss.detach()))
                    eligible = int(supcon_result.counts.get("eligible_anchors", 0))
                    zero_positive = int(supcon_result.counts.get("zero_positive_anchors", 0))
                    epoch_supcon["eligible"] += eligible
                    epoch_supcon["zero_positive"] += zero_positive
                    supcon_support["eligible_anchors"] += eligible
                    supcon_support["zero_positive_anchors"] += zero_positive
                    if supcon_result.available:
                        epoch_supcon["available"] += 1
                        supcon_support["available_batches"] += 1
                if paired_result is not None:
                    paired_values.append(float(paired_result.loss.detach()))
                    masters = int(paired_result.counts.get("eligible_masters", 0))
                    pairs = int(paired_result.counts.get("pairs", 0))
                    epoch_paired["masters"] += masters
                    epoch_paired["pairs"] += pairs
                    paired_support["eligible_masters"] += masters
                    paired_support["pairs"] += pairs
                    if paired_result.available:
                        epoch_paired["available"] += 1
                        paired_support["available_batches"] += 1

                _check_memory()
                _check_deadline()

            metrics, embedding_variance, embedding_norm_mean = _evaluate(
                model, evaluation_tensor, labels, classes, torch_device
            )
            _check_deadline()
            _check_memory()

            record: dict[str, Any] = {
                "epoch": epoch,
                "chemical_ce": _mean(ce_values),
                "total_loss": _mean(total_values),
                "supcon_loss": _mean(supcon_values),
                "paired_loss": _mean(paired_values),
                "supcon_enabled": bool(lambda_supcon > 0.0),
                "paired_enabled": bool(lambda_pair > 0.0),
                "supcon_available_batches": int(epoch_supcon["available"]),
                "paired_available_batches": int(epoch_paired["available"]),
                "eligible_anchor_count": int(epoch_supcon["eligible"]),
                "zero_positive_anchor_count": int(epoch_supcon["zero_positive"]),
                "paired_master_count": int(epoch_paired["masters"]),
                "gradient_norm_mean": _mean(gradient_norms),
                "gradient_norm_max": float(max(gradient_norms)) if gradient_norms else 0.0,
                "head_gradient_norm_mean": _mean(head_norms),
                "backbone_gradient_norm_mean": _mean(backbone_norms),
                "clipped_fraction": _mean(clipped),
                "embedding_variance": float(embedding_variance),
                "embedding_norm_mean": float(embedding_norm_mean),
                "train_ba": float(metrics["balanced_accuracy"]),
                "train_nll": float(metrics["negative_log_likelihood"]),
                "train_predicted_class_count": int(metrics["predicted_class_count"]),
                "optimizer_steps": int(optimizer_steps),
            }
            history.append(record)
            if on_epoch is not None:
                on_epoch(dict(record))
            _check_deadline()
            _check_memory()

        _check_deadline()
        _check_memory()
    except _StopFit as stop:
        return _finalize(stop.status, stop.reason)
    except FloatingPointError as exc:
        return _finalize(
            TERMINAL_NUMERICAL, "augmentation_numerical_failure", _traceback_digest(exc)
        )
    except Exception as exc:  # noqa: BLE001 - sanitized into a terminal result
        return _finalize(TERMINAL_FIT, f"{phase}_{_safe_class(exc)}", _traceback_digest(exc))

    return _finalize(TERMINAL_COMPLETE, None)


__all__ = [
    "DEFAULT_CUDA_BYTES",
    "DEFAULT_FIT_SECONDS",
    "EVALUATION_CHUNK",
    "GRADIENT_CLIP_NORM",
    "LEARNING_RATE",
    "MAXIMUM_BATCHES_PER_EPOCH",
    "MAXIMUM_EPOCHS",
    "MAXIMUM_FIT_SECONDS",
    "MAXIMUM_OPTIMIZER_STEPS",
    "MINMAX_TOLERANCE",
    "RECIPE_SPECIFICATIONS",
    "SmokeFitResult",
    "TERMINAL_COMPLETE",
    "TERMINAL_FIT",
    "TERMINAL_NUMERICAL",
    "TERMINAL_RESOURCE",
    "WEIGHT_DECAY",
    "train_smoke_fit",
]
