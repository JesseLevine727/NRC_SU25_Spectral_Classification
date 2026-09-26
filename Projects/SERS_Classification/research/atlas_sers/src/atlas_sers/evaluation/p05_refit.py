"""Source-only refit training kernel for the P05 core ladder.

This module owns the single in-memory refit used once the P05 development
schedule has fixed a recipe and epoch budget.  It fits a source-fitting-only
role for an exact, pre-declared number of epochs, reusing the frozen P05
sampler, objectives, augmentation, architecture, optimizer and initialization
exactly as the development kernel does.  It accepts no validation or test
inputs, performs no early stopping and no best-checkpoint selection, and
returns an honest terminal record.  It performs no file I/O, no calibration,
no outer predictions and no scientific launch.  Held validation/test arrays
never enter the kernel.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
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

from .p04_runtime import _augment, _seed_from, _state_hash
from .p05_development import BATCHES_PER_EPOCH, _prepare_noise, _prepare_role
from .p05_objectives import (
    paired_consistency,
    supervised_contrastive,
    weighted_cross_entropy,
)
from .p05_sampling import Observation, sample_master_views
from .p05_smoke import (
    _ALLOWED_CUBLAS_WORKSPACE_CONFIG,
    _CUBLAS_ENV_VARIABLE,
    DEFAULT_CUDA_BYTES,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAXIMUM_FIT_SECONDS,
    RECIPE_SPECIFICATIONS,
    TERMINAL_COMPLETE,
    TERMINAL_FIT,
    TERMINAL_NUMERICAL,
    TERMINAL_RESOURCE,
    WEIGHT_DECAY,
    _clone_state,
    _configure_determinism,
    _eligible_pairs,
    _grad_norm,
    _mean,
    _require_finite_number,
    _safe_class,
    _state_digest,
    _StopFit,
    _traceback_digest,
)

MINIMUM_REFIT_EPOCHS = 30
MAXIMUM_REFIT_EPOCHS = 200


@dataclass
class RefitResult:
    """Terminal record of one source-only refit, success or failure."""

    status: str
    reason_code: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    epochs: int = 0
    epochs_completed: int = 0
    parameter_count: int = 0
    optimizer_steps: int = 0
    zero_gradient_batches: int = 0
    initial_state_digest: str | None = None
    terminal_state_digest: str | None = None
    initial_backbone_digest: str | None = None
    terminal_backbone_digest: str | None = None
    initial_head_digest: str | None = None
    terminal_head_digest: str | None = None
    terminal_state_dict: dict[str, torch.Tensor] | None = None
    state_capture_failed: bool = False
    classes: tuple[str, ...] = ()
    source_noise_levels: tuple[float, ...] = ()
    augmentation_digest: str | None = None
    sampling_digest: str | None = None
    pair_digest: str | None = None
    finite_gradient_batches: int = 0
    nonzero_gradient_elements: int = 0
    supcon_support: dict[str, int] = field(default_factory=dict)
    paired_support: dict[str, int] = field(default_factory=dict)
    role_id: str | None = None
    recipe: str | None = None
    seed: int | None = None
    elapsed_seconds: float = 0.0
    peak_cuda_bytes: int = 0
    traceback_digest: str | None = None


def train_refit(
    *,
    values: np.ndarray,
    observations: list[Observation],
    noise_metadata: pd.DataFrame,
    role_id: str,
    recipe: str,
    seed: int,
    epochs: int,
    device: str | torch.device,
    maximum_fit_seconds: float = 120.0,
    global_deadline: float | None = None,
    maximum_cuda_allocated_bytes: int = DEFAULT_CUDA_BYTES,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> RefitResult:
    """Run one bounded, source-only refit for an exact number of epochs."""

    started = time.perf_counter()

    if not isinstance(role_id, str):
        raise TypeError("role_id must be a string")
    if not role_id.strip() or role_id != role_id.strip():
        raise ValueError("role_id must be a non-empty, unpadded string")
    if recipe not in RECIPE_SPECIFICATIONS:
        raise ValueError("recipe must be one of D0-M, D1, D2, D3")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(epochs, int) or isinstance(epochs, bool):
        raise TypeError("epochs must be an integer")
    if not MINIMUM_REFIT_EPOCHS <= epochs <= MAXIMUM_REFIT_EPOCHS:
        raise ValueError("epochs must lie in [30, 200]")
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

    fit_matrix, fit_rows, fit_classes, fit_labels, fit_uids = _prepare_role(
        values, observations, name="fitting"
    )
    noise_levels = _prepare_noise(noise_metadata, fit_uids)
    sample_master_views(fit_rows, role_id=role_id, seed=seed, epoch=1, batch_ordinal=0)

    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    zero_gradient_batches = 0
    finite_gradient_batches = 0
    nonzero_gradient_elements = 0
    parameter_count = 0
    initial_state: dict[str, torch.Tensor] | None = None
    terminal_state: dict[str, torch.Tensor] | None = None
    model: torch.nn.Module | None = None
    sampling_hasher = hashlib.sha256()
    augmentation_hasher = hashlib.sha256()
    pair_hasher = hashlib.sha256()
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

    capture_failed = False

    def _base_result(
        status: str,
        reason_code: str | None,
        traceback_digest: str | None = None,
        *,
        terminal_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> RefitResult:
        return RefitResult(
            status=status,
            reason_code=reason_code,
            history=list(history),
            epochs=epochs,
            epochs_completed=len(history),
            parameter_count=parameter_count,
            optimizer_steps=optimizer_steps,
            zero_gradient_batches=zero_gradient_batches,
            initial_state_digest=None if initial_state is None else _state_hash(initial_state),
            terminal_state_digest=(None if terminal_state is None else _state_hash(terminal_state)),
            initial_backbone_digest=(
                None if initial_state is None else _state_digest(initial_state, "backbone.")
            ),
            terminal_backbone_digest=(
                None if terminal_state is None else _state_digest(terminal_state, "backbone.")
            ),
            initial_head_digest=(
                None if initial_state is None else _state_digest(initial_state, "projection.")
            ),
            terminal_head_digest=(
                None if terminal_state is None else _state_digest(terminal_state, "projection.")
            ),
            terminal_state_dict=terminal_state_dict,
            state_capture_failed=capture_failed,
            classes=tuple(fit_classes),
            source_noise_levels=tuple(float(value) for value in noise_levels.tolist()),
            augmentation_digest=augmentation_hasher.hexdigest(),
            sampling_digest=sampling_hasher.hexdigest(),
            pair_digest=pair_hasher.hexdigest(),
            finite_gradient_batches=finite_gradient_batches,
            nonzero_gradient_elements=nonzero_gradient_elements,
            supcon_support=dict(supcon_support),
            paired_support=dict(paired_support),
            role_id=role_id,
            recipe=recipe,
            seed=seed,
            elapsed_seconds=float(time.perf_counter() - started),
            peak_cuda_bytes=_peak_cuda_bytes(),
            traceback_digest=traceback_digest,
        )

    def _capture_terminal() -> dict[str, torch.Tensor] | None:
        nonlocal terminal_state, capture_failed
        try:
            if model is not None:
                terminal_state = _clone_state(model)
            return (
                None
                if terminal_state is None
                else {name: value.clone() for name, value in terminal_state.items()}
            )
        except Exception:
            capture_failed = True
            return None if terminal_state is None else dict(terminal_state)

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

        def _sync() -> None:
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

        def _check_deadline() -> None:
            _sync()
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

        _check_deadline()
        _check_memory()

        torch.manual_seed(_seed_from("initialization", role_id, seed))
        model = AcquisitionClassifier(class_count=len(fit_classes), use_projection=use_projection)
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

        _check_deadline()
        _check_memory()

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
            epoch_zero_gradient = 0
            epoch_steps = 0

            for ordinal in range(BATCHES_PER_EPOCH):
                _check_deadline()
                _check_memory()

                batch = sample_master_views(
                    fit_rows, role_id=role_id, seed=seed, epoch=epoch, batch_ordinal=ordinal
                )
                indices = np.asarray(batch.indices, dtype=int)
                batch_rows = [fit_rows[index] for index in indices]
                batch_values = fit_matrix[indices]
                batch_uids = fit_uids[indices]
                batch_labels = fit_labels[indices]

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

                backbone_norm = _grad_norm([parameter.grad for parameter in backbone_parameters])
                head_norm = _grad_norm([parameter.grad for parameter in head_parameters])
                total_norm = math.sqrt(backbone_norm**2 + head_norm**2)
                if not math.isfinite(total_norm):
                    raise _StopFit(TERMINAL_NUMERICAL, "nonfinite_gradient_norm")

                finite_gradient_batches += 1
                nonzero_gradient_elements += nonzero
                if nonzero == 0:
                    zero_gradient_batches += 1
                    epoch_zero_gradient += 1
                gradient_norms.append(total_norm)
                backbone_norms.append(backbone_norm)
                head_norms.append(head_norm)
                clipped.append(1.0 if total_norm > GRADIENT_CLIP_NORM else 0.0)

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
                optimizer_steps += 1
                epoch_steps += 1

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
                "paired_pair_count": int(epoch_paired["pairs"]),
                "gradient_norm_mean": _mean(gradient_norms),
                "gradient_norm_max": float(max(gradient_norms)) if gradient_norms else 0.0,
                "head_gradient_norm_mean": _mean(head_norms),
                "backbone_gradient_norm_mean": _mean(backbone_norms),
                "clipped_fraction": _mean(clipped),
                "zero_gradient_batches": int(epoch_zero_gradient),
                "epoch_optimizer_steps": int(epoch_steps),
                "total_optimizer_steps": int(optimizer_steps),
                "sampling_digest": sampling_hasher.hexdigest(),
                "augmentation_digest": augmentation_hasher.hexdigest(),
                "pair_digest": pair_hasher.hexdigest(),
            }
            history.append(record)
            if on_epoch is not None:
                phase = "callback"
                on_epoch(dict(record))
                phase = "fit"
            _check_deadline()
            _check_memory()

        phase = "finalize"
        _check_deadline()
        _check_memory()
        terminal_clone = _capture_terminal()
        result = _base_result(TERMINAL_COMPLETE, None, terminal_state_dict=terminal_clone)
        _check_deadline()
        _check_memory()
        result.elapsed_seconds = float(time.perf_counter() - started)
        return result
    except _StopFit as stop:
        return _base_result(
            stop.status,
            stop.reason,
            _traceback_digest(stop),
            terminal_state_dict=_capture_terminal(),
        )
    except FloatingPointError as exc:
        return _base_result(
            TERMINAL_NUMERICAL,
            "augmentation_numerical_failure",
            _traceback_digest(exc),
            terminal_state_dict=_capture_terminal(),
        )
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        return _base_result(
            TERMINAL_RESOURCE,
            "out_of_memory",
            _traceback_digest(exc),
            terminal_state_dict=_capture_terminal(),
        )
    except Exception as exc:  # noqa: BLE001 - sanitized into a terminal result
        return _base_result(
            TERMINAL_FIT,
            f"{phase}_{_safe_class(exc)}",
            _traceback_digest(exc),
            terminal_state_dict=_capture_terminal(),
        )


__all__ = [
    "MAXIMUM_REFIT_EPOCHS",
    "MINIMUM_REFIT_EPOCHS",
    "RefitResult",
    "train_refit",
]
