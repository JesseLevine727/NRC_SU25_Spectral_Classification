"""Deterministic training primitives for the locked P04 compact D0 model."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn

from atlas_sers.evaluation.classical import (
    apply_temperature,
    fit_temperature,
)
from atlas_sers.governance.canonical import canonical_json_bytes
from atlas_sers.models.deep import CompactSERSClassifier

TERMINAL_STATUSES = {
    "complete",
    "numerical_failure",
    "resource_failure",
    "data_failure",
    "fit_failure",
    "excluded_by_protocol",
}


@dataclass(frozen=True)
class P04Dataset:
    intensity: np.ndarray
    metadata: pd.DataFrame
    uid_to_index: dict[str, int]

    @classmethod
    def from_frozen_representation(
        cls,
        *,
        intensity: np.ndarray,
        representation_uids: np.ndarray,
        metadata: pd.DataFrame,
    ) -> P04Dataset:
        matrix = np.asarray(intensity, dtype=np.float32)
        uids = np.asarray(representation_uids).astype(str)
        frame = metadata.copy().reset_index(drop=True)
        required = {
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "station",
            "first_difference_noise_mad",
            "intensity_range",
        }
        if not required <= set(frame):
            raise ValueError(f"P04 metadata misses fields: {sorted(required - set(frame))}")
        if matrix.shape != (len(frame), 1401):
            raise ValueError("P04 expects a 1,401-channel row for every manifest record.")
        if not np.array_equal(uids, frame.observation_uid.astype(str).to_numpy()):
            raise ValueError("P04 representation UID order differs from its manifest.")
        if len(set(uids)) != len(uids) or not np.isfinite(matrix).all():
            raise ValueError("P04 representation UIDs or intensities are invalid.")
        return cls(matrix, frame, {uid: index for index, uid in enumerate(uids)})

    def subset(self, uids: pd.Series | list[str]) -> tuple[np.ndarray, pd.DataFrame]:
        requested = [str(value) for value in uids]
        if len(requested) != len(set(requested)):
            raise ValueError("A P04 role contains duplicate observation UIDs.")
        missing = [uid for uid in requested if uid not in self.uid_to_index]
        if missing:
            raise ValueError(f"A P04 role contains {len(missing)} unknown UIDs.")
        indices = np.asarray([self.uid_to_index[uid] for uid in requested], dtype=int)
        return self.intensity[indices], self.metadata.iloc[indices].reset_index(drop=True)


@dataclass
class DeepFitResult:
    status: str
    reason_code: str | None
    history: pd.DataFrame
    predictions: pd.DataFrame
    best_epoch: int | None
    best_validation_balanced_accuracy: float | None
    best_validation_nll: float | None
    best_validation_macro_f1: float | None
    best_training_balanced_accuracy: float | None
    checkpoint_state_sha256: str | None
    state_dict: dict[str, torch.Tensor] | None
    augmentation: dict[str, Any]
    diagnostic: str | None
    elapsed_seconds: float
    peak_cuda_bytes: int
    traceback_digest: str | None
    failure_message: str | None


def _seed_from(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json_bytes([str(value) for value in parts])).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _class_setup(metadata: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    classes = tuple(sorted(metadata.target_analyte.astype(str).unique()))
    if len(classes) != 3:
        raise ValueError("P04 station-local roles must contain exactly three classes.")
    lookup = {label: index for index, label in enumerate(classes)}
    labels = metadata.target_analyte.astype(str).map(lookup).to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=len(classes)).astype(float)
    if (counts == 0).any():
        raise ValueError("A P04 fitting role is missing a station-local class.")
    weights = len(labels) / (len(classes) * counts)
    return classes, labels, weights.astype(np.float32)


def _noise_quantiles(metadata: pd.DataFrame) -> np.ndarray:
    scale = metadata.intensity_range.to_numpy(dtype=float)
    noise = metadata.first_difference_noise_mad.to_numpy(dtype=float)
    ratio = noise / np.maximum(scale, np.finfo(float).eps)
    ratio = ratio[np.isfinite(ratio) & (ratio >= 0)]
    if len(ratio) == 0:
        raise ValueError("P04 cannot estimate source-only normalized noise.")
    return np.asarray([0.0, *np.quantile(ratio, [0.25, 0.5, 0.75])], dtype=float)


def _augment(
    values: np.ndarray,
    uids: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_levels: np.ndarray,
    digest: hashlib._Hash,
) -> np.ndarray:
    output = values.astype(np.float64, copy=True)
    axis = np.arange(400, 1801, dtype=float)
    intensity = rng.uniform(0.9, 1.1, len(output))
    baseline_span = rng.uniform(-0.05, 0.05, len(output))
    shift = rng.uniform(-2.0, 2.0, len(output))
    noise_index = rng.integers(0, len(noise_levels), len(output))
    noise_sigma = noise_levels[noise_index] / 0.9538725524
    baseline_axis = np.linspace(-0.5, 0.5, output.shape[1])
    for index, row in enumerate(output):
        row *= intensity[index]
        row += baseline_span[index] * baseline_axis
        if noise_sigma[index] > 0:
            row += rng.normal(0.0, noise_sigma[index], len(row))
        row = np.interp(axis - shift[index], axis, row, left=row[0], right=row[-1])
        span = float(row.max() - row.min())
        if not np.isfinite(span) or span <= np.finfo(float).eps:
            raise FloatingPointError("P04 augmentation produced an invalid row range.")
        output[index] = (row - row.min()) / span
        record = {
            "observation_uid": str(uids[index]),
            "intensity_scale": float(intensity[index]),
            "linear_baseline_span_fraction": float(baseline_span[index]),
            "translation_cm1": float(shift[index]),
            "noise_level_index": int(noise_index[index]),
            "noise_sigma": float(noise_sigma[index]),
        }
        digest.update(canonical_json_bytes(record))
    return output.astype(np.float32)


def _predict(
    model: nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size, None, :]).to(device)
            result.append(model(batch).detach().cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float64)


def _metric_values(
    labels: np.ndarray, logits: np.ndarray, classes: tuple[str, ...]
) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    nll = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-7, 1 - 1e-7)
    ).mean()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "negative_log_likelihood": float(nll),
        "predicted_class_count": int(len(set(predicted.tolist()))),
    }


def _prediction_frame(
    metadata: pd.DataFrame,
    logits: np.ndarray,
    *,
    classes: tuple[str, ...],
    fit_id: str,
) -> pd.DataFrame:
    predicted = np.asarray(classes)[logits.argmax(axis=1)]
    frame = metadata[
        ["observation_uid", "master_sample_id", "instrument", "station", "target_analyte"]
    ].copy()
    frame = frame.rename(columns={"target_analyte": "true_label"})
    frame["predicted_label"] = predicted
    frame["fit_id"] = fit_id
    frame["class_vocabulary"] = json.dumps(list(classes), separators=(",", ":"))
    for index in range(len(classes)):
        frame[f"logit_{index}"] = logits[:, index]
    return frame


def _diagnostic(
    *,
    training_ba: float,
    validation_ba: float,
    predicted_class_count: int,
    history: pd.DataFrame,
) -> str:
    chance = 1.0 / 3.0
    if predicted_class_count < 2:
        return "collapse"
    if training_ba <= chance + 0.05:
        return "underfit"
    if training_ba - validation_ba > 0.20:
        return "overfit"
    tail = history.validation_balanced_accuracy.tail(10)
    if (
        len(tail) >= 10
        and float(tail.std(ddof=0)) > 0.15
        or float(history.gradient_clipped_fraction.mean()) > 0.5
    ):
        return "optimization_instability"
    return "none"


def train_with_validation(
    *,
    fit_id: str,
    train_values: np.ndarray,
    train_metadata: pd.DataFrame,
    validation_values: np.ndarray,
    validation_metadata: pd.DataFrame,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    maximum_epochs: int,
    minimum_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    device: torch.device,
) -> DeepFitResult:
    started = time.perf_counter()
    empty = pd.DataFrame()
    try:
        classes, train_labels, class_weights = _class_setup(train_metadata)
        validation_classes = set(validation_metadata.target_analyte.astype(str))
        if validation_classes != set(classes):
            raise ValueError("P04 validation role does not contain the fitting vocabulary.")
        label_lookup = {label: index for index, label in enumerate(classes)}
        validation_labels = (
            validation_metadata.target_analyte.astype(str).map(label_lookup).to_numpy(dtype=int)
        )
        noise_levels = _noise_quantiles(train_metadata)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = CompactSERSClassifier(len(classes)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).to(device=device)
        )
        best_key: tuple[float, float, int] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch: int | None = None
        best_metrics: dict[str, float] | None = None
        best_train_ba: float | None = None
        nonimproving = 0
        augmentation_digest = hashlib.sha256()
        augmentation_draws = 0
        histories: list[dict[str, Any]] = []
        uids = train_metadata.observation_uid.astype(str).to_numpy()
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            epoch_rng = np.random.default_rng(_seed_from(fit_id, seed, epoch, "p04-aug-v1"))
            order = epoch_rng.permutation(len(train_values))
            total_loss = 0.0
            total_seen = 0
            clipped = 0
            steps = 0
            gradient_norms: list[float] = []
            for batch_ordinal, start in enumerate(range(0, len(order), batch_size)):
                indices = order[start : start + batch_size]
                batch_rng = np.random.default_rng(
                    _seed_from(fit_id, seed, epoch, batch_ordinal, "p04-aug-v1")
                )
                augmented = _augment(
                    train_values[indices],
                    uids[indices],
                    rng=batch_rng,
                    noise_levels=noise_levels,
                    digest=augmentation_digest,
                )
                augmentation_draws += len(indices)
                values = torch.from_numpy(augmented[:, None, :]).to(device)
                labels = torch.from_numpy(train_labels[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("P04 training loss became nonfinite.")
                loss.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("P04 gradient norm became nonfinite.")
                clipped += int(norm > gradient_clip_norm)
                gradient_norms.append(norm)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(indices)
                total_seen += len(indices)
                steps += 1
            train_logits = _predict(
                model, train_values, device=device, batch_size=batch_size
            )
            validation_logits = _predict(
                model, validation_values, device=device, batch_size=batch_size
            )
            train_metrics = _metric_values(train_labels, train_logits, classes)
            validation_metrics = _metric_values(
                validation_labels, validation_logits, classes
            )
            histories.append(
                {
                    "fit_id": fit_id,
                    "epoch": epoch,
                    "training_loss": total_loss / total_seen,
                    "training_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "validation_balanced_accuracy": validation_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_negative_log_likelihood": validation_metrics[
                        "negative_log_likelihood"
                    ],
                    "validation_predicted_class_count": validation_metrics[
                        "predicted_class_count"
                    ],
                    "mean_gradient_norm": float(np.mean(gradient_norms)),
                    "maximum_gradient_norm": float(np.max(gradient_norms)),
                    "gradient_clipped_fraction": clipped / steps,
                    "optimizer_steps": steps,
                }
            )
            key = (
                -validation_metrics["balanced_accuracy"],
                validation_metrics["negative_log_likelihood"],
                epoch,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = validation_metrics
                best_train_ba = train_metrics["balanced_accuracy"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                nonimproving = 0
            else:
                nonimproving += 1
            if epoch >= minimum_epochs and nonimproving >= patience:
                break
        if best_state is None or best_metrics is None or best_epoch is None:
            raise RuntimeError("P04 fitting produced no finite checkpoint.")
        model.load_state_dict(best_state)
        validation_logits = _predict(
            model, validation_values, device=device, batch_size=batch_size
        )
        predictions = _prediction_frame(
            validation_metadata, validation_logits, classes=classes, fit_id=fit_id
        )
        history = pd.DataFrame(histories)
        diagnostic = _diagnostic(
            training_ba=float(best_train_ba),
            validation_ba=float(best_metrics["balanced_accuracy"]),
            predicted_class_count=int(best_metrics["predicted_class_count"]),
            history=history,
        )
        return DeepFitResult(
            status="complete",
            reason_code=None,
            history=history,
            predictions=predictions,
            best_epoch=best_epoch,
            best_validation_balanced_accuracy=float(best_metrics["balanced_accuracy"]),
            best_validation_nll=float(best_metrics["negative_log_likelihood"]),
            best_validation_macro_f1=float(best_metrics["macro_f1"]),
            best_training_balanced_accuracy=float(best_train_ba),
            checkpoint_state_sha256=_state_hash(best_state),
            state_dict=best_state,
            augmentation={
                "version": "p04-aug-v1",
                "source_noise_levels": noise_levels.tolist(),
                "draw_count": augmentation_draws,
                "draw_sha256": augmentation_digest.hexdigest(),
            },
            diagnostic=diagnostic,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=(
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            traceback_digest=None,
            failure_message=None,
        )
    except Exception as error:
        trace = traceback.format_exc()
        if isinstance(error, (FloatingPointError, OverflowError)):
            status = "numerical_failure"
        elif isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
            status = "resource_failure"
        elif isinstance(error, ValueError):
            status = "data_failure"
        else:
            status = "fit_failure"
        return DeepFitResult(
            status=status,
            reason_code=type(error).__name__,
            history=empty,
            predictions=empty,
            best_epoch=None,
            best_validation_balanced_accuracy=None,
            best_validation_nll=None,
            best_validation_macro_f1=None,
            best_training_balanced_accuracy=None,
            checkpoint_state_sha256=None,
            state_dict=None,
            augmentation={},
            diagnostic=None,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=0,
            traceback_digest=hashlib.sha256(trace.encode()).hexdigest(),
            failure_message=str(error),
        )


def select_candidate(
    fit_status: pd.DataFrame, candidates: pd.DataFrame, expected_unit_count: int
) -> tuple[pd.Series | None, pd.DataFrame]:
    complete = fit_status[
        fit_status.status.eq("complete")
        & fit_status.best_validation_balanced_accuracy.notna()
    ].copy()
    per_unit = (
        complete.groupby(["candidate_id", "selection_unit_id"], as_index=False)
        .agg(
            balanced_accuracy=("best_validation_balanced_accuracy", "mean"),
            macro_f1=("best_validation_macro_f1", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    summaries = []
    for candidate in candidates.itertuples(index=False):
        evidence = per_unit[per_unit.candidate_id.eq(candidate.candidate_id)]
        supported = len(evidence) == expected_unit_count and evidence.seed_count.eq(3).all()
        summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "learning_rate": candidate.learning_rate,
                "weight_decay": candidate.weight_decay,
                "hyperparameter_sha256": candidate.hyperparameter_sha256,
                "complexity_rank": candidate.complexity_rank,
                "declared_candidate_order": candidate.declared_candidate_order,
                "complete_support": supported,
                "selection_unit_count": len(evidence),
                "mean_balanced_accuracy": (
                    float(evidence.balanced_accuracy.mean()) if supported else np.nan
                ),
                "worst_balanced_accuracy": (
                    float(evidence.balanced_accuracy.min()) if supported else np.nan
                ),
                "mean_macro_f1": float(evidence.macro_f1.mean()) if supported else np.nan,
            }
        )
    trace = pd.DataFrame(summaries)
    eligible = trace[trace.complete_support].sort_values(
        [
            "mean_balanced_accuracy",
            "worst_balanced_accuracy",
            "mean_macro_f1",
            "complexity_rank",
            "declared_candidate_order",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    trace["selected"] = False
    if eligible.empty:
        return None, trace
    winner_index = eligible.index[0]
    trace.loc[winner_index, "selected"] = True
    return trace.loc[winner_index], trace


def _master_equal_calibration(
    predictions: pd.DataFrame, classes: tuple[str, ...]
) -> Any:
    logit_columns = [f"logit_{index}" for index in range(len(classes))]
    labels_per_master = predictions.groupby("master_sample_id").true_label.nunique()
    if not labels_per_master.eq(1).all():
        raise ValueError("P04 calibration masters have inconsistent labels.")
    master = (
        predictions.groupby(["master_sample_id", "true_label"], as_index=False)[
            logit_columns
        ]
        .mean()
        .sort_values("master_sample_id", kind="stable")
    )
    return fit_temperature(
        master[logit_columns].to_numpy(),
        master.true_label.astype(str).to_numpy(),
        class_vocabulary=classes,
        observation_uids=np.asarray(
            [f"calibration-master:{value}" for value in master.master_sample_id],
            dtype=str,
        ),
        master_ids=master.master_sample_id.astype(str).to_numpy(),
    )


def train_fixed_epochs(
    *,
    fit_id: str,
    train_values: np.ndarray,
    train_metadata: pd.DataFrame,
    test_values: np.ndarray,
    test_metadata: pd.DataFrame,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    epochs: int,
    gradient_clip_norm: float,
    device: torch.device,
    calibration_predictions: pd.DataFrame,
) -> tuple[DeepFitResult, bytes | None, Any | None]:
    started = time.perf_counter()
    try:
        classes, train_labels, class_weights = _class_setup(train_metadata)
        noise_levels = _noise_quantiles(train_metadata)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = CompactSERSClassifier(len(classes)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
        augmentation_digest = hashlib.sha256()
        augmentation_draws = 0
        histories = []
        uids = train_metadata.observation_uid.astype(str).to_numpy()
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_rng = np.random.default_rng(_seed_from(fit_id, seed, epoch, "p04-aug-v1"))
            order = epoch_rng.permutation(len(train_values))
            losses = 0.0
            seen = 0
            clipped = 0
            norms = []
            steps = 0
            for batch_ordinal, start in enumerate(range(0, len(order), batch_size)):
                indices = order[start : start + batch_size]
                batch_rng = np.random.default_rng(
                    _seed_from(fit_id, seed, epoch, batch_ordinal, "p04-aug-v1")
                )
                augmented = _augment(
                    train_values[indices],
                    uids[indices],
                    rng=batch_rng,
                    noise_levels=noise_levels,
                    digest=augmentation_digest,
                )
                augmentation_draws += len(indices)
                values = torch.from_numpy(augmented[:, None, :]).to(device)
                labels = torch.from_numpy(train_labels[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("P04 final-refit loss became nonfinite.")
                loss.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("P04 final-refit gradient became nonfinite.")
                clipped += int(norm > gradient_clip_norm)
                norms.append(norm)
                optimizer.step()
                losses += float(loss.detach().cpu()) * len(indices)
                seen += len(indices)
                steps += 1
            train_logits = _predict(model, train_values, device=device, batch_size=batch_size)
            train_metrics = _metric_values(train_labels, train_logits, classes)
            histories.append(
                {
                    "fit_id": fit_id,
                    "epoch": epoch,
                    "training_loss": losses / seen,
                    "training_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "validation_balanced_accuracy": np.nan,
                    "validation_macro_f1": np.nan,
                    "validation_negative_log_likelihood": np.nan,
                    "validation_predicted_class_count": np.nan,
                    "mean_gradient_norm": float(np.mean(norms)),
                    "maximum_gradient_norm": float(np.max(norms)),
                    "gradient_clipped_fraction": clipped / steps,
                    "optimizer_steps": steps,
                }
            )
        calibration = _master_equal_calibration(calibration_predictions, classes)
        test_logits = _predict(model, test_values, device=device, batch_size=batch_size)
        probabilities = apply_temperature(test_logits, calibration)
        prediction = _prediction_frame(test_metadata, test_logits, classes=classes, fit_id=fit_id)
        prediction["predicted_label"] = np.asarray(classes)[probabilities.argmax(axis=1)]
        for index in range(len(classes)):
            prediction[f"probability_{index}"] = probabilities[:, index]
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        checkpoint = {
            "schema_version": "nato-sers-p04-checkpoint-v1",
            "fit_id": fit_id,
            "model_id": "D0-ERM",
            "class_vocabulary": classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "epochs": epochs,
            "temperature": calibration.temperature,
            "state_dict": state,
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        result = DeepFitResult(
            status="complete",
            reason_code=None,
            history=pd.DataFrame(histories),
            predictions=prediction,
            best_epoch=epochs,
            best_validation_balanced_accuracy=None,
            best_validation_nll=None,
            best_validation_macro_f1=None,
            best_training_balanced_accuracy=float(
                histories[-1]["training_balanced_accuracy"]
            ),
            checkpoint_state_sha256=_state_hash(state),
            state_dict=state,
            augmentation={
                "version": "p04-aug-v1",
                "source_noise_levels": noise_levels.tolist(),
                "draw_count": augmentation_draws,
                "draw_sha256": augmentation_digest.hexdigest(),
            },
            diagnostic="not_applicable_final_refit",
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=(
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            traceback_digest=None,
            failure_message=None,
        )
        return result, buffer.getvalue(), calibration
    except Exception as error:
        trace = traceback.format_exc()
        if isinstance(error, (FloatingPointError, OverflowError)):
            status = "numerical_failure"
        elif isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
            status = "resource_failure"
        elif isinstance(error, ValueError):
            status = "data_failure"
        else:
            status = "fit_failure"
        result = DeepFitResult(
            status=status,
            reason_code=type(error).__name__,
            history=pd.DataFrame(),
            predictions=pd.DataFrame(),
            best_epoch=None,
            best_validation_balanced_accuracy=None,
            best_validation_nll=None,
            best_validation_macro_f1=None,
            best_training_balanced_accuracy=None,
            checkpoint_state_sha256=None,
            state_dict=None,
            augmentation={},
            diagnostic=None,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=0,
            traceback_digest=hashlib.sha256(trace.encode()).hexdigest(),
            failure_message=str(error),
        )
        return result, None, None


def status_record(
    fit: pd.Series, result: DeepFitResult, *, selected_candidate_id: str | None = None
) -> dict[str, Any]:
    if result.status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown P04 terminal fit status: {result.status}")
    return {
        "fit_id": str(fit.fit_id),
        "context_id": str(fit.context_id),
        "experiment_id": str(fit.experiment_id),
        "stage": str(fit.stage),
        "candidate_id": (
            selected_candidate_id
            if str(fit.candidate_id) == "selected_after_inner"
            else str(fit.candidate_id)
        ),
        "selection_unit_id": str(fit.selection_unit_id),
        "seed": int(fit.seed),
        "status": result.status,
        "reason_code": result.reason_code,
        "best_epoch": result.best_epoch,
        "epochs_completed": len(result.history),
        "best_validation_balanced_accuracy": result.best_validation_balanced_accuracy,
        "best_validation_nll": result.best_validation_nll,
        "best_validation_macro_f1": result.best_validation_macro_f1,
        "best_training_balanced_accuracy": result.best_training_balanced_accuracy,
        "checkpoint_state_sha256": result.checkpoint_state_sha256,
        "augmentation_draw_count": result.augmentation.get("draw_count"),
        "augmentation_draw_sha256": result.augmentation.get("draw_sha256"),
        "source_noise_levels": json.dumps(
            result.augmentation.get("source_noise_levels"), separators=(",", ":")
        ),
        "diagnostic": result.diagnostic,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_cuda_bytes": result.peak_cuda_bytes,
        "traceback_digest": result.traceback_digest,
        "failure_message": result.failure_message,
    }
