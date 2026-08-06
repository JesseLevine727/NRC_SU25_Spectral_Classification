"""Frozen row-local P01 spectral representations and preservation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata

from atlas_sers.governance.canonical import sha256_bytes, sha256_value

EPSILON = np.finfo(float).eps


@dataclass(frozen=True)
class RepresentationBundle:
    representation_id: str
    axis: np.ndarray
    intensity: np.ndarray
    valid_rows: np.ndarray
    reason_codes: tuple[str, ...]
    specification: dict[str, Any]


def row_minmax(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    low = matrix.min(axis=1, keepdims=True)
    span = matrix.max(axis=1, keepdims=True) - low
    valid = np.isfinite(matrix).all(axis=1) & (span[:, 0] > EPSILON)
    output = (matrix - low) / np.maximum(span, EPSILON)
    reasons = tuple("included" if flag else "nonfinite_or_zero_range" for flag in valid)
    return output, valid, reasons


def row_snv(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    valid = np.isfinite(matrix).all(axis=1) & (scale[:, 0] > EPSILON)
    output = centered / np.maximum(scale, EPSILON)
    reasons = tuple("included" if flag else "nonfinite_or_zero_scale" for flag in valid)
    return output, valid, reasons


def row_vector(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    valid = np.isfinite(matrix).all(axis=1) & (norm[:, 0] > EPSILON)
    output = matrix / np.maximum(norm, EPSILON)
    reasons = tuple("included" if flag else "nonfinite_or_zero_norm" for flag in valid)
    return output, valid, reasons


def row_area(
    matrix: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    shifted = matrix - matrix.min(axis=1, keepdims=True)
    area = np.trapezoid(shifted, x=axis, axis=1)[:, None]
    valid = np.isfinite(matrix).all(axis=1) & (area[:, 0] > EPSILON)
    output = shifted / np.maximum(area, EPSILON)
    reasons = tuple("included" if flag else "nonfinite_or_zero_area" for flag in valid)
    return output, valid, reasons


def despike(matrix: np.ndarray, *, window: int, threshold: float) -> np.ndarray:
    result = matrix.copy()
    for index, row in enumerate(matrix):
        median = pd.Series(row).rolling(window, center=True, min_periods=1).median().to_numpy()
        residual = row - median
        mad = np.median(np.abs(residual - np.median(residual)))
        flagged = np.abs(residual) > threshold * max(mad / 0.67448975, EPSILON)
        neighborhood = np.convolve(flagged.astype(int), np.ones(3, dtype=int), mode="same")
        isolated = flagged & (neighborhood <= 2)
        result[index, isolated] = median[isolated]
    return result


def _penalty_matrix(points: int) -> sparse.csc_matrix:
    difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(points - 2, points))
    return (difference.T @ difference).tocsc()


def arpls_correct(
    matrix: np.ndarray,
    *,
    lam: float,
    iterations: int,
    relative_tolerance: float,
    logistic_clip: float,
) -> np.ndarray:
    penalty = _penalty_matrix(matrix.shape[1])
    output = np.empty_like(matrix)
    for index, row in enumerate(matrix):
        weights = np.ones(len(row))
        baseline = np.zeros_like(row)
        for _ in range(iterations):
            baseline = spsolve(
                sparse.diags(weights, format="csc") + lam * penalty,
                weights * row,
            )
            negative = (row - baseline)[row < baseline]
            if len(negative) < 2 or np.std(negative) <= EPSILON:
                break
            mean, std = float(np.mean(negative)), float(np.std(negative))
            argument = np.clip(
                2.0 * ((row - baseline) - (2 * std - mean)) / std,
                -logistic_clip,
                logistic_clip,
            )
            updated = 1.0 / (1.0 + np.exp(argument))
            relative_change = np.linalg.norm(updated - weights) / max(
                np.linalg.norm(weights), 1e-12
            )
            weights = updated
            if relative_change < relative_tolerance:
                break
        output[index] = row - baseline
    return output


def _subset(
    axis: np.ndarray, matrix: np.ndarray, bounds: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    mask = (axis >= bounds[0]) & (axis <= bounds[1])
    selected_axis = np.asarray(axis[mask], dtype=float)
    expected_points = int(bounds[1] - bounds[0] + 1)
    if len(selected_axis) != expected_points or not np.array_equal(
        selected_axis, np.arange(bounds[0], bounds[1] + 1, dtype=float)
    ):
        raise ValueError(f"Representation axis {bounds} is incomplete or irregular.")
    return selected_axis, np.asarray(matrix[:, mask], dtype=float)


def _transform(
    representation_id: str,
    matrix: np.ndarray,
    axis: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if representation_id.startswith("R_MIN_"):
        return row_minmax(matrix)
    if representation_id == "R_SNV_400_1800":
        return row_snv(matrix)
    if representation_id == "R_VECTOR_400_1800":
        return row_vector(matrix)
    if representation_id == "R_AREA_400_1800":
        return row_area(matrix, axis)
    clean = despike(
        matrix,
        window=int(parameters["despike_window"]),
        threshold=float(parameters["despike_mad_threshold"]),
    )
    if representation_id == "R_SG_400_1800":
        smooth = savgol_filter(
            clean,
            int(parameters["savgol_window"]),
            int(parameters["savgol_polynomial"]),
            axis=1,
            mode=str(parameters["savgol_mode"]),
        )
        return row_minmax(smooth)
    if representation_id == "R_ARPLS_400_1800":
        corrected = arpls_correct(
            clean,
            lam=float(parameters["arpls_lambda"]),
            iterations=int(parameters["arpls_iterations"]),
            relative_tolerance=float(parameters["arpls_relative_weight_tolerance"]),
            logistic_clip=float(parameters["arpls_logistic_clip"]),
        )
        return row_minmax(corrected)
    if representation_id == "R_D1_400_1800":
        derivative = savgol_filter(
            clean,
            int(parameters["savgol_window"]),
            int(parameters["savgol_polynomial"]),
            deriv=1,
            delta=float(np.median(np.diff(axis))),
            axis=1,
            mode=str(parameters["savgol_mode"]),
        )
        return row_snv(derivative)
    raise KeyError(f"Unknown representation {representation_id}")


def build_representations(
    axis: np.ndarray,
    raw: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, RepresentationBundle]:
    bundles: dict[str, RepresentationBundle] = {}
    parameters = contract["transformation_parameters"]
    for specification in contract["representations"]:
        representation_id = specification["representation_id"]
        selected_axis, selected_raw = _subset(axis, raw, specification["range_cm1"])
        transformed, valid, reasons = _transform(
            representation_id, selected_raw, selected_axis, parameters
        )
        bundles[representation_id] = RepresentationBundle(
            representation_id=representation_id,
            axis=selected_axis.astype(np.float32),
            intensity=transformed.astype(np.float32),
            valid_rows=valid,
            reason_codes=reasons,
            specification=specification,
        )
    return bundles


def representation_invariants(bundle: RepresentationBundle) -> dict[str, Any]:
    matrix = bundle.intensity.astype(float)
    finite = np.isfinite(matrix).all()
    tolerance = 1e-6
    checks: dict[str, bool] = {
        "finite": bool(finite),
        "all_rows_valid": bool(bundle.valid_rows.all()),
        "axis_strict": bool(np.all(np.diff(bundle.axis) > 0)),
    }
    identifier = bundle.representation_id
    if identifier.startswith("R_MIN_") or identifier in {
        "R_SG_400_1800",
        "R_ARPLS_400_1800",
    }:
        checks["row_min_zero"] = bool(np.allclose(matrix.min(axis=1), 0, atol=tolerance))
        checks["row_max_one"] = bool(np.allclose(matrix.max(axis=1), 1, atol=tolerance))
    elif identifier in {"R_SNV_400_1800", "R_D1_400_1800"}:
        checks["row_mean_zero"] = bool(np.allclose(matrix.mean(axis=1), 0, atol=tolerance))
        checks["row_std_one"] = bool(np.allclose(matrix.std(axis=1), 1, atol=tolerance))
    elif identifier == "R_VECTOR_400_1800":
        checks["row_l2_one"] = bool(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=tolerance))
    elif identifier == "R_AREA_400_1800":
        checks["row_nonnegative"] = bool(np.min(matrix) >= -tolerance)
        checks["row_area_one"] = bool(
            np.allclose(np.trapezoid(matrix, x=bundle.axis, axis=1), 1, atol=tolerance)
        )
    return {"checks": checks, "status": "pass" if all(checks.values()) else "fail"}


def representation_registry(
    bundles: dict[str, RepresentationBundle],
    observation_uids: np.ndarray,
    *,
    source_bundle_sha256: str,
    code_sha256: str,
    config_sha256: str,
    run_id: str,
) -> pd.DataFrame:
    uid_bytes = "\n".join(observation_uids.astype(str)).encode()
    rows: list[dict[str, Any]] = []
    for identifier, bundle in bundles.items():
        invariants = representation_invariants(bundle)
        rows.append(
            {
                "representation_id": identifier,
                "scope": bundle.specification["scope"],
                "rows": bundle.intensity.shape[0],
                "features": bundle.intensity.shape[1],
                "dtype": str(bundle.intensity.dtype),
                "axis_start_cm1": float(bundle.axis[0]),
                "axis_end_cm1": float(bundle.axis[-1]),
                "axis_sha256": sha256_bytes(bundle.axis.tobytes()),
                "array_sha256": sha256_bytes(bundle.intensity.tobytes()),
                "row_order_sha256": sha256_bytes(uid_bytes),
                "transformation_sha256": sha256_value(bundle.specification),
                "source_bundle_sha256": source_bundle_sha256,
                "code_sha256": code_sha256,
                "config_sha256": config_sha256,
                "run_id": run_id,
                "invalid_rows": int((~bundle.valid_rows).sum()),
                "invariant_status": invariants["status"],
            }
        )
    return pd.DataFrame(rows)


def reproduce_frozen_candidates(
    axis: np.ndarray,
    raw: np.ndarray,
    frozen: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    clean = despike(
        raw,
        window=int(parameters["despike_window"]),
        threshold=float(parameters["despike_mad_threshold"]),
    )
    smooth = savgol_filter(
        clean,
        int(parameters["savgol_window"]),
        int(parameters["savgol_polynomial"]),
        axis=1,
        mode=str(parameters["savgol_mode"]),
    )
    corrected = arpls_correct(
        clean,
        lam=float(parameters["arpls_lambda"]),
        iterations=int(parameters["arpls_iterations"]),
        relative_tolerance=float(parameters["arpls_relative_weight_tolerance"]),
        logistic_clip=float(parameters["arpls_logistic_clip"]),
    )
    derivative = savgol_filter(
        clean,
        int(parameters["savgol_window"]),
        int(parameters["savgol_polynomial"]),
        deriv=1,
        delta=float(np.median(np.diff(axis))),
        axis=1,
        mode=str(parameters["savgol_mode"]),
    )
    generated = {
        "minimal_minmax": row_minmax(raw)[0],
        "minimal_snv": row_snv(raw)[0],
        "minimal_vector": row_vector(raw)[0],
        "minimal_area": row_area(raw, axis)[0],
        "despike_savgol_minmax": row_minmax(smooth)[0],
        "arpls_minmax": row_minmax(corrected)[0],
        "savgol_derivative1_snv": row_snv(derivative)[0],
    }
    comparisons: dict[str, Any] = {}
    for name, values in generated.items():
        difference = np.abs(values.astype(np.float32) - frozen[name].astype(np.float32))
        comparisons[name] = {
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "exact_float32_fraction": float(np.mean(difference == 0)),
        }
    return comparisons


def _peak_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = savgol_filter(row_minmax(reference[None, :])[0][0], 11, 3)
    test = savgol_filter(row_minmax(candidate[None, :])[0][0], 11, 3)
    ref_peaks, ref_properties = find_peaks(ref, prominence=0.04, distance=4)
    test_peaks, _ = find_peaks(test, prominence=0.025, distance=3)
    if len(ref_peaks):
        selected = ref_peaks[np.argsort(ref_properties["prominences"])[-10:]]
        distances = np.asarray(
            [np.min(np.abs(test_peaks - peak)) if len(test_peaks) else np.inf for peak in selected]
        )
        recall = float(np.mean(distances <= 5))
        displacement = (
            float(np.median(distances[np.isfinite(distances)]))
            if np.isfinite(distances).any()
            else np.nan
        )
    else:
        recall = np.nan
        displacement = np.nan
    return {
        "reference_peak_count": len(ref_peaks),
        "candidate_peak_count": len(test_peaks),
        "top_peak_recall_pm5cm1": recall,
        "median_peak_displacement_cm1": displacement,
    }


def preservation_metrics(
    raw_axis: np.ndarray,
    raw: np.ndarray,
    bundles: dict[str, RepresentationBundle],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identifier, bundle in bundles.items():
        mask = (raw_axis >= bundle.axis[0]) & (raw_axis <= bundle.axis[-1])
        reference = raw[:, mask].astype(float)
        candidate = bundle.intensity.astype(float)
        reference_scaled = row_minmax(reference)[0]
        candidate_scaled = row_minmax(candidate)[0]
        for index in range(len(manifest)):
            left = reference_scaled[index]
            right = candidate_scaled[index]
            correlation = float(np.corrcoef(left, right)[0, 1])
            cosine = float(
                np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), EPSILON)
            )
            cosine = float(np.clip(cosine, -1, 1))
            rank_correlation = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
            baseline = percentile_filter(right, percentile=10, size=101, mode="nearest")
            record = manifest.iloc[index]
            rows.append(
                {
                    "observation_uid": record.observation_uid,
                    "instrument": record.instrument,
                    "sensor_family": record.sensor_family,
                    "station": record.station,
                    "representation_id": identifier,
                    "shape_correlation": correlation,
                    "spectral_angle_radians": float(np.arccos(cosine)),
                    "rank_correlation": rank_correlation,
                    "first_difference_roughness": float(np.median(np.abs(np.diff(right)))),
                    "baseline_span": float(np.ptp(baseline)),
                    "changed_point_fraction": float(np.mean(np.abs(left - right) > 1e-7)),
                    "clipped_fraction": float(np.mean((right <= 0) | (right >= 1))),
                    **_peak_metrics(reference[index], candidate[index]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_preservation(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"source_scan_id"}
    ]
    rows: list[dict[str, Any]] = []
    for (representation, instrument), group in metrics.groupby(
        ["representation_id", "instrument"], sort=True
    ):
        row: dict[str, Any] = {
            "representation_id": representation,
            "instrument": instrument,
            "n_spectra": len(group),
        }
        for column in numeric:
            row[f"median_{column}"] = float(group[column].median())
            row[f"q10_{column}"] = float(group[column].quantile(0.1))
            row[f"q90_{column}"] = float(group[column].quantile(0.9))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_seed_stability(label_sets: list[np.ndarray]) -> float:
    from sklearn.metrics import adjusted_rand_score

    scores = [adjusted_rand_score(left, right) for left, right in combinations(label_sets, 2)]
    return float(np.median(scores)) if scores else 1.0
