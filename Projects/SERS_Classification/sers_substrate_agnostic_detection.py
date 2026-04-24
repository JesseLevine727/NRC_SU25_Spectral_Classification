#!/usr/bin/env python3
"""Evaluate SERS chemical detection on unseen substrates.

This script intentionally predicts the chemical label only. Substrate is used
only as the held-out evaluation group, which prevents the chemical-substrate
pair leakage present in ordinary stratified row splits.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


DEFAULT_DATA = Path("Workspace/consolidated_SERS.csv")


def snv_l2(X: np.ndarray) -> np.ndarray:
    """Per-spectrum standard-normal-variate plus L2 normalization."""
    X = np.asarray(X, dtype=np.float64)
    centered = X - np.median(X, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    X = centered / scale
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


def snv(X: np.ndarray) -> np.ndarray:
    """Per-spectrum standard normal variate without final L2 normalization."""
    X = np.asarray(X, dtype=np.float64)
    centered = X - np.median(X, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return centered / scale


def smooth_snv_l2(X: np.ndarray) -> np.ndarray:
    return snv_l2(savgol_filter(np.asarray(X, dtype=np.float64), 17, 3, axis=1))


def derivative_snv_l2(order: int):
    def transform(X: np.ndarray) -> np.ndarray:
        transformed = savgol_filter(snv(X), 17, 3, deriv=order, axis=1)
        return snv_l2(transformed)

    return transform


def peak_emphasis(X: np.ndarray) -> np.ndarray:
    """Suppress broad substrate/background shape and retain positive peaks."""
    X = savgol_filter(snv(X), 15, 3, axis=1)
    trend = savgol_filter(X, 101, 3, axis=1)
    return snv_l2(np.maximum(X - trend, 0))


@dataclass(frozen=True)
class FoldResult:
    held_out_substrate: str
    feature: str
    model: str
    n_train: int
    n_test: int
    test_labels: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


def spectral_columns(df: pd.DataFrame, crop_min: float | None, crop_max: float | None) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in {"Label", "Substrate", "Class"}:
            continue
        try:
            wav = float(col)
        except ValueError:
            continue
        if (crop_min is None or wav >= crop_min) and (crop_max is None or wav <= crop_max):
            cols.append(col)
    if not cols:
        raise ValueError("No spectral numeric columns found.")
    return cols


def load_dataset(
    path: Path,
    crop_min: float | None,
    crop_max: float | None,
    min_substrates: int,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    required = {"Label", "Substrate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    cols = spectral_columns(df, crop_min, crop_max)
    valid_labels = (
        df.groupby("Label")["Substrate"].nunique().loc[lambda s: s >= min_substrates].index
    )
    filtered = df[df["Label"].isin(valid_labels)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No labels have enough substrates for substrate-held-out evaluation.")
    return filtered, cols


def build_feature_transforms() -> dict[str, FunctionTransformer]:
    return {
        "snv_l2": FunctionTransformer(snv_l2, validate=False),
        "smooth_snv_l2": FunctionTransformer(smooth_snv_l2, validate=False),
        "derivative_1": FunctionTransformer(derivative_snv_l2(1), validate=False),
        "derivative_2": FunctionTransformer(derivative_snv_l2(2), validate=False),
        "peak_emphasis": FunctionTransformer(peak_emphasis, validate=False),
    }


def build_models() -> dict[str, object]:
    return {
        "nearest_centroid": NearestCentroid(metric="euclidean"),
        "cosine_knn_3": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "linear_svm": make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel="linear", C=1.0, class_weight="balanced"),
        ),
    }


def evaluate(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["Label"].astype(str).to_numpy()
    groups = df["Substrate"].astype(str).to_numpy()
    features = build_feature_transforms()
    models = build_models()

    rows: list[FoldResult] = []
    reports: dict[str, pd.DataFrame] = {}

    for substrate in sorted(np.unique(groups)):
        test_mask = groups == substrate
        train_mask = ~test_mask

        y_train = y[train_mask]
        y_test = y[test_mask]
        known_mask = np.isin(y_test, np.unique(y_train))
        if not np.all(known_mask):
            X_test = X[test_mask][known_mask]
            y_test = y_test[known_mask]
        else:
            X_test = X[test_mask]

        if len(y_test) == 0:
            continue

        for feature_name, transformer in features.items():
            for model_name, estimator in models.items():
                model = make_pipeline(clone(transformer), clone(estimator))
                model.fit(X[train_mask], y_train)
                pred = model.predict(X_test)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    balanced_accuracy = balanced_accuracy_score(y_test, pred)
                rows.append(
                    FoldResult(
                        held_out_substrate=substrate,
                        feature=feature_name,
                        model=model_name,
                        n_train=int(train_mask.sum()),
                        n_test=int(len(y_test)),
                        test_labels=",".join(sorted(np.unique(y_test))),
                        accuracy=accuracy_score(y_test, pred),
                        balanced_accuracy=balanced_accuracy,
                        macro_f1=f1_score(y_test, pred, average="macro"),
                    )
                )

                labels = sorted(np.unique(np.concatenate([y_test, pred])))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    matrix = confusion_matrix(y_test, pred, labels=labels)
                report = pd.DataFrame(
                    matrix,
                    index=[f"true:{label}" for label in labels],
                    columns=[f"pred:{label}" for label in labels],
                )
                reports[f"{substrate}__{feature_name}__{model_name}"] = report

    return pd.DataFrame(rows), reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--crop-min",
        type=float,
        default=330.0,
        help="Lowest wavenumber retained. Default follows the report artifact cutoff.",
    )
    parser.add_argument("--crop-max", type=float, default=1800.0)
    parser.add_argument("--min-substrates", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("Workspace/substrate_agnostic_results.csv"))
    parser.add_argument("--confusions-dir", type=Path, default=Path("Workspace/substrate_agnostic_confusions"))
    args = parser.parse_args()

    df, cols = load_dataset(args.data, args.crop_min, args.crop_max, args.min_substrates)
    results, confusions = evaluate(df, cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    args.confusions_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in confusions.items():
        matrix.to_csv(args.confusions_dir / f"{name}.csv")

    print("Dataset:", args.data)
    print("Rows evaluated:", len(df))
    print("Labels:", ", ".join(sorted(df["Label"].unique())))
    print("Substrates:", ", ".join(sorted(df["Substrate"].unique())))
    print(f"Spectral window: {min(map(float, cols)):.3f} to {max(map(float, cols)):.3f} cm^-1")
    print("\nPer-fold results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nMean by feature/model:")
    summary = (
        results.groupby(["feature", "model"])[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to {args.out}")
    print(f"Saved confusion matrices to {args.confusions_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
