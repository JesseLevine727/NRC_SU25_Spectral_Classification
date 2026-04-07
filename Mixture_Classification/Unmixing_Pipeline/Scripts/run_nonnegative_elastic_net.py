from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from unmixing_common import (
    RESULTS_ROOT,
    SpectrumRecord,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "nonnegative_elastic_net"
)
PAIR_NNLS_RESULTS_DIR = RESULTS_ROOT / "exhaustive_pair_nnls"

ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
L1_RATIOS = [0.2, 0.5, 0.8, 0.95, 1.0]
SHARE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
def build_targets(records: list[SpectrumRecord], classes: list[str]) -> np.ndarray:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    for idx, record in enumerate(records):
        y_true[idx, class_to_i[record.true_labels[0]]] = 1
        y_true[idx, class_to_i[record.true_labels[1]]] = 1
    return y_true


def fit_coefficients(records: list[SpectrumRecord], dictionary: np.ndarray, alpha: float, l1_ratio: float):
    coefficients = np.zeros((len(records), dictionary.shape[1]), dtype=np.float64)
    residuals = np.zeros(len(records), dtype=np.float64)
    signals = np.column_stack([record.spectrum for record in records])
    for idx, record in enumerate(records):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=True,
            fit_intercept=False,
            max_iter=50000,
            tol=1e-4,
            selection="cyclic",
        )
        model.fit(dictionary, record.spectrum)
        coef = model.coef_.clip(min=0.0)
        coefficients[idx] = coef
        residuals[idx] = float(np.linalg.norm(record.spectrum - dictionary @ coef))
    return coefficients, residuals


def coefficients_to_predictions(coefficients: np.ndarray, share_threshold: float) -> np.ndarray:
    coef_sum = coefficients.sum(axis=1, keepdims=True)
    shares = np.divide(coefficients, coef_sum, out=np.zeros_like(coefficients), where=coef_sum > 0)
    pred = (shares >= share_threshold).astype(int)

    empty = pred.sum(axis=1) == 0
    if np.any(empty):
        top_idx = np.argmax(coefficients[empty], axis=1)
        pred[empty, top_idx] = 1
    return pred


def summarize_predictions(
    records: list[SpectrumRecord],
    classes: list[str],
    coefficients: np.ndarray,
    residuals: np.ndarray,
    share_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true = build_targets(records, classes)
    y_pred = coefficients_to_predictions(coefficients, share_threshold)
    metrics = compute_metrics(y_true, y_pred)
    metrics["zero_prediction_rate"] = float(np.mean(y_pred.sum(axis=1) == 0))
    shares = np.divide(
        coefficients,
        coefficients.sum(axis=1, keepdims=True),
        out=np.zeros_like(coefficients),
        where=coefficients.sum(axis=1, keepdims=True) > 0,
    )

    rows = []
    for record, coef, share, pred, residual in zip(records, coefficients, shares, y_pred, residuals):
        ranked = np.argsort(coef)[::-1][:5]
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(classes[i] for i in np.where(pred == 1)[0]),
                "residual_norm": float(residual),
                "top1_label": classes[ranked[0]],
                "top1_coef": float(coef[ranked[0]]),
                "top1_share": float(share[ranked[0]]),
                "top2_label": classes[ranked[1]],
                "top2_coef": float(coef[ranked[1]]),
                "top2_share": float(share[ranked[1]]),
                "top3_label": classes[ranked[2]],
                "top3_coef": float(coef[ranked[2]]),
                "top3_share": float(share[ranked[2]]),
                "top4_label": classes[ranked[3]],
                "top4_coef": float(coef[ranked[3]]),
                "top4_share": float(share[ranked[3]]),
                "top5_label": classes[ranked[4]],
                "top5_coef": float(coef[ranked[4]]),
                "top5_share": float(share[ranked[4]]),
            }
        )
    return metrics, pd.DataFrame(rows), y_pred


def tune_on_existing(records: list[SpectrumRecord], classes: list[str], dictionary: np.ndarray):
    y_true = build_targets(records, classes)
    rows = []
    best = None

    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            coefficients, residuals = fit_coefficients(records, dictionary, alpha, l1_ratio)
            for share_threshold in SHARE_THRESHOLDS:
                y_pred = coefficients_to_predictions(coefficients, share_threshold)
                metrics = compute_metrics(y_true, y_pred)
                score = (
                    metrics["exact_match"],
                    -abs(metrics["mean_predicted_labels"] - 2.0),
                    metrics["micro_f1"],
                )
                row = {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "share_threshold": share_threshold,
                    **metrics,
                    "mean_residual_norm": float(residuals.mean()),
                }
                rows.append(row)
                if best is None or score > best["score"]:
                    best = {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "share_threshold": share_threshold,
                        "score": score,
                        "metrics": metrics,
                    }

    assert best is not None
    return best, pd.DataFrame(rows).sort_values(
        ["exact_match", "micro_f1"], ascending=[False, False]
    )


def per_source_summary(records: list[SpectrumRecord], classes: list[str], y_pred: np.ndarray) -> list[dict]:
    y_true = build_targets(records, classes)
    summaries = []
    for source in sorted({record.source for record in records if record.dataset == "pt2_real"}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        metrics["source"] = source
        metrics["samples"] = len(idx)
        metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        summaries.append(metrics)
    return summaries


def run_mode(mode: str) -> dict:
    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, dictionary = build_mean_dictionary(expanded_ref, mode)

    existing_records = load_existing_real_records(mode)
    pt2_records = load_pt2_mixture_records(wav_axis, mode)

    best, tuning_df = tune_on_existing(existing_records, classes, dictionary)

    existing_coef, existing_resid = fit_coefficients(
        existing_records, dictionary, best["alpha"], best["l1_ratio"]
    )
    pt2_coef, pt2_resid = fit_coefficients(
        pt2_records, dictionary, best["alpha"], best["l1_ratio"]
    )

    existing_metrics, existing_pred_df, _existing_y_pred = summarize_predictions(
        existing_records, classes, existing_coef, existing_resid, best["share_threshold"]
    )
    pt2_metrics, pt2_pred_df, pt2_y_pred = summarize_predictions(
        pt2_records, classes, pt2_coef, pt2_resid, best["share_threshold"]
    )

    mode_dir = RESULTS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(mode_dir / "tuning_results_existing_real.csv", index=False)
    existing_pred_df.to_csv(mode_dir / "existing_real_predictions.csv", index=False)
    pt2_pred_df.to_csv(mode_dir / "pt2_real_predictions.csv", index=False)

    summary = {
        "mode": mode,
        "dictionary_classes": classes,
        "pair_nnls_results_dir": str(PAIR_NNLS_RESULTS_DIR),
        "selected_hyperparameters": {
            "alpha": best["alpha"],
            "l1_ratio": best["l1_ratio"],
            "share_threshold": best["share_threshold"],
        },
        "existing_real": existing_metrics,
        "pt2_real": {
            "overall": pt2_metrics,
            "per_mixture": per_source_summary(pt2_records, classes, pt2_y_pred),
        },
    }
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRunning non-negative elastic net with mode={mode}")
    print(
        f"  selected alpha={best['alpha']} l1_ratio={best['l1_ratio']} "
        f"share_threshold={best['share_threshold']}"
    )
    print(
        f"  existing_real exact={existing_metrics['exact_match']:.3f} "
        f"micro_f1={existing_metrics['micro_f1']:.3f} "
        f"pred/sample={existing_metrics['mean_predicted_labels']:.2f}"
    )
    print(
        f"  pt2_real      exact={pt2_metrics['exact_match']:.3f} "
        f"micro_f1={pt2_metrics['micro_f1']:.3f} "
        f"pred/sample={pt2_metrics['mean_predicted_labels']:.2f}"
    )
    for row in summary["pt2_real"]["per_mixture"]:
        print(
            f"    {row['source']:6s} {row['true_labels']:<45s} "
            f"exact={row['exact_match']:.3f} "
            f"micro_f1={row['micro_f1']:.3f} "
            f"pred/sample={row['mean_predicted_labels']:.2f}"
        )

    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for mode in ("raw", "baseline_corrected"):
        all_results[mode] = run_mode(mode)
    (RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
