from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from unmixing_common import (
    RESULTS_ROOT,
    build_compound_atom_sets,
    build_expanded_reference,
    compute_metrics,
    constant_baseline_atom,
    load_existing_real_records,
    load_original_pure_records,
    load_pt2_mixture_records,
    load_pt2_pure_records,
    load_reference,
    search_best_pair_with_atom_sets,
)


RESULTS_DIR = RESULTS_ROOT / "binary_anchor_operational_calibration"
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
PURE_REJECT_TARGET = 0.995
MIXTURE_COVERAGE_TARGET = 0.98


def labels_to_str(labels: tuple[str, ...]) -> str:
    return " + ".join(labels)


def evaluate_records(records, classes, atom_sets, baseline_atom):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for row_idx, record in enumerate(records):
        best, second = search_best_pair_with_atom_sets(record.spectrum, classes, atom_sets, baseline_atom)
        pred_labels = tuple(best["labels"])
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
        y_pred[row_idx, class_to_i[pred_labels[0]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[1]]] = 1

        pair_sum = float(best["left_sum"]) + float(best["right_sum"]) + 1e-12
        minor_share = float(min(float(best["left_sum"]), float(best["right_sum"])) / pair_sum)
        residual_rel = float(float(best["residual"]) / (np.linalg.norm(record.spectrum) + 1e-12))
        gap_ratio = float(
            (float(second["residual"]) - float(best["residual"])) / (float(best["residual"]) + 1e-12)
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": labels_to_str(record.true_labels),
                "predicted_labels": labels_to_str(pred_labels),
                "correct": bool(pred_labels == record.true_labels),
                "residual_norm": float(best["residual"]),
                "residual_rel": residual_rel,
                "minor_share": minor_share,
                "gap_ratio": gap_ratio,
                "baseline_coef_sum": float(best["baseline_sum"]),
            }
        )

    return compute_metrics(y_true, y_pred), pd.DataFrame(rows), y_true, y_pred


def choose_thresholds(mixture_df: pd.DataFrame, pure_df: pd.DataFrame) -> dict[str, float]:
    residual_values = np.concatenate([mixture_df["residual_rel"].to_numpy(), pure_df["residual_rel"].to_numpy()])
    minor_values = np.concatenate([mixture_df["minor_share"].to_numpy(), pure_df["minor_share"].to_numpy()])
    gap_values = np.concatenate([mixture_df["gap_ratio"].to_numpy(), pure_df["gap_ratio"].to_numpy()])

    residual_candidates = sorted(set(np.round(np.quantile(residual_values, np.linspace(0.55, 1.0, 36)), 4)))
    minor_candidates = sorted(set(np.round(np.quantile(minor_values, np.linspace(0.0, 0.8, 33)), 4)))
    gap_candidates = sorted(set(np.round(np.quantile(gap_values, np.linspace(0.0, 0.8, 33)), 4)))

    best = None
    for residual_thr in residual_candidates:
        for minor_thr in minor_candidates:
            for gap_thr in gap_candidates:
                mixture_accept = (
                    (mixture_df["residual_rel"] <= residual_thr)
                    & (mixture_df["minor_share"] >= minor_thr)
                    & (mixture_df["gap_ratio"] >= gap_thr)
                )
                pure_accept = (
                    (pure_df["residual_rel"] <= residual_thr)
                    & (pure_df["minor_share"] >= minor_thr)
                    & (pure_df["gap_ratio"] >= gap_thr)
                )

                mixture_coverage = float(mixture_accept.mean())
                pure_reject = float((~pure_accept).mean())
                balanced_accuracy = 0.5 * (mixture_coverage + pure_reject)

                if mixture_accept.any():
                    accepted_exact = float(mixture_df.loc[mixture_accept, "correct"].mean())
                else:
                    accepted_exact = 0.0

                score = (
                    int(pure_reject >= PURE_REJECT_TARGET),
                    int(mixture_coverage >= MIXTURE_COVERAGE_TARGET),
                    balanced_accuracy,
                    mixture_coverage,
                    pure_reject,
                    accepted_exact,
                    -float(gap_thr),
                    -float(minor_thr),
                    float(residual_thr),
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "residual_rel_threshold": float(residual_thr),
                        "minor_share_threshold": float(minor_thr),
                        "gap_ratio_threshold": float(gap_thr),
                        "balanced_accuracy": float(balanced_accuracy),
                        "mixture_coverage_calibration": mixture_coverage,
                        "pure_reject_rate_calibration": pure_reject,
                        "accepted_exact_match_calibration": accepted_exact,
                    }

    assert best is not None
    return best


def apply_binary_gate(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return (
        (df["residual_rel"] <= thresholds["residual_rel_threshold"])
        & (df["minor_share"] >= thresholds["minor_share_threshold"])
        & (df["gap_ratio"] >= thresholds["gap_ratio_threshold"])
    )


def summarize_accepted_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, float]:
    coverage = float(accepted.mean())
    if accepted.sum() == 0:
        return {
            "coverage": coverage,
            "reject_rate": 1.0,
            "accepted_exact_match": 0.0,
            "accepted_micro_precision": 0.0,
            "accepted_micro_recall": 0.0,
            "accepted_micro_f1": 0.0,
            "mean_residual_rel_accepted": 0.0,
            "mean_gap_ratio_accepted": 0.0,
        }

    accepted_metrics = compute_metrics(y_true[accepted], y_pred[accepted])
    accepted_df = df.loc[accepted]
    return {
        "coverage": coverage,
        "reject_rate": float(1.0 - coverage),
        "accepted_exact_match": accepted_metrics["exact_match"],
        "accepted_micro_precision": accepted_metrics["micro_precision"],
        "accepted_micro_recall": accepted_metrics["micro_recall"],
        "accepted_micro_f1": accepted_metrics["micro_f1"],
        "mean_residual_rel_accepted": float(accepted_df["residual_rel"].mean()),
        "mean_gap_ratio_accepted": float(accepted_df["gap_ratio"].mean()),
    }


def summarize_pure_gate(df: pd.DataFrame, accepted: np.ndarray) -> dict[str, float]:
    accepted_df = df.loc[accepted]
    return {
        "binary_accept_rate": float(accepted.mean()),
        "binary_reject_rate": float((~accepted).mean()),
        "mean_residual_rel_accepted": float(accepted_df["residual_rel"].mean()) if len(accepted_df) else 0.0,
        "mean_gap_ratio_accepted": float(accepted_df["gap_ratio"].mean()) if len(accepted_df) else 0.0,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)
    original_pure_records = load_original_pure_records(MODE)
    pt2_pure_records = load_pt2_pure_records(wav_axis, MODE)

    existing_metrics, existing_df, existing_y_true, existing_y_pred = evaluate_records(
        existing_records, classes, atom_sets, baseline_atom
    )
    pt2_metrics, pt2_df, pt2_y_true, pt2_y_pred = evaluate_records(
        pt2_records, classes, atom_sets, baseline_atom
    )
    original_pure_metrics, original_pure_df, original_pure_y_true, original_pure_y_pred = evaluate_records(
        original_pure_records, classes, atom_sets, baseline_atom
    )
    pt2_pure_metrics, pt2_pure_df, pt2_pure_y_true, pt2_pure_y_pred = evaluate_records(
        pt2_pure_records, classes, atom_sets, baseline_atom
    )

    pure_df = pd.concat([original_pure_df, pt2_pure_df], ignore_index=True)
    thresholds = choose_thresholds(existing_df, pure_df)

    existing_accept = apply_binary_gate(existing_df, thresholds).to_numpy()
    pt2_accept = apply_binary_gate(pt2_df, thresholds).to_numpy()
    original_pure_accept = apply_binary_gate(original_pure_df, thresholds).to_numpy()
    pt2_pure_accept = apply_binary_gate(pt2_pure_df, thresholds).to_numpy()
    pure_accept = apply_binary_gate(pure_df, thresholds).to_numpy()

    existing_df["binary_accept"] = existing_accept
    pt2_df["binary_accept"] = pt2_accept
    original_pure_df["binary_accept"] = original_pure_accept
    pt2_pure_df["binary_accept"] = pt2_pure_accept
    pure_df["binary_accept"] = pure_accept

    summary = {
        "mode": MODE,
        "n_extra_reps": N_EXTRA_REPS,
        "calibration_targets": {
            "pure_reject_target": PURE_REJECT_TARGET,
            "mixture_coverage_target": MIXTURE_COVERAGE_TARGET,
            "mixture_calibration_dataset": "existing_real",
            "negative_calibration_datasets": ["original_pure", "pt2_pure"],
        },
        "selected_thresholds": {
            k: v for k, v in thresholds.items() if k != "score"
        },
        "anchor_metrics": {
            "existing_real": existing_metrics,
            "pt2_real": pt2_metrics,
            "original_pure_as_forced_binary": original_pure_metrics,
            "pt2_pure_as_forced_binary": pt2_pure_metrics,
        },
        "operational_binary": {
            "existing_real": summarize_accepted_predictions(
                existing_df, existing_y_true, existing_y_pred, existing_accept
            ),
            "pt2_real": summarize_accepted_predictions(pt2_df, pt2_y_true, pt2_y_pred, pt2_accept),
            "original_pure": summarize_pure_gate(original_pure_df, original_pure_accept),
            "pt2_pure": summarize_pure_gate(pt2_pure_df, pt2_pure_accept),
            "all_pure": summarize_pure_gate(pure_df, pure_accept),
        },
    }

    existing_df.to_csv(RESULTS_DIR / "existing_real_predictions.csv", index=False)
    pt2_df.to_csv(RESULTS_DIR / "pt2_real_predictions.csv", index=False)
    original_pure_df.to_csv(RESULTS_DIR / "original_pure_predictions.csv", index=False)
    pt2_pure_df.to_csv(RESULTS_DIR / "pt2_pure_predictions.csv", index=False)
    pd.concat([existing_df, pt2_df, original_pure_df, pt2_pure_df], ignore_index=True).to_csv(
        RESULTS_DIR / "all_predictions.csv", index=False
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Binary anchor operational calibration")
    print(
        f"  thresholds: residual_rel<={thresholds['residual_rel_threshold']:.4f} "
        f"minor_share>={thresholds['minor_share_threshold']:.4f} "
        f"gap_ratio>={thresholds['gap_ratio_threshold']:.4f}"
    )
    print(
        f"  existing_real coverage={summary['operational_binary']['existing_real']['coverage']:.3f} "
        f"accepted_exact={summary['operational_binary']['existing_real']['accepted_exact_match']:.3f}"
    )
    print(
        f"  pt2_real      coverage={summary['operational_binary']['pt2_real']['coverage']:.3f} "
        f"accepted_exact={summary['operational_binary']['pt2_real']['accepted_exact_match']:.3f}"
    )
    print(
        f"  all_pure      binary_reject_rate={summary['operational_binary']['all_pure']['binary_reject_rate']:.3f}"
    )
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
