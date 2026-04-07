from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from unmixing_common import (
    build_expanded_reference,
    compute_metrics,
    constant_baseline_atom,
    farthest_point_subset,
    load_existing_real_records,
    load_original_pure_records,
    load_pt2_mixture_records,
    load_pt2_pure_records,
    load_reference,
    preprocess_spectrum,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "full_library_sparse_support_selection"
)
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
ALPHAS = [3e-4, 1e-3, 3e-3, 1e-2]
L1_RATIOS = [0.5, 0.8, 0.95]
SHARE_THRESHOLDS = [0.05, 0.08, 0.10, 0.12]
CAL_EXISTING_MAX = 240
CAL_PURE_MAX = 240


def build_atom_dictionary(ref_df: pd.DataFrame, mode: str, n_extra_reps: int):
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    classes = sorted(ref_df["Label"].unique())
    atom_columns = []
    atom_names = []
    atom_to_compound = []

    for label in classes:
        spectra = ref_df.loc[ref_df["Label"] == label, wav_cols].to_numpy(dtype=np.float64)
        proc = np.vstack([preprocess_spectrum(spec, mode) for spec in spectra])
        mean_atom = proc.mean(axis=0, keepdims=True)
        reps = farthest_point_subset(proc, n_extra_reps)
        atoms = np.vstack([mean_atom, reps])
        for idx, atom in enumerate(atoms):
            atom_columns.append(atom.astype(np.float64))
            atom_names.append(f"{label}__atom_{idx}")
            atom_to_compound.append(label)

    atom_columns.append(constant_baseline_atom(proc.shape[1]).ravel())
    atom_names.append("baseline_constant")
    atom_to_compound.append("__baseline__")

    design = np.column_stack(atom_columns)
    return classes, design, atom_names, atom_to_compound


def fit_coefficients(records, design: np.ndarray, alpha: float, l1_ratio: float):
    coef = np.zeros((len(records), design.shape[1]), dtype=np.float64)
    residual_rel = np.zeros(len(records), dtype=np.float64)
    for i, record in enumerate(records):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=True,
            fit_intercept=False,
            max_iter=50000,
            tol=1e-4,
            selection="cyclic",
        )
        model.fit(design, record.spectrum)
        row = model.coef_.clip(min=0.0)
        coef[i] = row
        recon = design @ row
        residual_rel[i] = float(np.linalg.norm(record.spectrum - recon) / (np.linalg.norm(record.spectrum) + 1e-12))
    return coef, residual_rel


def aggregate_compound_coefficients(
    coef: np.ndarray, classes: list[str], atom_to_compound: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    compound_coef = np.zeros((coef.shape[0], len(classes)), dtype=np.float64)
    baseline_coef = np.zeros(coef.shape[0], dtype=np.float64)
    for atom_idx, label in enumerate(atom_to_compound):
        if label == "__baseline__":
            baseline_coef += coef[:, atom_idx]
        else:
            compound_coef[:, class_to_i[label]] += coef[:, atom_idx]
    return compound_coef, baseline_coef


def coefficients_to_support(compound_coef: np.ndarray, share_threshold: float) -> np.ndarray:
    total = compound_coef.sum(axis=1, keepdims=True)
    shares = np.divide(compound_coef, total, out=np.zeros_like(compound_coef), where=total > 0)
    y_pred = (shares >= share_threshold).astype(int)
    empty = y_pred.sum(axis=1) == 0
    if np.any(empty):
        top = np.argmax(compound_coef[empty], axis=1)
        y_pred[empty, top] = 1
    return y_pred


def build_targets(records, classes: list[str]) -> np.ndarray:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    for i, record in enumerate(records):
        for label in record.true_labels:
            y_true[i, class_to_i[label]] = 1
    return y_true


def score_config(
    existing_metrics: dict[str, float],
    pure_metrics: dict[str, float],
) -> tuple[float, float, float, float]:
    combined_exact = 0.7 * existing_metrics["exact_match"] + 0.3 * pure_metrics["exact_match"]
    return (
        combined_exact,
        existing_metrics["exact_match"],
        pure_metrics["exact_match"],
        existing_metrics["micro_f1"],
    )


def summarize_dataset(
    records,
    classes: list[str],
    y_pred: np.ndarray,
    residual_rel: np.ndarray,
    baseline_coef: np.ndarray,
):
    y_true = build_targets(records, classes)
    metrics = compute_metrics(y_true, y_pred)
    metrics["zero_prediction_rate"] = float(np.mean(y_pred.sum(axis=1) == 0))
    cardinality, counts = np.unique(y_pred.sum(axis=1), return_counts=True)
    metrics["predicted_cardinality_hist"] = {str(int(k)): int(v) for k, v in zip(cardinality, counts)}
    metrics["mean_residual_rel"] = float(np.mean(residual_rel))
    metrics["mean_baseline_coef"] = float(np.mean(baseline_coef))
    return metrics


def per_source_summary(
    records,
    classes: list[str],
    y_pred: np.ndarray,
    residual_rel: np.ndarray,
):
    y_true = build_targets(records, classes)
    rows = []
    for source in sorted({r.source for r in records}):
        idx = [i for i, r in enumerate(records) if r.source == source]
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        metrics["source"] = source
        metrics["samples"] = len(idx)
        metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        metrics["mean_residual_rel"] = float(np.mean(residual_rel[idx]))
        rows.append(metrics)
    return rows


def predictions_dataframe(
    records,
    classes: list[str],
    compound_coef: np.ndarray,
    y_pred: np.ndarray,
    residual_rel: np.ndarray,
    baseline_coef: np.ndarray,
) -> pd.DataFrame:
    total = compound_coef.sum(axis=1, keepdims=True)
    shares = np.divide(compound_coef, total, out=np.zeros_like(compound_coef), where=total > 0)
    rows = []
    for record, coef, share, pred, resid, base in zip(
        records, compound_coef, shares, y_pred, residual_rel, baseline_coef
    ):
        ranked = np.argsort(coef)[::-1][:5]
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(classes[i] for i in np.where(pred == 1)[0]),
                "predicted_cardinality": int(pred.sum()),
                "residual_rel": float(resid),
                "baseline_coef": float(base),
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
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_ref, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(original_ref, wav_axis)
    classes, design, atom_names, atom_to_compound = build_atom_dictionary(
        expanded_ref, MODE, N_EXTRA_REPS
    )

    existing_records = load_existing_real_records(MODE)
    original_pure_records = load_original_pure_records(MODE)
    pt2_pure_records = load_pt2_pure_records(wav_axis, MODE)
    pt2_mixture_records = load_pt2_mixture_records(wav_axis, MODE)

    rng = np.random.default_rng(42)
    cal_existing_idx = rng.choice(
        len(existing_records),
        size=min(CAL_EXISTING_MAX, len(existing_records)),
        replace=False,
    )
    cal_pure_idx = rng.choice(
        len(original_pure_records),
        size=min(CAL_PURE_MAX, len(original_pure_records)),
        replace=False,
    )
    calibration_existing = [existing_records[i] for i in np.sort(cal_existing_idx)]
    calibration_pure = [original_pure_records[i] for i in np.sort(cal_pure_idx)]
    calibration_records = calibration_existing + calibration_pure
    calibration_tuning = []
    best = None

    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            coef_cal, resid_cal = fit_coefficients(calibration_records, design, alpha, l1_ratio)
            compound_cal, baseline_cal = aggregate_compound_coefficients(coef_cal, classes, atom_to_compound)

            n_existing = len(calibration_existing)
            for share_threshold in SHARE_THRESHOLDS:
                pred_cal = coefficients_to_support(compound_cal, share_threshold)
                pred_existing = pred_cal[:n_existing]
                pred_pure = pred_cal[n_existing:]

                existing_metrics = summarize_dataset(
                    calibration_existing,
                    classes,
                    pred_existing,
                    resid_cal[:n_existing],
                    baseline_cal[:n_existing],
                )
                pure_metrics = summarize_dataset(
                    calibration_pure,
                    classes,
                    pred_pure,
                    resid_cal[n_existing:],
                    baseline_cal[n_existing:],
                )
                score = score_config(existing_metrics, pure_metrics)
                row = {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "share_threshold": share_threshold,
                    "score_combined_exact": score[0],
                    "calibration_n_existing": len(calibration_existing),
                    "calibration_n_pure": len(calibration_pure),
                    "existing_exact": existing_metrics["exact_match"],
                    "existing_micro_f1": existing_metrics["micro_f1"],
                    "existing_mean_predicted_labels": existing_metrics["mean_predicted_labels"],
                    "pure_exact": pure_metrics["exact_match"],
                    "pure_micro_f1": pure_metrics["micro_f1"],
                    "pure_mean_predicted_labels": pure_metrics["mean_predicted_labels"],
                }
                calibration_tuning.append(row)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "share_threshold": share_threshold,
                    }

    assert best is not None
    tuning_df = pd.DataFrame(calibration_tuning).sort_values(
        ["score_combined_exact", "existing_exact", "pure_exact"], ascending=[False, False, False]
    )
    tuning_df.to_csv(RESULTS_DIR / "calibration_tuning.csv", index=False)

    eval_sets = {
        "existing_real": existing_records,
        "original_pure": original_pure_records,
        "pt2_pure": pt2_pure_records,
        "pt2_real": pt2_mixture_records,
    }

    all_prediction_frames = []
    summaries = {}
    calibration_residuals = []

    for name, records in eval_sets.items():
        coef, resid = fit_coefficients(records, design, best["alpha"], best["l1_ratio"])
        compound_coef, baseline_coef = aggregate_compound_coefficients(coef, classes, atom_to_compound)
        y_pred = coefficients_to_support(compound_coef, best["share_threshold"])
        summaries[name] = summarize_dataset(records, classes, y_pred, resid, baseline_coef)
        if name == "pt2_real":
            summaries[name]["per_mixture"] = per_source_summary(records, classes, y_pred, resid)

        df = predictions_dataframe(records, classes, compound_coef, y_pred, resid, baseline_coef)
        all_prediction_frames.append(df)
        df.to_csv(RESULTS_DIR / f"{name}_predictions.csv", index=False)

        if name in {"existing_real", "original_pure"}:
            calibration_residuals.extend(list(resid))

    residual_reject_threshold = float(np.quantile(calibration_residuals, 0.99))
    summaries["reject_thresholds"] = {
        "residual_rel_99pct_calibration": residual_reject_threshold,
    }
    for name in eval_sets:
        df = pd.read_csv(RESULTS_DIR / f"{name}_predictions.csv")
        summaries[name]["reject_rate"] = float((df["residual_rel"] > residual_reject_threshold).mean())

    pd.concat(all_prediction_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "all_predictions.csv", index=False
    )

    summary = {
        "mode": MODE,
        "n_extra_reps": N_EXTRA_REPS,
        "selected_hyperparameters": {
            "alpha": best["alpha"],
            "l1_ratio": best["l1_ratio"],
            "share_threshold": best["share_threshold"],
        },
        "calibration_subset_sizes": {
            "existing_real": len(calibration_existing),
            "original_pure": len(calibration_pure),
        },
        "atom_dictionary": {
            "n_total_atoms": int(design.shape[1]),
            "n_compounds": len(classes),
            "atom_names_head": atom_names[:10],
        },
        **summaries,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Selected hyperparameters")
    print(
        f"  alpha={best['alpha']} l1_ratio={best['l1_ratio']} "
        f"share_threshold={best['share_threshold']}"
    )
    for name in ("existing_real", "original_pure", "pt2_pure", "pt2_real"):
        sec = summaries[name]
        print(
            f"{name:13s} exact={sec['exact_match']:.3f} "
            f"micro_f1={sec['micro_f1']:.3f} "
            f"precision={sec['micro_precision']:.3f} "
            f"recall={sec['micro_recall']:.3f} "
            f"pred/sample={sec['mean_predicted_labels']:.2f} "
            f"reject_rate={sec['reject_rate']:.3f}"
        )
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
