from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

from run_deep_binary_coefficient_regressor import (
    build_synthetic_dataset,
    normalize_dictionary,
    normalize_spectrum,
    per_source_summary,
    predict_shares,
    set_seed,
    split_synthetic_dataset,
    summarize_real_records,
    summarize_synthetic_split,
    train_model,
)
from unmixing_common import (
    RESULTS_ROOT,
    SpectrumRecord,
    build_compound_atom_sets,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    constant_baseline_atom,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "deep_hybrid_pair_rerank"
)
DEEP_BASELINE_RESULTS_DIR = RESULTS_ROOT / "deep_binary_coefficient_regressor"
PAIR_BENCHMARK_RESULTS_DIR = RESULTS_ROOT / "pair_nnls_replicate_dictionary"

MODE = "baseline_corrected"
N_EXTRA_REPS = 9
ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


def score_all_pairs(
    spectrum: np.ndarray,
    classes: list[str],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
) -> list[dict]:
    rows = []
    for left_label, right_label in combinations(classes, 2):
        left_atoms = atom_sets[left_label]
        right_atoms = atom_sets[right_label]
        design = np.column_stack([left_atoms, right_atoms, baseline_atom])
        coef, _ = nnls(design, spectrum)
        recon = design @ coef
        residual = float(np.linalg.norm(spectrum - recon))
        n_left = left_atoms.shape[1]
        n_right = right_atoms.shape[1]
        rows.append(
            {
                "labels": tuple(sorted((left_label, right_label))),
                "residual": residual,
                "left_sum": float(coef[:n_left].sum()),
                "right_sum": float(coef[n_left : n_left + n_right].sum()),
                "baseline_sum": float(coef[-1]),
            }
        )
    return rows


def choose_pair_with_hybrid_score(
    pair_rows: list[dict],
    deep_shares: np.ndarray,
    class_to_i: dict[str, int],
    alpha: float,
) -> dict:
    residuals = np.array([row["residual"] for row in pair_rows], dtype=np.float64)
    residual_span = residuals.max() - residuals.min()
    if residual_span <= 0:
        residual_norm = np.zeros_like(residuals)
    else:
        residual_norm = (residuals - residuals.min()) / residual_span

    pair_prior = np.array(
        [
            deep_shares[class_to_i[row["labels"][0]]] + deep_shares[class_to_i[row["labels"][1]]]
            for row in pair_rows
        ],
        dtype=np.float64,
    )
    combined = residual_norm - alpha * pair_prior
    best_idx = int(np.argmin(combined))
    row = dict(pair_rows[best_idx])
    row["deep_pair_prior"] = float(pair_prior[best_idx])
    row["hybrid_score"] = float(combined[best_idx])
    return row


def summarize_hybrid_predictions(
    records: list[SpectrumRecord],
    classes: list[str],
    pair_scores_by_record: list[list[dict]],
    deep_share_matrix: np.ndarray,
    alpha: float,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for idx, record in enumerate(records):
        for label in record.true_labels:
            y_true[idx, class_to_i[label]] = 1

        selected = choose_pair_with_hybrid_score(
            pair_scores_by_record[idx], deep_share_matrix[idx], class_to_i, alpha
        )
        pred_labels = selected["labels"]
        y_pred[idx, class_to_i[pred_labels[0]]] = 1
        y_pred[idx, class_to_i[pred_labels[1]]] = 1

        ranked = np.argsort(deep_share_matrix[idx])[::-1][:5]
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(pred_labels),
                "pair_residual_norm": float(selected["residual"]),
                "baseline_coef_sum": float(selected["baseline_sum"]),
                "deep_pair_prior": float(selected["deep_pair_prior"]),
                "hybrid_score": float(selected["hybrid_score"]),
                "deep_top1_label": classes[ranked[0]],
                "deep_top1_share": float(deep_share_matrix[idx, ranked[0]]),
                "deep_top2_label": classes[ranked[1]],
                "deep_top2_share": float(deep_share_matrix[idx, ranked[1]]),
                "deep_top3_label": classes[ranked[2]],
                "deep_top3_share": float(deep_share_matrix[idx, ranked[2]]),
            }
        )

    return compute_metrics(y_true, y_pred), pd.DataFrame(rows), y_pred


def precompute_pair_scores(
    records: list[SpectrumRecord],
    classes: list[str],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
) -> list[list[dict]]:
    return [score_all_pairs(record.spectrum, classes, atom_sets, baseline_atom) for record in records]


def tune_alpha(
    records: list[SpectrumRecord],
    classes: list[str],
    pair_scores_by_record: list[list[dict]],
    deep_share_matrix: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for alpha in ALPHAS:
        metrics, _, _ = summarize_hybrid_predictions(
            records, classes, pair_scores_by_record, deep_share_matrix, alpha
        )
        row = {"alpha": alpha, **metrics}
        rows.append(row)
        score = (metrics["exact_match"], metrics["micro_f1"])
        if best is None or score > best["score"]:
            best = {"alpha": alpha, "metrics": metrics, "score": score}

    assert best is not None
    return best, pd.DataFrame(rows).sort_values(
        ["exact_match", "micro_f1", "alpha"], ascending=[False, False, True]
    )


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, mean_dictionary = build_mean_dictionary(expanded_ref, MODE)
    decoder = normalize_dictionary(mean_dictionary).T

    dataset = build_synthetic_dataset(expanded_ref, classes, MODE)
    split_idx = split_synthetic_dataset(dataset)
    deep_model, history_df = train_model(dataset, split_idx, decoder, device)

    train_metrics, train_pred_df = summarize_synthetic_split(
        dataset, "train", split_idx, deep_model, classes, device
    )
    val_metrics, val_pred_df = summarize_synthetic_split(
        dataset, "val", split_idx, deep_model, classes, device
    )
    test_metrics, test_pred_df = summarize_synthetic_split(
        dataset, "test", split_idx, deep_model, classes, device
    )

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)
    existing_deep_metrics, existing_deep_pred_df, _ = summarize_real_records(
        existing_records, classes, deep_model, device
    )
    pt2_deep_metrics, pt2_deep_pred_df, _ = summarize_real_records(
        pt2_records, classes, deep_model, device
    )

    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    existing_pair_scores = precompute_pair_scores(existing_records, classes, atom_sets, baseline_atom)
    pt2_pair_scores = precompute_pair_scores(pt2_records, classes, atom_sets, baseline_atom)

    existing_spectra = np.vstack([normalize_spectrum(record.spectrum) for record in existing_records]).astype(np.float32)
    pt2_spectra = np.vstack([normalize_spectrum(record.spectrum) for record in pt2_records]).astype(np.float32)
    existing_deep_shares = predict_shares(deep_model, existing_spectra, device)
    pt2_deep_shares = predict_shares(deep_model, pt2_spectra, device)

    best, tuning_df = tune_alpha(existing_records, classes, existing_pair_scores, existing_deep_shares)
    existing_hybrid_metrics, existing_hybrid_pred_df, _ = summarize_hybrid_predictions(
        existing_records, classes, existing_pair_scores, existing_deep_shares, best["alpha"]
    )
    pt2_hybrid_metrics, pt2_hybrid_pred_df, pt2_hybrid_y_pred = summarize_hybrid_predictions(
        pt2_records, classes, pt2_pair_scores, pt2_deep_shares, best["alpha"]
    )

    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    tuning_df.to_csv(RESULTS_DIR / "alpha_tuning_existing_real.csv", index=False)
    train_pred_df.to_csv(RESULTS_DIR / "synthetic_train_predictions.csv", index=False)
    val_pred_df.to_csv(RESULTS_DIR / "synthetic_val_predictions.csv", index=False)
    test_pred_df.to_csv(RESULTS_DIR / "synthetic_test_predictions.csv", index=False)
    existing_deep_pred_df.to_csv(RESULTS_DIR / "existing_real_deep_predictions.csv", index=False)
    pt2_deep_pred_df.to_csv(RESULTS_DIR / "pt2_real_deep_predictions.csv", index=False)
    existing_hybrid_pred_df.to_csv(RESULTS_DIR / "existing_real_hybrid_predictions.csv", index=False)
    pt2_hybrid_pred_df.to_csv(RESULTS_DIR / "pt2_real_hybrid_predictions.csv", index=False)

    summary = {
        "mode": MODE,
        "model_type": "deep_hybrid_pair_rerank",
        "deep_baseline_results_dir": str(DEEP_BASELINE_RESULTS_DIR),
        "pair_benchmark_results_dir": str(PAIR_BENCHMARK_RESULTS_DIR),
        "device": device.type,
        "n_extra_representatives_per_compound": N_EXTRA_REPS,
        "selected_alpha": best["alpha"],
        "deep_model": {
            "synthetic": {
                "train_top2_binary": train_metrics,
                "val_top2_binary": val_metrics,
                "test_top2_binary": test_metrics,
            },
            "existing_real_top2_binary": existing_deep_metrics,
            "pt2_real_top2_binary": pt2_deep_metrics,
        },
        "hybrid_model": {
            "existing_real_top2_binary": existing_hybrid_metrics,
            "pt2_real_top2_binary": {
                "overall": pt2_hybrid_metrics,
                "per_mixture": per_source_summary(pt2_records, classes, pt2_hybrid_y_pred),
            },
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRunning deep hybrid pair rerank with mode={MODE}")
    print(f"  selected alpha={best['alpha']}")
    print(
        f"  deep existing exact={existing_deep_metrics['exact_match']:.3f} "
        f"micro_f1={existing_deep_metrics['micro_f1']:.3f}"
    )
    print(
        f"  hybrid existing exact={existing_hybrid_metrics['exact_match']:.3f} "
        f"micro_f1={existing_hybrid_metrics['micro_f1']:.3f}"
    )
    print(
        f"  hybrid pt2 exact={pt2_hybrid_metrics['exact_match']:.3f} "
        f"micro_f1={pt2_hybrid_metrics['micro_f1']:.3f}"
    )
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
