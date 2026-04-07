from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from unmixing_common import (
    PT2_DIR,
    PURE_LABEL_ALIASES,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    load_existing_real_records,
    load_original_pure_records,
    load_pt2_mixture_records,
    load_reference,
    load_txt_spectrum,
    load_pt2_pure_records,
    RESULTS_ROOT,
    SpectrumRecord,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "pair_nnls_reject_and_split_validation"
)
BASELINE_RESULTS_DIR = RESULTS_ROOT / "exhaustive_pair_nnls"
MODE = "baseline_corrected"


def search_pairs(spectrum: np.ndarray, dictionary: np.ndarray, pair_defs: list[tuple[int, int]]):
    best = None
    second = None
    for i, j in pair_defs:
        atoms = dictionary[:, [i, j]]
        coefs, _ = nnls(atoms, spectrum)
        recon = atoms @ coefs
        residual = float(np.linalg.norm(spectrum - recon))
        entry = {
            "pair_idx": (i, j),
            "coefs": coefs,
            "residual": residual,
        }
        if best is None or residual < best["residual"]:
            second = best
            best = entry
        elif second is None or residual < second["residual"]:
            second = entry
    assert best is not None
    if second is None:
        second = best
    return best, second


def evaluate_mixture_records(records: list[SpectrumRecord], classes: list[str], dictionary: np.ndarray):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    pair_defs = list(combinations(range(len(classes)), 2))

    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []
    for idx, record in enumerate(records):
        best, second = search_pairs(record.spectrum, dictionary, pair_defs)
        pred_labels = tuple(sorted((classes[best["pair_idx"][0]], classes[best["pair_idx"][1]])))
        y_true[idx, class_to_i[record.true_labels[0]]] = 1
        y_true[idx, class_to_i[record.true_labels[1]]] = 1
        y_pred[idx, class_to_i[pred_labels[0]]] = 1
        y_pred[idx, class_to_i[pred_labels[1]]] = 1

        coef_sum = float(best["coefs"].sum()) + 1e-12
        minor_share = float(np.min(best["coefs"]) / coef_sum)
        residual_rel = float(best["residual"] / (np.linalg.norm(record.spectrum) + 1e-12))
        gap_ratio = float((second["residual"] - best["residual"]) / (best["residual"] + 1e-12))
        correct = pred_labels == record.true_labels
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(pred_labels),
                "correct": bool(correct),
                "residual_norm": float(best["residual"]),
                "residual_rel": residual_rel,
                "minor_share": minor_share,
                "gap_ratio": gap_ratio,
            }
        )

    return compute_metrics(y_true, y_pred), pd.DataFrame(rows), y_true, y_pred


def evaluate_pure_rows(pure_rows: list[SpectrumRecord], classes: list[str], dictionary: np.ndarray) -> pd.DataFrame:
    pair_defs = list(combinations(range(len(classes)), 2))
    rows = []
    for record in pure_rows:
        best, second = search_pairs(record.spectrum, dictionary, pair_defs)
        pred_labels = tuple(sorted((classes[best["pair_idx"][0]], classes[best["pair_idx"][1]])))
        coef_sum = float(best["coefs"].sum()) + 1e-12
        minor_share = float(np.min(best["coefs"]) / coef_sum)
        residual_rel = float(best["residual"] / (np.linalg.norm(record.spectrum) + 1e-12))
        gap_ratio = float((second["residual"] - best["residual"]) / (best["residual"] + 1e-12))
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_label": record.true_labels[0],
                "predicted_pair": " + ".join(pred_labels),
                "residual_norm": float(best["residual"]),
                "residual_rel": residual_rel,
                "minor_share": minor_share,
                "gap_ratio": gap_ratio,
            }
        )
    return pd.DataFrame(rows)


def choose_reject_thresholds(mixture_df: pd.DataFrame, pure_df: pd.DataFrame) -> dict[str, float]:
    residual_candidates = sorted(
        set(np.round(np.concatenate([mixture_df["residual_rel"].to_numpy(), pure_df["residual_rel"].to_numpy()]), 4))
    )
    minor_candidates = sorted(
        set(np.round(np.concatenate([mixture_df["minor_share"].to_numpy(), pure_df["minor_share"].to_numpy()]), 4))
    )

    best = None
    y_true = np.concatenate(
        [
            np.ones(len(mixture_df), dtype=int),
            np.zeros(len(pure_df), dtype=int),
        ]
    )

    for residual_thr in residual_candidates:
        for minor_thr in minor_candidates:
            y_pred = np.concatenate(
                [
                    ((mixture_df["residual_rel"] <= residual_thr) & (mixture_df["minor_share"] >= minor_thr)).astype(int),
                    ((pure_df["residual_rel"] <= residual_thr) & (pure_df["minor_share"] >= minor_thr)).astype(int),
                ]
            )
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            score = 0.5 * (tpr + tnr)
            if best is None or score > best["balanced_accuracy"]:
                best = {
                    "residual_rel_threshold": float(residual_thr),
                    "minor_share_threshold": float(minor_thr),
                    "balanced_accuracy": float(score),
                    "tpr_mixture_accept": float(tpr),
                    "tnr_pure_reject": float(tnr),
                }
    assert best is not None
    return best


def apply_reject(mixture_df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float]:
    accepted = (
        (mixture_df["residual_rel"] <= thresholds["residual_rel_threshold"])
        & (mixture_df["minor_share"] >= thresholds["minor_share_threshold"])
    )
    coverage = float(accepted.mean())
    accepted_df = mixture_df.loc[accepted]
    if len(accepted_df) == 0:
        return {"coverage": 0.0, "accepted_exact_match": 0.0}
    return {
        "coverage": coverage,
        "accepted_exact_match": float(accepted_df["correct"].mean()),
    }


def build_split_reference(ref_df: pd.DataFrame, wav_axis: np.ndarray, seed: int, n_per_class: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pure_dir_name, label in PURE_LABEL_ALIASES.items():
        txt_paths = sorted((PT2_DIR / pure_dir_name / "txt").glob("*.txt"))
        indices = rng.choice(len(txt_paths), size=min(n_per_class, len(txt_paths)), replace=False)
        for idx in sorted(indices):
            spectrum = load_txt_spectrum(txt_paths[idx], wav_axis)
            row = {"Label": label}
            row.update({wav: val for wav, val in zip(wav_axis, spectrum)})
            rows.append(row)
    extra_df = pd.DataFrame(rows, columns=ref_df.columns)
    return pd.concat([ref_df, extra_df], ignore_index=True)


def split_validation(ref_df: pd.DataFrame, wav_axis: np.ndarray, pt2_records: list[SpectrumRecord]) -> list[dict]:
    rows = []
    for seed in range(5):
        split_ref = build_split_reference(ref_df, wav_axis, seed=seed, n_per_class=18)
        classes, dictionary = build_dictionary(split_ref, MODE)
        metrics, preds_df, _, _ = evaluate_mixture_records(pt2_records, classes, dictionary)
        rows.append(
            {
                "seed": seed,
                "pt2_exact_match": metrics["exact_match"],
                "pt2_micro_f1": metrics["micro_f1"],
                "pt2_mean_residual_rel": float(preds_df["residual_rel"].mean()),
            }
        )
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, dictionary = build_mean_dictionary(expanded_ref, MODE)

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)
    pure_rows = load_original_pure_records(MODE) + load_pt2_pure_records(wav_axis, MODE)

    existing_metrics, existing_pred_df, _, _ = evaluate_mixture_records(existing_records, classes, dictionary)
    pt2_metrics, pt2_pred_df, _, _ = evaluate_mixture_records(pt2_records, classes, dictionary)
    pure_pred_df = evaluate_pure_rows(pure_rows, classes, dictionary)

    thresholds = choose_reject_thresholds(existing_pred_df, pure_pred_df)
    reject_existing = apply_reject(existing_pred_df, thresholds)
    reject_pt2 = apply_reject(pt2_pred_df, thresholds)
    pure_accept_rate = float(
        (
            (pure_pred_df["residual_rel"] <= thresholds["residual_rel_threshold"])
            & (pure_pred_df["minor_share"] >= thresholds["minor_share_threshold"])
        ).mean()
    )

    split_rows = split_validation(ref_df, wav_axis, pt2_records)

    existing_pred_df.to_csv(RESULTS_DIR / "existing_mixture_predictions.csv", index=False)
    pt2_pred_df.to_csv(RESULTS_DIR / "pt2_mixture_predictions.csv", index=False)
    pure_pred_df.to_csv(RESULTS_DIR / "pure_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(RESULTS_DIR / "pt2_split_validation.csv", index=False)

    summary = {
        "mode": MODE,
        "baseline_reference_results_dir": str(BASELINE_RESULTS_DIR),
        "existing_real": existing_metrics,
        "pt2_real": pt2_metrics,
        "reject_thresholds": thresholds,
        "reject_existing_real": reject_existing,
        "reject_pt2_real": reject_pt2,
        "pure_binary_accept_rate": pure_accept_rate,
        "pt2_split_validation": split_rows,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Mode: {MODE}")
    print(
        f"Existing real mixtures: exact={existing_metrics['exact_match']:.3f} "
        f"micro_f1={existing_metrics['micro_f1']:.3f}"
    )
    print(
        f"PT2 real mixtures:      exact={pt2_metrics['exact_match']:.3f} "
        f"micro_f1={pt2_metrics['micro_f1']:.3f}"
    )
    print("\nReject thresholds")
    print(
        f"  residual_rel <= {thresholds['residual_rel_threshold']:.4f}"
        f", minor_share >= {thresholds['minor_share_threshold']:.4f}"
    )
    print(
        f"  calibration balanced_acc={thresholds['balanced_accuracy']:.3f} "
        f"mixture_accept_tpr={thresholds['tpr_mixture_accept']:.3f} "
        f"pure_reject_tnr={thresholds['tnr_pure_reject']:.3f}"
    )
    print("\nAfter reject rule")
    print(
        f"  existing_real coverage={reject_existing['coverage']:.3f} "
        f"accepted_exact={reject_existing['accepted_exact_match']:.3f}"
    )
    print(
        f"  pt2_real      coverage={reject_pt2['coverage']:.3f} "
        f"accepted_exact={reject_pt2['accepted_exact_match']:.3f}"
    )
    print(f"  pure_accept_rate={pure_accept_rate:.3f}")

    print("\nPT2 split validation")
    for row in split_rows:
        print(
            f"  seed={row['seed']} exact={row['pt2_exact_match']:.3f} "
            f"micro_f1={row['pt2_micro_f1']:.3f} "
            f"mean_residual_rel={row['pt2_mean_residual_rel']:.4f}"
        )

    print(f"\nSaved outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
