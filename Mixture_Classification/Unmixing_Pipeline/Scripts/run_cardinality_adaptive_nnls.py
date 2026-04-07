from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from unmixing_common import (
    SpectrumRecord,
    build_compound_atom_sets,
    build_expanded_reference,
    compute_metrics,
    constant_baseline_atom,
    load_existing_real_records,
    load_original_pure_records,
    load_pt2_mixture_records,
    load_pt2_pure_records,
    load_reference,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "cardinality_adaptive_nnls"
)
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
MAX_SUPPORT_SIZE = 3
TOP_PAIR_CANDIDATES = 10
TOP_SINGLE_CANDIDATES = 6
MAX_SHORTLIST_SIZE = 8
SIZE_PENALTIES = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]
MIN_SHARE_THRESHOLDS = [0.0, 0.03, 0.05, 0.08, 0.10]


def build_targets(records: list[SpectrumRecord], classes: list[str]) -> np.ndarray:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    for row_idx, record in enumerate(records):
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
    return y_true


def compute_support_fit(
    spectrum: np.ndarray,
    support: tuple[str, ...],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
):
    support = tuple(sorted(support))
    design_parts = []
    support_slices: dict[str, slice] = {}
    start = 0
    for label in support:
        atoms = atom_sets[label]
        design_parts.append(atoms)
        stop = start + atoms.shape[1]
        support_slices[label] = slice(start, stop)
        start = stop
    design_parts.append(baseline_atom)
    design = np.column_stack(design_parts)

    coef, _ = nnls(design, spectrum)
    recon = design @ coef
    residual_norm = float(np.linalg.norm(spectrum - recon))
    residual_rel = float(residual_norm / (np.linalg.norm(spectrum) + 1e-12))

    compound_coef = {}
    for label in support:
        compound_coef[label] = float(coef[support_slices[label]].sum())
    baseline_coef = float(coef[-1])
    total_compound_coef = float(sum(compound_coef.values()))
    shares = {
        label: (value / total_compound_coef if total_compound_coef > 0 else 0.0)
        for label, value in compound_coef.items()
    }
    minor_share = min(shares.values()) if shares else 0.0

    return {
        "support": support,
        "support_size": len(support),
        "residual_norm": residual_norm,
        "residual_rel": residual_rel,
        "compound_coef": compound_coef,
        "compound_shares": shares,
        "total_compound_coef": total_compound_coef,
        "baseline_coef": baseline_coef,
        "minor_share": float(minor_share),
    }


def build_record_candidate_table(
    record: SpectrumRecord,
    classes: list[str],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
) -> dict[tuple[str, ...], dict]:
    candidates: dict[tuple[str, ...], dict] = {}

    for label in classes:
        support = (label,)
        candidates[support] = compute_support_fit(record.spectrum, support, atom_sets, baseline_atom)

    pair_scores = []
    for support in combinations(classes, 2):
        fit = compute_support_fit(record.spectrum, support, atom_sets, baseline_atom)
        candidates[support] = fit
        pair_scores.append((fit["residual_rel"], support))

    top_singles = sorted(
        ((entry["residual_rel"], support[0]) for support, entry in candidates.items() if len(support) == 1),
        key=lambda item: item[0],
    )[:TOP_SINGLE_CANDIDATES]
    top_pairs = sorted(pair_scores, key=lambda item: item[0])[:TOP_PAIR_CANDIDATES]

    shortlist = {label for _score, label in top_singles}
    for _score, support in top_pairs:
        shortlist.update(support)

    shortlist = sorted(shortlist)
    if len(shortlist) > MAX_SHORTLIST_SIZE:
        ranked = {}
        for idx, label in enumerate(shortlist):
            single_rank = next(
                (rank for rank, (_score, candidate) in enumerate(top_singles) if candidate == label),
                TOP_SINGLE_CANDIDATES,
            )
            pair_rank = next(
                (
                    rank
                    for rank, (_score, pair_support) in enumerate(top_pairs)
                    if label in pair_support
                ),
                TOP_PAIR_CANDIDATES,
            )
            ranked[label] = min(single_rank, pair_rank)
        shortlist = sorted(shortlist, key=lambda label: (ranked[label], label))[:MAX_SHORTLIST_SIZE]

    for support_size in range(3, MAX_SUPPORT_SIZE + 1):
        for support in combinations(shortlist, support_size):
            if support not in candidates:
                candidates[support] = compute_support_fit(
                    record.spectrum, support, atom_sets, baseline_atom
                )

    return candidates


def select_prediction(
    candidate_table: dict[tuple[str, ...], dict],
    size_penalty: float,
    min_share_threshold: float,
) -> dict:
    best = None
    best_score = None
    for fit in candidate_table.values():
        score = fit["residual_rel"] + size_penalty * (fit["support_size"] - 1)
        if best_score is None or score < best_score:
            best_score = score
            best = fit

    assert best is not None
    predicted = tuple(
        sorted(
            label
            for label, share in best["compound_shares"].items()
            if share >= min_share_threshold
        )
    )
    if not predicted:
        predicted = (max(best["compound_coef"], key=best["compound_coef"].get),)

    return {
        **best,
        "selected_score": float(best_score),
        "predicted_labels": predicted,
        "predicted_cardinality": len(predicted),
    }


def summarize_dataset_from_candidates(
    records: list[SpectrumRecord],
    classes: list[str],
    candidate_tables: list[dict[tuple[str, ...], dict]],
    size_penalty: float,
    min_share_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true = build_targets(records, classes)
    y_pred = np.zeros_like(y_true)
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    rows = []

    for row_idx, (record, candidate_table) in enumerate(zip(records, candidate_tables)):
        selected = select_prediction(candidate_table, size_penalty, min_share_threshold)
        for label in selected["predicted_labels"]:
            y_pred[row_idx, class_to_i[label]] = 1

        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "selected_support": " + ".join(selected["support"]),
                "predicted_labels": " + ".join(selected["predicted_labels"]),
                "selected_support_size": selected["support_size"],
                "predicted_cardinality": selected["predicted_cardinality"],
                "selected_score": selected["selected_score"],
                "residual_norm": selected["residual_norm"],
                "residual_rel": selected["residual_rel"],
                "minor_share": selected["minor_share"],
                "baseline_coef": selected["baseline_coef"],
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    metrics["zero_prediction_rate"] = float(np.mean(y_pred.sum(axis=1) == 0))
    cardinality, counts = np.unique(y_pred.sum(axis=1), return_counts=True)
    metrics["predicted_cardinality_hist"] = {
        str(int(cardinality_value)): int(count)
        for cardinality_value, count in zip(cardinality, counts)
    }
    return metrics, pd.DataFrame(rows), y_pred


def per_source_summary(
    records: list[SpectrumRecord],
    classes: list[str],
    y_pred: np.ndarray,
    pred_df: pd.DataFrame,
) -> list[dict]:
    y_true = build_targets(records, classes)
    rows = []
    for source in sorted({record.source for record in records}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        metrics["zero_prediction_rate"] = float(np.mean(y_pred[idx].sum(axis=1) == 0))
        metrics["source"] = source
        metrics["samples"] = len(idx)
        metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        metrics["mean_predicted_labels"] = float(y_pred[idx].sum(axis=1).mean())
        metrics["mean_residual_rel"] = float(pred_df.iloc[idx]["residual_rel"].mean())
        rows.append(metrics)
    return rows


def score_config(existing_metrics: dict[str, float], pure_metrics: dict[str, float]) -> tuple[float, ...]:
    combined_exact = 0.7 * existing_metrics["exact_match"] + 0.3 * pure_metrics["exact_match"]
    return (
        combined_exact,
        existing_metrics["exact_match"],
        pure_metrics["exact_match"],
        existing_metrics["micro_f1"],
        -abs(existing_metrics["mean_predicted_labels"] - 2.0),
        -abs(pure_metrics["mean_predicted_labels"] - 1.0),
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    eval_sets = {
        "existing_real": load_existing_real_records(MODE),
        "original_pure": load_original_pure_records(MODE),
        "pt2_pure": load_pt2_pure_records(wav_axis, MODE),
        "pt2_real": load_pt2_mixture_records(wav_axis, MODE),
    }

    candidate_tables = {
        name: [
            build_record_candidate_table(record, classes, atom_sets, baseline_atom)
            for record in records
        ]
        for name, records in eval_sets.items()
    }

    tuning_rows = []
    best = None
    for size_penalty in SIZE_PENALTIES:
        for min_share_threshold in MIN_SHARE_THRESHOLDS:
            existing_metrics, _, _ = summarize_dataset_from_candidates(
                eval_sets["existing_real"],
                classes,
                candidate_tables["existing_real"],
                size_penalty,
                min_share_threshold,
            )
            pure_metrics, _, _ = summarize_dataset_from_candidates(
                eval_sets["original_pure"],
                classes,
                candidate_tables["original_pure"],
                size_penalty,
                min_share_threshold,
            )
            score = score_config(existing_metrics, pure_metrics)
            row = {
                "size_penalty": size_penalty,
                "min_share_threshold": min_share_threshold,
                "score_combined_exact": score[0],
                "existing_exact": existing_metrics["exact_match"],
                "existing_micro_f1": existing_metrics["micro_f1"],
                "existing_mean_predicted_labels": existing_metrics["mean_predicted_labels"],
                "pure_exact": pure_metrics["exact_match"],
                "pure_micro_f1": pure_metrics["micro_f1"],
                "pure_mean_predicted_labels": pure_metrics["mean_predicted_labels"],
            }
            tuning_rows.append(row)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "size_penalty": size_penalty,
                    "min_share_threshold": min_share_threshold,
                }

    assert best is not None
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["score_combined_exact", "existing_exact", "pure_exact"],
        ascending=[False, False, False],
    )
    tuning_df.to_csv(RESULTS_DIR / "calibration_tuning.csv", index=False)

    summaries = {}
    all_prediction_frames = []
    residuals_for_reject = []

    for name, records in eval_sets.items():
        metrics, pred_df, y_pred = summarize_dataset_from_candidates(
            records,
            classes,
            candidate_tables[name],
            best["size_penalty"],
            best["min_share_threshold"],
        )
        pred_df.to_csv(RESULTS_DIR / f"{name}_predictions.csv", index=False)
        all_prediction_frames.append(pred_df)
        summaries[name] = metrics
        summaries[name]["mean_residual_rel"] = float(pred_df["residual_rel"].mean())
        summaries[name]["mean_baseline_coef"] = float(pred_df["baseline_coef"].mean())
        if name == "pt2_real":
            summaries[name]["per_mixture"] = per_source_summary(records, classes, y_pred, pred_df)
        if name in {"existing_real", "original_pure"}:
            residuals_for_reject.extend(pred_df["residual_rel"].tolist())

    residual_reject_threshold = float(np.quantile(residuals_for_reject, 0.99))
    summaries["reject_thresholds"] = {
        "residual_rel_99pct_calibration": residual_reject_threshold,
    }
    for name in eval_sets:
        pred_df = pd.read_csv(RESULTS_DIR / f"{name}_predictions.csv")
        summaries[name]["reject_rate"] = float((pred_df["residual_rel"] > residual_reject_threshold).mean())

    pd.concat(all_prediction_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "all_predictions.csv", index=False
    )

    summary = {
        "mode": MODE,
        "n_extra_reps": N_EXTRA_REPS,
        "max_support_size": MAX_SUPPORT_SIZE,
        "shortlist_strategy": {
            "top_single_candidates": TOP_SINGLE_CANDIDATES,
            "top_pair_candidates": TOP_PAIR_CANDIDATES,
            "max_shortlist_size": MAX_SHORTLIST_SIZE,
        },
        "selected_hyperparameters": {
            "size_penalty": best["size_penalty"],
            "min_share_threshold": best["min_share_threshold"],
        },
        **summaries,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Selected hyperparameters")
    print(
        f"  size_penalty={best['size_penalty']} "
        f"min_share_threshold={best['min_share_threshold']}"
    )
    for name in ("existing_real", "original_pure", "pt2_pure", "pt2_real"):
        section = summaries[name]
        print(
            f"{name:13s} exact={section['exact_match']:.3f} "
            f"micro_f1={section['micro_f1']:.3f} "
            f"pred/sample={section['mean_predicted_labels']:.2f} "
            f"reject_rate={section['reject_rate']:.3f}"
        )

    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
