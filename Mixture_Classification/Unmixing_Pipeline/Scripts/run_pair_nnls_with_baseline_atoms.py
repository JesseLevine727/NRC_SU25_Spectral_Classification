from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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
    Path(__file__).resolve().parents[1] / "Results" / "pair_nnls_with_baseline_atoms"
)
PAIR_NNLS_RESULTS_DIR = RESULTS_ROOT / "exhaustive_pair_nnls"


def bernstein_basis(length: int, degree: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    basis = []
    for k in range(degree + 1):
        b = comb(degree, k) * (x**k) * ((1.0 - x) ** (degree - k))
        b = b.astype(np.float64)
        norm = np.linalg.norm(b)
        if norm > 0:
            b = b / norm
        basis.append(b)
    return np.column_stack(basis)
def evaluate_records(
    records: list[SpectrumRecord],
    classes: list[str],
    dictionary: np.ndarray,
    baseline_atoms: np.ndarray,
):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    pair_defs = list(combinations(range(len(classes)), 2))

    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for row_idx, record in enumerate(records):
        best = None
        for i, j in pair_defs:
            compound_atoms = dictionary[:, [i, j]]
            design = np.column_stack([compound_atoms, baseline_atoms])
            coefs, _ = nnls(design, record.spectrum)
            recon = design @ coefs
            residual = float(np.linalg.norm(record.spectrum - recon))
            if best is None or residual < best["residual"]:
                best = {
                    "pair_idx": (i, j),
                    "compound_coefs": coefs[:2],
                    "baseline_coefs": coefs[2:],
                    "residual": residual,
                }

        assert best is not None
        pred_labels = tuple(sorted((classes[best["pair_idx"][0]], classes[best["pair_idx"][1]])))
        y_true[row_idx, class_to_i[record.true_labels[0]]] = 1
        y_true[row_idx, class_to_i[record.true_labels[1]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[0]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[1]]] = 1

        baseline_strength = float(best["baseline_coefs"].sum())
        compound_strength = float(best["compound_coefs"].sum())
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(pred_labels),
                "residual_norm": best["residual"],
                "compound_coef_sum": compound_strength,
                "baseline_coef_sum": baseline_strength,
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    pred_df = pd.DataFrame(rows)
    per_source = []
    for source in sorted({record.source for record in records if record.dataset == "pt2_real"}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        src_metrics = compute_metrics(y_true[idx], y_pred[idx])
        src_metrics["source"] = source
        src_metrics["samples"] = len(idx)
        src_metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        src_metrics["mean_residual_norm"] = float(pred_df.iloc[idx]["residual_norm"].mean())
        src_metrics["mean_baseline_coef_sum"] = float(pred_df.iloc[idx]["baseline_coef_sum"].mean())
        per_source.append(src_metrics)

    return metrics, per_source, pred_df


def run_config(mode: str, degree: int) -> dict:
    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, dictionary = build_mean_dictionary(expanded_ref, mode)
    baseline_atoms = bernstein_basis(dictionary.shape[0], degree)

    existing_records = load_existing_real_records(mode)
    pt2_records = load_pt2_mixture_records(wav_axis, mode)

    existing_metrics, _, existing_pred_df = evaluate_records(
        existing_records, classes, dictionary, baseline_atoms
    )
    pt2_metrics, pt2_per_source, pt2_pred_df = evaluate_records(
        pt2_records, classes, dictionary, baseline_atoms
    )

    config_dir = RESULTS_DIR / f"{mode}_bernstein_deg_{degree}"
    config_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([existing_pred_df, pt2_pred_df], ignore_index=True).to_csv(
        config_dir / "predictions.csv", index=False
    )

    summary = {
        "mode": mode,
        "baseline_basis": "bernstein",
        "degree": degree,
        "n_baseline_atoms": degree + 1,
        "pair_nnls_results_dir": str(PAIR_NNLS_RESULTS_DIR),
        "existing_real": existing_metrics,
        "pt2_real": {
            "overall": pt2_metrics,
            "per_mixture": pt2_per_source,
        },
    }
    (config_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"mode={mode:18s} degree={degree} "
        f"existing exact={existing_metrics['exact_match']:.3f} "
        f"micro_f1={existing_metrics['micro_f1']:.3f} "
        f"pt2 exact={pt2_metrics['exact_match']:.3f} "
        f"pt2 micro_f1={pt2_metrics['micro_f1']:.3f}"
    )

    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    best = None

    for mode in ("raw", "baseline_corrected"):
        for degree in (0, 1, 2, 3):
            summary = run_config(mode, degree)
            all_results.append(summary)
            score = (
                summary["existing_real"]["exact_match"],
                summary["existing_real"]["micro_f1"],
                summary["pt2_real"]["overall"]["exact_match"],
                summary["pt2_real"]["overall"]["micro_f1"],
            )
            if best is None or score > best["score"]:
                best = {"score": score, "summary": summary}

    assert best is not None
    out = {"best": best["summary"], "all_results": all_results}
    (RESULTS_DIR / "all_results.json").write_text(json.dumps(out, indent=2))

    best_summary = best["summary"]
    print("\nBest configuration")
    print(
        f"  mode={best_summary['mode']} degree={best_summary['degree']} "
        f"n_baseline_atoms={best_summary['n_baseline_atoms']}"
    )
    print(
        f"  existing exact={best_summary['existing_real']['exact_match']:.3f} "
        f"micro_f1={best_summary['existing_real']['micro_f1']:.3f}"
    )
    print(
        f"  pt2 exact={best_summary['pt2_real']['overall']['exact_match']:.3f} "
        f"micro_f1={best_summary['pt2_real']['overall']['micro_f1']:.3f}"
    )
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
