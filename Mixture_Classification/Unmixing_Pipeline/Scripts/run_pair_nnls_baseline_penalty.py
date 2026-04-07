from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from unmixing_common import (
    RESULTS_ROOT,
    build_compound_atom_sets,
    build_expanded_reference,
    compute_metrics,
    constant_baseline_atom,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
)


RESULTS_DIR = RESULTS_ROOT / "pair_nnls_baseline_penalty"
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
BASELINE_PENALTY_OPTIONS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def evaluate_dataset(records, classes, atom_sets, baseline_atom, baseline_penalty: float):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    pair_defs = list(combinations(classes, 2))
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for row_idx, record in enumerate(records):
        best = None
        for left_label, right_label in pair_defs:
            left_atoms = atom_sets[left_label]
            right_atoms = atom_sets[right_label]
            design = np.column_stack([left_atoms, right_atoms, baseline_atom])
            coef, _ = nnls(design, record.spectrum)
            recon = design @ coef
            residual_norm = float(np.linalg.norm(record.spectrum - recon))
            residual_rel = float(residual_norm / (np.linalg.norm(record.spectrum) + 1e-12))
            baseline_coef = float(coef[-1])
            baseline_rel = float(baseline_coef / (np.linalg.norm(record.spectrum) + 1e-12))
            score = residual_rel + baseline_penalty * baseline_rel

            if best is None or score < best["score"]:
                n_left = left_atoms.shape[1]
                n_right = right_atoms.shape[1]
                best = {
                    "labels": tuple(sorted((left_label, right_label))),
                    "score": float(score),
                    "residual_norm": residual_norm,
                    "residual_rel": residual_rel,
                    "baseline_coef_sum": baseline_coef,
                    "baseline_rel": baseline_rel,
                    "left_atom_coef_sum": float(coef[:n_left].sum()),
                    "right_atom_coef_sum": float(coef[n_left : n_left + n_right].sum()),
                }

        assert best is not None
        pred_labels = best["labels"]
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
        y_pred[row_idx, class_to_i[pred_labels[0]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[1]]] = 1
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(pred_labels),
                "score": best["score"],
                "residual_norm": best["residual_norm"],
                "residual_rel": best["residual_rel"],
                "baseline_coef_sum": best["baseline_coef_sum"],
                "baseline_rel": best["baseline_rel"],
                "left_atom_coef_sum": best["left_atom_coef_sum"],
                "right_atom_coef_sum": best["right_atom_coef_sum"],
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    pred_df = pd.DataFrame(rows)
    return metrics, pred_df


def dominant_pair_accuracy(pred_df: pd.DataFrame, true_pair: str) -> float:
    group = pred_df.loc[pred_df["true_labels"] == true_pair]
    if len(group) == 0:
        return 0.0
    return float((group["true_labels"] == group["predicted_labels"]).mean())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)

    all_results = []
    best = None
    for baseline_penalty in BASELINE_PENALTY_OPTIONS:
        existing_metrics, existing_df = evaluate_dataset(
            existing_records, classes, atom_sets, baseline_atom, baseline_penalty
        )
        pt2_metrics, pt2_df = evaluate_dataset(
            pt2_records, classes, atom_sets, baseline_atom, baseline_penalty
        )

        summary = {
            "mode": MODE,
            "n_extra_reps": N_EXTRA_REPS,
            "baseline_penalty": baseline_penalty,
            "existing_real": existing_metrics,
            "pt2_real": pt2_metrics,
            "target_pair_accuracy": {
                "6-mercapto-1-hexanol + pyridine": dominant_pair_accuracy(
                    existing_df, "6-mercapto-1-hexanol + pyridine"
                ),
                "1-dodecanethiol + meoh": dominant_pair_accuracy(existing_df, "1-dodecanethiol + meoh"),
            },
        }

        config_name = f"baseline_penalty_{str(baseline_penalty).replace('.', 'p')}"
        config_dir = RESULTS_DIR / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([existing_df, pt2_df], ignore_index=True).to_csv(config_dir / "predictions.csv", index=False)
        (config_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        all_results.append(summary)
        score = (
            summary["existing_real"]["exact_match"],
            summary["existing_real"]["micro_f1"],
            summary["pt2_real"]["exact_match"],
            summary["pt2_real"]["micro_f1"],
            summary["target_pair_accuracy"]["6-mercapto-1-hexanol + pyridine"],
            summary["target_pair_accuracy"]["1-dodecanethiol + meoh"],
        )
        if best is None or score > best["score"]:
            best = {"score": score, "summary": summary}

        print(
            f"penalty={baseline_penalty:>4.2f} "
            f"existing exact={existing_metrics['exact_match']:.3f} "
            f"micro_f1={existing_metrics['micro_f1']:.3f} "
            f"pt2 exact={pt2_metrics['exact_match']:.3f} "
            f"target mh+pyr={summary['target_pair_accuracy']['6-mercapto-1-hexanol + pyridine']:.3f} "
            f"target ddt+meoh={summary['target_pair_accuracy']['1-dodecanethiol + meoh']:.3f}"
        )

    assert best is not None
    (RESULTS_DIR / "all_results.json").write_text(
        json.dumps({"best": best["summary"], "all_results": all_results}, indent=2)
    )

    print("\nBest configuration")
    print(json.dumps(best["summary"], indent=2))


if __name__ == "__main__":
    main()
