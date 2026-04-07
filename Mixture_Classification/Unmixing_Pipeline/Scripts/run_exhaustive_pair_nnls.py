from __future__ import annotations

import json
from itertools import combinations

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


RESULTS_DIR = RESULTS_ROOT / "exhaustive_pair_nnls"


def evaluate_pair_search(records: list[SpectrumRecord], classes: list[str], dictionary: np.ndarray):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    pair_defs = list(combinations(range(len(classes)), 2))
    predictions = []
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)

    for row_idx, record in enumerate(records):
        best = None
        for i, j in pair_defs:
            atoms = dictionary[:, [i, j]]
            coefs, residual = nnls(atoms, record.spectrum)
            recon = atoms @ coefs
            residual_norm = float(np.linalg.norm(record.spectrum - recon))
            if best is None or residual_norm < best["residual"]:
                best = {
                    "pair_idx": (i, j),
                    "pair_labels": (classes[i], classes[j]),
                    "coefs": coefs,
                    "residual": residual_norm,
                }

        assert best is not None
        true_left, true_right = record.true_labels
        pred_left, pred_right = best["pair_labels"]
        y_true[row_idx, class_to_i[true_left]] = 1
        y_true[row_idx, class_to_i[true_right]] = 1
        y_pred[row_idx, class_to_i[pred_left]] = 1
        y_pred[row_idx, class_to_i[pred_right]] = 1

        predictions.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(best["pair_labels"]),
                "coef_1": float(best["coefs"][0]),
                "coef_2": float(best["coefs"][1]),
                "residual_norm": best["residual"],
            }
        )

    summary = compute_metrics(y_true, y_pred)
    per_source = []
    for source in sorted({record.source for record in records if record.dataset == "pt2_real"}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        src_metrics = compute_metrics(y_true[idx], y_pred[idx])
        src_metrics["source"] = source
        src_metrics["samples"] = len(idx)
        src_metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        src_metrics["mean_residual_norm"] = float(
            np.mean([predictions[i]["residual_norm"] for i in idx])
        )
        per_source.append(src_metrics)

    return summary, per_source, pd.DataFrame(predictions)
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    expanded_ref.to_csv(RESULTS_DIR / "reference_v2_plus_pt2.csv", index=False)

    all_results = {}
    for mode in ("raw", "baseline_corrected"):
        print(f"\nRunning exhaustive pair NNLS with mode={mode}")
        classes, dictionary = build_mean_dictionary(expanded_ref, mode)

        existing_records = load_existing_real_records(mode)
        pt2_records = load_pt2_mixture_records(wav_axis, mode)

        existing_summary, _, existing_predictions = evaluate_pair_search(
            existing_records, classes, dictionary
        )
        pt2_summary, pt2_per_source, pt2_predictions = evaluate_pair_search(
            pt2_records, classes, dictionary
        )

        mode_dir = RESULTS_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([existing_predictions, pt2_predictions], ignore_index=True).to_csv(
            mode_dir / "predictions.csv", index=False
        )

        result = {
            "mode": mode,
            "classes": classes,
            "existing_real": existing_summary,
            "pt2_real": {
                "overall": pt2_summary,
                "per_mixture": pt2_per_source,
            },
        }
        (mode_dir / "summary.json").write_text(json.dumps(result, indent=2))
        all_results[mode] = result

        print(
            f"  existing_real exact={existing_summary['exact_match']:.3f} "
            f"micro_f1={existing_summary['micro_f1']:.3f}"
        )
        print(
            f"  pt2_real      exact={pt2_summary['exact_match']:.3f} "
            f"micro_f1={pt2_summary['micro_f1']:.3f}"
        )
        for row in pt2_per_source:
            print(
                f"    {row['source']:6s} {row['true_labels']:<45s} "
                f"exact={row['exact_match']:.3f} "
                f"micro_f1={row['micro_f1']:.3f} "
                f"residual={row['mean_residual_norm']:.3f}"
            )

    (RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
