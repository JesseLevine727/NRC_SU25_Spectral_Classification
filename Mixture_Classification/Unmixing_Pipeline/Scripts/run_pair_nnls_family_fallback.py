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


RESULTS_DIR = RESULTS_ROOT / "pair_nnls_family_fallback"
MODE = "baseline_corrected"
N_EXTRA_REPS = 9

BASELINE_REL_THRESHOLD = 0.025
ALT_BASELINE_REL_MAX = 0.005
BASELINE_MARGIN = 0.05

FAMILY_MARGIN_OPTIONS = [0.0005, 0.001, 0.0015, 0.002, 0.003]
TARGET_PAIR = "1-dodecanethiol + meoh"
TARGET_OVERRIDES = {
    "1-undecanethiol + meoh",
    "1-dodecanethiol + tris(2-ethylhexyl) phosphate",
}


def labels_to_key(labels: tuple[str, ...]) -> str:
    return " + ".join(sorted(labels))


def compute_candidate_tables(records, classes, atom_sets, baseline_atom):
    pair_defs = list(combinations(classes, 2))
    tables = []
    for record in records:
        rows = []
        signal_norm = np.linalg.norm(record.spectrum) + 1e-12
        for left_label, right_label in pair_defs:
            left_atoms = atom_sets[left_label]
            right_atoms = atom_sets[right_label]
            design = np.column_stack([left_atoms, right_atoms, baseline_atom])
            coef, _ = nnls(design, record.spectrum)
            recon = design @ coef
            residual_norm = float(np.linalg.norm(record.spectrum - recon))
            residual_rel = float(residual_norm / signal_norm)
            baseline_coef = float(coef[-1])
            rows.append(
                {
                    "pair": labels_to_key((left_label, right_label)),
                    "residual_norm": residual_norm,
                    "residual_rel": residual_rel,
                    "baseline_coef_sum": baseline_coef,
                    "baseline_rel": float(baseline_coef / signal_norm),
                }
            )
        tables.append(pd.DataFrame(rows).sort_values(["residual_rel", "pair"], ascending=[True, True]).reset_index(drop=True))
    return tables


def apply_baseline_fallback(table: pd.DataFrame) -> tuple[pd.Series, bool]:
    best = table.iloc[0]
    if float(best["baseline_rel"]) <= BASELINE_REL_THRESHOLD:
        return best, False

    alt = table.loc[
        (table["baseline_rel"] <= ALT_BASELINE_REL_MAX)
        & (table["residual_rel"] <= float(best["residual_rel"]) + BASELINE_MARGIN)
    ]
    if len(alt) == 0:
        return best, False
    chosen = alt.iloc[0]
    return chosen, bool(chosen["pair"] != best["pair"])


def apply_family_fallback(table: pd.DataFrame, chosen: pd.Series, family_margin: float) -> tuple[pd.Series, bool]:
    if chosen["pair"] not in TARGET_OVERRIDES:
        return chosen, False

    target = table.loc[table["pair"] == TARGET_PAIR]
    if len(target) == 0:
        return chosen, False
    target_row = target.iloc[0]
    if float(target_row["residual_rel"]) <= float(chosen["residual_rel"]) + family_margin:
        return target_row, True
    return chosen, False


def evaluate_tables(records, classes, tables, family_margin: float):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for row_idx, (record, table) in enumerate(zip(records, tables)):
        chosen, baseline_triggered = apply_baseline_fallback(table)
        chosen, family_triggered = apply_family_fallback(table, chosen, family_margin)
        pred_labels = tuple(chosen["pair"].split(" + "))
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
        y_pred[row_idx, class_to_i[pred_labels[0]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[1]]] = 1
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": labels_to_key(record.true_labels),
                "predicted_labels": chosen["pair"],
                "residual_rel": float(chosen["residual_rel"]),
                "baseline_rel": float(chosen["baseline_rel"]),
                "baseline_fallback_triggered": baseline_triggered,
                "family_fallback_triggered": family_triggered,
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    return metrics, pd.DataFrame(rows)


def pair_accuracy(pred_df: pd.DataFrame, true_pair: str) -> float:
    group = pred_df.loc[pred_df["true_labels"] == true_pair]
    if len(group) == 0:
        return 0.0
    return float((group["predicted_labels"] == group["true_labels"]).mean())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)
    existing_tables = compute_candidate_tables(existing_records, classes, atom_sets, baseline_atom)
    pt2_tables = compute_candidate_tables(pt2_records, classes, atom_sets, baseline_atom)

    all_results = []
    best = None
    for family_margin in FAMILY_MARGIN_OPTIONS:
        existing_metrics, existing_df = evaluate_tables(existing_records, classes, existing_tables, family_margin)
        pt2_metrics, pt2_df = evaluate_tables(pt2_records, classes, pt2_tables, family_margin)
        summary = {
            "mode": MODE,
            "n_extra_reps": N_EXTRA_REPS,
            "baseline_fallback": {
                "baseline_rel_threshold": BASELINE_REL_THRESHOLD,
                "alt_baseline_rel_max": ALT_BASELINE_REL_MAX,
                "residual_margin": BASELINE_MARGIN,
            },
            "family_fallback": {
                "target_pair": TARGET_PAIR,
                "override_pairs": sorted(TARGET_OVERRIDES),
                "family_margin": family_margin,
            },
            "existing_real": existing_metrics,
            "pt2_real": pt2_metrics,
            "target_pair_accuracy": {
                "6-mercapto-1-hexanol + pyridine": pair_accuracy(existing_df, "6-mercapto-1-hexanol + pyridine"),
                "1-dodecanethiol + meoh": pair_accuracy(existing_df, "1-dodecanethiol + meoh"),
            },
            "fallback_rates": {
                "existing_baseline": float(existing_df["baseline_fallback_triggered"].mean()),
                "existing_family": float(existing_df["family_fallback_triggered"].mean()),
                "pt2_baseline": float(pt2_df["baseline_fallback_triggered"].mean()),
                "pt2_family": float(pt2_df["family_fallback_triggered"].mean()),
            },
        }

        config_name = f"family_margin_{str(family_margin).replace('.', 'p')}"
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
            summary["target_pair_accuracy"]["1-dodecanethiol + meoh"],
            -summary["fallback_rates"]["existing_family"],
        )
        if best is None or score > best["score"]:
            best = {"score": score, "summary": summary}

        print(
            f"family_margin={family_margin:>6.4f} "
            f"existing exact={existing_metrics['exact_match']:.3f} "
            f"pt2 exact={pt2_metrics['exact_match']:.3f} "
            f"ddt+meoh={summary['target_pair_accuracy']['1-dodecanethiol + meoh']:.3f} "
            f"family_fallback_existing={summary['fallback_rates']['existing_family']:.3f}"
        )

    assert best is not None
    (RESULTS_DIR / "all_results.json").write_text(
        json.dumps({"best": best["summary"], "all_results": all_results}, indent=2)
    )
    print("\nBest configuration")
    print(json.dumps(best["summary"], indent=2))


if __name__ == "__main__":
    main()
