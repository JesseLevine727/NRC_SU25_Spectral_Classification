from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from unmixing_common import (
    build_expanded_reference,
    build_compound_atom_sets,
    constant_baseline_atom,
    evaluate_pair_records_with_atom_sets,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "pair_nnls_replicate_dictionary"
)
MODE = "baseline_corrected"
EXTRA_REP_OPTIONS = [0, 2, 4, 9]


def run_config(n_extra_reps: int) -> dict:
    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, n_extra_reps)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)

    existing_metrics, _, existing_pred_df = evaluate_pair_records_with_atom_sets(
        existing_records, classes, atom_sets, baseline_atom
    )
    pt2_metrics, pt2_per_source, pt2_pred_df = evaluate_pair_records_with_atom_sets(
        pt2_records, classes, atom_sets, baseline_atom
    )

    config_dir = RESULTS_DIR / f"baseline_corrected_extra_reps_{n_extra_reps}"
    config_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([existing_pred_df, pt2_pred_df], ignore_index=True).to_csv(
        config_dir / "predictions.csv", index=False
    )

    summary = {
        "mode": MODE,
        "n_extra_representatives_per_compound": n_extra_reps,
        "total_atoms_per_compound_upper_bound": 1 + n_extra_reps,
        "existing_real": existing_metrics,
        "pt2_real": {
            "overall": pt2_metrics,
            "per_mixture": pt2_per_source,
        },
    }
    (config_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"extra_reps={n_extra_reps:2d} "
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

    for n_extra_reps in EXTRA_REP_OPTIONS:
        summary = run_config(n_extra_reps)
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
        f"  extra_reps={best_summary['n_extra_representatives_per_compound']} "
        f"atoms_per_compound<= {best_summary['total_atoms_per_compound_upper_bound']}"
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
