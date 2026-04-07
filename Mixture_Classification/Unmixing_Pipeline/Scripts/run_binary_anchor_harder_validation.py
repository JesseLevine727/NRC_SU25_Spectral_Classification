from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from unmixing_common import (
    PT2_DIR,
    PURE_LABEL_ALIASES,
    build_compound_atom_sets,
    constant_baseline_atom,
    evaluate_pair_records_with_atom_sets,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
    load_txt_spectrum,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "binary_anchor_harder_validation"
)
MODE = "baseline_corrected"
TRAIN_CAP_OPTIONS = [1, 2, 4, 8, 16, None]
SEEDS = [0, 1, 2, 3, 4]
N_EXTRA_REPS = 9


def sample_reference_subset(ref_df: pd.DataFrame, cap_per_class: int | None, seed: int) -> pd.DataFrame:
    if cap_per_class is None:
        return ref_df.copy()

    rng = np.random.default_rng(seed)
    parts = []
    for label in sorted(ref_df["Label"].unique()):
        group = ref_df.loc[ref_df["Label"] == label].copy()
        if len(group) <= cap_per_class:
            parts.append(group)
            continue
        idx = rng.choice(len(group), size=cap_per_class, replace=False)
        parts.append(group.iloc[np.sort(idx)].copy())
    return pd.concat(parts, ignore_index=True)


def load_pt2_pure_df(wav_axis: np.ndarray) -> pd.DataFrame:
    rows = []
    for pure_dir_name, label in PURE_LABEL_ALIASES.items():
        for txt_path in sorted((PT2_DIR / pure_dir_name / "txt").glob("*.txt")):
            spectrum = load_txt_spectrum(txt_path, wav_axis)
            row = {"Label": label}
            row.update({wav: val for wav, val in zip(wav_axis, spectrum)})
            rows.append(row)
    return pd.DataFrame(rows)


def run_once(
    original_ref: pd.DataFrame,
    pt2_pure_df: pd.DataFrame,
    cap_original: int | None,
    cap_pt2: int | None,
    seed: int,
):
    orig_subset = sample_reference_subset(original_ref, cap_original, seed)
    pt2_subset = sample_reference_subset(pt2_pure_df, cap_pt2, seed)
    ref_subset = pd.concat([orig_subset, pt2_subset], ignore_index=True)

    atom_sets = build_compound_atom_sets(ref_subset, MODE, N_EXTRA_REPS)
    classes = sorted(ref_subset["Label"].unique())
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])

    ref_counts = ref_subset["Label"].value_counts().sort_index().to_dict()
    wav_axis = np.array([c for c in ref_subset.columns if c != "Label"], dtype=float)
    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)

    existing_metrics, _, _ = evaluate_pair_records_with_atom_sets(
        existing_records, classes, atom_sets, baseline_atom
    )
    pt2_metrics, pt2_per_source, _ = evaluate_pair_records_with_atom_sets(
        pt2_records, classes, atom_sets, baseline_atom
    )

    return {
        "seed": seed,
        "cap_original": cap_original if cap_original is not None else "all",
        "cap_pt2": cap_pt2 if cap_pt2 is not None else "all",
        "reference_counts": ref_counts,
        "existing_real": existing_metrics,
        "pt2_real": {
            "overall": pt2_metrics,
            "per_mixture": pt2_per_source,
        },
    }


def summarize_group(rows: list[dict]) -> dict:
    def collect(path: list[str]) -> list[float]:
        vals = []
        for row in rows:
            cur = row
            for key in path:
                cur = cur[key]
            vals.append(float(cur))
        return vals

    summary = {}
    for dataset in ("existing_real", "pt2_real"):
        prefix = [dataset] if dataset == "existing_real" else [dataset, "overall"]
        exact = collect(prefix + ["exact_match"])
        f1 = collect(prefix + ["micro_f1"])
        prec = collect(prefix + ["micro_precision"])
        rec = collect(prefix + ["micro_recall"])
        summary[dataset] = {
            "exact_mean": float(np.mean(exact)),
            "exact_std": float(np.std(exact)),
            "micro_f1_mean": float(np.mean(f1)),
            "micro_f1_std": float(np.std(f1)),
            "micro_precision_mean": float(np.mean(prec)),
            "micro_precision_std": float(np.std(prec)),
            "micro_recall_mean": float(np.mean(rec)),
            "micro_recall_std": float(np.std(rec)),
        }
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_ref, wav_axis = load_reference()
    pt2_pure_df = load_pt2_pure_df(wav_axis)

    all_runs = []
    grouped = []
    for cap in TRAIN_CAP_OPTIONS:
        rows = []
        for seed in SEEDS:
            result = run_once(
                original_ref=original_ref,
                pt2_pure_df=pt2_pure_df,
                cap_original=cap,
                cap_pt2=cap,
                seed=seed,
            )
            rows.append(result)
            all_runs.append(result)

        group_summary = {
            "train_cap_per_compound": cap if cap is not None else "all",
            **summarize_group(rows),
        }
        grouped.append(group_summary)

        print(
            f"cap={group_summary['train_cap_per_compound']:>3} "
            f"existing exact={group_summary['existing_real']['exact_mean']:.3f}±{group_summary['existing_real']['exact_std']:.3f} "
            f"micro_f1={group_summary['existing_real']['micro_f1_mean']:.3f}±{group_summary['existing_real']['micro_f1_std']:.3f} "
            f"pt2 exact={group_summary['pt2_real']['exact_mean']:.3f}±{group_summary['pt2_real']['exact_std']:.3f} "
            f"pt2 micro_f1={group_summary['pt2_real']['micro_f1_mean']:.3f}±{group_summary['pt2_real']['micro_f1_std']:.3f}"
        )

    out = {"mode": MODE, "n_extra_reps": N_EXTRA_REPS, "grouped": grouped, "runs": all_runs}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(
        [
            {
                "train_cap_per_compound": row["train_cap_per_compound"],
                **{f"existing_{k}": v for k, v in row["existing_real"].items()},
                **{f"pt2_{k}": v for k, v in row["pt2_real"].items()},
            }
            for row in grouped
        ]
    ).to_csv(RESULTS_DIR / "grouped_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "train_cap_per_compound": row["cap_original"],
                "existing_exact": row["existing_real"]["exact_match"],
                "existing_micro_f1": row["existing_real"]["micro_f1"],
                "existing_micro_precision": row["existing_real"]["micro_precision"],
                "existing_micro_recall": row["existing_real"]["micro_recall"],
                "pt2_exact": row["pt2_real"]["overall"]["exact_match"],
                "pt2_micro_f1": row["pt2_real"]["overall"]["micro_f1"],
                "pt2_micro_precision": row["pt2_real"]["overall"]["micro_precision"],
                "pt2_micro_recall": row["pt2_real"]["overall"]["micro_recall"],
            }
            for row in all_runs
        ]
    ).to_csv(RESULTS_DIR / "per_run_summary.csv", index=False)

    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
