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
    load_reference,
    preprocess_spectrum,
)


RESULTS_DIR = RESULTS_ROOT / "targeted_pair_diagnostics"
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
TOP_K = 5
TARGET_TRUE_PAIRS = {
    "6-mercapto-1-hexanol + pyridine",
    "1-dodecanethiol + meoh",
}
TARGET_COMPOUNDS = {
    "6-mercapto-1-hexanol",
    "benzenethiol",
    "pyridine",
    "1-dodecanethiol",
    "1-undecanethiol",
    "meoh",
    "diethylamine",
    "tris(2-ethylhexyl) phosphate",
}


def labels_to_key(labels: tuple[str, ...]) -> str:
    return " + ".join(sorted(labels))


def pair_fit_details(
    spectrum: np.ndarray,
    support: tuple[str, str],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
) -> dict[str, float | str]:
    left_label, right_label = tuple(sorted(support))
    left_atoms = atom_sets[left_label]
    right_atoms = atom_sets[right_label]
    design = np.column_stack([left_atoms, right_atoms, baseline_atom])
    coef, _ = nnls(design, spectrum)
    recon = design @ coef
    residual_norm = float(np.linalg.norm(spectrum - recon))
    residual_rel = float(residual_norm / (np.linalg.norm(spectrum) + 1e-12))
    left_sum = float(coef[: left_atoms.shape[1]].sum())
    right_sum = float(coef[left_atoms.shape[1] : left_atoms.shape[1] + right_atoms.shape[1]].sum())
    pair_total = left_sum + right_sum + 1e-12
    return {
        "pair": labels_to_key((left_label, right_label)),
        "residual_norm": residual_norm,
        "residual_rel": residual_rel,
        "left_sum": left_sum,
        "right_sum": right_sum,
        "minor_share": float(min(left_sum, right_sum) / pair_total),
        "baseline_coef_sum": float(coef[-1]),
    }


def summarize_problem_pair(
    records,
    pair_defs,
    atom_sets,
    baseline_atom,
):
    sample_rows = []
    aggregate_rows = []
    for record in records:
        fits = [pair_fit_details(record.spectrum, support, atom_sets, baseline_atom) for support in pair_defs]
        fits_df = pd.DataFrame(fits).sort_values(["residual_rel", "pair"], ascending=[True, True]).reset_index(drop=True)
        fits_df["rank"] = np.arange(1, len(fits_df) + 1)
        true_pair = labels_to_key(record.true_labels)
        true_row = fits_df.loc[fits_df["pair"] == true_pair].iloc[0]
        best_row = fits_df.iloc[0]
        second_row = fits_df.iloc[1]
        sample_rows.append(
            {
                "sample_id": record.sample_id,
                "true_pair": true_pair,
                "best_pair": best_row["pair"],
                "best_residual_rel": float(best_row["residual_rel"]),
                "second_pair": second_row["pair"],
                "second_residual_rel": float(second_row["residual_rel"]),
                "true_pair_rank": int(true_row["rank"]),
                "true_pair_residual_rel": float(true_row["residual_rel"]),
                "best_vs_true_delta": float(true_row["residual_rel"] - best_row["residual_rel"]),
                "true_minor_share": float(true_row["minor_share"]),
                "best_minor_share": float(best_row["minor_share"]),
            }
        )

        top_k = fits_df.head(TOP_K).copy()
        top_k.insert(0, "sample_id", record.sample_id)
        top_k.insert(1, "true_pair", true_pair)
        aggregate_rows.append(top_k)

    return pd.DataFrame(sample_rows), pd.concat(aggregate_rows, ignore_index=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def build_compound_similarity_table(ref_df: pd.DataFrame) -> pd.DataFrame:
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    rows = []
    proc_by_label = {}
    mean_by_label = {}
    for label in sorted(ref_df["Label"].unique()):
        spectra = ref_df.loc[ref_df["Label"] == label, wav_cols].to_numpy(dtype=np.float64)
        proc = np.vstack([preprocess_spectrum(spec, MODE) for spec in spectra])
        proc_by_label[label] = proc
        mean_by_label[label] = proc.mean(axis=0)

    for left, right in combinations(sorted(TARGET_COMPOUNDS), 2):
        left_mean = mean_by_label[left]
        right_mean = mean_by_label[right]
        all_pair_dists = np.linalg.norm(
            proc_by_label[left][:, None, :] - proc_by_label[right][None, :, :], axis=2
        )
        rows.append(
            {
                "compound_a": left,
                "compound_b": right,
                "mean_cosine_similarity": cosine_similarity(left_mean, right_mean),
                "mean_euclidean_distance": float(np.linalg.norm(left_mean - right_mean)),
                "nearest_atom_distance": float(all_pair_dists.min()),
                "mean_atom_distance": float(all_pair_dists.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["nearest_atom_distance", "mean_cosine_similarity"], ascending=[True, False]
    )


def make_summary(sample_df: pd.DataFrame, topk_df: pd.DataFrame) -> dict:
    top1_counts = (
        sample_df.groupby(["true_pair", "best_pair"]).size().reset_index(name="samples")
        .sort_values(["true_pair", "samples", "best_pair"], ascending=[True, False, True])
    )
    summary = {
        "pairs_audited": sorted(sample_df["true_pair"].unique().tolist()),
        "sample_count": int(len(sample_df)),
        "per_true_pair": [],
        "top1_confusions": top1_counts.to_dict(orient="records"),
    }
    for true_pair, group in sample_df.groupby("true_pair", sort=True):
        summary["per_true_pair"].append(
            {
                "true_pair": true_pair,
                "samples": int(len(group)),
                "top1_exact_rate": float((group["best_pair"] == true_pair).mean()),
                "mean_true_pair_rank": float(group["true_pair_rank"].mean()),
                "mean_best_vs_true_delta": float(group["best_vs_true_delta"].mean()),
                "max_true_pair_rank": int(group["true_pair_rank"].max()),
                "mean_best_residual_rel": float(group["best_residual_rel"].mean()),
                "mean_true_pair_residual_rel": float(group["true_pair_residual_rel"].mean()),
            }
        )
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])
    classes = sorted(expanded_ref["Label"].unique())
    pair_defs = list(combinations(classes, 2))

    existing_records = load_existing_real_records(MODE)
    target_records = [r for r in existing_records if labels_to_key(r.true_labels) in TARGET_TRUE_PAIRS]

    sample_df, topk_df = summarize_problem_pair(target_records, pair_defs, atom_sets, baseline_atom)
    summary = make_summary(sample_df, topk_df)
    similarity_df = build_compound_similarity_table(expanded_ref)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    sample_df.to_csv(RESULTS_DIR / "sample_summary.csv", index=False)
    topk_df.to_csv(RESULTS_DIR / "topk_pair_fits.csv", index=False)
    similarity_df.to_csv(RESULTS_DIR / "compound_similarity.csv", index=False)

    print(f"Saved diagnostics in {RESULTS_DIR}")
    for row in summary["per_true_pair"]:
        print(
            f"{row['true_pair']}: top1_exact={row['top1_exact_rate']:.3f} "
            f"mean_true_rank={row['mean_true_pair_rank']:.2f} "
            f"mean_delta={row['mean_best_vs_true_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
