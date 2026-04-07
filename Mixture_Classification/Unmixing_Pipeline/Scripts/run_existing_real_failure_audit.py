from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from unmixing_common import RESULTS_ROOT


BEST_PAIR_PREDICTIONS = (
    RESULTS_ROOT
    / "pair_nnls_replicate_dictionary"
    / "baseline_corrected_extra_reps_9"
    / "predictions.csv"
)
OPERATIONAL_PREDICTIONS = (
    RESULTS_ROOT / "binary_anchor_operational_calibration" / "existing_real_predictions.csv"
)
RESULTS_DIR = RESULTS_ROOT / "existing_real_failure_audit"


def safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def parse_sample_index(sample_id: str) -> int:
    return int(sample_id.split("/")[-1])


def summarize_true_pair(true_labels: str, group: pd.DataFrame) -> dict[str, float | str | int]:
    accepted = group["binary_accept"].astype(bool)
    accepted_group = group.loc[accepted]
    return {
        "true_labels": str(true_labels),
        "samples": int(len(group)),
        "anchor_exact_rate": safe_rate(group["correct"]),
        "reject_rate": float((~accepted).mean()),
        "accepted_exact_rate": safe_rate(accepted_group["correct"]) if len(accepted_group) else 0.0,
        "accepted_error_rate": safe_rate(~accepted_group["correct"]) if len(accepted_group) else 0.0,
        "mean_residual_rel": float(group["residual_rel"].mean()),
        "mean_minor_share": float(group["minor_share"].mean()),
        "mean_gap_ratio": float(group["gap_ratio"].mean()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(BEST_PAIR_PREDICTIONS)
    best_df = best_df.loc[best_df["dataset"] == "existing_real"].copy()

    op_df = pd.read_csv(OPERATIONAL_PREDICTIONS).copy()
    op_df["sample_index"] = op_df["sample_id"].map(parse_sample_index)
    best_df["sample_index"] = best_df["sample_id"].map(parse_sample_index)

    merged = best_df.merge(
        op_df[
            [
                "sample_id",
                "correct",
                "residual_rel",
                "minor_share",
                "gap_ratio",
                "binary_accept",
            ]
        ],
        on="sample_id",
        how="inner",
        suffixes=("_anchor", ""),
    )
    merged["accepted_error"] = merged["binary_accept"] & (~merged["correct"])
    merged["rejected"] = ~merged["binary_accept"]

    failures = merged.loc[~merged["correct"]].copy()
    accepted_failures = merged.loc[merged["accepted_error"]].copy()
    rejections = merged.loc[merged["rejected"]].copy()

    per_true_pair = pd.DataFrame(
        [
            summarize_true_pair(true_labels, group)
            for true_labels, group in merged.groupby("true_labels", sort=True)
        ]
    ).sort_values(
        ["anchor_exact_rate", "reject_rate", "mean_residual_rel", "true_labels"],
        ascending=[True, False, False, True],
    )

    confusion = (
        failures.groupby(["true_labels", "predicted_labels"])
        .size()
        .reset_index(name="samples")
        .sort_values(["samples", "true_labels", "predicted_labels"], ascending=[False, True, True])
    )

    rejection_reasons = pd.DataFrame(
        {
            "sample_id": rejections["sample_id"],
            "true_labels": rejections["true_labels"],
            "predicted_labels": rejections["predicted_labels"],
            "correct_anchor_prediction": rejections["correct"],
            "residual_rel": rejections["residual_rel"],
            "minor_share": rejections["minor_share"],
            "gap_ratio": rejections["gap_ratio"],
        }
    ).sort_values(["correct_anchor_prediction", "residual_rel"], ascending=[True, False])

    summary = {
        "dataset": "existing_real",
        "samples": int(len(merged)),
        "anchor_exact_match": safe_rate(merged["correct"]),
        "anchor_error_count": int((~merged["correct"]).sum()),
        "accepted_count": int(merged["binary_accept"].sum()),
        "rejected_count": int((~merged["binary_accept"]).sum()),
        "accepted_exact_match": safe_rate(merged.loc[merged["binary_accept"], "correct"]),
        "accepted_error_count": int(accepted_failures.shape[0]),
        "rejected_correct_count": int(rejections["correct"].sum()),
        "rejected_incorrect_count": int((~rejections["correct"]).sum()),
        "top_failure_modes": confusion.head(10).to_dict(orient="records"),
        "worst_true_pairs": per_true_pair.head(10).to_dict(orient="records"),
    }

    lines = [
        "# Existing Real Failure Audit",
        "",
        f"- samples: `{summary['samples']}`",
        f"- anchor exact match: `{summary['anchor_exact_match']:.4f}`",
        f"- anchor errors: `{summary['anchor_error_count']}`",
        f"- operational accepted count: `{summary['accepted_count']}`",
        f"- operational rejected count: `{summary['rejected_count']}`",
        f"- operational accepted exact match: `{summary['accepted_exact_match']:.4f}`",
        f"- accepted errors: `{summary['accepted_error_count']}`",
        f"- rejected but anchor-correct: `{summary['rejected_correct_count']}`",
        f"- rejected and anchor-incorrect: `{summary['rejected_incorrect_count']}`",
        "",
        "## Top Failure Modes",
        "",
    ]
    for row in summary["top_failure_modes"]:
        lines.append(
            f"- `{row['true_labels']}` -> `{row['predicted_labels']}`: `{row['samples']}` samples"
        )
    lines.extend(["", "## Worst True Pairs", ""])
    for row in summary["worst_true_pairs"]:
        lines.append(
            "- "
            f"`{row['true_labels']}`: anchor_exact=`{row['anchor_exact_rate']:.3f}`, "
            f"reject_rate=`{row['reject_rate']:.3f}`, "
            f"accepted_exact=`{row['accepted_exact_rate']:.3f}`, "
            f"mean_residual_rel=`{row['mean_residual_rel']:.3f}`, "
            f"mean_minor_share=`{row['mean_minor_share']:.3f}`"
        )

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    per_true_pair.to_csv(RESULTS_DIR / "per_true_pair_summary.csv", index=False)
    confusion.to_csv(RESULTS_DIR / "failure_confusions.csv", index=False)
    rejection_reasons.to_csv(RESULTS_DIR / "rejections.csv", index=False)
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")

    print(f"Saved audit in {RESULTS_DIR}")
    print(
        f"anchor_exact={summary['anchor_exact_match']:.4f} "
        f"accepted_exact={summary['accepted_exact_match']:.4f} "
        f"accepted_errors={summary['accepted_error_count']} "
        f"rejected_correct={summary['rejected_correct_count']} "
        f"rejected_incorrect={summary['rejected_incorrect_count']}"
    )


if __name__ == "__main__":
    main()
