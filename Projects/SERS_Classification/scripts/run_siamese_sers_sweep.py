#!/usr/bin/env python3
"""Small CUDA sweep for substrate-agnostic SERS Siamese settings."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Workspace/substrate_agnostic/sweeps/siamese_feature_loss_sweep"),
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--group-metal-substrates",
        action="store_true",
        help="Pass corrected Ag/AgNP and Au/AuNP grouping to each Siamese run.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = list(
        itertools.product(
            ["derivative_1", "peak_emphasis", "derivative_2"],
            ["contrastive", "triplet", "batch_hard_triplet"],
            ["substrate_balanced", "row_mean"],
            [int(seed) for seed in args.seeds.split(",")],
        )
    )

    rows = []
    for feature, loss, prototype_mode, seed in configs:
        stem = f"{feature}__{loss}__{prototype_mode}__seed{seed}"
        result_path = args.out_dir / f"{stem}.csv"
        conf_dir = args.out_dir / f"{stem}_confusions"
        cmd = [
            args.python,
            str(SCRIPT_DIR / "sers_siamese_substrate_agnostic.py"),
            "--feature",
            feature,
            "--loss",
            loss,
            "--prototype-mode",
            prototype_mode,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(seed),
            "--out",
            str(result_path),
            "--confusions-dir",
            str(conf_dir),
        ]
        if args.group_metal_substrates:
            cmd.append("--group-metal-substrates")
        run(cmd)
        df = pd.read_csv(result_path)
        rows.append(
            {
                "feature": feature,
                "loss": loss,
                "prototype_mode": prototype_mode,
                "seed": seed,
                "accuracy": df["accuracy"].mean(),
                "balanced_accuracy": df["balanced_accuracy"].mean(),
                "macro_f1": df["macro_f1"].mean(),
                "min_fold_accuracy": df["accuracy"].min(),
                "worst_fold": df.loc[df["accuracy"].idxmin(), "held_out_substrate"],
                "result_path": str(result_path),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "min_fold_accuracy"],
        ascending=False,
    )
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSweep summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
