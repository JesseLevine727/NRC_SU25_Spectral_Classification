#!/usr/bin/env python3
"""Generate average spectra panels for measured SERS chemical-substrate pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sers_siamese_substrate_agnostic as sers  # noqa: E402

DATA = ROOT / "Workspace" / "data" / "processed" / "consolidated_SERS.csv"
FIG_DIR = ROOT / "docs" / "cscce_2026_poster" / "figures"
OUT_PNG = FIG_DIR / "average_spectra_by_pair.png"
OUT_COMPACT_PNG = FIG_DIR / "average_spectra_by_pair_compact.png"
OUT_CSV = FIG_DIR / "average_spectra_by_pair.csv"
OUT_PREPROCESSED_PNG = FIG_DIR / "average_spectra_by_pair_preprocessed.png"
OUT_PREPROCESSED_COMPACT_PNG = FIG_DIR / "average_spectra_by_pair_preprocessed_compact.png"
OUT_PREPROCESSED_CSV = FIG_DIR / "average_spectra_by_pair_preprocessed.csv"
OUT_MODEL_PREPROCESSED_PNG = FIG_DIR / "average_spectra_by_pair_preprocessed_model.png"
OUT_MODEL_PREPROCESSED_COMPACT_PNG = FIG_DIR / "average_spectra_by_pair_preprocessed_model_compact.png"
OUT_MODEL_PREPROCESSED_CSV = FIG_DIR / "average_spectra_by_pair_preprocessed_model.csv"

CHEMICAL_ORDER = ["4np", "benzenethiol", "pyridine", "n,n-dimethylformamide"]
MODEL_CHEMICAL_ORDER = ["4np", "benzenethiol", "pyridine"]
CHEMICAL_DISPLAY = {
    "4np": "4NP",
    "benzenethiol": "Benzenethiol",
    "pyridine": "Pyridine",
    "n,n-dimethylformamide": "DMF",
}
SUBSTRATE_ORDER = ["Ag", "Au", "PICO", "pSERS"]
COLORS = {
    "4np": "#4C78A8",
    "benzenethiol": "#F58518",
    "pyridine": "#54A24B",
    "n,n-dimethylformamide": "#7A5195",
}


def minmax(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    spread = float(np.nanmax(y) - np.nanmin(y))
    if spread == 0:
        return np.zeros_like(y)
    return (y - np.nanmin(y)) / spread


def maxabs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    scale = float(np.nanmax(np.abs(y)))
    if scale == 0:
        return np.zeros_like(y)
    return y / scale


def write_average_csv(
    df: pd.DataFrame,
    cols: list[str],
    out_csv: Path,
    mean_column: str,
    display_column: str,
    scale_mode: str,
    chemical_order: list[str] | None = None,
) -> None:
    x = np.array([float(col) for col in cols])
    rows: list[dict[str, str | int | float]] = []
    for chemical in chemical_order or CHEMICAL_ORDER:
        for substrate in SUBSTRATE_ORDER:
            mask = (df["Label"] == chemical) & (df["Substrate"] == substrate)
            count = int(mask.sum())
            if not count:
                continue
            mean = df.loc[mask, cols].to_numpy(dtype=float).mean(axis=0)
            scaled = minmax(mean) if scale_mode == "minmax" else maxabs(mean)
            for wavenumber, intensity, display_intensity in zip(x, mean, scaled):
                rows.append(
                    {
                        "label": chemical,
                        "substrate": substrate,
                        "n": count,
                        "raman_shift_cm-1": float(wavenumber),
                        mean_column: float(intensity),
                        display_column: float(display_intensity),
                    }
                )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def plot_grid(
    df: pd.DataFrame,
    cols: list[str],
    out_png: Path,
    compact: bool,
    preprocessed: bool = False,
    chemical_order: list[str] | None = None,
) -> None:
    x = np.array([float(col) for col in cols])
    chemicals = chemical_order or CHEMICAL_ORDER
    figsize = (12.0, 3.35 if len(chemicals) == 3 else 4.2) if compact else (10.8, 6.0 if len(chemicals) == 3 else 7.2)
    fig, axes = plt.subplots(
        len(chemicals),
        len(SUBSTRATE_ORDER),
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    for row_idx, chemical in enumerate(chemicals):
        for col_idx, substrate in enumerate(SUBSTRATE_ORDER):
            ax = axes[row_idx, col_idx]
            mask = (df["Label"] == chemical) & (df["Substrate"] == substrate)
            count = int(mask.sum())

            ax.set_facecolor("#FCFAF5")
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)
                spine.set_color("#333333")

            if count:
                mean = df.loc[mask, cols].to_numpy(dtype=float).mean(axis=0)
                scaled = maxabs(mean) if preprocessed else minmax(mean)
                ax.plot(x, scaled, color=COLORS[chemical], linewidth=1.25 if compact else 1.6)
                if preprocessed:
                    ax.axhline(0, color="#777777", linewidth=0.4, alpha=0.65)
                ax.text(
                    0.04,
                    0.78 if compact else 0.86,
                    f"n={count}",
                    transform=ax.transAxes,
                    fontsize=6.6 if compact else 8,
                    weight="bold",
                    color="#222222",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "not\nmeasured",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.8 if compact else 8,
                    color="#555555",
                )
                ax.set_facecolor("#F2EFE8")

            if row_idx == 0:
                ax.set_title(substrate, fontsize=8.8 if compact else 11, weight="bold", pad=1 if compact else 4)
            if col_idx == 0:
                ax.set_ylabel(CHEMICAL_DISPLAY[chemical], fontsize=8.0 if compact else 10, weight="bold")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(-1.08, 1.08) if preprocessed else ax.set_ylim(-0.05, 1.05)
            if preprocessed:
                ax.set_yticks([-1, 0, 1])
                ax.tick_params(
                    axis="y",
                    labelsize=5.8 if compact else 7,
                    length=2,
                    labelleft=col_idx == 0,
                )
            else:
                ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=6 if compact else 7, length=2)

    for ax in axes[-1, :]:
        ax.set_xlabel("cm$^{-1}$" if compact else "Raman shift (cm$^{-1}$)", fontsize=7 if compact else 9)

    if compact:
        if preprocessed:
            fig.text(
                0.010,
                0.50,
                "Scaled first-derivative model input",
                rotation=90,
                va="center",
                ha="center",
                fontsize=7.2,
            )
            fig.tight_layout(rect=[0.030, 0.03, 1, 0.98], h_pad=0.12, w_pad=0.22)
        else:
            fig.tight_layout(rect=[0.015, 0.03, 1, 0.98], h_pad=0.12, w_pad=0.22)
    else:
        title = (
            "Average model-input spectra by chemical-substrate family"
            if preprocessed
            else "Average SERS spectra by chemical-substrate family"
        )
        footer = (
            "Each measured cell is the mean cropped, SNV-normalized, first-derivative, row-L2-normalized model input; panels are max-absolute scaled independently."
            if preprocessed
            else "Each measured cell is the mean raw spectrum after cropping to 330-1800 cm$^{-1}$; panels are min-max scaled independently."
        )
        fig.suptitle(
            title,
            fontsize=15,
            weight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.012,
            footer,
            ha="center",
            fontsize=9,
        )
        if preprocessed:
            fig.text(
                0.012,
                0.50,
                "Scaled first-derivative model input",
                rotation=90,
                va="center",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=[0.04, 0.04, 1, 0.96], h_pad=0.7, w_pad=0.45)
        else:
            fig.tight_layout(rect=[0.02, 0.04, 1, 0.96], h_pad=0.7, w_pad=0.45)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def preprocessed_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X_raw = df[cols].to_numpy(dtype=np.float64)
    X = sers.prepare_features(X_raw, type("Args", (), {"feature": "derivative_1"})())
    return pd.concat(
        [
            df[["Label", "Substrate"]].reset_index(drop=True),
            pd.DataFrame(X, columns=cols),
        ],
        axis=1,
    )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, cols = sers.load_dataset(
        DATA,
        crop_min=330.0,
        crop_max=1800.0,
        min_substrates=1,
        canonicalize_labels=True,
        group_metal_substrates=True,
    )
    processed = preprocessed_frame(df, cols)
    write_average_csv(df, cols, OUT_CSV, "mean_raw_intensity", "display_minmax_intensity", "minmax")
    write_average_csv(
        processed,
        cols,
        OUT_PREPROCESSED_CSV,
        "mean_model_input",
        "display_maxabs_intensity",
        "maxabs",
    )
    write_average_csv(
        processed,
        cols,
        OUT_MODEL_PREPROCESSED_CSV,
        "mean_model_input",
        "display_maxabs_intensity",
        "maxabs",
        chemical_order=MODEL_CHEMICAL_ORDER,
    )
    plot_grid(df, cols, OUT_PNG, compact=False)
    plot_grid(df, cols, OUT_COMPACT_PNG, compact=True)
    plot_grid(processed, cols, OUT_PREPROCESSED_PNG, compact=False, preprocessed=True)
    plot_grid(processed, cols, OUT_PREPROCESSED_COMPACT_PNG, compact=True, preprocessed=True)
    plot_grid(
        processed,
        cols,
        OUT_MODEL_PREPROCESSED_PNG,
        compact=False,
        preprocessed=True,
        chemical_order=MODEL_CHEMICAL_ORDER,
    )
    plot_grid(
        processed,
        cols,
        OUT_MODEL_PREPROCESSED_COMPACT_PNG,
        compact=True,
        preprocessed=True,
        chemical_order=MODEL_CHEMICAL_ORDER,
    )
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_COMPACT_PNG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PREPROCESSED_PNG}")
    print(f"Wrote {OUT_PREPROCESSED_COMPACT_PNG}")
    print(f"Wrote {OUT_PREPROCESSED_CSV}")
    print(f"Wrote {OUT_MODEL_PREPROCESSED_PNG}")
    print(f"Wrote {OUT_MODEL_PREPROCESSED_COMPACT_PNG}")
    print(f"Wrote {OUT_MODEL_PREPROCESSED_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
