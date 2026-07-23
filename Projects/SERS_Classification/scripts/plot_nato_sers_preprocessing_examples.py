#!/usr/bin/env python3
"""Plot deterministic example spectra for every NATO SERS Raman instrument.

One quality-pass medoid is selected per instrument using correlation distance
in the frozen minimal-minmax representation. The examples are therefore
deterministic and are not hand-picked for visual appearance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INSTRUMENT_ORDER = (
    "Agilent-1",
    "Agilent-3",
    "Mira-1",
    "Mira-2",
    "Mira-3",
    "Pendar-1",
    "Pendar-2",
    "Pendar-3",
    "RMX-1",
    "RMX-2",
)

COLORS = {
    "minimal_minmax": "#0072B2",
    "arpls_minmax": "#D55E00",
    "derivative_1": "#009E73",
    "baseline": "#E69F00",
    "spike": "#CC79A7",
}


def configure_style() -> None:
    """Apply a compact, colorblind-safe publication-style configuration."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def correlation_normalize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1.0e-12)


def select_instrument_medoids(
    manifest: pd.DataFrame, minimal: np.ndarray
) -> pd.DataFrame:
    """Select the observed row minimizing median within-instrument distance."""
    records: list[dict[str, object]] = []
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    for instrument in INSTRUMENT_ORDER:
        candidates = np.flatnonzero(
            (manifest["instrument"].astype(str).to_numpy() == instrument) & quality
        )
        used_quality_fallback = False
        if not len(candidates):
            candidates = np.flatnonzero(
                manifest["instrument"].astype(str).to_numpy() == instrument
            )
            used_quality_fallback = True
        if not len(candidates):
            raise ValueError(f"No frozen spectrum found for {instrument}")

        normalized = correlation_normalize(minimal[candidates])
        distances = 1.0 - normalized @ normalized.T
        median_distance = np.median(distances, axis=1)
        chosen_local = int(np.argmin(median_distance))
        chosen = int(candidates[chosen_local])
        row = manifest.iloc[chosen]
        records.append(
            {
                "archive_row": chosen,
                "observation_uid": row["observation_uid"],
                "instrument": instrument,
                "target_analyte": row["target_analyte"],
                "sensor_family": row["sensor_family"],
                "master_sample_id": row["master_sample_id"],
                "quality_pass": bool(row["include_sers_qc_pass"]),
                "quality_fallback_required": used_quality_fallback,
                "candidate_spike_count": int(row["candidate_spike_count"]),
                "median_correlation_distance_to_instrument": float(
                    median_distance[chosen_local]
                ),
                "instrument_quality_candidate_count": int(len(candidates)),
            }
        )
    return pd.DataFrame(records)


def prettify(value: object) -> str:
    return str(value).replace("_", " ")


def add_common_caption(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.005,
        "One deterministic quality-pass medoid per instrument. Targets and sensors "
        "differ because the field design is not fully crossed; compare transforms "
        "within an instrument, not cross-row chemistry.",
        ha="center",
        va="bottom",
        fontsize=8,
    )


def plot_overlay(
    output_stem: Path,
    axis_cm1: np.ndarray,
    manifest: pd.DataFrame,
    archive: np.lib.npyio.NpzFile,
    selection: pd.DataFrame,
) -> None:
    """Plot minimal and arPLS representations together for each instrument."""
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(11.0, 12.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for plot_index, instrument in enumerate(INSTRUMENT_ORDER):
        ax = axes.flat[plot_index]
        chosen = selection.loc[selection["instrument"] == instrument].iloc[0]
        row_index = int(chosen["archive_row"])
        minimal = archive["minimal_minmax"][row_index]
        arpls = archive["arpls_minmax"][row_index]
        ax.plot(
            axis_cm1,
            minimal,
            color=COLORS["minimal_minmax"],
            linewidth=1.0,
            label="Minimal min–max",
        )
        ax.plot(
            axis_cm1,
            arpls,
            color=COLORS["arpls_minmax"],
            linewidth=1.0,
            linestyle="--",
            label="arPLS + min–max",
        )
        spike_positions = archive["spike_mask"][row_index]
        if spike_positions.any():
            ax.scatter(
                axis_cm1[spike_positions],
                minimal[spike_positions],
                color=COLORS["spike"],
                marker="x",
                s=24,
                linewidth=1.0,
                zorder=5,
                label="Candidate spike",
            )
        correlation = float(np.corrcoef(minimal, arpls)[0, 1])
        ax.set_title(
            f"{instrument} · {prettify(chosen['target_analyte'])} · "
            f"{prettify(chosen['sensor_family'])}\n"
            f"minimal/arPLS correlation = {correlation:.3f}",
            loc="left",
            fontweight="semibold",
        )
        ax.set_xlim(float(axis_cm1.min()), float(axis_cm1.max()))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([400, 800, 1200, 1600, 1800])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.55)
        if plot_index % 2 == 0:
            ax.set_ylabel("Scaled intensity")
        if plot_index >= 8:
            ax.set_xlabel("Raman shift (cm$^{-1}$)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(unique),
        frameon=False,
    )
    fig.suptitle(
        "Effect of frozen domain-blind baseline preprocessing by instrument",
        y=0.998,
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.945, bottom=0.065, hspace=0.42, wspace=0.13
    )
    add_common_caption(fig)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_representation_matrix(
    output_stem: Path,
    axis_cm1: np.ndarray,
    archive: np.lib.npyio.NpzFile,
    selection: pd.DataFrame,
) -> None:
    """Show all three frozen representations for the ten instrument examples."""
    chosen_rows = selection.set_index("instrument").loc[list(INSTRUMENT_ORDER)]
    derivative_limit = 1.05 * float(
        np.max(np.abs(archive["derivative_1"][chosen_rows["archive_row"].astype(int)]))
    )
    fig, axes = plt.subplots(
        len(INSTRUMENT_ORDER),
        3,
        figsize=(12.0, 17.0),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )
    column_specs = (
        ("minimal_minmax", "Minimal min–max", COLORS["minimal_minmax"], "-"),
        ("arpls_minmax", "arPLS + min–max", COLORS["arpls_minmax"], "--"),
        ("derivative_1", "SNV + first derivative + L2", COLORS["derivative_1"], "-"),
    )
    for row_number, instrument in enumerate(INSTRUMENT_ORDER):
        chosen = chosen_rows.loc[instrument]
        archive_row = int(chosen["archive_row"])
        for column_number, (key, title, color, linestyle) in enumerate(column_specs):
            ax = axes[row_number, column_number]
            values = archive[key][archive_row]
            ax.plot(
                axis_cm1,
                values,
                color=color,
                linewidth=0.95,
                linestyle=linestyle,
            )
            ax.set_xlim(float(axis_cm1.min()), float(axis_cm1.max()))
            ax.set_xticks([400, 800, 1200, 1600, 1800])
            ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.5)
            if key == "derivative_1":
                ax.axhline(0, color="#666666", linewidth=0.5, alpha=0.8)
                ax.set_ylim(-derivative_limit, derivative_limit)
                ax.set_yticks([-derivative_limit, 0.0, derivative_limit])
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
            else:
                ax.set_ylim(-0.03, 1.03)
                ax.set_yticks([0.0, 0.5, 1.0])
            if row_number == 0:
                ax.set_title(title, fontsize=10, fontweight="bold", pad=9)
            if row_number == len(INSTRUMENT_ORDER) - 1:
                ax.set_xlabel("Raman shift (cm$^{-1}$)")

        row_label = (
            f"{instrument}\n"
            f"{prettify(chosen['target_analyte'])}\n"
            f"{prettify(chosen['sensor_family'])}"
        )
        axes[row_number, 0].text(
            -0.31,
            0.5,
            row_label,
            transform=axes[row_number, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="semibold",
        )

    fig.suptitle(
        "Frozen NATO SERS representations across all Raman instruments",
        y=0.997,
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.17, right=0.99, top=0.965, bottom=0.047, hspace=0.26, wspace=0.17
    )
    add_common_caption(fig)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "figures"
        / "preprocessing_examples",
    )
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    manifest = pd.read_csv(bundle / "core_preprocessing_manifest.csv")
    archive = np.load(bundle / "candidate_spectra_core.npz")
    if not np.array_equal(
        manifest["observation_uid"].astype(str).to_numpy(),
        archive["observation_uid"].astype(str),
    ):
        raise ValueError("Manifest and candidate archive order do not match")

    selection = select_instrument_medoids(manifest, archive["minimal_minmax"])
    selection.to_csv(output_dir / "preprocessing_example_selection.csv", index=False)
    axis_cm1 = archive["axis_cm1"].astype(float)
    plot_overlay(
        output_dir / "preprocessing_effect_overlay_by_instrument",
        axis_cm1,
        manifest,
        archive,
        selection,
    )
    plot_representation_matrix(
        output_dir / "frozen_representations_by_instrument",
        axis_cm1,
        archive,
        selection,
    )
    print(
        selection[
            [
                "instrument",
                "observation_uid",
                "target_analyte",
                "sensor_family",
                "candidate_spike_count",
                "median_correlation_distance_to_instrument",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
