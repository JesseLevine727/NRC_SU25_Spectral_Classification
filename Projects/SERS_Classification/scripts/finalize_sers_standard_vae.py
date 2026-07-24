#!/usr/bin/env python3
"""Finalize figures, decisions, and consolidated records for standard VAE v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import run_sers_representation_baselines as selection


COLORS = {
    "PCA-logistic": "#0072B2",
    "Siamese": "#009E73",
    "AE": "#E69F00",
    "DAE": "#D55E00",
    "VAE": "#CC79A7",
}


def write_json(path: Path, value: Any) -> None:
    selection.write_json(path, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def t_interval(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan, 0
    if len(array) == 1:
        return float(array[0]), np.nan, 1
    half = float(
        t.ppf(0.975, len(array) - 1)
        * np.std(array, ddof=1)
        / np.sqrt(len(array))
    )
    return float(np.mean(array)), half, len(array)


def cohort_from_scenario(frame: pd.DataFrame) -> pd.Series:
    return frame["scenario"].str.extract(r"test_(.+)$")[0]


def baseline_display(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    result[
        (frame["model_family"] == "classical")
        & (frame["model"] == "pca_logistic")
        & (frame["representation"] == "arpls_minmax")
    ] = "PCA-logistic"
    result[frame["model_family"] == "siamese"] = "Siamese"
    result[
        (frame["model_family"] == "ae")
        & (frame["representation"] == "arpls_minmax")
    ] = "AE"
    result[
        (frame["model_family"] == "dae")
        & (frame["representation"] == "arpls_minmax")
    ] = "DAE"
    return result


def aggregate_outer(
    baseline: pd.DataFrame, vae: pd.DataFrame
) -> pd.DataFrame:
    first = baseline.copy()
    first["cohort"] = cohort_from_scenario(first)
    first["display_model"] = baseline_display(first)
    first = first[first["display_model"].notna()]
    second = vae[vae["representation"] == "arpls_minmax"].copy()
    second["cohort"] = cohort_from_scenario(second)
    second["display_model"] = "VAE"
    combined = pd.concat([first, second], ignore_index=True)
    return (
        combined.groupby(
            ["cohort", "display_model", "outer_fold"], as_index=False
        )["balanced_accuracy_supported"]
        .mean()
        .rename(columns={"balanced_accuracy_supported": "balanced_accuracy"})
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_outer(
    output_dir: Path, baseline: pd.DataFrame, vae: pd.DataFrame
) -> None:
    units = aggregate_outer(baseline, vae)
    cohorts = [
        ("strict_core", "Strict core (598)"),
        ("quality_pass", "Quality pass (500)"),
        ("field_quality_stress", "Field stress (98)"),
    ]
    order = ["PCA-logistic", "Siamese", "AE", "DAE", "VAE"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True)
    for ax, (cohort, title) in zip(axes, cohorts):
        part = units[units["cohort"] == cohort]
        for index, model in enumerate(order):
            values = part.loc[
                part["display_model"] == model, "balanced_accuracy"
            ].to_numpy()
            if not len(values):
                continue
            mean, half, _ = t_interval(values)
            ax.scatter(
                np.full(len(values), index),
                values,
                s=16,
                alpha=0.55,
                color=COLORS[model],
            )
            ax.errorbar(
                index,
                mean,
                yerr=half,
                fmt="D",
                color=COLORS[model],
                capsize=2,
                markersize=5,
            )
        ax.axhline(1 / 7, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Balanced accuracy")
    fig.suptitle("Frozen-baseline comparison with standard VAE")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "outer_comparison")


def figure_corruption(
    output_dir: Path,
    baseline_corruption: pd.DataFrame,
    vae_corruption: pd.DataFrame,
) -> None:
    b = baseline_corruption[
        (baseline_corruption["scenario"].str.endswith("test_strict_core"))
        & (baseline_corruption["representation"] == "arpls_minmax")
        & (baseline_corruption["corruption"] == "composite")
        & (baseline_corruption["severity"] == 1.0)
        & baseline_corruption["model_family"].isin(["ae", "dae"])
    ].copy()
    b["display_model"] = b["model_family"].map({"ae": "AE", "dae": "DAE"})
    v = vae_corruption[
        (vae_corruption["scenario"].str.endswith("test_strict_core"))
        & (vae_corruption["representation"] == "arpls_minmax")
        & (vae_corruption["corruption"] == "composite")
        & (vae_corruption["severity"] == 1.0)
    ].copy()
    v["display_model"] = "VAE"
    frame = pd.concat([b, v], ignore_index=True)
    summary = frame.groupby("display_model")[
        [
            "balanced_accuracy_supported",
            "prediction_agreement",
            "latent_cosine_drift",
            "reconstruction_mse",
        ]
    ].mean()
    metrics = [
        ("balanced_accuracy_supported", "Chemical BA", True),
        ("prediction_agreement", "Prediction agreement", True),
        ("latent_cosine_drift", "Latent drift", False),
        ("reconstruction_mse", "Recovery MSE", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    order = ["AE", "DAE", "VAE"]
    for ax, (metric, title, _) in zip(axes, metrics):
        values = [summary.loc[name, metric] for name in order]
        ax.bar(
            order,
            values,
            color=[COLORS[name] for name in order],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Strict-core composite corruption, severity 1")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "corruption_comparison")


def figure_poster(
    output_dir: Path, baseline: pd.DataFrame, vae: pd.DataFrame
) -> None:
    b = baseline.copy()
    b["display_model"] = baseline_display(b)
    b = b[b["display_model"].notna()]
    v = vae[vae["representation"] == "arpls_minmax"].copy()
    v["display_model"] = "VAE"
    frame = pd.concat([b, v], ignore_index=True)
    means = (
        frame.groupby("display_model")["balanced_accuracy_supported"]
        .mean()
        .reindex(["PCA-logistic", "Siamese", "AE", "DAE", "VAE"])
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(
        means.index,
        means.values,
        color=[COLORS[name] for name in means.index],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Descriptive leave-substrate-family-out transfer")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "poster_comparison")


def consolidate(output_dir: Path) -> None:
    table_groups = {
        "per_spectrum_predictions.csv": [
            "outer_fold_predictions.csv",
            "poster_predictions.csv",
            "domain_transfer_predictions.csv",
        ],
        "reconstruction_metrics.csv": [
            "outer_fold_reconstruction_metrics.csv",
            "poster_reconstruction_metrics.csv",
            "domain_transfer_reconstruction_metrics.csv",
        ],
        "corruption_metrics.csv": [
            "outer_fold_corruption_metrics.csv",
            "poster_corruption_metrics.csv",
            "domain_transfer_corruption_metrics.csv",
        ],
        "training_histories.csv": [
            "selection_training_histories.csv",
            "outer_fold_training_histories.csv",
            "poster_training_histories.csv",
            "domain_transfer_training_histories.csv",
        ],
    }
    for target, sources in table_groups.items():
        frames = [
            pd.read_csv(output_dir / source, low_memory=False)
            for source in sources
        ]
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            output_dir / target, index=False
        )
    metrics = pd.concat(
        [
            pd.read_csv(output_dir / "outer_fold_metrics.csv"),
            pd.read_csv(output_dir / "poster_metrics.csv"),
            pd.read_csv(output_dir / "domain_transfer_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    variational_columns = [
        column for column in metrics if column.startswith("vae_")
    ]
    metrics[
        [
            "stage",
            "scenario",
            "model",
            "representation",
            "seed",
        ]
        + variational_columns
    ].to_csv(output_dir / "variational_metrics.csv", index=False)
    predictions = pd.read_csv(
        output_dir / "per_spectrum_predictions.csv", low_memory=False
    )
    failures = predictions[
        (~predictions["correct"].astype(bool))
        | (~predictions["test_class_supported"].astype(bool))
    ].copy()
    failures["failure_type"] = np.where(
        ~failures["test_class_supported"].astype(bool),
        "unsupported_true_class",
        "misclassification",
    )
    failures.to_csv(output_dir / "failure_cases.csv", index=False)
    class_rows: list[dict[str, Any]] = []
    context_columns = [
        "stage",
        "scenario",
        "model",
        "representation",
        "seed",
    ]
    for keys, group in predictions.groupby(context_columns, dropna=False):
        for label in sorted(
            set(group["true_label"].astype(str))
            | set(group["predicted_label"].astype(str))
        ):
            true = group["true_label"].astype(str) == label
            predicted = group["predicted_label"].astype(str) == label
            true_positive = int((true & predicted).sum())
            support = int(true.sum())
            predicted_count = int(predicted.sum())
            class_rows.append(
                {
                    **dict(zip(context_columns, keys)),
                    "class_label": label,
                    "support": support,
                    "predicted_count": predicted_count,
                    "true_positive": true_positive,
                    "recall": (
                        true_positive / support if support else np.nan
                    ),
                    "precision": (
                        true_positive / predicted_count
                        if predicted_count
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(class_rows).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    poster = predictions[
        (predictions["stage"] == "poster")
        & (predictions["heldout_substrate_family"] == "Ag")
        & (predictions["true_label"] == "4np")
    ].copy()
    poster["localized_case"] = "heldout_Ag__all_4np"
    poster.to_csv(
        output_dir / "poster_localized_4np_failures.csv", index=False
    )


def canonicalize_selection_records(output_dir: Path) -> None:
    """Remove cache-resume serialization differences from selection records."""
    for name in (
        "selection_fold_metrics.csv",
        "selection_core_summary.csv",
        "selection_summary.csv",
        "minimal_sensitivity_summary.csv",
    ):
        path = output_dir / name
        frame = pd.read_csv(path, low_memory=False).sort_index(axis=1)
        frame.to_csv(path, index=False)
    history_path = output_dir / "selection_training_histories.csv"
    history = pd.read_csv(history_path, low_memory=False).sort_index(axis=1)
    float_columns = history.select_dtypes(include=["float32", "float64"]).columns
    history.loc[:, float_columns] = history.loc[:, float_columns].round(12)
    history.to_csv(history_path, index=False)


def uncertainty(
    output_dir: Path,
    baseline_bundle: Path,
    outer_vae: pd.DataFrame,
    poster_vae: pd.DataFrame,
    domain_vae: pd.DataFrame,
) -> None:
    rows: list[dict[str, Any]] = []
    outer = outer_vae[outer_vae["representation"] == "arpls_minmax"].copy()
    outer["cohort"] = cohort_from_scenario(outer)
    units = (
        outer.groupby(["cohort", "outer_fold"])[
            "balanced_accuracy_supported"
        ]
        .mean()
        .reset_index()
    )
    for cohort, group in units.groupby("cohort"):
        mean, half, n = t_interval(group["balanced_accuracy_supported"])
        rows.append(
            {
                "evaluation": "outer",
                "subset": cohort,
                "domain_type": "",
                "protocol": "grouped_outer_fold",
                "model": "VAE",
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "outer_fold",
                "n_units": n,
            }
        )
    poster = poster_vae[poster_vae["representation"] == "arpls_minmax"]
    poster_units = poster.groupby("heldout_substrate_family")[
        "balanced_accuracy_supported"
    ].mean()
    mean, half, n = t_interval(poster_units)
    rows.append(
        {
            "evaluation": "poster",
            "subset": "chemical_only_275",
            "domain_type": "substrate_family",
            "protocol": "leave_one_substrate_family_out",
            "model": "VAE",
            "mean_balanced_accuracy": mean,
            "ci95_half_width": half,
            "independent_unit": (
                "substrate_family_descriptive_not_independent_preparation"
            ),
            "n_units": n,
        }
    )
    for keys, group in domain_vae.groupby(
        ["evaluation_subset", "domain_protocol", "domain_type"]
    ):
        subset, protocol, domain_type = keys
        units = group.groupby("heldout_domain")[
            "balanced_accuracy_supported"
        ].mean()
        mean, half, n = t_interval(units)
        rows.append(
            {
                "evaluation": "domain_transfer",
                "subset": subset,
                "domain_type": domain_type,
                "protocol": protocol,
                "model": "VAE",
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "heldout_domain",
                "n_units": n,
            }
        )
    baseline_uncertainty = pd.read_csv(
        baseline_bundle / "uncertainty_summary.csv"
    )
    pd.concat(
        [baseline_uncertainty, pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "uncertainty_summary.csv", index=False)


def decisions(
    output_dir: Path,
    baseline_bundle: Path,
    selected: dict[str, Any],
    outer: pd.DataFrame,
    corruption: pd.DataFrame,
    poster: pd.DataFrame,
    domain: pd.DataFrame,
) -> dict[str, Any]:
    baseline_decisions = json.loads(
        (baseline_bundle / "final_decisions.json").read_text()
    )
    outer = outer.copy()
    outer["cohort"] = cohort_from_scenario(outer)
    outer_summary = (
        outer.groupby(["cohort", "representation"])[
            [
                "balanced_accuracy_supported",
                "macro_f1_supported",
                "reconstruction_median_row_correlation",
                "repeatable_peak_recall",
                "vae_kl_unnormalized_per_observation",
                "vae_active_units_var_mu_gt_0_01",
            ]
        ]
        .mean()
        .to_dict(orient="index")
    )
    composite = corruption[
        (corruption["scenario"].str.endswith("test_strict_core"))
        & (corruption["representation"] == "arpls_minmax")
        & (corruption["corruption"] == "composite")
        & (corruption["severity"] == 1.0)
    ]
    corruption_summary = composite[
        [
            "balanced_accuracy_supported",
            "prediction_agreement",
            "latent_cosine_drift",
            "reconstruction_mse",
        ]
    ].mean().to_dict()
    poster_arpls = poster[poster["representation"] == "arpls_minmax"]
    poster_mean = float(poster_arpls["balanced_accuracy_supported"].mean())
    poster_predictions = pd.read_csv(output_dir / "poster_predictions.csv")
    ag4np = poster_predictions[
        (poster_predictions["heldout_substrate_family"] == "Ag")
        & (poster_predictions["true_label"] == "4np")
        & (poster_predictions["representation"] == "arpls_minmax")
    ]
    domain_strict = domain[
        (domain["evaluation_subset"] == "strict_core")
        & (domain["domain_protocol"] == "domain_and_sample")
        & (domain["domain_type"] == "instrument")
    ]
    result = {
        "protocol": "sers-standard-vae-v1",
        "selected_configuration": selected["identifier"],
        "selected_kl_schedule": selected["kl_schedule"],
        "eligible_to_advance": selected["eligible_to_advance"],
        "selection_gate_failures": [
            name
            for name in (
                "gate_clean_correlation",
                "gate_peak_recall",
                "gate_chemical_probe",
                "gate_instrument_probe",
                "gate_same_master_distance",
                "gate_active_units",
                "gate_kl_dimensions",
                "gate_kl_range",
                "gate_finite",
                "gate_quality_sensitivity",
            )
            if not bool(
                selected["strict_core_and_quality_metrics"].get(name, True)
            )
        ],
        "outer": {
            f"{cohort}__{representation}": values
            for (cohort, representation), values in outer_summary.items()
        },
        "strict_core_composite_severity_1_arpls": corruption_summary,
        "poster_arpls_mean_balanced_accuracy": poster_mean,
        "poster_heldout_ag_4np_recall": float(ag4np["correct"].mean()),
        "strict_instrument_and_sample_balanced_accuracy": float(
            domain_strict["balanced_accuracy_supported"].mean()
        ),
        "frozen_comparator_context": baseline_decisions,
        "decision": (
            "Retain the standard VAE as a noncollapsed mixed-latent diagnostic "
            "and required comparator, but do not advance it as the final "
            "representation. It preserves competitive chemical classification "
            "and improves descriptive poster transfer, yet fails registered "
            "correlation, repeatable-peak, and same-master geometry gates and "
            "does not improve real NATO stress or unseen-instrument transfer."
        ),
        "next_stage": (
            "Proceed to a separately registered structured-latent experiment "
            "using this VAE as the mixed-latent reference. The next model must "
            "explicitly improve the chemical-versus-domain trade-off without "
            "further peak or reconstruction loss."
        ),
        "claim_limits": [
            "The one-block VAE is not a disentangled model.",
            "Poster rows are map locations without independent preparation IDs.",
            "Synthetic corruption and poster gains do not establish universal field denoising.",
            "Unsupported unseen analytes remain excluded from supported-class metrics and are listed separately.",
        ],
    }
    write_json(output_dir / "final_decisions.json", result)
    return result


def documentation(output_dir: Path, decision: dict[str, Any]) -> None:
    strict = decision["outer"]["strict_core__arpls_minmax"]
    quality = decision["outer"]["quality_pass__arpls_minmax"]
    stress = decision["outer"]["field_quality_stress__arpls_minmax"]
    readme = f"""# SERS standard VAE v1

This immutable bundle evaluates a one-block, unsupervised standard VAE after
the frozen deterministic baselines and before any structured/disentangled VAE.

## Outcome

The selected four-cycle KL schedule produced a noncollapsed posterior but did
not pass all predeclared advancement gates. It failed clean correlation,
repeatable-peak preservation, and same-master cross-instrument geometry.

- strict-core arPLS balanced accuracy: `{strict['balanced_accuracy_supported']:.6f}`;
- quality-pass arPLS balanced accuracy: `{quality['balanced_accuracy_supported']:.6f}`;
- field-stress arPLS balanced accuracy: `{stress['balanced_accuracy_supported']:.6f}`;
- descriptive poster transfer: `{decision['poster_arpls_mean_balanced_accuracy']:.6f}`;
- strict leave-instrument-and-sample transfer: `{decision['strict_instrument_and_sample_balanced_accuracy']:.6f}`.

The model remains a required mixed-latent comparator. It is not evidence of
disentanglement.

See `DECISION_REGISTRY.md`, `final_decisions.json`, and the tables/figures in
this directory for the complete result.
"""
    registry = f"""# Standard-VAE decision registry

1. **Schedule:** four-cycle KL annealing was selected using grouped inner data only.
2. **Posterior health:** the posterior is finite and broadly active; posterior collapse is not the failure mode.
3. **Chemical classification:** clean NATO performance is competitive with the deterministic AE but does not beat the frozen PCA reference.
4. **Preservation:** the VAE failed the clean correlation and repeatable-peak gates.
5. **Geometry:** instrument predictability decreased in inner probes, but same-master cross-instrument distance worsened and sealed instrument transfer did not improve.
6. **Robustness:** field-quality stress remains unsolved.
7. **Poster:** descriptive substrate transfer improved, including Ag/4NP, but lacks independent-preparation support.
8. **Decision:** retain as a diagnostic mixed-latent reference; proceed to a separately registered structured-latent model, not an unbounded post-hoc β search.

Failed gates: `{', '.join(decision['selection_gate_failures'])}`.
"""
    (output_dir / "README.md").write_text(readme)
    (output_dir / "DECISION_REGISTRY.md").write_text(registry)
    commands = """#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python scripts/run_sers_standard_vae_selection.py --device cuda
.venv/bin/python scripts/run_sers_standard_vae_final.py --stage all --training-device cuda --evaluation-device cpu
.venv/bin/python scripts/finalize_sers_standard_vae.py
.venv/bin/python scripts/run_sers_standard_vae_selection.py --device cuda --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/run_sers_standard_vae_final.py --stage all --training-device cuda --evaluation-device cpu --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/finalize_sers_standard_vae.py --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/compare_sers_standard_vae_rebuild.py
.venv/bin/python scripts/validate_sers_standard_vae.py --require-clean-rebuild
"""
    path = output_dir / "reproduction_commands.sh"
    path.write_text(commands)
    path.chmod(0o755)


def hash_artifacts(output_dir: Path) -> None:
    excluded = {
        "artifact_hashes.json",
        "validation_report.json",
        "clean_rebuild_comparison.json",
    }
    catalog = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name not in excluded
        and "selection_cache" not in path.parts
        and "run_cache" not in path.parts
    }
    write_json(
        output_dir / "artifact_hashes.json",
        {
            "algorithm": "sha256",
            "files": catalog,
        },
    )


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1",
    )
    parser.add_argument(
        "--baseline-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    baseline_bundle = args.baseline_bundle.resolve()
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    corruption = pd.read_csv(output_dir / "outer_fold_corruption_metrics.csv")
    baseline_outer = pd.read_csv(baseline_bundle / "outer_fold_metrics.csv")
    baseline_corruption = pd.read_csv(
        baseline_bundle / "outer_fold_corruption_metrics.csv"
    )
    baseline_poster = pd.read_csv(baseline_bundle / "poster_metrics.csv")
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    pd.read_csv(
        baseline_bundle / "split_seed_assignments.csv", low_memory=False
    ).to_csv(output_dir / "split_seed_assignments.csv", index=False)
    canonicalize_selection_records(output_dir)
    consolidate(output_dir)
    uncertainty(output_dir, baseline_bundle, outer, poster, domain)
    decision = decisions(
        output_dir,
        baseline_bundle,
        selected,
        outer,
        corruption,
        poster,
        domain,
    )
    figure_outer(output_dir, baseline_outer, outer)
    figure_corruption(
        output_dir, baseline_corruption, corruption
    )
    figure_poster(output_dir, baseline_poster, poster)
    documentation(output_dir, decision)
    hash_artifacts(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "decision": decision["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
