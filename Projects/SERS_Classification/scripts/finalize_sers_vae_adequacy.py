#!/usr/bin/env python3
"""Finalize the leakage-controlled SERS VAE adequacy result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import sers_baseline_common as baseline


COLORS = {
    "PCA-logistic": "#0072B2",
    "Siamese": "#009E73",
    "AE": "#E69F00",
    "DAE": "#D55E00",
    "VAE-100 β=1": "#8C564B",
    "VAE-500 β=0.25": "#CC79A7",
}
MODEL_ORDER = list(COLORS)
GEOMETRY_PATTERN = re.compile(
    r"^outer__(?P<scenario>nato_outer_o(?P<fold>\d+)__train_.+__test_.+)"
    r"__(?P<family>vae_adequacy|vae|ae|dae|siamese)"
    r"__(?P<representation>arpls_minmax|minimal_minmax|derivative_1)"
    r"(?:__(?P<model>.+))?__s(?P<seed>\d+)\.npz$"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


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
    return frame["scenario"].str.extract(r"__test_(.+)$")[0]


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


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def vectorized_geometry_metrics(
    features: np.ndarray,
    manifest: pd.DataFrame,
    domain_column: str = "instrument",
) -> dict[str, float]:
    """Equivalent pairwise correlation geometry without Python pair loops."""
    values = np.asarray(features, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1.0e-12)
    distances = 1.0 - normalized @ normalized.T
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    domains = manifest[domain_column].astype(str).to_numpy()
    masters = manifest["master_sample_id"].astype(str).to_numpy()
    targets = manifest["target_analyte"].astype(str).to_numpy()
    cross_domain = upper & (domains[:, None] != domains[None, :])
    same_mask = cross_domain & (masters[:, None] == masters[None, :])
    different_mask = cross_domain & (targets[:, None] != targets[None, :])
    same = distances[same_mask]
    different = distances[different_mask]
    same_mean = float(same.mean()) if len(same) else np.nan
    different_mean = (
        float(different.mean()) if len(different) else np.nan
    )
    return {
        "same_master_cross_domain_mean_distance": same_mean,
        "different_target_cross_domain_mean_distance": different_mean,
        "cross_domain_separation_margin": different_mean - same_mean,
        "same_master_pair_count": int(len(same)),
        "different_target_pair_count": int(len(different)),
    }


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
        "confirmatory_training_histories.csv": [
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
        ["stage", "scenario", "model", "representation", "seed"]
        + variational_columns
    ].to_csv(output_dir / "variational_metrics.csv", index=False)

    predictions = pd.read_csv(
        output_dir / "per_spectrum_predictions.csv", low_memory=False
    )
    supported = predictions["test_class_supported"].astype(bool)
    correct = predictions["correct"].astype(bool)
    failures = predictions[(~correct) | (~supported)].copy()
    failures["failure_type"] = np.where(
        ~failures["test_class_supported"].astype(bool),
        "unsupported_true_class",
        "misclassification",
    )
    failures.to_csv(output_dir / "failure_cases.csv", index=False)

    context_columns = [
        "stage",
        "scenario",
        "model",
        "representation",
        "seed",
    ]
    class_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(context_columns, dropna=False):
        true_labels = group["true_label"].astype(str)
        predicted_labels = group["predicted_label"].astype(str)
        for label in sorted(set(true_labels) | set(predicted_labels)):
            true = true_labels == label
            predicted = predicted_labels == label
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
                    "recall": true_positive / support if support else np.nan,
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

    instrument_rows = (
        predictions.assign(
            supported=supported,
            correct_supported=correct & supported,
        )
        .groupby(
            [
                "stage",
                "scenario",
                "representation",
                "seed",
                "instrument",
                "sensor_family",
            ],
            dropna=False,
        )
        .agg(
            observations=("observation_uid", "size"),
            supported_observations=("supported", "sum"),
            correct_supported=("correct_supported", "sum"),
            mean_confidence=("prediction_confidence", "mean"),
        )
        .reset_index()
    )
    instrument_rows["accuracy_supported"] = (
        instrument_rows["correct_supported"]
        / instrument_rows["supported_observations"].replace(0, np.nan)
    )
    instrument_rows.to_csv(
        output_dir / "per_instrument_failures.csv", index=False
    )


def compute_accounting(output_dir: Path) -> pd.DataFrame:
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    checkpoint = pd.read_csv(output_dir / "checkpoint_metrics.csv")
    stage_counts = {
        path.name: len(list(path.glob("*.pt")))
        for path in sorted((output_dir / "selection_cache").iterdir())
        if path.is_dir()
    }
    rows: list[dict[str, Any]] = []
    for stage, runs in stage_counts.items():
        stage_records = checkpoint[checkpoint["stage"].eq(stage)]
        rows.append(
            {
                "phase": "inner_selection",
                "stage": stage,
                "distinct_training_runs": runs,
                "epochs_per_run": 500,
                "total_training_epochs": runs * 500,
                "minimum_parameter_count": int(
                    stage_records["parameter_count"].min()
                ),
                "maximum_parameter_count": int(
                    stage_records["parameter_count"].max()
                ),
                "aggregate_metric_rows": len(stage_records),
                "canonical_training_device": "deterministic_cuda",
                "canonical_inference_device": "cpu",
            }
        )
    final_specs = [
        ("outer", 60, len(pd.read_csv(output_dir / "outer_fold_metrics.csv"))),
        ("poster", 24, len(pd.read_csv(output_dir / "poster_metrics.csv"))),
        (
            "domain",
            168,
            len(pd.read_csv(output_dir / "domain_transfer_metrics.csv")),
        ),
    ]
    for stage, runs, metric_rows in final_specs:
        rows.append(
            {
                "phase": "locked_confirmation",
                "stage": stage,
                "distinct_training_runs": runs,
                "epochs_per_run": int(selected["maximum_epoch"]),
                "total_training_epochs": (
                    runs * int(selected["maximum_epoch"])
                ),
                "minimum_parameter_count": 1082353,
                "maximum_parameter_count": 1082353,
                "aggregate_metric_rows": metric_rows,
                "canonical_training_device": "deterministic_cuda",
                "canonical_inference_device": "cpu",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        output_dir / "parameter_and_compute_accounting.csv", index=False
    )
    summary = {
        "protocol": "sers-vae-adequacy-v1",
        "distinct_inner_training_runs": int(
            result.loc[
                result["phase"].eq("inner_selection"),
                "distinct_training_runs",
            ].sum()
        ),
        "distinct_confirmatory_training_runs": int(
            result.loc[
                result["phase"].eq("locked_confirmation"),
                "distinct_training_runs",
            ].sum()
        ),
        "total_distinct_training_runs": int(
            result["distinct_training_runs"].sum()
        ),
        "total_training_epochs": int(
            result["total_training_epochs"].sum()
        ),
        "controlled_corruption_aggregate_rows": int(
            len(pd.read_csv(output_dir / "corruption_metrics.csv"))
        ),
        "spectrum_level_prediction_rows": int(
            len(pd.read_csv(output_dir / "per_spectrum_predictions.csv"))
        ),
        "spectrum_level_reconstruction_rows": int(
            len(pd.read_csv(output_dir / "reconstruction_metrics.csv"))
        ),
        "confirmatory_checkpoint_count": len(
            list((output_dir / "checkpoints").rglob("*.pt"))
        ),
        "selection_checkpoint_count": len(
            list((output_dir / "selection_cache").rglob("*.pt"))
        ),
        "parameter_range": [
            int(checkpoint["parameter_count"].min()),
            int(checkpoint["parameter_count"].max()),
        ],
        "selected_parameter_count": 1082353,
        "accounting_note": (
            "Epoch counts measure executed optimizer epochs. Outer field-stress "
            "evaluation reuses its matching quality-pass training checkpoints "
            "and is therefore counted as evaluation, not an additional training run."
        ),
    }
    write_json(output_dir / "compute_accounting.json", summary)
    return result


def same_master_geometry(
    output_dir: Path,
    nato_bundle: Path,
    standard_bundle: Path,
    baseline_bundle: Path,
) -> pd.DataFrame:
    datasets = {
        "strict_core": baseline.load_nato_dataset(nato_bundle),
    }
    # The quality and stress loaders are encoded directly in their frozen NPZs.
    for subset, manifest_name, npz_name in (
        (
            "quality_pass",
            "quality_preprocessing_manifest.csv",
            "final_model_inputs_quality.npz",
        ),
        (
            "field_quality_stress",
            "field_quality_stress_manifest.csv",
            "final_model_inputs_field_quality_stress.npz",
        ),
    ):
        manifest = pd.read_csv(nato_bundle / manifest_name)
        archive = np.load(nato_bundle / npz_name)
        representations = {
            key: archive[key].astype(np.float32)
            for key in ("arpls_minmax", "minimal_minmax", "derivative_1")
        }
        datasets[subset] = baseline.SpectralDataset(
            subset,
            archive["axis_cm1"],
            archive["observation_uid"].astype(str),
            representations,
            manifest,
        )

    sources = [
        (
            "adequacy500",
            output_dir / "embeddings",
            {"vae_adequacy"},
        ),
        (
            "standard100",
            standard_bundle / "embeddings",
            {"vae"},
        ),
        (
            "frozen_baseline",
            baseline_bundle / "embeddings",
            {"ae", "dae", "siamese"},
        ),
    ]
    rows: list[dict[str, Any]] = []
    raw_cache: dict[
        tuple[str, str, tuple[str, ...]], dict[str, float]
    ] = {}
    for source, directory, accepted_families in sources:
        for path in sorted(directory.glob("outer__*.npz")):
            match = GEOMETRY_PATTERN.match(path.name)
            if match is None or match.group("family") not in accepted_families:
                continue
            scenario = match.group("scenario")
            subset = scenario.rsplit("__test_", 1)[1]
            representation = match.group("representation")
            if representation not in datasets[subset].representations:
                continue
            family = match.group("family")
            model = match.group("model") or "siamese"
            if family == "ae" and "spectral_composite_clean" not in model:
                continue
            if family == "dae" and not (
                "mixed_uniform" in model or "mixed_progressive" in model
            ):
                continue
            archive = np.load(path)
            latent_key = (
                "posterior_mean"
                if "posterior_mean" in archive.files
                else "latent"
            )
            uids = archive["observation_uid"].astype(str)
            latent = archive[latent_key]
            dataset = datasets[subset]
            index = pd.Series(
                np.arange(len(dataset.manifest)),
                index=dataset.manifest["observation_uid"].astype(str),
            )
            positions = index.loc[uids].to_numpy(dtype=int)
            manifest = dataset.manifest.iloc[positions].reset_index(drop=True)
            raw = dataset.representations[representation][positions]
            raw_key = (scenario, representation, tuple(uids))
            if raw_key not in raw_cache:
                raw_cache[raw_key] = vectorized_geometry_metrics(
                    raw, manifest
                )
            raw_metrics = raw_cache[raw_key]
            latent_metrics = vectorized_geometry_metrics(latent, manifest)
            display = {
                "vae_adequacy": "VAE-500 β=0.25",
                "vae": "VAE-100 β=1",
                "ae": "AE",
                "dae": "DAE",
                "siamese": "Siamese",
            }[family]
            rows.append(
                {
                    "source": source,
                    "display_model": display,
                    "scenario": scenario,
                    "outer_fold": int(match.group("fold")),
                    "subset": subset,
                    "representation": representation,
                    "model_family": family,
                    "model": model,
                    "seed": int(match.group("seed")),
                    "raw_same_master_cross_instrument_distance": raw_metrics[
                        "same_master_cross_domain_mean_distance"
                    ],
                    "latent_same_master_cross_instrument_distance": latent_metrics[
                        "same_master_cross_domain_mean_distance"
                    ],
                    "same_master_distance_delta": (
                        latent_metrics[
                            "same_master_cross_domain_mean_distance"
                        ]
                        - raw_metrics[
                            "same_master_cross_domain_mean_distance"
                        ]
                    ),
                    "latent_different_target_cross_instrument_distance": (
                        latent_metrics[
                            "different_target_cross_domain_mean_distance"
                        ]
                    ),
                    "latent_cross_instrument_separation_margin": latent_metrics[
                        "cross_domain_separation_margin"
                    ],
                    "same_master_pair_count": latent_metrics[
                        "same_master_pair_count"
                    ],
                    "different_target_pair_count": latent_metrics[
                        "different_target_pair_count"
                    ],
                }
            )
    result = pd.DataFrame(rows).sort_values(
        [
            "subset",
            "representation",
            "display_model",
            "outer_fold",
            "seed",
        ]
    )
    result.to_csv(
        output_dir / "outer_fold_same_master_geometry.csv", index=False
    )
    summary = (
        result.groupby(
            ["subset", "representation", "display_model"], as_index=False
        )[
            [
                "raw_same_master_cross_instrument_distance",
                "latent_same_master_cross_instrument_distance",
                "same_master_distance_delta",
                "latent_cross_instrument_separation_margin",
            ]
        ]
        .mean()
        .sort_values(["subset", "representation", "display_model"])
    )
    summary.to_csv(
        output_dir / "same_master_geometry_summary.csv", index=False
    )
    return result


def comparator_summary(
    output_dir: Path, standard_bundle: Path, baseline_bundle: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        evaluation: str,
        subset: str,
        representation: str,
        display_model: str,
        metric: str,
        values: pd.Series,
        independent_units: int,
    ) -> None:
        array = pd.to_numeric(values, errors="coerce")
        rows.append(
            {
                "evaluation": evaluation,
                "subset": subset,
                "representation": representation,
                "display_model": display_model,
                "metric": metric,
                "mean": float(array.mean()),
                "standard_deviation": float(array.std(ddof=1)),
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "record_count": int(array.notna().sum()),
                "independent_unit_count": independent_units,
            }
        )

    outer_sources = [
        (
            "VAE-500 β=0.25",
            pd.read_csv(output_dir / "outer_fold_metrics.csv"),
            "adequacy",
        ),
        (
            "VAE-100 β=1",
            pd.read_csv(standard_bundle / "outer_fold_metrics.csv"),
            "standard",
        ),
    ]
    baseline_outer = pd.read_csv(baseline_bundle / "outer_fold_metrics.csv")
    baseline_outer["display_model"] = baseline_display(baseline_outer)
    for display, data, _ in outer_sources:
        data = data.copy()
        data["display_model"] = display
        data["cohort"] = cohort_from_scenario(data)
        for (subset, rep), group in data.groupby(
            ["cohort", "representation"]
        ):
            for metric in (
                "balanced_accuracy_supported",
                "macro_f1_supported",
                "reconstruction_median_row_correlation",
                "repeatable_peak_recall",
            ):
                add("outer", subset, rep, display, metric, group[metric], 5)
    baseline_outer["cohort"] = cohort_from_scenario(baseline_outer)
    for (subset, rep, display), group in baseline_outer[
        baseline_outer["display_model"].notna()
    ].groupby(["cohort", "representation", "display_model"]):
        add(
            "outer",
            subset,
            rep,
            display,
            "balanced_accuracy_supported",
            group["balanced_accuracy_supported"],
            5,
        )
        if display in {"AE", "DAE"}:
            for metric in (
                "reconstruction_median_row_correlation",
                "repeatable_peak_recall",
            ):
                add(
                    "outer",
                    subset,
                    rep,
                    display,
                    metric,
                    group[metric],
                    5,
                )

    for display, root in (
        ("VAE-500 β=0.25", output_dir),
        ("VAE-100 β=1", standard_bundle),
    ):
        domain = pd.read_csv(root / "domain_transfer_metrics.csv")
        for keys, group in domain.groupby(
            ["evaluation_subset", "domain_protocol", "domain_type"]
        ):
            subset, protocol, domain_type = keys
            units = group.groupby("heldout_domain")[
                "balanced_accuracy_supported"
            ].mean()
            add(
                "domain_transfer",
                f"{subset}__{protocol}__{domain_type}",
                "arpls_minmax",
                display,
                "balanced_accuracy_supported",
                units,
                len(units),
            )
        poster = pd.read_csv(root / "poster_metrics.csv")
        for rep, group in poster.groupby("representation"):
            units = group.groupby("heldout_substrate_family")[
                "balanced_accuracy_supported"
            ].mean()
            add(
                "poster",
                "leave_substrate_family_out",
                rep,
                display,
                "balanced_accuracy_supported",
                units,
                len(units),
            )
        corruption = pd.read_csv(root / "outer_fold_corruption_metrics.csv")
        corruption = corruption[
            corruption["scenario"].str.endswith("test_strict_core")
            & corruption["corruption"].eq("composite")
            & corruption["severity"].eq(1.0)
        ]
        for rep, group in corruption.groupby("representation"):
            for metric in (
                "balanced_accuracy_supported",
                "prediction_agreement",
                "latent_cosine_drift",
                "reconstruction_mse",
            ):
                add(
                    "corruption",
                    "strict_core__composite__severity1",
                    rep,
                    display,
                    metric,
                    group[metric],
                    5,
                )

    geometry = pd.read_csv(output_dir / "outer_fold_same_master_geometry.csv")
    for (subset, rep, display), group in geometry.groupby(
        ["subset", "representation", "display_model"]
    ):
        for metric in (
            "latent_same_master_cross_instrument_distance",
            "same_master_distance_delta",
            "latent_cross_instrument_separation_margin",
        ):
            add(
                "same_master_geometry",
                subset,
                rep,
                display,
                metric,
                group[metric],
                5,
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluation", "subset", "representation", "metric", "display_model"]
    )
    result.to_csv(output_dir / "comparator_summary.csv", index=False)
    return result


def uncertainty(
    output_dir: Path, standard_bundle: Path
) -> None:
    existing = pd.read_csv(standard_bundle / "uncertainty_summary.csv")
    existing.loc[existing["model"].eq("VAE"), "model"] = "VAE-100 β=1"
    rows: list[dict[str, Any]] = []
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    outer["cohort"] = cohort_from_scenario(outer)
    for (cohort, representation), group in outer.groupby(
        ["cohort", "representation"]
    ):
        units = group.groupby("outer_fold")[
            "balanced_accuracy_supported"
        ].mean()
        mean, half, n = t_interval(units)
        rows.append(
            {
                "evaluation": "outer",
                "subset": cohort,
                "domain_type": "",
                "protocol": "grouped_outer_fold",
                "model": "VAE-500 β=0.25",
                "representation": representation,
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "outer_fold",
                "n_units": n,
            }
        )
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    for representation, group in poster.groupby("representation"):
        units = group.groupby("heldout_substrate_family")[
            "balanced_accuracy_supported"
        ].mean()
        mean, half, n = t_interval(units)
        rows.append(
            {
                "evaluation": "poster",
                "subset": "chemical_only_275",
                "domain_type": "substrate_family",
                "protocol": "leave_one_substrate_family_out",
                "model": "VAE-500 β=0.25",
                "representation": representation,
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": (
                    "substrate_family_descriptive_not_independent_preparation"
                ),
                "n_units": n,
            }
        )
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    for keys, group in domain.groupby(
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
                "model": "VAE-500 β=0.25",
                "representation": "arpls_minmax",
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "heldout_domain",
                "n_units": n,
            }
        )
    pd.concat(
        [existing, pd.DataFrame(rows)], ignore_index=True, sort=False
    ).to_csv(output_dir / "uncertainty_summary.csv", index=False)


def figure_convergence(output_dir: Path) -> None:
    histories = pd.read_csv(output_dir / "training_histories.csv")
    histories = histories[histories["stage"].eq("stage_1_convergence")]
    summaries = pd.read_csv(output_dir / "stage_summaries.csv")
    labels = {
        "constant_lr": "constant 1e-3",
        "step_lr_300": "step to 1e-4 at epoch 301",
        "plateau_lr": "plateau",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for policy, group in histories.groupby(
        histories["configuration"].str.extract(
            r"__(constant_lr|step_lr_300|plateau_lr)__"
        )[0]
    ):
        epoch = group.groupby("epoch")["validation_loss"]
        mean = epoch.mean()
        q1 = epoch.quantile(0.25)
        q3 = epoch.quantile(0.75)
        axes[0, 0].plot(mean.index, mean, label=labels.get(policy, policy))
        axes[0, 0].fill_between(mean.index, q1, q3, alpha=0.16)
    axes[0, 0].axvline(100, linestyle="--", color="0.35", linewidth=0.9)
    axes[0, 0].axvline(300, linestyle=":", color="0.35", linewidth=0.9)
    axes[0, 0].set(
        title="Validation ELBO trajectory",
        xlabel="Epoch",
        ylabel="Validation loss",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    metrics = [
        (
            "clean_reconstruction_median_row_correlation",
            "Median reconstruction correlation",
            "Correlation",
            axes[0, 1],
        ),
        (
            "clean_repeatable_peak_recall",
            "Repeatable-peak recall",
            "Peak recall",
            axes[1, 0],
        ),
        (
            "latent_target_balanced_accuracy",
            "Chemical balanced accuracy",
            "Balanced accuracy",
            axes[1, 1],
        ),
    ]
    for metric, title, ylabel, ax in metrics:
        for policy, group in summaries.groupby("optimizer_policy"):
            group = group.sort_values("checkpoint_epoch")
            ax.plot(
                group["checkpoint_epoch"],
                group[metric],
                marker="o",
                markersize=3,
                label=labels.get(policy, policy),
            )
        ax.axvline(100, linestyle="--", color="0.35", linewidth=0.9)
        ax.set(title=title, xlabel="Checkpoint epoch", ylabel=ylabel)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Stage 1: 100 epochs was not a converged endpoint")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "convergence_diagnostics")


def figure_ablation(output_dir: Path) -> None:
    specs = [
        (
            "stage_2_architecture_summary.csv",
            "architecture",
            "Backbone",
        ),
        ("stage_2_loss_summary.csv", "reconstruction_loss", "Loss"),
        ("stage_2_latent_summary.csv", "latent_dimension", "Latent size"),
        ("stage_2_beta_summary.csv", "beta_target", "KL strength β"),
    ]
    metrics = [
        ("latent_target_balanced_accuracy", "Chemical BA", "#0072B2"),
        (
            "clean_reconstruction_median_row_correlation",
            "Correlation",
            "#009E73",
        ),
        ("clean_repeatable_peak_recall", "Peak recall", "#D55E00"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (filename, key, title) in zip(axes.flat, specs):
        frame = pd.read_csv(output_dir / filename)
        frame = frame.sort_values(key)
        x = np.arange(len(frame))
        width = 0.23
        for offset, (metric, label, color) in zip((-1, 0, 1), metrics):
            ax.bar(
                x + offset * width,
                frame[metric],
                width,
                label=label,
                color=color,
                alpha=0.86,
            )
        labels = [str(value).replace("_", "\n") for value in frame[key]]
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0.35, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for index, converged in enumerate(
            frame["converged"].fillna(False).astype(bool)
        ):
            ax.text(
                index,
                0.37,
                "converged" if converged else "not conv.",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=6,
            )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("Sequential inner-only architecture and objective ablations")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "bounded_ablation")


def figure_outer(
    output_dir: Path, standard_bundle: Path, baseline_bundle: Path
) -> None:
    frames: list[pd.DataFrame] = []
    base = pd.read_csv(baseline_bundle / "outer_fold_metrics.csv")
    base["display_model"] = baseline_display(base)
    base = base[base["display_model"].notna()]
    frames.append(base)
    for display, root in (
        ("VAE-100 β=1", standard_bundle),
        ("VAE-500 β=0.25", output_dir),
    ):
        frame = pd.read_csv(root / "outer_fold_metrics.csv")
        frame = frame[frame["representation"].eq("arpls_minmax")].copy()
        frame["display_model"] = display
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["cohort"] = cohort_from_scenario(combined)
    units = (
        combined.groupby(["cohort", "display_model", "outer_fold"])[
            "balanced_accuracy_supported"
        ]
        .mean()
        .reset_index()
    )
    cohorts = [
        ("strict_core", "Strict core (598)"),
        ("quality_pass", "Quality pass (500)"),
        ("field_quality_stress", "Field stress (98)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    for ax, (cohort, title) in zip(axes, cohorts):
        part = units[units["cohort"].eq(cohort)]
        for index, model in enumerate(MODEL_ORDER):
            values = part.loc[
                part["display_model"].eq(model), "balanced_accuracy_supported"
            ].to_numpy()
            if not len(values):
                continue
            mean, half, _ = t_interval(values)
            ax.scatter(
                np.full(len(values), index),
                values,
                color=COLORS[model],
                alpha=0.55,
                s=16,
            )
            ax.errorbar(
                index,
                mean,
                yerr=half,
                fmt="D",
                color=COLORS[model],
                capsize=2,
                markersize=4,
            )
        ax.axhline(1 / 7, linestyle="--", color="0.5", linewidth=0.8)
        ax.set_xticks(
            range(len(MODEL_ORDER)), MODEL_ORDER, rotation=50, ha="right"
        )
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Balanced accuracy")
    fig.suptitle("Locked grouped outer comparison (arPLS primary view)")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "outer_comparison")


def figure_preservation(
    output_dir: Path, standard_bundle: Path, baseline_bundle: Path
) -> None:
    frames = []
    for display, root in (
        ("VAE-500 β=0.25", output_dir),
        ("VAE-100 β=1", standard_bundle),
    ):
        frame = pd.read_csv(root / "outer_fold_metrics.csv")
        frame["display_model"] = display
        frames.append(frame)
    baseline_frame = pd.read_csv(baseline_bundle / "outer_fold_metrics.csv")
    baseline_frame["display_model"] = baseline_display(baseline_frame)
    frames.append(
        baseline_frame[
            baseline_frame["display_model"].isin(["AE", "DAE"])
        ]
    )
    data = pd.concat(frames, ignore_index=True)
    data["cohort"] = cohort_from_scenario(data)
    data = data[data["cohort"].isin(["strict_core", "quality_pass"])]
    summary = data.groupby(
        ["cohort", "representation", "display_model"], as_index=False
    )[
        [
            "reconstruction_median_row_correlation",
            "repeatable_peak_recall",
        ]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey="row")
    for column, representation in enumerate(
        ["arpls_minmax", "minimal_minmax"]
    ):
        for row, (metric, ylabel) in enumerate(
            [
                (
                    "reconstruction_median_row_correlation",
                    "Median row correlation",
                ),
                ("repeatable_peak_recall", "Repeatable-peak recall"),
            ]
        ):
            ax = axes[row, column]
            part = summary[summary["representation"].eq(representation)]
            x = np.arange(2)
            models = ["AE", "DAE", "VAE-100 β=1", "VAE-500 β=0.25"]
            width = 0.19
            for offset, model in enumerate(models):
                values = (
                    part[part["display_model"].eq(model)]
                    .set_index("cohort")[metric]
                    .reindex(["strict_core", "quality_pass"])
                )
                ax.bar(
                    x + (offset - 1.5) * width,
                    values,
                    width,
                    label=model,
                    color=COLORS[model],
                )
            ax.set_xticks(x, ["Strict", "Quality"])
            ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel)
            ax.set_title(representation.replace("_", " "))
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Reconstruction and peak preservation by preprocessing view")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "preservation_comparison")


def figure_corruption(output_dir: Path, standard_bundle: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True)
    corruption_order = [
        "gaussian_noise",
        "isolated_spikes",
        "smooth_baseline",
        "scale_offset",
        "edge_filled_shift",
        "composite",
    ]
    for ax, corruption_name in zip(axes.flat, corruption_order):
        for display, root in (
            ("VAE-100 β=1", standard_bundle),
            ("VAE-500 β=0.25", output_dir),
        ):
            frame = pd.read_csv(root / "outer_fold_corruption_metrics.csv")
            frame = frame[
                frame["scenario"].str.endswith("test_strict_core")
                & frame["representation"].eq("arpls_minmax")
                & frame["corruption"].eq(corruption_name)
            ]
            units = (
                frame.groupby(["outer_fold", "severity"])[
                    "balanced_accuracy_supported"
                ]
                .mean()
                .reset_index()
            )
            mean = units.groupby("severity")[
                "balanced_accuracy_supported"
            ].mean()
            ax.plot(
                mean.index,
                mean.values,
                marker="o",
                color=COLORS[display],
                label=display,
            )
        ax.set_title(corruption_name.replace("_", " "))
        ax.set_ylim(0.1, 0.8)
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Severity")
    axes[1, 1].set_xlabel("Severity")
    axes[1, 2].set_xlabel("Severity")
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Controlled corruption: classification robustness")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "corruption_robustness")


def figure_domain(output_dir: Path, standard_bundle: Path) -> None:
    frames = []
    for display, root in (
        ("VAE-500 β=0.25", output_dir),
        ("VAE-100 β=1", standard_bundle),
    ):
        frame = pd.read_csv(root / "domain_transfer_metrics.csv")
        frame["display_model"] = display
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    strict = data[
        data["evaluation_subset"].eq("strict_core")
        & data["domain_protocol"].eq("domain_and_sample")
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0))
    for ax, domain_type in zip(axes, ["instrument", "sensor_family"]):
        part = strict[strict["domain_type"].eq(domain_type)]
        table = part.pivot_table(
            index="heldout_domain",
            columns="display_model",
            values="balanced_accuracy_supported",
            aggfunc="mean",
        ).reindex(columns=["VAE-100 β=1", "VAE-500 β=0.25"])
        x = np.arange(len(table))
        width = 0.38
        for offset, model in enumerate(table.columns):
            ax.bar(
                x + (offset - 0.5) * width,
                table[model],
                width,
                label=model,
                color=COLORS[model],
            )
        ax.set_xticks(x, table.index, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"Held-out {domain_type.replace('_', ' ')}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Strict domain-and-sample transfer")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "domain_transfer")


def figure_poster(output_dir: Path, standard_bundle: Path) -> None:
    frames = []
    for display, root in (
        ("VAE-500 β=0.25", output_dir),
        ("VAE-100 β=1", standard_bundle),
    ):
        frame = pd.read_csv(root / "poster_metrics.csv")
        frame["display_model"] = display
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, representation in zip(
        axes, ["arpls_minmax", "minimal_minmax"]
    ):
        table = data[data["representation"].eq(representation)].pivot_table(
            index="heldout_substrate_family",
            columns="display_model",
            values="balanced_accuracy_supported",
            aggfunc="mean",
        )
        table = table.reindex(columns=["VAE-100 β=1", "VAE-500 β=0.25"])
        x = np.arange(len(table))
        width = 0.38
        for offset, model in enumerate(table.columns):
            ax.bar(
                x + (offset - 0.5) * width,
                table[model],
                width,
                label=model,
                color=COLORS[model],
            )
        ax.set_xticks(x, table.index)
        ax.set_ylim(0, 1)
        ax.set_title(representation.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Descriptive leave-substrate-family-out poster transfer")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "poster_transfer")


def instrument_reconstruction_summary(
    output_dir: Path, nato_bundle: Path
) -> pd.DataFrame:
    core = pd.read_csv(nato_bundle / "core_preprocessing_manifest.csv")
    quality = pd.read_csv(nato_bundle / "quality_preprocessing_manifest.csv")
    stress = pd.read_csv(nato_bundle / "field_quality_stress_manifest.csv")
    manifests = {
        "strict_core": core,
        "quality_pass": quality,
        "field_quality_stress": stress,
    }
    reconstruction = pd.read_csv(
        output_dir / "outer_fold_reconstruction_metrics.csv"
    )
    reconstruction["subset"] = reconstruction["scenario"].str.extract(
        r"__test_(.+)$"
    )[0]
    frames = []
    for subset, frame in reconstruction.groupby("subset"):
        metadata = manifests[subset][
            ["observation_uid", "instrument", "sensor_family", "target_analyte"]
        ]
        frames.append(frame.merge(metadata, on="observation_uid", how="left"))
    joined = pd.concat(frames, ignore_index=True)
    summary = (
        joined.groupby(
            ["subset", "representation", "instrument", "sensor_family"],
            dropna=False,
        )
        .agg(
            spectra=("observation_uid", "size"),
            median_row_correlation=("pearson_correlation", "median"),
            mean_row_correlation=("pearson_correlation", "mean"),
            repeatable_reference_peaks=(
                "repeatable_reference_peak_count",
                "sum",
            ),
            repeatable_matched_peaks=("repeatable_matched_peak_count", "sum"),
            median_peak_shift_cm1=("median_peak_shift_cm1", "median"),
            median_relative_peak_width_change=(
                "median_absolute_relative_peak_width_change",
                "median",
            ),
        )
        .reset_index()
    )
    summary["repeatable_peak_recall"] = (
        summary["repeatable_matched_peaks"]
        / summary["repeatable_reference_peaks"].replace(0, np.nan)
    )
    summary.to_csv(
        output_dir / "instrument_reconstruction_summary.csv", index=False
    )
    return summary


def figure_instrument_spectra(
    output_dir: Path, standard_bundle: Path, nato_bundle: Path
) -> None:
    dataset = baseline.load_nato_dataset(nato_bundle)
    manifest = dataset.manifest
    uid_to_position = pd.Series(
        np.arange(len(manifest)),
        index=manifest["observation_uid"].astype(str),
    )
    records: list[dict[str, Any]] = []
    for fold in range(5):
        scenario = (
            f"nato_outer_o{fold}__train_strict_core__test_strict_core"
        )
        suffix_new = (
            f"outer__{scenario}__vae_adequacy__arpls_minmax__"
            "base_maxpool__z64__spectral_composite__beta0p25__"
            "constant_lr__e500__s1729.npz"
        )
        candidates_old = list(
            (standard_bundle / "reconstructions").glob(
                f"outer__{scenario}__vae__arpls_minmax__*__s1729.npz"
            )
        )
        new_path = output_dir / "reconstructions" / suffix_new
        if not new_path.is_file() or len(candidates_old) != 1:
            continue
        new = np.load(new_path)
        old = np.load(candidates_old[0])
        old_index = {
            uid: index
            for index, uid in enumerate(old["observation_uid"].astype(str))
        }
        for index, uid in enumerate(new["observation_uid"].astype(str)):
            position = int(uid_to_position.loc[uid])
            clean = new["clean"][index]
            reconstructed = new["reconstructed"][index]
            corr = float(np.corrcoef(clean, reconstructed)[0, 1])
            records.append(
                {
                    "uid": uid,
                    "instrument": str(manifest.iloc[position]["instrument"]),
                    "target": str(
                        manifest.iloc[position]["target_analyte"]
                    ),
                    "clean": clean,
                    "new": reconstructed,
                    "old": old["reconstructed"][old_index[uid]],
                    "correlation": corr,
                }
            )
    selected = []
    records_frame = pd.DataFrame(
        [
            {
                "index": index,
                "instrument": row["instrument"],
                "correlation": row["correlation"],
            }
            for index, row in enumerate(records)
        ]
    )
    for instrument, group in records_frame.groupby("instrument"):
        median = group["correlation"].median()
        chosen = (group["correlation"] - median).abs().idxmin()
        selected.append(records[int(records_frame.loc[chosen, "index"])])
    selected.sort(key=lambda row: row["instrument"])
    n = len(selected)
    columns = 2
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(11.0, 2.1 * rows), sharex=True
    )
    axis = np.arange(400, 1801)
    for ax, record in zip(np.asarray(axes).flat, selected):
        ax.plot(axis, record["clean"], color="black", linewidth=0.9, label="Input")
        ax.plot(
            axis,
            record["old"],
            color=COLORS["VAE-100 β=1"],
            linewidth=0.8,
            alpha=0.9,
            label="VAE-100",
        )
        ax.plot(
            axis,
            record["new"],
            color=COLORS["VAE-500 β=0.25"],
            linewidth=0.8,
            alpha=0.9,
            label="VAE-500",
        )
        ax.set_title(
            f"{record['instrument']} · {record['target']}", fontsize=9
        )
        ax.grid(alpha=0.15)
    for ax in np.asarray(axes).flat[n:]:
        ax.axis("off")
    for ax in np.asarray(axes)[-1, :]:
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
    np.asarray(axes).flat[0].legend(frameon=False, fontsize=7, ncol=3)
    fig.suptitle(
        "Median-fidelity strict-core example from every instrument"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "instrument_spectra")


def decisions(
    output_dir: Path,
    standard_bundle: Path,
    baseline_bundle: Path,
) -> dict[str, Any]:
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    beta = pd.read_csv(output_dir / "stage_2_beta_summary.csv")
    architecture = pd.read_csv(
        output_dir / "stage_2_architecture_summary.csv"
    )
    loss = pd.read_csv(output_dir / "stage_2_loss_summary.csv")
    latent = pd.read_csv(output_dir / "stage_2_latent_summary.csv")
    stage1 = pd.read_csv(output_dir / "stage_summaries.csv")
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    outer["cohort"] = cohort_from_scenario(outer)
    standard_outer = pd.read_csv(standard_bundle / "outer_fold_metrics.csv")
    standard_outer["cohort"] = cohort_from_scenario(standard_outer)
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    geometry = pd.read_csv(output_dir / "same_master_geometry_summary.csv")

    def outer_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        result = (
            frame.groupby(["cohort", "representation"])[
                [
                    "balanced_accuracy_supported",
                    "macro_f1_supported",
                    "reconstruction_median_row_correlation",
                    "repeatable_peak_recall",
                ]
            ]
            .mean()
            .to_dict(orient="index")
        )
        return {
            f"{cohort}__{representation}": json_clean(values)
            for (cohort, representation), values in result.items()
        }

    selected_inner = beta[
        beta["identifier"].eq(selected["identifier"])
    ].iloc[0]
    gate_names = [
        "gate_clean_correlation",
        "gate_peak_recall",
        "gate_chemical_probe",
        "gate_instrument_probe",
        "gate_same_master_distance",
        "gate_active_units",
        "gate_kl_dimensions",
        "gate_kl_range",
        "gate_finite",
    ]
    failed_gates = [
        name for name in gate_names if not bool(selected_inner[name])
    ]
    old_strict = standard_outer[
        standard_outer["cohort"].eq("strict_core")
        & standard_outer["representation"].eq("arpls_minmax")
    ]
    new_strict = outer[
        outer["cohort"].eq("strict_core")
        & outer["representation"].eq("arpls_minmax")
    ]
    strict_domain = domain[
        domain["evaluation_subset"].eq("strict_core")
        & domain["domain_protocol"].eq("domain_and_sample")
        & domain["domain_type"].eq("instrument")
    ]
    sensor_domain = domain[
        domain["evaluation_subset"].eq("strict_core")
        & domain["domain_protocol"].eq("domain_and_sample")
        & domain["domain_type"].eq("sensor_family")
    ]
    poster_arpls = poster[poster["representation"].eq("arpls_minmax")]
    convergence500 = stage1[
        stage1["optimizer_policy"].eq("constant_lr")
        & stage1["checkpoint_epoch"].eq(500)
    ].iloc[0]
    convergence300 = stage1[
        stage1["optimizer_policy"].eq("constant_lr")
        & stage1["checkpoint_epoch"].eq(300)
    ].iloc[0]

    attribution = {
        "protocol": "sers-vae-adequacy-v1",
        "selected_configuration": selected["identifier"],
        "categories": {
            "convergence": {
                "verdict": "confirmed_material_contributor",
                "evidence": {
                    "epoch300_converged": bool(convergence300["converged"]),
                    "epoch500_converged": bool(convergence500["converged"]),
                    "epoch500_median_relative_elbo_improvement_final_50": float(
                        convergence500["median_relative_improvement_50"]
                    ),
                    "outer_strict_correlation_change_vs_100": float(
                        new_strict[
                            "reconstruction_median_row_correlation"
                        ].mean()
                        - old_strict[
                            "reconstruction_median_row_correlation"
                        ].mean()
                    ),
                    "outer_strict_peak_recall_change_vs_100": float(
                        new_strict["repeatable_peak_recall"].mean()
                        - old_strict["repeatable_peak_recall"].mean()
                    ),
                    "outer_strict_balanced_accuracy_change_vs_100": float(
                        new_strict["balanced_accuracy_supported"].mean()
                        - old_strict[
                            "balanced_accuracy_supported"
                        ].mean()
                    ),
                },
                "interpretation": (
                    "The 100-epoch boundary was premature for spectral "
                    "preservation, but extending training alone did not improve "
                    "ordinary strict-core classification materially."
                ),
            },
            "architecture_downsampling": {
                "verdict": "not_supported_as_primary_failure",
                "evidence": json_clean(
                    architecture[
                        [
                            "architecture",
                            "converged",
                            "latent_target_balanced_accuracy",
                            "clean_reconstruction_median_row_correlation",
                            "clean_repeatable_peak_recall",
                        ]
                    ].to_dict(orient="records")
                ),
                "interpretation": (
                    "Neither the residual/multiscale candidate nor the "
                    "single-pool peak-preserving candidate produced a "
                    "converged, consistent improvement over the base backbone."
                ),
            },
            "reconstruction_objective": {
                "verdict": "peak_aware_loss_not_a_solution",
                "evidence": json_clean(
                    loss[
                        [
                            "reconstruction_loss",
                            "latent_target_balanced_accuracy",
                            "clean_reconstruction_median_row_correlation",
                            "clean_repeatable_peak_recall",
                        ]
                    ].to_dict(orient="records")
                ),
                "interpretation": (
                    "The multiscale peak-aware loss improved correlation but "
                    "did not improve repeatable-peak recall and slightly reduced "
                    "chemical accuracy."
                ),
            },
            "latent_capacity_or_kl_pressure": {
                "verdict": "strong_tradeoff_confirmed",
                "evidence": {
                    "latent_candidates": json_clean(
                        latent[
                            [
                                "latent_dimension",
                                "converged",
                                "latent_target_balanced_accuracy",
                                "clean_reconstruction_median_row_correlation",
                                "clean_repeatable_peak_recall",
                            ]
                        ].to_dict(orient="records")
                    ),
                    "beta_candidates": json_clean(
                        beta[
                            [
                                "beta_target",
                                "converged",
                                "latent_target_balanced_accuracy",
                                "clean_reconstruction_median_row_correlation",
                                "clean_repeatable_peak_recall",
                                "latent_instrument_probe_increment",
                                "latent_same_master_cross_instrument_distance",
                                "vae_kl_unnormalized_per_observation",
                                "vae_active_units_var_mu_gt_0_01",
                            ]
                        ].to_dict(orient="records")
                    ),
                },
                "interpretation": (
                    "Lower β preserved spectra and peaks; higher β suppressed "
                    "instrument information but damaged chemistry and spectral "
                    "fidelity. Changing latent width did not remove this tradeoff."
                ),
            },
            "data_coverage_domain_shift": {
                "verdict": "dominant_unresolved_confirmatory_failure",
                "evidence": {
                    "strict_instrument_and_sample_balanced_accuracy": float(
                        strict_domain["balanced_accuracy_supported"].mean()
                    ),
                    "strict_sensor_and_sample_balanced_accuracy": float(
                        sensor_domain["balanced_accuracy_supported"].mean()
                    ),
                    "field_stress_arpls_balanced_accuracy": float(
                        outer[
                            outer["cohort"].eq("field_quality_stress")
                            & outer["representation"].eq("arpls_minmax")
                        ]["balanced_accuracy_supported"].mean()
                    ),
                    "poster_arpls_balanced_accuracy": float(
                        poster_arpls["balanced_accuracy_supported"].mean()
                    ),
                },
                "interpretation": (
                    "Field stress and unseen sensor families remain much harder "
                    "than grouped clean cohorts. Quality filtering does not erase "
                    "the sensor-family failure."
                ),
            },
            "unresolved_interaction": {
                "verdict": "standard_mixed_latent_inadequate_for_invariance",
                "evidence": {
                    "failed_inner_gates": failed_gates,
                    "outer_same_master_summary": json_clean(
                        geometry[
                            geometry["display_model"].eq(
                                "VAE-500 β=0.25"
                            )
                        ].to_dict(orient="records")
                    ),
                },
                "interpretation": (
                    "The same mixed latent must reconstruct nuisance-rich spectra "
                    "and support chemistry. The registered search found no "
                    "setting that achieved both preservation and invariance."
                ),
            },
        },
    }
    write_json(output_dir / "failure_attribution.json", attribution)

    result = {
        "protocol": "sers-vae-adequacy-v1",
        "selection_closed_without_locked_outcomes": bool(
            selected["selection_closed"]
            and not any(
                selected[key]
                for key in (
                    "outer_used",
                    "field_quality_stress_used",
                    "domain_used",
                    "poster_used",
                )
            )
        ),
        "selected_configuration": selected["identifier"],
        "parameter_count": int(selected_inner["parameter_count"]),
        "converged": bool(selected_inner["converged"]),
        "inner_gate_count": int(selected_inner["gate_count"]),
        "inner_gate_total": int(selected_inner["gate_total"]),
        "failed_inner_gates": failed_gates,
        "inner_primary_metrics": json_clean(selected_inner.to_dict()),
        "outer_selected": outer_metrics(outer),
        "outer_original_vae": outer_metrics(standard_outer),
        "strict_instrument_and_sample_balanced_accuracy": float(
            strict_domain["balanced_accuracy_supported"].mean()
        ),
        "strict_sensor_and_sample_balanced_accuracy": float(
            sensor_domain["balanced_accuracy_supported"].mean()
        ),
        "poster_arpls_mean_balanced_accuracy": float(
            poster_arpls["balanced_accuracy_supported"].mean()
        ),
        "adequacy_decision": (
            "The converged standard-VAE backbone is scientifically adequate as "
            "a reconstruction-capable mixed-latent comparator and initialization "
            "for the next study, but it is not adequate as an instrument- and "
            "substrate-invariant representation. It remains below the frozen "
            "PCA/logistic clean benchmark, fails the registered instrument and "
            "same-master gates, and does not solve field or sensor-family shift."
        ),
        "frozen_backbone": {
            "architecture": "base_maxpool",
            "channels": [8, 16],
            "latent_dimension": 64,
            "parameter_count": 1082353,
            "preprocessing_primary": "arpls_minmax",
            "preprocessing_sensitivity": "minimal_minmax",
            "reconstruction_loss": "spectral_composite",
            "beta_target": 0.25,
            "epoch_policy": (
                "exact four 25-epoch KL cycles for epochs 1-100, then fixed "
                "beta=0.25 through epoch 500"
            ),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "batch_size": 64,
            "gradient_clip_norm": 5.0,
            "selection_checkpoint": 500,
            "reason": (
                "It was the converged inner-selected candidate with the best "
                "registered preservation/chemical utility. Architecture, loss, "
                "latent-width, and beta ablations did not produce a standard "
                "mixed latent that passed every gate."
            ),
        },
        "next_goal_boundary": (
            "Use the frozen backbone and training policy in a separately "
            "preregistered structured/disentangled-latent study. Do not reopen "
            "preprocessing, backbone, epoch count, loss, latent width, or beta "
            "using locked outer outcomes."
        ),
        "claim_limits": [
            "This is still a standard mixed-latent VAE, not evidence of disentanglement.",
            "Previously observed outer cohorts are confirmatory, not human-blind.",
            "The poster split is descriptive and lacks independent preparation IDs.",
            "Synthetic corruption robustness does not establish universal real-field denoising.",
            "Unsupported unseen analyte classes remain excluded from supported-class metrics and are retained in failure tables.",
            "Only the 400–1800 cm-1 common axis and frozen preprocessing-v2 population are supported.",
        ],
        "frozen_comparator_context": json.loads(
            (baseline_bundle / "final_decisions.json").read_text()
        ),
    }
    write_json(output_dir / "final_decisions.json", result)
    return result


def documentation(output_dir: Path, decision: dict[str, Any]) -> None:
    strict = decision["outer_selected"]["strict_core__arpls_minmax"]
    quality = decision["outer_selected"]["quality_pass__arpls_minmax"]
    stress = decision["outer_selected"][
        "field_quality_stress__arpls_minmax"
    ]
    old = decision["outer_original_vae"]["strict_core__arpls_minmax"]
    readme = f"""# SERS VAE adequacy v1

This immutable bundle determines whether the original NATO SERS standard-VAE
failures were caused by undertraining, backbone/loss/capacity choices, or domain
shift. Selection used only master-sample-grouped nested inner validation.
Outer, field-stress, instrument/sensor, and poster outcomes were locked until
the configuration was frozen.

## Outcome

The original 100-epoch cap was premature. Constant-LR training converged at
epoch 500 and raised strict-core arPLS reconstruction correlation from
`{old['reconstruction_median_row_correlation']:.6f}` to
`{strict['reconstruction_median_row_correlation']:.6f}` and repeatable-peak
recall from `{old['repeatable_peak_recall']:.6f}` to
`{strict['repeatable_peak_recall']:.6f}`. It did not materially raise strict
chemical classification.

- selected: `{decision['selected_configuration']}`;
- parameters: `{decision['parameter_count']:,}`;
- inner gates: `{decision['inner_gate_count']}/{decision['inner_gate_total']}`;
- failed gates: `{', '.join(decision['failed_inner_gates'])}`;
- strict-core arPLS balanced accuracy: `{strict['balanced_accuracy_supported']:.6f}`;
- quality-pass arPLS balanced accuracy: `{quality['balanced_accuracy_supported']:.6f}`;
- field-stress arPLS balanced accuracy: `{stress['balanced_accuracy_supported']:.6f}`;
- strict leave-instrument-and-sample accuracy: `{decision['strict_instrument_and_sample_balanced_accuracy']:.6f}`;
- strict leave-sensor-family-and-sample accuracy: `{decision['strict_sensor_and_sample_balanced_accuracy']:.6f}`;
- descriptive poster arPLS accuracy: `{decision['poster_arpls_mean_balanced_accuracy']:.6f}`.

The experiment executed 512 distinct 500-epoch training runs (256,000
optimizer epochs): 260 grouped-inner selection runs and 252 locked
confirmatory runs. See `compute_accounting.json` and
`parameter_and_compute_accounting.csv`.

## Decision

{decision['adequacy_decision']}

The frozen backbone is the starting point for a separately registered
structured/disentangled-VAE goal. No chemical/nuisance latent partitioning was
performed here.

See `DECISION_REGISTRY.md`, `final_decisions.json`,
`failure_attribution.json`, the comparator/uncertainty tables, and `figures/`.
"""
    registry = f"""# SERS VAE adequacy decision registry

1. **Leakage control:** all epochs, architecture, loss, latent-width, and beta decisions used grouped inner NATO data only.
2. **Exact continuation:** the reproduced first 100 epochs agree with the original trajectories within the registered `1e-12` tolerance.
3. **Convergence:** epoch 300 remained insufficient; constant learning rate met the registered convergence definition at epoch 500.
4. **Architecture:** residual/multiscale and single-pool candidates did not give converged, consistent gains; the base two-pool backbone remains frozen.
5. **Loss:** the peak/multiscale objective improved correlation but not repeatable-peak recall; spectral-composite remains frozen.
6. **Latent width:** z32 and z128 did not outperform converged z64 consistently; z64 remains frozen.
7. **KL strength:** beta 0.25 best preserved chemistry and peaks, while beta 4 reduced instrument leakage at unacceptable spectral/chemical cost.
8. **Preprocessing:** arPLS+min-max remains the primary separability view; minimal+min-max remains mandatory because it preserves spectra and peaks better.
9. **Outer confirmation:** 500 epochs improved quality and field-stress performance over the original VAE but did not beat frozen PCA/logistic on ordinary grouped folds.
10. **Domain confirmation:** unseen-instrument transfer improved, but unseen sensor-family, field-stress, and same-master invariance remain unresolved.
11. **Poster:** minimal-view transfer improved; arPLS transfer fell, especially for held-out Ag and PICO, demonstrating a preservation/invariance trade-off.
12. **Adequacy:** retain the selected model as a converged mixed-latent comparator/backbone, not as an invariant or disentangled representation.
13. **Next boundary:** the next goal may partition chemical and nuisance latents, but must not reopen this frozen backbone using locked outcomes.

Failed inner gates: `{', '.join(decision['failed_inner_gates'])}`.
"""
    (output_dir / "README.md").write_text(readme)
    (output_dir / "DECISION_REGISTRY.md").write_text(registry)
    commands = """#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python scripts/run_sers_vae_adequacy_selection.py --device cuda
.venv/bin/python scripts/run_sers_vae_adequacy_ablation.py --device cuda
.venv/bin/python scripts/run_sers_vae_adequacy_final.py --stage all --training-device cuda --evaluation-device cpu
.venv/bin/python scripts/finalize_sers_vae_adequacy.py
.venv/bin/python scripts/validate_sers_vae_adequacy.py
.venv/bin/python scripts/run_sers_vae_adequacy_selection.py --device cuda --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/run_sers_vae_adequacy_ablation.py --device cuda --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/run_sers_vae_adequacy_final.py --stage all --training-device cuda --evaluation-device cpu --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/finalize_sers_vae_adequacy.py --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/compare_sers_vae_adequacy_rebuild.py
.venv/bin/python scripts/validate_sers_vae_adequacy.py --require-clean-rebuild
"""
    path = output_dir / "reproduction_commands.sh"
    path.write_text(commands)
    path.chmod(0o755)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a compact deterministic Markdown table."""
    rendered = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def results_report(
    output_dir: Path,
    standard_bundle: Path,
    decision: dict[str, Any],
) -> None:
    """Write the full results interpretation from canonical bundle tables."""
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    compute = json.loads((output_dir / "compute_accounting.json").read_text())
    audit = json.loads((output_dir / "existing_run_audit.json").read_text())
    reproduction = json.loads(
        (output_dir / "first_100_reproduction.json").read_text()
    )
    stage1 = pd.read_csv(output_dir / "stage_summaries.csv")
    architecture = pd.read_csv(
        output_dir / "stage_2_architecture_summary.csv"
    )
    loss = pd.read_csv(output_dir / "stage_2_loss_summary.csv")
    latent = pd.read_csv(output_dir / "stage_2_latent_summary.csv")
    beta = pd.read_csv(output_dir / "stage_2_beta_summary.csv")
    quality = pd.read_csv(
        output_dir / "stage_2_quality_confirmation_summary.csv"
    )
    minimal = pd.read_csv(output_dir / "minimal_sensitivity_metrics.csv")
    comparator = pd.read_csv(output_dir / "comparator_summary.csv")
    uncertainty = pd.read_csv(output_dir / "uncertainty_summary.csv")
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    outer["cohort"] = cohort_from_scenario(outer)
    old_outer = pd.read_csv(standard_bundle / "outer_fold_metrics.csv")
    old_outer["cohort"] = cohort_from_scenario(old_outer)
    corruption = pd.read_csv(output_dir / "corruption_metrics.csv")
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    geometry = pd.read_csv(output_dir / "same_master_geometry_summary.csv")
    per_class = pd.read_csv(output_dir / "per_class_metrics.csv")
    per_instrument = pd.read_csv(
        output_dir / "per_instrument_failures.csv"
    )

    def mean_value(
        frame: pd.DataFrame, column: str, **filters: Any
    ) -> float:
        part = frame
        for key, value in filters.items():
            part = part[part[key].eq(value)]
        return float(part[column].mean())

    def f6(value: Any) -> str:
        numeric = float(value)
        return "NA" if not np.isfinite(numeric) else f"{numeric:.6f}"

    convergence_rows: list[list[Any]] = []
    for policy in ("constant_lr", "step_lr_300"):
        for checkpoint in (100, 300, 400, 500):
            row = stage1[
                stage1["optimizer_policy"].eq(policy)
                & stage1["checkpoint_epoch"].eq(checkpoint)
            ]
            if row.empty:
                continue
            record = row.iloc[0]
            convergence_rows.append(
                [
                    policy,
                    checkpoint,
                    f6(record["latent_target_balanced_accuracy"]),
                    f6(
                        record[
                            "clean_reconstruction_median_row_correlation"
                        ]
                    ),
                    f6(record["clean_repeatable_peak_recall"]),
                    (
                        "not assessed"
                        if pd.isna(record["converged"])
                        else str(bool(record["converged"]))
                    ),
                    (
                        "NA"
                        if pd.isna(
                            record["median_relative_improvement_50"]
                        )
                        else f"{100 * float(record['median_relative_improvement_50']):.3f}%"
                    ),
                    (
                        "NA"
                        if pd.isna(
                            record[
                                "folds_improving_at_least_1_percent"
                            ]
                        )
                        else f"{100 * float(record['folds_improving_at_least_1_percent']):.1f}%"
                    ),
                ]
            )

    def ablation_rows(
        frame: pd.DataFrame, label_column: str
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for record in frame.sort_values(label_column).itertuples(index=False):
            rows.append(
                [
                    getattr(record, label_column),
                    f"{int(record.parameter_count):,}",
                    str(bool(record.converged)),
                    f"{int(record.gate_count)}/9",
                    f6(record.latent_target_balanced_accuracy),
                    f6(record.clean_reconstruction_median_row_correlation),
                    f6(record.clean_repeatable_peak_recall),
                    f6(record.latent_instrument_probe_increment),
                    f6(record.latent_same_master_cross_instrument_distance),
                ]
            )
        return rows

    outer_rows: list[list[Any]] = []
    for cohort in (
        "strict_core",
        "quality_pass",
        "field_quality_stress",
    ):
        for representation in ("arpls_minmax", "minimal_minmax"):
            new = outer[
                outer["cohort"].eq(cohort)
                & outer["representation"].eq(representation)
            ]
            old = old_outer[
                old_outer["cohort"].eq(cohort)
                & old_outer["representation"].eq(representation)
            ]
            outer_rows.append(
                [
                    cohort,
                    representation,
                    f6(new["balanced_accuracy_supported"].mean()),
                    f6(old["balanced_accuracy_supported"].mean()),
                    f6(new["macro_f1_supported"].mean()),
                    f6(
                        new[
                            "reconstruction_median_row_correlation"
                        ].mean()
                    ),
                    f6(
                        old[
                            "reconstruction_median_row_correlation"
                        ].mean()
                    ),
                    f6(new["repeatable_peak_recall"].mean()),
                    f6(old["repeatable_peak_recall"].mean()),
                ]
            )

    comparator_rows: list[list[Any]] = []
    for cohort in (
        "strict_core",
        "quality_pass",
        "field_quality_stress",
    ):
        for model in (
            "PCA-logistic",
            "AE",
            "DAE",
            "Siamese",
            "VAE-100 β=1",
            "VAE-500 β=0.25",
        ):
            representation = (
                "derivative_1" if model == "Siamese" else "arpls_minmax"
            )
            part = comparator[
                comparator["evaluation"].eq("outer")
                & comparator["subset"].eq(cohort)
                & comparator["representation"].eq(representation)
                & comparator["display_model"].eq(model)
                & comparator["metric"].eq("balanced_accuracy_supported")
            ]
            if not part.empty:
                comparator_rows.append(
                    [cohort, model, f6(part["mean"].iloc[0])]
                )

    ci_rows: list[list[Any]] = []
    for cohort in (
        "strict_core",
        "quality_pass",
        "field_quality_stress",
    ):
        for representation in ("arpls_minmax", "minimal_minmax"):
            part = uncertainty[
                uncertainty["evaluation"].eq("outer")
                & uncertainty["subset"].eq(cohort)
                & uncertainty["model"].eq("VAE-500 β=0.25")
                & uncertainty["representation"].eq(representation)
            ]
            if not part.empty:
                row = part.iloc[0]
                ci_rows.append(
                    [
                        cohort,
                        representation,
                        f6(row["mean_balanced_accuracy"]),
                        f"±{float(row['ci95_half_width']):.6f}",
                        int(row["n_units"]),
                    ]
                )

    corruption_outer = corruption[corruption["stage"].eq("outer")].copy()
    corruption_outer["cohort"] = cohort_from_scenario(corruption_outer)
    corruption_rows: list[list[Any]] = []
    for representation in ("arpls_minmax", "minimal_minmax"):
        for severity in (0.5, 1.0, 1.5):
            part = corruption_outer[
                corruption_outer["cohort"].eq("strict_core")
                & corruption_outer["representation"].eq(representation)
                & corruption_outer["severity"].eq(severity)
            ]
            corruption_rows.append(
                [
                    representation,
                    severity,
                    f6(part["balanced_accuracy_supported"].mean()),
                    f6(part["prediction_agreement"].mean()),
                    f6(part["latent_cosine_drift"].mean()),
                    f6(
                        part[
                            "reconstruction_median_row_correlation"
                        ].mean()
                    ),
                    f6(part["repeatable_peak_recall"].mean()),
                ]
            )
    composite_rows: list[list[Any]] = []
    for representation in ("arpls_minmax", "minimal_minmax"):
        for model in ("VAE-100 β=1", "VAE-500 β=0.25"):
            values: dict[str, float] = {}
            for metric in (
                "balanced_accuracy_supported",
                "prediction_agreement",
                "latent_cosine_drift",
                "reconstruction_mse",
            ):
                part = comparator[
                    comparator["evaluation"].eq("corruption")
                    & comparator["subset"].eq(
                        "strict_core__composite__severity1"
                    )
                    & comparator["representation"].eq(representation)
                    & comparator["display_model"].eq(model)
                    & comparator["metric"].eq(metric)
                ]
                values[metric] = (
                    float(part["mean"].iloc[0]) if not part.empty else np.nan
                )
            composite_rows.append(
                [
                    representation,
                    model,
                    f6(values["balanced_accuracy_supported"]),
                    f6(values["prediction_agreement"]),
                    f6(values["latent_cosine_drift"]),
                    f6(values["reconstruction_mse"]),
                ]
            )

    geometry_rows: list[list[Any]] = []
    for representation in (
        "arpls_minmax",
        "minimal_minmax",
        "derivative_1",
    ):
        for model in (
            "AE",
            "DAE",
            "Siamese",
            "VAE-100 β=1",
            "VAE-500 β=0.25",
        ):
            part = geometry[
                geometry["subset"].eq("strict_core")
                & geometry["representation"].eq(representation)
                & geometry["display_model"].eq(model)
            ]
            if part.empty:
                continue
            row = part.iloc[0]
            geometry_rows.append(
                [
                    representation,
                    model,
                    f6(row["raw_same_master_cross_instrument_distance"]),
                    f6(
                        row[
                            "latent_same_master_cross_instrument_distance"
                        ]
                    ),
                    f6(row["same_master_distance_delta"]),
                    f6(
                        row["latent_cross_instrument_separation_margin"]
                    ),
                ]
            )

    domain_rows: list[list[Any]] = []
    for subset in ("strict_core", "quality_pass"):
        for protocol in ("domain_and_sample", "domain_only"):
            for domain_type in ("instrument", "sensor_family"):
                new = mean_value(
                    domain,
                    "balanced_accuracy_supported",
                    evaluation_subset=subset,
                    domain_protocol=protocol,
                    domain_type=domain_type,
                )
                old = comparator[
                    comparator["evaluation"].eq("domain_transfer")
                    & comparator["subset"].eq(
                        f"{subset}__{protocol}__{domain_type}"
                    )
                    & comparator["display_model"].eq("VAE-100 β=1")
                    & comparator["metric"].eq(
                        "balanced_accuracy_supported"
                    )
                ]["mean"]
                domain_rows.append(
                    [
                        subset,
                        protocol,
                        domain_type,
                        f6(new),
                        f6(old.iloc[0]),
                        f"{new - float(old.iloc[0]):+.6f}",
                    ]
                )

    heldout_rows: list[list[Any]] = []
    heldout = domain[domain["domain_protocol"].eq("domain_and_sample")]
    heldout_grouped = (
        heldout.groupby(
            ["evaluation_subset", "domain_type", "heldout_domain"],
            dropna=False,
        )["balanced_accuracy_supported"]
        .mean()
        .reset_index()
    )
    for record in heldout_grouped.itertuples(index=False):
        heldout_rows.append(
            [
                record.evaluation_subset,
                record.domain_type,
                record.heldout_domain,
                f6(record.balanced_accuracy_supported),
            ]
        )

    poster_rows: list[list[Any]] = []
    poster_grouped = (
        poster.groupby(
            ["representation", "heldout_substrate_family"], dropna=False
        )[
            [
                "balanced_accuracy_supported",
                "reconstruction_median_row_correlation",
                "repeatable_peak_recall",
            ]
        ]
        .mean()
        .reset_index()
    )
    for record in poster_grouped.itertuples(index=False):
        poster_rows.append(
            [
                record.representation,
                record.heldout_substrate_family,
                f6(record.balanced_accuracy_supported),
                f6(record.reconstruction_median_row_correlation),
                f6(record.repeatable_peak_recall),
            ]
        )

    class_rows: list[list[Any]] = []
    classes = per_class[per_class["stage"].eq("outer")].copy()
    classes["cohort"] = cohort_from_scenario(classes)
    classes = classes[classes["representation"].eq("arpls_minmax")]
    class_grouped = (
        classes.groupby(["cohort", "class_label"])
        .agg(
            true_positive=("true_positive", "sum"),
            support=("support", "sum"),
            predicted=("predicted_count", "sum"),
        )
        .reset_index()
    )
    for record in class_grouped.itertuples(index=False):
        class_rows.append(
            [
                record.cohort,
                record.class_label,
                int(record.support),
                f6(record.true_positive / record.support),
                (
                    f6(record.true_positive / record.predicted)
                    if record.predicted
                    else "NA"
                ),
            ]
        )

    instrument_rows: list[list[Any]] = []
    instruments = per_instrument[
        per_instrument["stage"].eq("outer")
    ].copy()
    instruments["cohort"] = cohort_from_scenario(instruments)
    instruments = instruments[
        instruments["representation"].eq("arpls_minmax")
    ]
    instrument_grouped = (
        instruments.groupby(["cohort", "instrument"])
        .agg(
            correct=("correct_supported", "sum"),
            support=("supported_observations", "sum"),
        )
        .reset_index()
    )
    for record in instrument_grouped.itertuples(index=False):
        instrument_rows.append(
            [
                record.cohort,
                record.instrument,
                int(record.support),
                f6(record.correct / record.support),
            ]
        )

    minimal_rows: list[list[Any]] = []
    minimal_grouped = (
        minimal.groupby(["subset"])[
            [
                "latent_target_balanced_accuracy",
                "clean_reconstruction_median_row_correlation",
                "clean_repeatable_peak_recall",
                "latent_instrument_probe_increment",
                "latent_same_master_cross_instrument_distance",
                "vae_kl_unnormalized_per_observation",
                "vae_active_units_var_mu_gt_0_01",
            ]
        ]
        .mean()
        .reset_index()
    )
    for record in minimal_grouped.itertuples(index=False):
        minimal_rows.append(
            [
                record.subset,
                f6(record.latent_target_balanced_accuracy),
                f6(record.clean_reconstruction_median_row_correlation),
                f6(record.clean_repeatable_peak_recall),
                f6(record.latent_instrument_probe_increment),
                f6(record.latent_same_master_cross_instrument_distance),
                f6(record.vae_kl_unnormalized_per_observation),
                f6(record.vae_active_units_var_mu_gt_0_01),
            ]
        )

    quality_rows = [
        [
            record.identifier,
            f6(record.strict_latent_target_balanced_accuracy),
            f6(record.quality_latent_target_balanced_accuracy),
            f"{float(record.quality_target_delta):+.6f}",
            f6(record.selection_utility),
        ]
        for record in quality.itertuples(index=False)
    ]

    old_audit = audit["groups"]["strict_core__arpls_minmax"]
    selected_inner = decision["inner_primary_metrics"]
    report = f"""# NATO SERS standard-VAE training and architecture adequacy — full results

Protocol: `sers-vae-adequacy-v1`  
Status: complete, leakage-controlled selection followed by locked confirmation  
Selected configuration: `{decision['selected_configuration']}`

## Executive answer

The idea was partly right, in a scientifically useful way:

1. **The original VAE was undertrained for reconstruction.** At 100 epochs,
   validation loss was still improving in every evaluable strict-core run;
   {old_audit['best_epoch_at_least_95_count']}/20 runs had their best epoch at
   or after 95. Extending the exact optimization path to 500 epochs materially
   improved correlation and repeatable-peak recovery.
2. **Undertraining was not the main classification or invariance failure.**
   Strict-core arPLS balanced accuracy changed only from
   `{mean_value(old_outer, 'balanced_accuracy_supported', cohort='strict_core', representation='arpls_minmax'):.6f}`
   to
   `{mean_value(outer, 'balanced_accuracy_supported', cohort='strict_core', representation='arpls_minmax'):.6f}`.
   The converged latent still failed the target-adjusted instrument-predictability
   and same-master cross-instrument-distance gates.
3. **The tested ordinary mixed-latent VAE cannot simultaneously preserve narrow
   chemistry-bearing structure and remove acquisition nuisance.** Lower beta
   preserves spectra and peaks but retains more instrument structure; higher
   beta suppresses some instrument information but damages chemistry and
   spectral fidelity.
4. **The frozen standard VAE is adequate as a converged reconstruction-capable
   comparator and initialization, not as an instrument/substrate-invariant
   representation.** This is the precise justification for a separately
   preregistered structured/disentangled-latent study.

No chemical/nuisance latent partitioning, adversarial loss, conditioned
decoder, or supervised contrastive loss was tested here.

## What was held fixed

- Population: 598 strict-core spectra, including the 500 quality-pass spectra
  and retaining the 98 field-quality-stress spectra as a locked stress cohort.
- Axis: 400–1800 cm⁻¹ inclusive, 1 cm⁻¹ step, 1,401 points.
- Grouping: `master_sample_id`; related observations could not cross a
  selection split.
- Primary view: `arpls_minmax`.
- Mandatory sensitivity view: `minimal_minmax`.
- Locked comparators: PCA/logistic, Siamese, AE, DAE, and the original
  100-epoch beta-1 VAE.
- Selection boundary: architecture, epoch, loss, latent width, beta, and
  optimizer decisions used only 20 grouped inner folds. Outer folds, field
  stress, held-out domains, and poster results were unavailable to selection.

The outer data had been seen in earlier projects, so the final stage is
confirmatory but not a human-blind external test.

## What was actually trained

- {compute['distinct_inner_training_runs']} distinct grouped-inner selection
  runs.
- {compute['distinct_confirmatory_training_runs']} distinct locked
  confirmatory runs.
- {compute['total_distinct_training_runs']} total model fits, each executing
  500 optimizer epochs.
- {compute['total_training_epochs']:,} optimizer epochs in total.
- Parameter range across ablations:
  {compute['parameter_range'][0]:,}–{compute['parameter_range'][1]:,}.
- {compute['spectrum_level_prediction_rows']:,} spectrum-level prediction rows
  and {compute['spectrum_level_reconstruction_rows']:,} reconstruction rows.
- 252 final checkpoints, 260 selection caches with model/optimizer states, 282
  embeddings, and 282 reconstruction arrays.

The chosen model has {decision['parameter_count']:,} parameters but only about
354–363 training spectra per strict inner fold. This unfavorable
data-to-parameter ratio was treated as a reason to keep the ablation bounded,
not as permission for an unrestricted architecture search.

## Frozen model and training policy

- 1-D convolutional encoder with channels 8→16 and two max-pooling stages.
- Mirrored decoder; no encoder–decoder skip connections.
- 64-dimensional Gaussian mixed latent.
- Spectral-composite reconstruction loss: Smooth L1 + 0.1 spectral angle +
  0.1 first-derivative loss.
- Adam, learning rate 0.001, weight decay 0.00001, batch size 64, gradient clip
  norm 5.
- Four fixed 25-epoch KL cycles during epochs 1–100. After epoch 100, beta is
  held at 0.25 through epoch 500.
- Canonical inference on CPU after deterministic CUDA training.

## Metric glossary: what the numbers mean

- **Balanced accuracy (BA):** mean class recall over supported analyte classes;
  each class has equal weight despite unequal spectrum counts. Higher is
  better.
- **Macro F1:** unweighted mean class F1 over supported classes; it penalizes
  both missed examples and false positives. Higher is better.
- **Median row correlation:** median Pearson correlation between each input
  spectrum and its reconstruction. It tests shape preservation, not absolute
  amplitude. Higher is better.
- **Repeatable-peak recall:** fraction of prominent reference peaks that recur
  across instruments for the same master sample and are reconstructed within
  ±5 cm⁻¹. Higher is better.
- **Instrument probe increment:** target-adjusted instrument-classification
  score above a target-only null model. Lower means less instrument information
  remains after accounting for analyte. Zero is ideal.
- **Same-master distance:** mean correlation distance between spectra of the
  same `master_sample_id` measured on different instruments. Lower is better.
- **Cross-instrument separation margin:** different-analyte distance minus
  same-master distance. Positive and larger is better.
- **KL per observation:** unnormalized divergence of the approximate posterior
  from the prior. Near zero can indicate posterior collapse; very high values
  indicate weak regularization.
- **Active units:** latent dimensions whose posterior-mean variance exceeds
  0.01. It measures use, not disentanglement.
- **Prediction agreement:** fraction whose predicted class is unchanged after
  controlled corruption. Higher is better.
- **Latent cosine drift:** change in latent direction after corruption. Lower
  is better.

Correlation and peak metrics do not prove that a VAE has removed noise:
identity-like reconstruction can score highly. Instrument probes,
same-master geometry, corruptions, and held-out domains are necessary
complements.

## Stage 0 — why 100 epochs was suspect

For strict-core arPLS, the original runs had median best epoch
{old_audit['best_epoch_median']:.0f}, with
{old_audit['best_epoch_at_100_count']}/20 exactly at epoch 100 and
{old_audit['best_epoch_at_least_95_count']}/20 at or beyond epoch 95.
Validation loss improved from epoch 90 to 100 in
{100 * old_audit['epoch_90_to_100_improved_fraction']:.0f}% of the
{old_audit['epoch_90_to_100_run_count']} evaluable runs, by a median
{abs(old_audit['epoch_90_to_100_validation_loss_percent_change_median']):.3f}%.
The fourth KL cycle reached beta 1 only at epoch
{audit['cycle_interaction']['fourth_cycle_reaches_beta_one_epoch']}, leaving
{audit['cycle_interaction']['beta_one_epochs_before_cap']} beta-1 epochs before
the cap—less than the original early-stopping patience of
{audit['cycle_interaction']['early_stopping_patience']}.

The reproduced first 100 epochs matched the original histories to a maximum
absolute difference of `{reproduction['maximum_absolute_difference']:.3e}`,
inside the preregistered `1e-12` tolerance. The extension therefore tested
training duration rather than silently changing the original trajectory.

## Stage 1 — convergence isolation

Convergence required both: median validation-ELBO improvement over the final
50 epochs below 0.5%, and fewer than 25% of grouped folds improving by at
least 1%.

{markdown_table(
    [
        'policy',
        'epoch',
        'inner BA',
        'correlation',
        'peak recall',
        'converged',
        'median final-50 improvement',
        'folds improving ≥1%',
    ],
    convergence_rows,
)}

At epoch 300, constant learning rate was not converged: median improvement was
1.099% and 55% of folds improved by at least 1%. At epoch 500 it was converged:
0.254% median improvement and 10% of folds above 1%. The step-down policy did
not improve the registered scientific utility, so constant 0.001 was frozen.

This establishes that 100 epochs was a real spectral-fidelity problem.
It does not establish that longer training creates invariance.

## Stage 2A — architecture

{markdown_table(
    [
        'architecture',
        'parameters',
        'converged',
        'gates',
        'inner BA',
        'correlation',
        'peak recall',
        'instrument probe',
        'same-master distance',
    ],
    ablation_rows(architecture, 'architecture'),
)}

The residual/multiscale model did not converge and reduced BA and peak recall.
The one-pool model modestly raised peak recall but did not converge, did not
improve BA, and retained more instrument information. The original two-pool
backbone was therefore not supported as the primary cause of failure.

## Stage 2B — reconstruction loss

{markdown_table(
    [
        'loss',
        'parameters',
        'converged',
        'gates',
        'inner BA',
        'correlation',
        'peak recall',
        'instrument probe',
        'same-master distance',
    ],
    ablation_rows(loss, 'reconstruction_loss'),
)}

The peak/multiscale loss raised correlation from 0.927692 to 0.932659 but peak
recall was essentially unchanged (0.437320 versus 0.437414) and BA fell
slightly. Extra derivative and multiscale terms therefore did not solve the
peak or nuisance problem.

## Stage 2C — latent width

{markdown_table(
    [
        'latent dimensions',
        'parameters',
        'converged',
        'gates',
        'inner BA',
        'correlation',
        'peak recall',
        'instrument probe',
        'same-master distance',
    ],
    ablation_rows(latent, 'latent_dimension'),
)}

The 32-dimensional latent had eight gates but failed convergence and lost
chemical accuracy. The 128-dimensional latent doubled parameters, failed
convergence, and improved neither chemistry nor preservation. Width alone did
not filter nuisance; z64 remained the only converged option.

## Stage 2D — KL strength and the key trade-off

{markdown_table(
    [
        'beta',
        'parameters',
        'converged',
        'gates',
        'inner BA',
        'correlation',
        'peak recall',
        'instrument probe',
        'same-master distance',
    ],
    ablation_rows(beta, 'beta_target'),
)}

- Beta 0.25 best preserved shape and repeatable peaks and used about 33.35
  active dimensions, but retained the most nuisance structure.
- Beta 1 had the best strict inner BA, but lower spectral and peak fidelity.
- Beta 4 reduced the instrument probe and same-master distance, but damaged BA,
  correlation, and peaks.

This is the central result: a single unsupervised latent is being asked both
to reconstruct nuisance-rich spectra and to discard that nuisance. Stronger
prior pressure does not tell the model which variation is chemical.

The top two beta candidates both passed the quality-sensitivity rule:

{markdown_table(
    ['candidate', 'strict BA', 'quality BA', 'quality−strict', 'utility'],
    quality_rows,
)}

Beta 0.25 was selected by the preregistered multi-objective utility, not by
outer outcomes. Although beta 1 had higher strict inner BA, beta 0.25 provided
the stronger registered preservation/chemistry compromise and slightly higher
quality BA.

## Eligibility-gate result

The selected strict-core arPLS model passed 7/9 gates. It passed clean
correlation, repeatable peaks, chemical probe, active units, KL dimensions, KL
range, and finite-output checks. It failed:

1. **Instrument probe:** `{selected_inner['latent_instrument_probe_increment']:.6f}`.
   Too much target-adjusted instrument information remained.
2. **Same-master distance:** `{selected_inner['latent_same_master_cross_instrument_distance']:.6f}`,
   versus raw `{selected_inner['raw_same_master_cross_instrument_distance']:.6f}`.
   Encoding increased rather than decreased the cross-instrument distance of
   replicate master samples by `{selected_inner['same_master_distance_delta']:+.6f}`.

The model was therefore selected as the rigorously defined least-failing,
converged backbone—not declared fully adequate.

## Mandatory preprocessing sensitivity

`minimal_minmax` preserves the instrument-delivered shape before common-axis
scaling; `arpls_minmax` removes more baseline and is the primary separability
view. The two views answer different questions and neither replaces the other.

{markdown_table(
    [
        'subset',
        'inner BA',
        'correlation',
        'peak recall',
        'instrument probe',
        'same-master distance',
        'KL/observation',
        'active units',
    ],
    minimal_rows,
)}

Minimal preprocessing improved correlation and peak recovery substantially,
but its instrument predictability and same-master geometry were worse. This
means scaling every spectrum to the same 0–1 range does not itself make
spectra instrument-invariant; baseline/shape differences remain encoded.
arPLS remains primary for classification, while minimal remains mandatory to
detect peak destruction or preprocessing-dependent conclusions.

## Locked grouped-outer confirmation

Each value below averages five grouped outer folds and three registered neural
seeds. “Old” is the original 100-epoch beta-1 VAE.

{markdown_table(
    [
        'cohort',
        'view',
        'new BA',
        'old BA',
        'new macro F1',
        'new correlation',
        'old correlation',
        'new peak recall',
        'old peak recall',
    ],
    outer_rows,
)}

The strict arPLS changes from 100 to 500 epochs were:

- BA: +{mean_value(outer, 'balanced_accuracy_supported', cohort='strict_core', representation='arpls_minmax') - mean_value(old_outer, 'balanced_accuracy_supported', cohort='strict_core', representation='arpls_minmax'):.6f}
  — effectively unchanged.
- Correlation: +{mean_value(outer, 'reconstruction_median_row_correlation', cohort='strict_core', representation='arpls_minmax') - mean_value(old_outer, 'reconstruction_median_row_correlation', cohort='strict_core', representation='arpls_minmax'):.6f}.
- Repeatable-peak recall: +{mean_value(outer, 'repeatable_peak_recall', cohort='strict_core', representation='arpls_minmax') - mean_value(old_outer, 'repeatable_peak_recall', cohort='strict_core', representation='arpls_minmax'):.6f}.

Quality-pass and field-stress BA improved, but field BA remained only 0.368545
on arPLS and 0.396561 on minimal. The model therefore became a better
reconstructor and somewhat better stress classifier without becoming a
strong field model.

The 95% t-interval summaries use the five outer folds as independent units:

{markdown_table(
    ['cohort', 'view', 'mean BA', '95% half-width', 'folds'],
    ci_rows,
)}

These intervals are descriptive across only five folds and should not be read
as precise population intervals.

## Comparison with frozen model families

Grouped-outer balanced accuracy:

{markdown_table(['cohort', 'model', 'BA'], comparator_rows)}

The new VAE did not beat PCA/logistic on strict, quality, or field cohorts. It
also did not produce a general classification advantage over AE/DAE. The
correct claim is therefore “converged VAE comparator,” not “best classifier.”
The Siamese model remains a useful metric-learning control and has excellent
same-master alignment, but it is not a reconstruction or denoising model.

## Controlled-corruption behavior

The next table averages all seven registered corruptions at each severity for
the strict cohort:

{markdown_table(
    [
        'view',
        'severity',
        'BA',
        'prediction agreement',
        'latent drift',
        'correlation',
        'peak recall',
    ],
    corruption_rows,
)}

Composite-corruption severity 1, directly compared with the original VAE:

{markdown_table(
    ['view', 'model', 'BA', 'agreement', 'latent drift', 'MSE'],
    composite_rows,
)}

The converged VAE improved composite BA, agreement, and reconstruction MSE on
arPLS, but latent drift increased slightly. On minimal, the BA/agreement
change was negligible while MSE improved. Smooth baselines and scale/offset
were more damaging than Gaussian broadening, Gaussian noise, or isolated
spikes. Synthetic robustness is useful diagnostic evidence, not proof of
universal field denoising.

## Same-master cross-instrument geometry

Lower latent distance and negative delta are desirable. A positive separation
margin means different analytes remain farther apart than same-master
replicates.

{markdown_table(
    [
        'view',
        'model',
        'raw same-master',
        'latent same-master',
        'delta',
        'separation margin',
    ],
    geometry_rows,
)}

On strict arPLS, VAE-500 increased same-master distance by +0.102069; the
original VAE increased it by +0.082548. AE and DAE reduced it, while Siamese
reduced it most. The longer/lower-beta VAE therefore preserved more spectral
detail but made replicate geometry less invariant. Field-stress distances
shrank for all autoencoders, but their separation margins were negative and
field classification was poor, so that shrinkage is not useful invariance.

## Held-out instrument and sensor transfer

`domain_only` holds out a domain; `domain_and_sample` additionally prevents
master-sample overlap and is the stronger generalization test.

{markdown_table(
    ['subset', 'protocol', 'domain', 'new BA', 'old BA', 'change'],
    domain_rows,
)}

Instrument transfer improved in all four new-versus-old comparisons. Sensor
family transfer was mixed: strict domain-and-sample fell from 0.623912 to
0.584317, while strict domain-only rose from 0.385036 to 0.423433. These
aggregates also conceal severe domain-specific failures:

{markdown_table(
    ['subset', 'domain type', 'held-out domain', 'BA'],
    heldout_rows,
)}

The NRC Canadian SERS family was especially weak (~0.22–0.24 BA). Quality
domain-and-sample Mira-2 was 0.0. Some 1.0 values come from small or unusually
supported held-out partitions; unsupported analyte classes remain excluded
from supported-class BA and are retained in failure tables. `NA` means no
supported-class BA could be computed, not perfect or zero performance.

## Descriptive poster substrate transfer

The poster data lack independent preparation IDs, so these are descriptive
leave-one-substrate-family-out results, not an independent validation set.

{markdown_table(
    ['view', 'held-out substrate', 'BA', 'correlation', 'peak recall'],
    poster_rows,
)}

Mean arPLS poster BA fell from 0.927222 for VAE-100 to 0.787778 for VAE-500,
concentrated in held-out Ag and PICO. Mean minimal BA rose from 0.682778 to
0.807778. This direction reversal is another preprocessing-dependent
preservation/invariance warning, not evidence that either view universally
wins.

## Class-level behavior

Pooled spectrum counts below include repeated outer-seed predictions, so they
describe error concentration and are not independent sample counts:

{markdown_table(
    ['cohort', 'class', 'pooled support', 'recall', 'precision'],
    class_rows,
)}

In strict-core arPLS, ethyl paraoxon was the weakest class (recall 0.506173,
precision 0.482353); benzyl fentanyl was strongest (recall 0.805556). In field
stress the model overpredicted acetaminophen: recall was 0.933333 but precision
only 0.210000. Ethanol, ethyl paraoxon, and 4-ANPP recalls were all below 0.18.

## Instrument-level behavior

These pooled accuracies also repeat spectra across seeds and are diagnostic,
not independent confidence units:

{markdown_table(
    ['cohort', 'instrument', 'pooled supported rows', 'accuracy'],
    instrument_rows,
)}

Strict-core performance ranged from 0.628571 on Pendar-3 to 0.862069 on
Agilent-1. The field cohort was dominated by failures on Pendar-1, Pendar-2,
and Pendar-3; instruments with only three or six pooled supported predictions
must not be overinterpreted.

## Failure attribution

1. **Convergence — confirmed material contributor to spectral fidelity.**
   Five hundred epochs repaired correlation and peak recall, but did not
   materially change strict classification.
2. **Architecture/downsampling — not supported as the primary failure.**
   Neither the residual/multiscale nor one-pool candidate gave a converged,
   consistent gain.
3. **Reconstruction objective — peak-aware loss not a solution.** It improved
   correlation but not repeatable-peak recall.
4. **Latent capacity/KL pressure — strong trade-off confirmed.** Width changes
   did not solve it; beta moved preservation and nuisance retention in opposite
   directions.
5. **Data coverage/domain shift — dominant unresolved confirmatory failure.**
   Field stress, NRC Canadian SERS, and several held-out instruments remain
   weak.
6. **Unresolved interaction — ordinary mixed latent inadequate for the desired
   invariance.** A standard VAE has no label telling it which reconstructable
   variance is chemical and which is nuisance.

## Final decision

{decision['adequacy_decision']}

The frozen backbone and training policy may initialize the next structured
study. The next goal should partition or condition chemical and nuisance
information and test that design with the same grouped and locked boundaries.
It must not reopen preprocessing, the 8→16 two-pool backbone, z64 width,
spectral-composite loss, beta 0.25, or 500-epoch policy based on the already
observed locked outcomes.

## Claim limits

"""
    report += "\n".join(f"- {item}" for item in decision["claim_limits"])
    report += """

## Reproducibility status

The authoritative bundle and a clean rebuild from an empty output directory
are required to agree exactly for canonical tables, JSON decisions, model
tensors, optimizer tensors, embeddings, and reconstructions; floating training
histories use the preregistered `1e-12` tolerance. See
`clean_rebuild_comparison.json`, `validation_report.json`, and
`artifact_hashes.json`.
"""
    (output_dir / "RESULTS_REPORT.md").write_text(report)


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
        {"algorithm": "sha256", "files": catalog},
    )


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--standard-bundle",
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
    nato_bundle = args.nato_bundle.resolve()
    standard_bundle = args.standard_bundle.resolve()
    baseline_bundle = args.baseline_bundle.resolve()
    consolidate(output_dir)
    compute_accounting(output_dir)
    same_master_geometry(
        output_dir, nato_bundle, standard_bundle, baseline_bundle
    )
    comparator_summary(output_dir, standard_bundle, baseline_bundle)
    uncertainty(output_dir, standard_bundle)
    instrument_reconstruction_summary(output_dir, nato_bundle)
    decision = decisions(output_dir, standard_bundle, baseline_bundle)
    figure_convergence(output_dir)
    figure_ablation(output_dir)
    figure_outer(output_dir, standard_bundle, baseline_bundle)
    figure_preservation(output_dir, standard_bundle, baseline_bundle)
    figure_corruption(output_dir, standard_bundle)
    figure_domain(output_dir, standard_bundle)
    figure_poster(output_dir, standard_bundle)
    figure_instrument_spectra(output_dir, standard_bundle, nato_bundle)
    documentation(output_dir, decision)
    results_report(output_dir, standard_bundle, decision)
    hash_artifacts(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "decision": decision["adequacy_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
