#!/usr/bin/env python3
"""Export real-pair latent swaps for every locked NATO outer scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_sers_baseline_final as baseline_final
import run_sers_structured_vae_selection as selection
import sers_baseline_common as baseline
import sers_structured_vae_common as structured


REPRESENTATIONS = ("arpls_minmax", "minimal_minmax")


def scenario_from_name(name: str) -> tuple[str, str]:
    prefix = "locked_outer__"
    if not name.startswith(prefix):
        raise ValueError(f"Not a locked-outer artifact: {name}")
    body = name.removeprefix(prefix).removesuffix(".npz")
    for representation in REPRESENTATIONS:
        marker = f"__{representation}__"
        if marker in body:
            scenario = body.split(marker, maxsplit=1)[0]
            return scenario, representation
    raise ValueError(f"Representation missing from artifact name: {name}")


def training_scenario(scenario: str) -> str:
    prefix, test_subset = scenario.rsplit("__test_", maxsplit=1)
    del test_subset
    if "__train_strict_core" in prefix:
        return prefix
    if "__train_quality_pass" in prefix:
        return prefix
    raise ValueError(f"Unknown outer scenario: {scenario}")


def test_subset(scenario: str) -> str:
    return scenario.rsplit("__test_", maxsplit=1)[1]


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0:
        return np.empty(0, dtype=np.float64)
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered, axis=1
    )
    return numerator / np.maximum(denominator, 1.0e-12)


def mean_square(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return float("nan")
    return float(np.mean((left - right) ** 2))


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def export_scenario(
    output_dir: Path,
    dataset: baseline.SpectralDataset,
    embedding_path: Path,
    reconstruction_path: Path,
    registry: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    scenario, representation = scenario_from_name(embedding_path.name)
    with np.load(embedding_path, allow_pickle=False) as archive:
        embeddings = {key: archive[key] for key in archive.files}
    with np.load(reconstruction_path, allow_pickle=False) as archive:
        reconstructions = {key: archive[key] for key in archive.files}
    uids = embeddings["observation_uid"].astype(str)
    if not np.array_equal(
        uids, reconstructions["observation_uid"].astype(str)
    ):
        raise ValueError(f"Embedding/reconstruction order differs: {scenario}")

    dataset_lookup = pd.Series(
        np.arange(len(dataset.observation_uid)),
        index=dataset.observation_uid.astype(str),
    )
    if not set(uids).issubset(set(dataset_lookup.index)):
        raise ValueError(f"Unknown observation UID in scenario: {scenario}")
    rows = dataset_lookup.loc[uids].to_numpy(dtype=int)
    manifest = dataset.manifest.iloc[rows].reset_index(drop=True)
    clean_values = dataset.representations[representation][rows]
    if not np.array_equal(
        clean_values.astype(np.float32),
        reconstructions["clean"].astype(np.float32),
    ):
        raise ValueError(f"Saved clean spectra differ: {scenario}")

    checkpoint = output_dir / registry["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = selection.config_from_record(payload["metadata"]["configuration"])
    target_mapping = payload["metadata"]["target_mapping"]
    instrument_mapping = payload["metadata"]["instrument_mapping"]
    sensor_mapping = payload["metadata"]["sensor_mapping"]
    model = structured.build_model_from_state(
        clean_values.shape[1],
        config,
        len(target_mapping),
        len(instrument_mapping),
        len(sensor_mapping),
        payload["state"],
        device,
    )
    instrument_indices = (
        manifest["instrument"].astype(str).map(instrument_mapping).to_numpy()
    )
    sensor_indices = (
        manifest["sensor_family"].astype(str).map(sensor_mapping).to_numpy()
    )
    if pd.isna(instrument_indices).any() or pd.isna(sensor_indices).any():
        raise ValueError(f"Unmapped domain label in scenario: {scenario}")
    instrument_indices = instrument_indices.astype(np.int64)
    sensor_indices = sensor_indices.astype(np.int64)
    swap_outputs = {
        "chemical_mu": embeddings["chemical_mu"].astype(np.float32),
        "nuisance_mu": embeddings["nuisance_mu"].astype(np.float32),
    }
    partners, valid = selection.partner_indices(
        manifest, int(registry["run_seed"])
    )
    selected = np.flatnonzero(valid)
    if len(selected):
        swapped, partner_rows, selected_rows = (
            selection.swapped_reconstruction(
                model,
                swap_outputs,
                manifest,
                instrument_indices,
                sensor_indices,
                int(registry["run_seed"]),
                device,
            )
        )
    else:
        selected_rows = np.empty(0, dtype=int)
        partner_rows = np.empty(0, dtype=int)
        swapped = np.empty(
            (0, clean_values.shape[1]), dtype=np.float32
        )
    del model

    source_values = clean_values[selected_rows].astype(np.float32)
    partner_values = clean_values[partner_rows].astype(np.float32)
    standard_reconstruction = reconstructions["reconstructed"][
        selected_rows
    ].astype(np.float32)
    repeatable = baseline_final.repeatable_test_positions(
        clean_values, manifest, clean_values, manifest
    )
    partner_repeatable = (
        [repeatable[index] for index in partner_rows]
        if repeatable is not None
        else None
    )
    if len(selected_rows):
        spectral = baseline.aggregate_reconstruction_metrics(
            baseline.reconstruction_metrics(
                partner_values,
                swapped,
                manifest.iloc[selected_rows]["observation_uid"]
                .astype(str)
                .to_numpy(),
                partner_repeatable,
            )
        )
    else:
        spectral = {
            key: float("nan")
            for key in (
                "reconstruction_mse",
                "reconstruction_smooth_l1",
                "reconstruction_spectral_angle",
                "reconstruction_median_row_correlation",
                "reconstruction_first_derivative_mae",
                "repeatable_peak_recall",
                "median_peak_shift_cm1",
                "median_absolute_relative_peak_width_change",
                "median_absolute_peak_prominence_change",
            )
        }

    swap_dir = output_dir / "swaps"
    swap_dir.mkdir(parents=True, exist_ok=True)
    destination = swap_dir / embedding_path.name
    def text_rows(column: str, indices: np.ndarray) -> np.ndarray:
        return (
            manifest.iloc[indices][column]
            .astype(str)
            .to_numpy(dtype=str)
        )

    np.savez_compressed(
        destination,
        axis_cm1=dataset.axis_cm1.astype(np.float32),
        source_observation_uid=text_rows("observation_uid", selected_rows),
        partner_observation_uid=text_rows("observation_uid", partner_rows),
        source_master_sample_id=text_rows(
            "master_sample_id", selected_rows
        ),
        partner_master_sample_id=text_rows(
            "master_sample_id", partner_rows
        ),
        source_target_analyte=text_rows("target_analyte", selected_rows),
        partner_target_analyte=text_rows("target_analyte", partner_rows),
        source_instrument=text_rows("instrument", selected_rows),
        partner_instrument=text_rows("instrument", partner_rows),
        source_sensor_family=text_rows("sensor_family", selected_rows),
        partner_sensor_family=text_rows("sensor_family", partner_rows),
        source_clean=source_values,
        partner_clean=partner_values,
        source_standard_reconstruction=standard_reconstruction,
        source_chemical_mu=embeddings["chemical_mu"][
            selected_rows
        ].astype(np.float32),
        partner_nuisance_mu=embeddings["nuisance_mu"][
            partner_rows
        ].astype(np.float32),
        swapped_reconstruction=swapped.astype(np.float32),
    )

    same_master = np.array_equal(
        manifest.iloc[selected_rows]["master_sample_id"].astype(str).to_numpy(),
        manifest.iloc[partner_rows]["master_sample_id"].astype(str).to_numpy(),
    )
    different_instrument = bool(
        (
            manifest.iloc[selected_rows]["instrument"].astype(str).to_numpy()
            != manifest.iloc[partner_rows]["instrument"]
            .astype(str)
            .to_numpy()
        ).all()
    )
    same_target = np.array_equal(
        manifest.iloc[selected_rows]["target_analyte"].astype(str).to_numpy(),
        manifest.iloc[partner_rows]["target_analyte"].astype(str).to_numpy(),
    )
    return {
        "outer_fold": int(
            scenario.split("__", maxsplit=1)[0].removeprefix("nato_outer_o")
        ),
        "training_scenario": training_scenario(scenario),
        "scenario": scenario,
        "test_subset": test_subset(scenario),
        "representation": representation,
        "configuration": config.identifier,
        "run_seed": int(registry["run_seed"]),
        "real_pair_count": len(selected_rows),
        "same_master_invariant": same_master,
        "different_instrument_invariant": different_instrument,
        "same_target_invariant": same_target,
        "source_partner_mse": mean_square(source_values, partner_values),
        "swap_source_mse": mean_square(swapped, source_values),
        "swap_partner_mse": mean_square(swapped, partner_values),
        "standard_reconstruction_source_mse": mean_square(
            standard_reconstruction, source_values
        ),
        "source_partner_median_correlation": median_or_nan(
            row_correlation(source_values, partner_values)
        ),
        "swap_source_median_correlation": median_or_nan(
            row_correlation(swapped, source_values)
        ),
        "swap_partner_median_correlation": median_or_nan(
            row_correlation(swapped, partner_values)
        ),
        **{key: clean(value) for key, value in spectral.items()},
        "artifact": str(destination.relative_to(output_dir)),
    }


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository / "Workspace/nato_sers_field_trial/preprocessing_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace/sers_structured_vae/structured_vae_v1",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    datasets = {
        "strict_core": baseline.load_nato_dataset(args.nato_bundle),
        "quality_pass": baseline_final.load_nato_subset(
            args.nato_bundle, "quality_pass"
        ),
        "field_quality_stress": baseline_final.load_nato_subset(
            args.nato_bundle, "field_quality_stress"
        ),
    }
    registry_records = json.loads(
        (output_dir / "locked_outer_run_registry.json").read_text()
    )
    registry = {
        (record["training_scenario"], record["representation"]): record
        for record in registry_records
    }
    rows: list[dict[str, Any]] = []
    embedding_paths = sorted(
        (output_dir / "embeddings").glob("locked_outer__*.npz")
    )
    if len(embedding_paths) != 30:
        raise ValueError(
            f"Expected 30 locked-outer embeddings, found {len(embedding_paths)}"
        )
    for embedding_path in embedding_paths:
        scenario, representation = scenario_from_name(embedding_path.name)
        reconstruction_path = (
            output_dir / "reconstructions" / embedding_path.name
        )
        key = (training_scenario(scenario), representation)
        if key not in registry:
            raise ValueError(f"No locked checkpoint for {key}")
        rows.append(
            export_scenario(
                output_dir,
                datasets[test_subset(scenario)],
                embedding_path,
                reconstruction_path,
                registry[key],
                device,
            )
        )
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "representation": representation,
                    "pairs": rows[-1]["real_pair_count"],
                }
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values(
        ["test_subset", "outer_fold", "representation"]
    )
    frame.to_csv(output_dir / "locked_outer_swap_metrics.csv", index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "scenario_count": len(frame),
                "swap_artifact_count": len(
                    list((output_dir / "swaps").glob("*.npz"))
                ),
                "real_pair_count": int(frame["real_pair_count"].sum()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
