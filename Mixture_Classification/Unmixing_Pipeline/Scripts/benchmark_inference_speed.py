from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_deep_binary_coefficient_regressor import (
    build_synthetic_dataset,
    normalize_dictionary,
    normalize_spectrum,
    set_seed,
    split_synthetic_dataset,
    support_from_top2,
)
from run_deep_similarity_supervision import train_model_with_similarity_supervision
from unmixing_common import (
    build_compound_atom_sets,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    constant_baseline_atom,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
    search_best_pair_with_atom_sets,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "Results" / "inference_speed_benchmark"
MODE = "baseline_corrected"
N_EXTRA_REPS = 9
N_REPEATS = 5


def perf_counter_seconds() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def build_real_records():
    ref_df, wav_axis = load_reference()
    existing_records = load_existing_real_records(MODE)
    pt2_records = load_pt2_mixture_records(wav_axis, MODE)
    return ref_df, wav_axis, existing_records + pt2_records


def build_nnls_state(ref_df: pd.DataFrame, wav_axis: np.ndarray):
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].unique())
    atom_sets = build_compound_atom_sets(expanded_ref, MODE, N_EXTRA_REPS)
    baseline_atom = constant_baseline_atom(next(iter(atom_sets.values())).shape[0])
    return classes, atom_sets, baseline_atom


def benchmark_nnls(records, classes, atom_sets, baseline_atom) -> dict:
    durations = []
    predictions = None

    for _ in range(N_REPEATS):
        rows = []
        start = perf_counter_seconds()
        for record in records:
            best, _ = search_best_pair_with_atom_sets(
                record.spectrum,
                classes,
                atom_sets,
                baseline_atom,
            )
            rows.append(tuple(best["labels"]))
        duration = perf_counter_seconds() - start
        durations.append(duration)
        predictions = rows

    assert predictions is not None
    return {
        "repeats": N_REPEATS,
        "total_seconds_mean": float(np.mean(durations)),
        "total_seconds_std": float(np.std(durations)),
        "milliseconds_per_sample_mean": float(1000.0 * np.mean(durations) / len(records)),
        "milliseconds_per_sample_std": float(1000.0 * np.std(durations) / len(records)),
        "predictions": predictions,
    }


def train_or_load_deep_model(ref_df: pd.DataFrame, wav_axis: np.ndarray, device: torch.device):
    model_path = RESULTS_DIR / "similarity_supervised_regressor_baseline_corrected.pt"
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, dictionary = build_mean_dictionary(expanded_ref, MODE)
    decoder = normalize_dictionary(dictionary).T

    dataset = build_synthetic_dataset(expanded_ref, classes, MODE)
    split_idx = split_synthetic_dataset(dataset)

    if model_path.exists():
        from run_deep_binary_coefficient_regressor import CoefficientRegressor

        model = CoefficientRegressor(dataset.spectra.shape[1], dataset.coefficients.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        trained_now = False
    else:
        model, _ = train_model_with_similarity_supervision(dataset, split_idx, decoder, device)
        torch.save(model.state_dict(), model_path)
        trained_now = True

    return classes, model, trained_now


def build_real_target(records, classes: list[str]) -> np.ndarray:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    for row_idx, record in enumerate(records):
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
    return y_true


def benchmark_deep(records, classes, model, device: torch.device) -> dict:
    spectra = np.vstack([normalize_spectrum(record.spectrum) for record in records]).astype(np.float32)
    tensor = torch.tensor(spectra, dtype=torch.float32, device=device)

    with torch.no_grad():
        _ = model(tensor[: min(32, len(tensor))])

    batch_durations = []
    batch_predictions = None
    with torch.no_grad():
        for _ in range(N_REPEATS):
            start = perf_counter_seconds()
            _, shares = model(tensor)
            y_pred = support_from_top2(shares.detach().cpu().numpy())
            duration = perf_counter_seconds() - start
            batch_durations.append(duration)
            batch_predictions = y_pred

    single_durations = []
    with torch.no_grad():
        for _ in range(N_REPEATS):
            preds = []
            start = perf_counter_seconds()
            for row in tensor:
                _, shares = model(row[None, :])
                preds.append(support_from_top2(shares.detach().cpu().numpy())[0])
            duration = perf_counter_seconds() - start
            single_durations.append(duration)

    assert batch_predictions is not None
    y_true = build_real_target(records, classes)
    metrics = compute_metrics(y_true, batch_predictions)
    return {
        "repeats": N_REPEATS,
        "device": device.type,
        "batched_total_seconds_mean": float(np.mean(batch_durations)),
        "batched_total_seconds_std": float(np.std(batch_durations)),
        "batched_milliseconds_per_sample_mean": float(1000.0 * np.mean(batch_durations) / len(records)),
        "batched_milliseconds_per_sample_std": float(1000.0 * np.std(batch_durations) / len(records)),
        "single_total_seconds_mean": float(np.mean(single_durations)),
        "single_total_seconds_std": float(np.std(single_durations)),
        "single_milliseconds_per_sample_mean": float(1000.0 * np.mean(single_durations) / len(records)),
        "single_milliseconds_per_sample_std": float(1000.0 * np.std(single_durations) / len(records)),
        "metrics": metrics,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    ref_df, wav_axis, records = build_real_records()
    classes_nnls, atom_sets, baseline_atom = build_nnls_state(ref_df, wav_axis)

    print(f"Benchmarking {len(records)} real mixture spectra")
    print("Timing replicate-dictionary pair NNLS inference...")
    nnls = benchmark_nnls(records, classes_nnls, atom_sets, baseline_atom)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Preparing similarity-supervised coefficient regressor...")
    classes_deep, model, trained_now = train_or_load_deep_model(ref_df, wav_axis, device)
    print("Timing coefficient-regressor inference...")
    deep = benchmark_deep(records, classes_deep, model, device)

    summary = {
        "dataset": {
            "mode": MODE,
            "n_real_mixture_spectra": len(records),
            "existing_real_spectra": sum(record.dataset == "existing_real" for record in records),
            "pt2_real_spectra": sum(record.dataset == "pt2_real" for record in records),
        },
        "nnls": {
            k: v for k, v in nnls.items() if k != "predictions"
        },
        "deep": deep,
        "deep_model_trained_during_this_run": trained_now,
        "notes": [
            "Timings exclude data loading and deep-model training.",
            "NNLS timing is an exhaustive search over all 136 binary supports using the replicate dictionary.",
            "Deep batched timing runs all 904 spectra in one tensor batch.",
            "Deep single timing loops one spectrum at a time through the trained network.",
        ],
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {
                "method": "Replicate-Dictionary Pair NNLS",
                "mode": "per-spectrum loop",
                "total_seconds_mean": summary["nnls"]["total_seconds_mean"],
                "total_seconds_std": summary["nnls"]["total_seconds_std"],
                "milliseconds_per_sample_mean": summary["nnls"]["milliseconds_per_sample_mean"],
                "milliseconds_per_sample_std": summary["nnls"]["milliseconds_per_sample_std"],
            },
            {
                "method": "Similarity-Supervised Coefficient Regressor",
                "mode": "batched forward pass",
                "total_seconds_mean": summary["deep"]["batched_total_seconds_mean"],
                "total_seconds_std": summary["deep"]["batched_total_seconds_std"],
                "milliseconds_per_sample_mean": summary["deep"]["batched_milliseconds_per_sample_mean"],
                "milliseconds_per_sample_std": summary["deep"]["batched_milliseconds_per_sample_std"],
            },
            {
                "method": "Similarity-Supervised Coefficient Regressor",
                "mode": "single-spectrum loop",
                "total_seconds_mean": summary["deep"]["single_total_seconds_mean"],
                "total_seconds_std": summary["deep"]["single_total_seconds_std"],
                "milliseconds_per_sample_mean": summary["deep"]["single_milliseconds_per_sample_mean"],
                "milliseconds_per_sample_std": summary["deep"]["single_milliseconds_per_sample_std"],
            },
        ]
    ).to_csv(RESULTS_DIR / "timing_summary.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
