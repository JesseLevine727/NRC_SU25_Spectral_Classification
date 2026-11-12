#!/usr/bin/env python3
"""Quality-tier sensitivity for the identifiable and aggregate NATO SERS tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from run_nato_sers_classical_fresh import estimator


def evaluate(frame, matrix, folds, repeats, seed, trees):
    rows = []
    for model_name in ("rbf_svm", "random_forest"):
        for repeat in range(repeats):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            for fold, (train, test) in enumerate(splitter.split(matrix, frame.target_analyte,
                                                                 frame.master_sample_id)):
                model = estimator(model_name, seed + repeat * 10 + fold, trees)
                model.fit(matrix[train], frame.target_analyte.iloc[train])
                rows.append({"model": model_name, "repeat": repeat, "fold": fold,
                             "balanced_accuracy": balanced_accuracy_score(
                                 frame.target_analyte.iloc[test], model.predict(matrix[test]))})
    return rows


def main():
    repository = Path(__file__).resolve().parents[1]
    root = repository / "Workspace" / "nato_sers_raw_reexploration_v1"
    output = root / "quality_sensitivity"; output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / "tables" / "analysis_manifest.csv")
    bundle = np.load(root / "preprocessing" / "preprocessing_candidates.npz")
    axis = bundle["axis_cm1"]; full_matrix = bundle["despike_savgol_minmax"][:, (axis >= 400) & (axis <= 1800)]
    tier_masks = {
        "unique_attributable": np.ones(len(manifest), dtype=bool),
        "notes_clear": manifest.tier_notes_clear_sers.to_numpy(bool),
        "exclude_mira1_failed_suitability": manifest.instrument.ne("Mira-1").to_numpy(),
        "notes_clear_and_exclude_mira1": manifest.tier_notes_clear_sers.to_numpy(bool) & manifest.instrument.ne("Mira-1").to_numpy(),
    }
    rows = []
    for tier, tier_mask in tier_masks.items():
        selected = np.flatnonzero(tier_mask); frame = manifest.iloc[selected].reset_index(drop=True); matrix = full_matrix[selected]
        for result in evaluate(frame, matrix, 5, 3, 20260805, 300):
            rows.append({"tier": tier, "task": "aggregate_seven_class", "station": "all",
                         "spectra": len(frame), "masters": frame.master_sample_id.nunique(), **result})
        for station in sorted(frame.station.unique()):
            local_indices = np.flatnonzero(frame.station.eq(station).to_numpy())
            local, local_matrix = frame.iloc[local_indices].reset_index(drop=True), matrix[local_indices]
            minimum_masters = local.groupby("target_analyte").master_sample_id.nunique().min()
            folds = min(4, int(minimum_masters))
            if folds < 2:
                continue
            for result in evaluate(local, local_matrix, folds, 3, 20260805, 300):
                rows.append({"tier": tier, "task": "within_station_three_class", "station": station,
                             "spectra": len(local), "masters": local.master_sample_id.nunique(), **result})
        print(tier, len(frame), flush=True)
    results = pd.DataFrame(rows); results.to_csv(output / "fold_metrics.csv", index=False)
    summary = results.groupby(["tier", "task", "station", "model"]).agg(
        spectra=("spectra", "first"), masters=("masters", "first"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"tiers": {
        tier: {"spectra": int(mask.sum()), "masters": int(manifest.loc[mask, "master_sample_id"].nunique())}
        for tier, mask in tier_masks.items()}}, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
