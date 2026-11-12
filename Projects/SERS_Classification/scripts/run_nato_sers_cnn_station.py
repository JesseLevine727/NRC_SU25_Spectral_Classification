#!/usr/bin/env python3
"""Bounded 1D-CNN control on chemistry-identifiable within-station tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_nato_sers_cnn_fresh import SmallCNN, evaluate, metric, seed_everything


def main():
    repository = Path(__file__).resolve().parents[1]
    root = repository / "Workspace" / "nato_sers_raw_reexploration_v1"
    output = root / "identifiability" / "cnn"; output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / "tables" / "analysis_manifest.csv")
    bundle = np.load(root / "preprocessing" / "preprocessing_candidates.npz")
    axis = bundle["axis_cm1"]; x_all = bundle["minimal_minmax"][:, (axis >= 400) & (axis <= 1800)].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for station in sorted(manifest.station.unique()):
        global_indices = np.flatnonzero(manifest.station.eq(station).to_numpy())
        frame = manifest.iloc[global_indices].reset_index(drop=True); x = x_all[global_indices]
        labels = sorted(frame.target_analyte.unique()); mapping = {label: index for index, label in enumerate(labels)}
        y = frame.target_analyte.map(mapping).to_numpy(np.int64); groups = frame.master_sample_id.to_numpy()
        outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260805)
        for fold, (outer_train, test) in enumerate(outer.split(x, y, groups)):
            seed = 20260805 + fold; seed_everything(seed)
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 100)
            train_rel, validation_rel = next(inner.split(x[outer_train], y[outer_train], groups[outer_train]))
            train, validation = outer_train[train_rel], outer_train[validation_rel]
            counts = np.bincount(y[train], minlength=len(labels)).astype(float)
            weights = counts.sum() / np.maximum(counts * len(labels), 1)
            model = SmallCNN(len(labels)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            loader = DataLoader(TensorDataset(torch.from_numpy(x[train, None, :]), torch.from_numpy(y[train])),
                                batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
            best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
            for epoch in range(100):
                model.train()
                for values, target in loader:
                    values, target = values.to(device), target.to(device); optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(values), target); loss.backward(); optimizer.step()
                logits = evaluate(model, x[validation], y[validation], device, 64)
                validation_loss = float(nn.functional.cross_entropy(
                    torch.from_numpy(logits), torch.from_numpy(y[validation]),
                    weight=torch.tensor(weights, dtype=torch.float32)).item())
                if validation_loss < best_loss - 1e-4:
                    best_loss, stale, best_epoch = validation_loss, 0, epoch + 1
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 12: break
            model.load_state_dict(best_state); logits = evaluate(model, x[test], y[test], device, 64)
            spectrum_bal, spectrum_f1 = metric(y[test], logits.argmax(axis=1))
            local_test = frame.iloc[test].reset_index(drop=True); master_truth, master_pred = [], []
            for _, group in local_test.reset_index().groupby("master_sample_id"):
                positions = group["index"].to_numpy(); master_truth.append(y[test[positions]][0])
                master_pred.append(int(logits[positions].mean(axis=0).argmax()))
            master_bal, master_f1 = metric(master_truth, master_pred)
            rows.append({"station": station, "fold": fold, "best_epoch": best_epoch,
                         "n_test_masters": local_test.master_sample_id.nunique(),
                         "spectrum_balanced_accuracy": spectrum_bal, "spectrum_macro_f1": spectrum_f1,
                         "master_balanced_accuracy": master_bal, "master_macro_f1": master_f1})
            print(json.dumps(rows[-1]), flush=True)
    results = pd.DataFrame(rows); results.to_csv(output / "fold_metrics.csv", index=False)
    summary = results.groupby("station").agg(
        spectrum_balanced_accuracy_mean=("spectrum_balanced_accuracy", "mean"),
        spectrum_balanced_accuracy_std=("spectrum_balanced_accuracy", "std"),
        master_balanced_accuracy_mean=("master_balanced_accuracy", "mean"),
        master_balanced_accuracy_std=("master_balanced_accuracy", "std"),
        best_epoch_mean=("best_epoch", "mean")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps({"device": str(device),
        "architecture_parameters": sum(p.numel() for p in SmallCNN(3).parameters()),
        "preprocessing": "minimal_minmax", "range_cm1": [400, 1800], "outer_folds": 4}, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
