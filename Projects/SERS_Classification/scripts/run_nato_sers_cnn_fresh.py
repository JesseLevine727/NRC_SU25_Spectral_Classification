#!/usr/bin/env python3
"""Bounded, nested-validation 1D-CNN baseline for the fresh NATO SERS audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 32, 128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, classes),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def evaluate(model, x, y, device, batch_size):
    model.eval(); logits = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None, :]), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for values, _ in loader:
            logits.append(model(values.to(device)).cpu().numpy())
    return np.concatenate(logits)


def metric(truth, prediction):
    return (float(balanced_accuracy_score(truth, prediction)),
            float(f1_score(truth, prediction, average="macro", zero_division=0)))


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=repository / "configs" / "nato_sers_cnn_fresh_v1.json")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    input_dir = repository / config["input_workspace"]
    prep_dir = repository / config["preprocessing_workspace"]
    output = repository / config["output_workspace"]; output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(input_dir / "tables" / "analysis_manifest.csv")
    bundle = np.load(prep_dir / "preprocessing_candidates.npz")
    axis = bundle["axis_cm1"]; low, high = config["range_cm1"]
    x = bundle[config["preprocessing"]][:, (axis >= low) & (axis <= high)].astype(np.float32)
    labels = sorted(manifest.target_analyte.unique()); mapping = {name: i for i, name in enumerate(labels)}
    y = manifest.target_analyte.map(mapping).to_numpy(np.int64)
    groups = manifest.master_sample_id.to_numpy()
    outer = StratifiedGroupKFold(n_splits=int(config["folds"]), shuffle=True,
                                 random_state=int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold, (outer_train, test) in enumerate(outer.split(x, y, groups)):
        seed = int(config["random_seed"]) + fold; seed_everything(seed)
        inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 100)
        inner_train_rel, validation_rel = next(inner_splitter.split(
            x[outer_train], y[outer_train], groups[outer_train]
        ))
        train, validation = outer_train[inner_train_rel], outer_train[validation_rel]
        counts = np.bincount(y[train], minlength=len(labels)).astype(float)
        weights = counts.sum() / np.maximum(counts * len(labels), 1.0)
        model = SmallCNN(len(labels)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(x[train, None, :]), torch.from_numpy(y[train])),
                            batch_size=int(config["batch_size"]), shuffle=True, generator=generator)
        best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
        history = []
        for epoch in range(int(config["max_epochs"])):
            model.train(); losses = []
            for values, target in loader:
                values, target = values.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True); loss = criterion(model(values), target)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            validation_logits = evaluate(model, x[validation], y[validation], device, int(config["batch_size"]))
            validation_loss = float(nn.functional.cross_entropy(
                torch.from_numpy(validation_logits), torch.from_numpy(y[validation]),
                weight=torch.tensor(weights, dtype=torch.float32)).item())
            history.append({"fold": fold, "epoch": epoch + 1, "train_loss": np.mean(losses),
                            "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-4:
                best_loss, stale, best_epoch = validation_loss, 0, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(config["patience"]): break
        model.load_state_dict(best_state)
        logits = evaluate(model, x[test], y[test], device, int(config["batch_size"]))
        prediction = logits.argmax(axis=1)
        spectrum_bal, spectrum_f1 = metric(y[test], prediction)
        frame = manifest.iloc[test].reset_index(drop=True); master_truth, master_prediction = [], []
        for _, group in frame.reset_index().groupby("master_sample_id"):
            master_truth.append(y[test[group["index"].to_numpy()]][0])
            master_prediction.append(int(logits[group["index"]].mean(axis=0).argmax()))
        master_bal, master_f1 = metric(master_truth, master_prediction)
        rows.append({"fold": fold, "best_epoch": best_epoch, "epochs_run": len(history),
                     "n_train_masters": manifest.iloc[train].master_sample_id.nunique(),
                     "n_validation_masters": manifest.iloc[validation].master_sample_id.nunique(),
                     "n_test_masters": manifest.iloc[test].master_sample_id.nunique(),
                     "spectrum_balanced_accuracy": spectrum_bal, "spectrum_macro_f1": spectrum_f1,
                     "master_balanced_accuracy": master_bal, "master_macro_f1": master_f1})
        pd.DataFrame(history).to_csv(output / f"training_history_fold_{fold}.csv", index=False)
        print(json.dumps(rows[-1]), flush=True)
    results = pd.DataFrame(rows); results.to_csv(output / "fold_metrics.csv", index=False)
    summary = {column: {"mean": float(results[column].mean()), "std": float(results[column].std())}
               for column in ["spectrum_balanced_accuracy", "spectrum_macro_f1",
                              "master_balanced_accuracy", "master_macro_f1", "best_epoch"]}
    summary.update({"device": str(device), "architecture_parameters": sum(p.numel() for p in SmallCNN(len(labels)).parameters()),
                    "preprocessing": config["preprocessing"], "range_cm1": config["range_cm1"]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig = go.Figure()
    for fold in range(int(config["folds"])):
        history = pd.read_csv(output / f"training_history_fold_{fold}.csv")
        fig.add_trace(go.Scatter(x=history.epoch, y=history.validation_loss, mode="lines", name=f"fold {fold}"))
    fig.update_layout(template="plotly_white", title="Bounded 1D-CNN inner-validation histories",
                      xaxis_title="Epoch", yaxis_title="Weighted validation loss")
    fig.write_html(output / "training_history.html", include_plotlyjs=True, full_html=True)
    provenance = {"protocol_version": config["protocol_version"],
                  "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "preprocessing_bundle_sha256": hashlib.sha256((prep_dir / "preprocessing_candidates.npz").read_bytes()).hexdigest(),
                  "labels": labels}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
