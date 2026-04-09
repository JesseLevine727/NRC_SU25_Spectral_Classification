from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

BASE_TRAIN_PATH = DATA_DIR / "reference_subset_1_resampled.csv"
FEB26_ALIGNED_PATH = RESULTS_DIR / "feb26_aligned_resampled.csv"
PRETRAINED_MODEL_PATH = MODELS_DIR / "siamese_raman_resampled.pth"

TRAIN_MIX_PATH = RESULTS_DIR / "cross_device_train_mix.csv"
CONTROL_HOLDOUT_PATH = RESULTS_DIR / "feb26_known_controls_holdout.csv"
UNKNOWN_HOLDOUT_PATH = RESULTS_DIR / "feb26_unknown_holdout.csv"
MODEL_OUT_PATH = MODELS_DIR / "siamese_raman_cross_device_finetuned.pth"
SUMMARY_PATH = RESULTS_DIR / "cross_device_finetune_summary.json"

KNOWN_OVERLAP_LABELS = ["benzene", "pyridine"]
UNKNOWN_LABELS = ["aniline", "dcm", "diethylamine", "n-hexane"]
N_FEB26_TRAIN_PER_CLASS = 10
SEED = 2026
PENALTY_CACHE = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def baseline_als(y, lam=1e4, p=0.01, niter=10):
    length = len(y)
    cache_key = (length, lam)
    penalty = PENALTY_CACHE.get(cache_key)
    if penalty is None:
        second_diff = np.diff(np.eye(length), 2)
        penalty = lam * second_diff.dot(second_diff.T)
        PENALTY_CACHE[cache_key] = penalty
    weights = np.ones(length)
    for _ in range(niter):
        baseline = np.linalg.solve(np.diag(weights) + penalty, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def preprocess(arr):
    out = np.zeros_like(arr)
    for idx, spectrum in enumerate(arr):
        baseline = baseline_als(spectrum)
        corrected = spectrum - baseline
        norm = np.linalg.norm(corrected)
        out[idx] = corrected / norm if norm > 0 else corrected
    return out


def augment(spec, noise_std=0.01, shift_max=2):
    noisy = spec + np.random.normal(0, noise_std, size=spec.shape)
    shift = np.random.randint(-shift_max, shift_max + 1)
    return np.roll(noisy, shift)


class RamanPairDataset(Dataset):
    def __init__(self, specs, labels, augment_fn=None):
        self.specs = specs
        self.labels = labels
        self.augment = augment_fn
        self.by_label = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        x1 = self.specs[idx]
        y1 = self.labels[idx]
        if np.random.rand() < 0.5:
            j = np.random.choice(self.by_label[y1])
            label = 1.0
        else:
            neg_labels = [label for label in self.by_label if label != y1]
            y2 = np.random.choice(neg_labels)
            j = np.random.choice(self.by_label[y2])
            label = 0.0
        x2 = self.specs[j]
        if self.augment:
            x1 = self.augment(x1)
            x2 = self.augment(x2)
        return (
            torch.tensor(x1, dtype=torch.float32).unsqueeze(0),
            torch.tensor(x2, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32),
        )


class SiameseNet(nn.Module):
    def __init__(self, input_len, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((input_len // 4) * 32, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=1)


def contrastive_loss(z1, z2, label, margin=1.0):
    dist = F.pairwise_distance(z1, z2)
    loss_pos = label * dist**2
    loss_neg = (1 - label) * F.relu(margin - dist) ** 2
    return (loss_pos + loss_neg).mean()


def build_splits():
    base_train = pd.read_csv(BASE_TRAIN_PATH)
    feb26 = pd.read_csv(FEB26_ALIGNED_PATH)
    axis_cols = [col for col in base_train.columns if col != "Label"]

    train_parts = [base_train[axis_cols + ["Label"]].copy()]
    control_parts = []
    for label in KNOWN_OVERLAP_LABELS:
        group = feb26[feb26["Label"] == label].copy()
        train_part = group.sample(n=N_FEB26_TRAIN_PER_CLASS, random_state=SEED)
        control_part = group.drop(train_part.index).copy()
        train_parts.append(train_part[axis_cols + ["Label"]].copy())
        control_parts.append(control_part[axis_cols + ["Label", "SourceFolder", "SourceFile"]].copy())

    unknown_holdout = feb26[feb26["Label"].isin(UNKNOWN_LABELS)].copy()

    train_mix = pd.concat(train_parts, ignore_index=True)
    control_holdout = pd.concat(control_parts, ignore_index=True)
    unknown_holdout = unknown_holdout[axis_cols + ["Label", "SourceFolder", "SourceFile"]].copy()

    train_mix.to_csv(TRAIN_MIX_PATH, index=False)
    control_holdout.to_csv(CONTROL_HOLDOUT_PATH, index=False)
    unknown_holdout.to_csv(UNKNOWN_HOLDOUT_PATH, index=False)
    return train_mix, control_holdout, unknown_holdout, axis_cols


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_mix, control_holdout, unknown_holdout, axis_cols = build_splits()

    labels = train_mix["Label"].astype(str).to_numpy()
    raw_specs = train_mix[axis_cols].to_numpy(dtype=float)
    spectra = preprocess(raw_specs)

    dataset = RamanPairDataset(spectra, labels, augment_fn=augment)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SiameseNet(spectra.shape[1], embed_dim=64)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location="cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 40
    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, lbl in loader:
            z1 = model(x1)
            z2 = model(x2)
            loss = contrastive_loss(z1, z2, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_OUT_PATH)

    summary = {
        "seed": SEED,
        "epochs": epochs,
        "learning_rate": 1e-4,
        "n_feb26_train_per_class": N_FEB26_TRAIN_PER_CLASS,
        "train_counts": train_mix["Label"].value_counts().sort_index().to_dict(),
        "control_holdout_counts": control_holdout["Label"].value_counts().sort_index().to_dict(),
        "unknown_holdout_counts": unknown_holdout["Label"].value_counts().sort_index().to_dict(),
        "final_loss": float(loss_history[-1]),
        "min_loss": float(min(loss_history)),
        "artifacts": {
            "train_mix_csv": TRAIN_MIX_PATH.name,
            "control_holdout_csv": CONTROL_HOLDOUT_PATH.name,
            "unknown_holdout_csv": UNKNOWN_HOLDOUT_PATH.name,
            "model_path": MODEL_OUT_PATH.name,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
