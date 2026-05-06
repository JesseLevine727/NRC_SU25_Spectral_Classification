#!/usr/bin/env python3
"""Siamese SERS chemical detection with leave-one-substrate-out evaluation.

This keeps the original notebook's Siamese idea, but fixes the target:
the contrastive labels are chemical `Label` values, not `Label__Substrate`.
Each fold trains on all but one substrate and evaluates chemical detection
on the held-out substrate.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA = Path("Workspace/data/processed/consolidated_SERS.csv")
DEFAULT_CANONICAL_LABELS = {
    "bt": "benzenethiol",
}
DEFAULT_SUBSTRATE_GROUPS = {
    "AgNP": "Ag",
    "AuNP": "Au",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spectral_columns(df: pd.DataFrame, crop_min: float | None, crop_max: float | None) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in {"Label", "Substrate", "Class"}:
            continue
        try:
            wav = float(col)
        except ValueError:
            continue
        if (crop_min is None or wav >= crop_min) and (crop_max is None or wav <= crop_max):
            cols.append(col)
    if not cols:
        raise ValueError("No spectral numeric columns found in the requested range.")
    return cols


def load_dataset(
    path: Path,
    crop_min: float | None,
    crop_max: float | None,
    min_substrates: int,
    canonicalize_labels: bool = True,
    group_metal_substrates: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    missing = {"Label", "Substrate"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if canonicalize_labels:
        df = df.copy()
        df["Label"] = df["Label"].replace(DEFAULT_CANONICAL_LABELS)
    if group_metal_substrates:
        df = df.copy()
        df["Substrate"] = df["Substrate"].replace(DEFAULT_SUBSTRATE_GROUPS)

    valid_labels = (
        df.groupby("Label")["Substrate"].nunique().loc[lambda s: s >= min_substrates].index
    )
    df = df[df["Label"].isin(valid_labels)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No labels have enough substrates for substrate-held-out evaluation.")
    return df, spectral_columns(df, crop_min, crop_max)


def baseline_als(y: np.ndarray, lam: float = 1e4, p: float = 0.01, niter: int = 10) -> np.ndarray:
    length = y.shape[0]
    diff = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(length - 2, length),
        format="csc",
    )
    smoothness = lam * diff.T @ diff
    weights = np.ones(length)
    for _ in range(niter):
        baseline = spsolve(sparse.diags(weights, 0, format="csc") + smoothness, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def preprocess_spectra(X: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    processed = np.zeros_like(X, dtype=np.float32)
    for idx, spectrum in enumerate(np.asarray(X, dtype=np.float64)):
        corrected = spectrum - baseline_als(spectrum, lam=lam, p=p, niter=niter)
        norm = np.linalg.norm(corrected)
        processed[idx] = corrected / norm if norm > 0 else corrected
    return processed


def snv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    centered = X - np.median(X, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return centered / scale


def l2_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (X / norm).astype(np.float32)


def peak_emphasis(X: np.ndarray) -> np.ndarray:
    X = savgol_filter(snv(X), 15, 3, axis=1)
    trend = savgol_filter(X, 101, 3, axis=1)
    return l2_rows(np.maximum(X - trend, 0))


def prepare_features(X_raw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.feature == "raw":
        return np.asarray(X_raw, dtype=np.float32)
    if args.feature == "als":
        return preprocess_spectra(X_raw, args.baseline_lam, args.baseline_p, args.baseline_niter)
    if args.feature == "snv_l2":
        return l2_rows(snv(X_raw))
    if args.feature == "derivative_1":
        return l2_rows(savgol_filter(snv(X_raw), 17, 3, deriv=1, axis=1))
    if args.feature == "derivative_2":
        return l2_rows(savgol_filter(snv(X_raw), 17, 3, deriv=2, axis=1))
    if args.feature == "peak_emphasis":
        return peak_emphasis(X_raw)
    raise ValueError(f"Unsupported feature transform: {args.feature}")


def augment(spec: np.ndarray, noise_std: float, shift_max: int) -> np.ndarray:
    noisy = spec + np.random.normal(0, noise_std, size=spec.shape)
    shift = np.random.randint(-shift_max, shift_max + 1)
    return np.roll(noisy, shift)


class RamanPairDataset(Dataset):
    def __init__(self, specs: np.ndarray, labels: np.ndarray, noise_std: float, shift_max: int):
        self.specs = specs
        self.labels = labels
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.by_label = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        x1 = self.specs[idx]
        y1 = self.labels[idx]
        if np.random.rand() < 0.5:
            partner_idx = np.random.choice(self.by_label[y1])
            label = 1.0
        else:
            negative_labels = [label for label in self.by_label if label != y1]
            y2 = np.random.choice(negative_labels)
            partner_idx = np.random.choice(self.by_label[y2])
            label = 0.0

        x2 = self.specs[partner_idx]
        x1 = augment(x1, self.noise_std, self.shift_max)
        x2 = augment(x2, self.noise_std, self.shift_max)
        return (
            torch.tensor(x1, dtype=torch.float32).unsqueeze(0),
            torch.tensor(x2, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32),
        )


class RamanTripletDataset(Dataset):
    """Substrate-aware triplets for chemical-invariant embeddings.

    Anchor and positive share the same chemical label, with different substrates
    preferred when available. Negative has a different chemical label, with the
    anchor substrate preferred to make substrate-matched hard negatives.
    """

    def __init__(
        self,
        specs: np.ndarray,
        labels: np.ndarray,
        substrates: np.ndarray,
        noise_std: float,
        shift_max: int,
    ):
        self.specs = specs
        self.labels = labels
        self.substrates = substrates
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.by_label = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        self.by_label_substrate = {
            (label, substrate): np.where((labels == label) & (substrates == substrate))[0]
            for label in np.unique(labels)
            for substrate in np.unique(substrates)
        }
        self.by_substrate = {
            substrate: np.where(substrates == substrate)[0] for substrate in np.unique(substrates)
        }

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        anchor = self.specs[idx]
        anchor_label = self.labels[idx]
        anchor_substrate = self.substrates[idx]

        positive_candidates = self.by_label[anchor_label]
        cross_substrate_positive = positive_candidates[self.substrates[positive_candidates] != anchor_substrate]
        if len(cross_substrate_positive) > 0:
            positive_idx = np.random.choice(cross_substrate_positive)
        else:
            same_label_other = positive_candidates[positive_candidates != idx]
            positive_idx = np.random.choice(same_label_other if len(same_label_other) > 0 else positive_candidates)

        same_substrate = self.by_substrate[anchor_substrate]
        hard_negative_candidates = same_substrate[self.labels[same_substrate] != anchor_label]
        if len(hard_negative_candidates) > 0:
            negative_idx = np.random.choice(hard_negative_candidates)
        else:
            negative_labels = [label for label in self.by_label if label != anchor_label]
            negative_label = np.random.choice(negative_labels)
            negative_idx = np.random.choice(self.by_label[negative_label])

        positive = self.specs[positive_idx]
        negative = self.specs[negative_idx]
        anchor = augment(anchor, self.noise_std, self.shift_max)
        positive = augment(positive, self.noise_std, self.shift_max)
        negative = augment(negative, self.noise_std, self.shift_max)
        return (
            torch.tensor(anchor, dtype=torch.float32).unsqueeze(0),
            torch.tensor(positive, dtype=torch.float32).unsqueeze(0),
            torch.tensor(negative, dtype=torch.float32).unsqueeze(0),
        )


class RamanEmbeddingDataset(Dataset):
    def __init__(
        self,
        specs: np.ndarray,
        label_ids: np.ndarray,
        substrate_ids: np.ndarray,
        noise_std: float,
        shift_max: int,
    ):
        self.specs = specs
        self.label_ids = label_ids.astype(np.int64)
        self.substrate_ids = substrate_ids.astype(np.int64)
        self.noise_std = noise_std
        self.shift_max = shift_max

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = augment(self.specs[idx], self.noise_std, self.shift_max)
        return (
            torch.tensor(spec, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.label_ids[idx], dtype=torch.long),
            torch.tensor(self.substrate_ids[idx], dtype=torch.long),
        )


class BalancedLabelBatchSampler:
    """Balanced batches for batch-hard metric learning.

    Each batch samples several chemical labels and several spectra per label.
    Within each label, sampling cycles across available substrates where possible.
    """

    def __init__(
        self,
        label_ids: np.ndarray,
        substrate_ids: np.ndarray,
        labels_per_batch: int,
        samples_per_label: int,
        batches_per_epoch: int,
    ):
        self.label_ids = label_ids.astype(np.int64)
        self.substrate_ids = substrate_ids.astype(np.int64)
        self.labels = np.unique(self.label_ids)
        self.labels_per_batch = min(labels_per_batch, len(self.labels))
        self.samples_per_label = samples_per_label
        self.batches_per_epoch = batches_per_epoch
        self.by_label_substrate: dict[int, dict[int, np.ndarray]] = {}
        for label in self.labels:
            by_substrate: dict[int, np.ndarray] = {}
            label_mask = self.label_ids == label
            for substrate in np.unique(self.substrate_ids[label_mask]):
                by_substrate[int(substrate)] = np.where(label_mask & (self.substrate_ids == substrate))[0]
            self.by_label_substrate[int(label)] = by_substrate

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_labels = np.random.choice(self.labels, size=self.labels_per_batch, replace=False)
            batch: list[int] = []
            for label in chosen_labels:
                groups = self.by_label_substrate[int(label)]
                substrates = list(groups)
                np.random.shuffle(substrates)
                for i in range(self.samples_per_label):
                    substrate = substrates[i % len(substrates)]
                    batch.append(int(np.random.choice(groups[substrate])))
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


class SiameseNet(nn.Module):
    def __init__(self, input_len: int, embed_dim: int = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, label: torch.Tensor, margin: float) -> torch.Tensor:
    dist = F.pairwise_distance(z1, z2)
    loss_pos = label * dist**2
    loss_neg = (1 - label) * F.relu(margin - dist) ** 2
    return (loss_pos + loss_neg).mean()


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    substrates: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distances = torch.cdist(embeddings, embeddings, p=2)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_label = labels[:, None] == labels[None, :]
    same_substrate = substrates[:, None] == substrates[None, :]

    positive_mask = same_label & ~eye
    cross_substrate_positive = positive_mask & ~same_substrate
    positive_mask = torch.where(
        cross_substrate_positive.any(dim=1, keepdim=True),
        cross_substrate_positive,
        positive_mask,
    )

    negative_mask = ~same_label
    same_substrate_negative = negative_mask & same_substrate
    negative_mask = torch.where(
        same_substrate_negative.any(dim=1, keepdim=True),
        same_substrate_negative,
        negative_mask,
    )

    hard_positive = distances.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
    hard_negative = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values
    valid = torch.isfinite(hard_positive) & torch.isfinite(hard_negative)
    if not valid.any():
        return embeddings.sum() * 0.0
    return F.relu(hard_positive[valid] - hard_negative[valid] + margin).mean()


def embed(model: SiameseNet, specs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            batch = torch.tensor(specs[start : start + batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            out.append(model(batch).cpu().numpy())
    return np.vstack(out)


@dataclass(frozen=True)
class FoldResult:
    held_out_substrate: str
    prototype_mode: str
    n_train: int
    n_test: int
    test_labels: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    final_loss: float


def evaluate_fold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    substrate: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FoldResult, pd.DataFrame]:
    train_mask = groups != substrate
    test_mask = groups == substrate

    train_labels = y[train_mask]
    train_groups = groups[train_mask]
    test_labels = y[test_mask]
    known_mask = np.isin(test_labels, np.unique(train_labels))
    X_test = X[test_mask][known_mask]
    test_labels = test_labels[known_mask]

    if args.loss == "batch_hard_triplet":
        label_to_id = {label: idx for idx, label in enumerate(sorted(np.unique(train_labels)))}
        substrate_to_id = {group: idx for idx, group in enumerate(sorted(np.unique(train_groups)))}
        train_label_ids = np.array([label_to_id[label] for label in train_labels], dtype=np.int64)
        train_substrate_ids = np.array([substrate_to_id[group] for group in train_groups], dtype=np.int64)
        dataset = RamanEmbeddingDataset(
            X[train_mask],
            train_label_ids,
            train_substrate_ids,
            args.noise_std,
            args.shift_max,
        )
        batches_per_epoch = args.batches_per_epoch or max(1, int(np.ceil(len(dataset) / args.batch_size)))
        sampler = BalancedLabelBatchSampler(
            train_label_ids,
            train_substrate_ids,
            args.labels_per_batch,
            args.samples_per_label,
            batches_per_epoch,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    elif args.loss == "triplet":
        dataset = RamanTripletDataset(X[train_mask], train_labels, train_groups, args.noise_std, args.shift_max)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = RamanPairDataset(X[train_mask], train_labels, args.noise_std, args.shift_max)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SiameseNet(X.shape[1], embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    final_loss = 0.0
    model.train()
    for _ in range(args.epochs):
        total = 0.0
        for batch in loader:
            if args.loss == "batch_hard_triplet":
                spectra, label_ids, substrate_ids = (item.to(device) for item in batch)
                loss = batch_hard_triplet_loss(model(spectra), label_ids, substrate_ids, args.margin)
                batch_size = spectra.size(0)
            elif args.loss == "triplet":
                anchor, positive, negative = (item.to(device) for item in batch)
                loss = triplet_loss(model(anchor), model(positive), model(negative), args.margin)
                batch_size = anchor.size(0)
            else:
                x1, x2, lbl = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                lbl = lbl.to(device)
                loss = contrastive_loss(model(x1), model(x2), lbl, args.margin)
                batch_size = x1.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_size
        final_loss = total / len(dataset)

    train_embeddings = embed(model, X[train_mask], device, args.batch_size)
    test_embeddings = embed(model, X_test, device, args.batch_size)
    prototype_labels = sorted(np.unique(train_labels))
    prototypes = []
    for label in prototype_labels:
        label_mask = train_labels == label
        if args.prototype_mode == "substrate_balanced":
            substrate_centroids = []
            for group in sorted(np.unique(train_groups[label_mask])):
                group_mask = label_mask & (train_groups == group)
                substrate_centroids.append(train_embeddings[group_mask].mean(axis=0))
            prototypes.append(np.vstack(substrate_centroids).mean(axis=0))
        else:
            prototypes.append(train_embeddings[label_mask].mean(axis=0))
    prototypes = np.vstack(prototypes)

    distances = np.linalg.norm(test_embeddings[:, None, :] - prototypes[None, :, :], axis=2)
    pred = np.array([prototype_labels[idx] for idx in np.argmin(distances, axis=1)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = balanced_accuracy_score(test_labels, pred)

    result = FoldResult(
        held_out_substrate=substrate,
        prototype_mode=args.prototype_mode,
        n_train=int(train_mask.sum()),
        n_test=int(len(test_labels)),
        test_labels=",".join(sorted(np.unique(test_labels))),
        accuracy=accuracy_score(test_labels, pred),
        balanced_accuracy=balanced_accuracy,
        macro_f1=f1_score(test_labels, pred, average="macro"),
        final_loss=final_loss,
    )
    labels = sorted(np.unique(np.concatenate([test_labels, pred])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        matrix = confusion_matrix(test_labels, pred, labels=labels)
    confusion = pd.DataFrame(
        matrix,
        index=[f"true:{label}" for label in labels],
        columns=[f"pred:{label}" for label in labels],
    )
    return result, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--crop-min", type=float, default=330.0)
    parser.add_argument("--crop-max", type=float, default=1800.0)
    parser.add_argument("--min-substrates", type=int, default=2)
    parser.add_argument(
        "--no-canonicalize-labels",
        action="store_true",
        help="Disable chemical label canonicalization such as bt -> benzenethiol.",
    )
    parser.add_argument(
        "--group-metal-substrates",
        action="store_true",
        help="Group AgNP with Ag and AuNP with Au before leave-substrate-out evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--loss",
        choices=["batch_hard_triplet", "triplet", "contrastive"],
        default="batch_hard_triplet",
        help="Metric-learning objective. Batch-hard triplet is substrate-aware and is the default.",
    )
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument(
        "--prototype-mode",
        choices=["row_mean", "substrate_balanced"],
        default="substrate_balanced",
        help="How chemical prototypes are formed from training embeddings.",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Override balanced batches per epoch. Defaults to roughly one pass over the training rows.",
    )
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--shift-max", type=int, default=2)
    parser.add_argument("--baseline-lam", type=float, default=1e4)
    parser.add_argument("--baseline-p", type=float, default=0.01)
    parser.add_argument("--baseline-niter", type=int, default=10)
    parser.add_argument(
        "--feature",
        choices=["raw", "als", "snv_l2", "derivative_1", "derivative_2", "peak_emphasis"],
        default="als",
        help="Input representation for the Siamese encoder. `als` matches the original notebooks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cuda", "auto", "cpu"],
        default="cuda",
        help="Training device. Defaults to CUDA and fails if no GPU is available.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU training. Without this flag, CPU execution is rejected.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Workspace/substrate_agnostic/current/best_siamese_triplet/results.csv"),
    )
    parser.add_argument(
        "--confusions-dir",
        type=Path,
        default=Path("Workspace/substrate_agnostic/current/best_siamese_triplet/confusions"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CPU training is disabled. Use --device cuda on a CUDA machine, "
            "or pass --allow-cpu only for an intentional CPU debug run."
        )

    df, cols = load_dataset(
        args.data,
        args.crop_min,
        args.crop_max,
        args.min_substrates,
        canonicalize_labels=not args.no_canonicalize_labels,
        group_metal_substrates=args.group_metal_substrates,
    )
    X_raw = df[cols].to_numpy(dtype=np.float64)
    X = prepare_features(X_raw, args)
    y = df["Label"].astype(str).to_numpy()
    groups = df["Substrate"].astype(str).to_numpy()

    rows: list[FoldResult] = []
    confusions: dict[str, pd.DataFrame] = {}
    for substrate in sorted(np.unique(groups)):
        result, confusion = evaluate_fold(X, y, groups, substrate, args, device)
        if result.n_test == 0:
            continue
        rows.append(result)
        confusions[substrate] = confusion
        print(
            f"{substrate}: acc={result.accuracy:.3f}, "
            f"bal_acc={result.balanced_accuracy:.3f}, macro_f1={result.macro_f1:.3f}"
        )

    results = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.confusions_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    for substrate, matrix in confusions.items():
        matrix.to_csv(args.confusions_dir / f"{substrate}.csv")

    print("\nDataset:", args.data)
    print("Device:", device)
    print("Feature:", args.feature)
    print("Loss:", args.loss)
    print("Canonical labels:", not args.no_canonicalize_labels)
    print("Grouped metal substrates:", args.group_metal_substrates)
    print("Rows evaluated:", len(df))
    print("Labels:", ", ".join(sorted(df["Label"].unique())))
    print("Substrates:", ", ".join(sorted(df["Substrate"].unique())))
    print(f"Spectral window: {min(map(float, cols)):.3f} to {max(map(float, cols)):.3f} cm^-1")
    print("\nPer-fold results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nMean:")
    print(results[["accuracy", "balanced_accuracy", "macro_f1"]].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to {args.out}")
    print(f"Saved confusion matrices to {args.confusions_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
