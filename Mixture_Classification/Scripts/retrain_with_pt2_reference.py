from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


SEED = 42
LAM, P, NITER = 1e4, 0.01, 10
RATIOS = np.arange(0.05, 1.0, 0.05)
NOISE_LEVEL = 0.01
N_PER_RATIO = 10
SIAMESE_BATCH_SIZE = 32
MLP_BATCH_SIZE = 64
SIAMESE_RAMAN_EPOCHS = 10
SIAMESE_FFT_EPOCHS = 20
MLP_EPOCHS = 200


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "Mixture_Classification" / "Notebooks"
PT2_DIR = REPO_ROOT / "Jesse dataset pt2"
OUTPUT_DIR = NOTEBOOK_DIR / "pt2_augmented_reference_experiment"


PURE_LABEL_ALIASES = {
    "Acetonitrile": "acetonitrile",
    "DCM": "dichloromethane",
    "Toluene": "toluene",
    "diethylamine": "diethylamine",
    "n-hexane": "n-hexane",
}


MIXTURE_LABEL_ALIASES = {
    "mercaptohexanol": "6-mercapto-1-hexanol",
    "dichloromethane": "dichloromethane",
    "acetonitrile": "acetonitrile",
    "n-hexane": "n-hexane",
    "diethylamine": "diethylamine",
    "toluene": "toluene",
}


@dataclass
class MixtureRecord:
    sample_id: str
    source: str
    path: Path
    true_labels: tuple[str, str]


class SiameseNet(nn.Module):
    def __init__(self, input_len: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((input_len // 4) * 32, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(z, dim=1)


class PresenceNetLogits(nn.Module):
    def __init__(self, d_input: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RamanPairDataset(Dataset):
    def __init__(self, specs: np.ndarray, pair_labels: list[tuple[str, str]], augment: bool):
        self.specs = specs.astype(np.float32)
        self.labels = pair_labels
        self.augment = augment
        self.by_label: dict[tuple[str, str], list[int]] = {}
        for idx, label in enumerate(pair_labels):
            self.by_label.setdefault(label, []).append(idx)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        x1 = self.specs[idx]
        label = self.labels[idx]

        if random.random() < 0.5:
            other_idx = random.choice(self.by_label[label])
            target = 1.0
        else:
            negative_labels = [k for k in self.by_label if k != label]
            negative_label = random.choice(negative_labels)
            other_idx = random.choice(self.by_label[negative_label])
            target = 0.0

        x2 = self.specs[other_idx]
        if self.augment:
            x1 = augment_spectrum(x1)
            x2 = augment_spectrum(x2)

        return (
            torch.tensor(x1, dtype=torch.float32).unsqueeze(0),
            torch.tensor(x2, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target, dtype=torch.float32),
        )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def baseline_penalty(length: int):
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    return (LAM * (diff.T @ diff)).tocsc()


def baseline_als(y: np.ndarray) -> np.ndarray:
    penalty = baseline_penalty(len(y))
    weights = np.ones(len(y), dtype=np.float64)
    signal = y.astype(np.float64)
    for _ in range(NITER):
        system = sparse.diags(weights, offsets=0, format="csc") + penalty
        baseline = spsolve(system, weights * signal)
        weights = np.where(signal > baseline, P, 1 - P)
    return baseline


def preprocess_pair(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrected = spectrum.astype(np.float64) - baseline_als(spectrum)
    raman = corrected.copy()
    raman_norm = np.linalg.norm(raman)
    if raman_norm > 0:
        raman = raman / raman_norm
    raman = np.abs(raman).astype(np.float32)

    fft = np.abs(np.fft.rfft(corrected))
    fft = np.log1p(fft)
    fft_norm = np.linalg.norm(fft)
    if fft_norm > 0:
        fft = fft / fft_norm
    fft = fft.astype(np.float32)
    return raman, fft


def preprocess_matrix(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raman = []
    fft = []
    for spec in spectra:
        r, f = preprocess_pair(spec)
        raman.append(r)
        fft.append(f)
    return np.vstack(raman), np.vstack(fft)


def augment_spectrum(spec: np.ndarray, noise_std: float = 0.01, shift_max: int = 2) -> np.ndarray:
    noisy = spec + np.random.normal(0.0, noise_std, size=spec.shape)
    shift = np.random.randint(-shift_max, shift_max + 1)
    return np.roll(noisy, shift).astype(np.float32)


def floatify_cols(df: pd.DataFrame) -> None:
    converted = []
    for col in df.columns:
        if col in {"Label", "Label 1", "Label 2"}:
            converted.append(col)
        else:
            converted.append(float(col))
    df.columns = converted


def exact_match_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "exact_match": exact_match_rate(y_true, y_pred),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "mean_true_labels": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels": float(y_pred.sum(axis=1).mean()),
        "zero_prediction_rate": float(np.mean(y_pred.sum(axis=1) == 0)),
    }


def load_txt_spectrum(path: Path, wav_axis: np.ndarray) -> np.ndarray:
    arr = np.loadtxt(path)
    wn = arr[:, 0].astype(float)
    intensity = arr[:, 1].astype(float)
    order = np.argsort(wn)
    wn = wn[order]
    intensity = intensity[order]
    lo, hi = wav_axis.min(), wav_axis.max()
    mask = (wn >= lo) & (wn <= hi)
    wn = wn[mask]
    intensity = intensity[mask]
    if len(wn) < 2:
        raise ValueError(f"Insufficient spectral coverage in {path}")
    return np.interp(wav_axis, wn, intensity)


def load_reference() -> tuple[pd.DataFrame, np.ndarray]:
    ref_df = pd.read_csv(NOTEBOOK_DIR / "reference_v2.csv")
    floatify_cols(ref_df)
    wav_axis = np.array([c for c in ref_df.columns if c != "Label"], dtype=float)
    return ref_df, wav_axis


def build_expanded_reference(ref_df: pd.DataFrame, wav_axis: np.ndarray) -> pd.DataFrame:
    rows = []
    for pure_dir_name, label in PURE_LABEL_ALIASES.items():
        txt_dir = PT2_DIR / pure_dir_name / "txt"
        for txt_path in sorted(txt_dir.glob("*.txt")):
            spectrum = load_txt_spectrum(txt_path, wav_axis)
            row = {"Label": label}
            row.update({wav: val for wav, val in zip(wav_axis, spectrum)})
            rows.append(row)
    extra_df = pd.DataFrame(rows, columns=ref_df.columns)
    return pd.concat([ref_df, extra_df], ignore_index=True)


def generate_synthetic_mixtures(ref_df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    ref_specs = ref_df[wav_cols].to_numpy(dtype=np.float32)
    ref_labels = ref_df["Label"].astype(str).to_numpy()
    classes = sorted(np.unique(ref_labels))

    synth_specs = []
    synth_labels: list[tuple[str, str]] = []
    for (_, class_i), (_, class_j) in combinations(enumerate(classes), 2):
        idx_i = np.where(ref_labels == class_i)[0]
        idx_j = np.where(ref_labels == class_j)[0]
        for ratio in RATIOS:
            for _ in range(N_PER_RATIO):
                spec_i = ref_specs[np.random.choice(idx_i)]
                spec_j = ref_specs[np.random.choice(idx_j)]
                mixed = ratio * spec_i + (1.0 - ratio) * spec_j
                mixed += np.random.normal(scale=NOISE_LEVEL, size=mixed.shape)
                synth_specs.append(mixed.astype(np.float32))
                synth_labels.append((class_i, class_j))
    return np.vstack(synth_specs), synth_labels, classes


def train_siamese(
    proc_inputs: np.ndarray,
    pair_labels: list[tuple[str, str]],
    input_len: int,
    epochs: int,
    device: torch.device,
) -> SiameseNet:
    x_train, _, y_train, _ = train_test_split(
        proc_inputs, pair_labels, test_size=0.2, random_state=SEED
    )
    train_ds = RamanPairDataset(x_train, y_train, augment=True)
    loader = DataLoader(
        train_ds,
        batch_size=SIAMESE_BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = SiameseNet(input_len=input_len, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x1, x2, labels in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            distances = F.pairwise_distance(z1, z2)
            loss = (labels * distances.square() + (1 - labels) * F.relu(1.0 - distances).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
        print(f"Siamese epoch {epoch:03d}/{epochs:03d} loss={total_loss / len(train_ds):.4f}")

    model.eval()
    return model


def embed_inputs(model: SiameseNet, proc_inputs: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.tensor(proc_inputs, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True).unsqueeze(1)
            outputs.append(model(xb).cpu().numpy())
    return np.vstack(outputs)


def build_multihot_labels(
    pair_labels: list[tuple[str, str]], classes: list[str], class_to_i: dict[str, int]
) -> np.ndarray:
    labels = np.zeros((len(pair_labels), len(classes)), dtype=np.int64)
    for idx, (left, right) in enumerate(pair_labels):
        labels[idx, class_to_i[left]] = 1
        labels[idx, class_to_i[right]] = 1
    return labels


def train_mlp(x_synth: np.ndarray, y_synth: np.ndarray, device: torch.device):
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x_synth, y_synth, test_size=0.10, random_state=0
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=0.1111, random_state=0
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=MLP_BATCH_SIZE,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=MLP_BATCH_SIZE,
        pin_memory=device.type == "cuda",
    )

    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor((neg / pos).clip(min=1.0), dtype=torch.float32, device=device)

    model = PresenceNetLogits(d_input=x_synth.shape[1], n_classes=y_synth.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == MLP_EPOCHS:
            print(
                f"MLP epoch {epoch:03d}/{MLP_EPOCHS:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_probs = predict_probabilities(model, x_test, device)
    test_pred = (test_probs >= 0.5).astype(int)
    synthetic_test_metrics = compute_metrics(y_test, test_pred)

    return model, synthetic_test_metrics, {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "test_loader": test_loader,
    }


def predict_probabilities(model: PresenceNetLogits, x: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.vstack(outputs)


def parse_mix_manifest() -> dict[str, tuple[str, str]]:
    manifest = {}
    for line in (PT2_DIR / "Mixtures.txt").read_text().splitlines():
        line = line.strip()
        if not line.startswith("Mix "):
            continue
        mix_name, labels = line.split(" - ", 1)
        left, right = labels.split(" + ")
        left = MIXTURE_LABEL_ALIASES.get(left.strip().lower(), left.strip().lower())
        right = MIXTURE_LABEL_ALIASES.get(right.strip().lower(), right.strip().lower())
        manifest[mix_name] = (left, right)
    return manifest


def build_pt2_mixture_records() -> list[MixtureRecord]:
    records = []
    manifest = parse_mix_manifest()
    for mix_name, labels in sorted(manifest.items()):
        txt_dir = PT2_DIR / mix_name / "txt"
        for txt_path in sorted(txt_dir.glob("*.txt")):
            records.append(
                MixtureRecord(
                    sample_id=f"{mix_name}/{txt_path.name}",
                    source=mix_name,
                    path=txt_path,
                    true_labels=labels,
                )
            )
    return records


def evaluate_mixture_set(
    records: list[MixtureRecord],
    wav_axis: np.ndarray,
    siamese_raman: SiameseNet,
    siamese_fft: SiameseNet,
    model: PresenceNetLogits,
    classes: list[str],
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    spectra = np.vstack([load_txt_spectrum(rec.path, wav_axis) for rec in records]).astype(np.float32)
    raman, fft = preprocess_matrix(spectra)
    emb_raman = embed_inputs(siamese_raman, raman, device)
    emb_fft = embed_inputs(siamese_fft, fft, device)
    embeds = np.hstack([emb_raman, emb_fft]).astype(np.float32)
    probs = predict_probabilities(model, embeds, device)

    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=np.int64)
    for idx, rec in enumerate(records):
        y_true[idx, class_to_i[rec.true_labels[0]]] = 1
        y_true[idx, class_to_i[rec.true_labels[1]]] = 1

    y_pred = (probs >= 0.5).astype(int)
    overall = compute_metrics(y_true, y_pred)

    per_source = []
    rows = []
    for source in sorted({rec.source for rec in records}):
        idx = [i for i, rec in enumerate(records) if rec.source == source]
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        metrics["source"] = source
        metrics["samples"] = len(idx)
        metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        per_source.append(metrics)

    for rec, row_probs, row_pred in zip(records, probs, y_pred):
        ranked = np.argsort(row_probs)[::-1][:5]
        rows.append(
            {
                "sample_id": rec.sample_id,
                "source": rec.source,
                "true_labels": " + ".join(rec.true_labels),
                "predicted_labels": " + ".join(classes[i] for i in np.where(row_pred == 1)[0]),
                "top1_label": classes[ranked[0]],
                "top1_prob": float(row_probs[ranked[0]]),
                "top2_label": classes[ranked[1]],
                "top2_prob": float(row_probs[ranked[1]]),
                "top3_label": classes[ranked[2]],
                "top3_prob": float(row_probs[ranked[2]]),
                "top4_label": classes[ranked[3]],
                "top4_prob": float(row_probs[ranked[3]]),
                "top5_label": classes[ranked[4]],
                "top5_prob": float(row_probs[ranked[4]]),
            }
        )

    return {"overall": overall, "per_mixture": per_source}, pd.DataFrame(rows)


def evaluate_existing_real_mixtures(
    wav_axis: np.ndarray,
    siamese_raman: SiameseNet,
    siamese_fft: SiameseNet,
    model: PresenceNetLogits,
    classes: list[str],
    device: torch.device,
) -> dict[str, float]:
    mix_df = pd.read_csv(NOTEBOOK_DIR / "mixtures_dataset.csv")
    floatify_cols(mix_df)
    wav_cols = [c for c in mix_df.columns if c not in {"Label 1", "Label 2"}]
    spectra = mix_df[wav_cols].to_numpy(dtype=np.float32)
    raman, fft = preprocess_matrix(spectra)
    emb_raman = embed_inputs(siamese_raman, raman, device)
    emb_fft = embed_inputs(siamese_fft, fft, device)
    embeds = np.hstack([emb_raman, emb_fft]).astype(np.float32)
    probs = predict_probabilities(model, embeds, device)

    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(mix_df), len(classes)), dtype=np.int64)
    for idx, row in mix_df.iterrows():
        y_true[idx, class_to_i[row["Label 1"]]] = 1
        y_true[idx, class_to_i[row["Label 2"]]] = 1
    y_pred = (probs >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    expanded_ref.to_csv(OUTPUT_DIR / "reference_v2_plus_pt2.csv", index=False)
    print(f"Expanded reference shape: {expanded_ref.shape}")
    print(expanded_ref["Label"].value_counts().sort_index().to_string())

    synth_specs, synth_labels, classes = generate_synthetic_mixtures(expanded_ref)
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    print(f"Synthetic raw spectra: {synth_specs.shape}")
    print(f"Expanded classes ({len(classes)}): {classes}")

    synth_raman, synth_fft = preprocess_matrix(synth_specs)
    print(f"Synthetic Raman preproc: {synth_raman.shape}")
    print(f"Synthetic FFT preproc: {synth_fft.shape}")

    print("\nTraining Raman Siamese")
    siamese_raman = train_siamese(
        synth_raman, synth_labels, input_len=synth_raman.shape[1], epochs=SIAMESE_RAMAN_EPOCHS, device=device
    )
    print("\nTraining FFT Siamese")
    siamese_fft = train_siamese(
        synth_fft, synth_labels, input_len=synth_fft.shape[1], epochs=SIAMESE_FFT_EPOCHS, device=device
    )

    torch.save(siamese_raman.state_dict(), OUTPUT_DIR / "siamese_mixture_pt2_augmented.pth")
    torch.save(siamese_fft.state_dict(), OUTPUT_DIR / "siamese_mixture_fft_pt2_augmented.pth")

    synth_emb_raman = embed_inputs(siamese_raman, synth_raman, device)
    synth_emb_fft = embed_inputs(siamese_fft, synth_fft, device)
    x_synth = np.hstack([synth_emb_raman, synth_emb_fft]).astype(np.float32)
    y_synth = build_multihot_labels(synth_labels, classes, class_to_i)
    print(f"Synthetic embeddings: {x_synth.shape}")

    print("\nTraining MLP")
    model, synthetic_test_metrics, _ = train_mlp(x_synth, y_synth, device)
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes},
        OUTPUT_DIR / "presence_net_logits_pt2_augmented.pth",
    )

    existing_metrics = evaluate_existing_real_mixtures(
        wav_axis, siamese_raman, siamese_fft, model, classes, device
    )
    pt2_records = build_pt2_mixture_records()
    pt2_metrics, pt2_predictions = evaluate_mixture_set(
        pt2_records, wav_axis, siamese_raman, siamese_fft, model, classes, device
    )

    pt2_predictions.to_csv(OUTPUT_DIR / "pt2_augmented_reference_predictions.csv", index=False)
    summary = {
        "device": str(device),
        "expanded_classes": classes,
        "synthetic": {
            "n_spectra": int(len(synth_specs)),
            "metrics_test_default_0_5": synthetic_test_metrics,
        },
        "existing_real_mixtures_default_0_5": existing_metrics,
        "pt2_real_mixtures_default_0_5": pt2_metrics,
    }
    (OUTPUT_DIR / "pt2_augmented_reference_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nExisting real mixtures after retraining")
    print(
        f"  exact={existing_metrics['exact_match']:.3f} "
        f"micro_f1={existing_metrics['micro_f1']:.3f} "
        f"pred/sample={existing_metrics['mean_predicted_labels']:.2f}"
    )

    print("\nPT2 real mixtures after expanded-reference retraining")
    overall = pt2_metrics["overall"]
    print(
        f"  exact={overall['exact_match']:.3f} "
        f"micro_f1={overall['micro_f1']:.3f} "
        f"pred/sample={overall['mean_predicted_labels']:.2f}"
    )
    for row in pt2_metrics["per_mixture"]:
        print(
            f"  {row['source']:6s} {row['true_labels']:<45s} "
            f"exact={row['exact_match']:.3f} "
            f"micro_f1={row['micro_f1']:.3f} "
            f"pred/sample={row['mean_predicted_labels']:.2f}"
        )

    print(f"\nSaved outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
