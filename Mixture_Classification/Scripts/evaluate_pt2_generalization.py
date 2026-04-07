from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


LAM, P, NITER = 1e4, 0.01, 10
BATCH_SIZE = 256


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "Mixture_Classification" / "Notebooks"
PT2_DIR = REPO_ROOT / "Jesse dataset pt2"
OUTPUT_DIR = NOTEBOOK_DIR


LABEL_ALIASES = {
    "benzene": "benzene",
    "benzenethiol": "benzenethiol",
    "bt": "benzenethiol",
    "pyridine": "pyridine",
    "meoh": "meoh",
    "methanol": "meoh",
    "etoh": "etoh",
    "ethanol": "etoh",
    "dmmp": "dmmp",
    "dimethyl methylphosphanate": "dmmp",
    "n,n-dimethylformamide": "n,n-dimethylformamide",
    "mercaptohexanol": "6-mercapto-1-hexanol",
    "6-mercapto-1-hexanol": "6-mercapto-1-hexanol",
    "1-dodecanethiol": "1-dodecanethiol",
    "1-undecanethiol": "1-undecanethiol",
    "1,9-nonanedithiol": "1,9-nonanedithiol",
    "tris(2-ethylhexyl) phosphate": "tris(2-ethylhexyl) phosphate",
}


@dataclass
class SampleRecord:
    sample_id: str
    group: str
    source: str
    path: Path
    true_raw_labels: tuple[str, ...]
    true_known_labels: tuple[str, ...]


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


def baseline_als(y: np.ndarray) -> np.ndarray:
    length = len(y)
    penalty = baseline_penalty(length)
    weights = np.ones(length, dtype=np.float64)
    for _ in range(NITER):
        system = sparse.diags(weights, offsets=0, format="csc") + penalty
        baseline = spsolve(system, weights * y)
        weights = np.where(y > baseline, P, 1 - P)
    return baseline


@lru_cache(maxsize=None)
def baseline_penalty(length: int):
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    return (LAM * (diff.T @ diff)).tocsc()


def baseline_correct_single(spectrum: np.ndarray) -> np.ndarray:
    corrected = spectrum.astype(np.float64) - baseline_als(spectrum.astype(np.float64))
    return corrected


def preprocess_raman_from_corrected(corrected: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(corrected)
    if norm > 0:
        corrected = corrected / norm
    return np.abs(corrected)


def preprocess_fft_from_corrected(corrected: np.ndarray) -> np.ndarray:
    mag = np.abs(np.fft.rfft(corrected))
    mag = np.log1p(mag)
    norm = np.linalg.norm(mag)
    if norm > 0:
        mag = mag / norm
    return mag


def floatify_cols(df: pd.DataFrame) -> None:
    converted = []
    for col in df.columns:
        if col in {"Label", "Label 1", "Label 2"}:
            converted.append(col)
        else:
            converted.append(float(col))
    df.columns = converted


def canonicalize_label(label: str) -> str:
    cleaned = label.strip().lower()
    cleaned = cleaned.replace(" - map 3", "")
    cleaned = cleaned.replace(" - map3", "")
    cleaned = cleaned.replace(" map1", "")
    cleaned = cleaned.replace(" map2", "")
    cleaned = cleaned.replace(" map3", "")
    return LABEL_ALIASES.get(cleaned, cleaned)


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_df = pd.read_csv(NOTEBOOK_DIR / "reference_v2.csv")
    floatify_cols(ref_df)
    classes = sorted(ref_df["Label"].unique())
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    wav_axis = np.array([c for c in ref_df.columns if c != "Label"], dtype=float)

    input_len_raman = len(wav_axis)
    input_len_fft = input_len_raman // 2 + 1

    siamese_raman = SiameseNet(input_len=input_len_raman).to(device)
    siamese_fft = SiameseNet(input_len=input_len_fft).to(device)
    model = PresenceNetLogits(d_input=128, n_classes=len(classes)).to(device)

    siamese_raman.load_state_dict(
        torch.load(NOTEBOOK_DIR / "siamese_mixture.pth", map_location=device)
    )
    siamese_fft.load_state_dict(
        torch.load(NOTEBOOK_DIR / "siamese_mixture_fft.pth", map_location=device)
    )
    checkpoint = torch.load(NOTEBOOK_DIR / "presence_net_logits.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    siamese_raman.eval()
    siamese_fft.eval()
    model.eval()

    threshold_dict = json.loads((NOTEBOOK_DIR / "calibrated_thresholds.json").read_text())
    calibrated = np.array([threshold_dict[label] for label in classes], dtype=np.float32)
    default = np.full(len(classes), 0.5, dtype=np.float32)

    return {
        "device": device,
        "classes": classes,
        "class_to_i": class_to_i,
        "wav_axis": wav_axis,
        "siamese_raman": siamese_raman,
        "siamese_fft": siamese_fft,
        "model": model,
        "default_thresholds": default,
        "calibrated_thresholds": calibrated,
    }


def load_txt_spectrum(path: Path, wav_axis: np.ndarray) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected txt format in {path}")
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


def build_known_target(
    labels: tuple[str, ...], class_to_i: dict[str, int], n_classes: int
) -> np.ndarray:
    target = np.zeros(n_classes, dtype=int)
    for label in labels:
        if label in class_to_i:
            target[class_to_i[label]] = 1
    return target


def extract_embeddings(spectra: np.ndarray, resources: dict) -> np.ndarray:
    raman = []
    fft = []
    for spectrum in spectra:
        corrected = baseline_correct_single(spectrum)
        raman.append(preprocess_raman_from_corrected(corrected))
        fft.append(preprocess_fft_from_corrected(corrected))
    raman = np.vstack(raman)
    fft = np.vstack(fft)

    dataset = TensorDataset(
        torch.tensor(raman, dtype=torch.float32),
        torch.tensor(fft, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings = []
    device = resources["device"]
    with torch.no_grad():
        for xb_r, xb_f in loader:
            xb_r = xb_r.to(device).unsqueeze(1)
            xb_f = xb_f.to(device).unsqueeze(1)
            emb_r = resources["siamese_raman"](xb_r)
            emb_f = resources["siamese_fft"](xb_f)
            embeddings.append(torch.cat([emb_r, emb_f], dim=1).cpu().numpy())
    return np.vstack(embeddings)


def predict_probabilities(embeddings: np.ndarray, resources: dict) -> np.ndarray:
    device = resources["device"]
    with torch.no_grad():
        logits = resources["model"](
            torch.tensor(embeddings, dtype=torch.float32, device=device)
        )
    return torch.sigmoid(logits).cpu().numpy()


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


def summarize_records(
    name: str,
    indices: list[int],
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    pred = (probs[indices] >= thresholds).astype(int)
    truth = y_true[indices]
    summary = {"samples": len(indices)}
    summary.update(compute_metrics(truth, pred))
    return {name: summary}


def parse_mix_manifest(path: Path) -> dict[str, tuple[str, str]]:
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.lower().startswith("mix "):
            continue
        lhs, rhs = line.split(" - ", 1)
        parts = tuple(canonicalize_label(part) for part in rhs.split(" + "))
        if len(parts) == 2:
            mapping[lhs] = parts
    return mapping


def build_pt2_records(classes: list[str]) -> list[SampleRecord]:
    manifest = parse_mix_manifest(PT2_DIR / "Mixtures.txt")
    known = set(classes)
    records: list[SampleRecord] = []

    for mix_name, labels in sorted(manifest.items()):
        txt_dir = PT2_DIR / mix_name / "txt"
        for txt_path in sorted(txt_dir.glob("*.txt")):
            known_labels = tuple(label for label in labels if label in known)
            records.append(
                SampleRecord(
                    sample_id=f"{mix_name}/{txt_path.name}",
                    group="mixture",
                    source=mix_name,
                    path=txt_path,
                    true_raw_labels=labels,
                    true_known_labels=known_labels,
                )
            )

    mix_dirs = set(manifest)
    for pure_dir in sorted(p for p in PT2_DIR.iterdir() if p.is_dir() and p.name not in mix_dirs):
        txt_dir = pure_dir / "txt"
        if not txt_dir.exists():
            continue
        raw_label = canonicalize_label(pure_dir.name)
        known_labels = (raw_label,) if raw_label in known else ()
        for txt_path in sorted(txt_dir.glob("*.txt")):
            records.append(
                SampleRecord(
                    sample_id=f"{pure_dir.name}/{txt_path.name}",
                    group="pure",
                    source=pure_dir.name,
                    path=txt_path,
                    true_raw_labels=(raw_label,),
                    true_known_labels=known_labels,
                )
            )
    return records


def analyze_pt2(records: list[SampleRecord], probs: np.ndarray, resources: dict) -> dict:
    class_to_i = resources["class_to_i"]
    n_classes = len(resources["classes"])
    y_true = np.vstack(
        [build_known_target(rec.true_known_labels, class_to_i, n_classes) for rec in records]
    )

    results = {}
    for threshold_name in ("default_thresholds", "calibrated_thresholds"):
        thresholds = resources[threshold_name]
        pred = (probs >= thresholds).astype(int)
        key = threshold_name.replace("_thresholds", "")

        all_mix = [i for i, rec in enumerate(records) if rec.group == "mixture"]
        all_pure = [i for i, rec in enumerate(records) if rec.group == "pure"]
        true0 = [i for i, rec in enumerate(records) if len(rec.true_known_labels) == 0]
        true1 = [i for i, rec in enumerate(records) if len(rec.true_known_labels) == 1]
        true2 = [i for i, rec in enumerate(records) if len(rec.true_known_labels) == 2]

        summary = {}
        summary.update(summarize_records("all_mixtures", all_mix, y_true, probs, thresholds))
        summary.update(summarize_records("all_pure", all_pure, y_true, probs, thresholds))
        summary.update(summarize_records("known_labels_0", true0, y_true, probs, thresholds))
        summary.update(summarize_records("known_labels_1", true1, y_true, probs, thresholds))
        summary.update(summarize_records("known_labels_2", true2, y_true, probs, thresholds))

        per_source = []
        for source in sorted({rec.source for rec in records if rec.group == "mixture"}):
            idx = [i for i, rec in enumerate(records) if rec.source == source]
            metrics = compute_metrics(y_true[idx], pred[idx])
            metrics["source"] = source
            metrics["samples"] = len(idx)
            metrics["true_labels"] = " + ".join(records[idx[0]].true_raw_labels)
            per_source.append(metrics)
        summary["per_mixture"] = per_source

        ood_pure = []
        for source in sorted({rec.source for rec in records if rec.group == "pure"}):
            idx = [i for i, rec in enumerate(records) if rec.source == source]
            metrics = compute_metrics(y_true[idx], pred[idx])
            metrics["source"] = source
            metrics["samples"] = len(idx)
            ood_pure.append(metrics)
        summary["per_pure"] = ood_pure
        results[key] = summary

    pred_default = (probs >= resources["default_thresholds"]).astype(int)
    top_rows = []
    for rec, row_probs, row_pred in zip(records, probs, pred_default):
        ranked = np.argsort(row_probs)[::-1][:3]
        top_rows.append(
            {
                "sample_id": rec.sample_id,
                "group": rec.group,
                "source": rec.source,
                "true_raw_labels": " + ".join(rec.true_raw_labels),
                "true_known_labels": " + ".join(rec.true_known_labels),
                "predicted_labels_default": " + ".join(
                    resources["classes"][i] for i in np.where(row_pred == 1)[0]
                ),
                "top1_label": resources["classes"][ranked[0]],
                "top1_prob": float(row_probs[ranked[0]]),
                "top2_label": resources["classes"][ranked[1]],
                "top2_prob": float(row_probs[ranked[1]]),
                "top3_label": resources["classes"][ranked[2]],
                "top3_prob": float(row_probs[ranked[2]]),
            }
        )
    return results, pd.DataFrame(top_rows)


def evaluate_existing_set(resources: dict) -> dict:
    mix_df = pd.read_csv(NOTEBOOK_DIR / "mixtures_dataset.csv")
    floatify_cols(mix_df)
    wav_cols = [c for c in mix_df.columns if c not in {"Label 1", "Label 2"}]
    spectra = mix_df[wav_cols].to_numpy(dtype=float)
    embeddings = extract_embeddings(spectra, resources)
    probs = predict_probabilities(embeddings, resources)

    y_true = np.zeros((len(mix_df), len(resources["classes"])), dtype=int)
    for idx, row in mix_df.iterrows():
        for key in ("Label 1", "Label 2"):
            label = row[key]
            y_true[idx, resources["class_to_i"][label]] = 1

    return {
        "default": compute_metrics(y_true, (probs >= resources["default_thresholds"]).astype(int)),
        "calibrated": compute_metrics(
            y_true, (probs >= resources["calibrated_thresholds"]).astype(int)
        ),
    }


def print_summary(existing: dict, pt2_results: dict) -> None:
    print("Existing in-repo mixture benchmark")
    for mode, metrics in existing.items():
        print(
            f"  {mode:10s} exact={metrics['exact_match']:.3f} "
            f"micro_f1={metrics['micro_f1']:.3f} "
            f"pred_labels/sample={metrics['mean_predicted_labels']:.2f}"
        )

    print("\nPT2 mixture generalization")
    for mode, summary in pt2_results.items():
        print(f"  Thresholds: {mode}")
        for bucket in ("all_mixtures", "known_labels_2", "known_labels_1", "known_labels_0"):
            stats = summary[bucket]
            print(
                f"    {bucket:14s} n={stats['samples']:3d} "
                f"exact={stats['exact_match']:.3f} "
                f"micro_f1={stats['micro_f1']:.3f} "
                f"pred/sample={stats['mean_predicted_labels']:.2f} "
                f"zero_pred={stats['zero_prediction_rate']:.3f}"
            )

        print("    Per mixture")
        for row in summary["per_mixture"]:
            print(
                f"      {row['source']:6s} {row['true_labels']:<45s} "
                f"exact={row['exact_match']:.3f} "
                f"micro_f1={row['micro_f1']:.3f} "
                f"pred/sample={row['mean_predicted_labels']:.2f}"
            )

        print("    New pure compounds")
        for row in summary["per_pure"]:
            print(
                f"      {row['source']:<14s} exact={row['exact_match']:.3f} "
                f"pred/sample={row['mean_predicted_labels']:.2f} "
                f"zero_pred={row['zero_prediction_rate']:.3f}"
            )


def main() -> None:
    if not PT2_DIR.exists():
        raise FileNotFoundError(f"Expected extracted dataset at {PT2_DIR}")

    resources = load_models()
    existing = evaluate_existing_set(resources)

    records = build_pt2_records(resources["classes"])
    spectra = np.vstack([load_txt_spectrum(rec.path, resources["wav_axis"]) for rec in records])
    embeddings = extract_embeddings(spectra, resources)
    probs = predict_probabilities(embeddings, resources)

    pt2_results, predictions_df = analyze_pt2(records, probs, resources)
    print_summary(existing, pt2_results)

    predictions_path = OUTPUT_DIR / "pt2_generalization_predictions.csv"
    summary_path = OUTPUT_DIR / "pt2_generalization_summary.json"
    predictions_df.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps({"existing": existing, "pt2": pt2_results}, indent=2))

    print(f"\nSaved predictions to {predictions_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
