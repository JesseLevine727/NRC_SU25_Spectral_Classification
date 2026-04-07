from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.sparse.linalg import spsolve
from sklearn.metrics import precision_recall_fscore_support


LAM, P, NITER = 1e4, 0.01, 10


MIXTURE_ROOT = Path(__file__).resolve().parents[2]
UNMIXING_ROOT = MIXTURE_ROOT / "Unmixing_Pipeline"
DATA_ROOT = UNMIXING_ROOT / "Data"
REFERENCE_DATA_DIR = DATA_ROOT / "reference"
PT2_DIR = DATA_ROOT / "pt2"
RESULTS_ROOT = UNMIXING_ROOT / "Results"


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
class SpectrumRecord:
    sample_id: str
    dataset: str
    source: str
    true_labels: tuple[str, ...]
    spectrum: np.ndarray


@lru_cache(maxsize=None)
def baseline_penalty(length: int):
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    return (LAM * (diff.T @ diff)).tocsc()


def baseline_als(y: np.ndarray) -> np.ndarray:
    signal = y.astype(np.float64)
    weights = np.ones(len(signal), dtype=np.float64)
    penalty = baseline_penalty(len(signal))
    for _ in range(NITER):
        system = sparse.diags(weights, offsets=0, format="csc") + penalty
        baseline = spsolve(system, weights * signal)
        weights = np.where(signal > baseline, P, 1 - P)
    return baseline


def preprocess_spectrum(spectrum: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return spectrum.astype(np.float64)
    if mode == "baseline_corrected":
        return spectrum.astype(np.float64) - baseline_als(spectrum)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def floatify_cols(df: pd.DataFrame) -> None:
    converted = []
    for col in df.columns:
        if col in {"Label", "Label 1", "Label 2"}:
            converted.append(col)
        else:
            converted.append(float(col))
    df.columns = converted


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
    ref_df = pd.read_csv(REFERENCE_DATA_DIR / "reference_v2.csv")
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


def build_mean_dictionary(ref_df: pd.DataFrame, mode: str) -> tuple[list[str], np.ndarray]:
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    classes = sorted(ref_df["Label"].unique())
    atoms = []
    for label in classes:
        spectra = ref_df.loc[ref_df["Label"] == label, wav_cols].to_numpy(dtype=np.float64)
        proc = np.vstack([preprocess_spectrum(spec, mode) for spec in spectra])
        atoms.append(proc.mean(axis=0))
    dictionary = np.column_stack(atoms)
    return classes, dictionary


def constant_baseline_atom(length: int) -> np.ndarray:
    atom = np.ones(length, dtype=np.float64)
    atom /= np.linalg.norm(atom)
    return atom[:, None]


def farthest_point_subset(spectra: np.ndarray, n_select: int) -> np.ndarray:
    if n_select <= 0 or len(spectra) == 0:
        return np.empty((0, spectra.shape[1]), dtype=np.float64)
    if len(spectra) <= n_select:
        return spectra.astype(np.float64)

    mean_spec = spectra.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(spectra - mean_spec, axis=1)))]
    min_dist = np.linalg.norm(spectra - spectra[selected[0]], axis=1)

    while len(selected) < n_select:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = np.linalg.norm(spectra - spectra[idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    selected = sorted(set(selected))
    return spectra[selected].astype(np.float64)


def build_compound_atom_sets(ref_df: pd.DataFrame, mode: str, n_extra_reps: int) -> dict[str, np.ndarray]:
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    atom_sets: dict[str, np.ndarray] = {}
    for label in sorted(ref_df["Label"].unique()):
        spectra = ref_df.loc[ref_df["Label"] == label, wav_cols].to_numpy(dtype=np.float64)
        proc = np.vstack([preprocess_spectrum(spec, mode) for spec in spectra])
        mean_atom = proc.mean(axis=0, keepdims=True)
        reps = farthest_point_subset(proc, n_extra_reps)
        atoms = np.vstack([mean_atom, reps])
        atom_sets[label] = atoms.T
    return atom_sets


def parse_pt2_manifest() -> dict[str, tuple[str, str]]:
    mapping = {}
    for line in (PT2_DIR / "Mixtures.txt").read_text().splitlines():
        line = line.strip()
        if not line.startswith("Mix "):
            continue
        mix_name, rhs = line.split(" - ", 1)
        left, right = rhs.split(" + ")
        left = MIXTURE_LABEL_ALIASES.get(left.strip().lower(), left.strip().lower())
        right = MIXTURE_LABEL_ALIASES.get(right.strip().lower(), right.strip().lower())
        mapping[mix_name] = tuple(sorted((left, right)))
    return mapping


def load_existing_real_records(mode: str) -> list[SpectrumRecord]:
    mix_df = pd.read_csv(REFERENCE_DATA_DIR / "mixtures_dataset.csv")
    floatify_cols(mix_df)
    wav_cols = [c for c in mix_df.columns if c not in {"Label 1", "Label 2"}]
    records = []
    for idx, row in mix_df.iterrows():
        labels = tuple(sorted((row["Label 1"], row["Label 2"])))
        spectrum = preprocess_spectrum(row[wav_cols].to_numpy(dtype=np.float64), mode)
        records.append(
            SpectrumRecord(
                sample_id=f"existing/{idx}",
                dataset="existing_real",
                source="existing_real",
                true_labels=labels,
                spectrum=spectrum,
            )
        )
    return records


def load_pt2_mixture_records(wav_axis: np.ndarray, mode: str) -> list[SpectrumRecord]:
    manifest = parse_pt2_manifest()
    records = []
    for mix_name, labels in sorted(manifest.items()):
        for txt_path in sorted((PT2_DIR / mix_name / "txt").glob("*.txt")):
            spectrum = preprocess_spectrum(load_txt_spectrum(txt_path, wav_axis), mode)
            records.append(
                SpectrumRecord(
                    sample_id=f"{mix_name}/{txt_path.name}",
                    dataset="pt2_real",
                    source=mix_name,
                    true_labels=labels,
                    spectrum=spectrum,
                )
            )
    return records


def load_original_pure_records(mode: str) -> list[SpectrumRecord]:
    ref_df = pd.read_csv(REFERENCE_DATA_DIR / "reference_v2.csv")
    floatify_cols(ref_df)
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    records = []
    for idx, row in ref_df.iterrows():
        label = str(row["Label"])
        spectrum = preprocess_spectrum(row[wav_cols].to_numpy(dtype=np.float64), mode)
        records.append(
            SpectrumRecord(
                sample_id=f"orig_pure/{idx}",
                dataset="original_pure",
                source=label,
                true_labels=(label,),
                spectrum=spectrum,
            )
        )
    return records


def load_pt2_pure_records(wav_axis: np.ndarray, mode: str) -> list[SpectrumRecord]:
    records = []
    for pure_dir_name, label in PURE_LABEL_ALIASES.items():
        for txt_path in sorted((PT2_DIR / pure_dir_name / "txt").glob("*.txt")):
            spectrum = preprocess_spectrum(load_txt_spectrum(txt_path, wav_axis), mode)
            records.append(
                SpectrumRecord(
                    sample_id=f"{pure_dir_name}/{txt_path.name}",
                    dataset="pt2_pure",
                    source=pure_dir_name,
                    true_labels=(label,),
                    spectrum=spectrum,
                )
            )
    return records


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
    }


def evaluate_pair_records_with_atom_sets(
    records: list[SpectrumRecord],
    classes: list[str],
    atom_sets: dict[str, np.ndarray],
    baseline_atom: np.ndarray,
):
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    pair_defs = list(combinations(classes, 2))
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)
    rows = []

    for row_idx, record in enumerate(records):
        best = None
        for left_label, right_label in pair_defs:
            left_atoms = atom_sets[left_label]
            right_atoms = atom_sets[right_label]
            design = np.column_stack([left_atoms, right_atoms, baseline_atom])
            coef, _ = nnls(design, record.spectrum)
            recon = design @ coef
            residual = float(np.linalg.norm(record.spectrum - recon))
            if best is None or residual < best["residual"]:
                n_left = left_atoms.shape[1]
                n_right = right_atoms.shape[1]
                best = {
                    "labels": tuple(sorted((left_label, right_label))),
                    "left_sum": float(coef[:n_left].sum()),
                    "right_sum": float(coef[n_left : n_left + n_right].sum()),
                    "baseline_sum": float(coef[-1]),
                    "residual": residual,
                    "n_left_atoms": n_left,
                    "n_right_atoms": n_right,
                }

        assert best is not None
        pred_labels = best["labels"]
        for label in record.true_labels:
            y_true[row_idx, class_to_i[label]] = 1
        y_pred[row_idx, class_to_i[pred_labels[0]]] = 1
        y_pred[row_idx, class_to_i[pred_labels[1]]] = 1

        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(pred_labels),
                "residual_norm": best["residual"],
                "left_atom_coef_sum": best["left_sum"],
                "right_atom_coef_sum": best["right_sum"],
                "baseline_coef_sum": best["baseline_sum"],
                "left_atom_count": best["n_left_atoms"],
                "right_atom_count": best["n_right_atoms"],
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    pred_df = pd.DataFrame(rows)
    per_source = []
    for source in sorted({record.source for record in records if record.dataset == "pt2_real"}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        src_metrics = compute_metrics(y_true[idx], y_pred[idx])
        src_metrics["source"] = source
        src_metrics["samples"] = len(idx)
        src_metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        src_metrics["mean_residual_norm"] = float(pred_df.iloc[idx]["residual_norm"].mean())
        src_metrics["mean_baseline_coef_sum"] = float(pred_df.iloc[idx]["baseline_coef_sum"].mean())
        per_source.append(src_metrics)

    return metrics, per_source, pred_df
