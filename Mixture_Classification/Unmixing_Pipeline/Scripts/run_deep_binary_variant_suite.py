from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from unmixing_common import (
    RESULTS_ROOT,
    SpectrumRecord,
    build_compound_atom_sets,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
    preprocess_spectrum,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "deep_binary_variant_suite"
)
BASELINE_RESULTS_DIR = RESULTS_ROOT / "deep_binary_coefficient_regressor"
PAIR_BENCHMARK_RESULTS_DIR = RESULTS_ROOT / "pair_nnls_replicate_dictionary"

SEED = 42
RATIOS = np.arange(0.1, 1.0, 0.1)
N_PER_RATIO = 10
NOISE_SCALE = 0.01
GLOBAL_SCALE_RANGE = (0.85, 1.15)
BATCH_SIZE = 256
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SUPPORT_LOSS_WEIGHT = 0.35
RECON_LOSS_WEIGHT = 0.15
N_EXTRA_REPS = 9


@dataclass
class SyntheticDataset:
    spectra: np.ndarray
    coefficients: np.ndarray
    support: np.ndarray
    pair_ids: np.ndarray


@dataclass
class VariantSpec:
    name: str
    model_type: str
    decoder_type: str
    decoder: np.ndarray
    output_to_compound: np.ndarray
    classes: list[str]
    build_model: callable


class CNNClassRegressor(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(self.features(x.unsqueeze(1)))
        positive = F.softplus(logits)
        shares = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return logits, shares


class MLPAtomRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, n_atoms: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.support_head = nn.Linear(hidden_dim, n_classes)
        self.atom_head = nn.Linear(hidden_dim, n_atoms)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        support_logits = self.support_head(hidden)
        atom_logits = self.atom_head(hidden)
        positive = F.softplus(atom_logits)
        atom_shares = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return support_logits, atom_shares


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    spectrum = spectrum.astype(np.float32, copy=False)
    norm = np.linalg.norm(spectrum)
    if norm <= 0:
        return spectrum.copy()
    return spectrum / norm


def normalize_dictionary(dictionary: np.ndarray) -> np.ndarray:
    normalized = dictionary.astype(np.float32).copy()
    norms = np.linalg.norm(normalized, axis=0, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return normalized / norms


def build_synthetic_dataset(ref_df: pd.DataFrame, classes: list[str], mode: str) -> SyntheticDataset:
    wav_cols = [c for c in ref_df.columns if c != "Label"]
    raw_by_label = {
        label: ref_df.loc[ref_df["Label"] == label, wav_cols].to_numpy(dtype=np.float64)
        for label in classes
    }
    class_to_i = {label: idx for idx, label in enumerate(classes)}

    spectra = []
    coefficients = []
    support = []
    pair_ids = []
    for left_idx, left_label in enumerate(classes[:-1]):
        for right_label in classes[left_idx + 1 :]:
            left_specs = raw_by_label[left_label]
            right_specs = raw_by_label[right_label]
            pair_id = f"{left_label}|{right_label}"
            for ratio in RATIOS:
                for _ in range(N_PER_RATIO):
                    left_raw = left_specs[np.random.randint(len(left_specs))]
                    right_raw = right_specs[np.random.randint(len(right_specs))]
                    mixed = ratio * left_raw + (1.0 - ratio) * right_raw
                    mixed *= np.random.uniform(*GLOBAL_SCALE_RANGE)
                    mixed += np.random.normal(scale=NOISE_SCALE * np.std(mixed), size=mixed.shape)
                    proc = preprocess_spectrum(mixed, mode)
                    spectra.append(normalize_spectrum(proc))

                    coef = np.zeros(len(classes), dtype=np.float32)
                    coef[class_to_i[left_label]] = float(ratio)
                    coef[class_to_i[right_label]] = float(1.0 - ratio)
                    coefficients.append(coef)
                    support.append((coef > 0).astype(np.float32))
                    pair_ids.append(pair_id)

    return SyntheticDataset(
        spectra=np.vstack(spectra).astype(np.float32),
        coefficients=np.vstack(coefficients).astype(np.float32),
        support=np.vstack(support).astype(np.float32),
        pair_ids=np.array(pair_ids),
    )


def split_synthetic_dataset(dataset: SyntheticDataset) -> dict[str, np.ndarray]:
    indices = np.arange(len(dataset.spectra))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED,
        stratify=dataset.pair_ids,
    )
    val_fraction = 0.10 / 0.90
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_fraction,
        random_state=SEED,
        stratify=dataset.pair_ids[train_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_loader(
    spectra: np.ndarray,
    coefficients: np.ndarray,
    support: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(spectra, dtype=torch.float32),
        torch.tensor(coefficients, dtype=torch.float32),
        torch.tensor(support, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def support_from_top2(shares: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(shares, dtype=int)
    top2 = np.argsort(shares, axis=1)[:, -2:]
    rows = np.arange(len(shares))[:, None]
    pred[rows, top2] = 1
    return pred


def build_variant_specs(ref_df: pd.DataFrame, classes: list[str], mode: str) -> list[VariantSpec]:
    identity = np.eye(len(classes), dtype=np.float32)
    _, mean_dictionary = build_mean_dictionary(ref_df, mode)
    mean_decoder = normalize_dictionary(mean_dictionary).T

    atom_sets = build_compound_atom_sets(ref_df, mode, n_extra_reps=N_EXTRA_REPS)
    atom_blocks = []
    output_to_compound = []
    for class_idx, label in enumerate(classes):
        atoms = atom_sets[label]
        atom_blocks.append(atoms)
        mapping = np.zeros((atoms.shape[1], len(classes)), dtype=np.float32)
        mapping[:, class_idx] = 1.0
        output_to_compound.append(mapping)
    atom_dictionary = normalize_dictionary(np.column_stack(atom_blocks)).T
    atom_to_compound = np.vstack(output_to_compound)

    return [
        VariantSpec(
            name="cnn_encoder",
            model_type="cnn_nonnegative_coefficient_regressor",
            decoder_type="class_mean_decoder",
            decoder=mean_decoder,
            output_to_compound=identity,
            classes=classes,
            build_model=lambda input_dim, n_classes=len(classes): CNNClassRegressor(input_dim, n_classes),
        ),
        VariantSpec(
            name="replicate_decoder",
            model_type="mlp_replicate_decoder_regressor",
            decoder_type="replicate_atom_decoder",
            decoder=atom_dictionary,
            output_to_compound=atom_to_compound,
            classes=classes,
            build_model=lambda input_dim, n_classes=len(classes), n_atoms=atom_dictionary.shape[0]: MLPAtomRegressor(
                input_dim, 256, n_classes, n_atoms
            ),
        ),
    ]


def train_variant(
    dataset: SyntheticDataset,
    split_idx: dict[str, np.ndarray],
    variant: VariantSpec,
    device: torch.device,
):
    decoder_tensor = torch.tensor(variant.decoder, dtype=torch.float32, device=device)
    output_to_compound_tensor = torch.tensor(
        variant.output_to_compound, dtype=torch.float32, device=device
    )
    train_loader = build_loader(
        dataset.spectra[split_idx["train"]],
        dataset.coefficients[split_idx["train"]],
        dataset.support[split_idx["train"]],
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=device,
    )
    val_loader = build_loader(
        dataset.spectra[split_idx["val"]],
        dataset.coefficients[split_idx["val"]],
        dataset.support[split_idx["val"]],
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    pos = dataset.support[split_idx["train"]].sum(axis=0)
    neg = len(split_idx["train"]) - pos
    pos_weight = torch.tensor((neg / pos).clip(min=1.0), dtype=torch.float32, device=device)

    model = variant.build_model(dataset.spectra.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    support_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for xb, coef_target, support_target in train_loader:
            xb = xb.to(device, non_blocking=True)
            coef_target = coef_target.to(device, non_blocking=True)
            support_target = support_target.to(device, non_blocking=True)

            support_logits, output_shares = model(xb)
            compound_shares = output_shares @ output_to_compound_tensor
            recon = output_shares @ decoder_tensor
            coeff_loss = F.mse_loss(compound_shares, coef_target)
            support_loss = support_criterion(support_logits, support_target)
            recon_loss = F.mse_loss(recon, xb)
            loss = coeff_loss + SUPPORT_LOSS_WEIGHT * support_loss + RECON_LOSS_WEIGHT * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            train_loss += loss.item() * batch_size
            train_items += batch_size

        train_loss /= max(train_items, 1)

        model.eval()
        val_loss = 0.0
        val_items = 0
        val_compound_shares = []
        val_support = []
        with torch.no_grad():
            for xb, coef_target, support_target in val_loader:
                xb = xb.to(device, non_blocking=True)
                coef_target = coef_target.to(device, non_blocking=True)
                support_target = support_target.to(device, non_blocking=True)

                support_logits, output_shares = model(xb)
                compound_shares = output_shares @ output_to_compound_tensor
                recon = output_shares @ decoder_tensor
                coeff_loss = F.mse_loss(compound_shares, coef_target)
                support_loss = support_criterion(support_logits, support_target)
                recon_loss = F.mse_loss(recon, xb)
                loss = coeff_loss + SUPPORT_LOSS_WEIGHT * support_loss + RECON_LOSS_WEIGHT * recon_loss

                batch_size = xb.size(0)
                val_loss += loss.item() * batch_size
                val_items += batch_size
                val_compound_shares.append(compound_shares.cpu().numpy())
                val_support.append(support_target.cpu().numpy().astype(int))

        val_loss /= max(val_items, 1)
        val_compound_shares_np = np.vstack(val_compound_shares)
        val_support_np = np.vstack(val_support)
        val_pred = support_from_top2(val_compound_shares_np)
        val_metrics = compute_metrics(val_support_np, val_pred)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_exact_match": float(val_metrics["exact_match"]),
                "val_micro_f1": float(val_metrics["micro_f1"]),
            }
        )

        score = (val_metrics["exact_match"], val_metrics["micro_f1"], -val_loss)
        if best is None or score > best:
            best = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    {variant.name} epoch {epoch:03d}/{EPOCHS:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_exact={val_metrics['exact_match']:.3f} "
                f"val_micro_f1={val_metrics['micro_f1']:.3f}"
            )

        if epochs_without_improvement >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def predict_compound_shares(
    model: nn.Module,
    spectra: np.ndarray,
    output_to_compound: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    output_to_compound_tensor = torch.tensor(
        output_to_compound, dtype=torch.float32, device=device
    )
    dataset = TensorDataset(torch.tensor(spectra, dtype=torch.float32))
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
            _, output_shares = model(xb)
            compound_shares = output_shares @ output_to_compound_tensor
            outputs.append(compound_shares.cpu().numpy())
    return np.vstack(outputs)


def summarize_synthetic_split(
    dataset: SyntheticDataset,
    split_name: str,
    split_idx: dict[str, np.ndarray],
    model: nn.Module,
    variant: VariantSpec,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    idx = split_idx[split_name]
    compound_shares = predict_compound_shares(
        model, dataset.spectra[idx], variant.output_to_compound, device
    )
    y_true = dataset.support[idx].astype(int)
    y_pred = support_from_top2(compound_shares)
    metrics = compute_metrics(y_true, y_pred)

    rows = []
    for sample_i, coeff_true, share_pred, pair_id in zip(
        idx, dataset.coefficients[idx], compound_shares, dataset.pair_ids[idx]
    ):
        ranked = np.argsort(share_pred)[::-1][:5]
        rows.append(
            {
                "sample_index": int(sample_i),
                "split": split_name,
                "pair_id": pair_id,
                "true_labels": " + ".join(variant.classes[j] for j in np.where(coeff_true > 0)[0]),
                "predicted_labels": " + ".join(variant.classes[j] for j in ranked[:2]),
                "top1_label": variant.classes[ranked[0]],
                "top1_share": float(share_pred[ranked[0]]),
                "top2_label": variant.classes[ranked[1]],
                "top2_share": float(share_pred[ranked[1]]),
                "top3_label": variant.classes[ranked[2]],
                "top3_share": float(share_pred[ranked[2]]),
                "top4_label": variant.classes[ranked[3]],
                "top4_share": float(share_pred[ranked[3]]),
                "top5_label": variant.classes[ranked[4]],
                "top5_share": float(share_pred[ranked[4]]),
            }
        )
    return metrics, pd.DataFrame(rows)


def summarize_real_records(
    records: list[SpectrumRecord],
    variant: VariantSpec,
    model: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    spectra = np.vstack([normalize_spectrum(record.spectrum) for record in records]).astype(np.float32)
    compound_shares = predict_compound_shares(
        model, spectra, variant.output_to_compound, device
    )
    y_pred = support_from_top2(compound_shares)

    class_to_i = {label: idx for idx, label in enumerate(variant.classes)}
    y_true = np.zeros((len(records), len(variant.classes)), dtype=int)
    rows = []
    for idx, (record, share_pred) in enumerate(zip(records, compound_shares)):
        for label in record.true_labels:
            y_true[idx, class_to_i[label]] = 1
        ranked = np.argsort(share_pred)[::-1][:5]
        pred_idx = np.where(y_pred[idx] == 1)[0]
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "source": record.source,
                "true_labels": " + ".join(record.true_labels),
                "predicted_labels": " + ".join(variant.classes[j] for j in pred_idx),
                "top1_label": variant.classes[ranked[0]],
                "top1_share": float(share_pred[ranked[0]]),
                "top2_label": variant.classes[ranked[1]],
                "top2_share": float(share_pred[ranked[1]]),
                "top3_label": variant.classes[ranked[2]],
                "top3_share": float(share_pred[ranked[2]]),
                "top4_label": variant.classes[ranked[3]],
                "top4_share": float(share_pred[ranked[3]]),
                "top5_label": variant.classes[ranked[4]],
                "top5_share": float(share_pred[ranked[4]]),
                "true_top_share_sum": float(sum(share_pred[class_to_i[label]] for label in record.true_labels)),
            }
        )
    return compute_metrics(y_true, y_pred), pd.DataFrame(rows), y_pred


def per_source_summary(
    records: list[SpectrumRecord],
    classes: list[str],
    y_pred: np.ndarray,
) -> list[dict]:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(records), len(classes)), dtype=int)
    for idx, record in enumerate(records):
        for label in record.true_labels:
            y_true[idx, class_to_i[label]] = 1

    summaries = []
    for source in sorted({record.source for record in records if record.dataset == "pt2_real"}):
        idx = [i for i, record in enumerate(records) if record.source == source]
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        metrics["source"] = source
        metrics["samples"] = len(idx)
        metrics["true_labels"] = " + ".join(records[idx[0]].true_labels)
        summaries.append(metrics)
    return summaries


def run_variant_mode(
    expanded_ref: pd.DataFrame,
    wav_axis: np.ndarray,
    dataset: SyntheticDataset,
    split_idx: dict[str, np.ndarray],
    variant: VariantSpec,
    mode: str,
    device: torch.device,
) -> dict:
    print(f"\nRunning {variant.name} with mode={mode}")
    model, history_df = train_variant(dataset, split_idx, variant, device)

    train_metrics, train_pred_df = summarize_synthetic_split(
        dataset, "train", split_idx, model, variant, device
    )
    val_metrics, val_pred_df = summarize_synthetic_split(
        dataset, "val", split_idx, model, variant, device
    )
    test_metrics, test_pred_df = summarize_synthetic_split(
        dataset, "test", split_idx, model, variant, device
    )

    existing_records = load_existing_real_records(mode)
    pt2_records = load_pt2_mixture_records(wav_axis, mode)
    existing_metrics, existing_pred_df, _ = summarize_real_records(
        existing_records, variant, model, device
    )
    pt2_metrics, pt2_pred_df, pt2_y_pred = summarize_real_records(
        pt2_records, variant, model, device
    )

    mode_dir = RESULTS_DIR / variant.name / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(mode_dir / "training_history.csv", index=False)
    train_pred_df.to_csv(mode_dir / "synthetic_train_predictions.csv", index=False)
    val_pred_df.to_csv(mode_dir / "synthetic_val_predictions.csv", index=False)
    test_pred_df.to_csv(mode_dir / "synthetic_test_predictions.csv", index=False)
    existing_pred_df.to_csv(mode_dir / "existing_real_predictions.csv", index=False)
    pt2_pred_df.to_csv(mode_dir / "pt2_real_predictions.csv", index=False)

    summary = {
        "variant": variant.name,
        "mode": mode,
        "model_type": variant.model_type,
        "decoder_type": variant.decoder_type,
        "device": device.type,
        "baseline_results_dir": str(BASELINE_RESULTS_DIR),
        "pair_benchmark_results_dir": str(PAIR_BENCHMARK_RESULTS_DIR),
        "synthetic_generation": {
            "ratios": [float(x) for x in RATIOS],
            "n_per_ratio": N_PER_RATIO,
            "noise_scale": NOISE_SCALE,
            "global_scale_range": [float(GLOBAL_SCALE_RANGE[0]), float(GLOBAL_SCALE_RANGE[1])],
        },
        "training": {
            "epochs_max": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "support_loss_weight": SUPPORT_LOSS_WEIGHT,
            "recon_loss_weight": RECON_LOSS_WEIGHT,
        },
        "decoder_shape": {
            "output_units": int(variant.decoder.shape[0]),
            "spectrum_dim": int(variant.decoder.shape[1]),
            "compound_classes": int(len(variant.classes)),
        },
        "synthetic": {
            "samples_total": int(len(dataset.spectra)),
            "train_samples": int(len(split_idx["train"])),
            "val_samples": int(len(split_idx["val"])),
            "test_samples": int(len(split_idx["test"])),
            "train_top2_binary": train_metrics,
            "val_top2_binary": val_metrics,
            "test_top2_binary": test_metrics,
        },
        "existing_real_top2_binary": existing_metrics,
        "pt2_real_top2_binary": {
            "overall": pt2_metrics,
            "per_mixture": per_source_summary(pt2_records, variant.classes, pt2_y_pred),
        },
    }
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"  synthetic test exact={test_metrics['exact_match']:.3f} "
        f"micro_f1={test_metrics['micro_f1']:.3f}"
    )
    print(
        f"  existing_real exact={existing_metrics['exact_match']:.3f} "
        f"micro_f1={existing_metrics['micro_f1']:.3f}"
    )
    print(
        f"  pt2_real      exact={pt2_metrics['exact_match']:.3f} "
        f"micro_f1={pt2_metrics['micro_f1']:.3f}"
    )

    return summary


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes = sorted(expanded_ref["Label"].astype(str).unique())

    all_results = {}
    for mode in ("raw", "baseline_corrected"):
        dataset = build_synthetic_dataset(expanded_ref, classes, mode)
        split_idx = split_synthetic_dataset(dataset)
        variants = build_variant_specs(expanded_ref, classes, mode)

        mode_results = {}
        for variant in variants:
            mode_results[variant.name] = run_variant_mode(
                expanded_ref, wav_axis, dataset, split_idx, variant, mode, device
            )
        all_results[mode] = mode_results

    (RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
