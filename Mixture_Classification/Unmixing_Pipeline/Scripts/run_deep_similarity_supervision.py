from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_deep_binary_coefficient_regressor import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NOISE_SCALE,
    GLOBAL_SCALE_RANGE,
    N_PER_RATIO,
    PATIENCE,
    RATIOS,
    RECON_LOSS_WEIGHT,
    SUPPORT_LOSS_WEIGHT,
    WEIGHT_DECAY,
    CoefficientRegressor,
    build_synthetic_dataset,
    build_loader,
    normalize_dictionary,
    per_source_summary,
    set_seed,
    split_synthetic_dataset,
    summarize_real_records,
    summarize_synthetic_split,
)
from unmixing_common import (
    RESULTS_ROOT,
    build_expanded_reference,
    build_mean_dictionary,
    compute_metrics,
    load_existing_real_records,
    load_pt2_mixture_records,
    load_reference,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "Results" / "deep_similarity_supervision"
)
BASELINE_RESULTS_DIR = RESULTS_ROOT / "deep_binary_coefficient_regressor"

MARGIN = 0.08
MARGIN_LOSS_WEIGHT = 0.25
SIMILARITY_FALSE_LOSS_WEIGHT = 0.20


def build_similarity_tensor(decoder: np.ndarray, device: torch.device) -> torch.Tensor:
    similarity = decoder @ decoder.T
    similarity = np.clip(similarity, 0.0, None)
    np.fill_diagonal(similarity, 0.0)
    return torch.tensor(similarity, dtype=torch.float32, device=device)


def margin_ranking_loss(pred_shares: torch.Tensor, support_target: torch.Tensor) -> torch.Tensor:
    true_mask = support_target > 0.5
    false_mask = ~true_mask

    true_min = pred_shares.masked_fill(~true_mask, float("inf")).amin(dim=1)
    false_max = pred_shares.masked_fill(~false_mask, float("-inf")).amax(dim=1)
    return F.relu(MARGIN - (true_min - false_max)).mean()


def similarity_weighted_false_loss(
    pred_shares: torch.Tensor,
    support_target: torch.Tensor,
    similarity_tensor: torch.Tensor,
) -> torch.Tensor:
    n_true = support_target.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_similarity = (support_target @ similarity_tensor) / n_true
    false_weights = mean_similarity * (1.0 - support_target)
    return (pred_shares * false_weights).sum(dim=1).mean()


def train_model_with_similarity_supervision(
    dataset,
    split_idx: dict[str, np.ndarray],
    decoder: np.ndarray,
    device: torch.device,
):
    decoder_tensor = torch.tensor(decoder, dtype=torch.float32, device=device)
    similarity_tensor = build_similarity_tensor(decoder, device)

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

    model = CoefficientRegressor(dataset.spectra.shape[1], dataset.coefficients.shape[1]).to(device)
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

            support_logits, pred_shares = model(xb)
            recon = pred_shares @ decoder_tensor
            coeff_loss = F.mse_loss(pred_shares, coef_target)
            support_loss = support_criterion(support_logits, support_target)
            recon_loss = F.mse_loss(recon, xb)
            ranking_loss = margin_ranking_loss(pred_shares, support_target)
            weighted_false_loss = similarity_weighted_false_loss(
                pred_shares, support_target, similarity_tensor
            )
            loss = (
                coeff_loss
                + SUPPORT_LOSS_WEIGHT * support_loss
                + RECON_LOSS_WEIGHT * recon_loss
                + MARGIN_LOSS_WEIGHT * ranking_loss
                + SIMILARITY_FALSE_LOSS_WEIGHT * weighted_false_loss
            )

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
        val_shares = []
        val_support = []
        with torch.no_grad():
            for xb, coef_target, support_target in val_loader:
                xb = xb.to(device, non_blocking=True)
                coef_target = coef_target.to(device, non_blocking=True)
                support_target = support_target.to(device, non_blocking=True)

                support_logits, pred_shares = model(xb)
                recon = pred_shares @ decoder_tensor
                coeff_loss = F.mse_loss(pred_shares, coef_target)
                support_loss = support_criterion(support_logits, support_target)
                recon_loss = F.mse_loss(recon, xb)
                ranking_loss = margin_ranking_loss(pred_shares, support_target)
                weighted_false_loss = similarity_weighted_false_loss(
                    pred_shares, support_target, similarity_tensor
                )
                loss = (
                    coeff_loss
                    + SUPPORT_LOSS_WEIGHT * support_loss
                    + RECON_LOSS_WEIGHT * recon_loss
                    + MARGIN_LOSS_WEIGHT * ranking_loss
                    + SIMILARITY_FALSE_LOSS_WEIGHT * weighted_false_loss
                )

                batch_size = xb.size(0)
                val_loss += loss.item() * batch_size
                val_items += batch_size
                val_shares.append(pred_shares.cpu().numpy())
                val_support.append(support_target.cpu().numpy().astype(int))

        val_loss /= max(val_items, 1)
        val_shares_np = np.vstack(val_shares)
        val_support_np = np.vstack(val_support)
        val_pred = np.zeros_like(val_shares_np, dtype=int)
        top2 = np.argsort(val_shares_np, axis=1)[:, -2:]
        val_pred[np.arange(len(val_pred))[:, None], top2] = 1

        precision = compute_metrics(val_support_np, val_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_exact_match": float(precision["exact_match"]),
                "val_micro_f1": float(precision["micro_f1"]),
            }
        )

        score = (precision["exact_match"], precision["micro_f1"], -val_loss)
        if best is None or score > best:
            best = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    epoch {epoch:03d}/{EPOCHS:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_exact={precision['exact_match']:.3f} "
                f"val_micro_f1={precision['micro_f1']:.3f}"
            )

        if epochs_without_improvement >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def run_mode(mode: str, device: torch.device) -> dict:
    print(f"\nRunning deep similarity supervision with mode={mode}")
    ref_df, wav_axis = load_reference()
    expanded_ref = build_expanded_reference(ref_df, wav_axis)
    classes, dictionary = build_mean_dictionary(expanded_ref, mode)
    decoder = normalize_dictionary(dictionary).T

    dataset = build_synthetic_dataset(expanded_ref, classes, mode)
    split_idx = split_synthetic_dataset(dataset)
    model, history_df = train_model_with_similarity_supervision(dataset, split_idx, decoder, device)

    train_metrics, train_pred_df = summarize_synthetic_split(
        dataset, "train", split_idx, model, classes, device
    )
    val_metrics, val_pred_df = summarize_synthetic_split(
        dataset, "val", split_idx, model, classes, device
    )
    test_metrics, test_pred_df = summarize_synthetic_split(
        dataset, "test", split_idx, model, classes, device
    )

    existing_records = load_existing_real_records(mode)
    pt2_records = load_pt2_mixture_records(wav_axis, mode)
    existing_metrics, existing_pred_df, _ = summarize_real_records(
        existing_records, classes, model, device
    )
    pt2_metrics, pt2_pred_df, pt2_y_pred = summarize_real_records(
        pt2_records, classes, model, device
    )

    mode_dir = RESULTS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(mode_dir / "training_history.csv", index=False)
    train_pred_df.to_csv(mode_dir / "synthetic_train_predictions.csv", index=False)
    val_pred_df.to_csv(mode_dir / "synthetic_val_predictions.csv", index=False)
    test_pred_df.to_csv(mode_dir / "synthetic_test_predictions.csv", index=False)
    existing_pred_df.to_csv(mode_dir / "existing_real_predictions.csv", index=False)
    pt2_pred_df.to_csv(mode_dir / "pt2_real_predictions.csv", index=False)

    summary = {
        "mode": mode,
        "model_type": "mlp_similarity_supervision_regressor",
        "baseline_results_dir": str(BASELINE_RESULTS_DIR),
        "device": device.type,
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
            "margin": MARGIN,
            "margin_loss_weight": MARGIN_LOSS_WEIGHT,
            "similarity_false_loss_weight": SIMILARITY_FALSE_LOSS_WEIGHT,
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
            "per_mixture": per_source_summary(pt2_records, classes, pt2_y_pred),
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

    all_results = {}
    for mode in ("raw", "baseline_corrected"):
        all_results[mode] = run_mode(mode, device)

    (RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
