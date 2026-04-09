from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from retrain_with_pt2_reference import (
    PresenceNetLogits,
    SiameseNet,
    compute_metrics,
    embed_inputs,
    floatify_cols,
    load_reference,
    predict_probabilities,
    preprocess_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "Mixture_Classification" / "Legacy_Siamese_Pipeline" / "Notebooks"
AUGMENTED_DIR = NOTEBOOK_DIR / "pt2_augmented_reference_experiment"
OUTPUT_DIR = NOTEBOOK_DIR / "pt2_augmented_existing_real_recalibrated"


def _report_text(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def _report_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )


def _best_threshold_f1(y_true_bin: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, probs)
    f1_scores = 2.0 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        best_thr = float(thresholds[best_idx])
    else:
        best_thr = 0.5
    return best_thr, float(f1_scores[best_idx])


def _load_augmented_models(
    wav_axis: np.ndarray,
) -> tuple[torch.device, list[str], SiameseNet, SiameseNet, PresenceNetLogits]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_len_raman = len(wav_axis)
    input_len_fft = input_len_raman // 2 + 1

    siamese_raman = SiameseNet(input_len=input_len_raman, embed_dim=64).to(device)
    siamese_fft = SiameseNet(input_len=input_len_fft, embed_dim=64).to(device)

    checkpoint = torch.load(
        AUGMENTED_DIR / "presence_net_logits_pt2_augmented.pth",
        map_location=device,
    )
    classes = list(checkpoint["classes"])
    model = PresenceNetLogits(d_input=128, n_classes=len(classes)).to(device)

    siamese_raman.load_state_dict(
        torch.load(AUGMENTED_DIR / "siamese_mixture_pt2_augmented.pth", map_location=device)
    )
    siamese_fft.load_state_dict(
        torch.load(AUGMENTED_DIR / "siamese_mixture_fft_pt2_augmented.pth", map_location=device)
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    siamese_raman.eval()
    siamese_fft.eval()
    model.eval()
    return device, classes, siamese_raman, siamese_fft, model


def _build_existing_real_dataset(
    wav_axis: np.ndarray,
    classes: list[str],
    device: torch.device,
    siamese_raman: SiameseNet,
    siamese_fft: SiameseNet,
    model: PresenceNetLogits,
) -> dict:
    mix_df = pd.read_csv(NOTEBOOK_DIR / "mixtures_dataset.csv")
    floatify_cols(mix_df)
    wav_cols = [c for c in mix_df.columns if c not in {"Label 1", "Label 2"}]
    spectra = mix_df[wav_cols].to_numpy(dtype=np.float32)

    raman, fft = preprocess_matrix(spectra)
    emb_raman = embed_inputs(siamese_raman, raman, device)
    emb_fft = embed_inputs(siamese_fft, fft, device)
    x_real = np.hstack([emb_raman, emb_fft]).astype(np.float32)
    probs_all = predict_probabilities(model, x_real, device)

    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_real = np.zeros((len(mix_df), len(classes)), dtype=np.int64)
    for idx, row in mix_df.iterrows():
        y_real[idx, class_to_i[row["Label 1"]]] = 1
        y_real[idx, class_to_i[row["Label 2"]]] = 1

    supports = y_real.sum(axis=0)
    valid_idx = np.where(supports > 0)[0]
    valid_names = [classes[i] for i in valid_idx]

    return {
        "mix_df": mix_df,
        "x_real": x_real,
        "y_real": y_real,
        "probs_all": probs_all,
        "supports": supports,
        "valid_idx": valid_idx,
        "valid_names": valid_names,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, wav_axis = load_reference()
    device, classes, siamese_raman, siamese_fft, model = _load_augmented_models(wav_axis)
    existing = _build_existing_real_dataset(
        wav_axis,
        classes,
        device,
        siamese_raman,
        siamese_fft,
        model,
    )

    y_real = existing["y_real"]
    probs_all = existing["probs_all"]
    valid_idx = existing["valid_idx"]
    valid_names = existing["valid_names"]

    # Notebook-style filtered full-set report at default 0.5 threshold.
    y_pred_default = (probs_all >= 0.5).astype(int)
    filtered_default_report_text = _report_text(
        y_real[:, valid_idx],
        y_pred_default[:, valid_idx],
        valid_names,
    )
    filtered_default_report_dict = _report_dict(
        y_real[:, valid_idx],
        y_pred_default[:, valid_idx],
        valid_names,
    )
    all_class_default_metrics = compute_metrics(y_real, y_pred_default)

    # Recalibrate thresholds for the expanded 17-class head on a 25/75 split.
    sample_ids = np.arange(len(existing["x_real"]))
    idx_cal, idx_test, y_cal, y_test, probs_cal, probs_test = train_test_split(
        sample_ids,
        y_real,
        probs_all,
        test_size=0.75,
        random_state=42,
    )

    best_thresholds = np.ones(len(classes), dtype=np.float32) * 0.5
    calibration_rows = []
    for class_idx, class_name in enumerate(classes):
        support_cal = int(y_cal[:, class_idx].sum())
        if support_cal < 1:
            calibration_rows.append(
                {
                    "Class": class_name,
                    "Best_Thr": 0.5,
                    "Calib_F1": None,
                    "Support_Calib": support_cal,
                }
            )
            continue

        best_thr, best_f1 = _best_threshold_f1(y_cal[:, class_idx], probs_cal[:, class_idx])
        best_thresholds[class_idx] = best_thr
        calibration_rows.append(
            {
                "Class": class_name,
                "Best_Thr": best_thr,
                "Calib_F1": best_f1,
                "Support_Calib": support_cal,
            }
        )

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(OUTPUT_DIR / "calibration_thresholds_table.csv", index=False)

    thresholds_dict = {
        class_name: float(thr) for class_name, thr in zip(classes, best_thresholds.tolist())
    }
    (OUTPUT_DIR / "calibrated_thresholds_17class.json").write_text(
        json.dumps(thresholds_dict, indent=2)
    )

    y_pred_default_test = (probs_test >= 0.5).astype(int)
    y_pred_tuned_test = (probs_test >= best_thresholds).astype(int)

    supports_test = y_test.sum(axis=0)
    valid_idx_test = np.where(supports_test > 0)[0]
    valid_names_test = [classes[i] for i in valid_idx_test]

    filtered_default_test_report_text = _report_text(
        y_test[:, valid_idx_test],
        y_pred_default_test[:, valid_idx_test],
        valid_names_test,
    )
    filtered_tuned_test_report_text = _report_text(
        y_test[:, valid_idx_test],
        y_pred_tuned_test[:, valid_idx_test],
        valid_names_test,
    )
    filtered_default_test_report_dict = _report_dict(
        y_test[:, valid_idx_test],
        y_pred_default_test[:, valid_idx_test],
        valid_names_test,
    )
    filtered_tuned_test_report_dict = _report_dict(
        y_test[:, valid_idx_test],
        y_pred_tuned_test[:, valid_idx_test],
        valid_names_test,
    )

    all_class_default_test_metrics = compute_metrics(y_test, y_pred_default_test)
    all_class_tuned_test_metrics = compute_metrics(y_test, y_pred_tuned_test)

    weighted_f1_default_test = float(
        f1_score(y_test, y_pred_default_test, average="weighted", zero_division=0)
    )
    weighted_f1_tuned_test = float(
        f1_score(y_test, y_pred_tuned_test, average="weighted", zero_division=0)
    )

    text_lines = [
        "PT2-expanded model re-evaluation on original real mixtures",
        "",
        f"Expanded classes ({len(classes)}): {classes}",
        f"Existing real samples: {len(existing['x_real'])}",
        f"Notebook-style valid classes: {valid_names}",
        "",
        "==================================================",
        "FULL ORIGINAL REAL SET, NOTEBOOK-STYLE FILTER, DEFAULT 0.5",
        "==================================================",
        filtered_default_report_text,
        "",
        f"All-class default metrics: {json.dumps(all_class_default_metrics, indent=2)}",
        "",
        "==================================================",
        f"CALIBRATION SPLIT: {len(idx_cal)} tune / {len(idx_test)} test",
        "==================================================",
        calibration_df.sort_values("Calib_F1", ascending=False, na_position="last").to_string(index=False),
        "",
        "==================================================",
        "TEST SPLIT, NOTEBOOK-STYLE FILTER, DEFAULT 0.5",
        "==================================================",
        filtered_default_test_report_text,
        "",
        "==================================================",
        "TEST SPLIT, NOTEBOOK-STYLE FILTER, CALIBRATED 17-CLASS THRESHOLDS",
        "==================================================",
        filtered_tuned_test_report_text,
        "",
        f"Weighted F1 on test split: default={weighted_f1_default_test:.4f}, calibrated={weighted_f1_tuned_test:.4f}, delta={weighted_f1_tuned_test - weighted_f1_default_test:+.4f}",
        "",
        f"All-class default test metrics: {json.dumps(all_class_default_test_metrics, indent=2)}",
        "",
        f"All-class calibrated test metrics: {json.dumps(all_class_tuned_test_metrics, indent=2)}",
    ]
    (OUTPUT_DIR / "reports.txt").write_text("\n".join(text_lines))

    summary = {
        "augmented_model_dir": str(AUGMENTED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "expanded_classes": classes,
        "existing_real_samples": int(len(existing["x_real"])),
        "notebook_style_valid_classes": valid_names,
        "full_existing_real_default_0_5": {
            "all_classes_metrics": all_class_default_metrics,
            "filtered_report": filtered_default_report_dict,
        },
        "calibration_split": {
            "random_state": 42,
            "calibration_samples": int(len(idx_cal)),
            "test_samples": int(len(idx_test)),
            "thresholds": thresholds_dict,
            "calibration_table": calibration_rows,
            "test_default_0_5": {
                "all_classes_metrics": all_class_default_test_metrics,
                "filtered_report": filtered_default_test_report_dict,
                "weighted_f1_all_classes": weighted_f1_default_test,
            },
            "test_calibrated_17class": {
                "all_classes_metrics": all_class_tuned_test_metrics,
                "filtered_report": filtered_tuned_test_report_dict,
                "weighted_f1_all_classes": weighted_f1_tuned_test,
            },
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved outputs in {OUTPUT_DIR}")
    print("")
    print("Full original real set, notebook-style filter, default 0.5:")
    print(filtered_default_report_text)
    print("")
    print(
        "Test split weighted F1: "
        f"default={weighted_f1_default_test:.4f}, "
        f"calibrated={weighted_f1_tuned_test:.4f}, "
        f"delta={weighted_f1_tuned_test - weighted_f1_default_test:+.4f}"
    )


if __name__ == "__main__":
    main()
