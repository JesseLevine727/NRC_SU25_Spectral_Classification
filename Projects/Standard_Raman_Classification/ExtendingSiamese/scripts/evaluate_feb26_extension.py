from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

FEB26_DIR = DATA_DIR / "Feb26_Spectra"
BASE_REF_PATH = DATA_DIR / "reference_siamese_resampled.csv"
BASE_QRY_PATH = DATA_DIR / "reference_v2.csv"
MODEL_PATH = MODELS_DIR / "siamese_raman_resampled.pth"

ALIGNED_FEB26_PATH = RESULTS_DIR / "feb26_aligned_resampled.csv"
EXPANDED_REF_PATH = RESULTS_DIR / "reference_siamese_resampled_plus_feb26.csv"
EXPANDED_QRY_PATH = RESULTS_DIR / "reference_v2_plus_feb26.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions_plus_feb26.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_plus_feb26.json"
PENALTY_CACHE = {}


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


def canonical_label(folder_name):
    mapping = {
        "Aniline": "aniline",
        "Benzene": "benzene",
        "DCM": "dcm",
        "Diethylamine": "diethylamine",
        "Pyridine": "pyridine",
        "n-hexane": "n-hexane",
    }
    return mapping[folder_name]


def load_target_axis(csv_path):
    df = pd.read_csv(csv_path, nrows=1)
    axis_cols = [col for col in df.columns if col != "Label"]
    axis = np.array([float(col) for col in axis_cols], dtype=float)
    return axis, axis_cols


def align_feb26(axis, axis_cols):
    rows = []
    for chem_dir in sorted(FEB26_DIR.iterdir()):
        if not chem_dir.is_dir():
            continue
        label = canonical_label(chem_dir.name)
        for txt_path in sorted(chem_dir.glob("*.txt")):
            arr = np.loadtxt(txt_path)
            native_x = arr[:, 0]
            native_y = arr[:, 1]
            aligned = np.interp(axis, native_x, native_y)
            row = {axis_cols[i]: aligned[i] for i in range(len(axis_cols))}
            row["Label"] = label
            row["SourceFolder"] = chem_dir.name
            row["SourceFile"] = txt_path.name
            rows.append(row)
    return pd.DataFrame(rows)


def split_reference_query(feb26_df):
    ref_parts = []
    qry_parts = []
    for label, group in feb26_df.groupby("Label", sort=True):
        group = group.sort_values(["SourceFolder", "SourceFile"]).reset_index(drop=True)
        ref_parts.append(group.iloc[[0]].copy())
        qry_parts.append(group.iloc[1:].copy())
    feb26_ref = pd.concat(ref_parts, ignore_index=True)
    feb26_qry = pd.concat(qry_parts, ignore_index=True)
    return feb26_ref, feb26_qry


def topk_distinct_labels(sorted_indices, ref_labels, k):
    distinct = []
    for idx in sorted_indices:
        label = ref_labels[idx]
        if label not in distinct:
            distinct.append(label)
        if len(distinct) == k:
            break
    while len(distinct) < k:
        distinct.append(None)
    return distinct


def evaluate_model(ref_df, qry_df, axis_cols):
    ref_specs = ref_df[axis_cols].to_numpy(dtype=float)
    qry_specs = qry_df[axis_cols].to_numpy(dtype=float)
    ref_labels = ref_df["Label"].astype(str).to_numpy()
    qry_labels = qry_df["Label"].astype(str).to_numpy()

    ref_proc = preprocess(ref_specs)
    qry_proc = preprocess(qry_specs)

    input_len = ref_proc.shape[1]
    model = SiameseNet(input_len, embed_dim=64)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        ref_embeds = model(torch.tensor(ref_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()
        qry_embeds = model(torch.tensor(qry_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()

    predictions = []
    for idx, q_embed in enumerate(qry_embeds):
        dists = np.linalg.norm(ref_embeds - q_embed, axis=1)
        order = np.argsort(dists)
        top1, top2 = topk_distinct_labels(order, ref_labels, k=2)
        predictions.append(
            {
                "Label": qry_labels[idx],
                "PredTop1": top1,
                "PredTop2": top2,
                "NearestDistance": float(dists[order[0]]),
                "SecondDistinctDistance": float(dists[order[1]]) if len(order) > 1 else np.nan,
            }
        )
    pred_df = qry_df[["Label"]].copy()
    pred_df["PredTop1"] = [row["PredTop1"] for row in predictions]
    pred_df["PredTop2"] = [row["PredTop2"] for row in predictions]
    pred_df["NearestDistance"] = [row["NearestDistance"] for row in predictions]
    pred_df["SecondDistinctDistance"] = [row["SecondDistinctDistance"] for row in predictions]
    if "SourceFolder" in qry_df.columns:
        pred_df["SourceFolder"] = qry_df["SourceFolder"].values
        pred_df["SourceFile"] = qry_df["SourceFile"].values

    top1_acc = float((pred_df["Label"] == pred_df["PredTop1"]).mean())
    top2_acc = float(
        np.mean(
            [
                truth in (pred1, pred2)
                for truth, pred1, pred2 in zip(
                    pred_df["Label"], pred_df["PredTop1"], pred_df["PredTop2"]
                )
            ]
        )
    )
    return pred_df, top1_acc, top2_acc


def accuracy_by_label(pred_df):
    rows = []
    for label, group in pred_df.groupby("Label", sort=True):
        rows.append(
            {
                "label": label,
                "n": int(len(group)),
                "top1_acc": float((group["Label"] == group["PredTop1"]).mean()),
                "top2_acc": float(
                    np.mean(
                        [
                            truth in (pred1, pred2)
                            for truth, pred1, pred2 in zip(
                                group["Label"], group["PredTop1"], group["PredTop2"]
                            )
                        ]
                    )
                ),
            }
        )
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    axis, axis_cols = load_target_axis(BASE_QRY_PATH)

    base_ref = pd.read_csv(BASE_REF_PATH)
    base_qry = pd.read_csv(BASE_QRY_PATH)
    base_labels = set(base_qry["Label"].astype(str))

    feb26_aligned = align_feb26(axis, axis_cols)
    feb26_ref, feb26_qry = split_reference_query(feb26_aligned)

    expanded_ref = pd.concat(
        [base_ref, feb26_ref[axis_cols + ["Label"]]],
        ignore_index=True,
    )
    expanded_qry = pd.concat(
        [base_qry, feb26_qry[axis_cols + ["Label", "SourceFolder", "SourceFile"]]],
        ignore_index=True,
    )

    expanded_pred, expanded_top1, expanded_top2 = evaluate_model(expanded_ref, expanded_qry, axis_cols)

    expanded_feb26_pred = expanded_pred[expanded_pred["SourceFolder"].notna()].copy()
    original_label_pred = expanded_pred[expanded_pred["Label"].isin(base_labels)].copy()
    only_new_labels = sorted(set(feb26_qry["Label"].astype(str)) - base_labels)
    new_only_pred = expanded_pred[expanded_pred["Label"].isin(only_new_labels)].copy()

    feb26_aligned.to_csv(ALIGNED_FEB26_PATH, index=False)
    expanded_ref.to_csv(EXPANDED_REF_PATH, index=False)
    expanded_qry.to_csv(EXPANDED_QRY_PATH, index=False)
    expanded_pred.to_csv(PREDICTIONS_PATH, index=False)

    summary = {
        "expanded_query": {
            "n_query": int(len(expanded_qry)),
            "top1_acc": expanded_top1,
            "top2_acc": expanded_top2,
        },
        "expanded_original_label_subset": {
            "n_query": int(len(original_label_pred)),
            "top1_acc": float((original_label_pred["Label"] == original_label_pred["PredTop1"]).mean()),
            "top2_acc": float(
                np.mean(
                    [
                        truth in (pred1, pred2)
                        for truth, pred1, pred2 in zip(
                            original_label_pred["Label"],
                            original_label_pred["PredTop1"],
                            original_label_pred["PredTop2"],
                        )
                    ]
                )
            ),
        },
        "feb26_query_subset": {
            "n_query": int(len(expanded_feb26_pred)),
            "top1_acc": float((expanded_feb26_pred["Label"] == expanded_feb26_pred["PredTop1"]).mean()),
            "top2_acc": float(
                np.mean(
                    [
                        truth in (pred1, pred2)
                        for truth, pred1, pred2 in zip(
                            expanded_feb26_pred["Label"],
                            expanded_feb26_pred["PredTop1"],
                            expanded_feb26_pred["PredTop2"],
                        )
                    ]
                )
            ),
            "per_label": accuracy_by_label(expanded_feb26_pred),
        },
        "true_new_labels_only": {
            "labels": only_new_labels,
            "n_query": int(len(new_only_pred)),
            "top1_acc": float((new_only_pred["Label"] == new_only_pred["PredTop1"]).mean()),
            "top2_acc": float(
                np.mean(
                    [
                        truth in (pred1, pred2)
                        for truth, pred1, pred2 in zip(
                            new_only_pred["Label"],
                            new_only_pred["PredTop1"],
                            new_only_pred["PredTop2"],
                        )
                    ]
                )
            ),
            "per_label": accuracy_by_label(new_only_pred),
        },
        "artifacts": {
            "aligned_feb26_csv": str(ALIGNED_FEB26_PATH.name),
            "expanded_reference_csv": str(EXPANDED_REF_PATH.name),
            "expanded_query_csv": str(EXPANDED_QRY_PATH.name),
            "predictions_csv": str(PREDICTIONS_PATH.name),
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
