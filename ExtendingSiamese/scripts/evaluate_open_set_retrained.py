from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

MODEL_PATH = MODELS_DIR / "siamese_raman_cross_device_finetuned.pth"
BASE_REF_PATH = DATA_DIR / "reference_siamese_resampled.csv"
BASE_QRY_PATH = DATA_DIR / "reference_v2.csv"
CONTROL_HOLDOUT_PATH = RESULTS_DIR / "feb26_known_controls_holdout.csv"
UNKNOWN_HOLDOUT_PATH = RESULTS_DIR / "feb26_unknown_holdout.csv"

HIST_PATH = FIGURES_DIR / "open_set_distance_histograms_retrained.png"
SWEEP_PATH = RESULTS_DIR / "open_set_threshold_sweep_retrained.csv"
SUMMARY_PATH = RESULTS_DIR / "open_set_summary_retrained.json"
PRED_PATH = RESULTS_DIR / "open_set_query_distances_retrained.csv"

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


def load_axis_cols(csv_path):
    df = pd.read_csv(csv_path, nrows=1)
    return [col for col in df.columns if col != "Label"]


def load_model(input_len):
    model = SiameseNet(input_len, embed_dim=64)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def embed_queries(model, qry_df, axis_cols):
    qry_specs = qry_df[axis_cols].to_numpy(dtype=float)
    qry_proc = preprocess(qry_specs)
    with torch.no_grad():
        return model(torch.tensor(qry_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()


def score_queries(ref_embeds, ref_labels, qry_embeds, qry_df):
    rows = []
    for idx, q_embed in enumerate(qry_embeds):
        dists = np.linalg.norm(ref_embeds - q_embed, axis=1)
        order = np.argsort(dists)
        nearest_idx = order[0]
        rows.append(
            {
                "Label": qry_df.iloc[idx]["Label"],
                "PredTop1": ref_labels[nearest_idx],
                "NearestDistance": float(dists[nearest_idx]),
                "MarginToSecond": float(dists[order[1]] - dists[order[0]]) if len(order) > 1 else np.nan,
            }
        )
    out = qry_df.copy().reset_index(drop=True)
    out["PredTop1"] = [row["PredTop1"] for row in rows]
    out["NearestDistance"] = [row["NearestDistance"] for row in rows]
    out["MarginToSecond"] = [row["MarginToSecond"] for row in rows]
    return out


def sweep_thresholds(known_df, unknown_df):
    all_dists = np.concatenate(
        [
            known_df["NearestDistance"].to_numpy(dtype=float),
            unknown_df["NearestDistance"].to_numpy(dtype=float),
        ]
    )
    thresholds = np.unique(np.quantile(all_dists, np.linspace(0.0, 1.0, 401)))

    rows = []
    for threshold in thresholds:
        known_accept = known_df["NearestDistance"] <= threshold
        unknown_reject = unknown_df["NearestDistance"] > threshold
        known_correct = ((known_df["PredTop1"] == known_df["Label"]) & known_accept).mean()
        unknown_reject_rate = unknown_reject.mean()
        rows.append(
            {
                "threshold": float(threshold),
                "known_correct_and_accept_rate": float(known_correct),
                "known_accept_rate": float(known_accept.mean()),
                "unknown_reject_rate": float(unknown_reject_rate),
                "unknown_false_accept_rate": float(1.0 - unknown_reject_rate),
                "open_set_accuracy": float(
                    (known_correct * len(known_df) + unknown_reject_rate * len(unknown_df))
                    / (len(known_df) + len(unknown_df))
                ),
                "balanced_accuracy": float(0.5 * (known_correct + unknown_reject_rate)),
            }
        )
    return pd.DataFrame(rows)


def pick_operating_points(known_df, unknown_df, sweep_df):
    best_balanced = sweep_df.sort_values(
        ["balanced_accuracy", "open_set_accuracy", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    p95_threshold = float(np.quantile(known_df["NearestDistance"], 0.95))
    p99_threshold = float(np.quantile(known_df["NearestDistance"], 0.99))

    def metrics_at(threshold):
        known_accept = known_df["NearestDistance"] <= threshold
        unknown_reject = unknown_df["NearestDistance"] > threshold
        known_correct = ((known_df["PredTop1"] == known_df["Label"]) & known_accept).mean()
        return {
            "threshold": float(threshold),
            "known_correct_and_accept_rate": float(known_correct),
            "known_accept_rate": float(known_accept.mean()),
            "unknown_reject_rate": float(unknown_reject.mean()),
            "unknown_false_accept_rate": float((~unknown_reject).mean()),
            "open_set_accuracy": float(
                (known_correct * len(known_df) + unknown_reject.mean() * len(unknown_df))
                / (len(known_df) + len(unknown_df))
            ),
            "balanced_accuracy": float(0.5 * (known_correct + unknown_reject.mean())),
        }

    return {
        "best_balanced_accuracy": best_balanced.to_dict(),
        "known_distance_p95": metrics_at(p95_threshold),
        "known_distance_p99": metrics_at(p99_threshold),
    }


def per_label_reject_rates(df, threshold):
    rows = []
    for label, group in df.groupby("Label", sort=True):
        rows.append(
            {
                "label": label,
                "n": int(len(group)),
                "reject_rate": float((group["NearestDistance"] > threshold).mean()),
                "mean_distance": float(group["NearestDistance"].mean()),
                "min_distance": float(group["NearestDistance"].min()),
                "max_distance": float(group["NearestDistance"].max()),
            }
        )
    return rows


def make_histograms(known_df, unknown_df, control_df):
    plt.figure(figsize=(10, 6))
    bins = 40
    plt.hist(known_df["NearestDistance"], bins=bins, alpha=0.6, label=f"Known queries (n={len(known_df)})", color="#4c78a8")
    plt.hist(unknown_df["NearestDistance"], bins=bins, alpha=0.6, label=f"Unknown holdout (n={len(unknown_df)})", color="#e45756")
    plt.hist(control_df["NearestDistance"], bins=bins, alpha=0.6, label=f"Cross-device known controls (n={len(control_df)})", color="#72b7b2")
    plt.xlabel("Nearest reference embedding distance")
    plt.ylabel("Count")
    plt.title("Open-Set Distance Histograms After Cross-Device Fine-Tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=180)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    axis_cols = load_axis_cols(BASE_REF_PATH)
    ref_df = pd.read_csv(BASE_REF_PATH)
    base_qry = pd.read_csv(BASE_QRY_PATH)
    control_qry = pd.read_csv(CONTROL_HOLDOUT_PATH)
    unknown_qry = pd.read_csv(UNKNOWN_HOLDOUT_PATH)

    ref_specs = ref_df[axis_cols].to_numpy(dtype=float)
    ref_proc = preprocess(ref_specs)
    model = load_model(ref_proc.shape[1])

    with torch.no_grad():
        ref_embeds = model(torch.tensor(ref_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()
    ref_labels = ref_df["Label"].astype(str).to_numpy()

    known_scored = score_queries(ref_embeds, ref_labels, embed_queries(model, base_qry, axis_cols), base_qry)
    control_scored = score_queries(ref_embeds, ref_labels, embed_queries(model, control_qry, axis_cols), control_qry)
    unknown_scored = score_queries(ref_embeds, ref_labels, embed_queries(model, unknown_qry, axis_cols), unknown_qry)

    combined_known = pd.concat([known_scored, control_scored], ignore_index=True)
    sweep_df = sweep_thresholds(combined_known, unknown_scored)
    operating_points = pick_operating_points(combined_known, unknown_scored, sweep_df)
    best_threshold = float(operating_points["best_balanced_accuracy"]["threshold"])

    pred_df = pd.concat(
        [
            known_scored.assign(QuerySet="known"),
            control_scored.assign(QuerySet="known_control"),
            unknown_scored.assign(QuerySet="unknown"),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(PRED_PATH, index=False)
    sweep_df.to_csv(SWEEP_PATH, index=False)
    make_histograms(known_scored, unknown_scored, control_scored)

    summary = {
        "query_counts": {
            "known": int(len(known_scored)),
            "known_control": int(len(control_scored)),
            "unknown": int(len(unknown_scored)),
        },
        "distance_summary": {
            "known_mean": float(known_scored["NearestDistance"].mean()),
            "known_max": float(known_scored["NearestDistance"].max()),
            "known_control_mean": float(control_scored["NearestDistance"].mean()),
            "known_control_max": float(control_scored["NearestDistance"].max()),
            "unknown_mean": float(unknown_scored["NearestDistance"].mean()),
            "unknown_min": float(unknown_scored["NearestDistance"].min()),
        },
        "operating_points": operating_points,
        "known_control_reject_rates_at_best_threshold": per_label_reject_rates(control_scored, best_threshold),
        "unknown_reject_rates_at_best_threshold": per_label_reject_rates(unknown_scored, best_threshold),
        "artifacts": {
            "histograms_png": HIST_PATH.name,
            "threshold_sweep_csv": SWEEP_PATH.name,
            "query_distances_csv": PRED_PATH.name,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
