from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap.umap_ as umap


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

BASE_DATA_PATH = DATA_DIR / "reference_v2.csv"
FEB26_ALIGNED_PATH = RESULTS_DIR / "feb26_aligned_resampled.csv"
FEB26_RAW_DIR = DATA_DIR / "Feb26_Spectra"
BASELINE_MODEL_PATH = MODELS_DIR / "siamese_raman_resampled.pth"
RETRAINED_MODEL_PATH = MODELS_DIR / "siamese_raman_cross_device_finetuned.pth"

SPECTRA_GALLERY_PATH = FIGURES_DIR / "chemical_spectra_gallery.png"
CENTROID_HEATMAPS_PATH = FIGURES_DIR / "centroid_similarity_panels.png"
CENTROID_UMAP_PATH = FIGURES_DIR / "centroid_umap_panels.png"
DEVICE_SHIFT_STACK_PATH = FIGURES_DIR / "device_shift_stacked_spectra.png"
RAW_SIM_CSV = RESULTS_DIR / "centroid_similarity_raw.csv"
BASELINE_SIM_CSV = RESULTS_DIR / "centroid_similarity_embedding_baseline.csv"
RETRAINED_SIM_CSV = RESULTS_DIR / "centroid_similarity_embedding_retrained.csv"
SUMMARY_PATH = RESULTS_DIR / "chemical_view_summary.json"

PENALTY_CACHE = {}


def clean_label(label):
    return str(label).strip().strip('"').strip()


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


def canonical_feb26_label(folder_name):
    mapping = {
        "Aniline": "aniline",
        "Benzene": "benzene",
        "DCM": "dcm",
        "Diethylamine": "diethylamine",
        "Pyridine": "pyridine",
        "n-hexane": "n-hexane",
    }
    return mapping[folder_name]


def ensure_aligned_feb26(axis_cols):
    if FEB26_ALIGNED_PATH.exists():
        feb26 = pd.read_csv(FEB26_ALIGNED_PATH)
        feb26["Label"] = feb26["Label"].map(clean_label)
        return feb26

    axis = np.array([float(col) for col in axis_cols], dtype=float)
    rows = []
    for chem_dir in sorted(FEB26_RAW_DIR.iterdir()):
        if not chem_dir.is_dir():
            continue
        label = canonical_feb26_label(chem_dir.name)
        for txt_path in sorted(chem_dir.glob("*.txt")):
            arr = np.loadtxt(txt_path)
            aligned = np.interp(axis, arr[:, 0], arr[:, 1])
            row = {axis_cols[i]: aligned[i] for i in range(len(axis_cols))}
            row["Label"] = label
            row["SourceFolder"] = chem_dir.name
            row["SourceFile"] = txt_path.name
            rows.append(row)
    feb26 = pd.DataFrame(rows)
    feb26.to_csv(FEB26_ALIGNED_PATH, index=False)
    return feb26


def load_combined_data():
    base = pd.read_csv(BASE_DATA_PATH)
    axis_cols = [col for col in base.columns if col != "Label"]
    base["Label"] = base["Label"].map(clean_label)
    base["Device"] = "old"
    base["Group"] = base["Device"] + ":" + base["Label"]

    feb26 = ensure_aligned_feb26(axis_cols)
    feb26["Label"] = feb26["Label"].map(clean_label)
    feb26["Device"] = "feb26"
    feb26["Group"] = feb26["Device"] + ":" + feb26["Label"]

    base = base[axis_cols + ["Label", "Device", "Group"]].copy()
    feb26 = feb26[axis_cols + ["Label", "Device", "Group"]].copy()
    combined = pd.concat([base, feb26], ignore_index=True)
    return combined, axis_cols


def ordered_labels(df):
    labels = sorted(df["Label"].unique())
    return labels


def ordered_groups(df):
    source_order = {"old": 0, "feb26": 1}

    def sort_key(group_name):
        device, label = group_name.split(":", 1)
        return (label, source_order.get(device, 99), device)

    return sorted(df["Group"].unique(), key=sort_key)


def plot_spectra_gallery(df, axis_cols):
    axis = np.array([float(col) for col in axis_cols], dtype=float)
    labels = ordered_labels(df)
    ncols = 4
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    colors = {"old": "#4c78a8", "feb26": "#f58518"}

    for idx, label in enumerate(labels):
        ax = axes[idx]
        label_df = df[df["Label"] == label]
        for device in ["old", "feb26"]:
            group = label_df[label_df["Device"] == device]
            if group.empty:
                continue
            specs = group[axis_cols].to_numpy(dtype=float)
            sample_count = min(len(specs), 6)
            for spec in specs[:sample_count]:
                ax.plot(axis, spec, color=colors[device], alpha=0.08, linewidth=0.8)
            mean_spec = specs.mean(axis=0)
            std_spec = specs.std(axis=0)
            ax.plot(axis, mean_spec, color=colors[device], linewidth=1.5, label=device)
            ax.fill_between(axis, mean_spec - std_spec, mean_spec + std_spec, color=colors[device], alpha=0.12)
        ax.set_title(label)
        ax.set_xlim(axis[0], axis[-1])
        if idx % ncols == 0:
            ax.set_ylabel("Intensity")
        if idx >= len(labels) - ncols:
            ax.set_xlabel("Wavenumber")
        ax.legend(loc="upper right", fontsize=8)

    for idx in range(len(labels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Chemical Spectra Gallery: old-device vs Feb26", fontsize=16)
    fig.tight_layout()
    fig.savefig(SPECTRA_GALLERY_PATH, dpi=180)
    plt.close(fig)


def plot_device_shift_stacks(df, axis_cols):
    axis = np.array([float(col) for col in axis_cols], dtype=float)
    compare_labels = ["benzene", "pyridine"]
    fig, axes = plt.subplots(len(compare_labels), 1, figsize=(16, 4.5 * len(compare_labels)), sharex=True)
    if len(compare_labels) == 1:
        axes = [axes]

    colors = {"old": "#4c78a8", "feb26": "#f58518"}

    for ax, label in zip(axes, compare_labels):
        label_df = df[df["Label"] == label]
        offset = 0.0
        for device in ["old", "feb26"]:
            group = label_df[label_df["Device"] == device]
            if group.empty:
                continue
            specs = preprocess(group[axis_cols].to_numpy(dtype=float))
            mean_spec = specs.mean(axis=0)
            sample_count = min(len(specs), 6)
            for spec in specs[:sample_count]:
                ax.plot(axis, spec + offset, color=colors[device], alpha=0.10, linewidth=0.8)
            ax.plot(axis, mean_spec + offset, color=colors[device], linewidth=1.8, label=f"{device} mean")
            ax.text(axis[5], offset + 0.15, f"{label} | {device}", color=colors[device], fontsize=10, weight="bold")
            offset += 0.45
        ax.set_title(f"Device Shift Stack: {label}")
        ax.set_ylabel("Normalized intensity + offset")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Wavenumber")
    fig.tight_layout()
    fig.savefig(DEVICE_SHIFT_STACK_PATH, dpi=180)
    plt.close(fig)


def cosine_similarity_matrix(vectors):
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def raw_centroids(df, axis_cols):
    groups = ordered_groups(df)
    vectors = []
    for group in groups:
        group_df = df[df["Group"] == group]
        vectors.append(group_df[axis_cols].to_numpy(dtype=float).mean(axis=0))
    sim = cosine_similarity_matrix(vectors)
    return groups, np.asarray(vectors), sim


def embedding_centroids(df, axis_cols, model_path):
    specs = df[axis_cols].to_numpy(dtype=float)
    proc = preprocess(specs)
    model = SiameseNet(proc.shape[1], embed_dim=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        embeds = model(torch.tensor(proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()

    groups = ordered_groups(df)
    vectors = []
    for group in groups:
        mask = df["Group"] == group
        vectors.append(embeds[mask.to_numpy()].mean(axis=0))
    vectors = np.asarray(vectors)
    sim = cosine_similarity_matrix(vectors)
    return groups, vectors, sim


def save_similarity_csv(groups, sim, out_path):
    pd.DataFrame(sim, index=groups, columns=groups).to_csv(out_path)


def plot_similarity_panels(groups, raw_sim, baseline_sim, retrained_sim):
    panels = [("Raw centroid cosine similarity", raw_sim)]
    if baseline_sim is not None:
        panels.append(("Baseline embedding centroid similarity", baseline_sim))
    if retrained_sim is not None:
        panels.append(("Fine-tuned embedding centroid similarity", retrained_sim))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, sim) in zip(axes, panels):
        im = ax.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=90, fontsize=8)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(CENTROID_HEATMAPS_PATH, dpi=180)
    plt.close(fig)


def plot_umap_panels(groups, raw_vectors, baseline_vectors, retrained_vectors):
    panels = [("Raw centroid UMAP", raw_vectors)]
    if baseline_vectors is not None:
        panels.append(("Baseline embedding centroid UMAP", baseline_vectors))
    if retrained_vectors is not None:
        panels.append(("Fine-tuned embedding centroid UMAP", retrained_vectors))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    color_map = {"old": "#4c78a8", "feb26": "#f58518"}
    marker_map = {"old": "o", "feb26": "s"}

    for ax, (title, vectors) in zip(axes, panels):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(6, max(2, len(groups) - 1)),
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(vectors)
        for idx, group in enumerate(groups):
            device, label = group.split(":", 1)
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                color=color_map.get(device, "#666666"),
                marker=marker_map.get(device, "o"),
                s=70,
            )
            ax.text(coords[idx, 0], coords[idx, 1], f" {group}", fontsize=8, va="center")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(CENTROID_UMAP_PATH, dpi=180)
    plt.close(fig)


def nearest_neighbors(groups, sim):
    rows = []
    for idx, group in enumerate(groups):
        order = np.argsort(-sim[idx])
        neighbors = []
        for neighbor_idx in order:
            if neighbor_idx == idx:
                continue
            neighbors.append(
                {
                    "group": groups[neighbor_idx],
                    "similarity": float(sim[idx, neighbor_idx]),
                }
            )
            if len(neighbors) == 3:
                break
        rows.append({"group": group, "neighbors": neighbors})
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    combined, axis_cols = load_combined_data()

    plot_spectra_gallery(combined, axis_cols)
    plot_device_shift_stacks(combined, axis_cols)

    groups, raw_vectors, raw_sim = raw_centroids(combined, axis_cols)
    save_similarity_csv(groups, raw_sim, RAW_SIM_CSV)

    baseline_sim = None
    baseline_vectors = None
    if BASELINE_MODEL_PATH.exists():
        _, baseline_vectors, baseline_sim = embedding_centroids(combined, axis_cols, BASELINE_MODEL_PATH)
        save_similarity_csv(groups, baseline_sim, BASELINE_SIM_CSV)

    retrained_sim = None
    retrained_vectors = None
    if RETRAINED_MODEL_PATH.exists():
        _, retrained_vectors, retrained_sim = embedding_centroids(combined, axis_cols, RETRAINED_MODEL_PATH)
        save_similarity_csv(groups, retrained_sim, RETRAINED_SIM_CSV)

    plot_similarity_panels(groups, raw_sim, baseline_sim, retrained_sim)
    plot_umap_panels(groups, raw_vectors, baseline_vectors, retrained_vectors)

    summary = {
        "n_total_spectra": int(len(combined)),
        "label_counts": combined["Label"].value_counts().sort_index().to_dict(),
        "group_counts": combined["Group"].value_counts().sort_index().to_dict(),
        "raw_nearest_centroids": nearest_neighbors(groups, raw_sim),
        "baseline_embedding_nearest_centroids": nearest_neighbors(groups, baseline_sim) if baseline_sim is not None else None,
        "retrained_embedding_nearest_centroids": nearest_neighbors(groups, retrained_sim) if retrained_sim is not None else None,
        "artifacts": {
            "spectra_gallery_png": SPECTRA_GALLERY_PATH.name,
            "device_shift_stack_png": DEVICE_SHIFT_STACK_PATH.name,
            "centroid_heatmaps_png": CENTROID_HEATMAPS_PATH.name,
            "centroid_umap_png": CENTROID_UMAP_PATH.name,
            "raw_similarity_csv": RAW_SIM_CSV.name,
            "baseline_similarity_csv": BASELINE_SIM_CSV.name if baseline_sim is not None else None,
            "retrained_similarity_csv": RETRAINED_SIM_CSV.name if retrained_sim is not None else None,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
