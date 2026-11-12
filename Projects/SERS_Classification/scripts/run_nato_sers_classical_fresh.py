#!/usr/bin/env python3
"""Leakage-safe fresh classical baselines for the NATO SERS audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def estimator(name: str, seed: int, trees: int):
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior", random_state=seed)
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0, class_weight="balanced", solver="lsqr"))
    if name == "nearest_centroid":
        return NearestCentroid(metric="euclidean")
    if name == "pca_lda":
        return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                             LinearDiscriminantAnalysis(solver="svd"))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", class_weight="balanced"))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=trees, max_features="sqrt", min_samples_leaf=1,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=trees, max_features="sqrt", min_samples_leaf=1,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
    raise KeyError(name)


def score_matrix(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x), np.asarray(model.classes_)
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores, np.asarray(model.classes_)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def aggregate_master(scores: np.ndarray, classes: np.ndarray, manifest: pd.DataFrame,
                     indices: np.ndarray) -> tuple[list[str], list[str]]:
    frame = manifest.iloc[indices].reset_index(drop=True)
    truth, predicted = [], []
    for _, group in frame.reset_index().groupby("master_sample_id"):
        truth.append(group.target_analyte.iloc[0])
        predicted.append(str(classes[np.argmax(scores[group["index"]].mean(axis=0))]))
    return truth, predicted


def fixed_splits(manifest: pd.DataFrame, repeats: int, folds: int, seed: int):
    y, groups = manifest.target_analyte, manifest.master_sample_id
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(manifest, y, groups)):
            yield repeat, fold, train, test


def closed_set(manifest: pd.DataFrame, matrices: dict[str, np.ndarray], config: dict) -> pd.DataFrame:
    rows = []
    splits = list(fixed_splits(manifest, int(config["repeats"]), int(config["folds"]), int(config["random_seed"])))
    for preprocessing in config["preprocessing_candidates"]:
        x = matrices[preprocessing]
        for model_name in config["models"]:
            for repeat, fold, train, test in splits:
                model = estimator(model_name, int(config["random_seed"]) + repeat * 10 + fold, int(config["trees"]))
                model.fit(x[train], manifest.target_analyte.iloc[train])
                scores, classes = score_matrix(model, x[test])
                prediction = classes[np.argmax(scores, axis=1)]
                spectrum = metrics(manifest.target_analyte.iloc[test], prediction)
                master_truth, master_prediction = aggregate_master(scores, classes, manifest, test)
                master_result = metrics(master_truth, master_prediction)
                rows.append({"preprocessing": preprocessing, "model": model_name,
                             "repeat": repeat, "fold": fold, "n_train_spectra": len(train),
                             "n_test_spectra": len(test), "n_train_masters": manifest.iloc[train].master_sample_id.nunique(),
                             "n_test_masters": manifest.iloc[test].master_sample_id.nunique(),
                             **{f"spectrum_{k}": v for k, v in spectrum.items()},
                             **{f"master_{k}": v for k, v in master_result.items()}})
            print(f"closed {preprocessing} {model_name}", flush=True)
    return pd.DataFrame(rows)


def metadata_baselines(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    feature_sets = {
        "station_only": ["station"],
        "acquisition_metadata": ["station", "instrument", "sensor_family", "sensor_variant"],
    }
    for label, columns in feature_sets.items():
        for repeat, fold, train, test in fixed_splits(
            manifest, int(config["repeats"]), int(config["folds"]), int(config["random_seed"])
        ):
            transformer = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), columns)])
            model = make_pipeline(transformer, LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            model.fit(manifest.iloc[train], manifest.target_analyte.iloc[train])
            prediction = model.predict(manifest.iloc[test])
            result = metrics(manifest.target_analyte.iloc[test], prediction)
            rows.append({"feature_set": label, "repeat": repeat, "fold": fold,
                         "n_test_masters": manifest.iloc[test].master_sample_id.nunique(), **result})
    return pd.DataFrame(rows)


def shifted_domain(manifest: pd.DataFrame, matrices: dict[str, np.ndarray], config: dict,
                   domain_field: str) -> pd.DataFrame:
    rows = []
    # One master split assignment is reused for every held-out domain.
    splits = list(fixed_splits(manifest, 1, int(config["folds"]), int(config["random_seed"])))
    for preprocessing in config["shift_preprocessing"]:
        x = matrices[preprocessing]
        for model_name in config["shift_models"]:
            for domain in sorted(manifest[domain_field].dropna().unique()):
                all_truth, all_prediction = [], []
                tested_masters = set()
                for _, fold, train_master_rows, test_master_rows in splits:
                    train = train_master_rows[manifest.iloc[train_master_rows][domain_field].ne(domain).to_numpy()]
                    test = test_master_rows[manifest.iloc[test_master_rows][domain_field].eq(domain).to_numpy()]
                    if not len(test) or manifest.target_analyte.iloc[train].nunique() < 2:
                        continue
                    model = estimator(model_name, int(config["random_seed"]) + fold, int(config["trees"]))
                    model.fit(x[train], manifest.target_analyte.iloc[train])
                    prediction = model.predict(x[test])
                    all_truth.extend(manifest.target_analyte.iloc[test])
                    all_prediction.extend(prediction)
                    tested_masters.update(manifest.master_sample_id.iloc[test])
                if not all_truth:
                    continue
                result = metrics(all_truth, all_prediction)
                rows.append({"domain_field": domain_field, "held_out_domain": domain,
                             "preprocessing": preprocessing, "model": model_name,
                             "n_test_spectra": len(all_truth), "n_test_masters": len(tested_masters),
                             "n_test_classes": len(set(all_truth)), "chance_for_observed_classes": 1 / len(set(all_truth)),
                             **result})
            print(f"shift {domain_field} {preprocessing} {model_name}", flush=True)
    return pd.DataFrame(rows)


def write_html(path: Path, figure: go.Figure) -> None:
    figure.update_layout(template="plotly_white", font={"family": "Arial", "size": 14},
                         margin={"l": 80, "r": 30, "t": 80, "b": 80})
    figure.write_html(path, include_plotlyjs=True, full_html=True,
                      config={"responsive": True, "scrollZoom": True, "displaylogo": False})


def make_plots(output: Path, summary: pd.DataFrame, metadata: pd.DataFrame,
               shift: pd.DataFrame) -> list[dict[str, str]]:
    html_dir = output / "html"; html_dir.mkdir(exist_ok=True)
    artifacts = []
    def save(name, title, description, figure):
        write_html(html_dir / name, figure); artifacts.append({"file": name, "title": title, "description": description})
    for level, metric in (("spectrum", "spectrum_balanced_accuracy"), ("master", "master_balanced_accuracy")):
        fig = go.Figure()
        for idx, (preprocessing, group) in enumerate(summary.groupby("preprocessing")):
            fig.add_trace(go.Bar(x=group.model, y=group[f"{metric}_mean"], name=preprocessing,
                                 error_y={"type": "data", "array": group[f"{metric}_std"], "visible": True}))
        fig.update_layout(barmode="group", title=f"Repeated master-grouped closed-set results ({level} level)",
                          xaxis_title="Classifier", yaxis_title="Balanced accuracy")
        save(f"0{len(artifacts)+1}_{level}_closed_set.html", f"{level.title()}-level closed set",
             "Mean ± SD across 15 master-grouped folds; folds, not rows, are the repeated units.", fig)
    meta_summary = metadata.groupby("feature_set").agg(mean=("balanced_accuracy", "mean"), sd=("balanced_accuracy", "std")).reset_index()
    fig = go.Figure(go.Bar(x=meta_summary.feature_set, y=meta_summary["mean"],
                           error_y={"type": "data", "array": meta_summary.sd, "visible": True}))
    fig.update_layout(title="How much chemistry can acquisition metadata predict?", yaxis_title="Balanced accuracy")
    save("03_metadata_only.html", "Metadata-only confounding baseline",
         "No spectral intensities are used; high performance reveals design confounding.", fig)
    for domain_field in shift.domain_field.unique():
        frame = shift[shift.domain_field.eq(domain_field)]
        fig = go.Figure()
        for idx, ((prep, model), group) in enumerate(frame.groupby(["preprocessing", "model"])):
            fig.add_trace(go.Bar(x=group.held_out_domain, y=group.balanced_accuracy,
                                 name=f"{prep} | {model}"))
        fig.update_layout(barmode="group", title=f"Unseen-{domain_field} transfer with held-out physical masters",
                          xaxis_title=f"Held-out {domain_field}", yaxis_title="Balanced accuracy")
        save(f"0{len(artifacts)+1}_leave_{domain_field}_out.html", f"Leave-{domain_field}-out transfer",
             "The held domain is absent from training and test physical masters are also absent from training.", fig)
    return artifacts


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=repository / "configs" / "nato_sers_classical_fresh_v1.json")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    input_dir = repository / config["input_workspace"]
    prep_dir = repository / config["preprocessing_workspace"]
    output = repository / config["output_workspace"]
    (output / "tables").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(input_dir / "tables" / "analysis_manifest.csv")
    bundle = np.load(prep_dir / "preprocessing_candidates.npz")
    axis = bundle["axis_cm1"]
    low, high = config["range_cm1"]; mask = (axis >= low) & (axis <= high)
    matrices = {name: bundle[name][:, mask] for name in set(config["preprocessing_candidates"] + config["shift_preprocessing"])}
    folds = closed_set(manifest, matrices, config)
    folds.to_csv(output / "tables" / "closed_set_fold_metrics.csv", index=False)
    aggregations = {f"{column}_mean": (column, "mean") for column in
                    ["spectrum_balanced_accuracy", "spectrum_macro_f1", "master_balanced_accuracy", "master_macro_f1"]}
    aggregations.update({f"{column}_std": (column, "std") for column in
                         ["spectrum_balanced_accuracy", "spectrum_macro_f1", "master_balanced_accuracy", "master_macro_f1"]})
    summary = folds.groupby(["preprocessing", "model"]).agg(**aggregations).reset_index()
    summary.to_csv(output / "tables" / "closed_set_summary.csv", index=False)
    metadata = metadata_baselines(manifest, config)
    metadata.to_csv(output / "tables" / "metadata_only_fold_metrics.csv", index=False)
    shift = pd.concat([
        shifted_domain(manifest, matrices, config, "instrument"),
        shifted_domain(manifest, matrices, config, "sensor_family"),
    ], ignore_index=True)
    shift.to_csv(output / "tables" / "acquisition_shift_metrics.csv", index=False)
    artifacts = make_plots(output, summary, metadata, shift)
    (output / "html_artifact_registry.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    links = "".join(f'<li><a href="html/{x["file"]}">{x["title"]}</a> — {x["description"]}</li>' for x in artifacts)
    (output / "index.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>NATO SERS baselines</title>"
        "<style>body{font:16px Arial;max-width:1100px;margin:40px auto;line-height:1.5}li{margin:12px 0}</style></head>"
        f"<body><h1>Fresh classical and acquisition-shift baselines</h1><ol>{links}</ol></body></html>")
    provenance = {"protocol_version": config["protocol_version"],
                  "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "preprocessing_bundle_sha256": hashlib.sha256((prep_dir / "preprocessing_candidates.npz").read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.sort_values("master_balanced_accuracy_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
