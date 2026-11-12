#!/usr/bin/env python3
"""Test chemistry tasks that are identifiable despite NATO station confounding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from run_nato_sers_classical_fresh import aggregate_master, estimator, score_matrix


def result_metrics(truth, prediction):
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0))}


def station_tasks(manifest, matrices, config):
    rows = []
    for station in sorted(manifest.station.unique()):
        selected = np.flatnonzero(manifest.station.eq(station).to_numpy())
        local = manifest.iloc[selected].reset_index(drop=True)
        for preprocessing, full_matrix in matrices.items():
            matrix = full_matrix[selected]
            for model_name in config["models"]:
                for repeat in range(int(config["repeats"])):
                    split = StratifiedGroupKFold(n_splits=int(config["folds"]), shuffle=True,
                                                 random_state=int(config["random_seed"]) + repeat)
                    for fold, (train, test) in enumerate(split.split(matrix, local.target_analyte,
                                                                     local.master_sample_id)):
                        model = estimator(model_name, int(config["random_seed"]) + repeat * 10 + fold,
                                          int(config["trees"]))
                        model.fit(matrix[train], local.target_analyte.iloc[train])
                        scores, classes = score_matrix(model, matrix[test])
                        prediction = classes[np.argmax(scores, axis=1)]
                        spectrum = result_metrics(local.target_analyte.iloc[test], prediction)
                        master_truth, master_pred = aggregate_master(scores, classes, local, test)
                        master = result_metrics(master_truth, master_pred)
                        rows.append({"station": station, "preprocessing": preprocessing, "model": model_name,
                                     "repeat": repeat, "fold": fold, "n_train_masters": local.iloc[train].master_sample_id.nunique(),
                                     "n_test_masters": local.iloc[test].master_sample_id.nunique(),
                                     **{f"spectrum_{k}": v for k, v in spectrum.items()},
                                     **{f"master_{k}": v for k, v in master.items()}})
                print(f"station {station} {preprocessing} {model_name}", flush=True)
    return pd.DataFrame(rows)


def station_metadata_baseline(manifest, config):
    rows = []; columns = ["instrument", "sensor_family", "sensor_variant", "team", "operator_initials"]
    for station in sorted(manifest.station.unique()):
        local = manifest[manifest.station.eq(station)].reset_index(drop=True)
        for repeat in range(int(config["repeats"])):
            split = StratifiedGroupKFold(n_splits=int(config["folds"]), shuffle=True,
                                         random_state=int(config["random_seed"]) + repeat)
            for fold, (train, test) in enumerate(split.split(local, local.target_analyte, local.master_sample_id)):
                transform = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), columns)])
                model = make_pipeline(transform, LogisticRegression(C=1, class_weight="balanced", max_iter=2000))
                model.fit(local.iloc[train], local.target_analyte.iloc[train])
                rows.append({"station": station, "repeat": repeat, "fold": fold,
                             **result_metrics(local.target_analyte.iloc[test], model.predict(local.iloc[test]))})
    return pd.DataFrame(rows)


def cross_station_binary(manifest, matrices, config):
    keep = manifest.target_analyte.isin(["4_ANPP", "benzyl_fentanyl"]) & manifest.station.isin(["pills", "surfaces"])
    local = manifest[keep].reset_index().rename(columns={"index": "global_index"})
    rows = []
    for train_station, test_station in (("pills", "surfaces"), ("surfaces", "pills")):
        train = np.flatnonzero(local.station.eq(train_station)); test = np.flatnonzero(local.station.eq(test_station))
        for preprocessing, full_matrix in matrices.items():
            matrix = full_matrix[local.global_index.to_numpy()]
            for model_name in config["models"]:
                model = estimator(model_name, int(config["random_seed"]), int(config["trees"]))
                model.fit(matrix[train], local.target_analyte.iloc[train])
                scores, classes = score_matrix(model, matrix[test]); pred = classes[np.argmax(scores, axis=1)]
                spectrum = result_metrics(local.target_analyte.iloc[test], pred)
                master_truth, master_pred = aggregate_master(scores, classes, local, test)
                master = result_metrics(master_truth, master_pred)
                rows.append({"train_station": train_station, "test_station": test_station,
                             "preprocessing": preprocessing, "model": model_name,
                             "n_train_spectra": len(train), "n_test_spectra": len(test),
                             "n_train_masters": local.iloc[train].master_sample_id.nunique(),
                             "n_test_masters": local.iloc[test].master_sample_id.nunique(),
                             **{f"spectrum_{k}": v for k, v in spectrum.items()},
                             **{f"master_{k}": v for k, v in master.items()}})
    return pd.DataFrame(rows)


def leave_instrument_within_station(manifest, matrices, config):
    rows = []
    for station in sorted(manifest.station.unique()):
        global_indices = np.flatnonzero(manifest.station.eq(station).to_numpy())
        local = manifest.iloc[global_indices].reset_index(drop=True)
        splitter = StratifiedGroupKFold(n_splits=int(config["folds"]), shuffle=True,
                                        random_state=int(config["random_seed"]))
        splits = list(splitter.split(local, local.target_analyte, local.master_sample_id))
        for instrument in sorted(local.instrument.unique()):
            for preprocessing, full_matrix in matrices.items():
                matrix = full_matrix[global_indices]
                for model_name in ["rbf_svm", "random_forest"]:
                    truth, prediction, masters = [], [], set()
                    for fold, (train_master, test_master) in enumerate(splits):
                        train = train_master[local.instrument.iloc[train_master].ne(instrument).to_numpy()]
                        test = test_master[local.instrument.iloc[test_master].eq(instrument).to_numpy()]
                        if not len(test) or local.target_analyte.iloc[train].nunique() < local.target_analyte.nunique():
                            continue
                        model = estimator(model_name, int(config["random_seed"]) + fold, int(config["trees"]))
                        model.fit(matrix[train], local.target_analyte.iloc[train])
                        prediction.extend(model.predict(matrix[test])); truth.extend(local.target_analyte.iloc[test])
                        masters.update(local.master_sample_id.iloc[test])
                    if len(set(truth)) < 2:
                        continue
                    rows.append({"station": station, "held_out_instrument": instrument,
                                 "preprocessing": preprocessing, "model": model_name,
                                 "n_test_spectra": len(truth), "n_test_masters": len(masters),
                                 "n_test_classes": len(set(truth)), "chance": 1 / len(set(truth)),
                                 **result_metrics(truth, prediction)})
            print(f"within-station shift {station} {instrument}", flush=True)
    return pd.DataFrame(rows)


def write_html(path, figure):
    figure.update_layout(template="plotly_white", font={"family": "Arial", "size": 14},
                         margin={"l": 80, "r": 30, "t": 80, "b": 80})
    figure.write_html(path, include_plotlyjs=True, full_html=True,
                      config={"responsive": True, "scrollZoom": True, "displaylogo": False})


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=repository / "configs" / "nato_sers_identifiability_v1.json")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    root = repository / config["input_workspace"]; output = repository / config["output_workspace"]
    tables = output / "tables"; html_dir = output / "html"; tables.mkdir(parents=True, exist_ok=True); html_dir.mkdir(exist_ok=True)
    manifest = pd.read_csv(root / "tables" / "analysis_manifest.csv")
    bundle = np.load(repository / config["preprocessing_workspace"] / "preprocessing_candidates.npz")
    axis = bundle["axis_cm1"]; low, high = config["range_cm1"]; mask = (axis >= low) & (axis <= high)
    matrices = {name: bundle[name][:, mask] for name in config["preprocessing"]}
    support = manifest.groupby(["station", "target_analyte"]).agg(
        spectra=("observation_uid", "size"), masters=("master_sample_id", "nunique"),
        instruments=("instrument", "nunique"), sensors=("sensor_family", "nunique")).reset_index()
    support.to_csv(tables / "station_task_support.csv", index=False)
    closed = station_tasks(manifest, matrices, config); closed.to_csv(tables / "within_station_fold_metrics.csv", index=False)
    summary = closed.groupby(["station", "preprocessing", "model"]).agg(
        spectrum_balanced_accuracy_mean=("spectrum_balanced_accuracy", "mean"),
        spectrum_balanced_accuracy_std=("spectrum_balanced_accuracy", "std"),
        master_balanced_accuracy_mean=("master_balanced_accuracy", "mean"),
        master_balanced_accuracy_std=("master_balanced_accuracy", "std")).reset_index()
    summary.to_csv(tables / "within_station_summary.csv", index=False)
    meta = station_metadata_baseline(manifest, config); meta.to_csv(tables / "within_station_metadata_baseline.csv", index=False)
    cross = cross_station_binary(manifest, matrices, config); cross.to_csv(tables / "cross_station_binary.csv", index=False)
    shift = leave_instrument_within_station(manifest, matrices, config); shift.to_csv(tables / "within_station_leave_instrument_out.csv", index=False)
    artifacts = []
    def save(name, title, description, fig):
        write_html(html_dir / name, fig); artifacts.append({"file": name, "title": title, "description": description})
    for metric, level in (("spectrum_balanced_accuracy_mean", "spectrum"), ("master_balanced_accuracy_mean", "master")):
        fig = go.Figure()
        for idx, ((prep, model), group) in enumerate(summary.groupby(["preprocessing", "model"])):
            fig.add_trace(go.Bar(x=group.station, y=group[metric], name=f"{prep} | {model}"))
        fig.update_layout(barmode="group", title=f"Three-class within-station {level}-level classification",
                          yaxis_title="Balanced accuracy")
        save(f"0{len(artifacts)+1}_within_station_{level}.html", f"Within-station {level} classification",
             "Chemistry is compared only within a station, preventing station identity from solving the task.", fig)
    fig = go.Figure()
    for idx, ((prep, model), group) in enumerate(cross.groupby(["preprocessing", "model"])):
        fig.add_trace(go.Bar(x=group.apply(lambda r:f"{r.train_station} → {r.test_station}",axis=1),
                             y=group.spectrum_balanced_accuracy, name=f"{prep} | {model}"))
    fig.update_layout(barmode="group", title="4-ANPP vs benzyl-fentanyl cross-station transfer",
                      yaxis_title="Spectrum balanced accuracy")
    save("03_cross_station_binary.html", "Cross-station binary transfer",
         "Only the two chemicals measured at both pills and surfaces are included.", fig)
    for station in sorted(shift.station.unique()):
        group = shift[shift.station.eq(station)]; fig = go.Figure()
        for idx, ((prep, model), values) in enumerate(group.groupby(["preprocessing", "model"])):
            fig.add_trace(go.Bar(x=values.held_out_instrument, y=values.balanced_accuracy,
                                 name=f"{prep} | {model}"))
        fig.update_layout(barmode="group", title=f"{station}: unseen instrument + unseen master",
                          yaxis_title="Balanced accuracy")
        save(f"0{len(artifacts)+1}_{station}_instrument_shift.html", f"{station} acquisition shift",
             "Only held instruments containing at least two target classes are reported.", fig)
    (output / "html_artifact_registry.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    links = "".join(f'<li><a href="html/{a["file"]}">{a["title"]}</a> — {a["description"]}</li>' for a in artifacts)
    (output / "index.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>NATO SERS identifiability</title>"
        "<style>body{font:16px Arial;max-width:1100px;margin:40px auto;line-height:1.5}li{margin:12px 0}</style></head>"
        f"<body><h1>Chemistry-identifiable task audit</h1><ol>{links}</ol></body></html>")
    provenance = {"protocol_version": config["protocol_version"],
                  "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "preprocessing_bundle_sha256": hashlib.sha256((repository / config["preprocessing_workspace"] / "preprocessing_candidates.npz").read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False)); print("\nCROSS STATION\n", cross.to_string(index=False))


if __name__ == "__main__":
    main()
