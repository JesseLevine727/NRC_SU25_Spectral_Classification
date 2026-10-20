"""Extract the aggregate tables behind the 2026-10-21 supervisor-meeting deck.

Public inputs are the released P08-U1 and P13 aggregate tables. Private inputs
(the P01 manifest and processed arrays) are used only for coverage counts and a
descriptive PCA; anything row-level is written to the git-ignored private_data/.

Scoring follows the 14 September report: the pooled four-fold estimand, so the
minimal-preprocessing Extra Trees score is the same 68.5% Li-Lin has already seen.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1]
P08 = ATLAS / "results/p08_universal/release/tables"
P13 = ATLAS / "results/p13_portability/tables"
# Private P01 run (processed spectra + manifest); located via the project's artifact-root variable.
P01_RUN = Path(os.environ["ATLAS_ARTIFACT_ROOT"]) / "p01/runs/P01-ba26c5a50087107e314912f7"

DATA = HERE / "data"
PRIVATE = HERE / "private_data"
DATA.mkdir(exist_ok=True)
PRIVATE.mkdir(exist_ok=True)

ESTIMAND = "pooled_four_fold"
MODELS = ["C-EXTRA-TREES", "C-RANDOM-FOREST", "C-RBF-SVM", "D0-M", "P05-SELECTED"]
MODEL_LABEL = {
    "C-EXTRA-TREES": "Extra Trees",
    "C-RANDOM-FOREST": "Random Forest",
    "C-RBF-SVM": "SVM",
    "D0-M": "CNN",
    "P05-SELECTED": "CNN (selected recipe)",
}
POLICY_LABEL = {"PP-U-MIN": "minimal", "PP-U-SG": "smoothed", "PP-U-ARPLS": "baseline"}
STATION_LABEL = {"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"}
CHEM_LABEL = {
    "4_nitrophenol": "4-nitrophenol",
    "ethanol": "ethanol",
    "ethyl_paraoxon": "ethyl paraoxon",
    "4_ANPP": "4-ANPP",
    "benzyl_fentanyl": "benzyl fentanyl",
    "blank": "blank",
    "acetaminophen": "acetaminophen",
}
STATION_CLASSES = {
    "cwa": ["4_nitrophenol", "ethanol", "ethyl_paraoxon"],
    "pills": ["4_ANPP", "benzyl_fentanyl", "blank"],
    "surfaces": ["4_ANPP", "acetaminophen", "benzyl_fentanyl"],
}
SUBSTRATE_LABEL = {
    "pSERS_Metrohm_silver": "pSERS silver",
    "H_SERS_H_Kit": "H-Kit",
    "NRC_Canadian_SERS": "NRC",
    "GaN_polymer": "GaN/polymer",
}
INSTRUMENTS = ["Agilent-1", "Agilent-3", "Mira-1", "Mira-2", "Mira-3",
               "Pendar-1", "Pendar-2", "Pendar-3", "RMX-1", "RMX-2"]


def model_by_preprocessing():
    ms = pd.read_csv(P08 / "model_summary.csv")
    ms = ms[(ms.estimand == ESTIMAND) & ms.model_id.isin(MODELS)]
    out = ms.pivot_table(index="model_id", columns=["endpoint", "policy_id"], values="balanced_accuracy")
    rows = []
    for i, m in enumerate(MODELS):
        rows.append({
            "y": len(MODELS) - i,
            "model": MODEL_LABEL[m],
            **{f"{ep}_{POLICY_LABEL[p]}": round(100 * out.loc[m, (ep, p)], 1)
               for ep in ("M01", "M06") for p in POLICY_LABEL},
        })
    pd.DataFrame(rows).to_csv(DATA / "model_by_preprocessing.csv", index=False)
    return pd.DataFrame(rows)


def confusion_tables():
    cf = pd.read_csv(P08 / "confusion.csv")
    cf = cf[(cf.estimand == ESTIMAND) & (cf.endpoint == "M01")]
    long_rows = []
    for policy in ("PP-U-ARPLS", "PP-U-MIN"):
        for station, classes in STATION_CLASSES.items():
            for m in MODELS:
                sub = cf[(cf.policy_id == policy) & (cf.station == station) & (cf.model_id == m)]
                for ti, t in enumerate(classes):
                    row = sub[sub.true_label == t]
                    total = row["count"].sum()
                    for pi, p in enumerate(classes):
                        n = row.loc[row.predicted_label == p, "count"].sum()
                        long_rows.append({
                            "policy": POLICY_LABEL[policy], "station": station, "model_id": m,
                            "true": t, "pred": p, "x": pi, "y": len(classes) - 1 - ti,
                            "pct": round(100 * n / total, 1) if total else np.nan,
                        })
    long = pd.DataFrame(long_rows)
    long.to_csv(DATA / "confusion_long.csv", index=False)
    for (policy, station, m), g in long.groupby(["policy", "station", "model_id"]):
        g[["x", "y", "pct"]].to_csv(DATA / f"cm_{policy}_{station}_{m}.csv", index=False)
    # Per-class recall is the confusion-matrix diagonal, so the two views agree exactly.
    diag = long[long.true == long.pred].copy()
    diag["station_label"] = diag.station.map(STATION_LABEL)
    diag["class_label"] = diag.true.map(CHEM_LABEL)
    diag["model"] = diag.model_id.map(MODEL_LABEL)
    wide = diag.pivot_table(index=["station", "true"], columns=["policy", "model_id"], values="pct")
    wide.to_csv(DATA / "per_class_recall.csv")
    return long


def per_instrument():
    dm = pd.read_csv(P08 / "domain_metrics.csv")
    dm = dm[(dm.estimand == ESTIMAND) & (dm.endpoint == "M01")]
    rows = []
    for m in MODELS:
        sub = dm[dm.model_id == m].pivot_table(index=["station", "instrument"], columns="policy_id",
                                               values="balanced_accuracy").reset_index()
        for _, r in sub.iterrows():
            rows.append({"model_id": m, "station": r.station, "instrument": r.instrument,
                         "minimal": round(100 * r["PP-U-MIN"], 1),
                         "baseline": round(100 * r["PP-U-ARPLS"], 1),
                         "change": round(100 * (r["PP-U-ARPLS"] - r["PP-U-MIN"]), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "per_instrument.csv", index=False)
    for st in STATION_CLASSES:
        out[(out.model_id == "C-EXTRA-TREES") & (out.station == st)][["minimal", "baseline", "instrument"]].to_csv(
            DATA / f"scatter_et_{st}.csv", index=False)
    return out


def substrate():
    dm = pd.read_csv(P13 / "domain_metrics.csv")
    conf = dm[dm.support_tier == "confirmatory"]
    keep = conf[conf.procedure_id.isin(["C-EXTRA-TREES", "C-SELECTED"])]
    rows = []
    order = (keep[["station", "substrate_family", "held_instrument"]].drop_duplicates()
             .sort_values(["substrate_family", "station", "held_instrument"]).reset_index(drop=True))
    for i, d in order.iterrows():
        rec = {"y": len(order) - i, "station": STATION_LABEL[d.station],
               "substrate": SUBSTRATE_LABEL[d.substrate_family], "instrument": d.held_instrument}
        for proc, tag in (("C-EXTRA-TREES", "et"), ("C-SELECTED", "sel")):
            for pol, ptag in (("PP-U-MIN", "min"), ("PP-U-ARPLS", "arpls")):
                r = keep[(keep.procedure_id == proc) & (keep.policy_id == pol) &
                         (keep.station == d.station) & (keep.substrate_family == d.substrate_family) &
                         (keep.held_instrument == d.held_instrument)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ok = r.endpoint_status == "complete"
                rec[f"{tag}_{ptag}_ba"] = round(100 * r.held_balanced_accuracy, 1) if ok else np.nan
                rec[f"{tag}_{ptag}_lo"] = round(100 * r.held_lcb95, 1) if ok else np.nan
                rec[f"{tag}_{ptag}_hi"] = round(100 * r.held_ucb95, 1) if ok else np.nan
                rec[f"{tag}_{ptag}_state"] = r.bounded_state
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "substrate_domains.csv", index=False)
    fl = pd.read_csv(P13 / "field_log_results.csv")
    fl = fl[fl.procedure_id == "C-SELECTED"][["substrate_family", "endpoint_type", "definite_views",
                                              "successful_views", "complete_case_estimate",
                                              "worst_case_missing_bound", "best_case_missing_bound"]]
    fl.to_csv(DATA / "field_log.csv", index=False)
    return out


def private_inputs():
    man = pd.read_csv(P01_RUN / "primary_manifest.csv", low_memory=False)
    # Coverage counts (aggregate only).
    cov = man.groupby(["sensor_family", "instrument"]).size().unstack(fill_value=0)
    cov = cov.reindex(columns=INSTRUMENTS, fill_value=0).reindex(list(SUBSTRATE_LABEL), fill_value=0)
    cov_long = [{"x": xi, "y": 3 - yi, "n": int(cov.iloc[yi, xi])}
                for yi in range(cov.shape[0]) for xi in range(cov.shape[1])]
    pd.DataFrame(cov_long).to_csv(DATA / "coverage_substrate_instrument.csv", index=False)
    chem_order = ["4_nitrophenol", "ethanol", "ethyl_paraoxon", "4_ANPP", "benzyl_fentanyl", "acetaminophen", "blank"]
    cs = man.groupby(["target_analyte", "sensor_family"]).master_sample_id.nunique().unstack(fill_value=0)
    cs = cs.reindex(index=chem_order, columns=list(SUBSTRATE_LABEL), fill_value=0)
    cs_long = [{"x": xi, "y": len(chem_order) - 1 - yi, "n": int(cs.iloc[yi, xi])}
               for yi in range(cs.shape[0]) for xi in range(cs.shape[1])]
    pd.DataFrame(cs_long).to_csv(DATA / "coverage_chemical_substrate.csv", index=False)
    ci = man.groupby(["target_analyte"]).instrument.nunique().reindex(chem_order)
    ci.to_csv(DATA / "instruments_per_chemical.csv")
    # Descriptive within-station PCA (row-level -> private).
    for rep, tag in (("R_MIN_400_1800", "minimal"), ("R_ARPLS_400_1800", "baseline")):
        z = np.load(P01_RUN / "representations" / f"{rep}.npz")
        frame = pd.DataFrame({"observation_uid": z["observation_uid"]})
        frame = frame.merge(man[["observation_uid", "station", "instrument", "target_analyte",
                                 "master_sample_id"]], on="observation_uid", how="left", validate="one_to_one")
        x = z["intensity"].astype(float)
        for st in STATION_CLASSES:
            mask = (frame.station == st).to_numpy()
            pca = PCA(n_components=2).fit(x[mask])
            pcs = pca.transform(x[mask])
            sub = frame[mask].copy()
            sub["pc1"], sub["pc2"] = pcs[:, 0], pcs[:, 1]
            sub["brand"] = sub.instrument.str.split("-").str[0]
            sub[["pc1", "pc2", "brand", "instrument", "target_analyte", "master_sample_id"]].to_csv(
                PRIVATE / f"pca_{tag}_{st}.csv", index=False)
            for b, g in sub.groupby("brand"):
                g[["pc1", "pc2"]].to_csv(PRIVATE / f"pca_{tag}_{st}_brand_{b}.csv", index=False)
            for c, g in sub.groupby("target_analyte"):
                g[["pc1", "pc2"]].to_csv(PRIVATE / f"pca_{tag}_{st}_chem_{c}.csv", index=False)
            print(f"PCA {tag} {st}: n={mask.sum()} var={np.round(100 * pca.explained_variance_ratio_, 1)}")


if __name__ == "__main__":
    print(model_by_preprocessing().to_string(index=False))
    confusion_tables()
    pi = per_instrument()
    print(pi[pi.model_id == "C-EXTRA-TREES"].to_string(index=False))
    print(substrate()[["substrate", "station", "instrument", "et_min_ba", "et_arpls_ba", "et_arpls_state"]].to_string(index=False))
    private_inputs()
