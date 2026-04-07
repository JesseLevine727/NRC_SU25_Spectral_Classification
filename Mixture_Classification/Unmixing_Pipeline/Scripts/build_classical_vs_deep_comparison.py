from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "Results"
OUTPUT_DIR = RESULTS_ROOT / "classical_vs_deep_comparison"


ORIGINAL_SUPPORTED_CHEMICALS = [
    "1-dodecanethiol",
    "6-mercapto-1-hexanol",
    "benzene",
    "benzenethiol",
    "etoh",
    "meoh",
    "n,n-dimethylformamide",
    "pyridine",
]


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def build_summary_rows() -> list[dict]:
    rows: list[dict] = []

    exhaustive = load_json(RESULTS_ROOT / "exhaustive_pair_nnls" / "all_results.json")
    for mode in ("raw", "baseline_corrected"):
        result = exhaustive[mode]
        rows.append(
            {
                "group": "Classical",
                "setup": f"Exhaustive pair NNLS, {mode}",
                "family": "Pair NNLS",
                "protocol": "All binary pairs, class-mean dictionary",
                "existing_exact": result["existing_real"]["exact_match"],
                "existing_micro_f1": result["existing_real"]["micro_f1"],
                "existing_mean_pred": result["existing_real"]["mean_predicted_labels"],
                "pt2_exact": result["pt2_real"]["overall"]["exact_match"],
                "pt2_micro_f1": result["pt2_real"]["overall"]["micro_f1"],
                "pt2_mean_pred": result["pt2_real"]["overall"]["mean_predicted_labels"],
                "prediction_csv": RESULTS_ROOT / "exhaustive_pair_nnls" / mode / "existing_real_predictions.csv",
                "notes": "clean classical baseline",
            }
        )

    elastic = load_json(RESULTS_ROOT / "nonnegative_elastic_net" / "all_results.json")
    for mode in ("raw", "baseline_corrected"):
        result = elastic[mode]
        rows.append(
            {
                "group": "Classical",
                "setup": f"Non-negative elastic net, {mode}",
                "family": "Sparse regression",
                "protocol": "Full-library positive ElasticNet",
                "existing_exact": result["existing_real"]["exact_match"],
                "existing_micro_f1": result["existing_real"]["micro_f1"],
                "existing_mean_pred": result["existing_real"]["mean_predicted_labels"],
                "pt2_exact": result["pt2_real"]["overall"]["exact_match"],
                "pt2_micro_f1": result["pt2_real"]["overall"]["micro_f1"],
                "pt2_mean_pred": result["pt2_real"]["overall"]["mean_predicted_labels"],
                "prediction_csv": RESULTS_ROOT / "nonnegative_elastic_net" / mode / "existing_real_predictions.csv",
                "notes": "clean classical baseline",
            }
        )

    baseline_atoms = load_json(RESULTS_ROOT / "pair_nnls_with_baseline_atoms" / "all_results.json")
    best = baseline_atoms["best"]
    mode = best["mode"]
    degree = best["degree"]
    rows.append(
        {
            "group": "Classical",
            "setup": f"Pair NNLS + nuisance baseline atom, {mode}",
            "family": "Pair NNLS",
            "protocol": f"Bernstein baseline atoms, degree {degree}",
            "existing_exact": best["existing_real"]["exact_match"],
            "existing_micro_f1": best["existing_real"]["micro_f1"],
            "existing_mean_pred": best["existing_real"]["mean_predicted_labels"],
            "pt2_exact": best["pt2_real"]["overall"]["exact_match"],
            "pt2_micro_f1": best["pt2_real"]["overall"]["micro_f1"],
            "pt2_mean_pred": best["pt2_real"]["overall"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT / "pair_nnls_with_baseline_atoms" / f"{mode}_bernstein_deg_{degree}" / "predictions.csv",
            "notes": "best nuisance-baseline classical",
        }
    )

    replicate = load_json(RESULTS_ROOT / "pair_nnls_replicate_dictionary" / "all_results.json")["best"]
    reps = replicate["n_extra_representatives_per_compound"]
    rows.append(
        {
            "group": "Classical",
            "setup": f"Replicate-aware pair NNLS, {replicate['mode']}",
            "family": "Pair NNLS",
            "protocol": f"Constant baseline atom, extra reps {reps}",
            "existing_exact": replicate["existing_real"]["exact_match"],
            "existing_micro_f1": replicate["existing_real"]["micro_f1"],
            "existing_mean_pred": replicate["existing_real"]["mean_predicted_labels"],
            "pt2_exact": replicate["pt2_real"]["overall"]["exact_match"],
            "pt2_micro_f1": replicate["pt2_real"]["overall"]["micro_f1"],
            "pt2_mean_pred": replicate["pt2_real"]["overall"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT / "pair_nnls_replicate_dictionary" / f"baseline_corrected_extra_reps_{reps}" / "predictions.csv",
            "notes": "frozen clean classical benchmark",
        }
    )

    cardinality = load_json(RESULTS_ROOT / "cardinality_adaptive_nnls" / "summary.json")
    rows.append(
        {
            "group": "Classical",
            "setup": "Cardinality-adaptive NNLS",
            "family": "Sparse support search",
            "protocol": "Support size 1/2/3 with calibrated penalty",
            "existing_exact": cardinality["existing_real"]["exact_match"],
            "existing_micro_f1": cardinality["existing_real"]["micro_f1"],
            "existing_mean_pred": cardinality["existing_real"]["mean_predicted_labels"],
            "pt2_exact": cardinality["pt2_real"]["exact_match"],
            "pt2_micro_f1": cardinality["pt2_real"]["micro_f1"],
            "pt2_mean_pred": cardinality["pt2_real"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT / "cardinality_adaptive_nnls" / "existing_real_predictions.csv",
            "notes": "open-cardinality classical prototype",
        }
    )

    low_baseline = load_json(RESULTS_ROOT / "pair_nnls_baseline_fallback" / "all_results.json")["best"]
    baseline_rel = low_baseline["baseline_rel_threshold"]
    residual_margin = low_baseline["residual_margin"]
    rows.append(
        {
            "group": "Classical",
            "setup": "Pair NNLS + selective low-baseline fallback",
            "family": "Pair NNLS",
            "protocol": (
                "Global rerank on baseline-heavy pair "
                f"(thr {baseline_rel}, alt {low_baseline['alt_baseline_rel_max']}, margin {residual_margin})"
            ),
            "existing_exact": low_baseline["existing_real"]["exact_match"],
            "existing_micro_f1": low_baseline["existing_real"]["micro_f1"],
            "existing_mean_pred": low_baseline["existing_real"]["mean_predicted_labels"],
            "pt2_exact": low_baseline["pt2_real"]["exact_match"],
            "pt2_micro_f1": low_baseline["pt2_real"]["micro_f1"],
            "pt2_mean_pred": low_baseline["pt2_real"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT
            / "pair_nnls_baseline_fallback"
            / f"baseline_rel_{str(baseline_rel).replace('.', 'p')}_margin_{str(residual_margin).replace('.', 'p')}"
            / "predictions.csv",
            "notes": "diagnostic classical variant",
        }
    )

    family = load_json(RESULTS_ROOT / "pair_nnls_family_fallback" / "all_results.json")["best"]
    family_margin = family["family_fallback"]["family_margin"]
    rows.append(
        {
            "group": "Classical",
            "setup": "Pair NNLS + family near-tie fallback",
            "family": "Pair NNLS",
            "protocol": f"Global baseline fallback + family margin {family_margin}",
            "existing_exact": family["existing_real"]["exact_match"],
            "existing_micro_f1": family["existing_real"]["micro_f1"],
            "existing_mean_pred": family["existing_real"]["mean_predicted_labels"],
            "pt2_exact": family["pt2_real"]["exact_match"],
            "pt2_micro_f1": family["pt2_real"]["micro_f1"],
            "pt2_mean_pred": family["pt2_real"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT
            / "pair_nnls_family_fallback"
            / f"family_margin_{str(family_margin).replace('.', 'p')}"
            / "predictions.csv",
            "notes": "diagnostic classical ceiling",
        }
    )

    deep_base = load_json(RESULTS_ROOT / "deep_binary_coefficient_regressor" / "all_results.json")
    for mode in ("raw", "baseline_corrected"):
        result = deep_base[mode]
        rows.append(
            {
                "group": "Deep",
                "setup": f"Deep coefficient regressor, {mode}",
                "family": "MLP coefficient regressor",
                "protocol": "Synthetic binary mixtures, top-2 support inference",
                "existing_exact": result["existing_real_top2_binary"]["exact_match"],
                "existing_micro_f1": result["existing_real_top2_binary"]["micro_f1"],
                "existing_mean_pred": result["existing_real_top2_binary"]["mean_predicted_labels"],
                "pt2_exact": result["pt2_real_top2_binary"]["overall"]["exact_match"],
                "pt2_micro_f1": result["pt2_real_top2_binary"]["overall"]["micro_f1"],
                "pt2_mean_pred": result["pt2_real_top2_binary"]["overall"]["mean_predicted_labels"],
                "prediction_csv": RESULTS_ROOT / "deep_binary_coefficient_regressor" / mode / "existing_real_predictions.csv",
                "notes": "first clean deep baseline",
            }
        )

    deep_variants = load_json(RESULTS_ROOT / "deep_binary_variant_suite" / "all_results.json")
    variant_labels = {
        "cnn_encoder": "CNN encoder",
        "replicate_decoder": "Replicate-aware decoder",
    }
    for mode in ("raw", "baseline_corrected"):
        for variant, label in variant_labels.items():
            result = deep_variants[mode][variant]
            rows.append(
                {
                    "group": "Deep",
                    "setup": f"{label}, {mode}",
                    "family": "Deep variant suite",
                    "protocol": result["decoder_type"].replace("_", " "),
                    "existing_exact": result["existing_real_top2_binary"]["exact_match"],
                    "existing_micro_f1": result["existing_real_top2_binary"]["micro_f1"],
                    "existing_mean_pred": result["existing_real_top2_binary"]["mean_predicted_labels"],
                    "pt2_exact": result["pt2_real_top2_binary"]["overall"]["exact_match"],
                    "pt2_micro_f1": result["pt2_real_top2_binary"]["overall"]["micro_f1"],
                    "pt2_mean_pred": result["pt2_real_top2_binary"]["overall"]["mean_predicted_labels"],
                    "prediction_csv": RESULTS_ROOT / "deep_binary_variant_suite" / variant / mode / "existing_real_predictions.csv",
                    "notes": "negative deep variant",
                }
            )

    deep_supervision = load_json(RESULTS_ROOT / "deep_similarity_supervision" / "all_results.json")
    for mode in ("raw", "baseline_corrected"):
        result = deep_supervision[mode]
        rows.append(
            {
                "group": "Deep",
                "setup": f"Similarity-supervised deep regressor, {mode}",
                "family": "MLP coefficient regressor",
                "protocol": "Top-2 margin + similarity-weighted false-compound loss",
                "existing_exact": result["existing_real_top2_binary"]["exact_match"],
                "existing_micro_f1": result["existing_real_top2_binary"]["micro_f1"],
                "existing_mean_pred": result["existing_real_top2_binary"]["mean_predicted_labels"],
                "pt2_exact": result["pt2_real_top2_binary"]["overall"]["exact_match"],
                "pt2_micro_f1": result["pt2_real_top2_binary"]["overall"]["micro_f1"],
                "pt2_mean_pred": result["pt2_real_top2_binary"]["overall"]["mean_predicted_labels"],
                "prediction_csv": RESULTS_ROOT / "deep_similarity_supervision" / mode / "existing_real_predictions.csv",
                "notes": "best clean deep supervision",
            }
        )

    hybrid = load_json(RESULTS_ROOT / "deep_hybrid_pair_rerank" / "summary.json")
    rows.append(
        {
            "group": "Deep",
            "setup": "Deep + pair-NNLS hybrid rerank",
            "family": "Hybrid",
            "protocol": f"Global residual/prior fusion, alpha {hybrid['selected_alpha']}",
            "existing_exact": hybrid["hybrid_model"]["existing_real_top2_binary"]["exact_match"],
            "existing_micro_f1": hybrid["hybrid_model"]["existing_real_top2_binary"]["micro_f1"],
            "existing_mean_pred": hybrid["hybrid_model"]["existing_real_top2_binary"]["mean_predicted_labels"],
            "pt2_exact": hybrid["hybrid_model"]["pt2_real_top2_binary"]["overall"]["exact_match"],
            "pt2_micro_f1": hybrid["hybrid_model"]["pt2_real_top2_binary"]["overall"]["micro_f1"],
            "pt2_mean_pred": hybrid["hybrid_model"]["pt2_real_top2_binary"]["overall"]["mean_predicted_labels"],
            "prediction_csv": RESULTS_ROOT / "deep_hybrid_pair_rerank" / "existing_real_hybrid_predictions.csv",
            "notes": "global deep/classical fusion",
        }
    )

    return rows


def compute_per_chemical_metrics(prediction_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(prediction_csv)
    if "dataset" in pred.columns:
        pred = pred.loc[pred["dataset"] == "existing_real"].copy()
    supported = ORIGINAL_SUPPORTED_CHEMICALS
    rows = []
    for chem in supported:
        pattern = rf"(?:^| \+ ){chem}(?:$| \+ )"
        y_true = pred["true_labels"].str.contains(pattern, regex=True)
        y_pred = pred["predicted_labels"].fillna("").str.contains(pattern, regex=True)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.astype(int), y_pred.astype(int), average="binary", zero_division=0
        )
        rows.append(
            {
                "chemical": chem,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def render_summary_table(summary_df: pd.DataFrame) -> str:
    body = []
    for _, row in summary_df.iterrows():
        body.append(
            " & ".join(
                [
                    tex_escape(row["setup"]),
                    tex_escape(row["group"]),
                    tex_escape(row["protocol"]),
                    fmt(row["existing_exact"]),
                    fmt(row["existing_micro_f1"]),
                    fmt(row["existing_mean_pred"], 2),
                    fmt(row["pt2_exact"]),
                    fmt(row["pt2_micro_f1"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(body)


def render_per_chemical_table(per_chemical: dict[str, pd.DataFrame], selected_methods: list[str]) -> str:
    header_groups = " & ".join(
        [rf"\multicolumn{{3}}{{c}}{{{tex_escape(method)}}}" for method in selected_methods]
    )
    cmidrules = " ".join(
        [rf"\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(len(selected_methods))]
    )

    rows = []
    for chemical in ORIGINAL_SUPPORTED_CHEMICALS:
        row_parts = [tex_escape(chemical)]
        for method in selected_methods:
            metrics = per_chemical[method].set_index("chemical").loc[chemical]
            row_parts.extend([fmt(metrics["precision"]), fmt(metrics["recall"]), fmt(metrics["f1"])])
        rows.append(" & ".join(row_parts) + r" \\")

    return f"""
\\begin{{tabular}}{{>{{\\raggedright\\arraybackslash}}p{{3.2cm}} {' '.join(['c c c' for _ in selected_methods])}}}
\\toprule
& {header_groups} \\\\
{cmidrules}
Chemical & {' & '.join(['P & R & F1' for _ in selected_methods])} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
"""


def build_latex(summary_df: pd.DataFrame, per_chemical: dict[str, pd.DataFrame], selected_methods: list[str]) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{threeparttable}}
\usepackage{{caption}}
\usepackage{{graphicx}}

\begin{{document}}

\begin{{table}}[ht]
\centering
\caption{{Direct comparison of classical and deep mixture-classification results on the original and PT2 real-mixture datasets}}
\begin{{threeparttable}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{3.8cm}} >{{\raggedright\arraybackslash}}p{{1.3cm}} >{{\raggedright\arraybackslash}}p{{4.4cm}} c c c c c}}
\toprule
Setup & Family & Evaluation protocol & Existing exact & Existing Micro-F1 & Existing avg. pred. chemicals & PT2 exact & PT2 Micro-F1 \\
\midrule
{render_summary_table(summary_df)}
\bottomrule
\end{{tabular}}
}}

\begin{{tablenotes}}\footnotesize
\item The table includes the main comparable classical and deep runs saved under \texttt{{Unmixing\_Pipeline/Results}}.
\item ``Existing'' refers to the original real-mixture dataset from \texttt{{mixtures\_dataset.csv}}.
\item ``PT2'' refers to the later real-mixture dataset defined by \texttt{{Data/pt2/Mixtures.txt}} and the corresponding spectra folders.
\item The clean classical benchmark is the replicate-aware pair-NNLS row. The diagnostic classical ceiling is the family-fallback row.
\item The strongest clean deep family is the baseline-corrected MLP coefficient-regressor line and its similarity-supervised follow-up.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Per-chemical precision, recall, and F1 on the eight original supported chemicals for the main classical and deep contenders}}
\begin{{threeparttable}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
{render_per_chemical_table(per_chemical, selected_methods)}
}}

\begin{{tablenotes}}\footnotesize
\item ``Clean classical'' is the frozen replicate-aware pair-NNLS benchmark.
\item ``Classical ceiling'' is the localized family-fallback variant and is included as an engineering upper bound rather than the main scientific benchmark.
\item ``Deep baseline'' is the baseline-corrected deep coefficient regressor.
\item ``Deep supervision'' is the baseline-corrected similarity-supervised follow-up.
\item ``Hybrid'' is the baseline-corrected global deep-plus-pair-NNLS reranker.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\end{{document}}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_summary_rows()
    summary_df = pd.DataFrame(rows).sort_values(
        ["group", "existing_exact", "existing_micro_f1"], ascending=[True, False, False]
    )
    summary_df["prediction_csv"] = summary_df["prediction_csv"].astype(str)
    summary_df.to_csv(OUTPUT_DIR / "classical_vs_deep_summary.csv", index=False)

    selected = {
        "Clean classical": next(
            row["prediction_csv"] for row in rows if row["setup"] == "Replicate-aware pair NNLS, baseline_corrected"
        ),
        "Classical ceiling": next(
            row["prediction_csv"] for row in rows if row["setup"] == "Pair NNLS + family near-tie fallback"
        ),
        "Deep baseline": next(
            row["prediction_csv"] for row in rows if row["setup"] == "Deep coefficient regressor, baseline_corrected"
        ),
        "Deep supervision": next(
            row["prediction_csv"] for row in rows if row["setup"] == "Similarity-supervised deep regressor, baseline_corrected"
        ),
        "Hybrid": next(
            row["prediction_csv"] for row in rows if row["setup"] == "Deep + pair-NNLS hybrid rerank"
        ),
    }

    per_chemical = {}
    per_chemical_rows = []
    for label, csv_path in selected.items():
        metrics_df = compute_per_chemical_metrics(Path(csv_path))
        metrics_df.to_csv(
            OUTPUT_DIR / f"per_chemical_{label.lower().replace(' ', '_').replace('+', 'plus')}.csv",
            index=False,
        )
        per_chemical[label] = metrics_df
        tmp = metrics_df.copy()
        tmp.insert(0, "method", label)
        per_chemical_rows.append(tmp)

    pd.concat(per_chemical_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "per_chemical_selected_methods.csv", index=False
    )

    latex = build_latex(summary_df, per_chemical, list(selected.keys()))
    (OUTPUT_DIR / "classical_vs_deep_comparison.tex").write_text(latex)

    print(f"Saved summary CSV to {OUTPUT_DIR / 'classical_vs_deep_summary.csv'}")
    print(f"Saved LaTeX comparison to {OUTPUT_DIR / 'classical_vs_deep_comparison.tex'}")


if __name__ == "__main__":
    main()
