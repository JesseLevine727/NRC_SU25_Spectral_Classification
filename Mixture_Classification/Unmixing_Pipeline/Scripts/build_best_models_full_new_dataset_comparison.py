from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


MIXTURE_ROOT = Path(__file__).resolve().parents[2]
UNMIXING_ROOT = MIXTURE_ROOT / "Unmixing_Pipeline"
RESULTS_ROOT = UNMIXING_ROOT / "Results"
OUTPUT_DIR = RESULTS_ROOT / "best_models_full_new_dataset_comparison"
NOTEBOOK_ROOT = MIXTURE_ROOT / "Notebooks"


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


def normalize_label_string(text: str) -> str:
    labels = sorted(part.strip() for part in str(text).split(" + "))
    return " + ".join(labels)


def label_set(text: str) -> tuple[str, ...]:
    return tuple(sorted(part.strip() for part in str(text).split(" + ")))


def compute_multilabel_metrics(df: pd.DataFrame, classes: list[str]) -> dict[str, float]:
    class_to_i = {label: idx for idx, label in enumerate(classes)}
    y_true = np.zeros((len(df), len(classes)), dtype=int)
    y_pred = np.zeros_like(y_true)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        for label in label_set(row.true_labels):
            y_true[row_idx, class_to_i[label]] = 1
        for label in label_set(row.predicted_labels):
            y_pred[row_idx, class_to_i[label]] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return {
        "exact_match": exact,
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "mean_true_labels": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels": float(y_pred.sum(axis=1).mean()),
    }


def compute_per_mixture(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    for source in sorted(df["source"].unique()):
        sub = df.loc[df["source"] == source].copy()
        metrics = compute_multilabel_metrics(sub, classes)
        rows.append(
            {
                "source": source,
                "true_labels": normalize_label_string(sub.iloc[0]["true_labels"]),
                "samples": int(len(sub)),
                "exact_match": metrics["exact_match"],
                "micro_f1": metrics["micro_f1"],
                "mean_predicted_labels": metrics["mean_predicted_labels"],
            }
        )
    return pd.DataFrame(rows)


def load_model_frames() -> tuple[list[str], list[dict], dict[str, pd.DataFrame]]:
    siamese_summary = json.loads(
        (NOTEBOOK_ROOT / "pt2_augmented_reference_experiment" / "pt2_augmented_reference_summary.json").read_text()
    )
    classes = list(siamese_summary["expanded_classes"])

    classical_df = pd.read_csv(
        RESULTS_ROOT / "pair_nnls_family_fallback" / "family_margin_0p002" / "predictions.csv"
    )
    classical_df = classical_df.loc[classical_df["dataset"] == "pt2_real", ["sample_id", "source", "true_labels", "predicted_labels"]].copy()
    classical_df["true_labels"] = classical_df["true_labels"].map(normalize_label_string)
    classical_df["predicted_labels"] = classical_df["predicted_labels"].map(normalize_label_string)

    deep_df = pd.read_csv(
        RESULTS_ROOT / "deep_similarity_supervision" / "baseline_corrected" / "pt2_real_predictions.csv"
    )
    deep_df = deep_df[["sample_id", "source", "true_labels", "predicted_labels"]].copy()
    deep_df["true_labels"] = deep_df["true_labels"].map(normalize_label_string)
    deep_df["predicted_labels"] = deep_df["predicted_labels"].map(normalize_label_string)

    siamese_df = pd.read_csv(
        NOTEBOOK_ROOT / "pt2_augmented_reference_experiment" / "pt2_augmented_reference_predictions.csv"
    )
    siamese_df = siamese_df[["sample_id", "source", "true_labels", "predicted_labels"]].copy()
    siamese_df["true_labels"] = siamese_df["true_labels"].map(normalize_label_string)
    siamese_df["predicted_labels"] = siamese_df["predicted_labels"].map(normalize_label_string)

    model_frames = {
        "Best classical": classical_df,
        "Best deep": deep_df,
        "Original Siamese+MLP 2-head": siamese_df,
    }

    rows = []
    metadata = {
        "Best classical": {
            "family": "Classical",
            "class_space": "17-compound expanded library",
            "protocol": "Best overall classical variant on PT2, pairwise NNLS with family fallback",
            "notes": "Best overall classical on current repo state; the clean classical anchor also reaches 1.000 on PT2.",
        },
        "Best deep": {
            "family": "Deep",
            "class_space": "17-compound expanded library",
            "protocol": "Similarity-supervised coefficient regressor, binary top-2 support inference",
            "notes": "Best promoted pure-deep model without pair-specific rules.",
        },
        "Original Siamese+MLP 2-head": {
            "family": "Original deep pipeline",
            "class_space": "17-class sigmoid head",
            "protocol": "PT2-expanded Siamese Raman+FT retrain, fixed threshold 0.5",
            "notes": "Original two-head Siamese+MLP architecture after expanding the reference to PT2 classes.",
        },
    }

    for name, df in model_frames.items():
        metrics = compute_multilabel_metrics(df, classes)
        rows.append(
            {
                "setup": name,
                "family": metadata[name]["family"],
                "class_space": metadata[name]["class_space"],
                "evaluation_protocol": metadata[name]["protocol"],
                "exact_match": metrics["exact_match"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
                "avg_pred_chemicals": metrics["mean_predicted_labels"],
                "notes": metadata[name]["notes"],
            }
        )

    return classes, rows, model_frames


def render_summary_table(summary_df: pd.DataFrame) -> str:
    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            " & ".join(
                [
                    tex_escape(row["setup"]),
                    tex_escape(row["family"]),
                    tex_escape(row["class_space"]),
                    tex_escape(row["evaluation_protocol"]),
                    fmt(row["exact_match"]),
                    fmt(row["micro_precision"]),
                    fmt(row["micro_recall"]),
                    fmt(row["micro_f1"]),
                    fmt(row["avg_pred_chemicals"], 2),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def render_per_mixture_table(per_mixture: dict[str, pd.DataFrame], method_order: list[str]) -> str:
    group_header = " & ".join(
        [rf"\multicolumn{{3}}{{c}}{{{tex_escape(name)}}}" for name in method_order]
    )
    cmidrules = " ".join(
        [rf"\cmidrule(lr){{{3 + 3*i}-{5 + 3*i}}}" for i in range(len(method_order))]
    )

    sources = list(per_mixture[method_order[0]]["source"])
    rows = []
    for source in sources:
        first = per_mixture[method_order[0]].set_index("source").loc[source]
        cells = [tex_escape(source), tex_escape(first["true_labels"])]
        for method in method_order:
            row = per_mixture[method].set_index("source").loc[source]
            cells.extend(
                [
                    fmt(float(row["exact_match"])),
                    fmt(float(row["micro_f1"])),
                    fmt(float(row["mean_predicted_labels"]), 2),
                ]
            )
        rows.append(" & ".join(cells) + r" \\")

    return rf"""
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{1.3cm}} >{{\raggedright\arraybackslash}}p{{3.3cm}} {' '.join(['c c c' for _ in method_order])}}}
\toprule
& & {group_header} \\
{cmidrules}
Mix & True labels & {' & '.join(['Exact & F1 & Avg. pred.' for _ in method_order])} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
"""


def build_method_explanations() -> str:
    return r"""
\section*{How Each Best Model Works}

\subsection*{Best Classical: Pair NNLS With Replicate Dictionary And Family Fallback}

The classical winner is still based on explicit spectral unmixing. For each candidate binary pair of compounds, the method builds a nonnegative linear model of the observed mixture spectrum using:
\begin{itemize}
\item replicate-level atoms for each candidate compound rather than only one class mean,
\item one nuisance baseline atom to absorb simple background structure,
\item nonnegative least squares to fit the candidate pair.
\end{itemize}

In symbols, for each candidate pair $(i, j)$ it solves a problem of the form
\[
x \approx A_{ij} c + b \mathbf{1} + e,
\]
where $x$ is the observed spectrum, $A_{ij}$ is the dictionary made from the replicate atoms of compounds $i$ and $j$, $c \ge 0$ are compound coefficients, $b \ge 0$ is a nuisance baseline coefficient, and $e$ is the residual. The solver evaluates every pair in the library and ranks them by reconstruction quality.

The version used here is the best overall classical result in the repo, so it adds one localized near-tie fallback after the NNLS search. That fallback does not change the underlying unmixing model; it only reranks very specific ambiguous pair solutions when the residual gap is tiny. This makes it the strongest classical number in the repo, though the clean scientific classical benchmark remains the replicate-aware pair NNLS anchor without that final family-specific rule.

\subsection*{Best Deep: Similarity-Supervised Coefficient Regressor}

The best promoted deep model is library-constrained rather than pair-identity based. It trains on synthetic binary mixtures generated directly from the expanded pure-spectrum library. Each synthetic sample is made by selecting two library compounds, mixing their spectra at a random ratio, adding mild intensity jitter and noise, and then applying the same preprocessing used at evaluation time.

The network is an MLP that reads the processed spectrum and outputs a nonnegative coefficient-share vector over the full library. Training uses several coupled signals:
\begin{itemize}
\item a coefficient regression loss so the two true compounds receive the right mixture shares,
\item a support loss so the active compounds are separated from the inactive ones,
\item a reconstruction loss through a fixed library decoder,
\item a similarity-aware penalty that pushes down false positives that are spectrally close to the true compounds.
\end{itemize}

At inference, the model does not threshold an arbitrary number of labels. Instead, because the real task here is binary mixtures, it takes the top two predicted compounds as the final support. This keeps the deployment framing aligned with the current dataset while still using a general library-level learned representation.

\subsection*{Original Siamese+MLP 2-Head Pipeline}

The original deep pipeline starts from synthetic binary mixtures too, but it learns mixture embeddings rather than explicit library coefficients. It has two Siamese-style branches:
\begin{itemize}
\item one branch processes the baseline-corrected Raman spectrum,
\item the second branch processes the Fourier-transformed representation.
\end{itemize}

Those learned embeddings are concatenated and passed to a multilabel MLP head with one sigmoid output per compound class. The model is then used as a multilabel presence detector: each class output is thresholded independently, and any class above the threshold is treated as present in the mixture.

That framing is weaker for this problem because it is not explicitly solving an additive unmixing problem. It is learning a representation and then using class-wise thresholding to infer support. In practice, on the PT2-expanded 17-class task, that produces too many extra labels on several real mixtures, which is why the project pivoted first to classical unmixing and then to library-constrained coefficient regression rather than continuing to optimize the old Siamese head.
"""


def build_latex(summary_df: pd.DataFrame, per_mixture: dict[str, pd.DataFrame], method_order: list[str]) -> str:
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
\caption{{Best classical, best deep, and original Siamese+MLP 2-head results on the full PT2 real-mixture dataset}}
\begin{{threeparttable}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{3.2cm}} >{{\raggedright\arraybackslash}}p{{1.7cm}} >{{\raggedright\arraybackslash}}p{{2.3cm}} >{{\raggedright\arraybackslash}}p{{4.3cm}} c c c c c}}
\toprule
Setup & Family & Class space & Evaluation protocol & Exact & Precision & Recall & Micro-F1 & Avg. pred. chemicals \\
\midrule
{render_summary_table(summary_df)}
\bottomrule
\end{{tabular}}
}}

\begin{{tablenotes}}\footnotesize
\item ``Full PT2 real-mixture dataset'' here means the nine real mixture families in \texttt{{Data/pt2}}: Mix 13 through Mix 22, for 324 measured mixture spectra total.
\item The classical row uses the strongest overall classical result in the repo. The clean classical benchmark also reaches perfect PT2 exact support recovery, but this row reflects the best classical number available in the current project state.
\item The deep row uses the strongest promoted pure-deep model rather than the hybrid deep-plus-classical reranker.
\item The Siamese row is the PT2-expanded 17-class retrain from \texttt{{pt2\_augmented\_reference\_experiment}}, evaluated with the default threshold 0.5 used in that experiment summary.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Per-mixture comparison on the PT2 real-mixture dataset}}
\begin{{threeparttable}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
{render_per_mixture_table(per_mixture, method_order)}
}}

\begin{{tablenotes}}\footnotesize
\item For each PT2 mixture family, the table reports exact support match, micro-F1, and mean predicted compound count.
\item The classical and deep best models are uniformly perfect on this PT2 evaluation set.
\item The original Siamese+MLP 2-head model succeeds on some PT2 mixtures but overpredicts support on several others, which is why its overall PT2 exact match remains low despite some perfect individual rows.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

{build_method_explanations()}

\end{{document}}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes, rows, model_frames = load_model_frames()
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "best_models_full_new_dataset_summary.csv", index=False)

    method_order = ["Best classical", "Best deep", "Original Siamese+MLP 2-head"]
    per_mixture = {}
    per_mix_rows = []
    for method in method_order:
        df = compute_per_mixture(model_frames[method], classes)
        per_mixture[method] = df
        tmp = df.copy()
        tmp.insert(0, "method", method)
        per_mix_rows.append(tmp)
        df.to_csv(
            OUTPUT_DIR / f"per_mixture_{method.lower().replace(' ', '_').replace('+', 'plus').replace('-', '_')}.csv",
            index=False,
        )

    pd.concat(per_mix_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "per_mixture_best_models.csv", index=False
    )

    latex = build_latex(summary_df, per_mixture, method_order)
    (OUTPUT_DIR / "best_models_full_new_dataset_comparison.tex").write_text(latex)

    print(f"Saved summary CSV to {OUTPUT_DIR / 'best_models_full_new_dataset_summary.csv'}")
    print(f"Saved LaTeX comparison to {OUTPUT_DIR / 'best_models_full_new_dataset_comparison.tex'}")


if __name__ == "__main__":
    main()
