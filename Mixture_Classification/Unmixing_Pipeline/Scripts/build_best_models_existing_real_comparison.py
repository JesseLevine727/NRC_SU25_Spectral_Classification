from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "Scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from evaluate_pt2_augmented_existing_real import (  # noqa: E402
    _build_existing_real_dataset,
    _load_augmented_models,
)
from retrain_with_pt2_reference import load_reference  # noqa: E402


RESULTS_DIR = REPO_ROOT / "Unmixing_Pipeline" / "Results" / "best_models_existing_real_comparison"
CLASSICAL_PREDICTIONS = (
    REPO_ROOT
    / "Unmixing_Pipeline"
    / "Results"
    / "pair_nnls_family_fallback"
    / "family_margin_0p002"
    / "predictions.csv"
)
DEEP_PREDICTIONS = (
    REPO_ROOT
    / "Unmixing_Pipeline"
    / "Results"
    / "deep_similarity_supervision"
    / "baseline_corrected"
    / "existing_real_predictions.csv"
)
SIAMESE_THRESHOLDS = (
    REPO_ROOT
    / "Notebooks"
    / "pt2_augmented_existing_real_recalibrated"
    / "calibrated_thresholds_17class.json"
)

ALL_CLASSES = [
    "1,9-nonanedithiol",
    "1-dodecanethiol",
    "1-undecanethiol",
    "6-mercapto-1-hexanol",
    "acetonitrile",
    "benzene",
    "benzenethiol",
    "dichloromethane",
    "diethylamine",
    "dmmp",
    "etoh",
    "meoh",
    "n,n-dimethylformamide",
    "n-hexane",
    "pyridine",
    "toluene",
    "tris(2-ethylhexyl) phosphate",
]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def parse_labels(value: str) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(" + ") if part.strip()]


def to_indicator_matrix(df: pd.DataFrame, true_col: str, pred_col: str) -> tuple[np.ndarray, np.ndarray]:
    class_to_idx = {name: idx for idx, name in enumerate(ALL_CLASSES)}
    y_true = np.zeros((len(df), len(ALL_CLASSES)), dtype=np.int64)
    y_pred = np.zeros((len(df), len(ALL_CLASSES)), dtype=np.int64)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for label in parse_labels(row[true_col]):
            y_true[row_idx, class_to_idx[label]] = 1
        for label in parse_labels(row[pred_col]):
            y_pred[row_idx, class_to_idx[label]] = 1
    return y_true, y_pred


def compute_micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = float(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    exact_match = float(np.all(y_true == y_pred, axis=1).mean())
    return {
        "exact_match": exact_match,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "avg_predicted_chemicals": float(y_pred.sum(axis=1).mean()),
        "avg_true_chemicals": float(y_true.sum(axis=1).mean()),
    }


def compute_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    predicted_support = y_pred.sum(axis=0)
    return pd.DataFrame(
        {
            "chemical": ALL_CLASSES,
            "support": support.astype(int),
            "predicted_support": predicted_support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def load_classical() -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.read_csv(CLASSICAL_PREDICTIONS)
    df = df[df["dataset"] == "existing_real"].copy()
    y_true, y_pred = to_indicator_matrix(df, "true_labels", "predicted_labels")
    return compute_micro_metrics(y_true, y_pred), compute_per_class(y_true, y_pred)


def load_deep() -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.read_csv(DEEP_PREDICTIONS)
    y_true, y_pred = to_indicator_matrix(df, "true_labels", "predicted_labels")
    return compute_micro_metrics(y_true, y_pred), compute_per_class(y_true, y_pred)


def load_calibrated_siamese() -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
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
    if classes != ALL_CLASSES:
        raise RuntimeError("Expanded class ordering does not match expected 17-class layout.")

    thresholds_dict = json.loads(SIAMESE_THRESHOLDS.read_text())
    thresholds = np.asarray([thresholds_dict[name] for name in classes], dtype=np.float32)
    y_true = existing["y_real"]
    probs = existing["probs_all"]
    y_pred = (probs >= thresholds).astype(np.int64)

    sample_rows: list[dict[str, object]] = []
    for row_idx, row in existing["mix_df"].iterrows():
        true_labels = [
            label
            for label, flag in zip(classes, y_true[row_idx].tolist())
            if flag
        ]
        predicted_labels = [
            label
            for label, flag in zip(classes, y_pred[row_idx].tolist())
            if flag
        ]
        sample_rows.append(
            {
                "sample_id": f"existing/{row_idx}",
                "source": "existing_real",
                "true_labels": " + ".join(true_labels),
                "predicted_labels": " + ".join(predicted_labels),
                "num_predicted_chemicals": int(y_pred[row_idx].sum()),
            }
        )

    predictions_df = pd.DataFrame(sample_rows)
    return (
        compute_micro_metrics(y_true, y_pred),
        compute_per_class(y_true, y_pred),
        predictions_df,
    )


def build_summary_table(metrics_by_model: dict[str, dict[str, float]]) -> str:
    rows = []
    order = [
        "Best classical",
        "Best deep",
        "Calibrated Siamese+MLP 2-head",
    ]
    for model_name in order:
        metrics = metrics_by_model[model_name]
        rows.append(
            "        "
            + " & ".join(
                [
                    tex_escape(model_name),
                    fmt(metrics["exact_match"]),
                    fmt(metrics["micro_precision"]),
                    fmt(metrics["micro_recall"]),
                    fmt(metrics["micro_f1"]),
                    fmt(metrics["avg_predicted_chemicals"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_per_class_table(per_class_by_model: dict[str, pd.DataFrame]) -> str:
    rows = []
    classical = per_class_by_model["Best classical"].set_index("chemical")
    deep = per_class_by_model["Best deep"].set_index("chemical")
    siamese = per_class_by_model["Calibrated Siamese+MLP 2-head"].set_index("chemical")

    for chemical in ALL_CLASSES:
        c_row = classical.loc[chemical]
        d_row = deep.loc[chemical]
        s_row = siamese.loc[chemical]
        support = int(max(c_row["support"], d_row["support"], s_row["support"]))
        rows.append(
            "        "
            + " & ".join(
                [
                    tex_escape(chemical),
                    str(support),
                    fmt(float(c_row["precision"])),
                    fmt(float(c_row["recall"])),
                    fmt(float(c_row["f1"])),
                    fmt(float(d_row["precision"])),
                    fmt(float(d_row["recall"])),
                    fmt(float(d_row["f1"])),
                    fmt(float(s_row["precision"])),
                    fmt(float(s_row["recall"])),
                    fmt(float(s_row["f1"])),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_latex(
    metrics_by_model: dict[str, dict[str, float]],
    per_class_by_model: dict[str, pd.DataFrame],
) -> str:
    summary_rows = build_summary_table(metrics_by_model)
    per_class_rows = build_per_class_table(per_class_by_model)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{array}}
\usepackage{{parskip}}

\begin{{document}}

\begin{{center}}
{{\LARGE Full 17-Class Existing-Real Comparison}}\\[0.4em]
{{\large Calibrated Siamese+MLP 2-head vs Best Classical vs Best Deep}}
\end{{center}}

\textbf{{Scope.}} All three models are evaluated on the same 580-sample original real-mixture dataset from \texttt{{mixtures\_dataset.csv}}, and all inference is scored against the full 17-class PT2-expanded label space.

\textbf{{Important note on the Siamese row.}} The calibrated Siamese thresholds come from the saved notebook recalibration experiment. To produce a full 17-class per-class comparison, those saved thresholds are applied here to the full 580-sample set. That makes this row a direct description of calibrated full-set behavior, not a held-out post-calibration estimate.

\begin{{table}}[ht]
\centering
\caption{{Full 17-class summary on the original real-mixture dataset.}}
\begin{{tabular}}{{lccccc}}
\toprule
Model & Exact match & Micro-precision & Micro-recall & Micro-F1 & Avg. returned chemicals \\
\midrule
{summary_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{landscape}}
\begin{{center}}
\small
\begin{{longtable}}{{lrrrrrrrrrr}}
\caption{{Per-class precision, recall, and F1 under full 17-class inference on the original real-mixture dataset.}}\\
\toprule
Chemical & Support & \multicolumn{{3}}{{c}}{{Best classical}} & \multicolumn{{3}}{{c}}{{Best deep}} & \multicolumn{{3}}{{c}}{{Calibrated Siamese}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}}
 &  & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule
\endfirsthead
\toprule
Chemical & Support & \multicolumn{{3}}{{c}}{{Best classical}} & \multicolumn{{3}}{{c}}{{Best deep}} & \multicolumn{{3}}{{c}}{{Calibrated Siamese}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}}
 &  & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule
\endhead
{per_class_rows}
\bottomrule
\end{{longtable}}
\end{{center}}
\end{{landscape}}

\textbf{{How the best classical model works.}} The classical system starts from a replicate-aware reference dictionary over the full 17-compound library. Each candidate compound contributes multiple measured pure spectra rather than one prototype. For one test spectrum, the solver enumerates all binary compound pairs, appends a nuisance baseline atom, and solves a nonnegative least-squares fit for each pair. The lowest-residual pair is the base prediction. The specific ``best classical'' row shown here is the strongest engineering variant in the repo: it keeps that exhaustive pairwise NNLS core, but adds a selective baseline-dependent fallback and a narrow near-tie family fallback to rescue the remaining dodecanethiol/methanol confusions. It always returns exactly two chemicals.

\textbf{{How the best deep model works.}} The promoted deep model is the similarity-supervised coefficient regressor. It trains on baseline-corrected synthetic binary mixtures built from the same 17-class reference library. The network takes a single spectrum as input and predicts a 17-dimensional nonnegative coefficient vector, one coefficient per library chemical. Training combines coefficient supervision with similarity-aware structure so chemically close spectra still have to separate in the output space. At inference time the model ranks all 17 coefficients and returns the top two chemicals. It does not use pair-specific thresholds or chemistry-specific fallback rules.

\textbf{{How the calibrated Siamese+MLP 2-head works.}} The original deep baseline uses two Siamese encoders: one for the Raman spectrum and one for its FFT representation. Their embeddings are concatenated and passed to a 17-way multilabel presence head that predicts one score per chemical. The calibrated variant then applies per-class thresholds learned on a calibration subset of the original real-mixture dataset. Unlike the classical and coefficient-regression models, it is not constrained to output two chemicals, so its calibrated thresholds can increase recall at the cost of extra false positives and a larger average predicted support.

\textbf{{Reading the comparison.}} The full 17-class table makes the operating difference visible. The classical and deep models concentrate almost all of their remaining error inside the thiol/methanol neighborhood while staying sparse, whereas the calibrated Siamese model overpredicts several classes that never appear in the original dataset at all, including \texttt{{1,9-nonanedithiol}}, \texttt{{1-undecanethiol}}, \texttt{{diethylamine}}, and \texttt{{acetonitrile}}. That is why its recall is high but its micro-precision and exact-match rate collapse under full 17-class scoring.

\end{{document}}
"""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    obsolete = [
        RESULTS_DIR / "per_chemical_best_models_existing_real.csv",
    ]
    for path in obsolete:
        if path.exists():
            path.unlink()

    classical_metrics, classical_per_class = load_classical()
    deep_metrics, deep_per_class = load_deep()
    siamese_metrics, siamese_per_class, siamese_predictions = load_calibrated_siamese()

    metrics_by_model = {
        "Best classical": classical_metrics,
        "Best deep": deep_metrics,
        "Calibrated Siamese+MLP 2-head": siamese_metrics,
    }
    per_class_by_model = {
        "Best classical": classical_per_class,
        "Best deep": deep_per_class,
        "Calibrated Siamese+MLP 2-head": siamese_per_class,
    }

    summary_df = pd.DataFrame(
        [
            {"model": name, **metrics}
            for name, metrics in metrics_by_model.items()
        ]
    )
    summary_csv = RESULTS_DIR / "best_models_existing_real_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    per_class_merged = classical_per_class[["chemical", "support"]].copy()
    for model_name, model_df in per_class_by_model.items():
        slug = (
            model_name.lower()
            .replace("+", "plus")
            .replace("-", "_")
            .replace(" ", "_")
        )
        per_class_merged = per_class_merged.merge(
            model_df[["chemical", "precision", "recall", "f1"]].rename(
                columns={
                    "precision": f"{slug}_precision",
                    "recall": f"{slug}_recall",
                    "f1": f"{slug}_f1",
                }
            ),
            on="chemical",
            how="left",
        )
    per_class_csv = RESULTS_DIR / "best_models_existing_real_per_class_full17.csv"
    per_class_merged.to_csv(per_class_csv, index=False)

    siamese_predictions_csv = RESULTS_DIR / "siamese_calibrated_existing_real_predictions.csv"
    siamese_predictions.to_csv(siamese_predictions_csv, index=False)

    tex = build_latex(metrics_by_model, per_class_by_model)
    tex_path = RESULTS_DIR / "best_models_existing_real_comparison.tex"
    tex_path.write_text(tex)

    print(f"Saved summary CSV to {summary_csv}")
    print(f"Saved per-class CSV to {per_class_csv}")
    print(f"Saved Siamese predictions CSV to {siamese_predictions_csv}")
    print(f"Saved LaTeX comparison to {tex_path}")


if __name__ == "__main__":
    main()
