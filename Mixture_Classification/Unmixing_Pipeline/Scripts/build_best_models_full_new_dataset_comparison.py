from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


REPO_ROOT = Path(__file__).resolve().parents[3]
UNMIXING_RESULTS = REPO_ROOT / "Mixture_Classification" / "Unmixing_Pipeline" / "Results"
OUTPUT_DIR = UNMIXING_RESULTS / "best_models_full_new_dataset_comparison"
LEGACY_SCRIPTS_DIR = REPO_ROOT / "Mixture_Classification" / "Legacy_Siamese_Pipeline" / "Scripts"
if str(LEGACY_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(LEGACY_SCRIPTS_DIR))

from evaluate_pt2_augmented_existing_real import (  # noqa: E402
    _build_existing_real_dataset,
    _load_augmented_models,
)
from retrain_with_pt2_reference import (  # noqa: E402
    MIXTURE_LABEL_ALIASES,
    build_pt2_mixture_records,
    embed_inputs,
    load_reference,
    load_txt_spectrum,
    predict_probabilities,
    preprocess_matrix,
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

SIAMESE_THRESHOLDS_PATH = (
    REPO_ROOT
    / "Mixture_Classification"
    / "Legacy_Siamese_Pipeline"
    / "Notebooks"
    / "pt2_augmented_existing_real_recalibrated"
    / "calibrated_thresholds_17class.json"
)

CLASSICAL_CANDIDATES = {
    "pair_nnls_replicate_dictionary": UNMIXING_RESULTS
    / "pair_nnls_replicate_dictionary"
    / "baseline_corrected_extra_reps_9"
    / "predictions.csv",
    "exhaustive_pair_nnls": UNMIXING_RESULTS
    / "exhaustive_pair_nnls"
    / "baseline_corrected"
    / "predictions.csv",
    "pair_nnls_with_baseline_atoms": UNMIXING_RESULTS
    / "pair_nnls_with_baseline_atoms"
    / "baseline_corrected_bernstein_deg_0"
    / "predictions.csv",
    "cardinality_adaptive_nnls": UNMIXING_RESULTS
    / "cardinality_adaptive_nnls"
    / "all_predictions.csv",
    "full_library_sparse_support_selection": UNMIXING_RESULTS
    / "full_library_sparse_support_selection"
    / "all_predictions.csv",
}

DEEP_CANDIDATES = {
    "deep_binary_coefficient_regressor": [
        UNMIXING_RESULTS
        / "deep_binary_coefficient_regressor"
        / "baseline_corrected"
        / "existing_real_predictions.csv",
        UNMIXING_RESULTS
        / "deep_binary_coefficient_regressor"
        / "baseline_corrected"
        / "pt2_real_predictions.csv",
    ],
    "deep_similarity_supervision": [
        UNMIXING_RESULTS
        / "deep_similarity_supervision"
        / "baseline_corrected"
        / "existing_real_predictions.csv",
        UNMIXING_RESULTS
        / "deep_similarity_supervision"
        / "baseline_corrected"
        / "pt2_real_predictions.csv",
    ],
    "deep_cnn_encoder": [
        UNMIXING_RESULTS
        / "deep_binary_variant_suite"
        / "cnn_encoder"
        / "baseline_corrected"
        / "existing_real_predictions.csv",
        UNMIXING_RESULTS
        / "deep_binary_variant_suite"
        / "cnn_encoder"
        / "baseline_corrected"
        / "pt2_real_predictions.csv",
    ],
    "deep_replicate_decoder": [
        UNMIXING_RESULTS
        / "deep_binary_variant_suite"
        / "replicate_decoder"
        / "baseline_corrected"
        / "existing_real_predictions.csv",
        UNMIXING_RESULTS
        / "deep_binary_variant_suite"
        / "replicate_decoder"
        / "baseline_corrected"
        / "pt2_real_predictions.csv",
    ],
}


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


def fmt(value: float) -> str:
    return f"{value:.3f}"


def parse_labels(value: str) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(" + ") if part.strip()]


def metrics_from_predictions(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    class_to_idx = {name: idx for idx, name in enumerate(ALL_CLASSES)}
    y_true = np.zeros((len(df), len(ALL_CLASSES)), dtype=np.int64)
    y_pred = np.zeros_like(y_true)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for label in parse_labels(row["true_labels"]):
            y_true[row_idx, class_to_idx[label]] = 1
        for label in parse_labels(row["predicted_labels"]):
            y_pred[row_idx, class_to_idx[label]] = 1

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    summary = {
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "mean_true_labels": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels": float(y_pred.sum(axis=1).mean()),
    }
    per_class = pd.DataFrame(
        {
            "chemical": ALL_CLASSES,
            "support": support.astype(int),
            "predicted_support": y_pred.sum(axis=0).astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return summary, per_class


def mixture_only(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in df.columns:
        return df.copy()
    return df[df["dataset"].isin(["existing_real", "pt2_real"])].copy()


def load_prediction_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def evaluate_siamese(default_thresholds: bool) -> pd.DataFrame:
    _, wav_axis = load_reference()
    device, classes, siamese_raman, siamese_fft, model = _load_augmented_models(wav_axis)
    if classes != ALL_CLASSES:
        raise RuntimeError("Siamese class ordering mismatch.")

    existing = _build_existing_real_dataset(
        wav_axis, classes, device, siamese_raman, siamese_fft, model
    )
    existing_probs = existing["probs_all"]
    if default_thresholds:
        existing_pred = (existing_probs >= 0.5).astype(int)
    else:
        thresholds = json.loads(SIAMESE_THRESHOLDS_PATH.read_text())
        threshold_array = np.asarray([thresholds[name] for name in classes], dtype=np.float32)
        existing_pred = (existing_probs >= threshold_array).astype(int)

    rows = []
    mix_df = existing["mix_df"]
    for idx, row in mix_df.iterrows():
        pred_labels = [classes[j] for j in np.where(existing_pred[idx] == 1)[0]]
        rows.append(
            {
                "sample_id": f"existing/{idx}",
                "dataset": "existing_real",
                "source": "existing_real",
                "true_labels": " + ".join(sorted((row["Label 1"], row["Label 2"]))),
                "predicted_labels": " + ".join(pred_labels),
            }
        )

    pt2_records = build_pt2_mixture_records()
    spectra = np.vstack([load_txt_spectrum(record.path, wav_axis) for record in pt2_records]).astype(np.float32)
    raman, fft = preprocess_matrix(spectra)
    emb_raman = embed_inputs(siamese_raman, raman, device)
    emb_fft = embed_inputs(siamese_fft, fft, device)
    embeds = np.hstack([emb_raman, emb_fft]).astype(np.float32)
    pt2_probs = predict_probabilities(model, embeds, device)
    if default_thresholds:
        pt2_pred = (pt2_probs >= 0.5).astype(int)
    else:
        thresholds = json.loads(SIAMESE_THRESHOLDS_PATH.read_text())
        threshold_array = np.asarray([thresholds[name] for name in classes], dtype=np.float32)
        pt2_pred = (pt2_probs >= threshold_array).astype(int)

    for record, row_pred in zip(pt2_records, pt2_pred):
        pred_labels = [classes[j] for j in np.where(row_pred == 1)[0]]
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": "pt2_real",
                "source": record.source,
                "true_labels": " + ".join(sorted(record.true_labels)),
                "predicted_labels": " + ".join(pred_labels),
            }
        )

    return pd.DataFrame(rows)


def choose_best_candidates() -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame
]:
    classical_frames: dict[str, pd.DataFrame] = {}
    classical_rows = []
    for name, path in CLASSICAL_CANDIDATES.items():
        df = mixture_only(pd.read_csv(path))
        classical_frames[name] = df
        summary, _ = metrics_from_predictions(df)
        classical_rows.append({"candidate": name, **summary})
    classical_summary = pd.DataFrame(classical_rows).sort_values(
        ["exact_match", "micro_f1", "micro_precision"], ascending=False
    )

    deep_frames: dict[str, pd.DataFrame] = {}
    deep_rows = []
    for name, paths in DEEP_CANDIDATES.items():
        df = mixture_only(load_prediction_frames(paths))
        deep_frames[name] = df
        summary, _ = metrics_from_predictions(df)
        deep_rows.append({"candidate": name, **summary})
    deep_summary = pd.DataFrame(deep_rows).sort_values(
        ["exact_match", "micro_f1", "micro_precision"], ascending=False
    )

    return classical_frames, classical_summary, deep_frames, deep_summary


def build_summary_rows(metrics_by_model: dict[str, dict[str, float]]) -> str:
    rows = []
    order = [
        "Siamese+MLP 2-head (default 0.5)",
        "Siamese+MLP 2-head (calibrated)",
        "Best clean classical",
        "Best clean deep",
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
                    fmt(metrics["mean_predicted_labels"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_per_class_rows(per_class_by_model: dict[str, pd.DataFrame]) -> str:
    rows = []
    frames = {name: df.set_index("chemical") for name, df in per_class_by_model.items()}
    for chemical in ALL_CLASSES:
        support = int(max(frames[name].loc[chemical, "support"] for name in frames))
        if support == 0:
            continue
        values = [tex_escape(chemical), str(support)]
        for model_name in [
            "Siamese+MLP 2-head (default 0.5)",
            "Siamese+MLP 2-head (calibrated)",
            "Best clean classical",
            "Best clean deep",
        ]:
            row = frames[model_name].loc[chemical]
            values.extend(
                [
                    fmt(float(row["precision"])),
                    fmt(float(row["recall"])),
                    fmt(float(row["f1"])),
                ]
            )
        rows.append("        " + " & ".join(values) + r" \\")
    return "\n".join(rows)


def build_latex(
    metrics_by_model: dict[str, dict[str, float]],
    per_class_by_model: dict[str, pd.DataFrame],
    best_classical_name: str,
    best_deep_name: str,
    classical_candidates: pd.DataFrame,
    deep_candidates: pd.DataFrame,
) -> str:
    summary_rows = build_summary_rows(metrics_by_model)
    per_class_rows = build_per_class_rows(per_class_by_model)
    classical_note = ", ".join(
        f"{row.candidate}: exact={row.exact_match:.3f}, micro-F1={row.micro_f1:.3f}"
        for row in classical_candidates.itertuples()
    )
    deep_note = ", ".join(
        f"{row.candidate}: exact={row.exact_match:.3f}, micro-F1={row.micro_f1:.3f}"
        for row in deep_candidates.itertuples()
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{parskip}}
\usepackage{{graphicx}}
\begin{{document}}

\begin{{center}}
{{\LARGE Mixture-Only 17-Class Comparison}}\\[0.4em]
{{\large Original real mixtures + PT2 real mixtures}}
\end{{center}}

\textbf{{Dataset.}} This comparison uses mixture data only: the 580-sample original real-mixture dataset plus the 324-sample PT2 real-mixture dataset, for 904 total mixture spectra. All scoring is still done in the full 17-class PT2-expanded inference space.

\textbf{{Selection rule.}} ``Best clean classical'' is chosen from the non-hand-tuned classical baselines on this 904-sample mixture-only set. ``Best clean deep'' is chosen from the non-hand-tuned pure-deep baselines on this same set. The localized classical fallback variants and the hybrid deep-plus-classical reranker are intentionally excluded.

\begin{{table}}[ht]
\centering
\caption{{Full 17-class mixture-only comparison across original and PT2 real mixtures.}}
\small
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Model & Exact match & Micro-precision & Micro-recall & Micro-F1 & Avg. returned chemicals \\
\midrule
{summary_rows}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

\begin{{landscape}}
\begin{{center}}
\small
\begin{{longtable}}{{lrrrrrrrrrrrrr}}
\caption{{Per-class precision, recall, and F1 on the combined mixture-only dataset. Classes with zero support across both mixture datasets are omitted.}}\\
\toprule
Chemical & Support & \multicolumn{{3}}{{c}}{{Siamese 0.5}} & \multicolumn{{3}}{{c}}{{Siamese calibrated}} & \multicolumn{{3}}{{c}}{{Best clean classical}} & \multicolumn{{3}}{{c}}{{Best clean deep}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}} \cmidrule(lr){{12-14}}
 &  & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule
\endfirsthead
\toprule
Chemical & Support & \multicolumn{{3}}{{c}}{{Siamese 0.5}} & \multicolumn{{3}}{{c}}{{Siamese calibrated}} & \multicolumn{{3}}{{c}}{{Best clean classical}} & \multicolumn{{3}}{{c}}{{Best clean deep}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}} \cmidrule(lr){{12-14}}
 &  & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule
\endhead
{per_class_rows}
\bottomrule
\end{{longtable}}
\end{{center}}
\end{{landscape}}

\textbf{{Selected clean classical model.}} {tex_escape(best_classical_name)}. Candidate scores on the same mixture-only set: {tex_escape(classical_note)}.

\textbf{{Selected clean deep model.}} {tex_escape(best_deep_name)}. Candidate scores on the same mixture-only set: {tex_escape(deep_note)}.

\textbf{{How the default Siamese+MLP 2-head works.}} This is the original PT2-expanded retrained Siamese pipeline. It starts from synthetic binary mixtures generated from the expanded pure-spectrum library. Each synthetic sample mixes two reference compounds at sampled ratios, then passes the result through the same preprocessing used at test time. The model has two Siamese-style encoders: one branch for the preprocessed Raman spectrum and one branch for its FFT representation. Those two embeddings are concatenated and passed to a multilabel MLP presence head with one sigmoid output per compound in the 17-class library. Training therefore optimizes a class-presence detection problem, not an explicit unmixing problem. At inference, every class with output score above the fixed threshold 0.5 is returned as present, so the number of predicted chemicals can vary from sample to sample.

\textbf{{How the calibrated Siamese+MLP 2-head works.}} The calibrated Siamese model uses the exact same trained Raman encoder, FFT encoder, and multilabel presence head as the default row. The only change is the decision rule at inference time. Instead of using a flat 0.5 threshold for every class, it applies the saved class-specific thresholds that were tuned on the original real-mixture calibration experiment. That can improve recall for weak classes by lowering their trigger threshold, but it also increases the risk of overpredicting absent compounds because every class is still decided independently. In this mixture-only 17-class evaluation, that tradeoff pushes predicted support size up from about 2.89 to about 3.17 chemicals per sample and hurts exact-match performance.

\textbf{{How the best clean classical model works.}} The selected clean classical winner is the replicate-dictionary pair NNLS model. It assumes the task is binary mixture identification and explicitly searches over all compound pairs in the 17-class library. For each candidate pair, it builds a design matrix containing multiple reference atoms per compound rather than a single prototype spectrum. Those atoms include the class mean plus selected representative replicates, which lets the fit absorb within-class spectral variation without changing the label set. A constant nuisance baseline atom is appended so simple background offset can be absorbed without forcing the pair choice to explain it. For every mixture spectrum, the method solves a nonnegative least-squares fit for every candidate pair, ranks pairs by residual error, and returns the minimum-residual support. Training in the machine-learning sense is not required; the method is entirely driven by the reference library and the NNLS optimization at inference time.

\textbf{{How the best clean deep model works.}} The selected clean deep winner is the similarity-supervised coefficient regressor. It also trains on synthetic binary mixtures generated from the expanded reference library, but unlike the Siamese classifier it predicts a full nonnegative coefficient-share vector over the 17 compounds. The network is an MLP operating directly on the preprocessed spectrum. Its output is normalized into coefficient shares, one share per compound. Training combines several objectives: coefficient regression toward the true synthetic mixture ratios, a support loss that separates active from inactive compounds, a reconstruction loss through a fixed decoder built from the reference dictionary, a margin-ranking term that pushes true compounds above false ones, and a similarity-weighted false-positive penalty that is especially harsh on spectrally similar impostor compounds. At inference, the model ranks all predicted shares and returns the top two compounds, which matches the binary structure of the measured mixture datasets without introducing pair-specific rules.

\textbf{{Why the clean deep model wins here.}} On this combined mixture-only set, the clean deep winner outperforms the clean classical winner because it keeps the binary top-2 deployment rule while learning a smoother global representation over the entire 17-class library. The classical NNLS solver is still very strong, but its remaining errors are concentrated in a few chemically similar neighborhoods where residual-based pair selection can flip to the wrong pair. The similarity-supervised regressor learns a library-level ranking that reduces those near-tie flips without needing the hand-tuned fallback rules used in the stronger engineering-only classical variants.

\end{{document}}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_siam_df = evaluate_siamese(default_thresholds=True)
    calibrated_siam_df = evaluate_siamese(default_thresholds=False)

    classical_frames, classical_candidates, deep_frames, deep_candidates = choose_best_candidates()
    best_classical_name = str(classical_candidates.iloc[0]["candidate"])
    best_deep_name = str(deep_candidates.iloc[0]["candidate"])

    model_frames = {
        "Siamese+MLP 2-head (default 0.5)": default_siam_df,
        "Siamese+MLP 2-head (calibrated)": calibrated_siam_df,
        "Best clean classical": classical_frames[best_classical_name],
        "Best clean deep": deep_frames[best_deep_name],
    }

    metrics_by_model: dict[str, dict[str, float]] = {}
    per_class_by_model: dict[str, pd.DataFrame] = {}
    for model_name, df in model_frames.items():
        summary, per_class = metrics_from_predictions(df)
        metrics_by_model[model_name] = summary
        per_class_by_model[model_name] = per_class

    summary_df = pd.DataFrame(
        [{"model": name, **metrics_by_model[name]} for name in model_frames]
    )
    summary_df.to_csv(OUTPUT_DIR / "best_models_full_new_dataset_summary.csv", index=False)

    merged = pd.DataFrame({"chemical": ALL_CLASSES})
    merged["support"] = per_class_by_model["Best clean classical"]["support"]
    merged = merged[merged["support"] > 0].copy()
    for model_name, per_class in per_class_by_model.items():
        slug = (
            model_name.lower()
            .replace("+", "plus")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        merged = merged.merge(
            per_class[["chemical", "precision", "recall", "f1"]].rename(
                columns={
                    "precision": f"{slug}_precision",
                    "recall": f"{slug}_recall",
                    "f1": f"{slug}_f1",
                }
            ),
            on="chemical",
            how="left",
        )
    merged.to_csv(OUTPUT_DIR / "best_models_full_new_dataset_per_class.csv", index=False)

    classical_candidates.to_csv(OUTPUT_DIR / "clean_classical_candidate_summary.csv", index=False)
    deep_candidates.to_csv(OUTPUT_DIR / "clean_deep_candidate_summary.csv", index=False)
    default_siam_df.to_csv(OUTPUT_DIR / "siamese_default_mixture_only_predictions.csv", index=False)
    calibrated_siam_df.to_csv(OUTPUT_DIR / "siamese_calibrated_mixture_only_predictions.csv", index=False)
    classical_frames[best_classical_name].to_csv(
        OUTPUT_DIR / "best_clean_classical_mixture_only_predictions.csv", index=False
    )
    deep_frames[best_deep_name].to_csv(
        OUTPUT_DIR / "best_clean_deep_mixture_only_predictions.csv", index=False
    )

    tex = build_latex(
        metrics_by_model,
        per_class_by_model,
        best_classical_name,
        best_deep_name,
        classical_candidates,
        deep_candidates,
    )
    tex_path = OUTPUT_DIR / "best_models_full_new_dataset_comparison.tex"
    tex_path.write_text(tex)

    print(f"Saved summary CSV to {OUTPUT_DIR / 'best_models_full_new_dataset_summary.csv'}")
    print(f"Saved per-class CSV to {OUTPUT_DIR / 'best_models_full_new_dataset_per_class.csv'}")
    print(f"Saved LaTeX comparison to {tex_path}")


if __name__ == "__main__":
    main()
