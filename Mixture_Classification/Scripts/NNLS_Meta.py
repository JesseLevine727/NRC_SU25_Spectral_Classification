import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# ------------------------ Data Loading ------------------------

def load_data(ref_path, query_path, crop_max=1500):
    ref_df = pd.read_csv(ref_path)
    query_df = pd.read_csv(query_path)

    all_cols = ref_df.columns.drop("Label")
    wavenumbers = pd.to_numeric(all_cols, errors="coerce")
    valid_cols = all_cols[wavenumbers < crop_max]

    ref_data = ref_df[valid_cols].values
    query_data = query_df[valid_cols].values
    ref_labels = ref_df["Label"].values

    return ref_df, query_df, ref_data, query_data, ref_labels, valid_cols


# ------------------------ Preprocessing ------------------------

def baseline_AsLS(y, lam=1e4, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        b = np.linalg.solve(np.diag(w) + D, w * y)
        w = p * (y > b) + (1 - p) * (y < b)
    return b

def preprocess(arr, lam=1e4, p=0.01, niter=10):
    out = np.zeros_like(arr)
    for i, spec in enumerate(arr):
        bkg = baseline_AsLS(spec, lam=lam, p=p, niter=niter)
        corr = spec - bkg
        nrm = np.linalg.norm(corr)
        normed = corr / nrm if nrm else corr
        out[i] = np.abs(normed)
    return out


# ------------------------ Characteristic Peak Selection ------------------------

def extract_cps(spectra, num_peaks=15, height=0.01, prominence=0.01):
    counts = np.zeros(spectra.shape[1])
    for spec in spectra:
        peaks, _ = find_peaks(spec, height=height, prominence=prominence)
        counts[peaks] += 1
    top_peaks = np.argsort(counts)[-num_peaks:]
    return sorted(top_peaks)


# ------------------------ NNLS Fitting ------------------------

def l1_nnls(A, b, alpha=0.001):
    lasso = Lasso(alpha=alpha, positive=True, max_iter=10000)
    lasso.fit(A, b)
    return lasso.coef_


# ------------------------ Meta-Model Training and Prediction ------------------------

def train_meta_models(weights, query_df, classes):
    y_multi = np.array([[label in (l1, l2) for label in classes]
                        for l1, l2 in zip(query_df["Label 1"], query_df["Label 2"])])

    models = {}
    for i, chem in enumerate(classes):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(weights, y_multi[:, i])
        models[chem] = clf
    return models


def predict_labels(models, weights, classes):
    proba_preds = np.array([
        models[c].predict([w])[0] for w in weights for c in classes
    ])
    proba_preds = proba_preds.reshape(len(weights), len(classes))
    binary_preds = (proba_preds > 0.5).astype(int)
    predicted_labels = [[c for j, c in enumerate(classes) if row[j]] for row in binary_preds]
    return binary_preds, predicted_labels


# ------------------------ Evaluation ------------------------

def evaluate_performance(true1, true2, predicted_labels, binary_preds, classes):
    metrics = {}
    for i, chem in enumerate(classes):
        y_true = [(chem in [l1, l2]) for l1, l2 in zip(true1, true2)]
        y_pred = binary_preds[:, i]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        metrics[chem] = {"Accuracy": acc, "Precision": prec, "Recall": rec}

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    filtered_metrics_df = metrics_df[~((metrics_df["Precision"] == 0) & (metrics_df["Recall"] == 0))]

    correct = [l1 in p and l2 in p for l1, l2, p in zip(true1, true2, predicted_labels)]
    total_accuracy = sum(correct) / len(correct)

    return filtered_metrics_df, total_accuracy


# ------------------------ Main Pipeline ------------------------

def run_pipeline(ref_path, query_path):
    ref_df, query_df, ref_data, query_data, ref_labels, valid_cols = load_data(ref_path, query_path)

    X_ref = preprocess(ref_data)
    X_query = preprocess(query_data)

    classes = np.unique(ref_labels)
    ref_spectra_avg = {c: X_ref[ref_labels == c].mean(axis=0) for c in classes}
    ref_matrix = np.array([ref_spectra_avg[c] for c in classes])

    cps = extract_cps(X_ref, num_peaks=15)
    ref_matrix_cps = ref_matrix[:, cps]
    X_query_cps = X_query[:, cps]

    weights_l1 = np.array([l1_nnls(ref_matrix_cps.T, q) for q in X_query_cps])
    meta_models = train_meta_models(weights_l1, query_df, classes)

    binary_preds, predicted_labels = predict_labels(meta_models, weights_l1, classes)

    true1 = query_df["Label 1"].values
    true2 = query_df["Label 2"].values

    metrics_df, total_accuracy = evaluate_performance(true1, true2, predicted_labels, binary_preds, classes)

    print(metrics_df)
    print(total_accuracy)


# ------------------------ Execute ------------------------

if __name__ == "__main__":
    run_pipeline("reference_v2.csv", "query_only_mixed.csv")
