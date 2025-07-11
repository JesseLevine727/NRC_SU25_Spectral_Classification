# Imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

# Loading data
reference_df = pd.read_csv('Jesse_Dataset/reference.csv')
query_df = pd.read_csv('Jesse_Dataset/query.csv')

# Functions Definitions
def baseline_als(y, lam=1e4, p=0.01, niter=10):
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
        bkg = baseline_als(spec, lam=lam, p=p, niter=niter)
        corr = spec - bkg
        nrm = np.linalg.norm(corr)
        out[i] = corr / nrm if nrm else corr
    return out

def smooth(spec, K_smooth=3):
    window = np.ones(K_smooth) / K_smooth
    return np.convolve(spec, window, mode='same')

def build_CP_library(R, Q_len, K_smooth, height, prominence, N_peak, w_max):
    CPs = {}
    Ref_comp = {}
    for chem in np.unique(R[:, -1]):  # assume last column of R is label
        refs = R[R[:, -1] == chem, :-1]
        counts = np.zeros(Q_len, int)
        for spec in refs:
            pks, _ = find_peaks(smooth(spec, K_smooth), height=height, prominence=prominence)
            counts[pks] += 1
        cp_idxs = sorted(np.argsort(counts)[-N_peak:])
        CPs[chem] = cp_idxs

        comp = []
        hw = w_max // 2
        for spec in refs:
            vec = [np.max(spec[max(0, i - hw): i + hw + 1]) for i in cp_idxs]
            comp.append(minmax_scale(vec))
        Ref_comp[chem] = np.vstack(comp)
    return CPs, Ref_comp

def CaPSim(query_vector, reference_matrix):
    """
    Compute the mean dot‐product similarity between one query_vector and each row of reference_matrix.
    """
    return np.dot(reference_matrix, query_vector).mean()

def identify_spectrum_pipeline(query_df, ref_df,
    crop_max=1700, lam=1e4, p=0.01, niter=10,
    K_smooth=3, N_peak=12, w_max=15,
    height=0.01, prominence=None):

    # select wavenumber columns
    wav_cols = query_df.columns[:-1]
    wavs = pd.to_numeric(wav_cols)
    keep = wav_cols[wavs < crop_max]

    q = query_df[keep].values.astype(float)
    r = ref_df[keep].values.astype(float)
    r_lbl = ref_df['Label'].values

    # preprocess
    Q = preprocess(q, lam, p, niter)
    R = preprocess(r, lam, p, niter)

    # build CP library (we pass labels along as last column)
    R_with_lbl = np.hstack([R, r_lbl.reshape(-1, 1)])
    CPs, Ref_comp = build_CP_library(R_with_lbl, Q.shape[1],
                                     K_smooth, height, prominence,
                                     N_peak, w_max)

    # identification
    all_rankings = []
    for spec in Q:
        scores = {}
        hw = w_max // 2
        for chem, Rvs in Ref_comp.items():
            vec = [np.max(spec[max(0, i - hw): i + hw + 1]) for i in CPs[chem]]
            qv = minmax_scale(vec)
            scores[chem] = CaPSim(qv, Rvs)
        ranking = sorted(scores, key=scores.get, reverse=True)
        all_rankings.append(ranking)

    return all_rankings, Q, CPs, Ref_comp

if __name__ == "__main__":
    rankings, Q, CPs, Ref_comp = identify_spectrum_pipeline(query_df, reference_df)
    true_labels = query_df['Label'].values
    preds_topk = np.array([[r[k] if k < len(r) else None for k in range(5)]
                        for r in rankings])

    # Compute Top-1 through Top-5 accuracies
    for k in range(1, 6):
        acc = np.mean([true in pred_row[:k]
                    for true, pred_row in zip(true_labels, preds_topk)])
        print(f"Top-{k} Accuracy: {acc:.2%}")