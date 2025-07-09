# Imports 
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Preprocessing functions
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
        bkg = baseline_AsLS(spec,lam=lam, p=p, niter=niter)
        corr = spec - bkg
        nrm = np.linalg.norm(corr)
        normed = corr / nrm if nrm else corr
        normed = np.abs(normed)  #Make all values positive, element-wise
        out[i] = normed
    return out

def smooth(spec, K_smooth=3):
    kernel = np.ones(K_smooth) / K_smooth
    return np.convolve(spec, kernel, mode='same') 

def extract_wavenumber_cols(df):
    return [col for col in df.columns if col.replace('.', '', 1).isdigit()]


# main function
def identify_multilabel_knn(query_df, ref_df,
    crop_max=1500, lam=1e4, p=0.01, niter=10,
    K_smooth=3, N_peak=12, w_max=15,
    height=0.01, prominence=0.01,
    n_neighbors=3):

    wav_cols = extract_wavenumber_cols(query_df)
    wavs = np.array(wav_cols, dtype=float)
    keep_cols = [col for col, w in zip(wav_cols, wavs) if w < crop_max]

    Q_raw = query_df[keep_cols].values.astype(float)
    R_raw = ref_df[keep_cols].values.astype(float)

    # Build multilabel targets
    ref_labels = list(zip(ref_df['Label 1'], ref_df['Label 2']))
    query_labels = list(zip(query_df['Label 1'], query_df['Label 2']))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(ref_labels)

    # Preprocess
    Q = preprocess(Q_raw)
    R = preprocess(R_raw, lam, p, niter)

    # CP detection
    CPs = {}
    for i, class_name in enumerate(mlb.classes_):
        specs = R[y[:, i] == 1]
        counts = np.zeros(Q.shape[1], int)
        for s in specs:
            pks, _ = find_peaks(smooth(s, K_smooth), height=height, prominence=prominence)
            counts[pks] += 1
        CPs[class_name] = sorted(np.argsort(counts)[-N_peak:])

    global_cp = sorted({i for idxs in CPs.values() for i in idxs})

    # Reference feature matrix
    X = []
    for s in R:
        vec = [np.max(s[max(0, i - w_max//2):i + w_max//2 + 1]) for i in global_cp]
        X.append(minmax_scale(vec))
    X = np.array(X)

    # Train multilabel kNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X, y)

    # Query feature matrix
    Q_feat = np.vstack([
        minmax_scale([np.max(s[max(0, i - w_max//2):i + w_max//2 + 1]) for i in global_cp])
        for s in Q
    ])
    y_pred = knn.predict(Q_feat)
    y_true = mlb.transform(query_labels)

    return y_true, y_pred, mlb

if __name__ == "__main__":
    ref_df = pd.read_csv("mixtures_dataset.csv")
    query_df = pd.read_csv("query_only_mixed.csv")
    y_true, y_pred, mlb = identify_multilabel_knn(query_df, ref_df)
    report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    print(report)

    

