# Imports 
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function definitions
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# --- Helper functions ---
def baseline_als(y, lam=1e4, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D @ D.T
    w = np.ones(L)
    for _ in range(niter):
        b = np.linalg.solve(np.diag(w) + D, w * y)
        w = p * (y > b) + (1 - p) * (y < b)
    return b

def preprocess(arr, lam=1e4, p=0.01, niter=10):
    out = np.zeros_like(arr)
    for i, s in enumerate(arr):
        b = baseline_als(s, lam=lam, p=p, niter=niter)
        c = s - b
        norm = np.linalg.norm(c)
        out[i] = c / norm if norm > 0 else c
    return out

def smooth(spec, K_smooth=3):
    kernel = np.ones(K_smooth) / K_smooth
    return np.convolve(spec, kernel, mode='same')

# --- Main pipeline ---
def identify_with_knn(query_df, ref_df_train, ref_df_test=None,
    crop_max=1700, lam=1e4, p=0.01, niter=10,
    K_smooth=3, N_peak=12, w_max=15,
    height=0.01, prominence=0.01,
    n_neighbors=3):

    wav_cols = query_df.columns[:-1]
    wavs = pd.to_numeric(wav_cols)
    keep_cols = wav_cols[wavs < crop_max]

    # build matrices
    R_raw = ref_df_train[keep_cols].values.astype(float)
    labels = ref_df_train['Label'].values
    classes = np.unique(labels)

    R = preprocess(R_raw, lam, p, niter)

    # Characteristic peaks
    CPs = {}
    for chem in classes:
        specs = R[labels == chem]
        counts = np.zeros(R.shape[1], int)
        for s in specs:
            pks, _ = find_peaks(smooth(s, K_smooth),
                                height=height, prominence=prominence)
            counts[pks] += 1
        cp_idxs = np.argsort(counts)[-N_peak:]
        CPs[chem] = sorted(cp_idxs)

    global_cp = sorted({i for idxs in CPs.values() for i in idxs})

    # Train features
    X = []
    y = []
    for chem in classes:
        for s in R[labels == chem]:
            vec = [np.max(s[max(0, i - w_max//2):i + w_max//2 + 1]) for i in global_cp]
            X.append(minmax_scale(vec))
            y.append(chem)
    X = np.array(X)
    y = np.array(y)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X, y)

    # Optional test set
    test_results = None
    if ref_df_test is not None:
        R_test = preprocess(ref_df_test[keep_cols].values.astype(float), lam, p, niter)
        y_true_test = ref_df_test['Label'].values
        X_test = np.vstack([
            minmax_scale([np.max(s[max(0, i - w_max//2):i + w_max//2 + 1]) for i in global_cp])
            for s in R_test
        ])
        y_pred_test = knn.predict(X_test)
        y_pred2_test = knn.kneighbors(X_test, n_neighbors=2, return_distance=False)
        y_pred2_test = np.array([[y[idx] for idx in row] for row in y_pred2_test])
        test_results = (y_true_test, y_pred_test, y_pred2_test)

    # Validation set (query)
    Q_raw = query_df[keep_cols].values.astype(float)
    Q = preprocess(Q_raw, lam, p, niter)
    Q_feat = np.vstack([
        minmax_scale([np.max(s[max(0, i - w_max//2):i + w_max//2 + 1]) for i in global_cp])
        for s in Q
    ])
    pred1 = knn.predict(Q_feat)
    neigh_idx = knn.kneighbors(Q_feat, n_neighbors=2, return_distance=False)
    pred2 = np.array([[y[idx] for idx in row] for row in neigh_idx])

    return pred1, pred2, knn, CPs, test_results

if __name__ == "__main__":
    ref_df = pd.read_csv('Jesse_Dataset/reference.csv')
    query_df = pd.read_csv('Jesse_Dataset/query.csv')

    # Split reference data 50/50
    ref_train, ref_test = train_test_split(ref_df, test_size=0.5, stratify=ref_df['Label'], random_state=42)

    # Run model
    pred_val, pred_val_top2, knn_model, CPs, test_results = identify_with_knn(query_df, ref_train, ref_test)

    # Validation accuracy
    true_val = query_df['Label'].values
    acc1_val = accuracy_score(true_val, pred_val)
    acc2_val = np.mean([t in row for t, row in zip(true_val, pred_val_top2)])
    print(f"[VALIDATION] Top-1 k-NN Acc: {acc1_val:.2%}")
    print(f"[VALIDATION] Top-2 k-NN Acc: {acc2_val:.2%}")

    # Test set accuracy
    if test_results:
        true_test, pred_test, pred2_test = test_results
        acc1_test = accuracy_score(true_test, pred_test)
        acc2_test = np.mean([t in row for t, row in zip(true_test, pred2_test)])
        print(f"[TEST] Top-1 k-NN Acc: {acc1_test:.2%}")
        print(f"[TEST] Top-2 k-NN Acc: {acc2_test:.2%}")

    
    