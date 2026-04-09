# Imports 
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function Defs
def baseline_AsLS(y, lam=1e4, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        b = np.linalg.solve(np.diag(w)+D, w*y)
        w = p*(y > b) + (1-p)*(y < b)
    return b
    
def preprocess(arr):
    out = np.zeros_like(arr)
    for i,s in enumerate(arr):
        b = baseline_AsLS(s)
        c = s - b
        out[i] = c/np.linalg.norm(c) if np.linalg.norm(c)>0 else c
    return out

def compute_centroids(R_pca, labels):
    classes = np.unique(labels)
    return {chem: R_pca[labels == chem].mean(axis=0) for chem in classes}

def rank_by_centroid(Q_pca, centroids):
    classes = list(centroids.keys())
    rankings = []
    for qv in Q_pca:
        dists = {chem: np.linalg.norm(qv - centroids[chem]) for chem in classes}
        ranking = sorted(dists, key=dists.get)
        rankings.append(ranking)
    return rankings

def identify_with_pca(query_df, ref_df, crop_max=2500, lam=1e4, p=0.01, niter=10, n_pca=10):
    wavs = pd.to_numeric(query_df.columns[:-1])
    keep_idxs = np.where(wavs < crop_max)[0]
    Q_raw = query_df.iloc[:, keep_idxs].values
    R_raw = ref_df.iloc[:, keep_idxs].values
    labels = ref_df['Label'].values

    Q = preprocess(Q_raw)
    R = preprocess(R_raw)

    pca = PCA(n_components=n_pca)
    R_pca = pca.fit_transform(R)
    Q_pca = pca.transform(Q)

    centroids = compute_centroids(R_pca, labels)
    rankings = rank_by_centroid(Q_pca, centroids)

    return rankings, Q_pca, R_pca, centroids

if __name__ == "__main__":
    query_df = pd.read_csv('Jesse_Dataset/query.csv') 
    ref_df = pd.read_csv('Jesse_Dataset/reference_subset_1.csv') 

    rankings, Q_pca, R_pca, centroids = identify_with_pca(query_df, ref_df)

    true  = query_df['Label'].values
    pred1 = [r[0] for r in rankings]
    pred2 = [r[1] for r in rankings]

    print("Top-1 PCA Acc:", accuracy_score(true, pred1))
    print("Top-2 PCA Acc:", np.mean([t in (p1,p2) for t,p1,p2 in zip(true,pred1,pred2)]))

    
    
