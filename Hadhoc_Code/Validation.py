import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale

def identify_spectrum_pipeline(query_df, ref_df, crop_max=1700, lam=1e4, p=0.01, niter=10, K_smooth=3, N_peak=15, w_max=36, height=0.02, prominence=0.02):
    """
    Returns a list of lists: each inner list is the species ranked
    by descending CaPSim score (Top-1 at index 0, Top-2 at index 1).
    """
    # --- crop ---
    wav_cols = query_df.columns[:-1]
    wavs_num = pd.to_numeric(wav_cols)
    keep     = wav_cols[wavs_num < crop_max]
    wavs     = wavs_num[wavs_num < crop_max].values

    q = query_df[keep].values.astype(float)
    r = ref_df  [keep].values.astype(float)
    r_lbl = ref_df['Species'].values

    # --- preprocess ---
    def baseline_als(y):
        L = len(y)
        D = np.diff(np.eye(L), 2); D = lam * D.dot(D.T)
        w = np.ones(L)
        for _ in range(niter):
            b = np.linalg.solve(np.diag(w) + D, w * y)
            w = p * (y > b) + (1 - p) * (y < b)
        return b

    def preprocess(arr):
        out = np.zeros_like(arr)
        for i, spec in enumerate(arr):
            bkg  = baseline_als(spec)
            corr = spec - bkg
            nrm  = np.linalg.norm(corr)
            out[i] = corr/nrm if nrm else corr
        return out

    Q = preprocess(q)
    R = preprocess(r)

    # --- smooth + CP library ---
    def smooth(spec):
        return np.convolve(spec, np.ones(K_smooth)/K_smooth, mode='same')

    CPs, Ref_comp = {}, {}
    for chem in np.unique(r_lbl):
        refs = R[r_lbl == chem]
        counts = np.zeros(len(wavs), int)
        for spec in refs:
            sm = smooth(spec)
            pks, _ = find_peaks(sm, height=height, prominence=prominence)
            counts[pks] += 1
        cp_idxs = sorted(np.argsort(counts)[-N_peak:])
        CPs[chem] = cp_idxs

        comp = []
        for spec in refs:
            vec = []
            hw  = w_max//2
            for i in cp_idxs:
                seg = spec[max(0,i-hw):i+hw+1]
                vec.append(np.max(seg))
            comp.append(minmax_scale(vec))
        Ref_comp[chem] = np.array(comp)

    # --- identification: return full ranking per query ---
    def CaPSim(qv, Rvs):
        return (Rvs @ qv).mean()

    all_rankings = []
    for spec in Q:
        scores = {}
        for chem, cp_idxs in CPs.items():
            vec = []
            hw  = w_max//2
            for i in cp_idxs:
                seg = spec[max(0,i-hw):i+hw+1]
                vec.append(np.max(seg))
            qv = minmax_scale(vec)
            scores[chem] = CaPSim(qv, Ref_comp[chem])
        # sorted chems high→low
        ranking = sorted(scores, key=scores.get, reverse=True)
        all_rankings.append(ranking)

    return all_rankings


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# 1) Prepare storage
split_top1 = []
split_top2 = []
split_cms  = []

# Get the consistent label order from split1
labels = sorted(pd.read_csv('Test_Data/query_dataset_split1.csv')['Species'].unique())

# 2) Loop over splits 1–5
for i in range(1, 6):
    # Load this split
    qdf = pd.read_csv(f'Test_Data/query_dataset_split{i}.csv')
    rdf = pd.read_csv(f'Test_Data/reference_dataset_split{i}.csv')
    
    # Get full rankings from your pipeline
    rankings = identify_spectrum_pipeline(qdf, rdf)
    
    true = qdf['Species'].values
    pred1 = [r[0] for r in rankings]
    pred2 = [r[:2] for r in rankings]
    
    # Compute Top-1 & Top-2
    t1 = accuracy_score(true, pred1)
    t2 = np.mean([true[j] in pred2[j] for j in range(len(true))])
    split_top1.append(t1)
    split_top2.append(t2)
    
    # Confusion matrix for Top-1
    cm = confusion_matrix(true, pred1, labels=labels)
    split_cms.append(cm)

# 3) Report per-split & aggregate
results_df = pd.DataFrame({
    'Split': range(1,6),
    'Top-1': split_top1,
    'Top-2': split_top2
})
mean1, std1 = results_df['Top-1'].mean(), results_df['Top-1'].std()
mean2, std2 = results_df['Top-2'].mean(), results_df['Top-2'].std()

print("Per-split accuracies:")
print(results_df)
print(f"\nMean Top-1: {mean1:.2%} ± {std1:.2%}")
print(f"Mean Top-2: {mean2:.2%} ± {std2:.2%}")

# 4) Plot confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for idx, ax in enumerate(axes):
    sns.heatmap(split_cms[idx], annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f"Split {idx+1}")
    ax.set_xlabel('Predicted')
    if idx == 0:
        ax.set_ylabel('True')
plt.tight_layout()
plt.show()
