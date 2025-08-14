# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Function definitions 
def baseline_AsLS(y,lam = 1e4,p=0.01,niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        b = np.linalg.solve(np.diag(w) + D, w * y)
        w = p * (y > b) + (1 - p) * (y < b)
    return b

def preprocess(arr):
    out = np.zeros_like(arr)
    for i, s in enumerate(arr):
        b = baseline_AsLS(s)
        c = s - b
        norm = np.linalg.norm(c)
        out[i] = c / norm if norm > 0 else c
    return out

def augment(spec, noise_std=0.01,shift_max =2):
    spec_noisy = spec+np.random.normal(0, noise_std, size=spec.shape)
    shift = np.random.randint(-shift_max, shift_max + 1)
    return np.roll(spec_noisy, shift)

# Custom Dataset for Siamese Network training.
# Returns pairs of spectra (same or different class) with a binary label:
# 1.0 if same class, 0.0 if different. Used to train the network to learn spectral similarity.
class RamanPairDataset(Dataset):
    def __init__(self, specs, labels, augment_fn=None):
        self.specs = specs
        self.labels = labels
        self.augment = augment_fn
        self.by_labels = {c: np.where(labels == c)[0] for c in np.unique(labels)}

    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, idx):
        x1 = self.specs[idx]
        y1 = self.labels[idx]
        if np.random.rand() < 0.5:
            j = np.random.choice(self.by_labels[y1])
            label = 1.0
        else:
            neg = [c for c in self.by_labels if c != y1]
            y2 = np.random.choice(neg)
            j = np.random.choice(self.by_labels[y2])
            label = 0.0
        x2 = self.specs[j]
        if self.augment:
            x1 = self.augment(x1)
            x2 = self.augment(x2)
        return (torch.tensor(x1, dtype=torch.float32).unsqueeze(0),
                torch.tensor(x2, dtype=torch.float32).unsqueeze(0),
                torch.tensor(label, dtype=torch.float32)
                )
    
# Siamese Network for learning spectral embeddings.
# The encoder maps 1D spectra to a fixed-length embedding vector.
# The network is trained using contrastive loss to bring similar pairs closer
# and push dissimilar pairs apart in embedding space.
class SiameseNetwork(nn.Module):
    def __init__(self,input_len, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (input_len // 4), embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=1)
    

# Contrastive loss function for Siamese Network
def contrastive_loss(z1,z2,label,margin = 1.0):
    dist = F.pairwise_distance(z1, z2)
    loss_pos = label * dist**2
    loss_neg = (1-label) * F.relu(margin - dist)**2
    return torch.mean(loss_pos + loss_neg)

if __name__ == "__main__":

    df = pd.read_csv('reference_subset_1.csv')
    labels = df['Label'].values
    raw_specs = df.drop(columns=['Label']).values.astype(float)

    spectra = preprocess(raw_specs)

    input_len = spectra.shape[1]
    datasaet = RamanPairDataset(spectra, labels, augment_fn=augment)
    loader = DataLoader(datasaet, batch_size=32, shuffle=True)
    model = SiameseNetwork(input_len=input_len, embed_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        for x1,x2, lbl in loader:
            z1,z2 = model(x1), model(x2)
            loss = contrastive_loss(z1, z2, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'siamese_model.pth')

    # Example inference
    ref_df = pd.read_csv('reference_siamese.csv')
    qry_df = pd.read_csv('query_siamese.csv')
    if 'Label' not in ref_df.columns:
        ref_df.rename(columns={'Species': 'Label'}, inplace=True)
    if 'Label' not in qry_df.columns:
        qry_df.rename(columns={'Species': 'Label'}, inplace=True)

    wav_cols = ref_df.columns[:-1]
    # numeric convert to ensure ordering
    wav_cols = [col for col in wav_cols]
    ref_specs = ref_df[wav_cols].values.astype(float)
    qry_specs = qry_df[wav_cols].values.astype(float)
    ref_labels = ref_df['Label'].values
    qry_labels = qry_df['Label'].values

    # 4. Preprocess spectra
    ref_proc = preprocess(ref_specs)
    qry_proc = preprocess(qry_specs)

    # 5. Instantiate and load model
    input_len = ref_proc.shape[1]
    model = SiameseNetwork(input_len, embed_dim=64)
    model.load_state_dict(torch.load("siamese_raman.pth"))
    model.eval()

    # 6. Compute embeddings
    with torch.no_grad():
        ref_embeds = model(torch.tensor(ref_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()
        qry_embeds = model(torch.tensor(qry_proc, dtype=torch.float32).unsqueeze(1)).cpu().numpy()

    # 7. Nearest neighbor classification
    pred1 = []
    pred2 = []
    for qz in qry_embeds:
        # compute distances
        dists = np.linalg.norm(ref_embeds - qz, axis=1)
        # sort ascending
        idxs = np.argsort(dists)
        pred1.append(ref_labels[idxs[0]])
        if len(idxs) > 1:
            pred2.append((ref_labels[idxs[0]], ref_labels[idxs[1]]))
        else:
            pred2.append((ref_labels[idxs[0]], None))

    # 8. Compute accuracy
    acc1 = accuracy_score(qry_labels, pred1)
    acc2 = np.mean([qry_labels[i] in pair for i, pair in enumerate(pred2)])

    print(f"Top-1 Accuracy: {acc1:.2%}")
    print(f"Top-2 Accuracy: {acc2:.2%}")

    # 9. Confusion matrix
    labels_unique = np.unique(qry_labels)
    cm = confusion_matrix(qry_labels, pred1, labels=labels_unique)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.xticks(range(len(labels_unique)), labels_unique, rotation=90)
    plt.yticks(range(len(labels_unique)), labels_unique)
    plt.colorbar()
    plt.title("Confusion Matrix (Siamese + 1-NN)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()