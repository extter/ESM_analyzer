##-------------------------------
# PASSO 2: MEAN POOLING PCA SPACE
##-------------------------------

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from esm import pretrained
import joblib
from tqdm import tqdm

# === 1. ADATTA: PATHS ===
df_path = './dataset_bilanciato_097.csv'  # dal Passo 1
pca_path = "../../pca/joblibs/Total_ipca_fitted.joblib"  # TUO joblib

# batching per evitare OOM
BATCH_SIZE = 8  # se crasha ancora -> 4


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# === 2. CARICA ===
df = pd.read_csv(df_path)
print(f'Dataset: {len(df)} seq')

pca = joblib.load(pca_path)
pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
print(f'PCA: {pca.n_components_} dim')

# === 3. ESM MODEL (TUO) ===
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

# === 4. ESM + PCA + MEAN POOLING (BATCH SAFE) ===
print('Calcolo embedding...')

all_mean = []
sequences = df['sequence'].tolist()

for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
    batch_seqs = sequences[i:i+BATCH_SIZE]
    data = [("seq", seq) for seq in batch_seqs]

    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[28], return_contacts=False)
        reps = out["representations"][28][:, 1:-1]  # [B, L, 1280]

        # PCA su GPU
        reps_pca = (reps - pca_mean) @ pca_components.T  # [B, L, 640]

        # MEAN POOLING immediato -> evita tensor enormi
        mean_pooled_batch = reps_pca.mean(dim=1)  # [B, 640]

    all_mean.append(mean_pooled_batch.cpu())

    # cleanup GPU
    del tokens, out, reps, reps_pca, mean_pooled_batch
    if device == "cuda":
        torch.cuda.empty_cache()

mean_pooled = torch.cat(all_mean).numpy()
print(f'Mean pooled: {mean_pooled.shape}')

# === 6. PC1 vs PC2 ===
df['PC1'] = mean_pooled[:, 0]
df['PC2'] = mean_pooled[:, 1]

# === 7. PLOT con Matplotlib ===

# Scatter 1: colore continuo "cosine"
plt.figure(figsize=(8,6))
sc = plt.scatter(df['PC1'], df['PC2'], c=df['cosine'], cmap='RdYlBu_r', s=50)
plt.colorbar(sc, label='cosine')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Passo 2: MEAN POOLING PCA space (>0.97 seq)')
plt.tight_layout()
plt.savefig('plot1_cosine_matplotlib.png', dpi=300)
plt.close()

# Scatter 2: colore discreto "chain_id"
plt.figure(figsize=(8,6))
# assegna un colore diverso a ogni catena
chain_ids = df['chain_id'].unique()
colors = plt.cm.tab20(np.linspace(0,1,len(chain_ids)))  # mappa di colori discreti
color_dict = dict(zip(chain_ids, colors))

plt.scatter(df['PC1'], df['PC2'], c=df['chain_id'].map(color_dict), s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Passo 2: per CATENA')
# legenda
for cid, color in color_dict.items():
    plt.scatter([], [], c=[color], label=cid)
plt.legend(title='chain_id', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plot2_chains_matplotlib.png', dpi=300)
plt.close()

# Salva dataframe
df.to_csv('df_con_pc.csv', index=False)
print('✅ SALVATO: plot1_cosine_matplotlib.png, plot2_chains_matplotlib.png, df_con_pc.csv')

