##-------------------------------
# PASSO 2: MEAN POOLING PCA SPACE
##-------------------------------

import pandas as pd
import numpy as np
import torch
import plotly.express as px
from esm import pretrained
import joblib
from tqdm import tqdm

# === 1. ADATTA: PATHS ===
df_path = './dataset_bilanciato_097.csv'  # dal Passo 1
pca_path = "../../pca/joblibs/Total_ipca_fitted.joblib"  # TUO joblib

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

# === 4. ESM + PCA + MEAN POOLING ===
print('Calcolo embedding...')
data = [("seq", seq) for seq in tqdm(df['sequence'])]
_, _, tokens = batch_converter(data)
tokens = tokens.to(device)

with torch.no_grad():
    out = model(tokens, repr_layers=[28], return_contacts=False)
    reps = out["representations"][28][:, 1:-1]  # [N, L, 1280]

# PCA su GPU (TUO codice)
reps_pca = (reps - pca_mean) @ pca_components.T  # [N, L, 640]
reps_pca = reps_pca.cpu().numpy()  # [8734, 207, 640]
print(f'reps_pca shape: {reps_pca.shape}')

# === 5. MEAN POOLING (globale) ===
mean_pooled = reps_pca.mean(axis=1)  # [8734, 640]
print(f'Mean pooled: {mean_pooled.shape}')

# === 6. PC1 vs PC2 ===
df['PC1'] = mean_pooled[:, 0]
df['PC2'] = mean_pooled[:, 1]

# === 7. PLOT ===
fig1 = px.scatter(df, x='PC1', y='PC2', 
                 color='cosine', 
                 color_continuous_scale='RdYlBu_r',
                 title='Passo 2: MEAN POOLING PCA space (>0.97 seq)',
                 hover_data=['chain_id'])
fig1.show()

fig2 = px.scatter(df, x='PC1', y='PC2', 
                 color='chain_id',
                 title='Passo 2: per CATENA')
fig2.show()

# Salva
fig1.write_html('plot1_cosine.html')
fig2.write_html('plot2_chains.html')
df.to_csv('df_con_pc.csv', index=False)
print('✅ SALVATO: plot1_cosine.html, plot2_chains.html, df_con_pc.csv')
