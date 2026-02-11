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
from mpl_toolkits.mplot3d import Axes3D  # necessario per 3D
from Bio.Align import substitution_matrices
import random

blosum62 = substitution_matrices.load("BLOSUM62")

# sequenza TonB
tonb_seq = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

def conservative_mutations(seq, n_mut=5):
    variants = []
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(n_mut):
        seq_mut = list(seq)
        pos = random.randint(0, len(seq)-1)
        original = seq_mut[pos]
        cons_aas = [aa for aa in aa_list if blosum62[original, aa] >= 1 and aa != original]
        if cons_aas:
            seq_mut[pos] = random.choice(cons_aas)
        variants.append("".join(seq_mut))
    return variants

tonb_variants = conservative_mutations(tonb_seq, n_mut=5)
print("Varianti conservative TonB:", tonb_variants)


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

# TonB originale
data_tonb = [("TonB", tonb_seq)]
_, _, tokens_tonb = batch_converter(data_tonb)
tokens_tonb = tokens_tonb.to(device)

with torch.no_grad():
    out_tonb = model(tokens_tonb, repr_layers=[28], return_contacts=False)
    reps_tonb = out_tonb["representations"][28][:, 1:-1]
    reps_pca_tonb = (reps_tonb - pca_mean) @ pca_components.T
    mean_pooled_tonb = reps_pca_tonb.mean(dim=1).cpu().numpy()  # [1, 640]

# TonB mutazioni conservative
mean_pooled_variants = []
for i, var_seq in enumerate(tonb_variants):
    data_var = [(f'TonB_mut{i+1}', var_seq)]
    _, _, tokens_var = batch_converter(data_var)
    tokens_var = tokens_var.to(device)

    with torch.no_grad():
        out_var = model(tokens_var, repr_layers=[28], return_contacts=False)
        reps_var = out_var["representations"][28][:, 1:-1]
        reps_pca_var = (reps_var - pca_mean) @ pca_components.T
        mean_pooled_var = reps_pca_var.mean(dim=1).cpu().numpy()

    mean_pooled_variants.append(mean_pooled_var)

mean_pooled_variants = np.concatenate(mean_pooled_variants, axis=0)



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


# Salva dataframe
df.to_csv('df_con_pc.csv', index=False)
print('✅ SALVATO: plot1_cosine_matplotlib.png, plot2_chains_matplotlib.png, df_con_pc.csv')



# === 8. PC1 vs PC2 vs PC3 (3D) ===
df['PC3'] = mean_pooled[:, 2]


# assegna colori
chain_ids = df['chain_id'].unique()
colors = plt.cm.tab20(np.linspace(0,1,len(chain_ids)))
color_dict = dict(zip(chain_ids, colors))

# Scatter 3D cosine
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['cosine'], cmap='RdYlBu_r', s=50)
ax.scatter(mean_pooled_tonb[:,0], mean_pooled_tonb[:,1], mean_pooled_tonb[:,2], c='black', s=60, marker='X', label='TonB')
ax.scatter(mean_pooled_variants[:,0], mean_pooled_variants[:,1], mean_pooled_variants[:,2], c='orange', s=50, marker='^', label='TonB mut cons')
fig.colorbar(p, ax=ax, label='cosine')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('Passo 2: MEAN POOLING PCA space 3D (>0.97 seq)')
plt.legend()
plt.tight_layout()
plt.savefig('plot3_cosine_3D_matplotlib.png', dpi=300)
plt.close()

# Scatter 3D chain_id
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
for cid in chain_ids:
    mask = df['chain_id'] == cid
    ax.scatter(df.loc[mask,'PC1'], df.loc[mask,'PC2'], df.loc[mask,'PC3'], color=color_dict[cid], s=50, label=cid)
ax.scatter(mean_pooled_tonb[:,0], mean_pooled_tonb[:,1], mean_pooled_tonb[:,2], c='black', s=60, marker='X', label='TonB')
ax.scatter(mean_pooled_variants[:,0], mean_pooled_variants[:,1], mean_pooled_variants[:,2], c='orange', s=50, marker='^', label='TonB mut cons')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('Passo 2: per CATENA (3D)')
ax.legend(title='chain_id', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('plot4_chains_3D_matplotlib.png', dpi=300)
plt.close()
