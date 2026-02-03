import torch
import torch.nn as nn
from esm import pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import joblib
import pandas as pd
from Bio import SeqIO
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
import random

import time
import torch
from tqdm import tqdm
import psutil
import sys
import warnings
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
import joblib
import esm.pretrained as pretrained
warnings.filterwarnings('ignore')

def fasta_to_tonb_csv(fasta_input, csv_output):
    # Lista per raccogliere i dati
    data = []
    
    # Leggiamo il file FASTA
    for record in SeqIO.parse(fasta_input, "fasta"):
        data.append({
            "id": record.id,
            "sequence": str(record.seq).upper() # Il tuo codice lavora su questa colonna
        })
    
    # Creiamo il DataFrame
    df = pd.DataFrame(data)
    
    # Salvataggio in CSV (senza indice per pulizia)
    df.to_csv(csv_output, index=False)
    print(f"Conversione completata: {len(df)} sequenze salvate in {csv_output}")

# Utilizzo
random_path = "/kaggle/input/random-dataset/02_random_proteins.fasta"
save_path = "/kaggle/working/Random_dataset.csv"
fasta_to_tonb_csv(random_path, save_path)

input_dir = '/kaggle/input/uniref50-sub'
fasta_files = [f for f in os.listdir(input_dir) if f.endswith(('.fasta', '.fa', '.fasta.gz'))]
print("File FASTA trovati:", fasta_files)

fasta_file = os.path.join(input_dir, fasta_files[0])  # Prendi il primo
print(f"Usando: {fasta_file}")

print("Estrazione subsample sequenze 150-700 AA...")
random.seed(42)
target_n = 100000
sequences = []  # Lista (id, sequenza)
short_records = []

with open(fasta_file, 'r') as handle:
    parser = SeqIO.parse(handle, 'fasta')
    for record in parser:
        if len(record.seq) > 150 and len(record.seq) < 700: # Modificato per 150-700 AA
            short_records.append(record)
        if len(short_records) >= target_n * 2:  # Buffer
            break

print(f"Trovate {len(short_records)} proteine nel range 150-700 AA")

# Random subsample dalle corte
subsample_records = random.sample(short_records, min(target_n, len(short_records)))
sequences = [(record.id, str(record.seq)) for record in subsample_records]

print(f"✓ {len(sequences)} sequenze salvate")
print(f"Media lunghezza: {np.mean([len(s[1]) for s in sequences]):.0f} AA")
print(f"Max: {max(len(s[1]) for s in sequences)} AA")
print(f"Tot AA: {sum(len(s[1]) for s in sequences)/1e6:.1f}M")
print("Esempio:", sequences[0])

# =========================
# CONFIG
# =========================
target_n = 100000

csv_1_path = "/kaggle/input/mutations-of-tonb/TonB_mutations_dataset.csv"
csv_2_path = "/kaggle/working/Random_dataset.csv"

output_csv = "/kaggle/working/dataset_proteine_finale.csv"

random.seed(42)

# =========================
# 1. PRENDI 100K DA sequences
# sequences = [(id, sequence), ...]
# =========================
assert len(sequences) > 0, "Lista sequences vuota"

selected = random.sample(
    sequences,
    min(target_n, len(sequences))
)

# tieni solo la sequenza (senza ID)
seq_only = [seq for _, seq in selected]

df_uniref = pd.DataFrame({"sequence": seq_only})

print(f"UniRef prese: {len(df_uniref)}")

# tieni solo la sequenza
seq_only = [seq for _, seq in selected]

df_uniref = pd.DataFrame(seq_only, columns=["sequence"])

print(f"UniRef: {len(df_uniref)}")

# =========================
# 2. CARICA GLI ALTRI CSV
# (una sola colonna: sequenza)
# =========================
df1 = pd.read_csv(csv_1_path, header=None, names=["sequence"])
df2 = pd.read_csv(csv_2_path, header=None, names=["sequence"])

# =========================
# 3. UNIONE
# =========================
df_final = pd.concat(
    [df_uniref, df1, df2],
    ignore_index=True
)

print("Totale prima dedup:", len(df_final))

# =========================
# 4. DEDUP + SHUFFLE
# =========================
df_final = df_final.drop_duplicates()
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)


print("Totale finale:", len(df_final))

print(df_final.iloc[0, 0])

# =========================
# 5. SALVA CSV
# =========================
df_final.to_csv(output_csv, index=False, header=False)

print(f"✓ Dataset finale salvato: {output_csv}")

csv_path = "/kaggle/working/dataset_proteine_finale.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca_components = 640
pca_batch_size = 64
max_seqs_for_pca = 297706

# ------------------------
# CARICA MODELLO ESM2 + DataParallel (invariato)
# ------------------------
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print("Uso", torch.cuda.device_count(), "GPU con DataParallel")
    model = nn.DataParallel(model, device_ids=[0, 1])
else:
    print("Uso una sola GPU")

def get_num_layers(model):
    if isinstance(model, nn.DataParallel):
        return model.module.num_layers
    else:
        return model.num_layers

# ------------------------
# FUNZIONI EMBEDDING (invariato)
# ------------------------
def forward_with_single_gpu_if_small_batch(model, tokens, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        single_model = model.module.to("cuda:0")
        return single_model(tokens.to("cuda:0"), **kwargs)
    else:
        return model(tokens, **kwargs)

@torch.no_grad()
def get_residue_embeddings_batch(sequences):
    data = [("seq", s) for s in sequences]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    num_layers = 28
    #num_layers = get_num_layers(model)
    out = forward_with_single_gpu_if_small_batch(
        model, batch_tokens, repr_layers=[num_layers], return_contacts=False
    )
    
    token_reps = out["representations"][num_layers]
    
    emb_list = []
    for i, seq in enumerate(batch_strs):
        L = len(seq)
        emb = token_reps[i, 1:1+L].detach().cpu().numpy()
        emb_list.append(emb)
    return emb_list

# ------------------------
# CARICA DATASET (invariato)
# ------------------------
df = pd.read_csv(csv_path, header=None, names=["sequence"])
sequences = df["sequence"].tolist()
print(f"Caricate {len(sequences)} sequenze dal CSV")

print(sequences[0])

if len(sequences) > max_seqs_for_pca:
    seqs_for_pca = sequences[:max_seqs_for_pca]
else:
    seqs_for_pca = sequences

# ------------------------
# 1) FIT INCREMENTAL PCA (invariato)
# ------------------------
print("Inizio IncrementalPCA (solo fit)...")
ipca = IncrementalPCA(n_components=pca_components, batch_size=None)

for i in tqdm(range(0, len(seqs_for_pca), pca_batch_size), desc="Fit IncrementalPCA"):
    batch_seqs = seqs_for_pca[i:i+pca_batch_size]
    emb_list = get_residue_embeddings_batch(batch_seqs)
    X_batch = np.concatenate(emb_list, axis=0)
    ipca.partial_fit(X_batch)

print("IncrementalPCA fit completato!")
print("Explained variance ratio (sum):", ipca.explained_variance_ratio_.sum())

# ------------------------
# 2) SALVA PCA (NUOVO - FINE CODICE!)
# ------------------------
joblib.dump(ipca, "Total_ipca_fitted.joblib")
print("✅ PCA salvata in Total_ipca_fitted.joblib (~2MB)")

# Opzionale: salva anche info dataset
joblib.dump({
    'pca_components': pca_components,
    'n_sequences_used': len(seqs_for_pca),
    'model_name': 'esm2_t33_650M_UR50D'
}, "Total_pca_metadata.joblib")
print("✅ Metadata salvati!")