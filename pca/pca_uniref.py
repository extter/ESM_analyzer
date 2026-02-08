import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
import random
import torch
from tqdm import tqdm
import sys
import warnings
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
import joblib
import esm.pretrained as pretrained
warnings.filterwarnings('ignore')

def print_overwrite(text):
    sys.stdout.write('\r\033[K' + text)
    sys.stdout.flush()

input_dir = '/kaggle/input/uniref'
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

# ------------------------
# CONFIG (invariato)
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca_components = 640
pca_batch_size = 64
max_seqs_for_pca = 100000

seqs_for_pca = [seq for _, seq in sequences[:max_seqs_for_pca]]  # Da sequences corrente!
print(f"Usando {len(seqs_for_pca)} sequenze correnti")

print(seqs_for_pca[0])

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

    num_layers = get_num_layers(model)
    out = forward_with_single_gpu_if_small_batch(
        model, batch_tokens, repr_layers=[28], return_contacts=False
    )
    
    token_reps = out["representations"][28]  # Usa il layer 28
    
    emb_list = []
    for i, seq in enumerate(batch_strs):
        L = len(seq)
        emb = token_reps[i, 1:1+L].detach().cpu().numpy()
        emb_list.append(emb)
    return emb_list

random.shuffle(seqs_for_pca)  # <--- aggiungi questo
print(f"Usando {len(seqs_for_pca)} sequenze correnti (mescolate)")

# ------------------------
# 1) FIT INCREMENTAL PCA (invariato)
# ------------------------
print("Inizio IncrementalPCA (solo fit)...")
ipca = IncrementalPCA(n_components=pca_components, batch_size=None)

for i in tqdm(range(0, len(seqs_for_pca), pca_batch_size), desc="Fit IncrementalPCA"):
    torch.cuda.empty_cache()
    #gc.collect()  # <--- aggiungi questo
    batch_seqs = seqs_for_pca[i:i+pca_batch_size]
    emb_list = get_residue_embeddings_batch(batch_seqs)
    X_batch = np.concatenate(emb_list, axis=0)
    ipca.partial_fit(X_batch)

print("IncrementalPCA fit completato!")
print("Explained variance ratio (sum):", ipca.explained_variance_ratio_.sum())

# ------------------------
# 2) SALVA PCA (NUOVO - FINE CODICE!)
# ------------------------
joblib.dump(ipca, "TonB_ipca_fitted_640comp.joblib")
print("✅ PCA salvata in TonB_ipca_fitted_640comp.joblib")


# Metadati
joblib.dump({
    'pca_components': pca_components,
    'n_sequences_used': len(seqs_for_pca),
    'model_name': 'esm2_t33_650M_UR50D',
    'layer_used': get_num_layers(model),  # <--- layer info
    'mean_seq_len': np.mean([len(s) for s in seqs_for_pca]),
    'date': pd.Timestamp.now().isoformat()
}, "TonB_pca_metadata.joblib")