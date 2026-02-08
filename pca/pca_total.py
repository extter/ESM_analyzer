
import os
# Ottimizzazione memoria CUDA per evitare frammentazione
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
import torch
import torch.nn as nn
from esm import pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import joblib
from Bio import SeqIO
import gc
import warnings
import sys

warnings.filterwarnings('ignore')

# ---------------------
# 0) SETUP E VARIABILI
# ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

CONFIG = {
    'uniref_dir': './datasets/uniref50_subsample.fasta',
    'random_csv': './datasets/Random_dataset.csv',  
    'tonb_csv': './datasets/TonB_mutations_dataset.csv',
    
    'final_csv': './datasets/dataset_proteine_balanced_150k.csv',
    'output_dir': './joblibs',
    'samples_per_category': 50000,
    'seq_len_range': (150, 700), 
    
    'num_layer': 28,          
    'pca_components': 640,    
    'pca_batch_size': 64,    
    'random_seed': 42
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ------------------------
# 1) FUNZIONI CARICAMENTO DATASET
# ------------------------

def load_uniref_subsample(input_dir, target_n, length_range):
    """Carica un campione casuale da un file FASTA filtrato per lunghezza."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File FASTA non trovato: {fasta_path}")
    
    valid_records = []
    min_len, max_len = length_range
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if min_len <= len(record.seq) <= max_len:
                valid_records.append(str(record.seq))
                if len(valid_records) >= target_n * 3: 
                    break
    
    if not valid_records:
        return []
    
    return random.sample(valid_records, min(target_n, len(valid_records)))

def load_csv_subsample(csv_path, target_n):
    """Carica un campione casuale da un file CSV"""
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path, header=None, names=["sequence"])
    seqs = df["sequence"].tolist()
    return random.sample(seqs, min(target_n, len(seqs)))

if not os.path.exists(CONFIG['final_csv']):
    print("Generazione dataset bilanciato...")
    u_seqs = load_uniref_subsample(CONFIG['uniref_dir'], CONFIG['samples_per_category'], CONFIG['seq_len_range'])
    r_seqs = load_csv_subsample(CONFIG['random_csv'], CONFIG['samples_per_category'])
    t_seqs = load_csv_subsample(CONFIG['tonb_csv'], CONFIG['samples_per_category'])
    
    all_seqs = u_seqs + r_seqs + t_seqs
    random.shuffle(all_seqs)
    pd.DataFrame(all_seqs, columns=["sequence"]).to_csv(CONFIG['final_csv'], index=False, header=False)
    print(f"Dataset salvato: {len(all_seqs)} sequenze.")
    del u_seqs, r_seqs, t_seqs, all_seqs; gc.collect()

# ----------------------
# 2) CARICA MODELLO ESM2 
# ----------------------
print("\n--- Caricamento Modello ESM-2 ---")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print(f"DataParallel Attivo su {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)
else:
    print("Uso una sola GPU")

# ------------------------
# 3) FUNZIONI EMBEDDING
# ------------------------

def forward_robust(model, tokens, layer):
    """Gestisce il forward pass prevenendo errori con batch=1."""
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        return model.module(tokens, repr_layers=[layer], return_contacts=False)
    return model(tokens, repr_layers=[layer], return_contacts=False)

@torch.no_grad()
def get_residue_embeddings_batch(sequences):
    """Estrae gli embedding per-residuo dal layer specificato di ESM."""
    data = [("seq", s) for s in sequences]
    _, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    out = forward_robust(model, batch_tokens, CONFIG['num_layer'])
    token_reps = out["representations"][CONFIG['num_layer']]
    
    return [token_reps[i, 1:1+len(s)].detach().cpu().numpy() for i, s in enumerate(batch_strs)]

# ------------------------
# 4) FIT INCREMENTAL PCA
# ------------------------
df = pd.read_csv(CONFIG['final_csv'], header=None, names=["sequence"])
sequences = df["sequence"].tolist()
ipca = IncrementalPCA(n_components=CONFIG['pca_components'])

print(f"Inizio PCA su {len(sequences)} sequenze...")
for i in tqdm(range(0, len(sequences), CONFIG['pca_batch_size']), desc="PCA Fitting"):
    batch_seqs = sequences[i : i + CONFIG['pca_batch_size']]
    try:
        emb_list = get_residue_embeddings_batch(batch_seqs)
        X_batch = np.concatenate(emb_list, axis=0)
        ipca.partial_fit(X_batch)
        del X_batch, emb_list
    except Exception as e:
        print(f"Errore nel batch {i}: {e}")

# ------------------------
# 5) SALVATAGGIO
# ------------------------
pca_path = os.path.join(CONFIG['output_dir'], "Total_ipca_fitted.joblib")
meta_path = os.path.join(CONFIG['output_dir'], "Total_pca_metadata.joblib")

joblib.dump(ipca, pca_path)
joblib.dump({
    'pca_components': CONFIG['pca_components'],
    'model_name': 'esm2_t33_650M_UR50D',
    'layer_used': CONFIG['num_layer'],
    'composition': '50k UniRef / 50k Random / 50k TonB'
}, meta_path)

print(f"PCA e Metadati salvati in {CONFIG['output_dir']}")