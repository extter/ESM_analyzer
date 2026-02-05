
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

# ==========================================
# 0. CONFIGURAZIONE GLOBALE
# ==========================================
CONFIG = {
    'uniref_dir': '/kaggle/input/uniref50-sub',
    'random_csv': '/kaggle/working/Random_dataset.csv',  
    'tonb_csv': '/kaggle/input/mutations-of-tonb/TonB_mutations_dataset.csv',
    
    'final_csv': '/kaggle/working/dataset_proteine_balanced_150k.csv',
    'pca_model_path': 'Total_ipca_fitted.joblib',
    'pca_meta_path': 'Total_pca_metadata.joblib',
    
    'samples_per_category': 50000,
    'seq_len_range': (150, 700), 
    
    'esm_layer': 28,          
    'pca_components': 640,    
    'batch_size_seqs': 64,    
    'random_seed': 42
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ==========================
# 1. DATA PREPARATION 
# ==========================

def load_uniref_subsample(input_dir, target_n, length_range):
    """Carica un subsample casuale di sequenze UniRef filtrate per lunghezza."""
    print(f"\n--- Caricamento UniRef50 (Target: {target_n}) ---")
    
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(('.fasta', '.fa', '.fasta.gz'))]
    if not fasta_files:
        raise FileNotFoundError("Nessun file FASTA trovato in UniRef dir.")
    
    fasta_path = os.path.join(input_dir, fasta_files[0])
    print(f"Reading: {fasta_path}")
    
    valid_records = []
    min_len, max_len = length_range
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            L = len(record.seq)
            if min_len <= L <= max_len:
                valid_records.append(str(record.seq))
                if len(valid_records) >= target_n * 3:
                    break
    
    print(f"Sequenze valide trovate: {len(valid_records)}")
    
    return random.sample(valid_records, target_n)

def load_csv_subsample(csv_path, target_n, name="Dataset"):
    """Carica un CSV e prende un subsample del datset considerato"""
    print(f"\n--- Caricamento {name} (Target: {target_n}) ---")
    if not os.path.exists(csv_path):
        print(f"File non trovato in {csv_path}")
        return []
        
    df = pd.read_csv(csv_path, header=None, names=["sequence"])
    seqs = df["sequence"].tolist()
    
    if len(seqs) > target_n:
        print(f"Subsampling da {len(seqs)} a {target_n}...")
        return random.sample(seqs, target_n)
    else:
        print(f"Trovate solo {len(seqs)} sequenze. Le prendo tutte.")
        return seqs

if not os.path.exists(CONFIG['final_csv']):
    uniref_seqs = load_uniref_subsample(CONFIG['uniref_dir'], CONFIG['samples_per_category'], CONFIG['seq_len_range'])
    random_seqs = load_csv_subsample(CONFIG['random_csv'], CONFIG['samples_per_category'], "Random")
    tonb_seqs = load_csv_subsample(CONFIG['tonb_csv'], CONFIG['samples_per_category'], "TonB")
    
    all_seqs = uniref_seqs + random_seqs + tonb_seqs
    random.shuffle(all_seqs) 
    
    df_final = pd.DataFrame(all_seqs, columns=["sequence"])
    df_final.to_csv(CONFIG['final_csv'], index=False, header=False)
    
    print(f"Dataset Bilanciato salvato: {CONFIG['final_csv']}")
    print(f"Totale sequenze: {len(df_final)}")
    

    del uniref_seqs, random_seqs, tonb_seqs, all_seqs, df_final
    gc.collect()
else:
    print(f"\nDataset già esistente: {CONFIG['final_csv']}")

# ==========================================
# 2. MODEL SETUP (ESM-2)
# ==========================================
print("\n--- Caricamento Modello ESM-2 ---")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print(f"DataParallel Attivo su {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)

@torch.no_grad()
def get_residue_embeddings(sequences_batch):
    """
    Calcola gli embedding per un batch di sequenze proteiche usando ESM-2.

    Questa funzione esegue i seguenti passaggi:
    1. Tokenizzazione: Converte le stringhe di amminoacidi in token numerici.
    2. Inferenza: Passa i token al modello ESM-2 (su GPU).
    3. Estrazione: Recupera le rappresentazioni interne (hidden states) dal layer specificato.
    4. Post-processing: Rimuove i token speciali di inizio/fine sequenza e converte in numpy array.

    Args:
        sequences (list of str): Lista di sequenze proteiche.

    Returns:
        list of np.ndarray: Una lista dove ogni elemento è una matrice (L x D) contenente
                            gli embedding per ogni residuo della sequenza.
                            L = lunghezza sequenza, D = dimensione embedding.

    """
    data = [("seq", s) for s in sequences_batch]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    out = forward_robust(model, tokens, CONFIG['esm_layer'])
    token_reps = out["representations"][CONFIG['esm_layer']]
    
    emb_list = []
    for i, seq in enumerate(sequences_batch):
        L = len(seq)
        emb = token_reps[i, 1:1+L].detach().cpu().numpy()
        emb_list.append(emb)
    return emb_list

# ==========================================
# 3. INCREMENTAL PCA PIPELINE
# ==========================================
print(f"\n--- Inizio Incremental PCA (n_components={CONFIG['pca_components']}) ---")

df = pd.read_csv(CONFIG['final_csv'], header=None, names=["sequence"])
sequences = df["sequence"].tolist()
total_seqs = len(sequences)
print(f"Sequenze da processare: {total_seqs}")

ipca = IncrementalPCA(n_components=CONFIG['pca_components'])

batch_size = CONFIG['batch_size_seqs']
processed_residues = 0

for i in tqdm(range(0, total_seqs, batch_size), desc="Fitting PCA"):
    batch_seqs = sequences[i : i + batch_size]
    
    try:
        emb_list = get_residue_embeddings(batch_seqs)
        
        X_batch = np.concatenate(emb_list, axis=0)
        
        ipca.partial_fit(X_batch)
        processed_residues += X_batch.shape[0]
        
        del emb_list, X_batch
        
    except Exception as e:
        print(f"Errore nel batch {i}: {e}")
        continue

print("\n--- Fit Completato! ---")
print(f"Totale residui processati: {processed_residues}")
print(f"Explained Variance Ratio (Cumulative): {ipca.explained_variance_ratio_.sum():.4f}")

# ==========================================
# 4. SALVATAGGIO
# ==========================================
joblib.dump(ipca, CONFIG['pca_model_path'])
print(f"Modello PCA salvato in: {CONFIG['pca_model_path']}")

metadata = {
    'pca_components': CONFIG['pca_components'],
    'n_sequences_total': total_seqs,
    'n_residues_total': processed_residues,
    'model_name': 'esm2_t33_650M_UR50D',
    'layer': CONFIG['esm_layer'],
    'source_split': '50k UniRef / 50k Random / 50k TonB'
}
joblib.dump(metadata, CONFIG['pca_meta_path'])
print("Metadati salvati.")