import os
# Ottimizzazione memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.nn as nn
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from Bio import SeqIO
import joblib
import gc
import warnings

warnings.filterwarnings('ignore')

# ------------------------
# 0) SETUP E VARIABILI 
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

CONFIG = {
    'input_dir': './datasets/uniref50_subsample.fasta'',
    'joblibs_dir':'./joblibs'
    'target_n': 100000,
    'seq_len_range': (150, 700),
    'num_layer': 28,
    
    'pca_components': 640,
    'pca_batch_size': 64,
    'random_seed': 42
}

random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ------------------------
# 1) CARICAMENTO E SUBSAMPLING
# ------------------------
def get_fasta_subsample(input_dir, target_n, length_range):
    """Estrae un subsample casuale di sequenze da file FASTA nel range indicato."""
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(('.fasta', '.fa', '.fasta.gz'))]
    if not fasta_files: raise FileNotFoundError("Nessun file FASTA trovato.")
    
    fasta_path = os.path.join(input_dir, fasta_files[0])
    print(f"Lettura da: {fasta_path}")
    
    valid_records = []
    min_l, max_l = length_range
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if min_l < len(record.seq) < max_l:
                valid_records.append(str(record.seq))
                if len(valid_records) >= target_n * 2: break # Buffer per shuffle
    
    return random.sample(valid_records, min(target_n, len(valid_records)))

sequences = get_fasta_subsample(CONFIG['input_dir'], CONFIG['target_n'], CONFIG['seq_len_range'])
random.shuffle(sequences)

print(f"{len(sequences)} sequenze caricate")
print(f"Media lunghezza: {np.mean([len(s) for s in sequences]):.0f} AA")
# ----------------------
# CARICA MODELLO ESM2 
# ----------------------
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print("Uso", torch.cuda.device_count(), "GPU con DataParallel")
    model = nn.DataParallel(model, device_ids=[0, 1])
else:
    print("Uso una sola GPU")

# ---------------------
# FUNZIONI EMBEDDING 
# ---------------------
def forward_with_single_gpu_if_small_batch(model, tokens, **kwargs):
    """
    Esegue il forward pass del modello gestendo il caso limite di batch_size=1 con DataParallel.
    
    Quando si usa nn.DataParallel con un batch di dimensione 1, PyTorch può sollevare errori
    o comportarsi in modo inefficiente perché tenta di dividere il batch tra le GPU.

    Args:
        model (nn.Module): Il modello PyTorch (può essere avvolto in DataParallel).
        tokens (torch.Tensor): Il tensore dei token di input.
        **kwargs: Argomenti aggiuntivi da passare al modello (es. repr_layers).

    Returns:
        dict: L'output del modello ESM (contiene 'logits', 'representations', etc.).
    """
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        single_model = model.module.to("cuda:0")
        return single_model(tokens.to("cuda:0"), **kwargs)
    else:
        return model(tokens, **kwargs)

@torch.no_grad()
def get_residue_embeddings_batch(sequences):
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

    data = [("seq", s) for s in sequences]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    num_layers = get_num_layers(model)
    out = forward_with_single_gpu_if_small_batch(
        model, batch_tokens, repr_layers=[28], return_contacts=False
    )
    
    token_reps = out["representations"][28] 
    
    emb_list = []
    for i, seq in enumerate(batch_strs):
        L = len(seq)
        emb = token_reps[i, 1:1+L].detach().cpu().numpy()
        emb_list.append(emb)
    return emb_list

random.shuffle(seqs_for_pca) 
print(f"Usando {len(seqs_for_pca)} sequenze correnti (mescolate)")

# ------------------------
# 4) FIT INCREMENTAL PCA
# ------------------------
print(f"Inizio IncrementalPCA su {len(sequences)} sequenze...")
ipca = IncrementalPCA(n_components=CONFIG['pca_components'])

for i in tqdm(range(0, len(sequences), CONFIG['pca_batch_size']), desc="PCA Fitting"):
    torch.cuda.empty_cache() # Libera memoria GPU ad ogni batch
    batch_seqs = sequences[i : i + CONFIG['pca_batch_size']]
    
    try:
        emb_list = get_residue_embeddings_batch(batch_seqs)
        X_batch = np.concatenate(emb_list, axis=0)
        ipca.partial_fit(X_batch)
        del X_batch, emb_list
    except Exception as e:
        print(f"Errore nel batch {i}: {e}")

print(f"Fit completato. Varianza totale spiegata: {ipca.explained_variance_ratio_.sum():.4f}")

# ------------------------
# 5) SALVATAGGIO
# ------------------------
pca_path = os.path.join(CONFIG['joblibs_dir'], "Uniref_ipca_fitted.joblib")
meta_path = os.path.join(CONFIG['joblibs_dir'], "Uniref_pca_metadata.joblib")

joblib.dump(ipca, pca_path)
joblib.dump({
    'pca_components': CONFIG['pca_components'],
    'model_name': 'esm2_t33_650M_UR50D',
    'layer_used': CONFIG['num_layer']
}, meta_path)

print(f"PCA e Metadati salvati in {CONFIG['joblibs_dir']}")