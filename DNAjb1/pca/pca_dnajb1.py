import os
# Ottimizzazione memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import math
import time
import torch
import torch.nn as nn
from esm import pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from Bio.Align import substitution_matrices
import joblib
import warnings 

warnings.filterwarnings('ignore')

# ------------------------
# 0) SETUP E VARIABILI 
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

CONFIG = {
    'target_name': 'DNAjb1',
    'n_samples': 100000,
    'max_mutations': 20,
    'seq_ref': "MGKDYYQTLGLARGASDDEIKRAYRRQALRYPDKNKEPGAEEKFKEIAEAYDVLSDPRKREIFDRYGEEGLKGGGPSGGSSGGANGTSFSYTFGDPAMFAEFFGGRNP",
    'output_dir': './datasets',
    'joblib_dir': './joblibs',
    'num_layer': 28,
    
    # Parametri Mutazione
    'T_blosum': 1.5,
    'p_mut': 0.8,
    'p_ins': 0.1,
    'p_del': 0.1,
    
    # Variabili per la PCA
    'max_seqs_for_pca': 100000,
    'pca_components': 640,
    'pca_batch_size': 64
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['joblib_dir'], exist_ok=True)

# Nome file CSV specifico per DNAjb1
csv_path = os.path.join(CONFIG['output_dir'], f"{CONFIG['target_name']}_mutations_dataset.csv")

# Inizializzazione BLOSUM62
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
try:
    blosum = substitution_matrices.load("BLOSUM62")
except:
    from Bio.Align import substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")

# ------------------------
# 1) FUNZIONI DI MUTAZIONE E GENERAZIONE
# ------------------------

def mutate_residue(seq, T=CONFIG['T_blosum']):
    """Esegue una mutazione puntiforme basata sulla matrice di sostituzione BLOSUM62"""
    seq = list(seq)
    idx = random.randrange(len(seq))
    original = seq[idx]
    scores, choices = [], []
    for aa in AA_LIST:
        if aa == original: continue
        key = (original, aa) if (original, aa) in blosum else (aa, original)
        if key in blosum:
            scores.append(blosum[key])
            choices.append(aa)
    if not choices: return ''.join(seq)
    exps = [math.exp(s/T) for s in scores]
    total = sum(exps)
    probs = [e/total for e in exps]
    seq[idx] = random.choices(choices, weights=probs)[0]
    return ''.join(seq)

def insert_residue(seq):
    """ Inserisce un amminoacido casuale in una posizione casuale della sequenza"""
    seq = list(seq)
    idx = random.randrange(len(seq)+1)
    seq.insert(idx, random.choice(AA_LIST))
    return ''.join(seq)

def delete_residue(seq):
    """Rimuove un amminoacido da una posizione casuale (se la lunghezza > 1)"""
    if len(seq) <= 1: return seq
    seq = list(seq)
    del seq[random.randrange(len(seq))]
    return ''.join(seq)

def markov_mutation(seq):
    """Seleziona stocasticamente un tipo di mutazione (Sostituzione, Inserzione, Delezione)"""
    r = random.random()
    if r < CONFIG['p_mut']: return mutate_residue(seq)
    elif r < CONFIG['p_mut'] + CONFIG['p_ins']: return insert_residue(seq)
    else: return delete_residue(seq)

if not os.path.exists(csv_path):
    dataset = []
    print(f"Generazione di {CONFIG['n_samples']} sequenze mutate per {CONFIG['target_name']}...")
    for _ in tqdm(range(CONFIG['n_samples']), desc=f"Mutating {CONFIG['target_name']}"):
        n_mut = random.randint(1, CONFIG['max_mutations'])
        seq = CONFIG['seq_ref']
        for _ in range(n_mut):
            seq = markov_mutation(seq)
        dataset.append(seq)
    
    df = pd.DataFrame({"sequence": dataset})
    df.to_csv(csv_path, index=False)
    print(f"Dataset salvato in: {csv_path}")
else:
    print(f"Dataset già presente in: {csv_path}")


# ------------------------
# 2) CARICA MODELLO ESM2
# ------------------------
print("Caricamento modello ESM2...")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print(f"Uso {torch.cuda.device_count()} GPU con DataParallel")
    model = nn.DataParallel(model)
else:
    print("Uso una sola GPU")


# ------------------------
# 3) FUNZIONI EMBEDDING
# ------------------------
def forward_with_single_gpu_if_small_batch(model, tokens, **kwargs):
    """
    Esegue il forward pass del modello gestendo il caso limite di batch_size=1 con DataParallel.
    """
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        return model.module(tokens, **kwargs)
    return model(tokens, **kwargs)

@torch.no_grad()
def get_residue_embeddings_batch(sequences):
    """
    Calcola gli embedding per un batch di sequenze proteiche usando ESM-2.
    """
    data = [("seq", s) for s in sequences]
    _, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    out = forward_with_single_gpu_if_small_batch(
        model, batch_tokens, repr_layers=[CONFIG['num_layer']], return_contacts=False
    )
    
    token_reps = out["representations"][CONFIG['num_layer']]
    emb_list = [token_reps[i, 1:1+len(seq)].detach().cpu().numpy() for i, seq in enumerate(batch_strs)]
    return emb_list

# ------------------------
# 4) FIT INCREMENTAL PCA
# ------------------------
df = pd.read_csv(csv_path)
sequences = df["sequence"].tolist()
seqs_for_pca = sequences[:CONFIG['max_seqs_for_pca']]

print(f"Inizio IncrementalPCA su {len(seqs_for_pca)} sequenze ({CONFIG['target_name']})...")
ipca = IncrementalPCA(n_components=CONFIG['pca_components'], batch_size=None)

batch_size = CONFIG['pca_batch_size']
for i in tqdm(range(0, len(seqs_for_pca), batch_size), desc="Fit IncrementalPCA"):
    batch_seqs = seqs_for_pca[i:i+batch_size]
    emb_list = get_residue_embeddings_batch(batch_seqs)
    X_batch = np.concatenate(emb_list, axis=0)
    ipca.partial_fit(X_batch)

print("IncrementalPCA fit completato!")
print(f"Explained variance ratio (sum): {ipca.explained_variance_ratio_.sum():.4f}")


# ------------------------
# 5) SALVATAGGIO
# ------------------------
# Nomi file specifici per DNAjb1
pca_filename = os.path.join(CONFIG['joblib_dir'], f"{CONFIG['target_name']}_ipca_fitted.joblib")
meta_filename = os.path.join(CONFIG['joblib_dir'], f"{CONFIG['target_name']}_pca_metadata.joblib")

joblib.dump(ipca, pca_filename)
joblib.dump({
    'pca_components': CONFIG['pca_components'],
    'n_sequences_used': len(seqs_for_pca),
    'model_name': 'esm2_t33_650M_UR50D',
    'layer_used': CONFIG['num_layer'],
    'ref_sequence': CONFIG['seq_ref'],
    'target_name': CONFIG['target_name']
}, meta_filename)

print(f"PCA salvata in {pca_filename}")
print(f"Metadata salvati in {meta_filename}")