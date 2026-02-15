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
import joblib
from Bio import SeqIO
from Bio.Align import substitution_matrices
import gc
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 0) CONFIGURAZIONE E SETUP
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

CONFIG = {
    # --- Target DNAjb1 ---
    'target_name': 'DNAjb1',
    'seq_target': "MGKDYYQTLGLARGASDDEIKRAYRRQALRYPDKNKEPGAEEKFKEIAEAYDVLSDPRKREIFDRYGEEGLKGGGPSGGSSGGANGTSFSYTFGDPAMFAEFFGGRNP",
    
    # --- Percorsi File Input ---
    'fasta_uniref': './datasets/uniref50_subsample.fasta',
    'csv_random': './datasets/Random_dataset.csv',
    
    # --- Percorsi Output ---
    'output_dir': './datasets',
    'joblib_dir': './joblibs',
    
    # --- Parametri Dataset ---
    'samples_per_category': 50000, # 50k Uniref + 50k Random + 50k DNAjb1
    'seq_len_range': (50, 700),    # Range lunghezza per UniRef
    
    # --- Parametri Mutazione ---
    'max_mutations': 20,
    'T_blosum': 1.5,
    'p_mut': 0.8,
    'p_ins': 0.1,
    'p_del': 0.1,
    
    # --- Parametri PCA e Modello ---
    'num_layer': 28,
    'pca_components': 640,
    'pca_batch_size': 64,
    'random_seed': 42
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['joblib_dir'], exist_ok=True)

# Seed
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

# Caricamento Matrice BLOSUM
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
try:
    blosum = substitution_matrices.load("BLOSUM62")
except:
    # Fallback import
    from Bio.Align import substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")

# Definizione percorsi file
file_dataset_balanced = os.path.join(CONFIG['output_dir'], f"dataset_{CONFIG['target_name']}_balanced_150k.csv")
file_pca_model = os.path.join(CONFIG['joblib_dir'], f"Total_{CONFIG['target_name']}_ipca_fitted.joblib")
file_pca_meta = os.path.join(CONFIG['joblib_dir'], f"Total_{CONFIG['target_name']}_pca_metadata.joblib")


# -----------------------------------------------------------------------------
# 1) FUNZIONI UTILI (Mutazione & Caricamento)
# -----------------------------------------------------------------------------

def mutate_residue(seq, T=CONFIG['T_blosum']):
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
    
    # Boltzmann sampling
    exps = [math.exp(s/T) for s in scores]
    total = sum(exps)
    probs = [e/total for e in exps]
    seq[idx] = random.choices(choices, weights=probs)[0]
    return ''.join(seq)

def insert_residue(seq):
    seq = list(seq)
    idx = random.randrange(len(seq)+1)
    seq.insert(idx, random.choice(AA_LIST))
    return ''.join(seq)

def delete_residue(seq):
    if len(seq) <= 1: return seq
    seq = list(seq)
    del seq[random.randrange(len(seq))]
    return ''.join(seq)

def markov_mutation(seq):
    r = random.random()
    if r < CONFIG['p_mut']: return mutate_residue(seq)
    elif r < CONFIG['p_mut'] + CONFIG['p_ins']: return insert_residue(seq)
    else: return delete_residue(seq)

def load_uniref_subsample(fasta_path, target_n, length_range):
    if not os.path.exists(fasta_path):
        print(f"ATTENZIONE: File {fasta_path} non trovato!")
        return []
    valid_records = []
    min_len, max_len = length_range
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if min_len <= len(record.seq) <= max_len:
                valid_records.append(str(record.seq))
                if len(valid_records) >= target_n * 5: break
    if not valid_records: return []
    return random.sample(valid_records, min(target_n, len(valid_records)))

def load_csv_subsample(csv_path, target_n):
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path, header=None)
        seqs = df[0].tolist()
        if seqs[0] == "sequence": seqs.pop(0)
        if len(seqs) > target_n: return random.sample(seqs, target_n)
        return seqs
    except: return []


# -----------------------------------------------------------------------------
# 2) GENERAZIONE DATASET BILANCIATO
# -----------------------------------------------------------------------------

if not os.path.exists(file_dataset_balanced):
    print(f"\n--- Generazione Dataset Bilanciato per {CONFIG['target_name']} ---")
    
    # A. Generazione Mutanti DNAjb1
    print(f"Generazione 50k mutazioni di {CONFIG['target_name']}...")
    mutants = []
    for _ in tqdm(range(CONFIG['samples_per_category']), desc="Mutating DNAjb1"):
        n_mut = random.randint(1, CONFIG['max_mutations'])
        seq = CONFIG['seq_target']
        for _ in range(n_mut):
            seq = markov_mutation(seq)
        mutants.append(seq)
        
    # B. Caricamento UniRef50
    print("Caricamento UniRef50...")
    uniref = load_uniref_subsample(CONFIG['fasta_uniref'], CONFIG['samples_per_category'], CONFIG['seq_len_range'])
    
    # C. Caricamento Random
    print("Caricamento Random Dataset...")
    random_seqs = load_csv_subsample(CONFIG['csv_random'], CONFIG['samples_per_category'])
    
    # D. Unione e Shuffle
    print(f"Conteggi: {CONFIG['target_name']}={len(mutants)}, UniRef={len(uniref)}, Random={len(random_seqs)}")
    all_seqs = mutants + uniref + random_seqs
    random.shuffle(all_seqs)
    
    df = pd.DataFrame(all_seqs, columns=["sequence"])
    df.to_csv(file_dataset_balanced, index=False, header=False)
    print(f"Dataset salvato: {file_dataset_balanced}")
    
    del mutants, uniref, random_seqs, all_seqs
    gc.collect()
else:
    print(f"Dataset bilanciato già presente: {file_dataset_balanced}")


# -----------------------------------------------------------------------------
# 3) CARICAMENTO MODELLO ESM2
# -----------------------------------------------------------------------------
print("\n--- Caricamento Modello ESM-2 ---")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print(f"DataParallel attivo su {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)

# Funzione embedding helper
def forward_robust(model, tokens, layer):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        return model.module(tokens, repr_layers=[layer], return_contacts=False)
    return model(tokens, repr_layers=[layer], return_contacts=False)

@torch.no_grad()
def get_residue_embeddings_batch(sequences):
    data = [("seq", s) for s in sequences]
    _, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    out = forward_robust(model, batch_tokens, CONFIG['num_layer'])
    token_reps = out["representations"][CONFIG['num_layer']]
    
    return [token_reps[i, 1:1+len(s)].detach().cpu().numpy() for i, s in enumerate(batch_strs)]


# -----------------------------------------------------------------------------
# 4) FIT INCREMENTAL PCA
# -----------------------------------------------------------------------------
print(f"\n--- Inizio Fit IncrementalPCA (Total {CONFIG['target_name']}) ---")

# Lettura dataset
df = pd.read_csv(file_dataset_balanced, header=None)
sequences = df[0].tolist()

ipca = IncrementalPCA(n_components=CONFIG['pca_components'], batch_size=None)
total_batches = (len(sequences) + CONFIG['pca_batch_size'] - 1) // CONFIG['pca_batch_size']

for i in tqdm(range(0, len(sequences), CONFIG['pca_batch_size']), total=total_batches, desc="PCA Fitting"):
    batch_seqs = sequences[i : i + CONFIG['pca_batch_size']]
    try:
        emb_list = get_residue_embeddings_batch(batch_seqs)
        X_batch = np.concatenate(emb_list, axis=0)
        
        ipca.partial_fit(X_batch)
        
        del X_batch, emb_list
        # Pulizia periodica cache GPU
        if i % (CONFIG['pca_batch_size'] * 10) == 0:
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Errore batch {i}: {e}")
        continue

print("IncrementalPCA fit completato!")
print(f"Explained variance ratio (sum): {ipca.explained_variance_ratio_.sum():.4f}")

# -----------------------------------------------------------------------------
# 5) SALVATAGGIO
# -----------------------------------------------------------------------------
joblib.dump(ipca, file_pca_model)
joblib.dump({
    'pca_components': CONFIG['pca_components'],
    'model_name': 'esm2_t33_650M_UR50D',
    'composition': f"50k UniRef / 50k Random / 50k {CONFIG['target_name']}",
    'target_seq': CONFIG['seq_target']
}, file_pca_meta)

print(f"File salvati in {CONFIG['joblib_dir']}:")
print(f"- {os.path.basename(file_pca_model)}")
print(f"- {os.path.basename(file_pca_meta)}")