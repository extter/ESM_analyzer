import os
# Ottimizzazione memoria CUDA
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
import warnings

warnings.filterwarnings('ignore')

# ------------------------
# 0) SETUP E VARIABILI 
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

CONFIG = {
    'n_samples': 100000,  
    'tonb_length': 239,
    'tonb_length_range': (200, 300),
    'output_dir': './datasets/random_dataset', 
    'num_layer': 28, 
    
    # Variabili per la PCA
    'max_seqs_for_pca': 100000, 
    'pca_components': 128,
    'pca_batch_size': 64    
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
csv_path = os.path.join(CONFIG['output_dir'], "Random_dataset.csv") # Path costruito sicuro

# ------------------------
# 1) GENERAZIONE DATASET RANDOM
# ------------------------

if not os.path.exists(csv_path):
    random_proteins = []
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    print(f"Generazione di {CONFIG['n_samples']} sequenze casuali...")

    for _ in tqdm(range(CONFIG['n_samples']), desc="Generating sequences"):
        length = random.randint(*CONFIG['tonb_length_range'])
        seq = "".join(random.choice(aa_alphabet) for _ in range(length))
        random_proteins.append(seq)

    print(f"Media lunghezza: {np.mean([len(seq) for seq in random_proteins]):.0f} aa")

    df_random = pd.DataFrame(random_proteins, columns=["sequence"])
    df_random.to_csv(csv_path, index=False, header=False)
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
        # Hack per DataParallel con batch=1
        single_model = model.module
        return single_model(tokens, **kwargs)
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
    
    out = forward_with_single_gpu_if_small_batch(
        model, batch_tokens, repr_layers=[CONFIG['num_layer']], return_contacts=False
    )
    
    token_reps = out["representations"][CONFIG['num_layer']]
    
    emb_list = []
    for i, seq in enumerate(batch_strs):
        L = len(seq)
        emb = token_reps[i, 1:1+L].detach().cpu().numpy()
        emb_list.append(emb)
    return emb_list

# ------------------------
# 4) FIT INCREMENTAL PCA
# ------------------------

df = pd.read_csv(csv_path, header=None, names=["sequence"])
sequences = df["sequence"].tolist()
print(f"Caricate {len(sequences)} sequenze")

if len(sequences) > CONFIG['max_seqs_for_pca']:
    seqs_for_pca = sequences[:CONFIG['max_seqs_for_pca']]
else:
    seqs_for_pca = sequences

print(f"Inizio IncrementalPCA su {len(seqs_for_pca)} sequenze...")
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
joblib.dump(ipca, "Total_ipca_fitted.joblib")
print("PCA salvata in Total_ipca_fitted.joblib")

joblib.dump({
    'pca_components': CONFIG['pca_components'],
    'n_sequences_used': len(seqs_for_pca),
    'model_name': 'esm2_t33_650M_UR50D',
    'layer_used': CONFIG['num_layer']
}, "Total_pca_metadata.joblib")
print("Metadata salvati!")