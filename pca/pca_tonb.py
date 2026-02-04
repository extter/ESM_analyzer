import random
import math
import numpy as np
from Bio.Align import substitution_matrices
import pandas as pd
from tqdm import tqdm  
import torch
import torch.nn as nn
from esm import pretrained
from sklearn.decomposition import IncrementalPCA
import joblib  # per salvare PCA

# ------------------------
# CONFIGURAZIONE
# ------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
blosum = substitution_matrices.load("BLOSUM62")
T_blosum = 1.5  # temperatura Boltzmann
p_mut = 0.8     # probabilità di mutazione
p_ins = 0.1     # probabilità di inserzione
p_del = 0.1     # probabilità di delezione

# Sequenza di riferimento TonB (esempio)
seq_ref = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

# Numero di sequenze da generare
n_sequences = 100000

# Numero massimo di mutazioni per sequenza
max_mutations = 20

# ------------------------
# FUNZIONI MUTAZIONE
# ------------------------
def mutate_residue(seq, T=T_blosum):
    seq = list(seq)
    idx = random.randrange(len(seq))
    original = seq[idx]

    scores, choices = [], []
    for aa in AA_LIST:
        if aa == original:
            continue
        key = (original, aa) if (original, aa) in blosum else (aa, original)
        if key in blosum:
            scores.append(blosum[key])
            choices.append(aa)

    if not choices:  # fallback
        return ''.join(seq)

    exps = [math.exp(s/T) for s in scores]
    total = sum(exps)
    probs = [e/total for e in exps]

    new_aa = random.choices(choices, weights=probs)[0]
    seq[idx] = new_aa
    return ''.join(seq)

def insert_residue(seq):
    seq = list(seq)
    idx = random.randrange(len(seq)+1)
    aa = random.choice(AA_LIST)
    seq.insert(idx, aa)
    return ''.join(seq)

def delete_residue(seq):
    if len(seq) <= 1:
        return seq
    seq = list(seq)
    idx = random.randrange(len(seq))
    del seq[idx]
    return ''.join(seq)

def markov_mutation(seq, p_mut=p_mut, p_ins=p_ins, p_del=p_del):
    r = random.random()
    if r < p_mut:
        return mutate_residue(seq)
    elif r < p_mut + p_ins:
        return insert_residue(seq)
    else:
        return delete_residue(seq)

# ------------------------
# GENERAZIONE DATASET CON PROGRESS BAR
# ------------------------
dataset = []

for _ in tqdm(range(n_sequences), desc="Generazione sequenze"):
    n_mut = random.randint(1, max_mutations)
    seq = seq_ref
    for _ in range(n_mut):
        seq = markov_mutation(seq)
    dataset.append(seq)

# ------------------------
# SALVATAGGIO SU CSV
# ------------------------
df = pd.DataFrame({"sequence": dataset})
df.to_csv("./datasets/TonB_mutations_dataset.csv", index=False)
print("Dataset salvato in datasets/TonB_mutations_dataset.csv")




# FINE CREAZIONE DATASET MUTAZIONI. STARTING TO FIT THE PCA



# ------------------------
# CONFIG (invariato)
# ------------------------

csv_path = "./datasets/TonB_mutations_dataset.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca_components = 640
pca_batch_size = 64
max_seqs_for_pca = 100000

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
df = pd.read_csv(csv_path)
sequences = df["sequence"].tolist()
print(f"Caricate {len(sequences)} sequenze dal CSV")

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
joblib.dump(ipca, "./joblibs/TonB_ipca_fitted.joblib")
print("✅ PCA salvata in joblibs/TonB_ipca_final.joblib")



'''
# Opzionale: salva anche info dataset
joblib.dump({
    'pca_components': pca_components,
    'n_sequences_used': len(seqs_for_pca),
    'model_name': 'esm2_t33_650M_UR50D'
}, "./joblibs/TonB_pca_metadata_final.joblib")
print("✅ Metadata salvati!")
'''