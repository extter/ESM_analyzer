import random
import matplotlib.pyplot as plt
import numpy as np
from Bio.Align import substitution_matrices
import pandas as pd
from tqdm import tqdm  
import torch
import torch.nn as nn
from esm import pretrained
from sklearn.decomposition import IncrementalPCA
from Bio import SeqIO 
blosum62 = substitution_matrices.load("BLOSUM62")

# ------------------------
# CONFIG
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
torch.set_grad_enabled(False)
batch_converter = alphabet.get_batch_converter()

segment_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  
layer = 28
n_random = 200
n_conservative = 200
n_random_baseline = 100
n_uniref = 200  

# CAMBIA QUESTO PATH con il tuo file FASTA UniRef50
uniref_fasta_path = "../pca/datasets/uniref50_subsample.fasta" 

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# ------------------------
# SEQUENZE
# ------------------------
DNAbj1_seq ="MGKDYYQTLGLARGASDDEIKRAYRRQALRYPDKNKEPGAEEKFKEIAEAYDVLSDPRKREIFDRYGEEGLKGGGPSGGSSGGANGTSFSYTFGDPAMFAEFFGGRNP"

seq_hb = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)

# ------------------------
# FUNZIONI ESM
# ------------------------
@torch.no_grad()
def get_residue_embeddings(seq):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer][0, 1:-1]  # [L, D]
    return reps.cpu().numpy()

def split_into_segments(residue_embs, n_segments):
    L, D = residue_embs.shape
    boundaries = np.linspace(0, L, n_segments + 1).astype(int)

    segments = np.zeros((n_segments, D))
    for i in range(n_segments):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            seg = residue_embs[start:end].mean(axis=0)
            seg /= np.linalg.norm(seg) + 1e-8
            segments[i] = seg
    return segments

def segmentwise_cosine(seg_a, seg_b):
    return np.sum(seg_a * seg_b, axis=1)

def global_similarity(seg_a, seg_b):
    return segmentwise_cosine(seg_a, seg_b).mean()

# ------------------------
# MUTAZIONI
# ------------------------

def conservative_mutation_blosum(seq, min_score=2):
    seq = list(seq)
    pos = random.randrange(len(seq))
    aa = seq[pos]

    candidates = []
    for aa2 in aa_list:
        if aa2 == aa:
            continue
        score = blosum62.get((aa, aa2), blosum62.get((aa2, aa), -10))
        if score > min_score:
            candidates.append(aa2)

    if not candidates:
        return ''.join(seq)

    seq[pos] = random.choice(candidates)
    return ''.join(seq)

print("Caricamento UniRef50...")
uniref_sequences = []

try:
    with open(uniref_fasta_path, "r") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta"), start=1):
            seq_str = str(record.seq)
            L = len(seq_str)

            if not (150 <= L <= 700):
                continue

            if len(uniref_sequences) < n_uniref:
                uniref_sequences.append(seq_str)
            else:
                j = random.randint(1, i)
                if j <= n_uniref:
                    uniref_sequences[j - 1] = seq_str



    print(f"Caricate {len(uniref_sequences)} sequenze UniRef50")

except FileNotFoundError:
    print(f"File {uniref_fasta_path} non trovato!")
    n_uniref = 0
    sims_DNAbj1_uniref = []


# ------------------------
# EMBEDDING UNIREF
# ------------------------
print("Calcolo embedding UniRef50...")
uniref_embs = []
for seq in tqdm(uniref_sequences, desc="UniRef embeddings"):
    emb = get_residue_embeddings(seq)
    uniref_embs.append(emb)

def random_sequence(length):
    return ''.join(random.choice(aa_list) for _ in range(length))

# ------------------------
# GENERA ENSEMBLE RANDOM
# ------------------------
print("Generazione ensemble random...")
random_seqs = [random_sequence(len(DNAbj1_seq) + random.randint(-50, 50)) for _ in range(n_random_baseline)]
all_random_segs_dict = {}  # cache embeddings per numero segmenti
print("Calcolo embedding random batch...")
for rseq in tqdm(random_seqs, desc="Random embeddings"):
    emb = get_residue_embeddings(rseq)
    all_random_segs_dict[rseq] = emb

# ------------------------
# LOOP SUI NUMERI DI SEGMENTI
# ------------------------
results = {}
DNAbj1_emb_full = get_residue_embeddings(DNAbj1_seq)  # embedding DNAbj1 intero

for n_seg in segment_list:
    print(f"\n=== Segmenti: {n_seg} ===")
    DNAbj1_seg = split_into_segments(DNAbj1_emb_full, n_seg)
    
    # ------------------------
    # DNAbj1 vs ensemble random
    # ------------------------
    ensemble_sims = []
    for rseq in random_seqs:
        rseg = split_into_segments(all_random_segs_dict[rseq], n_seg)
        ensemble_sims.append(global_similarity(DNAbj1_seg, rseg))
    mean_ensemble = np.mean(ensemble_sims)
    std_ensemble = np.std(ensemble_sims)
    
    # ------------------------
    # DNAbj1 vs conservative
    # ------------------------
    sims_cons = []
    for _ in range(n_conservative):
        mut_seq = conservative_mutation_blosum(DNAbj1_seq)
        mut_seg = split_into_segments(get_residue_embeddings(mut_seq), n_seg)
        sims_cons.append(global_similarity(DNAbj1_seg, mut_seg))
    mean_cons = np.mean(sims_cons)
    std_cons = np.std(sims_cons)

    # ------------------------
    # DNAbj1 vs UniRef50 random  
    # ------------------------
    sims_uniref = []
    for emb in uniref_embs:
        useg = split_into_segments(emb, n_seg)
        sims_uniref.append(global_similarity(DNAbj1_seg, useg))
    mean_uniref = np.mean(sims_uniref)
    std_uniref = np.std(sims_uniref)
    
    # ------------------------
    # DELTA
    # ------------------------
    delta_rand = mean_cons - mean_ensemble
    delta_uniref = mean_cons - mean_uniref
    
    print(f"DNAbj1 vs Random:   {mean_ensemble:.4f} ± {std_ensemble:.4f}")
    print(f"DNAbj1 vs UniRef:   {mean_uniref:.4f} ± {std_uniref:.4f}")
    print(f"DNAbj1 vs Cons:     {mean_cons:.4f} ± {std_cons:.4f}")
    print(f"Δ Cons–Random:   {delta_rand:.4f}")
    print(f"Δ Cons–UniRef:   {delta_uniref:.4f}")
    
    results[n_seg] = {
        "ensemble_sims": ensemble_sims,
        "cons_sims": sims_cons,
        "uniref_sims": sims_uniref,
        "delta_rand": delta_rand,
        "delta_uniref": delta_uniref
    }

# ------------------------
# GRAFICO BOX PLOT
# ------------------------
plt.figure(figsize=(18,6))
data_to_plot = []
labels = []

for n_seg in segment_list:
    data_to_plot.append(results[n_seg]["cons_sims"])
    data_to_plot.append(results[n_seg]["ensemble_sims"])
    data_to_plot.append(results[n_seg]["uniref_sims"])
    
    labels.append(f"Cons {n_seg}")
    labels.append(f"Random {n_seg}")
    labels.append(f"UniRef {n_seg}")

plt.boxplot(data_to_plot, labels=labels, showfliers=False)
plt.xticks(rotation=60)
plt.ylabel("Segment-wise cosine similarity")
plt.title("DNAbj1: Conservative vs Random vs UniRef50")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./result.png", dpi = 300)
plt.show()