import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from esm import pretrained
from tqdm import tqdm
from Bio.Align import substitution_matrices
from Bio import SeqIO  
from scipy.spatial.distance import euclidean
from collections import defaultdict

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

layer = 28
n_segments = 12
n_random = 200
n_conservative = 200
n_random_baseline = 100
n_uniref = 200  


uniref_fasta_path = "../pca/datasets/uniref50_subsample.fasta"  

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# --- Sequenza target ---
with open("../../sequences/dnajb1.txt") as f:
    DNAjb1_sequence = f.read().strip()
print(DNAjb1_sequence)

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
def random_sequence(length):
    return ''.join(random.choice(aa_list) for _ in range(length))

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
    print(f"❌ File {uniref_fasta_path} non trovato!")
    n_uniref = 0
    sims_DNAjb1_uniref = []

# ------------------------
# EMBEDDING BASE
# ------------------------
print("Calcolo embedding base...")

print("Calcolo embedding DNAjb1...")
print(device)
DNAjb1_res = get_residue_embeddings(DNAjb1_sequence)
DNAjb1_seg = split_into_segments(DNAjb1_res, n_segments)

print("Calcolo embedding Hb...")
hb_res = get_residue_embeddings(seq_hb)
hb_seg = split_into_segments(hb_res, n_segments)

# ------------------------
# STATISTICHE
# ------------------------
sims_DNAjb1_DNAjb1 = []
sims_DNAjb1_random = []
sims_DNAjb1_cons = []
sims_DNAjb1_uniref = []
sims_random_random = []

# DNAjb1 vs DNAjb1 (identico, per controllo numerico)
sims_DNAjb1_DNAjb1.append(global_similarity(DNAjb1_seg, DNAjb1_seg))

# DNAjb1 vs Hb
sim_DNAjb1_hb = global_similarity(DNAjb1_seg, hb_seg)

# DNAjb1 vs random
print("DNAjb1 vs random...")
for _ in tqdm(range(n_random), desc="DNAjb1 vs random"):
    rseq = random_sequence(len(DNAjb1_sequence))
    rseg = split_into_segments(get_residue_embeddings(rseq), n_segments)
    sims_DNAjb1_random.append(global_similarity(DNAjb1_seg, rseg))

# DNAjb1 vs mutazioni conservative
print("DNAjb1 vs conservative...")
for _ in tqdm(range(n_conservative), desc="DNAjb1 vs conservative"):
    mut = conservative_mutation_blosum(DNAjb1_sequence, min_score=2)
    mut_seg = split_into_segments(get_residue_embeddings(mut), n_segments)
    sims_DNAjb1_cons.append(global_similarity(DNAjb1_seg, mut_seg))

# DNAjb1 vs UniRef50
if n_uniref > 0:
    print("DNAjb1 vs UniRef50...")
    for seq_str in tqdm(uniref_sequences[:n_uniref], desc="DNAjb1 vs UniRef50"):
        uref_seg = split_into_segments(get_residue_embeddings(seq_str), n_segments)
        sims_DNAjb1_uniref.append(global_similarity(DNAjb1_seg, uref_seg))
else:
    sims_DNAjb1_uniref = []

# Random vs random (baseline) - sequenze casuali
print("Random vs random baseline (sequenze casuali)...")
random_seqs = []
for i in tqdm(range(n_random_baseline), desc="Generate random seqs"):
    rseq = random_sequence(len(DNAjb1_sequence) + random.randint(-50, 50))
    random_seqs.append(rseq)

print("Calcolo embedding random batch...")
all_random_segs = []
for rseq in tqdm(random_seqs, desc="Random embeddings"):
    rseg = split_into_segments(get_residue_embeddings(rseq), n_segments)
    all_random_segs.append(rseg)

# Calcola matrice upper-triangular per random vs random
sims_random_random = []
for i in tqdm(range(n_random_baseline), desc="Random vs random casuali"):
    for j in range(i+1, n_random_baseline):
        sim = global_similarity(all_random_segs[i], all_random_segs[j])
        sims_random_random.append(sim)

# ------------------------
# NUOVO: UniRef50 vs UniRef50 (random pairs)
# ------------------------
sims_uniref_uniref = []
if len(uniref_sequences) >= 20:  # minimo per coppie significative
    print("UniRef50 vs UniRef50 (random pairs)...")
    n_uniref_pairs = min(500, len(uniref_sequences)*(len(uniref_sequences)-1)//2)  # max coppie possibili
    
    # Precalcola tutti gli embedding UniRef per efficienza
    print("Precalcolo embedding UniRef50...")
    all_uniref_segs = []
    for seq_str in tqdm(uniref_sequences[:50], desc="UniRef embeddings"):  # max 50 per memoria/tempo
        uref_seg = split_into_segments(get_residue_embeddings(seq_str), n_segments)
        all_uniref_segs.append(uref_seg)
    
    # Campiona coppie random (upper triangular per evitare duplicati)
    n_pairs = min(500, len(all_uniref_segs)*(len(all_uniref_segs)-1)//2)
    pairs = []
    for i in range(len(all_uniref_segs)):
        for j in range(i+1, len(all_uniref_segs)):
            pairs.append((i,j))
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break
    
    print(f"Calcolo {len(pairs)} coppie UniRef50 vs UniRef50...")
    for i, j in tqdm(pairs, desc="UniRef vs UniRef"):
        sim = global_similarity(all_uniref_segs[i], all_uniref_segs[j])
        sims_uniref_uniref.append(sim)
else:
    print("UniRef50 troppo piccolo per coppie significative")
    sims_uniref_uniref = []
    
# ------------------------
# RISULTATI NUMERICI (MODIFICATO)
# ------------------------
print("\n=== RISULTATI ===")
print("DNAjb1 vs DNAjb1:                           %.4f" % sims_DNAjb1_DNAjb1[0])
print("DNAjb1 vs Hb:                             %.4f" % sim_DNAjb1_hb)
print("DNAjb1 vs Random: mean = %.4f ± %.4f" % (np.mean(sims_DNAjb1_random), np.std(sims_DNAjb1_random)))
print("DNAjb1 vs Conservative: mean = %.4f ± %.4f" % (np.mean(sims_DNAjb1_cons), np.std(sims_DNAjb1_cons)))
if len(sims_DNAjb1_uniref) > 0:
    print("DNAjb1 vs UniRef50 (n=%d): mean = %.4f ± %.4f" % (len(sims_DNAjb1_uniref), np.mean(sims_DNAjb1_uniref), np.std(sims_DNAjb1_uniref)))
print("Random vs Random (casuali): mean = %.4f ± %.4f" % (np.mean(sims_random_random), np.std(sims_random_random)))
if len(sims_uniref_uniref) > 0:
    print("UniRef50 vs UniRef50 (n=%d): mean = %.4f ± %.4f" % (len(sims_uniref_uniref), np.mean(sims_uniref_uniref), np.std(sims_uniref_uniref)))

print("\n=== DELTE vs BASELINE RandomRandom ===")
print("DNAjb1(Cons) - RandomRandom mean: %.4f" % (np.mean(sims_DNAjb1_cons) - np.mean(sims_random_random)))
print("DNAjb1(Hb)   - RandomRandom mean: %.4f" % (sim_DNAjb1_hb - np.mean(sims_random_random)))
print("DNAjb1(Rand) - RandomRandom mean: %.4f" % (np.mean(sims_DNAjb1_random) - np.mean(sims_random_random)))
if len(sims_DNAjb1_uniref) > 0:
    print("DNAjb1(UniRef) - RandomRandom mean: %.4f" % (np.mean(sims_DNAjb1_uniref) - np.mean(sims_random_random)))
if len(sims_uniref_uniref) > 0:
    print("UniRef vs UniRef - RandomRandom mean: %.4f" % (np.mean(sims_uniref_uniref) - np.mean(sims_random_random)))

# ------------------------
# BOXPLOT (MODIFICATO)
# ------------------------
plt.figure(figsize=(16, 6))

data = [sims_DNAjb1_DNAjb1, sims_DNAjb1_cons, sims_DNAjb1_random, sims_random_random, [sim_DNAjb1_hb]]
labels = ["DNAjb1 vs DNAjb1", "DNAjb1 vs Conservative", "DNAjb1 vs Random", "Random vs Random", "DNAjb1 vs Hb"]

if len(sims_DNAjb1_uniref) > 0:
    data.append(sims_DNAjb1_uniref)
    labels.append("DNAjb1 vs UniRef50")

if len(sims_uniref_uniref) > 0:
    data.append(sims_uniref_uniref)
    labels.append("UniRef50 vs UniRef50")

box_plot = plt.boxplot(data, labels=labels, showfliers=False)
plt.ylabel("Segment-wise cosine similarity (mean)")
plt.title("DNAjb1 similarity baselines (segment-based pooling, layer 28)")
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./baselines.png", dpi = 300)
plt.show()

# ------------------------
# HISTOGRAM DISTRIBUZIONI (MODIFICATO)
# ------------------------
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.hist(sims_DNAjb1_cons, bins=30, alpha=0.7, label="DNAjb1 vs Cons", color='green', density=True)
plt.hist(sims_DNAjb1_random, bins=30, alpha=0.7, label="DNAjb1 vs Random", color='orange', density=True)
plt.hist(sims_random_random, bins=30, alpha=0.7, label="Random vs Random", color='blue', density=True)
if len(sims_DNAjb1_uniref) > 0:
    plt.hist(sims_DNAjb1_uniref, bins=30, alpha=0.7, label="DNAjb1 vs UniRef50", color='purple', density=True)
plt.axvline(sim_DNAjb1_hb, color='red', linestyle='--', label=f"DNAjb1 vs Hb: {sim_DNAjb1_hb:.3f}")
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.legend()
plt.title("Distribuzioni simili")

plt.subplot(1, 4, 2)
plt.hist(sims_random_random, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(np.mean(sims_random_random), color='red', linestyle='--', 
           label=f'Mean: {np.mean(sims_random_random):.3f}')
plt.axvline(np.percentile(sims_random_random, 95), color='orange', linestyle='--', 
           label='95th percentile')
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.legend()
plt.title("Random vs Random")

plt.subplot(1, 4, 3)
if len(sims_DNAjb1_uniref) > 0:
    plt.hist(sims_DNAjb1_uniref, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(np.mean(sims_DNAjb1_uniref), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sims_DNAjb1_uniref):.3f}')
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("DNAjb1 vs UniRef50")

plt.subplot(1, 4, 4)
if len(sims_uniref_uniref) > 0:
    plt.hist(sims_uniref_uniref, bins=50, alpha=0.7, color='brown', edgecolor='black')
    plt.axvline(np.mean(sims_uniref_uniref), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sims_uniref_uniref):.3f}')
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("UniRef50 vs UniRef50")

plt.tight_layout()
plt.savefig("./comparison.png", dpi = 300)
plt.show()