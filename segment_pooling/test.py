import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from esm import pretrained
from tqdm import tqdm
from Bio.Align import substitution_matrices
from Bio import SeqIO  # per leggere FASTA UniRef50
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
n_uniref = 200  # numero proteine UniRef50 da campionare

# CAMBIA QUESTO PATH con il tuo file FASTA UniRef50
uniref_fasta_path = "../pca/datasets/uniref50_subsample.fasta"  # <-- ADATTA IL PATH!

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# ------------------------
# SEQUENZE
# ------------------------
tonb_seq = (
    "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"
)

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
    sims_tonb_uniref = []

# ------------------------
# EMBEDDING BASE
# ------------------------
print("Calcolo embedding base...")

print("Calcolo embedding TonB...")
print(device)
tonb_res = get_residue_embeddings(tonb_seq)
tonb_seg = split_into_segments(tonb_res, n_segments)

print("Calcolo embedding Hb...")
hb_res = get_residue_embeddings(seq_hb)
hb_seg = split_into_segments(hb_res, n_segments)

# ------------------------
# STATISTICHE
# ------------------------
sims_tonb_tonb = []
sims_tonb_random = []
sims_tonb_cons = []
sims_tonb_uniref = []
sims_random_random = []

# TonB vs TonB (identico, per controllo numerico)
sims_tonb_tonb.append(global_similarity(tonb_seg, tonb_seg))

# TonB vs Hb
sim_tonb_hb = global_similarity(tonb_seg, hb_seg)

# TonB vs random
print("TonB vs random...")
for _ in tqdm(range(n_random), desc="TonB vs random"):
    rseq = random_sequence(len(tonb_seq))
    rseg = split_into_segments(get_residue_embeddings(rseq), n_segments)
    sims_tonb_random.append(global_similarity(tonb_seg, rseg))

# TonB vs mutazioni conservative
print("TonB vs conservative...")
for _ in tqdm(range(n_conservative), desc="TonB vs conservative"):
    mut = conservative_mutation_blosum(tonb_seq, min_score=2)
    mut_seg = split_into_segments(get_residue_embeddings(mut), n_segments)
    sims_tonb_cons.append(global_similarity(tonb_seg, mut_seg))

# TonB vs UniRef50
if n_uniref > 0:
    print("TonB vs UniRef50...")
    for seq_str in tqdm(uniref_sequences[:n_uniref], desc="TonB vs UniRef50"):
        uref_seg = split_into_segments(get_residue_embeddings(seq_str), n_segments)
        sims_tonb_uniref.append(global_similarity(tonb_seg, uref_seg))
else:
    sims_tonb_uniref = []

# Random vs random (baseline) - sequenze casuali
print("Random vs random baseline (sequenze casuali)...")
random_seqs = []
for i in tqdm(range(n_random_baseline), desc="Generate random seqs"):
    rseq = random_sequence(len(tonb_seq) + random.randint(-50, 50))
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
print("TonB vs TonB:                           %.4f" % sims_tonb_tonb[0])
print("TonB vs Hb:                             %.4f" % sim_tonb_hb)
print("TonB vs Random: mean = %.4f ± %.4f" % (np.mean(sims_tonb_random), np.std(sims_tonb_random)))
print("TonB vs Conservative: mean = %.4f ± %.4f" % (np.mean(sims_tonb_cons), np.std(sims_tonb_cons)))
if len(sims_tonb_uniref) > 0:
    print("TonB vs UniRef50 (n=%d): mean = %.4f ± %.4f" % (len(sims_tonb_uniref), np.mean(sims_tonb_uniref), np.std(sims_tonb_uniref)))
print("Random vs Random (casuali): mean = %.4f ± %.4f" % (np.mean(sims_random_random), np.std(sims_random_random)))
if len(sims_uniref_uniref) > 0:
    print("UniRef50 vs UniRef50 (n=%d): mean = %.4f ± %.4f" % (len(sims_uniref_uniref), np.mean(sims_uniref_uniref), np.std(sims_uniref_uniref)))

print("\n=== DELTE vs BASELINE RandomRandom ===")
print("TonB(Cons) - RandomRandom mean: %.4f" % (np.mean(sims_tonb_cons) - np.mean(sims_random_random)))
print("TonB(Hb)   - RandomRandom mean: %.4f" % (sim_tonb_hb - np.mean(sims_random_random)))
print("TonB(Rand) - RandomRandom mean: %.4f" % (np.mean(sims_tonb_random) - np.mean(sims_random_random)))
if len(sims_tonb_uniref) > 0:
    print("TonB(UniRef) - RandomRandom mean: %.4f" % (np.mean(sims_tonb_uniref) - np.mean(sims_random_random)))
if len(sims_uniref_uniref) > 0:
    print("UniRef vs UniRef - RandomRandom mean: %.4f" % (np.mean(sims_uniref_uniref) - np.mean(sims_random_random)))

# ------------------------
# BOXPLOT (MODIFICATO)
# ------------------------
plt.figure(figsize=(16, 6))

data = [sims_tonb_tonb, sims_tonb_cons, sims_tonb_random, sims_random_random, [sim_tonb_hb]]
labels = ["TonB vs TonB", "TonB vs Conservative", "TonB vs Random", "Random vs Random", "TonB vs Hb"]

if len(sims_tonb_uniref) > 0:
    data.append(sims_tonb_uniref)
    labels.append("TonB vs UniRef50")

if len(sims_uniref_uniref) > 0:
    data.append(sims_uniref_uniref)
    labels.append("UniRef50 vs UniRef50")

box_plot = plt.boxplot(data, labels=labels, showfliers=False)
plt.ylabel("Segment-wise cosine similarity (mean)")
plt.title("TonB similarity baselines (segment-based pooling, layer 28)")
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
plt.hist(sims_tonb_cons, bins=30, alpha=0.7, label="TonB vs Cons", color='green', density=True)
plt.hist(sims_tonb_random, bins=30, alpha=0.7, label="TonB vs Random", color='orange', density=True)
plt.hist(sims_random_random, bins=30, alpha=0.7, label="Random vs Random", color='blue', density=True)
if len(sims_tonb_uniref) > 0:
    plt.hist(sims_tonb_uniref, bins=30, alpha=0.7, label="TonB vs UniRef50", color='purple', density=True)
plt.axvline(sim_tonb_hb, color='red', linestyle='--', label=f"TonB vs Hb: {sim_tonb_hb:.3f}")
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
if len(sims_tonb_uniref) > 0:
    plt.hist(sims_tonb_uniref, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(np.mean(sims_tonb_uniref), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sims_tonb_uniref):.3f}')
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("TonB vs UniRef50")

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