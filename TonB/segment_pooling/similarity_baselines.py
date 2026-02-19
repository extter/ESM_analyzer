import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from Bio.Align import substitution_matrices
from Bio import SeqIO
from esm import pretrained
from tqdm import tqdm
from typing import List, Tuple
from itertools import combinations

# ------------------------
# CONFIGURAZIONE E COSTANTI
# ------------------------
LAYER = 28
N_SEGMENTS = 12
N_RANDOM = 200
N_CONSERVATIVE = 200
N_RANDOM_BASELINE = 100
N_UNIREF = 200 
MAX_UNIREF_PAIRS = 500

TONB_SEQ_PATH = "../../sequences/tonb.txt"
UNIREF_FASTA_PATH = "../../pca/datasets/uniref50_subsample.fasta"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
BLOSUM62 = substitution_matrices.load("BLOSUM62")

SEQ_HB = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVA"
    "HVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)

# ------------------------
# INIZIALIZZAZIONE MODELLO
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inizializzazione ESM-2 su {device}...")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
torch.set_grad_enabled(False)
batch_converter = alphabet.get_batch_converter()

# ------------------------
# FUNZIONI CORE
# ------------------------
@torch.no_grad()
def get_residue_embeddings(seq: str) -> np.ndarray:
    """Estrae gli embedding dal layer scelto escludendo i token speciali <cls> e <eos>."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[LAYER], return_contacts=False)
    reps = out["representations"][LAYER][0, 1:-1]
    return reps.cpu().numpy()

def split_into_segments(residue_embs: np.ndarray, n_segments: int) -> np.ndarray:
    """Divide gli embedding in segmenti, calcola la media e li normalizza (L2)."""
    L, D = residue_embs.shape
    boundaries = np.linspace(0, L, n_segments + 1).astype(int)

    segments = np.zeros((n_segments, D))
    for i in range(n_segments):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            seg = residue_embs[start:end].mean(axis=0)
            norm = np.linalg.norm(seg)
            seg /= norm + 1e-8 if norm > 0 else 1
            segments[i] = seg
    return segments

def global_similarity(seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Calcola la media della cosine similarity tra segmenti corrispondenti."""
    return float(np.sum(seg_a * seg_b, axis=1).mean())

# ------------------------
# MUTAZIONI E DATASET
# ------------------------
def random_sequence(length: int) -> str:
    """Genera una sequenza casuale della lunghezza specificata."""
    return ''.join(random.choice(AA_LIST) for _ in range(length))

def conservative_mutation_blosum(seq: str, min_score: int = 2) -> str:
    """Genera una mutazione conservativa basata sulla matrice BLOSUM62."""
    seq_list = list(seq)
    pos = random.randrange(len(seq_list))
    aa = seq_list[pos]

    candidates = [
        aa2 for aa2 in AA_LIST 
        if aa2 != aa and BLOSUM62.get((aa, aa2), BLOSUM62.get((aa2, aa), -10)) > min_score
    ]

    if not candidates:
        return ''.join(seq_list)

    seq_list[pos] = random.choice(candidates)
    return ''.join(seq_list)

# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    
    # 1. Caricamento Sequenze
    with open(TONB_SEQ_PATH, "r") as f:
        tonb_seq = f.read().strip()
    print(f"TonB caricata (lunghezza: {len(tonb_seq)})")

    uniref_sequences: List[str] = []
    try:
        with open(UNIREF_FASTA_PATH, "r") as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta"), start=1):
                seq_str = str(record.seq)
                if not (150 <= len(seq_str) <= 700):
                    continue
                if len(uniref_sequences) < N_UNIREF:
                    uniref_sequences.append(seq_str)
                else:
                    j = random.randint(1, i)
                    if j <= N_UNIREF:
                        uniref_sequences[j - 1] = seq_str
        print(f"Caricate {len(uniref_sequences)} sequenze UniRef50")
    except FileNotFoundError:
        print(f"⚠️ File {UNIREF_FASTA_PATH} non trovato! Salto UniRef50.")

    # 2. Embedding Base
    print("\nCalcolo embedding base...")
    tonb_seg = split_into_segments(get_residue_embeddings(tonb_seq), N_SEGMENTS)
    hb_seg = split_into_segments(get_residue_embeddings(SEQ_HB), N_SEGMENTS)

    # 3. Statistiche Base (TonB)
    sims_tonb_tonb = [global_similarity(tonb_seg, tonb_seg)]
    sim_tonb_hb = global_similarity(tonb_seg, hb_seg)

    print("Calcolo TonB vs Random...")
    sims_tonb_random = [
        global_similarity(tonb_seg, split_into_segments(get_residue_embeddings(random_sequence(len(tonb_seq))), N_SEGMENTS))
        for _ in tqdm(range(N_RANDOM))
    ]

    print("Calcolo TonB vs Conservative...")
    sims_tonb_cons = [
        global_similarity(tonb_seg, split_into_segments(get_residue_embeddings(conservative_mutation_blosum(tonb_seq)), N_SEGMENTS))
        for _ in tqdm(range(N_CONSERVATIVE))
    ]

    sims_tonb_uniref = []
    if uniref_sequences:
        print("Calcolo TonB vs UniRef50...")
        sims_tonb_uniref = [
            global_similarity(tonb_seg, split_into_segments(get_residue_embeddings(seq), N_SEGMENTS))
            for seq in tqdm(uniref_sequences[:N_UNIREF])
        ]

    # 4. Statistiche Random Baseline
    print("\nCalcolo Random vs Random baseline...")
    random_seqs = [random_sequence(len(tonb_seq) + random.randint(-50, 50)) for _ in range(N_RANDOM_BASELINE)]
    all_random_segs = [split_into_segments(get_residue_embeddings(rseq), N_SEGMENTS) for rseq in tqdm(random_seqs, desc="Random embeddings")]
    
    sims_random_random = [global_similarity(a, b) for a, b in combinations(all_random_segs, 2)]

    # 5. UniRef50 vs UniRef50
    sims_uniref_uniref = []
    if len(uniref_sequences) >= 20:
        print("\nCalcolo UniRef50 vs UniRef50...")
        all_uniref_segs = [
            split_into_segments(get_residue_embeddings(seq), N_SEGMENTS) 
            for seq in tqdm(uniref_sequences[:50], desc="UniRef embeddings")
        ]
        
        pairs = list(combinations(all_uniref_segs, 2))[:MAX_UNIREF_PAIRS]
        sims_uniref_uniref = [global_similarity(a, b) for a, b in tqdm(pairs, desc="UniRef vs UniRef")]

    # 6. Print dei Risultati (f-strings)
    print("\n" + "="*20 + " RISULTATI " + "="*20)
    print(f"TonB vs TonB:                           {sims_tonb_tonb[0]:.4f}")
    print(f"TonB vs Hb:                             {sim_tonb_hb:.4f}")
    print(f"TonB vs Random: mean = {np.mean(sims_tonb_random):.4f} ± {np.std(sims_tonb_random):.4f}")
    print(f"TonB vs Conservative: mean = {np.mean(sims_tonb_cons):.4f} ± {np.std(sims_tonb_cons):.4f}")
    
    if sims_tonb_uniref:
        print(f"TonB vs UniRef50 (n={len(sims_tonb_uniref)}): mean = {np.mean(sims_tonb_uniref):.4f} ± {np.std(sims_tonb_uniref):.4f}")
    
    print(f"Random vs Random (casuali): mean = {np.mean(sims_random_random):.4f} ± {np.std(sims_random_random):.4f}")
    
    if sims_uniref_uniref:
        print(f"UniRef50 vs UniRef50 (n={len(sims_uniref_uniref)}): mean = {np.mean(sims_uniref_uniref):.4f} ± {np.std(sims_uniref_uniref):.4f}")

    print("\n" + "="*15 + " DELTA vs BASELINE (Random vs Random) " + "="*15)
    baseline_mean = np.mean(sims_random_random)
    print(f"TonB(Cons) - Baseline:   {np.mean(sims_tonb_cons) - baseline_mean:.4f}")
    print(f"TonB(Hb)   - Baseline:   {sim_tonb_hb - baseline_mean:.4f}")
    print(f"TonB(Rand) - Baseline:   {np.mean(sims_tonb_random) - baseline_mean:.4f}")
    if sims_tonb_uniref:
        print(f"TonB(UniRef) - Baseline: {np.mean(sims_tonb_uniref) - baseline_mean:.4f}")
    if sims_uniref_uniref:
        print(f"UniRef vs UniRef - Base: {np.mean(sims_uniref_uniref) - baseline_mean:.4f}")

    # 7. Generazione Grafici
    # --- Boxplot ---
    plt.figure(figsize=(16, 6))
    data = [sims_tonb_tonb, sims_tonb_cons, sims_tonb_random, sims_random_random, [sim_tonb_hb]]
    labels = ["TonB vs TonB", "TonB vs Conservative", "TonB vs Random", "Random vs Random", "TonB vs Hb"]

    if sims_tonb_uniref:
        data.append(sims_tonb_uniref)
        labels.append("TonB vs UniRef50")
    if sims_uniref_uniref:
        data.append(sims_uniref_uniref)
        labels.append("UniRef50 vs UniRef50")

    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(f"Segment-wise cosine similarity (mean, n={N_SEGMENTS})")
    plt.title(f"TonB similarity baselines (Layer {LAYER})")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./baselines.png", dpi=300)
    plt.close()

    # --- Istogrammi ---
    
    plt.figure(figsize=(18, 5))

    # Subplot 1: Distribuzioni Sovrapposte
    plt.subplot(1, 4, 1)
    plt.hist(sims_tonb_cons, bins=30, alpha=0.7, label="TonB vs Cons", color='green', density=True)
    plt.hist(sims_tonb_random, bins=30, alpha=0.7, label="TonB vs Random", color='orange', density=True)
    plt.hist(sims_random_random, bins=30, alpha=0.7, label="Random vs Random", color='blue', density=True)
    if sims_tonb_uniref:
        plt.hist(sims_tonb_uniref, bins=30, alpha=0.7, label="TonB vs UniRef50", color='purple', density=True)
    plt.axvline(sim_tonb_hb, color='red', linestyle='--', label=f"TonB vs Hb: {sim_tonb_hb:.3f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribuzioni simili")

    # Subplot 2: Random vs Random
    plt.subplot(1, 4, 2)
    plt.hist(sims_random_random, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(np.mean(sims_random_random), color='red', linestyle='--', label=f'Mean: {np.mean(sims_random_random):.3f}')
    plt.axvline(np.percentile(sims_random_random, 95), color='orange', linestyle='--', label='95th percentile')
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Random vs Random")

    # Subplot 3: TonB vs UniRef50
    plt.subplot(1, 4, 3)
    if sims_tonb_uniref:
        plt.hist(sims_tonb_uniref, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(np.mean(sims_tonb_uniref), color='red', linestyle='--', label=f'Mean: {np.mean(sims_tonb_uniref):.3f}')
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.legend()
        plt.title("TonB vs UniRef50")

    # Subplot 4: UniRef50 vs UniRef50
    plt.subplot(1, 4, 4)
    if sims_uniref_uniref:
        plt.hist(sims_uniref_uniref, bins=50, alpha=0.7, color='brown', edgecolor='black')
        plt.axvline(np.mean(sims_uniref_uniref), color='red', linestyle='--', label=f'Mean: {np.mean(sims_uniref_uniref):.3f}')
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.legend()
        plt.title("UniRef50 vs UniRef50")

    plt.tight_layout()
    plt.savefig("./comparison.png", dpi=300)
    plt.close()
    
    print("\nGrafici salvati in ./baselines.png e ./comparison.png")