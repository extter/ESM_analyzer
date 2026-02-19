import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from Bio.Align import substitution_matrices
from Bio import SeqIO
from esm import pretrained
from tqdm import tqdm
from typing import List, Dict

# ------------------------
# CONFIGURAZIONE E COSTANTI
# ------------------------
LAYER = 28
SEGMENT_LIST = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
N_RANDOM_BASELINE = 100
N_CONSERVATIVE = 200
N_UNIREF = 200

TONB_SEQ_PATH = "../../sequences/tonb.txt"
UNIREF_FASTA_PATH = "../../pca/datasets/uniref50_subsample.fasta"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
BLOSUM62 = substitution_matrices.load("BLOSUM62")

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
    reps = out["representations"][LAYER][0, 1:-1]  # Shape: [L, D]
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
    segmentwise_cosine = np.sum(seg_a * seg_b, axis=1)
    return float(segmentwise_cosine.mean())

# ------------------------
# MUTAZIONI E DATASET
# ------------------------

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

def random_sequence(length: int) -> str:
    """Genera una sequenza casuale della lunghezza specificata."""
    return ''.join(random.choice(AA_LIST) for _ in range(length))

# ------------------------
# ESECUZIONE MAIN
# ------------------------
if __name__ == "__main__":
    
    # 1. Caricamento sequenza TonB
    with open(TONB_SEQ_PATH, "r") as f:
        tonb_seq = f.read().strip()
    print(f"TonB caricata (lunghezza: {len(tonb_seq)})")

    # 2. Caricamento UniRef50
    print("Caricamento UniRef50...")
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
        print(f" File {UNIREF_FASTA_PATH} non trovato! Analisi UniRef saltata.")
        uniref_sequences = []

    # 3. Calcolo Embedding di Base
    print("Calcolo embedding UniRef50...")
    uniref_embs = [get_residue_embeddings(seq) for seq in tqdm(uniref_sequences, desc="UniRef embeddings")]

    print("Generazione e calcolo ensemble random...")
    random_seqs = [random_sequence(len(tonb_seq) + random.randint(-50, 50)) for _ in range(N_RANDOM_BASELINE)]
    all_random_segs_dict = {rseq: get_residue_embeddings(rseq) for rseq in tqdm(random_seqs, desc="Random embeddings")}

    tonb_emb_full = get_residue_embeddings(tonb_seq)

    # 4. Loop sui numeri di segmenti
    results: Dict[int, dict] = {}

    for n_seg in SEGMENT_LIST:
        print(f"\n=== Segmenti: {n_seg} ===")
        tonb_seg = split_into_segments(tonb_emb_full, n_seg)
        
        # TonB vs Random Ensemble
        ensemble_sims = [
            global_similarity(tonb_seg, split_into_segments(all_random_segs_dict[rseq], n_seg))
            for rseq in random_seqs
        ]
        
        # TonB vs Conservative
        sims_cons = []
        for _ in range(N_CONSERVATIVE):
            mut_seq = conservative_mutation_blosum(tonb_seq)
            mut_seg = split_into_segments(get_residue_embeddings(mut_seq), n_seg)
            sims_cons.append(global_similarity(tonb_seg, mut_seg))

        # TonB vs UniRef50  
        sims_uniref = [
            global_similarity(tonb_seg, split_into_segments(emb, n_seg))
            for emb in uniref_embs
        ]
        
        mean_cons = np.mean(sims_cons)
        mean_ensemble = np.mean(ensemble_sims)
        mean_uniref = np.mean(sims_uniref) if sims_uniref else 0.0
        
        print(f"TonB vs Random:   {mean_ensemble:.4f} ± {np.std(ensemble_sims):.4f}")
        if sims_uniref:
            print(f"TonB vs UniRef:   {mean_uniref:.4f} ± {np.std(sims_uniref):.4f}")
        print(f"TonB vs Cons:     {mean_cons:.4f} ± {np.std(sims_cons):.4f}")
        print(f"Δ Cons–Random:   {mean_cons - mean_ensemble:.4f}")
        
        results[n_seg] = {
            "ensemble_sims": ensemble_sims,
            "cons_sims": sims_cons,
            "uniref_sims": sims_uniref,
        }

    # 5. Generazione Grafico
    plt.figure(figsize=(18, 6))
    data_to_plot = []
    labels = []

    for n_seg in SEGMENT_LIST:
        data_to_plot.extend([
            results[n_seg]["cons_sims"], 
            results[n_seg]["ensemble_sims"]
        ])
        labels.extend([f"Cons {n_seg}", f"Random {n_seg}"])
        
        if results[n_seg]["uniref_sims"]:
            data_to_plot.append(results[n_seg]["uniref_sims"])
            labels.append(f"UniRef {n_seg}")

    plt.boxplot(data_to_plot, labels=labels, showfliers=False)
    plt.xticks(rotation=60)
    plt.ylabel("Segment-wise cosine similarity")
    plt.title("TonB: Conservative vs Random vs UniRef50")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./selection_plot.png", dpi=300)
    print("Grafico salvato in ./selection_plot.png")