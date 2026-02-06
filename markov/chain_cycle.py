import os
from datetime import datetime
import gc
# Configuration
import torch
import esm
import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import cosine
from tqdm import tqdm
from Bio.Align import substitution_matrices
import joblib
import random
import math
from esm import pretrained
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

K_PROPOSALS = 8     # numero di mutazioni per step 
TOP_M = 4         # scegli tra le migliori M
# 8,4 per fisso 3060; 4,2 per laptop 3070; non obbligatorio seguire
# 3070 laptop: 100 step in circa 15 secondi con k = 4 e m = 2
# 3060 desktop: 100 step in circa 26 secondi con k = 4 e m = 2
# 3060 desktop: 100 step in circa 48 secondi con k = 8 e m = 4



# ------------------------
# BLOSUM62
# ------------------------
blosum = substitution_matrices.load("BLOSUM62")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
T_blosum = 1.7  # temperatura alta = esplorazione ampia

PCA_PATH = "./../pca/joblibs/Total_ipca_fitted.joblib"
pca = joblib.load(PCA_PATH)
# Converti PCA su GPU
pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)  # [d_pca, D]
pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)             # [D]
d_pca = pca.n_components

print("PCA loaded")
print("n_components:", pca.n_components)

model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

# DEFINITIONS

# ------------------------
# FUNZIONI SEQUENZA
# ------------------------
def random_sequence(length):
    return ''.join(random.choice(AA_LIST) for _ in range(length))

def mutate_residue(seq, T=T_blosum):
    seq = list(seq)
    idx = random.randrange(len(seq))
    original = seq[idx]

    scores, choices = [], []
    for aa in AA_LIST:
        if aa == original:
            continue  # <-- ESCLUDE auto-mutazione
        key = (original, aa) if (original, aa) in blosum else (aa, original)
        if key in blosum:
            scores.append(blosum[key])
            choices.append(aa)

    # Boltzmann-style
    exps = [math.exp(s/T) for s in scores]
    total = sum(exps)
    probs = [e/total for e in exps]

    new_aa = random.choices(choices, weights=probs)[0]
    seq[idx] = new_aa
    prob = probs[choices.index(new_aa)]
    info = f"mutazione al residuo {idx+1}: {original}->{new_aa} (prob={prob:.2f})"
    return ''.join(seq), info

def insert_residue(seq):
    seq = list(seq)
    idx = random.randrange(len(seq)+1)
    aa = random.choice(AA_LIST)
    seq.insert(idx, aa)
    info = f"inserzione al residuo {idx+1}: {aa}"
    return ''.join(seq), info

def delete_residue(seq):
    if len(seq) <= 1:
        return seq, "nessuna delezione (sequenza troppo corta)"
    seq = list(seq)
    idx = random.randrange(len(seq))
    aa = seq[idx]
    del seq[idx]
    info = f"delezione al residuo {idx+1}: {aa}"
    return ''.join(seq), info

def markov_step(seq, p_mut=0.98, p_ins=0.01, p_del=0.01, T=T_blosum):
    r = random.random()
    if r < p_mut:
        return mutate_residue(seq, T=T)
    elif r < p_mut + p_ins:
        return insert_residue(seq)
    else:
        return delete_residue(seq)

def generate_proposals(seq_current, K):
    seqs = []
    infos = []
    for _ in range(K):
        s, info = markov_step(seq_current)
        seqs.append(s)
        infos.append(info)
    return seqs, infos


def segment_pooling_vectorized(reps, n_segments=24):
    """
    reps: torch.Tensor [L, D], su CPU o GPU
    return: torch.Tensor [n_segments, D]
    """
    L, D = reps.shape
    edges = torch.linspace(0, L, n_segments + 1, device=reps.device).round().long()
    
    segment_ids = torch.arange(L, device=reps.device).unsqueeze(0)
    start = edges[:-1].unsqueeze(1)
    end = edges[1:].unsqueeze(1)
    
    mask = (segment_ids >= start) & (segment_ids < end)
    
    # segmenti vuoti → almeno ultimo residuo
    mask_empty = mask.sum(dim=1) == 0
    mask[mask_empty, -1] = True
    
    # forza stesso dtype di reps
    mask = mask.to(reps.dtype)
    
    pooled = (mask @ reps) / mask.sum(dim=1, keepdim=True)
    return pooled


# ------------------------
# FUNZIONE EMBEDDING OTTIMIZZATA
# ------------------------
@torch.no_grad()
def get_sequence_embeddings_batch(sequences, layer=28, n_segments=24):
    """
    sequences: list[str]
    return: np.array [N, n_segments * d_pca]

    PCA e pooling completamente su GPU.
    """
    N = len(sequences)
    
    # --- converti in token per ESM ---
    data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    # --- forward nel modello ---
    out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer][:, 1:-1]  # [N, L, D]
    L_max, D = reps.shape[1], reps.shape[2]

    # --- PCA su GPU ---
    reps_pca = (reps - pca_mean) @ pca_components.T  # [N, L, d_pca], su GPU

    # --- segment pooling vettorizzato ---
    final_embeddings = []
    for i in range(N):
        pooled = segment_pooling_vectorized(reps_pca[i], n_segments=n_segments)  # [n_segments, d_pca]
        final_embeddings.append(pooled.flatten())

    final_embeddings = torch.stack(final_embeddings).cpu().numpy()  # [N, n_segments*d_pca]
    return final_embeddings



# --- Funzione per cosine similarity ---
def cosine_similarity(vec1, vec2):
    """
    Calcola la similarità coseno tra due vettori
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

beta_orig = 800
beta = beta_orig
n_steps = 7500
threshold = 0.9
lookback = 80   # numero di step per controllare se siamo bloccati
stuck_threshold = 0  # delta medio <=0 significa catena bloccata


while True:
    # --- Genera timestamp per file ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = f"./runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    output_file = f"./runs/{timestamp}/sequences_over_0.9_{timestamp}.txt"
    plot_file = f"./runs/{timestamp}/similarity_plot_{timestamp}.png"

    # --- Sequenza iniziale ---
    seq_target = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"
    layer = 28

    # --- Embedding target ---
    emb_target = get_sequence_embeddings_batch([seq_target], layer=layer)[0]

    # --- Random start ---
    seq_current = random_sequence(len(seq_target))
    emb_current = get_sequence_embeddings_batch([seq_current], layer=layer)[0]
    sim_current = cosine_similarity(emb_current, emb_target)

    sim_history = [sim_current]
    accepted_history = [True]
    step_info = ["iniziale"]

    beta = beta_orig
    accepted_counter = 0
    not_accepted_counter = 0
    counter_delta_positivo = 0

    print(f"\n=== Nuovo ciclo - timestamp {timestamp} ===")
    print("Random start:", seq_current)
    print(f"Cosine similarity iniziale: {sim_current:.4f}")
    print("-"*50)

    # --- CATENA DI MARKOV ---
    for step in range(n_steps):
        seq_candidates, infos = generate_proposals(seq_current, K_PROPOSALS)
        emb_candidates = get_sequence_embeddings_batch(seq_candidates, layer=layer)

        sims = np.array([cosine_similarity(emb_candidates[i], emb_target) for i in range(K_PROPOSALS)])
        order = np.argsort(sims)[::-1]
        top_idx = order[:TOP_M]

        positive_idx = [i for i in top_idx if sims[i] > sim_current]
        if positive_idx:
            chosen = random.choice(positive_idx)
        else:
            chosen = random.choice(top_idx)

        seq_next = seq_candidates[chosen]
        emb_next = emb_candidates[chosen]
        sim_next = sims[chosen]
        info = infos[chosen]
        delta = sim_next - sim_current

        if step >= lookback:
            recent_deltas = np.diff(sim_history[-lookback:])
            mean_delta = recent_deltas.mean()
            beta = beta_orig if mean_delta > stuck_threshold else beta_orig / 8

        if delta >= 0:
            accept = True
            accept_prob = 1.0
            counter_delta_positivo += 1
        else:
            x = max(beta * delta, -700)
            accept_prob = math.exp(x)
            accept = random.random() < accept_prob

        if accept:
            seq_current = seq_next
            emb_current = emb_next
            sim_current = sim_next
            accepted_counter += 1

            if sim_current > threshold:
                with open(output_file, "a") as f_out:
                    f_out.write(f">step={step+1} cosine_to_tonb={sim_current:.5f} length={len(seq_current)}\n")
                    f_out.write(seq_current + "\n")
        else:
            not_accepted_counter += 1

        sim_history.append(sim_current)
        accepted_history.append(accept)
        step_info.append(info)

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: {info}")
            print(f"Sim_current = {sim_current:.4f}, Sim_next = {sim_next:.4f}, P_accept = {accept_prob:.3f}, beta={beta}")
            print(f"→ Accettate: {accepted_counter}, Rifiutate: {not_accepted_counter},  Accettate con delta negativo: {accepted_counter - counter_delta_positivo}")
            print("-"*50)

    # --- Salva grafico ---
    steps = list(range(1, len(sim_history)+1))
    plt.figure(figsize=(12,6))
    plt.plot(steps, sim_history, marker='o', linestyle='-', color='blue', label='Cosine similarity')

    for i, accepted in enumerate(accepted_history):
        color = 'green' if accepted else 'red'
        plt.scatter(steps[i], sim_history[i], color=color, s=100, edgecolors='k', zorder=5)

    plt.xlabel("Step della Markov chain")
    plt.ylabel("Cosine similarity con target")
    plt.title(f"Evoluzione similitudine - timestamp {timestamp}")
    plt.grid(True)
    plt.legend(["Cosine similarity", "Accettata (verde) / Rifiutata (rosso)"])
    plt.savefig(plot_file)
    plt.close()

    print(f"Ciclo completato. File salvati:\n- {output_file}\n- {plot_file}")
    print("="*60)


    gc.collect()
    torch.cuda.empty_cache()