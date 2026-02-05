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

K_PROPOSALS = 4     # numero di mutazioni per step
TOP_M = 2         # scegli tra le migliori M


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
n_steps = 12500
threshold = 0.9
# ----------------------------------------------------------------

output_file = "./sequences_over_0.9.txt"
f_out = open(output_file, "w")

# ------------------------
# PARAMETRI SIMULAZIONE METROPOLIS
# ------------------------
seq_target = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"  # esempio
layer = 28 # attento qui 



# target embedding (una sola volta)
emb_target = get_sequence_embeddings_batch([seq_target], layer=layer)[0]

N_chains = 8  # numero di catene indipendenti
seq_currents = [random_sequence(len(seq_target)) for _ in range(N_chains)]
emb_currents = get_sequence_embeddings_batch(seq_currents, layer=layer)
sim_currents = np.array([cosine_similarity(emb_currents[i], emb_target) for i in range(N_chains)])

sim_histories = [[sim] for sim in sim_currents]
accepted_histories = [[True] for _ in range(N_chains)]
step_infos = [["iniziale"] for _ in range(N_chains)]

print("Random start:", seq_currents)
print("Cosine similarity iniziale per catena:")
for i, sim in enumerate(sim_currents):
    print(f"  Chain {i}: {sim:.4f}")
print("-"*50)


# ------------------------
# CATENA DI MARKOV OTTIMIZZATA
# ------------------------

lookback = 80   # numero di step per controllare se siamo bloccati
stuck_threshold = 0  # delta medio <=0 significa catena bloccata


accepted_counter = 0
not_accepted_counter = 0
counter_delta_positivo = 0



for step in range(n_steps):
    # Step di Markov: proposta
    # genera K proposte per ogni catena
    seq_candidates_all = []
    infos_all = []

    for seq in seq_currents:
        seqs, infos = generate_proposals(seq, K_PROPOSALS)
        seq_candidates_all.extend(seqs)
        infos_all.extend(infos)

    # Forward batch unico
    emb_candidates_all = get_sequence_embeddings_batch(seq_candidates_all, layer=layer)

    # --- controllo beta adattivo ---
    if step >= lookback:
        recent_deltas = []
        for chain_idx in range(N_chains):
            recent = np.diff(sim_histories[chain_idx][-lookback:])
            recent_deltas.extend(recent)
        mean_delta = np.mean(recent_deltas)
        if mean_delta <= stuck_threshold:
            beta = beta_orig / 8
        else:
            beta = beta_orig


    for chain_idx in range(N_chains):
        start = chain_idx * K_PROPOSALS
        end = start + K_PROPOSALS

        seq_candidates = seq_candidates_all[start:end]
        emb_candidates = emb_candidates_all[start:end]
        infos = infos_all[start:end]

        sim_candidates = np.array([
            cosine_similarity(emb_candidates[i], emb_target)
            for i in range(K_PROPOSALS)
        ])
        
        sim_current = sim_currents[chain_idx]

        order = np.argsort(sim_candidates)[::-1]
        top_idx = order[:TOP_M]

        positive_idx = [i for i in top_idx if sim_candidates[i] > sim_current]

        if positive_idx:
            chosen = random.choice(positive_idx)
        else:
            chosen = random.choice(top_idx)

        seq_next = seq_candidates[chosen]
        emb_next = emb_candidates[chosen]
        sim_next = sim_candidates[chosen]
        info = infos[chosen]

        delta = sim_next - sim_current

        # --- Metropolis ---
        if delta >= 0:
            accept = True
            accept_prob = 1.0
        else:
            x = max(beta * delta, -700)
            accept_prob = math.exp(x)
            accept = random.random() < accept_prob

        if accept:
            seq_currents[chain_idx] = seq_next
            emb_currents[chain_idx] = emb_next
            sim_currents[chain_idx] = sim_next
            accepted_counter += 1
            if delta > 0:
                counter_delta_positivo += 1
        else:
            not_accepted_counter += 1


            if sim_next > threshold:
                with open(output_file, "a") as f_out:
                    f_out.write(
                        f">chain={chain_idx} step={step+1} cosine_to_target={sim_next:.5f} length={len(seq_next)}\n"
                    )
                    f_out.write(seq_next + "\n")

        sim_histories[chain_idx].append(sim_currents[chain_idx])
        accepted_histories[chain_idx].append(accept)
        step_infos[chain_idx].append(info)

    if (step + 1) % 100 == 0:
        print(f"Step {step+1}: {info}")
        print(f"Sim_current = {sim_current:.4f}, Sim_next = {sim_next:.4f}, P_accept = {accept_prob:.3f}, beta={beta}")
        print(f"→ Accettate: {accepted_counter}, Rifiutate: {not_accepted_counter},  Accettate con delta negativo: {accepted_counter - counter_delta_positivo}")
        print("-"*50)




steps = list(range(n_steps + 1))  # +1 per includere lo step iniziale

f_out.close()


plt.figure(figsize=(12,6))
for chain_idx in range(N_chains):
    plt.plot(steps, sim_histories[chain_idx], marker='o', linestyle='-', label=f'Chain {chain_idx}')

for chain_idx in range(N_chains):
    for step_idx, accepted in enumerate(accepted_histories[chain_idx]):
        color = 'green' if accepted else 'red'
        plt.scatter(steps[step_idx], sim_histories[chain_idx][step_idx], color=color, s=50, edgecolors='k', zorder=5)


plt.xlabel("Step della Markov chain")
plt.ylabel("Cosine similarity con target")
plt.title("Evoluzione della similitudine nella catena di Markov")
plt.grid(True)
plt.legend(["Cosine similarity", "Accettata (verde) / Rifiutata (rosso)"])
plt.show()

print(accepted_counter)
print(counter_delta_positivo)
print(not_accepted_counter)
den = (accepted_counter - counter_delta_positivo) + not_accepted_counter
if den > 0:
    print((accepted_counter - counter_delta_positivo) / den)
else:
    print("No negative moves attempted")
