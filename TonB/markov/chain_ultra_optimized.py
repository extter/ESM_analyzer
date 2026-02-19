import os
import gc
import torch
import numpy as np
import random
import math
import joblib
import re
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from Bio.Align import substitution_matrices
from esm import pretrained

# -----------------------------------------
# 1. CONFIGURAZIONE E IPERPARAMETRI
# -----------------------------------------

# Target: TonB Sequence
SEQ_TARGET = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

# Percorsi
PCA_PATH = "./../pca/joblibs/Total_ipca_fitted.joblib"
INPUT_SEEDS_FILE = "./best_sequences_from_runs.txt"
RUNS_DIR = "./runs_ultra_optimized"

# Parametri ESM / PCA
LAYER = 28
N_SEGMENTS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parametri Evoluzione (FINE TUNING)
N_STEPS = 6000 
K_PROPOSALS = 32     
BETA_START = 5000       
BETA_MAX = 20000

# Soglie
THRESHOLD_RECORD = 0.992 

# -----------------------------------------
# 2. SETUP MODELLI E MATRICI
# -----------------------------------------

print(f"Device: {DEVICE}")

# Caricamento BLOSUM62
try:
    blosum = substitution_matrices.load("BLOSUM62")
except:
    from Bio.Align import substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")
    
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Caricamento PCA
if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(f"Errore: Non trovo il file PCA in {PCA_PATH}")

print("Caricamento PCA...")
pca_obj = joblib.load(PCA_PATH)
pca_components = torch.tensor(pca_obj.components_, dtype=torch.float32, device=DEVICE)
pca_mean = torch.tensor(pca_obj.mean_, dtype=torch.float32, device=DEVICE)
d_pca = pca_obj.n_components

print("Caricamento ESM-2 (t33_650M) in FP16...")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)
model = model.half() 
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------------------
# 3. FUNZIONI DI UTILITÀ (EMBEDDING, MUTAZIONE, SELEZIONE)
# -----------------------------------------

def segment_pooling_vectorized(reps, n_segments=24):
    """Pooling dei vettori in n segmenti fissi."""
    L, D = reps.shape
    edges = torch.linspace(0, L, n_segments + 1, device=reps.device).round().long()
    segment_ids = torch.arange(L, device=reps.device).unsqueeze(0)
    start = edges[:-1].unsqueeze(1)
    end = edges[1:].unsqueeze(1)
    mask = (segment_ids >= start) & (segment_ids < end)
    
    mask_empty = mask.sum(dim=1) == 0
    mask[mask_empty, -1] = True
    mask = mask.to(reps.dtype)
    
    pooled = (mask @ reps) / mask.sum(dim=1, keepdim=True)
    return pooled.flatten()

@torch.no_grad()
def get_sequence_embeddings_batch(sequences, layer=28, n_segments=24):
    """
    Calcola embedding PCA+Pooling ottimizzato.
    Usa FP16 per il modello e FP32 per la PCA.
    """
    # Tokenizzazione
    data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.amp.autocast('cuda'):
        out = model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out["representations"][layer][:, 1:-1] 

    reps = reps.to(torch.float32)

    
    reps_pca = (reps - pca_mean) @ pca_components.T

   
    L, D = reps_pca.shape[1], reps_pca.shape[2]
    
    
    edges = torch.linspace(0, L, n_segments + 1, device=DEVICE).round().long()
    seg_ids = torch.arange(L, device=DEVICE).unsqueeze(0)
    start = edges[:-1].unsqueeze(1)
    end = edges[1:].unsqueeze(1)
    
    
    mask = (seg_ids >= start) & (seg_ids < end)
    mask = mask.to(torch.float32) 
    
    pooled = torch.einsum('bld, sl -> bsd', reps_pca, mask)
    
    counts = mask.sum(dim=1)
    counts[counts==0] = 1 
    
    
    pooled = pooled / counts.view(1, -1, 1)

    return pooled.reshape(len(sequences), -1).cpu().numpy()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def mutate_residue(seq, T=1.5):
    seq_list = list(seq)
    idx = random.randrange(len(seq_list))
    original = seq_list[idx]

    scores, choices = [], []
    for aa in AA_LIST:
        if aa == original: continue
        key = (original, aa) if (original, aa) in blosum else (aa, original)
        score = blosum.get(key, -4.0) 
        scores.append(score)
        choices.append(aa)

    # Boltzmann Weights
    exps = [math.exp(s/T) for s in scores]
    total = sum(exps)
    probs = [e/total for e in exps]

    new_aa = random.choices(choices, weights=probs)[0]
    seq_list[idx] = new_aa
    return "".join(seq_list), f"Mut {original}{idx+1}{new_aa}"

def insert_residue(seq):
    seq_list = list(seq)
    idx = random.randrange(len(seq_list)+1)
    aa = random.choice(AA_LIST)
    seq_list.insert(idx, aa)
    return "".join(seq_list), f"Ins {aa}@{idx+1}"

def delete_residue(seq):
    if len(seq) <= 10: return seq, "NoDel"
    seq_list = list(seq)
    idx = random.randrange(len(seq_list))
    removed = seq_list.pop(idx)
    return "".join(seq_list), f"Del {removed}@{idx+1}"

def markov_step_adaptive(seq, current_sim, T=1.5):
    """Sceglie la mutazione con strategia a 3 stadi."""
    r = random.random()
    
    # --- STADIO 3: ENDGAME (Solo se siamo > 0.9935) ---
    if current_sim > 0.994:
        return mutate_residue(seq, T=0.8) 
    
    # --- STADIO 2: CLIMBING (Siamo tra 0.99 e 0.9935) ---
    elif current_sim > 0.99:
        return mutate_residue(seq, T=1.0) 
    
    # --- STADIO 1: APPROACH (< 0.99) ---
    else:
        if r < 0.90:
            return mutate_residue(seq, T=T)
        elif r < 0.95:
            return insert_residue(seq)
        else:
            return delete_residue(seq)

# -----------------------------------------
# 4. CARICAMENTO SEEDS
# -----------------------------------------

def load_seeds(filepath):
    valid_seeds = []
    if not os.path.exists(filepath):
        print(f"File seeds {filepath} non trovato.")
        return []
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    current_seq = ""
    for line in lines:
        line = line.strip()
        if not line: continue
        if not line.startswith(">") and not line.startswith("="):
            if set(line).issubset(set(AA_LIST)):
                valid_seeds.append(line)
    
    # Rimuovi duplicati mantenendo l'ordine
    return list(dict.fromkeys(valid_seeds))

seeds = load_seeds(INPUT_SEEDS_FILE)
print(f"Trovati {len(seeds)} seed validi per il fine-tuning.")
if len(seeds) == 0:
    print("Nessun seed trovato. Uso la sequenza target come test.")
    seeds = [SEQ_TARGET]

emb_target = get_sequence_embeddings_batch([SEQ_TARGET], layer=LAYER)[0]

# -----------------------------------------
# 5. LOOP DI EVOLUZIONE (FINE TUNING)
# -----------------------------------------

def run_finetuning(seed_seq, seed_idx):
    timestamp = datetime.now().strftime("%H%M%S")
    run_name = f"seed{seed_idx}_{timestamp}"
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    log_path = os.path.join(out_dir, "finetune_log.txt")
    best_seqs_path = os.path.join(out_dir, "best_candidates.fasta")
    
    # Inizializzazione
    current_seq = seed_seq
    current_emb = get_sequence_embeddings_batch([current_seq], layer=LAYER)[0]
    current_sim = cosine_similarity(current_emb, emb_target)
    
    best_sim_local = current_sim
    history = [current_sim]
    
    print(f"Start {run_name} | Init Sim: {current_sim:.6f}")
    
    with open(log_path, "w") as f:
        f.write(f"Start Fine-Tuning Seed {seed_idx}\nInit Sim: {current_sim}\n")

    accepted_count = 0
    
    pbar = tqdm(range(N_STEPS), desc=f"Seed {seed_idx}")
    
    for step in pbar:
        progress = step / N_STEPS
        beta = BETA_START + (BETA_MAX - BETA_START) * progress
        
        # 1. Generazione Batch (GPU)
        candidates = []
        infos = []
        for _ in range(K_PROPOSALS):
            temp = 1.5 - (1.0 * progress) 
            s, info = markov_step_adaptive(current_seq, current_sim, T=temp)
            candidates.append(s)
            infos.append(info)
            
        # 2. Valutazione Batch
        cand_embs = get_sequence_embeddings_batch(candidates, layer=LAYER)
        cand_sims = np.array([cosine_similarity(e, emb_target) for e in cand_embs])
        
        # 3. Selezione (Greedy + Metropolis)
        best_idx_batch = np.argmax(cand_sims)
        best_cand_sim = cand_sims[best_idx_batch]
        best_cand_seq = candidates[best_idx_batch]
        best_cand_info = infos[best_idx_batch]
        
        delta = best_cand_sim - current_sim
        
        accept = False
        if delta > 0:
            accept = True 
        else:
            prob = math.exp(delta * beta)
            if random.random() < prob:
                accept = True
        
        if accept:
            current_seq = best_cand_seq
            current_emb = cand_embs[best_idx_batch]
            current_sim = best_cand_sim
            accepted_count += 1
            
            if current_sim > best_sim_local or current_sim > THRESHOLD_RECORD:
                best_sim_local = max(best_sim_local, current_sim)
                with open(best_seqs_path, "a") as f:
                    f.write(f">step_{step}_sim_{current_sim:.6f} {best_cand_info}\n")
                    f.write(f"{current_seq}\n")
        
        history.append(current_sim)
        pbar.set_postfix({"Sim": f"{current_sim:.5f}", "Best": f"{best_sim_local:.5f}", "Acc": accepted_count})

    # Plot Finale
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Cosine Similarity")
    plt.axhline(y=THRESHOLD_RECORD, color='r', linestyle='--', alpha=0.5)
    plt.title(f"Fine-Tuning Trajectory (Max: {best_sim_local:.6f})")
    plt.xlabel("Step")
    plt.ylabel("Similarity")
    plt.savefig(os.path.join(out_dir, "trajectory.png"))
    plt.close()
    
    return best_sim_local

# -----------------------------------------
# 6. ESECUZIONE
# -----------------------------------------

print("\n--- INIZIO FINE TUNING MASSIVO ---")
for i, seed in enumerate(seeds):
    try:
        max_score = run_finetuning(seed, i)
        print(f"Seed {i} completato. Max Score: {max_score:.6f}")
        
        # Pulizia memoria GPU tra un seed e l'altro
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Errore sul seed {i}: {e}")
        continue

print("\nTutte le run completate.")