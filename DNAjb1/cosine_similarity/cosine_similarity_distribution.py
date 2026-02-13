tonb_sequence = "MGKDYYQTLGLARGASDDEIKRAYRRQALRYPDKNKEPGAEEKFKEIAEAYDVLSDPRKREIFDRYGEEGLKGGGPSGGSSGGANGTSFSYTFGDPAMFAEFFGGRNP"

import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from esm import pretrained
from tqdm import tqdm
from scipy.spatial.distance import cosine

# ---------------------
# CONFIG
# ---------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("MODEL LOADED")
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM allocated:",
          round(torch.cuda.memory_allocated()/1e9, 2), "GB")

print("==============================\n")

LAYER = 28
N_SAMPLES = 200
MAX_MUTATIONS = 8
BATCH_SIZE = 128
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# =====================
# LOAD MODEL
# =====================
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# =====================
# MUTATION FUNCTIONS
# =====================
def mutate_sequence(seq, n_mutations):
    seq = list(seq)
    positions = random.sample(range(len(seq)), n_mutations)
    for pos in positions:
        original = seq[pos]
        choices = [aa for aa in AMINO_ACIDS if aa != original]
        seq[pos] = random.choice(choices)
    return "".join(seq)

# =====================
# EMBEDDING FUNCTIONS
# =====================
@torch.no_grad()
def embed_batch(sequences):
    print(f"Embedding batch of size {len(sequences)}...")

    batch = [(f"seq{i}", s) for i, s in enumerate(sequences)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(DEVICE)

    start = time.time()

    out = model(tokens, repr_layers=[LAYER], return_contacts=False)
    reps = out["representations"][LAYER]

    elapsed = time.time() - start
    print(f"Forward pass done in {elapsed:.2f}s")

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = reps[i, 1:len(seq)+1].mean(dim=0)
        embeddings.append(emb.cpu().numpy())

    return embeddings

def embed_parallel(sequences, batch_size):
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Embedding batches",
                  leave=False):
        batch_seqs = sequences[i:i+batch_size]
        batch_embs = embed_batch(batch_seqs)
        all_embeddings.extend(batch_embs)

    return all_embeddings

# =====================
# MAIN ANALYSIS
# =====================
def run_mutation_similarity_analysis(tonb_sequence):

    global_start = time.time()

    print("Computing WT embedding...")
    wt_embedding = embed_parallel([tonb_sequence], 1)[0]
    print("WT embedding ready.\n")

    similarity_distributions = {}

    mutation_iterator = tqdm(
        range(1, MAX_MUTATIONS + 1),
        desc="Mutation levels"
    )

    for k in mutation_iterator:

        step_start = time.time()

        mutated_sequences = [
            mutate_sequence(tonb_sequence, k)
            for _ in range(N_SAMPLES)
        ]

        mutated_embeddings = embed_parallel(
            mutated_sequences,
            BATCH_SIZE
        )

        similarities = [
            1 - cosine(wt_embedding, emb)
            for emb in mutated_embeddings
        ]

        similarity_distributions[k] = similarities

        step_time = time.time() - step_start
        total_elapsed = time.time() - global_start
        avg = total_elapsed / k
        eta = avg * (MAX_MUTATIONS - k)

        mutation_iterator.set_postfix({
            "step_s": round(step_time, 1),
            "ETA_min": round(eta / 60, 1)
        })

    print("\nTOTAL TIME:",
          round((time.time() - global_start)/60, 2),
          "minutes")

    return similarity_distributions


# =====================
# PLOTTING
# =====================
def plot_distributions(similarity_distributions):
    plt.figure(figsize=(10, 6))

    for k, sims in similarity_distributions.items():
        plt.hist(
            sims,
            bins=40,
            alpha=0.4,
            density=True,
            label=f"{k} mutations"
        )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./result.png", dpi = 300)
    plt.show()

# =====================
# USAGE
# =====================
results = run_mutation_similarity_analysis(tonb_sequence)
plot_distributions(results)