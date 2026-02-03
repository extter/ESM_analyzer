import torch
import esm
import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import cosine
from tqdm import tqdm

wt_sequence = load_txt_sequence("/kaggle/input/betalactamase/beta-lactamase.txt")
print("WT length:", len(wt_sequence))


def load_txt_sequence(txt_path):
    with open(txt_path, "r") as f:
        seq = f.read().strip()  # rimuove spazi, newline, tab
    return seq
    


CONSERVATIVE_MUTATIONS = {
    "L": ["I", "V"],
    "I": ["L", "V"],
    "V": ["L", "I"],
    "D": ["E"],
    "E": ["D"],
    "S": ["T"],
    "T": ["S"],
    "K": ["R"],
    "R": ["K"],
}

NON_CONSERVATIVE = ["W", "P", "G"]


def generate_single_mutants(sequence, mutation_dict, max_per_position=1):
    mutants = []
    for i, aa in enumerate(sequence):
        if aa in mutation_dict:
            for new_aa in mutation_dict[aa][:max_per_position]:
                mutated = list(sequence)
                mutated[i] = new_aa
                mutants.append(("cons", i, aa, new_aa, "".join(mutated)))
    return mutants


def generate_non_conservative_mutants(sequence, max_mutations=200):
    mutants = []
    positions = np.random.choice(len(sequence), size=max_mutations, replace=False)
    for i in positions:
        aa = sequence[i]
        for new_aa in NON_CONSERVATIVE:
            if new_aa != aa:
                mutated = list(sequence)
                mutated[i] = new_aa
                mutants.append(("noncons", i, aa, new_aa, "".join(mutated)))
    return mutants


device = "cuda" if torch.cuda.is_available() else "cpu"

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()


@torch.no_grad()
def get_sequence_embedding(sequence, layer):
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer][0, 1:-1]  # rimuove CLS e EOS

    return reps.mean(dim=0).cpu().numpy()


LAYERS = list(range(20, 34))

def run_layer_analysis(wt_seq):
    results = {l: {"cons": [], "noncons": []} for l in LAYERS}

    wt_embeddings = {
        l: get_sequence_embedding(wt_seq, l)
        for l in LAYERS
    }

    cons_mutants = generate_single_mutants(wt_seq, CONSERVATIVE_MUTATIONS)
    noncons_mutants = generate_non_conservative_mutants(wt_seq)

    for label, i, aa, new_aa, seq in tqdm(cons_mutants):
        for l in LAYERS:
            emb = get_sequence_embedding(seq, l)
            d = cosine(wt_embeddings[l], emb)
            results[l]["cons"].append(d)

    for label, i, aa, new_aa, seq in tqdm(noncons_mutants):
        for l in LAYERS:
            emb = get_sequence_embedding(seq, l)
            d = cosine(wt_embeddings[l], emb)
            results[l]["noncons"].append(d)

    return results


def summarize_results(results):
    summary = []

    for l, data in results.items():
        mu_cons = np.mean(data["cons"])
        mu_non = np.mean(data["noncons"])
        sigma_cons = np.std(data["cons"])

        ratio = mu_non / mu_cons if mu_cons > 0 else np.nan

        summary.append((l, mu_cons, mu_non, ratio, sigma_cons))

    return sorted(summary, key=lambda x: (-x[3], x[1]))


results = run_layer_analysis(wt_sequence)
summary = summarize_results(results)

print("Layer | μ_cons | μ_noncons | Ratio | σ_cons")
for row in summary:
    print(f"{row[0]:>5} | {row[1]:.4e} | {row[2]:.4e} | {row[3]:.2f} | {row[4]:.2e}")
