import pandas as pd
import torch
import matplotlib.pyplot as plt
import esm
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


###################################
# FUNZIONI
###################################

def load_txt_sequence(txt_path):
    with open(txt_path, "r") as f:
        return f.read().strip()


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


def generate_single_mutants(sequence):
    mutants = []
    for i, aa in enumerate(sequence):
        if aa in CONSERVATIVE_MUTATIONS:
            for new_aa in CONSERVATIVE_MUTATIONS[aa]:
                mutated = list(sequence)
                mutated[i] = new_aa
                mutants.append("".join(mutated))
    return mutants


def generate_non_conservative_mutants(sequence, max_mutations=200):
    mutants = []
    positions = np.random.choice(len(sequence), size=max_mutations, replace=False)

    for i in positions:
        for new_aa in NON_CONSERVATIVE:
            mutated = list(sequence)
            mutated[i] = new_aa
            mutants.append("".join(mutated))

    return mutants


@torch.no_grad()
def get_embeddings_batch_multi(sequences, layers, batch_size=8):

    layer_outputs = {l: [] for l in layers}

    for i in range(0, len(sequences), batch_size):

        batch = sequences[i:i+batch_size]
        data = list(zip(map(str, range(len(batch))), batch))

        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=layers, return_contacts=False)

        for l in layers:
            reps = out["representations"][l][:, 1:-1]   # mean pooling
            mean_reps = reps.mean(dim=1).cpu().numpy()
            layer_outputs[l].extend(mean_reps)

    # converti tutto in numpy
    for l in layers:
        layer_outputs[l] = np.array(layer_outputs[l])

    return layer_outputs


LAYERS = list(range(20,34))


def run_layer_analysis(wt_seq):

    cons_mutants = generate_single_mutants(wt_seq)
    noncons_mutants = generate_non_conservative_mutants(wt_seq)

    # UNA SOLA FORWARD PER TUTTO
    wt_embs = get_embeddings_batch_multi([wt_seq], LAYERS)
    cons_embs = get_embeddings_batch_multi(cons_mutants, LAYERS)
    non_embs = get_embeddings_batch_multi(noncons_mutants, LAYERS)

    results = []

    for l in LAYERS:

        wt_emb = wt_embs[l][0]

        cons_dist = [cosine(wt_emb, e) for e in cons_embs[l]]
        non_dist = [cosine(wt_emb, e) for e in non_embs[l]]

        mu_cons = np.mean(cons_dist)
        mu_non = np.mean(non_dist)
        sigma_cons = np.std(cons_dist)

        ratio = mu_non / mu_cons if mu_cons > 0 else np.nan

        results.append((l, mu_cons, mu_non, ratio, sigma_cons))

    return results


###################################
# MODELLO 
###################################

device = "cuda" if torch.cuda.is_available() else "cpu"

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()


###################################
# MAIN
###################################

protein_paths = {
    "Betalactamase": "../sequences/beta-lactamase.txt",
    "DNAjb1": "../sequences/dnajb1.txt",
    "GB1": "../sequences/gb1.txt",
    "TonB": "../sequences/tonb.txt",
}

all_rows = []

for name, path in protein_paths.items():

    print(f"\nRunning analysis for {name}")

    wt_sequence = load_txt_sequence(path)
    summary = run_layer_analysis(wt_sequence)

    for l, mu_cons, mu_non, ratio, sigma_cons in summary:
        all_rows.append({
            "protein": name,
            "layer": l,
            "mu_cons": mu_cons,
            "mu_noncons": mu_non,
            "ratio": ratio,
            "sigma_cons": sigma_cons
        })

df = pd.DataFrame(all_rows)


plt.figure(figsize=(10,6))

for protein in df["protein"].unique():
    subset = df[df["protein"] == protein]
    plt.plot(subset["layer"], subset["ratio"], marker='o', label=protein)

plt.xlabel("Layer")
plt.ylabel("μ_noncons / μ_cons")
plt.title("Layer sensitivity to mutations")
plt.legend()
plt.grid()
plt.savefig("./layer_selection.png", dpi=300)
plt.show()


mean_df = df.groupby("layer")["ratio"].mean()

plt.plot(mean_df.index, mean_df.values, marker='o')
plt.xlabel("Layer")
plt.ylabel("Mean ratio")
plt.title("Average across proteins")
plt.grid()
plt.savefig("./mean_ratio.png", dpi=300)
plt.show()
