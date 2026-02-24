import pandas as pd
import torch
import matplotlib.pyplot as plt
import esm
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple
from pathlib import Path

#---------------------------
# COSTANTI
#---------------------------

CONSERVATIVE_MUTATIONS = {
    "L": ["I", "V"], "I": ["L", "V"], "V": ["L", "I"],
    "D": ["E"], "E": ["D"],
    "S": ["T"], "T": ["S"],
    "K": ["R"], "R": ["K"],
}
NON_CONSERVATIVE = ["W", "P", "G"]
LAYERS = list(range(20, 34))

#---------------------------
# FUNZIONI DI SUPPORTO
#---------------------------

def load_txt_sequence(txt_path: str | Path) -> str:
    """Carica una sequenza amminoacidica da un file di testo."""
    return Path(txt_path).read_text(encoding="utf-8").strip()

def generate_single_mutants(sequence: str) -> List[str]:
    """Genera tutte le possibili mutazioni conservative singole per la sequenza data."""
    mutants = []
    for i, aa in enumerate(sequence):
        if aa in CONSERVATIVE_MUTATIONS:
            for new_aa in CONSERVATIVE_MUTATIONS[aa]:
                mutated = list(sequence)
                mutated[i] = new_aa
                mutants.append("".join(mutated))
    return mutants

def generate_non_conservative_mutants(sequence: str, max_mutations: int = 200) -> List[str]:
    """Genera mutanti non conservativi campionando posizioni casuali."""
    mutants = []
    num_samples = min(len(sequence), max_mutations)
    positions = np.random.choice(len(sequence), size=num_samples, replace=False)

    for i in positions:
        for new_aa in NON_CONSERVATIVE:
            mutated = list(sequence)
            mutated[i] = new_aa
            mutants.append("".join(mutated))
    return mutants

#---------------------------
# CORE ESM-2
#---------------------------

@torch.no_grad()
def get_embeddings_batch_multi(
    sequences: List[str], 
    layers: List[int], 
    model: torch.nn.Module, 
    batch_converter: callable, 
    device: torch.device,
    batch_size: int = 8
) -> Dict[int, np.ndarray]:
    """
    Estrae gli embedding per un batch di sequenze calcolando il mean pooling.
    Ignora i token <cls> e <eos> (indici 1:-1).
    """
    layer_outputs = {l: [] for l in layers}

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        data = list(zip(map(str, range(len(batch))), batch))

        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=layers, return_contacts=False)

        for l in layers:
            # Mean pooling escludendo <cls> all'inizio e <eos> alla fine
            reps = out["representations"][l][:, 1:-1]
            mean_reps = reps.mean(dim=1).cpu().numpy()
            layer_outputs[l].extend(mean_reps)

    for l in layers:
        layer_outputs[l] = np.array(layer_outputs[l])

    return layer_outputs

def run_layer_analysis(
    wt_seq: str, 
    model: torch.nn.Module, 
    batch_converter: callable, 
    device: torch.device
) -> List[Tuple[int, float, float, float, float]]:
    """Esegue l'analisi delle distanze per tutti i layer definiti."""
    cons_mutants = generate_single_mutants(wt_seq)
    noncons_mutants = generate_non_conservative_mutants(wt_seq)

    wt_embs = get_embeddings_batch_multi([wt_seq], LAYERS, model, batch_converter, device)
    cons_embs = get_embeddings_batch_multi(cons_mutants, LAYERS, model, batch_converter, device)
    non_embs = get_embeddings_batch_multi(noncons_mutants, LAYERS, model, batch_converter, device)

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

#---------------------------
# MAIN EXECUTION
#---------------------------

def main():
    # 1. Inizializzazione modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM-2 model on {device}...")
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 2. Setup percorsi usando Pathlib (più sicuro cross-platform)
    base_dir = Path(__file__).parent.parent / "sequences"
    protein_paths = {
        "Betalactamase": base_dir / "beta-lactamase.txt",
        "DNAjb1": base_dir / "dnajb1.txt",
        "GB1": base_dir / "gb1.txt",
        "TonB": base_dir / "tonb.txt",
    }

    # 3. Esecuzione analisi
    all_rows = []
    for name, path in protein_paths.items():
        print(f"\nRunning analysis for {name}")
        try:
            wt_sequence = load_txt_sequence(path)
        except FileNotFoundError:
            print(f"Warning: File {path} not found. Skipping {name}.")
            continue
            
        summary = run_layer_analysis(wt_sequence, model, batch_converter, device)

        for l, mu_cons, mu_non, ratio, sigma_cons in summary:
            all_rows.append({
                "protein": name, "layer": l,
                "mu_cons": mu_cons, "mu_noncons": mu_non,
                "ratio": ratio, "sigma_cons": sigma_cons
            })

    if not all_rows:
        print("No data processed. Exiting.")
        return

    df = pd.DataFrame(all_rows)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    for protein in df["protein"].unique():
        subset = df[df["protein"] == protein]
        plt.plot(subset["layer"], subset["ratio"], marker='o', label=protein)

    plt.xlabel("ESM2 Layer", size = 14)
    plt.ylabel("μ_noncons / μ_cons", size = 14)
    plt.title("Layer sensitivity to mutations", size = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
    plt.savefig("layer_selection.png", dpi=300)
    plt.close()

    mean_df = df.groupby("layer")["ratio"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(mean_df.index, mean_df.values, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Mean ratio")
    plt.title("Average across proteins")
    plt.grid()
    plt.savefig("mean_ratio.png", dpi=300)
    plt.close()
    
    print("\nAnalysis complete. Plots saved.")

if __name__ == "__main__":
    main()