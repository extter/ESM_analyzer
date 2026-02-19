import os
import random
import torch
import torch.nn as nn
from esm import pretrained
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from Bio.Align import substitution_matrices
from Bio import SeqIO
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# COSTANTI E CONFIGURAZIONE
# -----------------------------------------------------------------------------
ESM_LAYER = 28
N_CONS = 500
N_RANDOM_TONB = 500
N_RANDOM_BASE = 500
N_UNIREF_TEST = 500
BATCH_SIZE = 8
N_SEGMENTS = 24

UNIREF_FASTA_PATH = "./datasets/uniref50_subsample.fasta"
OUT_DIR_MEAN = "comparison_results/plots_mean_pooling"
OUT_DIR_SEG = "comparison_results/plots_segment_pooling"
os.makedirs(OUT_DIR_MEAN, exist_ok=True)
os.makedirs(OUT_DIR_SEG, exist_ok=True)

SEQ_TONB = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"
SEQ_HB = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

PCA_CONFIGS = [
    {"name": "PCA_Random", "pca_path": "./joblibs/Random_ipca_fitted.joblib", "esm_layer": ESM_LAYER},
    {"name": "PCA_Uniref", "pca_path": "./joblibs/Uniref_ipca_fitted.joblib", "esm_layer": ESM_LAYER},
    {"name": "PCA_TonB", "pca_path": "./joblibs/Tonb_ipca_fitted.joblib", "esm_layer": ESM_LAYER},
    {"name": "PCA_Total", "pca_path": "./joblibs/Combined_ipca_fitted.joblib", "esm_layer": ESM_LAYER},
]

# -----------------------------------------------------------------------------
# INIZIALIZZAZIONE E PREPARAZIONE DATI
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

model, alphabet = pretrained.esm2_t33_650M_UR50D()
model.to(device).eval()
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    print(f"Uso {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)

try:
    blosum62 = substitution_matrices.load("BLOSUM62")
except Exception:
    print("Warning: Matrice BLOSUM62 non trovata.")
    blosum62 = {}

def blosum_score(a, b):
    return blosum62.get((a, b), blosum62.get((b, a), -10))

print(" Caricamento campione UniRef per i test...")
uniref_sequences = []
try:
    with open(UNIREF_FASTA_PATH) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            L = len(record.seq)
            if 150 <= L <= 700:
                uniref_sequences.append(str(record.seq))
                if len(uniref_sequences) >= N_UNIREF_TEST * 3:
                    break
    uniref_test = random.sample(uniref_sequences, min(N_UNIREF_TEST, len(uniref_sequences)))
    print(f"UniRef test pronti: {len(uniref_test)} sequenze")
except FileNotFoundError:
    print(f"Errore: {UNIREF_FASTA_PATH} non trovato. L'analisi su UniRef sarà vuota.")
    uniref_test = []

# --- Generazione Dati ---
def generate_conservative_mutants(seq, n=N_CONS):
    mutants = []
    L = len(seq)
    for _ in range(n):
        s = list(seq)
        for pos in random.sample(range(L), random.randint(1, 5)):
            aa = s[pos]
            candidates = [x for x in AA_ALPHABET if x != aa and blosum_score(aa, x) >= 2]
            if candidates:
                s[pos] = random.choice(candidates)
        mutants.append("".join(s))
    return mutants

def generate_random_proteins(n, length_range=(200, 400)):
    return ["".join(random.choices(AA_ALPHABET, k=random.randint(*length_range))) for _ in range(n)]

# -----------------------------------------------------------------------------
# MOTORE DI EMBEDDING ED ESTRAZIONE
# -----------------------------------------------------------------------------
@torch.no_grad()
def forward_esm(model, tokens, **kwargs):
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        device_0 = f"cuda:{model.device_ids[0]}" if hasattr(model, 'device_ids') else "cuda:0"
        return model.module.to(device_0)(tokens.to(device_0), **kwargs)
    return model(tokens.to(device), **kwargs)

def get_residue_embeddings_batch(seqs, esm_layer):
    data = [("seq", s) for s in seqs]
    _, batch_strs, tokens = batch_converter(data)
    
    out = forward_esm(model, tokens, repr_layers=[esm_layer], return_contacts=False)
    reps = out["representations"][esm_layer]
    return [reps[i, 1:1+len(s)].cpu().numpy() for i, s in enumerate(batch_strs)]

@torch.no_grad()
def get_global_embeddings_batch(seqs, ipca, esm_layer, pooling_mode="mean", n_segments=24):
    """
    Gestisce l'estrazione degli embedding applicando la PCA e il tipo di pooling richiesto.
    """
    all_vecs = []
    
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i+BATCH_SIZE]
        emb_list = get_residue_embeddings_batch(batch, esm_layer)
        
        for emb in emb_list:
            pca_emb = ipca.transform(emb)
            
            if pooling_mode == "mean":
                vec = pca_emb.mean(axis=0)
            
            elif pooling_mode == "segment":
                L, d = pca_emb.shape
                seg_size = max(1, L // n_segments)
                segment_vecs = []
                for j in range(n_segments):
                    start = j * seg_size
                    end = (j + 1) * seg_size if j < n_segments - 1 else L
                    seg_emb = pca_emb[start:end]
                    
                    if len(seg_emb) == 0:
                        segment_vecs.append(np.zeros(d))
                    else:
                        segment_vecs.append(seg_emb.mean(axis=0))
                vec = np.concatenate(segment_vecs)
            else:
                raise ValueError("Pooling mode deve essere 'mean' o 'segment'")

            vec /= (np.linalg.norm(vec) + 1e-8)
            all_vecs.append(vec)
            
        torch.cuda.empty_cache()
        
    return np.vstack(all_vecs)

# -----------------------------------------------------------------------------
# PLOTTING ED ESECUZIONE 
# -----------------------------------------------------------------------------
def plot_results(data_list, title, save_path):
    """Genera e salva il boxplot standard per l'analisi."""
    labels = ["TonB vs Cons", "TonB vs Rand", "TonB vs UniRef", "Rand vs Rand", "UniRef vs UniRef"]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data_list, widths=0.5, showfliers=True, patch_artist=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() 

def esegui_analisi(pooling_mode):
    """
    Esegue l'intera pipeline di analisi calcolando le cosine similarity
    sia per Mean Pooling che per Segment Pooling.
    """
    print(f"\n==================== ANALISI MULTI-PCA: {pooling_mode.upper()} POOLING ====================\n")
    
    cons_seqs = generate_conservative_mutants(SEQ_TONB)
    rand_seqs = generate_random_proteins(N_RANDOM_TONB, (len(SEQ_TONB)-50, len(SEQ_TONB)+50))
    rand_base_seqs = generate_random_proteins(N_RANDOM_BASE)
    
    for cfg in PCA_CONFIGS:
        print(f"Modello: {cfg['name']}")
        try:
            ipca = joblib.load(cfg["pca_path"])
        except FileNotFoundError:
            print(f"File {cfg['pca_path']} non trovato. Salto.")
            continue
            
        layer = cfg["esm_layer"]
        
        # Estrazione
        Z_tonb = get_global_embeddings_batch([SEQ_TONB], ipca, layer, pooling_mode)[0]
        Z_cons = get_global_embeddings_batch(cons_seqs, ipca, layer, pooling_mode)
        Z_rand = get_global_embeddings_batch(rand_seqs, ipca, layer, pooling_mode)
        Z_uniref = get_global_embeddings_batch(uniref_test, ipca, layer, pooling_mode)
        Z_hb = get_global_embeddings_batch([SEQ_HB], ipca, layer, pooling_mode)[0]
        Z_rand_base = get_global_embeddings_batch(rand_base_seqs, ipca, layer, pooling_mode)

        # Calcolo Similarità
        cos = cosine_similarity
        cos_tonb_cons = cos(Z_tonb[None], Z_cons)[0]
        cos_tonb_rand = cos(Z_tonb[None], Z_rand)[0]
        cos_tonb_uniref = cos(Z_tonb[None], Z_uniref)[0] if len(Z_uniref) > 0 else np.array([])
        cos_tonb_hb = cos(Z_tonb[None], Z_hb[None])[0, 0]

        cos_rand_rand = cos(Z_rand_base, Z_rand_base)[np.triu_indices(len(Z_rand_base), 1)]
        cos_uniref_uniref = cos(Z_uniref, Z_uniref)[np.triu_indices(len(Z_uniref), 1)] if len(Z_uniref) > 0 else np.array([])

        print(f"TonB vs Conservative: {cos_tonb_cons.mean():.4f} ± {cos_tonb_cons.std():.4f}")
        print(f"TonB vs Random:       {cos_tonb_rand.mean():.4f} ± {cos_tonb_rand.std():.4f}")
        print(f"TonB vs Emoglobina:   {cos_tonb_hb:.4f}")
        
        # Plotting
        data_to_plot = [cos_tonb_cons, cos_tonb_rand]
        if len(cos_tonb_uniref) > 0: data_to_plot.append(cos_tonb_uniref)
        else: data_to_plot.append(np.array([0]))
        
        data_to_plot.append(cos_rand_rand)
        
        if len(cos_uniref_uniref) > 0: data_to_plot.append(cos_uniref_uniref)
        else: data_to_plot.append(np.array([0]))

        out_dir = OUT_DIR_MEAN if pooling_mode == "mean" else OUT_DIR_SEG
        fname = f"boxplot_{pooling_mode}_pooling_{cfg['name']}.png"
        
        plot_results(data_to_plot, f"PCA: {cfg['name']} - {pooling_mode.capitalize()} Pooling", os.path.join(out_dir, fname))

if __name__ == "__main__":
    esegui_analisi(pooling_mode="mean")
    esegui_analisi(pooling_mode="segment")
    print("Tutte le analisi completate e i grafici salvati!")


