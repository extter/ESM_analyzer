import os
import torch
import torch.nn as nn
from esm import pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA
import joblib
from Bio.Align import substitution_matrices
import random
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

ESM_LAYER = 28

PCA_CONFIGS = [
    {
        "name": "PCA_Random",
        "pca_path": "./joblibs/Random_ipca_640comp_100k.joblib",
        "esm_layer": 28,
    },
    {
        "name": "PCA_Uniref",
        "pca_path": "./joblibs/Uniref_640comp_100k.joblib",
        "esm_layer": 28,
    },
    {
        "name": "PCA_TonB_mutations",
        "pca_path": "./joblibs/TonB_ipca_fitted_final.joblib",
        "esm_layer": 28,
    },
    {
        "name": "PCA_Total",
        "pca_path": "./joblibs/Total_ipca_fitted.joblib",
        "esm_layer": 28,
    },
]


N_CONS = 500
N_RANDOM_TONB = 500
N_RANDOM_BASE = 500
N_UNIREF_TEST = 500
BATCH_SIZE = 8

uniref_fasta_path = "/kaggle/input/uniref-subsample/uniref50_subsample.fasta"

seq_tonb = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

seq_hb = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model.to(device).eval()
batch_converter = alphabet.get_batch_converter()
if torch.cuda.device_count() > 1:
    print("Uso", torch.cuda.device_count(), "GPU")
    model = nn.DataParallel(model, device_ids=[0, 1])

print("🔄 Caricamento UniRef...")
uniref_sequences = []

with open(uniref_fasta_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        L = len(record.seq)
        if 150 <= L <= 700:
            uniref_sequences.append(str(record.seq))

random.shuffle(uniref_sequences)
uniref_test = uniref_sequences[:N_UNIREF_TEST]

print(f"UniRef test: {len(uniref_test)}")


blosum62 = substitution_matrices.load("BLOSUM62")

def blosum_score(a, b):
    return blosum62.get((a, b), blosum62.get((b, a), -10))

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


@torch.no_grad()
def forward_esm(model, tokens, **kwargs):
    # Se hai più GPU, meglio usare solo 1 per batch piccoli
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        return model.module(tokens.to("cuda:0"), **kwargs)
    # Altrimenti invia il batch al device principale
    tokens = tokens.to(device)
    return model(tokens, **kwargs)


def get_residue_embeddings_batch(seqs, esm_layer):
    data = [("seq", s) for s in seqs]
    _, batch_strs, tokens = batch_converter(data)
    tokens = tokens.to(device)

    # ← qui usiamo forward_esm, non il modello direttamente
    out = forward_esm(model, tokens, repr_layers=[esm_layer], return_contacts=False)
    reps = out["representations"][esm_layer]

    return [reps[i, 1:1+len(s)].cpu().numpy() for i, s in enumerate(batch_strs)]


@torch.no_grad()
def get_global_embeddings_after_ipca(seqs, ipca, esm_layer):
    all_vecs = []

    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i+BATCH_SIZE]
        data = [("seq", s) for s in batch]

        _, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        out = forward_esm(
            model,
            batch_tokens,
            repr_layers=[esm_layer],
            return_contacts=False
        )

        reps = out["representations"][esm_layer]

        for j, seq in enumerate(batch_strs):
            L = len(seq)
            emb = reps[j, 1:1+L].cpu().numpy()
            pca_emb = ipca.transform(emb)

            mean_vec = pca_emb.mean(axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-8)
            all_vecs.append(mean_vec)

        torch.cuda.empty_cache()

    return np.array(all_vecs)

OUT_DIR = "plots_mean_pooling"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def get_global_embedding_segment_pooling(seq, ipca, esm_layer, n_segments=24):
    """
    Embedding globale PCA + segment pooling.
    Divide la proteina in `n_segments` segmenti e fa la media di ogni segmento.
    """
    # 1. Embedding residue
    emb_list = get_residue_embeddings_batch([seq], esm_layer)
    emb = emb_list[0]  # [L, d]

    # 2. PCA
    emb_pca = ipca.transform(emb)  # [L, pca_components]
    L, d = emb_pca.shape

    # 3. Calcola segmenti
    seg_size = L // n_segments
    segment_vecs = []

    for i in range(n_segments):
        start = i * seg_size
        # l'ultimo segmento prende tutto ciò che resta
        end = (i + 1) * seg_size if i < n_segments - 1 else L
        seg_emb = emb_pca[start:end]
        mean_seg = seg_emb.mean(axis=0)
        segment_vecs.append(mean_seg)

    # 4. Concatenazione segmenti + normalizzazione
    global_seg_vec = np.concatenate(segment_vecs)
    global_seg_vec /= (np.linalg.norm(global_seg_vec) + 1e-8)
    return global_seg_vec


@torch.no_grad()
def get_global_embeddings_batch_segment_pooling(
    seqs, ipca, esm_layer, n_segments=24, batch_size=4
):
    all_vecs = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]

        emb_list = get_residue_embeddings_batch(batch, esm_layer)

        for emb in emb_list:
            emb_pca = ipca.transform(emb)
            L, d = emb_pca.shape

            seg_size = max(1, L // n_segments)
            segment_vecs = []

            for j in range(n_segments):
                start = j * seg_size
                end = (j + 1) * seg_size if j < n_segments - 1 else L
                seg_emb = emb_pca[start:end]

                if len(seg_emb) == 0:
                    segment_vecs.append(np.zeros(d))
                else:
                    segment_vecs.append(seg_emb.mean(axis=0))

            vec = np.concatenate(segment_vecs)
            vec /= (np.linalg.norm(vec) + 1e-8)
            all_vecs.append(vec)

        torch.cuda.empty_cache()

    return np.vstack(all_vecs)

OUT2_DIR = "plots_segment_pooling"
os.makedirs(OUT2_DIR, exist_ok=True)


print("\n==================== ANALISI MULTI-PCA su random, uniref, hb ====================\n")

for cfg in PCA_CONFIGS:
    print(f"\n🔬 {cfg['name']}")

    ipca = joblib.load(cfg["pca_path"])
    esm_layer = cfg["esm_layer"]

    print("🔧 CONFIG:")
    print(cfg)
    print("PCA components:", ipca.n_components)

    # ------------------------
    # EMBEDDING
    # ------------------------
    Z_tonb = get_global_embeddings_after_ipca([seq_tonb], ipca, esm_layer)[0]

    # Conservative
    cons_seqs = generate_conservative_mutants(seq_tonb)
    Z_cons = get_global_embeddings_after_ipca(cons_seqs, ipca, esm_layer)

    # Random (lunghezza simile a TonB)
    rand_seqs = generate_random_proteins(N_RANDOM_TONB, (len(seq_tonb)-50, len(seq_tonb)+50))
    Z_rand = get_global_embeddings_after_ipca(rand_seqs, ipca, esm_layer)

    # UniRef
    Z_uniref = get_global_embeddings_after_ipca(uniref_test, ipca, esm_layer)

    # Emoglobina
    Z_hb = get_global_embeddings_after_ipca([seq_hb], ipca, esm_layer)[0]

    # Baseline random vs random
    Z_rand_base = get_global_embeddings_after_ipca(
        generate_random_proteins(N_RANDOM_BASE), ipca, esm_layer
    )

    # ------------------------
    # COSINE SIMILARITY + STD
    # ------------------------
    cos = cosine_similarity

    cos_tonb_cons = cos(Z_tonb[None], Z_cons)[0]
    cos_tonb_rand = cos(Z_tonb[None], Z_rand)[0]
    cos_tonb_uniref = cos(Z_tonb[None], Z_uniref)[0]
    cos_tonb_hb = cos(Z_tonb[None], Z_hb[None])[0,0]

    cos_rand_rand = cos(Z_rand_base, Z_rand_base)[np.triu_indices(len(Z_rand_base),1)]
    cos_uniref_uniref = cos(Z_uniref, Z_uniref)[np.triu_indices(len(Z_uniref),1)]

    # ------------------------
    # STAMPA RISULTATI
    # ------------------------
    print("\n📊 RISULTATI COSINE SIMILARITY:")
    print(f"TonB vs Conservative: {cos_tonb_cons.mean():.4f} ± {cos_tonb_cons.std():.4f}")
    print(f"TonB vs Random:       {cos_tonb_rand.mean():.4f} ± {cos_tonb_rand.std():.4f}")
    print(f"TonB vs UniRef:       {cos_tonb_uniref.mean():.4f} ± {cos_tonb_uniref.std():.4f}")
    print(f"TonB vs Emoglobina:   {cos_tonb_hb:.4f}")
    print(f"Random vs Random:     {cos_rand_rand.mean():.4f} ± {cos_rand_rand.std():.4f}")
    print(f"UniRef vs UniRef:     {cos_uniref_uniref.mean():.4f} ± {cos_uniref_uniref.std():.4f}")

    # ------------------------
    # BOX PLOT
    # ------------------------
    data = [
        cos_tonb_cons,
        cos_tonb_rand,
        cos_tonb_uniref,
        cos_rand_rand,
        cos_uniref_uniref
    ]
    
    labels = [
        "TonB vs Conservative",
        "TonB vs Random",
        "TonB vs UniRef",
        "Random vs Random",
        "UniRef vs UniRef"
    ]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    ax.boxplot(
        data,
        widths=0.5,
        showfliers=True,
        patch_artist=True
    )
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"PCA: {cfg['name']} – Mean Pooling", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    fname = f"boxplot_mean_pooling_{cfg['name'].replace(' ', '_')}.png"
    save_path = os.path.join(OUT_DIR, fname)
    plt.savefig(save_path, dpi=300)
    plt.show()


    print("\n==================== ANALISI MULTI-PCA su random, uniref, hb (SEGMENT POOLING) ====================\n")

for cfg in PCA_CONFIGS:
    print(f"\n🔬 {cfg['name']}")

    ipca = joblib.load(cfg["pca_path"])
    esm_layer = cfg["esm_layer"]

    print("🔧 CONFIG:")
    print(cfg)
    print("PCA components:", ipca.n_components)

    # ------------------------
    # EMBEDDING (segment pooling 24)
    # ------------------------
    Z_tonb = get_global_embeddings_batch_segment_pooling([seq_tonb], ipca, esm_layer, n_segments=24)[0]
    
    cons_seqs = generate_conservative_mutants(seq_tonb)
    rand_seqs = generate_random_proteins(N_RANDOM_TONB, (len(seq_tonb)-50, len(seq_tonb)+50))
    
    Z_cons = get_global_embeddings_batch_segment_pooling(cons_seqs, ipca, esm_layer, 24)
    Z_rand = get_global_embeddings_batch_segment_pooling(rand_seqs, ipca, esm_layer, 24)
    Z_uniref = get_global_embeddings_batch_segment_pooling(uniref_test, ipca, esm_layer, 24)
    Z_hb = get_global_embeddings_batch_segment_pooling([seq_hb], ipca, esm_layer, n_segments=24)[0]
    Z_rand_base = get_global_embeddings_batch_segment_pooling(
        generate_random_proteins(N_RANDOM_BASE), ipca, esm_layer, n_segments=24
    )


    # ------------------------
    # COSINE SIMILARITY + STD
    # ------------------------
    cos = cosine_similarity

    cos_tonb_cons = cos(Z_tonb[None], Z_cons)[0]
    cos_tonb_rand = cos(Z_tonb[None], Z_rand)[0]
    cos_tonb_uniref = cos(Z_tonb[None], Z_uniref)[0]
    cos_tonb_hb = cos(Z_tonb[None], Z_hb[None])[0,0]

    cos_rand_rand = cos(Z_rand_base, Z_rand_base)[np.triu_indices(len(Z_rand_base),1)]
    cos_uniref_uniref = cos(Z_uniref, Z_uniref)[np.triu_indices(len(Z_uniref),1)]

    # ------------------------
    # STAMPA RISULTATI
    # ------------------------
    print("\n📊 RISULTATI COSINE SIMILARITY (Segment Pooling):")
    print(f"TonB vs Conservative: {cos_tonb_cons.mean():.4f} ± {cos_tonb_cons.std():.4f}")
    print(f"TonB vs Random:       {cos_tonb_rand.mean():.4f} ± {cos_tonb_rand.std():.4f}")
    print(f"TonB vs UniRef:       {cos_tonb_uniref.mean():.4f} ± {cos_tonb_uniref.std():.4f}")
    print(f"TonB vs Emoglobina:   {cos_tonb_hb:.4f}")
    print(f"Random vs Random:     {cos_rand_rand.mean():.4f} ± {cos_rand_rand.std():.4f}")
    print(f"UniRef vs UniRef:     {cos_uniref_uniref.mean():.4f} ± {cos_uniref_uniref.std():.4f}")

    # ------------------------
    # BOX PLOT
    # ------------------------
    data = [
        cos_tonb_cons,
        cos_tonb_rand,
        cos_tonb_uniref,
        cos_rand_rand,
        cos_uniref_uniref
    ]
    
    labels = [
        "TonB vs Conservative",
        "TonB vs Random",
        "TonB vs UniRef",
        "Random vs Random",
        "UniRef vs UniRef"
    ]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    ax.boxplot(
        data,
        widths=0.5,
        showfliers=True,
        patch_artist=True
    )
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"PCA: {cfg['name']} – Segment Pooling", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    fname = f"boxplot_segment_pooling_{cfg['name'].replace(' ', '_')}.png"
    save_path = os.path.join(OUT2_DIR, fname)
    plt.savefig(save_path, dpi=300)
    plt.show()


