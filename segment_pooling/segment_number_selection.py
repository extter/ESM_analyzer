layer = 28
segment_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # numeri di segmenti da provare
n_random_baseline = 300            # ensemble random
n_conservative = 300               # mutazioni conservative

# ------------------------
# CARICA UNIREF50 (150–700 aa)
# ------------------------
uniref_fasta_path = "./datasets/uniref50_subsample.fasta"
n_uniref = 300  # stesso ordine di grandezza degli altri baseline

print("Caricamento UniRef50...")
uniref_sequences = []

with open(uniref_fasta_path, "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        if 150 <= len(seq) <= 700:
            uniref_sequences.append(seq)
        if len(uniref_sequences) >= n_uniref:
            break

print(f"Caricate {len(uniref_sequences)} sequenze UniRef50")

# ------------------------
# EMBEDDING UNIREF
# ------------------------
print("Calcolo embedding UniRef50...")
uniref_embs = []
for seq in tqdm(uniref_sequences, desc="UniRef embeddings"):
    emb = get_residue_embeddings(seq)
    uniref_embs.append(emb)


# ------------------------
# GENERA ENSEMBLE RANDOM
# ------------------------
print("Generazione ensemble random...")
random_seqs = [random_sequence(len(tonb_seq) + random.randint(-50, 50)) for _ in range(n_random_baseline)]
all_random_segs_dict = {}  # cache embeddings per numero segmenti
print("Calcolo embedding random batch...")
for rseq in tqdm(random_seqs, desc="Random embeddings"):
    emb = get_residue_embeddings(rseq)
    all_random_segs_dict[rseq] = emb

# ------------------------
# LOOP SUI NUMERI DI SEGMENTI
# ------------------------
results = {}
tonb_emb_full = get_residue_embeddings(tonb_seq)  # embedding TonB intero

for n_seg in segment_list:
    print(f"\n=== Segmenti: {n_seg} ===")
    tonb_seg = split_into_segments(tonb_emb_full, n_seg)
    
    # ------------------------
    # TonB vs ensemble random
    # ------------------------
    ensemble_sims = []
    for rseq in random_seqs:
        rseg = split_into_segments(all_random_segs_dict[rseq], n_seg)
        ensemble_sims.append(global_similarity(tonb_seg, rseg))
    mean_ensemble = np.mean(ensemble_sims)
    std_ensemble = np.std(ensemble_sims)
    
    # ------------------------
    # TonB vs conservative
    # ------------------------
    sims_cons = []
    for _ in range(n_conservative):
        mut_seq = conservative_mutation_blosum(tonb_seq)
        mut_seg = split_into_segments(get_residue_embeddings(mut_seq), n_seg)
        sims_cons.append(global_similarity(tonb_seg, mut_seg))
    mean_cons = np.mean(sims_cons)
    std_cons = np.std(sims_cons)

    # ------------------------
    # TonB vs UniRef50 random  (NUOVO)
    # ------------------------
    sims_uniref = []
    for emb in uniref_embs:
        useg = split_into_segments(emb, n_seg)
        sims_uniref.append(global_similarity(tonb_seg, useg))
    mean_uniref = np.mean(sims_uniref)
    std_uniref = np.std(sims_uniref)
    
    # ------------------------
    # DELTA
    # ------------------------
    delta_rand = mean_cons - mean_ensemble
    delta_uniref = mean_cons - mean_uniref
    
    print(f"TonB vs Random:   {mean_ensemble:.4f} ± {std_ensemble:.4f}")
    print(f"TonB vs UniRef:   {mean_uniref:.4f} ± {std_uniref:.4f}")
    print(f"TonB vs Cons:     {mean_cons:.4f} ± {std_cons:.4f}")
    print(f"Δ Cons–Random:   {delta_rand:.4f}")
    print(f"Δ Cons–UniRef:   {delta_uniref:.4f}")
    
    results[n_seg] = {
        "ensemble_sims": ensemble_sims,
        "cons_sims": sims_cons,
        "uniref_sims": sims_uniref,
        "delta_rand": delta_rand,
        "delta_uniref": delta_uniref
    }

# ------------------------
# GRAFICO BOX PLOT
# ------------------------
plt.figure(figsize=(18,6))
data_to_plot = []
labels = []

for n_seg in segment_list:
    data_to_plot.append(results[n_seg]["cons_sims"])
    data_to_plot.append(results[n_seg]["ensemble_sims"])
    data_to_plot.append(results[n_seg]["uniref_sims"])
    
    labels.append(f"Cons {n_seg}")
    labels.append(f"Random {n_seg}")
    labels.append(f"UniRef {n_seg}")

plt.boxplot(data_to_plot, labels=labels, showfliers=False)
plt.xticks(rotation=60)
plt.ylabel("Segment-wise cosine similarity")
plt.title("TonB: Conservative vs Random vs UniRef50")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./result.png", dpi = 300)
plt.show()