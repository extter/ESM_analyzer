# DATASET 2: 100K RANDOM PROTEINE
# =============================================================================
print("\n🎲 2/4: 100k random proteins")
random_proteins = []
aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

for _ in tqdm(range(CONFIG['n_samples']), desc="Random proteins"):
    length = random.randint(*CONFIG['tonb_length_range'])
    seq = "".join(random.choice(aa_alphabet) for _ in range(length))
    random_proteins.append(seq)

with open(f"{CONFIG['output_dir']}/02_random_proteins.fasta", 'w') as f:
    for i, seq in enumerate(random_proteins):
        f.write(f">random_{i} len={len(seq)}\n{seq}\n")

print(f" 02_random_proteins.fasta: {len(random_proteins)} sequences")
print(f"Mean length: {np.mean([len(seq) for seq in random_proteins]):.0f} aa")