from Bio import SeqIO
from pathlib import Path
import math
from collections import Counter, defaultdict

# ------------------------
# CONFIG
# ------------------------
msa_dir = Path("./extracted/run_msas")  # cartella con i file allineati (_aligned.fasta)
output_file = Path("./extracted/run_stats.txt")

# ------------------------
# FUNZIONI
# ------------------------
def calc_conservation(column):
    """Ritorna la percentuale di conservazione della colonna"""
    counts = Counter(column)
    if '-' in counts:
        counts.pop('-')  # ignora gap
    if not counts:
        return 0.0
    most_common = counts.most_common(1)[0][1]
    return most_common / len(column) * 100

def calc_entropy(column):
    """Calcola entropia di Shannon della colonna"""
    counts = Counter(column)
    total = sum(counts.values())
    entropy = 0.0
    for aa, count in counts.items():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calc_gap_fraction(column):
    """Percentuale di gap nella colonna"""
    gaps = column.count('-')
    return gaps / len(column) * 100

# ------------------------
# PROCESS FILES
# ------------------------
results = []

for msa_file in sorted(msa_dir.glob("*_aligned*.fasta")):
    sequences = list(SeqIO.parse(msa_file, "fasta"))
    if not sequences:
        continue

    n_seqs = len(sequences)
    aln_len = sequences[0].seq.__len__()

    # Trasponi sequenze in colonne
    columns = []
    for i in range(aln_len):
        col = [str(seq.seq[i]) for seq in sequences]
        columns.append(col)

    cons_per_col = [calc_conservation(col) for col in columns]
    entropy_per_col = [calc_entropy(col) for col in columns]
    gap_per_col = [calc_gap_fraction(col) for col in columns]

    mean_cons = sum(cons_per_col) / aln_len
    mean_entropy = sum(entropy_per_col) / aln_len
    mean_gap = sum(gap_per_col) / aln_len

    run_id = msa_file.stem.replace("_aligned", "")
    results.append({
        "run": run_id,
        "n_seqs": n_seqs,
        "aln_len": aln_len,
        "mean_conservation": mean_cons,
        "mean_entropy": mean_entropy,
        "mean_gap": mean_gap
    })

# ------------------------
# WRITE OUTPUT
# ------------------------
with open(output_file, "w") as fout:
    header = ["run","n_seqs","aln_len","mean_conservation","mean_entropy","mean_gap"]
    fout.write("\t".join(header) + "\n")
    for r in results:
        fout.write("\t".join(str(r[h]) for h in header) + "\n")

print(f"✅ Analisi completata. Risultati salvati in {output_file}")
