from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from collections import Counter

# ------------------------
# CONFIG
# ------------------------
aligned_dir = Path("./extracted/run_msas")   # cartella con file *_aligned.fasta
tonb_file = Path("../sequences/tonb.txt")               # file con sequenza di TonB in fasta o txt
output_csv = Path("./consensus_vs_tonb.csv")

# ------------------------
# CARICO SEQUENZA TONB
# ------------------------
with open(tonb_file) as f:
    tonb_seq = "".join(line.strip() for line in f if not line.startswith(">"))

print(f"TonB length: {len(tonb_seq)}")

# ------------------------
# FUNZIONE CONSENSUS
# ------------------------
def consensus_sequence(seqs):
    """Restituisce la sequenza consenso a partire da una lista di sequenze (allineate, uguale lunghezza)."""
    if not seqs:
        return ""
    aln_len = len(seqs[0])
    consensus = []
    for i in range(aln_len):
        column = [s[i] for s in seqs]
        most_common, count = Counter(column).most_common(1)[0]
        consensus.append(most_common)
    return "".join(consensus)

# ------------------------
# PROCESSO FILE
# ------------------------
import csv

results = []

for aln_file in aligned_dir.glob("*_aligned.fasta"):
    run_id = aln_file.stem.replace("_aligned", "")
    seqs = [str(rec.seq) for rec in SeqIO.parse(aln_file, "fasta")]

    if not seqs:
        continue

    # calcolo consenso
    cons = consensus_sequence(seqs)

    # calcolo percentuale identità con TonB
    # confronto solo fino alla lunghezza minima
    min_len = min(len(cons), len(tonb_seq))
    matches = sum(1 for a, b in zip(cons[:min_len], tonb_seq[:min_len]) if a == b)
    identity_pct = matches / min_len * 100

    results.append({
        "run": run_id,
        "n_seqs": len(seqs),
        "aln_len": len(seqs[0]),
        "consensus_len": len(cons),
        "identity_vs_tonb": identity_pct
    })
    print(f"{run_id}: {len(seqs)} seqs, consensus_len={len(cons)}, identity vs TonB={identity_pct:.2f}%")

# ------------------------
# SCRIVO CSV
# ------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Finito! Risultati salvati in {output_csv}")
