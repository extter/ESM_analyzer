from collections import defaultdict
from Bio import SeqIO
from pathlib import Path

input_file = Path("./extracted/normalized_best_than_0.98000.txt")
output_dir = Path("./extracted/run_msas")
output_dir.mkdir(exist_ok=True)

runs = defaultdict(list)

for record in SeqIO.parse(input_file, "fasta"):
    # estrai run_id dal header: include data + ora
    # esempio header: ">time=2026-02-08 15:34:45 step=5019 cosine_to_tonb=0.98223 length=242"
    run_id = record.description.split(" ")[0].replace("time=", "") + " " + record.description.split(" ")[1]
    runs[run_id].append(record)

# Scrivi file per run (timestamp intero, safe per filesystem)
for run_id, seqs in runs.items():
    safe_run_id = run_id.replace(" ", "_").replace(":", "")
    run_file = output_dir / f"run_{safe_run_id}.fasta"
    SeqIO.write(seqs, run_file, "fasta")
    print(f"Run {run_id}: {len(seqs)} sequenze scritte in {run_file}")



#for f in ./extracted/run_msas/run_*.fasta; do
#    mafft --auto --thread -1 "$f" > "${f%.fasta}_aligned.fasta"
#done
