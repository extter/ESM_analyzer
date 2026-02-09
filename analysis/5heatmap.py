import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
aligned_dir = Path("./extracted/run_msas")  # cartella con i file _aligned.fasta
output_dir = aligned_dir / "heatmaps"
output_dir.mkdir(exist_ok=True)

aa_list = list("ACDEFGHIKLMNPQRSTVWY")  # aminoacidi standard

# -------------------------------
# LOOP SU FILE
# -------------------------------
for aligned_file in aligned_dir.glob("*_aligned.fasta"):
    seqs = [str(rec.seq) for rec in SeqIO.parse(aligned_file, "fasta")]
    n_seqs = len(seqs)
    aln_len = len(seqs[0])

    # costruisci matrice di frequenza
    freq_matrix = pd.DataFrame(0, index=aa_list, columns=range(1, aln_len+1))

    for seq in seqs:
        for i, aa in enumerate(seq):
            if aa in aa_list:
                freq_matrix.at[aa, i+1] += 1

    freq_matrix = freq_matrix / n_seqs

    # plot heatmap
    plt.figure(figsize=(20,6))
    plt.imshow(freq_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Frequenza')
    plt.yticks(range(len(aa_list)), aa_list)
    plt.xlabel("Posizione nell'allineamento")
    plt.ylabel("Aminoacido")
    plt.title(f"Frequenze AA - {aligned_file.stem}")
    plt.tight_layout()

    # salva
    output_file = output_dir / f"{aligned_file.stem}_heatmap.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Heatmap salvata: {output_file}")
