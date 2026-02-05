from Bio import AlignIO
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# VERY IMPORTANT: YOU HAVE TO RUN THE FOLLOWING COMMAND ON THE BASH BEFORE RUNNING THIS SCRIPT:
# mafft --auto --thread -1 filtered_cosine_gt_095.fasta > aligned_cosine_gt_095.fasta


# Leggi l'allineamento
alignment = AlignIO.read("aligned_cosine_gt_095.fasta", "fasta") # aggiustare il path con il nome della cartella giusta in RUNS
L = alignment.get_alignment_length()

consensus = ""
conservation = []

for i in range(L):
    # Prendi la colonna i
    column = alignment[:, i]
    # Conta solo residui veri (ignora gap '-')
    counts = Counter([aa for aa in column if aa != "-"])
    
    if counts:
        # amminoacido più frequente
        aa, freq = counts.most_common(1)[0]
        consensus += aa
        conservation.append(freq / sum(counts.values()))  # frazione di sequenze che hanno quel residuo
    else:
        consensus += "-"
        conservation.append(0)

print("Consensus:")
print(consensus)

print("\nConservation per posizione (frazione di sequenze che hanno il residuo consensus):")
print(np.round(conservation, 2))


tonb_seq = "METTI_LA_TUA_SEQUENZA_TONB_QUI"  # sequenza reale di TonB
consensus_seq = consensus  # dal passo precedente

with open("tonb_vs_consensus.fasta", "w") as f:
    f.write(">TONB_reference\n")
    f.write(tonb_seq + "\n")
    f.write(">Consensus\n")
    f.write(consensus_seq + "\n")

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# Allineamento globale (Needleman-Wunsch)
alignments = pairwise2.align.globalxx(tonb_seq, consensus_seq)  # 'xx' = match=1, mismatch=0

# Prendi il migliore
best_alignment = alignments[0]

print(format_alignment(*best_alignment))


aligned_tonb, aligned_consensus, score, start, end = best_alignment

# mappa conservation
aligned_conservation = []
c_idx = 0
for aa in aligned_consensus:
    if aa == "-":
        aligned_conservation.append(0)
    else:
        aligned_conservation.append(conservation[c_idx])
        c_idx += 1

# Stampa esempio
for i in range(len(aligned_tonb)):
    print(f"{aligned_tonb[i]} {aligned_consensus[i]} {aligned_conservation[i]:.2f}")



# Crea un array per i mismatch
# 0 = match, 1 = mismatch o gap
mismatch = np.array([0 if a == b else 1 for a, b in zip(aligned_tonb, aligned_consensus)])

# Trasforma aligned_conservation in numpy array
aligned_conservation = np.array(aligned_conservation)

# Combina mismatch e conservation in una matrice 2xL (due righe: mismatch, conservation)
heatmap_matrix = np.vstack([mismatch, aligned_conservation])

# Plot
plt.figure(figsize=(15,3))
plt.imshow(heatmap_matrix, aspect='auto', cmap='coolwarm', interpolation='nearest')

plt.yticks([0,1], ['Mismatch', 'Conservation'])
plt.xlabel("Position in alignment")
plt.colorbar(label="Mismatch / Conservation")
plt.title("Consensus vs TONB heatmap")
plt.savefig("tonb_vs_consensus_heatmap.png", dpi=300)
plt.show()
