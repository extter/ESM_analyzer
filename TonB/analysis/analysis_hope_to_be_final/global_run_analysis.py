import os
import subprocess
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter

# ---------------------
# CONFIGURAZIONE 
# ---------------------

MSA_DIR = "./msa_aln"              
OUTPUT_DIR = "./consensus_analysis"
SEQ_WT_NAME = "TonB_WT"
SEQ_WT_STR = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# 1. FUNZIONE CONSENSUS 
# ----------------------


def calculate_robust_consensus(aln_path, min_support=0.1):
    """
    Calcola il consensus IGNORANDO i gap (-, X o .), a meno che non siano la quasi totalità.
    Questo evita che le estremità vengano 'mangiate'.
    
    min_support=0.1: Basta che il 10% delle sequenze abbia un AA per considerarlo.
    """
    try:
        alignment = AlignIO.read(aln_path, "fasta")
        length = alignment.get_alignment_length()
        consensus = []

        for i in range(length):
            column = alignment[:, i]
            clean_column = [aa for aa in column if aa not in ['-', 'X', '.']]
            
            if not clean_column:
                consensus.append("") 
            else:
                counts = Counter(clean_column)
                most_common_aa, count = counts.most_common(1)[0]
                consensus.append(most_common_aa)
                
        return "".join(consensus)
        
    except Exception as e:
        print(f"Errore lettura {aln_path}: {e}")
        return None

# ----------------------
# 2. ESTRAZIONE 
# ----------------------


print("--- 1. Estrazione Consensus ---")
records_to_align = []

# WT
records_to_align.append(SeqRecord(Seq(SEQ_WT_STR), id=SEQ_WT_NAME, description=""))

aln_files = [f for f in os.listdir(MSA_DIR) if f.endswith(".aln") or f.endswith(".fasta")]

for f in aln_files:
    path = os.path.join(MSA_DIR, f)
    clean_cons = calculate_robust_consensus(path)
    
    if clean_cons:
        run_id = f.split('.')[0] 
        if len(clean_cons) > 50:
            records_to_align.append(SeqRecord(Seq(clean_cons), id=f"Run_{run_id}", description=""))
        else:
            print(f"Warning: Consensus per {run_id} troppo corto ({len(clean_cons)})")

print(f"Recuperate {len(records_to_align)-1} sequenze consensus valide.")

unaligned_path = os.path.join(OUTPUT_DIR, "unaligned.fasta")
SeqIO.write(records_to_align, unaligned_path, "fasta")

# --------------------------
# 3. ALLINEAMENTO FAMSA 
# --------------------------


aligned_path = os.path.join(OUTPUT_DIR, "super_alignment.fasta")
print("--- 2. Riesecuzione FAMSA ---")

famsa_cmd = ["famsa", unaligned_path, aligned_path]
subprocess.run(famsa_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"Nuovo allineamento salvato in: {aligned_path}")

# ----------------------
# 4. MAPPING E ANALISI 
# ----------------------


print("--- 3. Analisi e Mapping ---")
alignment = AlignIO.read(aligned_path, "fasta")

wt_idx = -1
for i, rec in enumerate(alignment):
    if rec.id == SEQ_WT_NAME:
        wt_idx = i
        break

mapped_data = []
wt_residue_counter = 0
n_cols = alignment.get_alignment_length()

for col_idx in range(n_cols):
    col_residues = alignment[:, col_idx]
    wt_aa = col_residues[wt_idx]
    
    if wt_aa != "-":
        wt_residue_counter += 1
        
        run_aas = [aa for i, aa in enumerate(col_residues) if i != wt_idx]
        valid_aas = [aa for aa in run_aas if aa != "-"]
        
        if valid_aas:
            most_common = Counter(valid_aas).most_common(1)[0]
            top_aa = most_common[0]
            support_abs = most_common[1]
        else:
            top_aa = "-" 
            support_abs = 0
            
        n_runs = len(run_aas)
        support_pct = (support_abs / n_runs) * 100
        gap_pct = (run_aas.count("-") / n_runs) * 100
        
        # Entropy
        entropy = 0
        if valid_aas:
            counts = Counter(valid_aas)
            total = len(valid_aas)
            for k in counts:
                p = counts[k] / total
                entropy -= p * np.log2(p)
        
        mapped_data.append({
            "WT_Pos": wt_residue_counter,
            "WT_AA": wt_aa,
            "Consensus_Global_AA": top_aa,
            "Support_Pct": round(support_pct, 2),
            "Gap_Pct": round(gap_pct, 2),
            "Entropy_Bits": round(entropy, 3),
            "Is_Conserved": (top_aa == wt_aa) and (gap_pct < 50),
            "Is_Mutated": (top_aa != wt_aa) and (top_aa != "-") and (support_pct > 30) # Soglia abbassata a 30% per vedere segnali deboli
        })

df_mapped = pd.DataFrame(mapped_data)
csv_out = os.path.join(OUTPUT_DIR, "consensus_mapped.csv")
df_mapped.to_csv(csv_out, index=False)

print(f"Analisi completata. Controlla {csv_out}")
