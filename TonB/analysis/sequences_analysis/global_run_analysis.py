import os
import subprocess
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter
import sys

# ---------------------------------------------
# ⚙️ SELETTORE MODALITÀ
# ---------------------------------------------
MODE = "NORMAL"  # Opzioni: "NORMAL" oppure "OPTIMIZED"

# ---------------------------------------------
# CONFIGURAZIONE DINAMICA
# ---------------------------------------------
if MODE == "NORMAL":
    MSA_DIR = "./msa_aln"
    OUTPUT_DIR = "./consensus_analysis"
    print(f"🔵 GLOBAL CONSENSUS: MODALITÀ NORMAL")
    print(f"   Input: {MSA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")

elif MODE == "OPTIMIZED":
    MSA_DIR = "./msa_aln_opt"
    OUTPUT_DIR = "./consensus_analysis_opt"
    print(f"🚀 GLOBAL CONSENSUS: MODALITÀ OPTIMIZED")
    print(f"   Input: {MSA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")

else:
    print("ERRORE: Mode non valido. Usa 'NORMAL' o 'OPTIMIZED'")
    sys.exit()

SEQ_WT_NAME = "TonB_WT"
SEQ_WT_STR = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------
# 1. FUNZIONE CONSENSUS (ROBUST)
# ---------------------------------------------
def calculate_robust_consensus(aln_path, min_support=0.1):
    """
    Calcola il consensus IGNORANDO i gap, utile per evitare che 
    sequenze parziali rovinino il consensus.
    """
    try:
        alignment = AlignIO.read(aln_path, "fasta")
        length = alignment.get_alignment_length()
        consensus = []

        for i in range(length):
            column = alignment[:, i]
            # Filtra gap e caratteri ambigui
            clean_column = [aa for aa in column if aa not in ['-', 'X', '.', '?']]
            
            if not clean_column:
                consensus.append("") 
            else:
                counts = Counter(clean_column)
                most_common_aa, count = counts.most_common(1)[0]
                consensus.append(most_common_aa)
                
        return "".join(consensus)
        
    except Exception as e:
        print(f"⚠️ Errore lettura {aln_path}: {e}")
        return None

# ---------------------------------------------
# 2. ESTRAZIONE E SUPER-ALLINEAMENTO
# ---------------------------------------------

def main():
    print("-" * 60)
    print("--- 1. Estrazione Consensus dalle Run ---")
    
    # Check input dir
    if not os.path.exists(MSA_DIR):
        print(f"ERRORE: La cartella {MSA_DIR} non esiste. Esegui prima lo Step 1 e 2.")
        return

    records_to_align = []

    # Aggiungi WT (Riferimento assoluto)
    records_to_align.append(SeqRecord(Seq(SEQ_WT_STR), id=SEQ_WT_NAME, description="Wild Type"))

    # Trova file allineati
    aln_files = [f for f in os.listdir(MSA_DIR) if f.endswith(".aln") or f.endswith(".fasta")]
    
    if not aln_files:
        print("Nessun file .aln trovato.")
        return

    print(f"Processando {len(aln_files)} file di allineamento...")
    
    count_valid = 0
    for f in aln_files:
        path = os.path.join(MSA_DIR, f)
        clean_cons = calculate_robust_consensus(path)
        
        if clean_cons:
            run_id = f.split('.')[0] 
            # Filtro lunghezza minima per evitare consensus vuoti
            if len(clean_cons) > 50:
                records_to_align.append(SeqRecord(Seq(clean_cons), id=f"Run_{run_id}", description=""))
                count_valid += 1
            else:
                print(f"   -> Skip {run_id}: Consensus troppo corto ({len(clean_cons)} aa)")

    print(f"Recuperate {count_valid} sequenze consensus valide + 1 WT.")

    # Salvataggio pre-allineamento
    unaligned_path = os.path.join(OUTPUT_DIR, "unaligned.fasta")
    SeqIO.write(records_to_align, unaligned_path, "fasta")

    # Esecuzione FAMSA (Super Alignment)
    aligned_path = os.path.join(OUTPUT_DIR, "super_alignment.fasta")
    print("--- 2. Esecuzione FAMSA (Super Alignment) ---")
    
    try:
        famsa_cmd = ["famsa", unaligned_path, aligned_path]
        subprocess.run(famsa_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Super allineamento salvato in: {aligned_path}")
    except FileNotFoundError:
        print("ERRORE: FAMSA non trovato. Assicurati che sia installato o nel PATH.")
        return
    except subprocess.CalledProcessError:
        print("ERRORE: FAMSA ha fallito l'allineamento.")
        return

    # ---------------------------------------------
    # 3. MAPPING E ANALISI STATISTICA
    # ---------------------------------------------
    print("--- 3. Analisi Statistica e Mapping su WT ---")
    alignment = AlignIO.read(aligned_path, "fasta")

    # Trova indice riga WT
    wt_idx = -1
    for i, rec in enumerate(alignment):
        if rec.id == SEQ_WT_NAME:
            wt_idx = i
            break
            
    if wt_idx == -1:
        print("ERRORE CRITICO: WT persa nell'allineamento!")
        return

    mapped_data = []
    wt_residue_counter = 0
    n_cols = alignment.get_alignment_length()

    for col_idx in range(n_cols):
        col_residues = alignment[:, col_idx]
        wt_aa = col_residues[wt_idx]
        
        # Mappa SOLO se c'è un residuo nel WT (ignora inserzioni globali rispetto al WT)
        if wt_aa != "-":
            wt_residue_counter += 1
            
            # Estrai tutti gli AA delle run in questa colonna
            run_aas = [aa for i, aa in enumerate(col_residues) if i != wt_idx]
            valid_aas = [aa for aa in run_aas if aa != "-"]
            
            n_runs = len(run_aas)
            if n_runs == 0: continue # Evita division by zero

            if valid_aas:
                most_common = Counter(valid_aas).most_common(1)[0]
                top_aa = most_common[0]
                support_abs = most_common[1]
            else:
                top_aa = "-" 
                support_abs = 0
                
            support_pct = (support_abs / n_runs) * 100
            gap_pct = (run_aas.count("-") / n_runs) * 100
            
            # Calcolo Entropia Shannon
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
                # Mutazione rilevante: diversa da WT, non è un gap, supporto decente
                "Is_Mutated": (top_aa != wt_aa) and (top_aa != "-") and (support_pct > 30)
            })

    # Salvataggio CSV
    df_mapped = pd.DataFrame(mapped_data)
    csv_out = os.path.join(OUTPUT_DIR, "consensus_mapped.csv")
    df_mapped.to_csv(csv_out, index=False)

    print(f"Analisi completata. Dati salvati in:")
    print(f"📄 {csv_out}")
    print("-" * 60)

if __name__ == "__main__":
    main()
