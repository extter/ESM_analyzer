
import os
import glob
import random
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

# ----------------------------------
# SELETTORE MODALITÀ
# ----------------------------------
MODE = "OPTIMIZED"  # Imposta su "NORMAL" o "OPTIMIZED"

# ----------------------------------
# CONFIGURAZIONE
# ----------------------------------
N_SAMPLE = 300       
MIN_SEQ_RUN = 300 

if MODE == "NORMAL":
    RUNS_DIR = '../../markov/runs'
    FASTA_DIR = Path('./msa_fasta')
    ALN_DIR = Path('./msa_aln')
    THRESHOLD = 0.975
    print(f"MODALITÀ NORMAL ATTIVA\nInput: {RUNS_DIR}\nSoglia Cosine: {THRESHOLD}")

elif MODE == "OPTIMIZED":
    RUNS_DIR = '../../markov/runs_ultra_optimized'
    FASTA_DIR = Path('./msa_fasta_opt')
    ALN_DIR = Path('./msa_aln_opt')
    THRESHOLD = 0.994 # ATTENZIONE: Assicurati che le run raggiungano questo valore!
    print(f"MODALITÀ OPTIMIZED ATTIVA\nInput: {RUNS_DIR}\nSoglia Cosine: {THRESHOLD}")

else:
    raise ValueError("Mode non valido. Usa 'NORMAL' o 'OPTIMIZED'")

# ----------------------------------
# LOGICA PARSING - OPTIMIZED (.fasta)
# ----------------------------------
def parse_fasta_optimized_run(fasta_path, run_id):
    """
    Legge best_candidates.fasta con header: >step_30_sim_0.983025 Mut M218C
    """
    valid_records = []
    
    # BioPython legge l'ID fino al primo spazio: "step_30_sim_0.983025"
    records = list(SeqIO.parse(fasta_path, "fasta"))
    
    for rec in records:
        try:
            # Splitta l'ID: ['step', '30', 'sim', '0.983025']
            parts = rec.id.split('_')
            
            if 'sim' in parts:
                idx = parts.index('sim')
                # Prende l'elemento subito dopo 'sim'
                score_str = parts[idx+1]
                score = float(score_str)
                
                # FILTRO SOGLIA
                if score >= THRESHOLD:
                    # Rinomina per unicità nel file finale
                    rec.id = f"{run_id}_{rec.id}"
                    rec.description = "" # Pulisce la descrizione
                    valid_records.append(rec)
                    
        except (ValueError, IndexError):
            continue
            
    return valid_records

def process_optimized_runs():
    """Cicla su tutte le cartelle seed, filtra e campiona"""
    Path(FASTA_DIR).mkdir(exist_ok=True)
    
    # Trova le sottocartelle (seed0_..., seed1_...)
    run_folders = [f for f in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, f))]
    print(f"Trovate {len(run_folders)} cartelle run.")
    
    fasta_files = []
    total_seqs_extracted = 0

    for run_folder in tqdm(run_folders, desc="Parsing Run"):
        input_fasta = os.path.join(RUNS_DIR, run_folder, "best_candidates.fasta")
        
        if not os.path.exists(input_fasta):
            continue
        
        # 1. Estrazione (Filtro Threshold)
        valid_seqs = parse_fasta_optimized_run(input_fasta, run_folder)
        
        if not valid_seqs:
            # Se la lista è vuota, significa che nessuna seq ha superato 0.995
            continue
            
        # 2. Campionamento (Random 300)
        if len(valid_seqs) > N_SAMPLE:
            selected_seqs = random.sample(valid_seqs, N_SAMPLE)
        else:
            selected_seqs = valid_seqs
            
        total_seqs_extracted += len(selected_seqs)

        # 3. Scrittura FASTA pulito per FAMSA
        out_path = FASTA_DIR / f"{run_folder}.fa"
        with open(out_path, "w") as f_out:
            SeqIO.write(selected_seqs, f_out, "fasta")
            
        fasta_files.append(out_path)
    
    print(f"Totale sequenze estratte (sopra {THRESHOLD}): {total_seqs_extracted}")
    return fasta_files

# ----------------------------------
# LOGICA PARSING - NORMAL (.txt)
# ----------------------------------
def parse_txt_run(fname, chain_id):
    """Legge i vecchi file .txt"""
    seq_data = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('>step='):
            header = lines[i].strip()[1:]
            parts = header.split()
            cosine_str = next((p.split('=')[1] for p in parts if 'cosine_to_tonb=' in p), None)
            
            if cosine_str:
                try:
                    cosine = float(cosine_str)
                    if cosine >= THRESHOLD:
                        i += 1
                        if i < len(lines):
                            seq_data.append({
                                'chain_id': chain_id,
                                'sequence': lines[i].strip(),
                                'cosine': cosine,
                                'step': parts[0].split('=')[1]
                            })
                except ValueError: pass
        i += 1
    return seq_data

def process_normal_runs():
    Path(FASTA_DIR).mkdir(exist_ok=True)
    df_all = []
    
    run_folders = os.listdir(RUNS_DIR)
    for run_folder in tqdm(run_folders, desc="Parsing Run"):
        txt_path = os.path.join(RUNS_DIR, run_folder, '*.txt')
        txt_files = glob.glob(txt_path)
        if not txt_files: continue
        
        chain_id = os.path.basename(run_folder)
        seq_data = parse_txt_run(txt_files[0], chain_id)
        if seq_data:
            df_all.append(pd.DataFrame(seq_data))
    
    if not df_all: return []

    df_total = pd.concat(df_all, ignore_index=True)
    counts = df_total['chain_id'].value_counts()
    good_runs = counts[counts >= MIN_SEQ_RUN].index
    
    fasta_files = []
    for chain in tqdm(good_runs, desc="Generating FASTA"):
        df_run = df_total[df_total['chain_id'] == chain]
        sample = df_run.sample(min(N_SAMPLE, len(df_run)), random_state=42)
        
        fa_path = FASTA_DIR / f"{chain}.fa"
        with open(fa_path, 'w') as f:
            for _, row in sample.iterrows():
                f.write(f">{chain}_{row['step']}_cos{row['cosine']:.3f}\n{row['sequence']}\n")
        fasta_files.append(fa_path)
    return fasta_files

# ----------------------------------
# ESECUZIONE FAMSA
# ----------------------------------
def run_famsa(fasta_files):
    Path(ALN_DIR).mkdir(exist_ok=True)
    aln_files = []
    skipped = 0
    
    for fa in tqdm(fasta_files, desc="Running FAMSA"):
        aln_path = ALN_DIR / f"{fa.stem}.aln"
        
        if aln_path.exists():
            skipped += 1
            aln_files.append(aln_path)
            continue
            
        # -v off per pulizia, se vuoi log usa check=False
        subprocess.run(['famsa', str(fa), str(aln_path)], capture_output=True)
        
        if aln_path.exists():
            aln_files.append(aln_path)

    print(f"Allineamenti generati: {len(aln_files)} ({skipped} già esistenti)")
    return aln_files

# ----------------------------------
# MAIN
# ----------------------------------
def main():
    print(f"--- STEP 1: PARSING & FILTERING ({MODE}) ---")
    
    if MODE == "NORMAL":
        fasta_files = process_normal_runs()
    else:
        fasta_files = process_optimized_runs()
        
    print(f"File FASTA pronti: {len(fasta_files)} in {FASTA_DIR}")
    
    if len(fasta_files) == 0:
        print("⚠️  ATTENZIONE: Nessun file generato. Controlla che i file esistano e che raggiungano la SOGLIA impostata.")
        return

    print(f"\n--- STEP 2: ALLINEAMENTO FAMSA ---")
    run_famsa(fasta_files)
    
    print("\n✅ PIPELINE 1 COMPLETATA.")

if __name__ == "__main__":
    main()
