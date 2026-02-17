
import os
import glob
import random
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

# ------------------------------------
# SELETTORE MODALITÀ 
# ------------------------------------
MODE = "OPTIMIZED"  # Opzioni: "NORMAL" oppure "OPTIMIZED"

# ------------------------------------
# CONFIGURAZIONE DINAMICA
# ------------------------------------
N_SAMPLE = 300
MIN_SEQ_RUN = 300  

if MODE == "NORMAL":
    RUNS_DIR = '../../markov/runs'
    FASTA_DIR = Path('./msa_fasta')
    ALN_DIR = Path('./msa_aln')
    THRESHOLD = 0.975
    print(f"MODALITÀ NORMAL ATTIVA (Input: {RUNS_DIR} | Soglia: {THRESHOLD})")

elif MODE == "OPTIMIZED":
    RUNS_DIR = '../../runs_ultra_optimized'
    FASTA_DIR = Path('./msa_fasta_opt')
    ALN_DIR = Path('./msa_aln_opt')
    THRESHOLD = 0.995
    print(f"MODALITÀ OPTIMIZED ATTIVA (Input: {RUNS_DIR} | Soglia: {THRESHOLD})")

else:
    raise ValueError("Mode non valido. Usa 'NORMAL' o 'OPTIMIZED'")

# ----------------------------------------------------
# LOGICA DI PARSING 
# ----------------------------------------------------
def parse_txt_run(fname, chain_id):
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
    df_all = []
    print(f"Parsing .txt da {RUNS_DIR}...")
    for run_folder in tqdm(os.listdir(RUNS_DIR), desc="Run Parsing"):
        txt_path = os.path.join(RUNS_DIR, run_folder, '*.txt')
        txt_files = glob.glob(txt_path)
        if not txt_files: continue
        
        chain_id = os.path.basename(run_folder)
        seq_data = parse_txt_run(txt_files[0], chain_id)
        if seq_data:
            df_all.append(pd.DataFrame(seq_data))
    
    if not df_all: return []
    df_total = pd.concat(df_all, ignore_index=True)
    
    # Filtro runs con poche sequenze
    counts = df_total['chain_id'].value_counts()
    good_runs = counts[counts >= MIN_SEQ_RUN].index
    
    fasta_files = []
    Path(FASTA_DIR).mkdir(exist_ok=True)
    
    for chain in good_runs:
        df_run = df_total[df_total['chain_id'] == chain]
        sample = df_run.sample(min(N_SAMPLE, len(df_run)), random_state=42)
        fa_path = FASTA_DIR / f"{chain}.fa"
        with open(fa_path, 'w') as f:
            for _, row in sample.iterrows():
                f.write(f">{chain}_{row['step']}_cos{row['cosine']:.3f}\n{row['sequence']}\n")
        fasta_files.append(fa_path)
    return fasta_files

# ----------------------------------------------------
# LOGICA DI PARSING (OPTIMIZED - FASTA)
# ----------------------------------------------------
def process_optimized_runs():
    Path(FASTA_DIR).mkdir(exist_ok=True)
    run_folders = [f for f in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, f))]
    fasta_files = []

    print(f"Parsing best_candidates.fasta da {RUNS_DIR}...")
    for run_folder in tqdm(run_folders, desc="Run Parsing"):
        input_fasta = os.path.join(RUNS_DIR, run_folder, "best_candidates.fasta")
        if not os.path.exists(input_fasta): continue
        
        seqs = list(SeqIO.parse(input_fasta, "fasta"))
        if not seqs: continue
        
        # Sampling se necessario
        selected = random.sample(seqs, N_SAMPLE) if len(seqs) > N_SAMPLE else seqs
        
        out_path = FASTA_DIR / f"{run_folder}.fa"
        with open(out_path, "w") as f_out:
            SeqIO.write(selected, f_out, "fasta")
        fasta_files.append(out_path)
    return fasta_files

# ----------------------------------------------------
# FAMSA RUNNER (COMUNE)
# ----------------------------------------------------
def run_famsa(fasta_files):
    Path(ALN_DIR).mkdir(exist_ok=True)
    aln_files = []
    for fa in tqdm(fasta_files, desc="FAMSA Alignment"):
        aln_path = ALN_DIR / f"{fa.stem}.aln"
        if not aln_path.exists():
            subprocess.run(['famsa', str(fa), str(aln_path)], capture_output=True)
        aln_files.append(aln_path)
    return aln_files

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    if MODE == "NORMAL":
        fasta_files = process_normal_runs()
    else:
        fasta_files = process_optimized_runs()
        
    print(f"Generati {len(fasta_files)} file FASTA.")
    
    aln_files = run_famsa(fasta_files)
    print(f"Allineamenti completati in: {ALN_DIR}")

if __name__ == "__main__":
    main()
