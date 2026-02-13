import os
import glob
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
THRESHOLD = 0.975
N_SAMPLE = 300
MIN_SEQ_RUN = 300
RUNS_DIR = '../../markov/runs'
FASTA_DIR = './msa_fasta'

# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(FASTA_DIR).mkdir(exist_ok=True)
    
    df_all = parse_all_runs()
    
    print(f"{len(df_all):,} seq totali >{THRESHOLD}")
    
    good_runs = select_good_runs(df_all)
    print(f"Run valide (>= {MIN_SEQ_RUN} seq): {len(good_runs)}")
    
    generate_fastas(good_runs, df_all)
    print(f" {len(list(Path(FASTA_DIR).glob('*.fa')))} FASTA pronti!")
    print(f"{FASTA_DIR}/")
    
def parse_all_runs():
    """Parse tutte run → df_all con seq >THRESHOLD"""
    df_all = []
    
    for run_folder in tqdm(os.listdir(RUNS_DIR), desc="Run"):
        txt_path = os.path.join(RUNS_DIR, run_folder, '*.txt')
        txt_files = glob.glob(txt_path)
        
        if not txt_files:
            continue
            
        fname = txt_files[0]
        chain_id = os.path.basename(run_folder)
        
        seq_data = parse_txt_file(fname, chain_id)
        if seq_data:
            df_run = pd.DataFrame(seq_data)
            df_all.append(df_run)
    
    return pd.concat(df_all, ignore_index=True) if df_all else pd.DataFrame()

def parse_txt_file(fname, chain_id):
    """Parse singolo .txt → seq_data >THRESHOLD"""
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
                            sequence = lines[i].strip()
                            seq_data.append({
                                'chain_id': chain_id,
                                'sequence': sequence,
                                'cosine': cosine,
                                'step': parts[0].split('=')[1]
                            })
                except ValueError:
                    pass
        
        i += 1
    
    return seq_data

def select_good_runs(df_all):
    """Filtra run con >= MIN_SEQ_RUN seq"""
    counts = df_all['chain_id'].value_counts()
    return counts[counts >= MIN_SEQ_RUN].index.tolist()

def generate_fastas(good_runs, df_all):
    """Genera .fa per ogni run valida"""
    for chain in tqdm(good_runs, desc="FASTA"):
        df_run = df_all[df_all['chain_id'] == chain]
        n = min(N_SAMPLE, len(df_run))
        sample = df_run.sample(n, random_state=42)
        
        fa_path = Path(FASTA_DIR) / f"{chain}.fa"
        with open(fa_path, 'w') as f:
            for _, row in sample.iterrows():
                header = f">{chain}_{row['step']}_cos{row['cosine']:.3f}"
                f.write(f"{header}\n{row['sequence']}\n")
        
        print(f"{chain}.fa: {n}")

if __name__ == "__main__":
    main()