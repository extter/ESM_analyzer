
import os
import glob
import random
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm


# ----------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------
THRESHOLD = 0.975
N_SAMPLE = 300
MIN_SEQ_RUN = 300

RUNS_DIR = '../../markov/runs'
FASTA_DIR = Path('./msa_fasta')
ALN_DIR = Path('./msa_aln')


# ----------------------------------------------------
# STEP 1: PARSING + FASTA GENERATION
# ----------------------------------------------------
def parse_all_runs():
    """Parse tutte run → df_all con seq >THRESHOLD"""
    df_all = []
    
    print(f"Parsing run da {RUNS_DIR}/...")
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
    
    df_total = pd.concat(df_all, ignore_index=True) if df_all else pd.DataFrame()
    print(f"{len(df_total):,} sequenze totali >{THRESHOLD}")
    return df_total


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


def generate_fastas(df_all):
    """Genera .fa per ogni run valida"""
    Path(FASTA_DIR).mkdir(exist_ok=True)
    
    counts = df_all['chain_id'].value_counts()
    good_runs = counts[counts >= MIN_SEQ_RUN].index
    print(f"Run valide (>= {MIN_SEQ_RUN} seq): {len(good_runs)}")
    
    fasta_files = []
    for chain in tqdm(good_runs, desc="Generazione FASTA"):
        df_run = df_all[df_all['chain_id'] == chain]
        n_total = len(df_run)
        n = min(N_SAMPLE, n_total)
        sample = df_run.sample(n, random_state=42)
        
        fa_path = FASTA_DIR / f"{chain}.fa"
        with open(fa_path, 'w') as f:
            for _, row in sample.iterrows():
                header = f">{chain}_{row['step']}_cos{row['cosine']:.3f}"
                f.write(f"{header}\n{row['sequence']}\n")
        
        fasta_files.append(fa_path)
        print(f"   {chain}: {n_total} → {n} seq")
    
    return fasta_files


# ----------------------------------------------------
# STEP 2: FAMSA MSA
# ----------------------------------------------------
def run_famsa(fasta_files):
    """Esegue FAMSA su tutti i FASTA"""
    Path(ALN_DIR).mkdir(exist_ok=True)
    
    aln_files = []
    skipped = 0
    
    for fa in tqdm(fasta_files, desc="FAMSA MSA"):
        aln_path = ALN_DIR / f"{fa.stem}.aln"
        
        if aln_path.exists():
            print(f"Skip {fa.name}: già allineato")
            aln_files.append(aln_path)
            skipped += 1
            continue
        
        print(f"Allineo {fa.name}...")
        result = subprocess.run(
            ['famsa', '-v', str(fa), str(aln_path)], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            aln_files.append(aln_path)
            print(f"{fa.stem}.aln creato")
        else:
            print(f"Errore FAMSA: {result.stderr[:200]}...")
    
    print(f"{len(aln_files)} allineamenti | {skipped} saltati")
    return aln_files


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    
    # Step 1: Parsing + FASTA
    print("STEP 1: PARSING + FASTA")
    df_all = parse_all_runs()
    
    fasta_files = generate_fastas(df_all)
    print(f"FASTA salvati: {FASTA_DIR}/ ({len(fasta_files)} file)")
    
    # Step 2: MSA
    print("STEP 2: FAMSA ALIGNMENT")
    aln_files = run_famsa(fasta_files)
    print(f"MSA salvati: {ALN_DIR}/ ({len(aln_files)} file)")
    
    print("PIPELINE COMPLETATA!")


if __name__ == "__main__":
    main()
