import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict

# === PARAMETRI ===
MIN_SEQ_PER_CHAIN = 100  # minimo seq >0.97 per includere la catena
N_SAMPLE_PER_CHAIN = 400  # quante seq campionare per catena
CUTOFF_COSINE = 0.97     # soglia per "buone"

# === 1. Scansiona /runs e carica TUTTE le seq >0.97 ===
df_all = []
runs_folder = '../../markov/runs'  # adatta al tuo path reale

for run_folder in os.listdir(runs_folder):
    run_path = os.path.join(runs_folder, run_folder, '*.txt')
    txt_files = glob.glob(run_path)
    
    if not txt_files: continue
        
    fname = txt_files[0]
    chain_id = os.path.basename(run_folder)
    
    print(f'Caricando {fname}...')
    
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    i = 0
    seq_data = []
    while i < len(lines):
        if lines[i].startswith('>step='):
            header = lines[i].strip()[1:]  # toglie '>'
            parts = header.split()
            
            # Estrai cosine
            cosine_str = None
            for p in parts:
                if 'cosine_to_tonb=' in p:
                    cosine_str = p.split('=')[1]
                    break
            
            if cosine_str:
                try:
                    cosine = float(cosine_str)
                    if cosine >= CUTOFF_COSINE:
                        # Sequenza è riga SUCCESSIVA
                        i += 1
                        if i < len(lines):
                            sequence = lines[i].strip()
                            seq_data.append({
                                'chain_id': chain_id,
                                'sequence': sequence,
                                'cosine': cosine,
                                'step': parts[0].split('=')[1]  # step=XXXX
                            })
                except:
                    pass
        
        i += 1
    
    if seq_data:
        run_df = pd.DataFrame(seq_data)
        df_all.append(run_df)

df_all = pd.concat(df_all, ignore_index=True)
print(f'✅ Totale seq >{CUTOFF_COSINE}: {len(df_all)}')

# === 2. Conta e filtra ===
counts_per_chain = df_all['chain_id'].value_counts()
good_chains = counts_per_chain[counts_per_chain >= MIN_SEQ_PER_CHAIN].index.tolist()
print(f'Chain buone: {len(good_chains)}')

# === 3. Bilancia ===
df_balanced = []
for chain in good_chains:
    chain_df = df_all[df_all['chain_id'] == chain]
    n_sample = min(N_SAMPLE_PER_CHAIN, len(chain_df))
    sampled = chain_df.sample(n=n_sample, random_state=42)
    df_balanced.append(sampled)
    print(f'{chain}: {n_sample}/{len(chain_df)}')

df_balanced = pd.concat(df_balanced, ignore_index=True)
print(f'\n🎉 Dataset bilanciato: {len(df_balanced)} seq')

# === 4. Salva ===
df_balanced.to_csv('dataset_bilanciato_097.csv', index=False)
print(df_balanced.head(3))
print('Salvato!')