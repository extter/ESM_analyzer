import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from Bio import SeqIO
import sys

# --------------------------------
# SELETTORE MODALITÀ
# --------------------------------
MODE = "OPTIMIZED"  # Opzioni: "NORMAL" oppure "OPTIMIZED"

# --------------------------------
# CONFIGURAZIONE DINAMICA
# --------------------------------
if MODE == "NORMAL":
    ALN_DIR = Path('./msa_aln')
    HEATMAP_DIR = Path('./msa_heatmaps')
    SUMMARY_PATH = './SUMMARY_MSA_ANALYSIS.csv'
    print(f"ANALISI NORMAL ATTIVA (Input: {ALN_DIR})")

elif MODE == "OPTIMIZED":
    ALN_DIR = Path('./msa_aln_opt')
    HEATMAP_DIR = Path('./msa_heatmaps_opt')
    SUMMARY_PATH = './SUMMARY_MSA_ANALYSIS_OPT.csv'
    print(f"ANALISI OPTIMIZED ATTIVA (Input: {ALN_DIR})")

else:
    print("ERRORE: Mode non valido.")
    sys.exit()

CONS_THRESH = 0.8
CONS_CRITICAL = 0.9
aa_order = 'ACDEFGHIKLMNPQRSTVWY'

# --------------------------------
# CORE ANALYSIS
# --------------------------------
def analyze_msa(aln_path):
    """Freq matrix + metriche + top conservati"""
    run_id = aln_path.stem
    
    align = list(SeqIO.parse(aln_path, 'fasta'))
    if not align:
        return None, None

    L, N = len(align[0]), len(align)
    
    freqs = np.zeros((L, 20))
    cons_profile = np.zeros(L)
    aa_dominant = [''] * L
    gap_cols = 0
    
    for pos in range(L):
        col = [str(rec.seq[pos]) for rec in align]
        n_gaps = col.count('-')
        
        if n_gaps / N > 0.9:
            gap_cols += 1
            continue
            
        counts = Counter(c for c in col if c != '-')
        total_valid = N - n_gaps
        
        if total_valid == 0:
            continue

        for i, aa in enumerate(aa_order):
            freqs[pos, i] = counts.get(aa, 0) / total_valid
        
        # Conservazione + AA dominante
        cons_profile[pos] = max(counts.values()) / total_valid
        aa_dominant[pos] = max(counts, key=counts.get) if counts else '-'
    
    # Metriche aggregate
    effective_L = L - gap_cols
    if effective_L == 0: effective_L = 1 # Evita division by zero
    
    # Calcolo entropia
    entropy_sum = 0
    for row in freqs:
        if np.sum(row) > 0:
            entropy_sum += np.sum(row * np.log2(row + 1e-10))
            
    results = {
        'run_id': run_id,
        'N_seq': N,
        'L_aln': L,
        'effective_L': effective_L,
        'C_mean': float(np.mean(cons_profile)),
        'H_mean': float(-entropy_sum / effective_L),
        'pos_conserved': int(np.sum(cons_profile > CONS_THRESH)),
        'pos_critical': int(np.sum(cons_profile > CONS_CRITICAL)),
    }
    
    # TOP 10 AA conservati per run
    cons_aa = []
    for pos in range(L):
        if cons_profile[pos] > CONS_THRESH:
            cons_aa.append((pos+1, aa_dominant[pos], cons_profile[pos]))
    
    results['top_conserved'] = sorted(cons_aa, key=lambda x: x[2], reverse=True)[:10]
    
    return results, freqs

# --------------------------------
# HEATMAP PLASMA
# --------------------------------
def plot_heatmap(freqs, results):
    """Heatmap freq AA x posizione - PLASMA"""
    Path(HEATMAP_DIR).mkdir(exist_ok=True)
    run_id = results['run_id']
    
    plt.figure(figsize=(20, 8))
    
    # Taglia la heatmap alla lunghezza effettiva se ci sono gap finali, 
    # ma qui usiamo L originale per coerenza visiva
    sns.heatmap(freqs.T, 
                cmap='plasma',  # ← PLASMA!
                vmin=0, vmax=1,
                yticklabels=aa_order,
                xticklabels=False,
                cbar_kws={'label': 'Frequenza AA'})
    
    plt.title(f'{run_id} ({MODE})\n'
              f'C_mean={results["C_mean"]:.3f} | H_mean={results["H_mean"]:.3f} | '
              f'Pos conserved={results["pos_conserved"]} | critical={results["pos_critical"]}')
    plt.ylabel('Aminoacidi')
    plt.xlabel('Posizione TonB (Allineamento)')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / f'{run_id}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------
# MAIN
# --------------------------------
def main():
    print(f"🔬 MSA ANALYSIS FINALE - {MODE}")
    print("-" * 60)
    
    # Check directory
    if not ALN_DIR.exists():
        print(f"ERRORE: La cartella {ALN_DIR} non esiste.")
        print("Hai eseguito lo Step 1?")
        return

    aln_files = list(ALN_DIR.glob('*.aln'))
    
    if not aln_files:
        print(f"Nessun file .aln trovato in {ALN_DIR}")
        return

    all_results = []
    
    print(f"Analizzo {len(aln_files)} MSA...")
    for aln_path in tqdm(aln_files, desc="MSA + Heatmap"):
        results, freqs = analyze_msa(aln_path)
        if results:
            plot_heatmap(freqs, results)
            all_results.append(results)
    
    if not all_results:
        print("Nessun risultato valido generato.")
        return

    # Summary
    df_summary = pd.DataFrame(all_results)
    df_summary['diversity'] = df_summary['H_mean'] * (1 - df_summary['C_mean'])
    df_summary = df_summary.sort_values('C_mean', ascending=False)  # Conservazione first
    df_summary.to_csv(SUMMARY_PATH, index=False)
    
    print(f"\n📊 SUMMARY: {SUMMARY_PATH}")
    print(df_summary[['run_id', 'C_mean', 'H_mean', 'pos_conserved', 'pos_critical', 'diversity']].round(3))
    
    print(f"\n🖼️  HEATMAP: {HEATMAP_DIR}/ ({len(aln_files)} file)")
    
    # Top conservati per run
    print("\n🔒 TOP CONSERVATI PER RUN (Prime 3 run):")
    for _, row in df_summary.head(3).iterrows():
        print(f"\n{row['run_id']}:")
        for pos, aa, c in row['top_conserved']:
            print(f"  Pos {pos}: {aa} (C={c:.3f})")

if __name__ == "__main__":
    main()