import os
import glob
import random
import subprocess
import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# -----------------------------------------------------------------------------
# COSTANTI GLOBALI
# -----------------------------------------------------------------------------
SEQ_WT_NAME = "TonB_WT"
SEQ_WT_STR = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'

# -----------------------------------------------------------------------------
# STEP 1: ESTRAZIONE E FILTRAGGIO 
# -----------------------------------------------------------------------------
def parse_txt_run(fname, chain_id, threshold):
    """Legge i vecchi file .txt (Modalità NORMAL)"""
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
                    if cosine >= threshold:
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

def parse_fasta_optimized_run(fasta_path, run_id, threshold):
    """Legge best_candidates.fasta (Modalità OPTIMIZED)"""
    valid_records = []
    records = list(SeqIO.parse(fasta_path, "fasta"))
    
    for rec in records:
        try:
            parts = rec.id.split('_')
            if 'sim' in parts:
                idx = parts.index('sim')
                score = float(parts[idx+1])
                if score >= threshold:
                    rec.id = f"{run_id}_{rec.id}"
                    rec.description = "" 
                    valid_records.append(rec)
        except (ValueError, IndexError):
            continue
    return valid_records

def step_extract(mode: str, dirs: dict, config: dict):
    print(f"\n--- STEP 1: ESTRAZIONE DATI ({mode}) ---")
    dirs['fasta'].mkdir(parents=True, exist_ok=True)
    
    fasta_files = []
    total_seqs = 0

    if mode == "NORMAL":
        df_all = []
        run_folders = [f for f in os.listdir(dirs['runs']) if os.path.isdir(os.path.join(dirs['runs'], f))]
        for run_folder in tqdm(run_folders, desc="Parsing Run TXT"):
            txt_files = glob.glob(os.path.join(dirs['runs'], run_folder, '*.txt'))
            if not txt_files: continue
            
            seq_data = parse_txt_run(txt_files[0], run_folder, config['threshold'])
            if seq_data: df_all.append(pd.DataFrame(seq_data))
            
        if df_all:
            df_total = pd.concat(df_all, ignore_index=True)
            counts = df_total['chain_id'].value_counts()
            good_runs = counts[counts >= config['min_seq_run']].index
            
            for chain in tqdm(good_runs, desc="Generazione FASTA"):
                df_run = df_total[df_total['chain_id'] == chain]
                sample = df_run.sample(min(config['n_sample'], len(df_run)), random_state=42)
                
                fa_path = dirs['fasta'] / f"{chain}.fa"
                with open(fa_path, 'w') as f:
                    for _, row in sample.iterrows():
                        f.write(f">{chain}_{row['step']}_cos{row['cosine']:.3f}\n{row['sequence']}\n")
                fasta_files.append(fa_path)
                total_seqs += len(sample)

    elif mode == "OPTIMIZED":
        run_folders = [f for f in os.listdir(dirs['runs']) if os.path.isdir(os.path.join(dirs['runs'], f))]
        for run_folder in tqdm(run_folders, desc="Parsing Run FASTA"):
            input_fasta = os.path.join(dirs['runs'], run_folder, "best_candidates.fasta")
            if not os.path.exists(input_fasta): continue
            
            valid_seqs = parse_fasta_optimized_run(input_fasta, run_folder, config['threshold'])
            if not valid_seqs: continue
            
            selected_seqs = random.sample(valid_seqs, config['n_sample']) if len(valid_seqs) > config['n_sample'] else valid_seqs
            out_path = dirs['fasta'] / f"{run_folder}.fa"
            SeqIO.write(selected_seqs, out_path, "fasta")
            fasta_files.append(out_path)
            total_seqs += len(selected_seqs)

    print(f"Estratte {total_seqs} sequenze. Salvate in {dirs['fasta']}")
    return fasta_files

# -----------------------------------------------------------------------------
# STEP 2: ALLINEAMENTO FAMSA
# -----------------------------------------------------------------------------
def step_align(fasta_files, aln_dir):
    print("\n--- STEP 2: ALLINEAMENTO FAMSA ---")
    aln_dir.mkdir(parents=True, exist_ok=True)
    aln_files = []
    
    if not fasta_files:
        fasta_files = list(Path(str(aln_dir).replace('aln', 'fasta')).glob("*.fa"))
        
    for fa in tqdm(fasta_files, desc="Running FAMSA"):
        aln_path = aln_dir / f"{fa.stem}.aln"
        if not aln_path.exists():
            subprocess.run(['famsa', str(fa), str(aln_path)], capture_output=True)
        if aln_path.exists():
            aln_files.append(aln_path)
    print(f"Generati {len(aln_files)} allineamenti in {aln_dir}")
    return aln_files

# -----------------------------------------------------------------------------
# STEP 3: ANALISI SINGOLE RUN E HEATMAPS 
# -----------------------------------------------------------------------------
def analyze_msa(aln_path):
    align = list(SeqIO.parse(aln_path, 'fasta'))
    if not align: return None, None
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
        if total_valid == 0: continue

        for i, aa in enumerate(AA_ORDER):
            freqs[pos, i] = counts.get(aa, 0) / total_valid
        
        cons_profile[pos] = max(counts.values()) / total_valid
        aa_dominant[pos] = max(counts, key=counts.get) if counts else '-'
    
    effective_L = max(1, L - gap_cols)
    entropy_sum = sum(np.sum(row * np.log2(row + 1e-10)) for row in freqs if np.sum(row) > 0)
            
    results = {
        'run_id': aln_path.stem, 'N_seq': N, 'L_aln': L, 'effective_L': effective_L,
        'C_mean': float(np.mean(cons_profile)), 'H_mean': float(-entropy_sum / effective_L),
        'pos_conserved': int(np.sum(cons_profile > 0.8)), 'pos_critical': int(np.sum(cons_profile > 0.9))
    }
    
    cons_aa = [(pos+1, aa_dominant[pos], cons_profile[pos]) for pos in range(L) if cons_profile[pos] > 0.8]
    results['top_conserved'] = sorted(cons_aa, key=lambda x: x[2], reverse=True)[:10]
    return results, freqs

def step_single_run_analysis(dirs):
    print("\n--- STEP 3: ANALISI RUN SINGOLE E HEATMAP ---")
    dirs['heatmaps'].mkdir(parents=True, exist_ok=True)
    aln_files = list(dirs['aln'].glob('*.aln'))
    
    all_results = []
    for aln_path in tqdm(aln_files, desc="Generazione Heatmap"):
        results, freqs = analyze_msa(aln_path)
        if results:
            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(freqs.T, cmap='plasma', vmin=0, vmax=1, yticklabels=list(AA_ORDER), xticklabels=50)
            plt.title(f"C_mean={results['C_mean']:.3f} | H_mean={results['H_mean']:.3f}", size=12, pad=8)
            plt.xticks(fontsize=10, rotation=0) 
            plt.yticks(fontsize=10, rotation=0)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)
            plt.tight_layout()
            plt.savefig(dirs['heatmaps'] / f"{results['run_id']}_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            all_results.append(results)
            
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(dirs['data'] / 'SUMMARY_MSA_ANALYSIS.csv', index=False)
        print(f"Analisi salvata in {dirs['data']}")

# -----------------------------------------------------------------------------
# STEP 4: GLOBAL CONSENSUS & PLOTTING 
# -----------------------------------------------------------------------------
def calculate_robust_consensus(aln_path):
    try:
        alignment = AlignIO.read(aln_path, "fasta")
        length = alignment.get_alignment_length()
        consensus = []
        for i in range(length):
            clean_column = [aa for aa in alignment[:, i] if aa not in ['-', 'X', '.', '?']]
            if not clean_column: consensus.append("") 
            else: consensus.append(Counter(clean_column).most_common(1)[0][0])
        return "".join(consensus)
    except Exception: return None

def step_global_consensus_and_plot(dirs, title_suffix):
    print("\n--- STEP 4: GLOBAL CONSENSUS E PLOTTING ---")
    dirs['consensus'].mkdir(parents=True, exist_ok=True)
    dirs['plots'].mkdir(parents=True, exist_ok=True)
    
    records_to_align = [SeqRecord(Seq(SEQ_WT_STR), id=SEQ_WT_NAME, description="Wild Type")]
    
    for f in tqdm(list(dirs['aln'].glob("*.aln")), desc="Calcolo Consensus"):
        clean_cons = calculate_robust_consensus(f)
        if clean_cons and len(clean_cons) > 50:
            records_to_align.append(SeqRecord(Seq(clean_cons), id=f"Run_{f.stem}"))
            
    unaligned_path = dirs['consensus'] / "unaligned.fasta"
    aligned_path = dirs['consensus'] / "super_alignment.fasta"
    SeqIO.write(records_to_align, unaligned_path, "fasta")
    
    subprocess.run(["famsa", str(unaligned_path), str(aligned_path)], check=True, stdout=subprocess.DEVNULL)
    
    alignment = AlignIO.read(aligned_path, "fasta")
    wt_idx = next(i for i, rec in enumerate(alignment) if rec.id == SEQ_WT_NAME)
    
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
            n_runs = len(run_aas)
            
            if n_runs == 0: continue
            
            if valid_aas:
                most_common = Counter(valid_aas).most_common(1)[0]
                top_aa = most_common[0]
                support_abs = most_common[1]
            else:
                top_aa = "-" 
                support_abs = 0
                
            support_pct = (support_abs / n_runs) * 100
            gap_pct = (run_aas.count("-") / n_runs) * 100
            
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
                "Is_Mutated": (top_aa != wt_aa) and (top_aa != "-") and (support_pct > 30)
            })
            
    df = pd.DataFrame(mapped_data)
    df.to_csv(dirs['consensus'] / "consensus_mapped.csv", index=False)
    
    # --------------------------------------------
    # PLOTTING STILE 
    # --------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    TONB_DOMAINS = [
        {"start": 1, "end": 32, "color": "gray", "alpha": 0.1, "label": "TM Anchor (Rigid)"},
        {"start": 65, "end": 105, "color": "orange", "alpha": 0.1, "label": "Proline Linker (Disordered)"},
        {"start": 150, "end": 239, "color": "green", "alpha": 0.1, "label": "C-Term Barrel (Folded)"}
    ]

    # --- PLOT 1: MUTATION LANDSCAPE ---
    print("Generazione Plot 1: Mutation Landscape...")
    fig, ax = plt.subplots(figsize=(16, 6))

    colors = []
    for _, row in df.iterrows():
        is_wt = row['Consensus_Global_AA'] == row['WT_AA']
        support = row['Support_Pct']
        if is_wt:
            colors.append('#4d4d4d' if support > 50 else '#bdbdbd')
        else:
            colors.append('#d62728' if support > 50 else '#ff9896')

    ax.bar(df['WT_Pos'], df['Support_Pct'], color=colors, width=1.0, edgecolor='none')

    for dom in TONB_DOMAINS:
        ax.axvspan(dom["start"], dom["end"], color=dom["color"], alpha=dom["alpha"], label=dom["label"])

    ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(80, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    strong_muts = df[(df['Consensus_Global_AA'] != df['WT_AA']) & (df['Support_Pct'] > 60)]
    for _, row in strong_muts.iterrows():
        label = f"{row['WT_AA']}{int(row['WT_Pos'])}{row['Consensus_Global_AA']}"
        ax.text(row['WT_Pos'], row['Support_Pct'] + 2, label, 
                ha='center', va='bottom', fontsize=9, rotation=90, fontweight='bold', color='#8b0000')

    ax.set_xlim(0, 240)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Residue Position", fontsize=12)
    ax.set_ylabel("Consensus Support (%)", fontsize=12)
    ax.set_title(f"ESM-2 Evolutionary Landscape of TonB {title_suffix}", fontsize=14, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#4d4d4d', label='Conserved (High Confidence)'),
        mpatches.Patch(color='#d62728', label='Mutated (High Confidence)'),
        mpatches.Patch(color='#bdbdbd', label='Uncertain/Variable Region')
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(dirs['plots'] / "1_Mutation_Landscape.png", dpi=300)
    plt.close()

    # --- PLOT 2: ENTROPY & DISORDER ---
    print("Generazione Plot 2: Entropy Profile...")
    fig, ax = plt.subplots(figsize=(16, 5))

    sns.lineplot(x=df['WT_Pos'], y=df['Entropy_Bits'], color='#1f77b4', linewidth=2, ax=ax)
    ax.fill_between(df['WT_Pos'], df['Entropy_Bits'], color='#1f77b4', alpha=0.1)

  
    for dom in TONB_DOMAINS:
        ax.axvspan(dom["start"], dom["end"], color=dom["color"], alpha=dom["alpha"], label=dom["label"])

    ax.set_xlim(0, 240)
    y_max = max(df['Entropy_Bits']) * 1.1 if max(df['Entropy_Bits']) > 0 else 1.0
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Residue Position", fontsize=12)
    ax.set_ylabel("Shannon Entropy (Bits)", fontsize=12)
    ax.set_title(f"Structural Flexibility Profile {title_suffix}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(dirs['plots'] / "2_Entropy_Profile.png", dpi=300)
    plt.close()

    # --- PLOT 3: TOP PILLARS OF STABILITY ---
    TOP_N_RESIDUES = 20
    print(f"Generazione Plot 3: Top {TOP_N_RESIDUES} Conserved Residues...")

    df_sorted = df.sort_values(by=['Support_Pct', 'Entropy_Bits'], ascending=[False, True])
    df_top = df_sorted.head(TOP_N_RESIDUES).copy()

    df_top['Label'] = df_top.apply(lambda x: f"{x['Consensus_Global_AA']}{int(x['WT_Pos'])}", axis=1)

    bar_colors = []
    for _, row in df_top.iterrows():
        if row['Consensus_Global_AA'] == row['WT_AA']: bar_colors.append('#1f77b4') 
        else: bar_colors.append('#d62728') 

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(df_top['Label'], df_top['Support_Pct'], color=bar_colors, edgecolor='black', alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    ax.set_title(f"Top {TOP_N_RESIDUES} Most Frequent Residues {title_suffix}", fontsize=15, fontweight='bold')
    ax.set_xlabel("Residue (Consensus AA + Position)", fontsize=13)
    ax.set_ylabel("Frequency / Support (%)", fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.axhline(100, color='green', linestyle=':', linewidth=1.5, alpha=0.6)

    legend_patches = [
        mpatches.Patch(color='#1f77b4', label='Native Anchor (Identical to WT)'),
        mpatches.Patch(color='#d62728', label='Evolved Anchor (Stable Mutation)')
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(dirs['plots'] / f"3_Top_{TOP_N_RESIDUES}_Residues.png", dpi=300)
    plt.close()

    print(f"Super-allineamento e Plot salvati in {dirs['plots']}")
    

# -----------------------------------------------------------------------------
# MAIN MANAGER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline di analisi MSA per le Catene di Markov")
    parser.add_argument("--mode", choices=["NORMAL", "OPTIMIZED"], required=True)
    parser.add_argument("--step", choices=["1", "2", "3", "4", "all"], default="all", help="Quale step eseguire")
    args = parser.parse_args()

    # Configurazione e Cartelle 
    base_dir = Path(f"./analysis_{args.mode.lower()}")
    dirs = {
        'runs': '../../markov/runs_ultra_optimized' if args.mode == "OPTIMIZED" else '../../markov/runs',
        'fasta': base_dir / '1_fasta',
        'aln': base_dir / '2_alignments',
        'data': base_dir / '3_data_summaries',
        'heatmaps': base_dir / '4_heatmaps',
        'consensus': base_dir / '5_consensus_data',
        'plots': base_dir / '6_final_plots'
    }

    for key, path in dirs.items():
        if key != 'runs': path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'threshold': 0.994 if args.mode == "OPTIMIZED" else 0.975,
        'n_sample': 300,
        'min_seq_run': 300
    }
    title_suffix = "(Gen 2 - Optimized > 0.995)" if args.mode == "OPTIMIZED" else "(Gen 1 - Cos > 0.90)"

    print(f"=== AVVIO PIPELINE MSA [{args.mode}] ===")
    
    fasta_files = []
    if args.step in ["1", "all"]: fasta_files = step_extract(args.mode, dirs, config)
    if args.step in ["2", "all"]: step_align(fasta_files, dirs['aln'])
    if args.step in ["3", "all"]: step_single_run_analysis(dirs)
    if args.step in ["4", "all"]: step_global_consensus_and_plot(dirs, title_suffix)
    
    print("\n=== PIPELINE COMPLETATA CON SUCCESSO ===")