import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO, AlignIO, SeqRecord, Seq
from collections import Counter
from tqdm import tqdm
import subprocess

# =============================================================================
# CONFIG
# =============================================================================
ALN_DIR = Path('./msa_aln')
CONSENSUS_DIR = Path('./consensus_msa')
WT_TONB_FASTA = 'TonB_wt.fa'  # File WT (crealo se non esiste)

CONSENSUS_FASTA = CONSENSUS_DIR / 'consensus_all_runs.fa'
CONSENSUS_ALN = CONSENSUS_DIR / 'consensus_all_runs.aln'
SUMMARY_CONSENSUS = './consensus_summary.csv'

# =============================================================================
# STEP 1: CONSENSUS PER RUN
# =============================================================================
def extract_consensus(aln_path, threshold=0.4):
    """Consensus SENZA Biopython"""
    run_id = aln_path.stem
    align = []
    with open(aln_path) as f:
        for record in SeqIO.parse(f, 'fasta'):
            align.append(str(record.seq))
    
    L = len(align[0])
    consensus_seq = ''
    
    for pos in range(L):
        col = [seq[pos] for seq in align]
        counts = Counter(c for c in col if c != '-')
        
        if counts:
            top_aa, top_freq = counts.most_common(1)[0]
            if top_freq / len(align) >= threshold:
                consensus_seq += top_aa
            else:
                consensus_seq += 'X'
        else:
            consensus_seq += '-'
    
    # Salva FASTA semplice
    fasta_path = CONSENSUS_DIR / f"{run_id}_consensus.fa"
    with open(fasta_path, 'w') as f:
        f.write(f">{run_id}_consensus\n{consensus_seq}\n")
    
    return consensus_seq
# =============================================================================
# STEP 2: MSA CONSENSUS + WT
# =============================================================================
def create_consensus_msa():
    Path(CONSENSUS_DIR).mkdir(exist_ok=True)
    
    consensus_seqs = {}
    aln_files = list(ALN_DIR.glob('*.aln'))
    
    for aln_path in tqdm(aln_files, desc="Consensus"):
        seq = extract_consensus(aln_path)
        run_id = aln_path.stem
        consensus_seqs[run_id] = seq
    
    # FASTA multiplo
    with open(CONSENSUS_FASTA, 'w') as f:
        for run_id, seq in consensus_seqs.items():
            f.write(f">{run_id}_consensus\n{seq}\n")
    
    # FAMSA
    subprocess.run(['famsa', '-v', str(CONSENSUS_FASTA), str(CONSENSUS_ALN)], check=True)
    return CONSENSUS_ALN
# =============================================================================
# STEP 3: ANALISI CONSENSUS MEDIO
# =============================================================================
def analyze_consensus_msa(aln_path):
    """Analizza MSA consensus → sequenza media"""
    align = AlignIO.read(aln_path, 'fasta')
    L = align.get_alignment_length()
    
    # Consensus finale (17 run)
    final_consensus = ''
    pos_stats = []
    
    for pos in range(L):
        col = [align[i][pos] for i in range(len(align))]
        counts = Counter(c for c in col if c != '-')
        
        if counts:
            top_aa, top_freq = counts.most_common(1)[0]
            final_consensus += top_aa
            pos_stats.append({
                'position': pos + 1,
                'consensus_aa': top_aa,
                'support': top_freq / len(align),
                'n_runs': top_freq,
                'entropy': -sum(f/len(align) * np.log2(f/len(align)+1e-10) 
                               for f in counts.values())
            })
        else:
            final_consensus += '-'
    
    return final_consensus, pd.DataFrame(pos_stats)

# =============================================================================
# CONFRONTO WT
# =============================================================================
def compare_to_wt(consensus_seq, wt_seq):
    """Differenze consensus vs WT"""
    diffs = []
    for i in range(min(len(consensus_seq), len(wt_seq))):
        if consensus_seq[i] != '-' and wt_seq[i] != '-' and consensus_seq[i] != wt_seq[i]:
            diffs.append({
                'position': i + 1,
                'wt_aa': wt_seq[i],
                'cons_aa': consensus_seq[i],
                'change': f"{wt_seq[i]}→{consensus_seq[i]}"
            })
    return diffs

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🔬 CONSENSUS PIPELINE - MEDIA ESM TONB")
    print("-" * 50)
    
    # STEP 1-2: Crea MSA consensus
    consensus_aln = create_consensus_msa()
    
    # STEP 3: Analizza
    print("\n📊 ANALISI CONSENSUS FINALE...")
    final_consensus, pos_stats = analyze_consensus_msa(consensus_aln)
    
    # Salva
    pos_stats.to_csv(SUMMARY_CONSENSUS, index=False)
    print(f"✅ Summary: {SUMMARY_CONSENSUS}")
    
    # WT confronto (manuale - sostituisci con tua seq WT)
    wt_seq = "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ"
    diffs = compare_to_wt(final_consensus, wt_seq)
    
    print(f"\n🎯 CONSENSUS FINALE (17 run):")
    print(f"Lunghezza: {len(final_consensus)}")
    print(f"Supporto medio: {pos_stats['support'].mean():.1%}")
    print(f"Mutazioni vs WT: {len(diffs)}")
    
    print("\n🔥 TOP 10 POSIZIONI CONSENSUS:")
    print(pos_stats.nlargest(10, 'support')[['position', 'consensus_aa', 'support', 'n_runs']])
    
    print("\n🧬 MUTAZIONI CONSENSUS vs WT:")
    for d in diffs[:10]:
        print(f"  Pos {d['position']}: {d['change']}")
    
    # Salva sequenza finale
    with open('TonB_ESM_consensus.fa', 'w') as f:
        f.write(f">TonB_ESM_consensus_17runs\n{final_consensus}\n")
    print(f"\n💾 Sequenza finale: TonB_ESM_consensus.fa")

if __name__ == "__main__":
    main()