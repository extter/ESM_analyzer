from random import sample
from Bio import SeqIO
from Bio import AlignIO
from pathlib import Path
import numpy as np
import pandas as pd

FASTA_DIR = Path('./msa_fasta_opt')
ALN_DIR = Path('./msa_aln_opt')

def hamming_distance(s1, s2):
    min_len = min(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len])) / min_len

def diagnose_fast(fasta_path):
    """Versione RAPIDA - 100 coppie random"""
    seqs = list(SeqIO.parse(fasta_path, "fasta"))
    n_seq = len(seqs)
    
    print(f"\n🔍 {fasta_path.name}: {n_seq} seq")
    
    if n_seq < 10:
        print("  ⚠️ Pochi dati")
        return None
    
    # 100 COPPIE RANDOM (non tutte!)
    pairs = sample(list(range(n_seq)), min(200, n_seq*2))
    hamming_vals = []
    
    for i in range(0, len(pairs), 2):
        if i+1 < len(pairs):
            s1, s2 = seqs[pairs[i]], seqs[pairs[i+1]]
            hamming_vals.append(hamming_distance(s1.seq, s2.seq))
    
    hamming_avg = np.mean(hamming_vals)
    
    lengths = [len(s.seq) for s in seqs]
    
    status = "✅" if hamming_avg > 0.05 else "⚠️" if hamming_avg > 0.02 else "❌"
    
    print(f"  Hamming: {hamming_avg:.2%} {status}")
    print(f"  Len: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    
    return {
        'run': fasta_path.stem,
        'hamming_avg_%': hamming_avg * 100,
        'n_pairs': len(hamming_vals),
        'len_mean': np.mean(lengths)
    }
def count_gaps(aln_path):
    """Conta % gap per colonna"""
    alignment = AlignIO.read(aln_path, "fasta")
    L = alignment.get_alignment_length()
    n_seq = len(alignment)
    
    gap_per_col = []
    for col in range(L):
        col_data = [alignment[i][col] for i in range(n_seq)]
        gap_pct = col_data.count('-') / n_seq * 100
        gap_per_col.append(gap_pct)
    
    return np.array(gap_per_col)

def main():
    fasta_files = sorted(FASTA_DIR.glob("*.fa"))
    aln_files = sorted(ALN_DIR.glob("*.aln"))
    print(f"Trovati {len(fasta_files)} FASTA\n")
    
    results = []
    for fa in fasta_files[:6]:  # 6 run
        res = diagnose_fast(fa)
        if res: results.append(res)

    for aln in aln_files[:5]:  # Prime 5
        gaps = count_gaps(aln)
        print(f"{aln.name}:")
        print(f"  Gap medio: {gaps.mean():.1f}%")
        print(f"  Gap max:   {gaps.max():.1f}%") 
        print(f"  Colonne >20% gap: {np.sum(gaps > 20)}")
        print(f"  Colonne >50% gap: {np.sum(gaps > 50)}")
        print()

    if results:
        df = pd.DataFrame(results)
        print("\n📊 SUMMARY:")
        print(df[['run', 'hamming_avg_%']].round(1))
        
        risky = df[df['hamming_avg_%'] < 3]
        if len(risky):
            print(f"\n🚨 FIX CONSIGLIATO per {len(risky)} run")
        else:
            print("\n✅ TUTTO OK!")

if __name__ == "__main__":
    main()