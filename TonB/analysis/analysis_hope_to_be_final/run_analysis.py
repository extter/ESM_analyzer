#!/usr/bin/env python3
"""
PIPELINE MSA ANALYSIS - FIXED PCA CLUSTERING
Gestisce MSA di lunghezze diverse per clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG (IDENTICO)
# =============================================================================
ALN_DIR = Path('./msa_aln')
HEATMAP_DIR = Path('./msa_heatmaps')
SUMMARY_PATH = './SUMMARY_MSA_RUNS.csv'
PCA_PLOT = './run_clusters_pca.png'
TSNE_PLOT = './run_clusters_tsne.png'

aa_order = 'ACDEFGHIKLMNPQRSTVWY'
CONS_THRESH = 0.8
VAR_THRESH = 0.3


# =============================================================================
# ANALISI SINGOLO MSA (IDENTICA)
# =============================================================================
def analyze_single_msa(aln_path):
    """Analizza MSA → freq matrix + metriche"""
    run_id = aln_path.stem
    
    try:
        align = list(SeqIO.parse(aln_path, 'fasta'))
        if not align:
            print(f"⚠️  Skip {run_id}: MSA vuoto")
            return None, None
            
        L, N = len(align[0]), len(align)
        print(f"Analizzo {run_id}: {N} seq × {L} pos")
        
        freqs = np.zeros((L, 20))
        gap_cols = 0
        
        for pos in range(L):
            col = [str(rec.seq[pos]) for rec in align]
            n_gaps = col.count('-')
            
            if n_gaps / N > 0.9:
                gap_cols += 1
                continue
                
            counts = Counter(c for c in col if c != '-')
            total_valid = N - n_gaps
            for i, aa in enumerate(aa_order):
                freqs[pos, i] = counts.get(aa, 0) / total_valid
        
        cons = np.nanmax(freqs, axis=1)
        entropy = -np.nansum(freqs * np.log2(freqs + 1e-10), axis=1)
        
        results = {
            'run_id': run_id,
            'N_seq': N,
            'L_aln': L,
            'gap_columns': gap_cols,
            'effective_L': L - gap_cols,
            'H_mean': float(np.nanmean(entropy)),
            'H_std': float(np.nanstd(entropy)),
            'C_mean': float(np.nanmean(cons)),
            'C_std': float(np.nanstd(cons)),
            'pos_conserved': int(np.sum(cons > CONS_THRESH)),
            'pos_variable': int(np.sum(cons < VAR_THRESH)),
            'diversity': float(np.nanmean(entropy) * (1 - np.nanmean(cons)))
        }
        
        return results, freqs
        
    except Exception as e:
        print(f"❌ Errore {run_id}: {e}")
        return None, None


# =============================================================================
# HEATMAP (IDENTICA)
# =============================================================================
def plot_heatmap(freqs, results, L_eff):
    Path(HEATMAP_DIR).mkdir(exist_ok=True)
    run_id = results['run_id']
    
    plt.figure(figsize=(20, 6))
    sns.heatmap(freqs[:L_eff].T, 
                cmap='Blues', vmin=0, vmax=1,
                yticklabels=aa_order, 
                xticklabels=False,
                cbar_kws={'label': 'Frequenza'})
    
    plt.title(f'{run_id}\n'
              f'N={results["N_seq"]} L_eff={L_eff} | '
              f'H={results["H_mean"]:.2f} C={results["C_mean"]:.2f} '
              f'Div={results["diversity"]:.3f}')
    
    plt.ylabel('Aminioacidi')
    plt.xlabel('Posizione MSA')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / f'{run_id}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# CLUSTERING ROBUSTO (FIX)
# =============================================================================
def robust_clustering(df_summary):
    """Clustering metriche aggregate - NO profili raw"""
    print("🔍 Clustering metriche scalari...")
    
    # Metriche fisse (non dipendono da lunghezza MSA)
    features = ['H_mean', 'H_std', 'C_mean', 'C_std', 'pos_conserved', 
                'pos_variable', 'diversity']
    X = StandardScaler().fit_transform(df_summary[features])
    
    print(f"Features shape: {X.shape} ({len(features)} metriche)")
    
    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # KMeans (max 1/3 del totale)
    n_clusters = min(4, len(df_summary) // 6 + 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    print(f"Clusters trovati: {len(set(clusters))}")
    return clusters, X_pca, pca.explained_variance_ratio_


# =============================================================================
# MAIN (FIXED)
# =============================================================================
def main():
    print("🔬 MSA ANALYSIS - FIXED CLUSTERING")
    print("-" * 60)
    
    aln_files = list(ALN_DIR.glob('*.aln'))
    print(f"📊 {len(aln_files)} MSA...")
    
    all_results = []
    all_profiles = []
    
    for aln_path in tqdm(aln_files, desc="Analysis"):
        result = analyze_single_msa(aln_path)
        if result[0] is None:
            continue
            
        results, freqs = result
        L_eff = results['effective_L']
        
        plot_heatmap(freqs, results, L_eff)
        all_results.append(results)
        all_profiles.append(freqs[:L_eff].flatten())  # Non usato per clustering
    
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values('diversity', ascending=False)
    df_summary.to_csv(SUMMARY_PATH, index=False)
    
    print(f"\n📈 SUMMARY: {SUMMARY_PATH}")
    display_cols = ['run_id', 'N_seq', 'effective_L', 'H_mean', 'C_mean', 'diversity']
    print(df_summary[display_cols].round(3))

    # CLUSTERING ROBUSTO (NUOVO)
    clusters, X_pca, var_ratio = robust_clustering(df_summary)
    df_summary['cluster'] = clusters
    df_summary.to_csv(SUMMARY_PATH, index=False)

    # PLOT PCA + BARPLOT
    plt.figure(figsize=(15, 6))

    # PCA scatter
    plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(clusters / len(set(clusters)))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], 
                        c=colors, s=200, alpha=0.85, edgecolors='black', linewidth=1.5)
    plt.colorbar(scatter, label='Cluster ID')

    for i, txt in enumerate(df_summary['run_id'].str[:12]):
        plt.annotate(txt, (X_pca[i,0], X_pca[i,1]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=9)

    plt.title(f'PCA Metriche MSA\nPC1={var_ratio[0]:.1%} + PC2={var_ratio[1]:.1%}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)

    # Diversity per cluster
    plt.subplot(1, 2, 2)
    cluster_stats = df_summary.groupby('cluster').agg({
        'diversity': 'mean', 'H_mean': 'mean', 'C_mean': 'mean', 'run_id': 'count'
    }).round(3)

    bars = plt.bar(cluster_stats.index, cluster_stats['diversity'], 
                color=plt.cm.tab10(cluster_stats.index / len(cluster_stats)))
    plt.title('Diversity Media per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Diversity Score')
    plt.xticks(range(len(cluster_stats)))

    for i, bar in enumerate(bars):
        height = cluster_stats['diversity'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.015, 
                f'{height:.3f}\nn={cluster_stats["run_id"].iloc[i]}', 
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PCA_PLOT, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n🎨 PLOT salvato: {PCA_PLOT}")
    print("\n📊 STATS CLUSTER:")
    print(cluster_stats)

    print("\n🔥 TOP 5 DIVERSE:")
    print(df_summary[['run_id', 'H_mean', 'C_mean', 'diversity', 'cluster']].head())

if __name__ == "__main__":
    main()
