import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# === 1. CONFIG ===
df_path = './dataset_bilanciato_097.csv' 
TonB_wt = 'MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ'

print(f'TonB lunghezza: {len(TonB_wt)}')

# === 2. CARICA DATASET ===
df = pd.read_csv(df_path)
N_seq, L = len(df), len(TonB_wt)
print(f'Dataset bilanciato: {N_seq} seq')

# === 3. PROFILO COMPLETO ===
print('Calcolo profilo mutazionale...')
profile_data = []

for pos in range(L):
    wt_aa = TonB_wt[pos]
    
    # Colonna posizione
    col_pos = df['sequence'].str[pos]
    
    # Statistiche
    counts = col_pos.value_counts()
    conservation = (col_pos == wt_aa).mean()
    entropy = -sum((c/N_seq)*np.log2(c/N_seq+1e-10) for c in counts.values)
    
    row = {
        'position': pos+1,
        'wt_aa': wt_aa,
        'conservation': conservation,
        'n_mutants': len(counts),
        'entropy': entropy,
        'top_mutant': counts.index[1] if len(counts)>1 else None,
        'freq_top_mut': counts.iloc[1]/N_seq if len(counts)>1 else 0
    }
    profile_data.append(row)

profile_df = pd.DataFrame(profile_data)

# === 4. SALVA ===
profile_df.to_csv('profilo_tolleranza_completo.csv', index=False)
print('Salvato: profilo_tolleranza_completo.csv')

# === 5. ANALISI TOP ===
print('\n=== TOP 15 POSIZIONI PIÙ RIGIDE ===')
top_rigide = profile_df.nlargest(15, 'conservation')
print(top_rigide[['position', 'wt_aa', 'conservation', 'n_mutants']].round(3))

print('\n=== TOP 15 POSIZIONI PIÙ VARIABILI ===')
top_variabili = profile_df.nsmallest(15, 'conservation')
print(top_variabili[['position', 'wt_aa', 'conservation', 'n_mutants']].round(3))

print('\n=== STATISTICHE GLOBALI ===')
print(f'Media conservation: {profile_df.conservation.mean():.3f}')
print(f'Media n_mutants: {profile_df.n_mutants.mean():.1f}')
print(f'Posizioni con >15 mutanti: {sum(profile_df.n_mutants > 15)}')

# === 6. PLOT MATPLOTLIB (NO HTML) ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analisi Profilo Tolleranza Mutazionale', fontsize=16)

# Plot 1: Conservation per posizione
axes[0,0].plot(profile_df['position'], profile_df['conservation'], 'o-', linewidth=1, markersize=3)
axes[0,0].axhline(0.9, color='red', linestyle='--', label='Cutoff 0.9', alpha=0.7)
axes[0,0].axhline(profile_df['conservation'].mean(), color='orange', linestyle=':', label='Media')
axes[0,0].set_title('Conservation per posizione')
axes[0,0].set_xlabel('Posizione')
axes[0,0].set_ylabel('Conservation WT')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Istogramma numero mutanti
axes[0,1].hist(profile_df['n_mutants'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0,1].axvline(profile_df['n_mutants'].median(), color='red', linestyle='--', label='Mediana')
axes[0,1].set_title('Distribuzione numero mutanti per posizione')
axes[0,1].set_xlabel('Numero mutanti distinti')
axes[0,1].set_ylabel('Frequenza')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Conservation vs Entropy
scatter = axes[1,0].scatter(profile_df['entropy'], profile_df['conservation'], 
                          c=profile_df['n_mutants'], cmap='viridis', alpha=0.6, s=30)
axes[1,0].set_xlabel('Entropia (diversità)')
axes[1,0].set_ylabel('Conservation')
axes[1,0].set_title('Conservation vs Entropia (size = n_mutants)')
plt.colorbar(scatter, ax=axes[1,0], label='N. mutanti')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Top 20 posizioni più rigide
top20 = profile_df.nlargest(20, 'conservation')
colors = plt.cm.Greens(np.linspace(0.3, 1, 20))
axes[1,1].barh(range(20), top20['conservation'], color=colors)
axes[1,1].set_yticks(range(20))
axes[1,1].set_yticklabels([f'P{row.position}\n({row.wt_aa})' for _,row in top20.iterrows()])
axes[1,1].set_title('Top 20 Posizioni più RIGIDE')
axes[1,1].set_xlabel('Conservation WT')
axes[1,1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('profilo_tolleranza_completo.png', dpi=300, bbox_inches='tight')
plt.show()

# === 7. HEATMAP TOP 50 POSIZIONI ===
plt.figure(figsize=(12, 8))
top50_pos = profile_df.nlargest(50, 'conservation')['position'].values
sns.heatmap(profile_df.set_index('position').loc[top50_pos, ['conservation', 'n_mutants', 'entropy']],
           annot=True, cmap='RdYlGn_r', center=0.5, fmt='.3f',
           cbar_kws={'label': 'Valore'})
plt.title('Top 50 Posizioni: Heatmap metriche')
plt.tight_layout()
plt.savefig('heatmap_top50.png', dpi=300, bbox_inches='tight')
plt.show()

# === 8. SUMMARY TABLE ===
summary = pd.DataFrame({
    'Regione': ['Proline-rich (70-105)', 'Lysine repeats (95-105)', 'C-term (190-239)', 'Globale'],
    'Conservation media': [
        profile_df.loc[69:104, 'conservation'].mean(),
        profile_df.loc[94:104, 'conservation'].mean(), 
        profile_df.loc[189:, 'conservation'].mean(),
        profile_df['conservation'].mean()
    ]
})
print('\n=== SUMMARY REGIONI TONB ===')
print(summary.round(3))

# === 9. SALVA TUTTO ===
profile_df.to_csv('profilo_tolleranza_completo.csv', index=False)
top_rigide.to_csv('top_rigide.csv', index=False)
top_variabili.to_csv('top_variabili.csv', index=False)
summary.to_csv('summary_regioni.csv', index=False)

print('\n✅ TUTTO SALVATO!')
print('File generati:')
print('- profilo_tolleranza_completo.csv (completo)')
print('- top_rigide.csv')
print('- top_variabili.csv')
print('- summary_regioni.csv')
print('- profilo_tolleranza_completo.png')
print('- heatmap_top50.png')
