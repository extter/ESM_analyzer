import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import sys

# =============================================================================
# ⚙️ SELETTORE MODALITÀ
# =============================================================================
MODE = "OPTIMIZED"  # Opzioni: "NORMAL" oppure "OPTIMIZED"

# =============================================================================
# CONFIGURAZIONE DINAMICA
# =============================================================================
if MODE == "NORMAL":
    CSV_PATH = "./consensus_analysis/consensus_mapped.csv"
    OUTPUT_IMG_DIR = "./plots_final"
    TITLE_SUFFIX = "(Gen 1 - Cos > 0.90)"
    print(f"🔵 PLOTTING NORMAL")
    print(f"   Input: {CSV_PATH}")
    print(f"   Output: {OUTPUT_IMG_DIR}")

elif MODE == "OPTIMIZED":
    CSV_PATH = "./consensus_analysis_opt/consensus_mapped.csv"
    OUTPUT_IMG_DIR = "./plots_final_opt"
    TITLE_SUFFIX = "(Gen 2 - Optimized > 0.995)"
    print(f"🚀 PLOTTING OPTIMIZED")
    print(f"   Input: {CSV_PATH}")
    print(f"   Output: {OUTPUT_IMG_DIR}")

else:
    print("ERRORE: Mode non valido.")
    sys.exit()

TOP_N_RESIDUES = 20
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERRORE CRITICO: Non trovo il file {CSV_PATH}")
    print("Hai eseguito lo Step 3 (Consensus Analysis)?")
    sys.exit()

# Caricamento dati
df = pd.read_csv(CSV_PATH)

# Setup Stile
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# =============================================================================
# PLOT 1: MUTATION LANDSCAPE
# =============================================================================
print("Generazione Plot 1: Mutation Landscape...")
fig, ax = plt.subplots(figsize=(16, 6))

colors = []
for _, row in df.iterrows():
    is_wt = row['Consensus_Global_AA'] == row['WT_AA']
    support = row['Support_Pct']
    
    if is_wt:
        colors.append('#4d4d4d' if support > 50 else '#bdbdbd') # Grigio scuro / chiaro
    else:
        colors.append('#d62728' if support > 50 else '#ff9896') # Rosso forte / pallido

# Bar Plot
ax.bar(df['WT_Pos'], df['Support_Pct'], color=colors, width=1.0, edgecolor='none')

# Annotazione Domini TonB
ax.axvspan(1, 32, color='blue', alpha=0.05, label='TM Anchor')
ax.axvspan(33, 100, color='orange', alpha=0.05, label='Proline-Rich Linker')
ax.axvspan(150, 239, color='green', alpha=0.05, label='C-Term Domain')

# Linee soglia
ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(80, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

# Etichette per mutazioni forti (Solo se diverse dal WT)
strong_muts = df[(df['Consensus_Global_AA'] != df['WT_AA']) & (df['Support_Pct'] > 60)]
for _, row in strong_muts.iterrows():
    label = f"{row['WT_AA']}{int(row['WT_Pos'])}{row['Consensus_Global_AA']}"
    ax.text(row['WT_Pos'], row['Support_Pct'] + 2, label, 
            ha='center', va='bottom', fontsize=9, rotation=90, fontweight='bold', color='#8b0000')

ax.set_xlim(0, 240)
ax.set_ylim(0, 115)
ax.set_xlabel("Residue Position", fontsize=12)
ax.set_ylabel("Consensus Support (%)", fontsize=12)
ax.set_title(f"ESM-2 Evolutionary Landscape of TonB {TITLE_SUFFIX}", fontsize=14, fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#4d4d4d', label='Conserved (High Confidence)'),
    mpatches.Patch(color='#d62728', label='Mutated (High Confidence)'),
    mpatches.Patch(color='#bdbdbd', label='Uncertain/Variable Region')
]
ax.legend(handles=legend_patches, loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMG_DIR}/1_Mutation_Landscape.png", dpi=300)
plt.close()

# =============================================================================
# PLOT 2: ENTROPY & DISORDER
# =============================================================================
print("Generazione Plot 2: Entropy Profile...")
fig, ax = plt.subplots(figsize=(16, 5))

sns.lineplot(x=df['WT_Pos'], y=df['Entropy_Bits'], color='#1f77b4', linewidth=2, ax=ax)
ax.fill_between(df['WT_Pos'], df['Entropy_Bits'], color='#1f77b4', alpha=0.1)

ax.axvspan(1, 32, color='gray', alpha=0.1, label='TM Anchor (Rigid)')
ax.axvspan(65, 105, color='orange', alpha=0.1, label='Proline Linker (Disordered?)')
ax.axvspan(150, 239, color='green', alpha=0.1, label='C-Term Barrel (Folded)')

ax.set_xlim(0, 240)
# Se l'entropia è zero ovunque (possibile in OPTIMIZED), fissa un minimo per il grafico
y_max = max(df['Entropy_Bits']) * 1.1 if max(df['Entropy_Bits']) > 0 else 1.0

ax.set_ylim(0, y_max)
ax.set_xlabel("Residue Position", fontsize=12)
ax.set_ylabel("Shannon Entropy (Bits)", fontsize=12)
ax.set_title(f"Structural Flexibility Profile {TITLE_SUFFIX}", fontsize=14, fontweight='bold')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMG_DIR}/2_Entropy_Profile.png", dpi=300)
plt.close()

# =============================================================================
# PLOT 3: TOP PILLARS OF STABILITY
# =============================================================================
print(f"Generazione Plot 3: Top {TOP_N_RESIDUES} Conserved Residues...")

# 1. Ordina per Supporto % Decrescente 
df_sorted = df.sort_values(by=['Support_Pct', 'Entropy_Bits'], ascending=[False, True])
df_top = df_sorted.head(TOP_N_RESIDUES).copy()

# 2. Crea etichetta 
df_top['Label'] = df_top.apply(
    lambda x: f"{x['Consensus_Global_AA']}{int(x['WT_Pos'])}", axis=1
)

# 3. Assegna Colori
bar_colors = []
for _, row in df_top.iterrows():
    if row['Consensus_Global_AA'] == row['WT_AA']:
        bar_colors.append('#1f77b4') # Blu (Native Conservation)
    else:
        bar_colors.append('#d62728') # Rosso (Novel Stability)

fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(df_top['Label'], df_top['Support_Pct'], color=bar_colors, edgecolor='black', alpha=0.85)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

ax.set_title(f"Top {TOP_N_RESIDUES} Most Frequent Residues {TITLE_SUFFIX}", fontsize=15, fontweight='bold')
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
plt.savefig(f"{OUTPUT_IMG_DIR}/3_Top_{TOP_N_RESIDUES}_Residues.png", dpi=300)
plt.close()

print(f"Grafici completati e salvati in: {OUTPUT_IMG_DIR}")
