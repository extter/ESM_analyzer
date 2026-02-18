import iFeatureOmegaCLI
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("features_folder", exist_ok=True)

# Estrazione (invariato)
protein = iFeatureOmegaCLI.iProtein("fastaprova.fasta")
descriptors = ["AAC", "CTDC", "CTDT", "CTriad"]
for desc in descriptors:
    print(f"Estraendo {desc}...")
    protein.get_descriptor(desc)
    protein.to_csv(os.path.join("features_folder", f"{desc}.csv"), index=True, header=True)

# Master corretto
dfs = []
for desc in descriptors:
    df_temp = pd.read_csv(os.path.join("features_folder", f"{desc}.csv"), index_col=0)
    dfs.append(df_temp)
master = pd.concat(dfs, axis=1)
master.to_csv(os.path.join("features_folder", "ALL_FEATURES.csv"))

# 🔥 ANALISI SIGNED CORRETTA
print("\n🔥 ANALISI SIGNED Z-SCORE vs TonB")
df = master.copy()
TonB = df.loc['TonB']

descs_dict = {
    'AAC': [col for col in df.columns if col.startswith('AAC')],
    'CTDC': [col for col in df.columns if col.startswith('CTDC')],
    'CTDT': [col for col in df.columns if col.startswith('CTDT')],
    'CTriad': [col for col in df.columns if col.startswith('CTriad')]
}

for name, cols in descs_dict.items():
    if not cols: continue
    
    sub_df = df[cols].copy()
    
    # Z-SCORE SIGNED per FEATURE (corretto!)
    z_signed = sub_df.sub(TonB, axis=1).div(sub_df.std(axis=0), axis=1)
    
    # Metriche pulite
    mean_z_run = z_signed.drop('TonB').mean()
    top_inc = mean_z_run.nlargest(3).to_dict()
    top_dec = mean_z_run.nsmallest(3).to_dict()
    corr_mean = sub_df.drop('TonB').corrwith(TonB).abs().mean()
    
    print(f"\n{name}:")
    print(f"  Corr media run: {corr_mean:.3f}")
    print(f"  Top AUMENTI: {top_inc}")
    print(f"  Top DIMINUZIONI: {top_dec}")
    
    # HEATMAP PULITA (prime 20 feature, indici corretti)
    plt.figure(figsize=(12, 6))
    top_cols = cols[:20]  # Prime 20
    sns.heatmap(z_signed[top_cols].T, 
                cmap='RdBu_r', center=0, 
                yticklabels=top_cols,
                cbar_kws={'label': 'Signed Z-score vs TonB'})
    plt.title(f"{name}: Signed Z-score (ROSSO=aumento, BLU=diminuzione)")
    plt.xlabel("Sample")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"features_folder/signed_z_{name.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\n✅ CORRETTO! File + heatmap in features_folder/")
