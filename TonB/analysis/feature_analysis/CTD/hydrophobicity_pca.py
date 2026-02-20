import iFeatureOmegaCLI
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("features_folder", exist_ok=True)

# Estrazione (invariato)
protein = iFeatureOmegaCLI.iProtein("../tonb_and_runs.fasta")
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
print("\n🔥 ANALISI SIGNED Z-SCORE vs TonB (CTD separato per proprietà)")

df = master.copy()
TonB = df.loc['TonB']

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n🔬 PCA su CTDC - Hydrophobicity")

df = master.copy()

# ==============================
# 1️⃣ Selezione colonne hydrophobicity CTDC
# ==============================
hydro_cols = [
    col for col in df.columns
    if col.startswith("CTDC_hydrophobicity")
]

print(f"Numero colonne hydrophobicity trovate: {len(hydro_cols)}")

if len(hydro_cols) == 0:
    print("⚠ Nessuna colonna trovata.")
else:
    df_runs = df.drop("TonB")
    df_tonb = df.loc[["TonB"]]

    # 🔹 Stampare le sequenze su cui facciamo PCA
    sequence_names = df_runs.index.tolist()
    print("\nSequenze usate per PCA (run only, TonB esclusa):")
    for name in sequence_names:
        print(" -", name)

    X_runs = df_runs[hydro_cols].values
    X_tonb = df_tonb[hydro_cols].values

    # ==============================
    # 2️⃣ Standardizzazione sulle run
    # ==============================
    scaler = StandardScaler()
    X_runs_scaled = scaler.fit_transform(X_runs)
    X_tonb_scaled = scaler.transform(X_tonb)

    # ==============================
    # 3️⃣ PCA sulle run
    # ==============================
    pca = PCA(n_components=2)
    pcs_runs = pca.fit_transform(X_runs_scaled)
    pcs_tonb = pca.transform(X_tonb_scaled)

    explained_pc1 = pca.explained_variance_ratio_[0]
    explained_pc2 = pca.explained_variance_ratio_[1]
    print(f"Varianza spiegata da PC1: {explained_pc1:.3f}")
    print(f"Varianza spiegata da PC2: {explained_pc2:.3f}")


    df_runs["Hydro_PC1"] = pcs_runs[:,0]
    df_runs["Hydro_PC2"] = pcs_runs[:,1]
    df_tonb["Hydro_PC1"] = pcs_tonb[:,0]
    df_tonb["Hydro_PC2"] = pcs_tonb[:,1]
    df_pca = pd.concat([df_tonb, df_runs])


    # Salvataggio
    df_pca[["Hydro_PC1","Hydro_PC2"]].to_csv("features_folder/hydrophobicity_pc1_pc2.csv")
    print("✅ PC salvato in features_folder/hydrophobicity_pc1.csv")

    # ==============================
    # 5️⃣ Plot semplice con tutte le sequenze
    # ==============================
    plt.figure(figsize=(8,6))

    # Punti delle run
    plt.scatter(df_runs["Hydro_PC1"], df_runs["Hydro_PC2"],
                c='skyblue', label='Run', s=100)

    # Punto di TonB
    plt.scatter(df_tonb["Hydro_PC1"], df_tonb["Hydro_PC2"],
                c='red', label='TonB', s=150, marker='*')

    # Annotazioni con i nomi delle run
    for i, name in enumerate(df_runs.index):
        plt.text(df_runs["Hydro_PC1"].iloc[i]+0.01,
                df_runs["Hydro_PC2"].iloc[i]+0.01,
                name, fontsize=9)

    plt.xlabel("Hydro PC1")
    plt.ylabel("Hydro PC2")
    plt.title("PCA 2D - Hydrophobicity (CTDC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("features_folder/hydrophobicity_pc1_pc2_plot.png", dpi=200)
    plt.show()




    embedding_distance = np.array([0.995184, 0.997061, 0.988305, 0.993196, 0.994195, 0.994333, 0.996836, 0.995257, 0.995800, 0.996014, 0.993456, 0.997139, 0.994098]) 

    # Coordinate di TonB nello spazio PCA
    tonb_coords = df_tonb[["Hydro_PC1", "Hydro_PC2"]].values[0]

    # Calcolo distanza euclidea per ogni run
    df_runs["Hydro_dist_to_TonB"] = df_runs.apply(
        lambda row: euclidean([row["Hydro_PC1"], row["Hydro_PC2"]], tonb_coords), axis=1
    )

    print("\nDistanze idrofobicità rispetto a TonB (PC1-PC2):")
    print(df_runs[["Hydro_PC1","Hydro_PC2","Hydro_dist_to_TonB"]])

    # --- Confronto con embedding ---
    if 'embedding_distance' in locals():
        rho, pval = spearmanr(df_runs["Hydro_dist_to_TonB"], embedding_distance)
        print(f"\nSpearman correlazione distanza idrofobicità vs distanza embedding: {rho:.3f}, p-value: {pval:.3g}")
    else:
        print("\n⚠ embedding_distance non definito, salta la correlazione")

        # --- Correlazione separata PC1 e PC2 con embedding ---
    rho_pc1, pval_pc1 = spearmanr(df_runs["Hydro_PC1"], embedding_distance)
    rho_pc2, pval_pc2 = spearmanr(df_runs["Hydro_PC2"], embedding_distance)

    print(f"\nSpearman correlazione PC1 vs embedding distance: {rho_pc1:.3f}, p-value: {pval_pc1:.3g}")
    print(f"Spearman correlazione PC2 vs embedding distance: {rho_pc2:.3f}, p-value: {pval_pc2:.3g}")
