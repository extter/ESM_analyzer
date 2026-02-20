import iFeatureOmegaCLI
from scipy.stats import spearmanr
import pandas as pd
import os
import numpy as np

os.makedirs("features_folder", exist_ok=True)

# ==============================
# 1️⃣ Estrazione CTD + altre features
# ==============================
protein = iFeatureOmegaCLI.iProtein("../tonb_and_runs.fasta")
descriptors = ["AAC", "CTDC", "CTDT", "CTriad"]

for desc in descriptors:
    print(f"Estraendo {desc}...")
    protein.get_descriptor(desc)
    protein.to_csv(os.path.join("features_folder", f"{desc}.csv"), index=True, header=True)

# ==============================
# 2️⃣ Creazione master dataframe
# ==============================
dfs = []
for desc in descriptors:
    df_temp = pd.read_csv(os.path.join("features_folder", f"{desc}.csv"), index_col=0)
    dfs.append(df_temp)

master = pd.concat(dfs, axis=1)
master.to_csv(os.path.join("features_folder", "ALL_FEATURES.csv"))
print("\n✅ Master dataframe salvato in features_folder/ALL_FEATURES.csv")

# ==============================
# 3️⃣ Preparazione embedding distance
# ==============================
embedding_distance = np.array([0.995184, 0.997061, 0.988305, 0.993196, 0.994195, 0.994333, 
                               0.996836, 0.995257, 0.995800, 0.996014, 0.993456, 0.997139, 0.994098]) 

# ==============================
# 4️⃣ Separazione TonB e run
# ==============================
df = master.copy()
df_runs = df.drop("TonB")
df_tonb = df.loc[["TonB"]]

# ==============================
# 5️⃣ Calcolo correlazioni Spearman per tutte le colonne CTD
# ==============================
# Qui consideriamo tutte le colonne CTD (CTDC + CTDT + CTriad)
ctd_cols = [col for col in df.columns if col.startswith("CTD")]

results = []
for col in ctd_cols:
    rho, pval = spearmanr(df_runs[col], embedding_distance)
    results.append({
        "Feature": col,
        "Spearman_rho": rho,
        "p_value": pval
    })

results_df = pd.DataFrame(results)

# Ordinamento per p-value crescente
results_df = results_df.sort_values("p_value").reset_index(drop=True)

# Salvataggio dei risultati
results_df.to_csv("features_folder/CTD_embedding_correlation.csv", index=False)
print("\n✅ Risultati correlazioni salvati in features_folder/CTD_embedding_correlation.csv")

# Stampa delle prime 10 per avere un'anteprima
print("\nPrime 10 correlazioni (per p-value):")
print(results_df.head(10))
