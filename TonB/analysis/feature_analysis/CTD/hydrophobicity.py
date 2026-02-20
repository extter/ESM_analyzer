import iFeatureOmegaCLI
import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 0️⃣ Cartella output
# ==============================
os.makedirs("features_folder", exist_ok=True)

# ==============================
# 1️⃣ Estrazione feature
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
print("✅ Master features salvato in features_folder/ALL_FEATURES.csv")

# ==============================
# 3️⃣ Selezione idrofobicità CTDC
# ==============================
df = master.copy()
df_runs = df.drop("TonB")
df_tonb = df.loc[["TonB"]]

hydro_cols = [col for col in df.columns if col.startswith("CTDC_hydrophobicity")]
print(f"Colonne idrofobicità trovate: {len(hydro_cols)}")
for col in hydro_cols:
    print(" -", col)

# ==============================
# 4️⃣ Distanza embedding
# ==============================
embedding_distance = np.array([0.995184, 0.997061, 0.988305, 0.993196, 
                               0.994195, 0.994333, 0.996836, 0.995257, 
                               0.995800, 0.996014, 0.993456, 0.997139, 
                               0.994098]) 

# ==============================
# 5️⃣ Calcolo correlazioni Spearman per ciascuna scala
# ==============================
results = []

for col in hydro_cols:
    # Differenza rispetto a TonB (puoi anche usare il valore diretto)
    hydro_values = df_runs[col].values
    tonb_value = df_tonb[col].values[0]
    
    # Se vuoi distanza vs TonB:
    diff_to_tonb = np.abs(hydro_values - tonb_value)
    
    # Correlazione Spearman con embedding_distance
    rho, pval = spearmanr(diff_to_tonb, embedding_distance)
    results.append({"Scale": col, "Spearman_rho": rho, "p_value": pval})

df_corr = pd.DataFrame(results)
print("\n📊 Risultati correlazioni Spearman per ciascuna scala di idrofobicità:")
print(df_corr)

# ==============================
# 6️⃣ Plot delle correlazioni
# ==============================
plt.figure(figsize=(12,5))
sns.barplot(x="Scale", y="Spearman_rho", data=df_corr, palette="coolwarm")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Spearman ρ vs embedding distance")
plt.title("Correlazione idrofobicità CTDC vs embedding distance")
plt.tight_layout()
plt.savefig("features_folder/hydro_scale_correlation.png", dpi=200)
plt.show()
