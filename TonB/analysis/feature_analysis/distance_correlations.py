import iFeatureOmegaCLI
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

os.makedirs("features_folder", exist_ok=True)

# ==============================
# 1️⃣ Estrazione features
# ==============================
protein = iFeatureOmegaCLI.iProtein("./tonb_and_runs.fasta")
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
# Qui inserisci le distanze estratte dai nomi delle sequenze
# Ordine delle sequenze deve corrispondere a quello del dataframe senza TonB
embedding_similarity = np.array([
    0.995184, 0.997061, 0.988305, 0.993196, 0.994195, 
    0.994333, 0.996836, 0.995257, 0.995800, 0.996014, 
    0.993456, 0.997139, 0.994098
])

embedding_distance = 1 - embedding_similarity


# ==============================
# 4️⃣ Separazione TonB e run
# ==============================
df = master.copy()
df_runs = df.drop("TonB")
df_tonb = df.loc[["TonB"]]

# ==============================
# 5️⃣ Calcolo distanza Euclidea rispetto a TonB per ogni run
# ==============================
tonb_vector = df_tonb.values[0]  # tutte le feature
df_runs["dist_to_TonB"] = df_runs.apply(
    lambda row: np.linalg.norm(row.values - tonb_vector),
    axis=1
)

# ==============================
# 6️⃣ Correlazione Spearman tra distanza e embedding distance
# ==============================
rho, pval = spearmanr(df_runs["dist_to_TonB"], embedding_distance)
print(f"\nSpearman correlazione distanza feature vs embedding distance: {rho:.3f}, p-value: {pval:.3g}")
r_pearson, pval_pearson = pearsonr(df_runs["dist_to_TonB"], embedding_distance)
print(f"Pearson correlazione distanza feature vs embedding distance: {r_pearson:.3f}, p-value: {pval_pearson:.3g}")


# ==============================
# 7️⃣ Correlazione separata per feature
# ==============================
results = []
for col in df_runs.columns[:-1]:  # escludiamo la colonna dist_to_TonB
    rho_f, pval_f = spearmanr(df_runs[col], embedding_distance)
    results.append({
        "Feature": col,
        "Spearman_rho": rho_f,
        "p_value": pval_f
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("p_value").reset_index(drop=True)
results_df.to_csv("features_folder/feature_embedding_correlation.csv", index=False)

print("\n✅ Risultati correlazioni salvati in features_folder/feature_embedding_correlation.csv")
print("\nPrime 10 correlazioni (per p-value):")
print(results_df.head(10))



plt.figure(figsize=(12,8))
sns.regplot(x=df_runs["dist_to_TonB"], y=embedding_distance, ci=None, scatter_kws={"s":100, "color":"black"}, line_kws={"color":"red"})
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel("Feature distance protein vs TonB", size=16, color='black')
plt.ylabel("Embedding distance protein vs TonB", size=16, color='black')
plt.title(
    f"Correlation between embedding and feature distances (best of 13 runs)\n"
    f"Spearman: ρ = {rho:.3f} (p = {pval:.2e})   |   "
    f"Pearson: r = {r_pearson:.3f} (p = {pval_pearson:.2e})",
    size=18
)
plt.grid(True)
plt.tight_layout()
plt.savefig("features_folder/distance_vs_embedding.png", dpi=200)
plt.show()
