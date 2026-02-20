import iFeatureOmegaCLI
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import re
from Bio import SeqIO

# ==============================
# 0️⃣ Impostazioni
# ==============================
fasta_file = "./run7_modificato.fasta"   # il tuo nuovo FASTA
output_folder = "features_folder"
os.makedirs(output_folder, exist_ok=True)

descriptors = ["AAC", "CTDC", "CTDT", "CTriad"]

# ==============================
# 1️⃣ Estrazione features con iFeatureOmegaCLI
# ==============================
protein = iFeatureOmegaCLI.iProtein(fasta_file)
for desc in descriptors:
    print(f"Estraendo {desc}...")
    protein.get_descriptor(desc)
    protein.to_csv(os.path.join(output_folder, f"{desc}.csv"), index=True, header=True)

# ==============================
# 2️⃣ Creazione master dataframe
# ==============================
dfs = []
for desc in descriptors:
    df_temp = pd.read_csv(os.path.join(output_folder, f"{desc}.csv"), index_col=0)
    dfs.append(df_temp)

master = pd.concat(dfs, axis=1)
master.to_csv(os.path.join(output_folder, "ALL_FEATURES.csv"))
print("\n✅ Master dataframe salvato in", os.path.join(output_folder, "ALL_FEATURES.csv"))

# ==============================
# 3️⃣ Estrazione embedding distance dai nomi delle sequenze
# ==============================
embedding_similarity = []

for record in SeqIO.parse(fasta_file, "fasta"):
    m = re.search(r"_sim_(\d+\.\d+)", record.id)
    if m:  # solo per le sequenze “run”
        embedding_similarity.append(float(m.group(1)))


embedding_similarity = np.array(embedding_similarity)
embedding_distance = 1 - embedding_similarity

# ==============================
# 4️⃣ Separazione TonB e run
# ==============================
df = master.copy()

# Separiamo TonB
df_tonb = df.loc[["TonB"]]  # TonB come riferimento
df_runs = df.drop("TonB")    # tutte le altre run

# Controllo che numero di embedding corrisponda alle run
assert len(df_runs) == len(embedding_similarity), "Numero sequenze e embedding distance non coincide!"



# ==============================
# 5️⃣ Calcolo distanza Euclidea rispetto a TonB
# ==============================
tonb_vector = df_tonb.values[0]  # vettore di riferimento
df_runs["dist_to_TonB"] = df_runs.apply(
    lambda row: np.linalg.norm(row.values - tonb_vector),
    axis=1
)

# ==============================
# 6️⃣ Correlazione Spearman e Pearson tra distanze
# ==============================
rho, pval = spearmanr(df_runs["dist_to_TonB"], embedding_distance)
r_pearson, pval_pearson = pearsonr(df_runs["dist_to_TonB"], embedding_distance)

print(f"\nSpearman correlazione distanza feature vs embedding distance: {rho:.3f}, p-value: {pval:.3g}")
print(f"Pearson correlazione distanza feature vs embedding distance: {r_pearson:.3f}, p-value: {pval_pearson:.3g}")

# ==============================
# 7️⃣ Correlazioni separate per feature
# ==============================
results = []
for col in df_runs.columns[:-1]:  # escludi dist_to_TonB
    rho_f, pval_f = spearmanr(df_runs[col], embedding_distance)
    results.append({
        "Feature": col,
        "Spearman_rho": rho_f,
        "p_value": pval_f
    })

results_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
results_df.to_csv(os.path.join(output_folder, "feature_embedding_correlation.csv"), index=False)

print("\n✅ Risultati correlazioni salvati in", os.path.join(output_folder, "feature_embedding_correlation.csv"))
print("\nPrime 10 correlazioni (per p-value):")
print(results_df.head(10))

# ==============================
# 8️⃣ Grafico distanza feature vs embedding distance
# ==============================
plt.figure(figsize=(7,6))
sns.regplot(
    x=df_runs["dist_to_TonB"], 
    y=embedding_distance, 
    ci=None, 
    scatter_kws={"s":100, "color":"skyblue"}, 
    line_kws={"color":"red"}
)
plt.xlabel("Distanza Euclidea feature vs TonB")
plt.ylabel("Cosine distance embedding vs TonB")
plt.title(f"Distanza feature vs embedding\nSpearman: {rho:.3f}, Pearson: {r_pearson:.3f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "distance_vs_embedding_per_run.png"), dpi=200)
plt.show()
