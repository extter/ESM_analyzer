import iFeatureOmegaCLI
import pandas as pd
import os

# Crea cartella (se non esiste)
os.makedirs("features_folder", exist_ok=True)

# Carica FASTA
protein = iFeatureOmegaCLI.iProtein("fastaprova.fasta")  # Batch 3 seq

# Lista descrittori
descriptors = ["AAC", "CTDC", "CTDT", "CTriad"]

# Estrai e salva OGNI file nella cartella
for desc in descriptors:
    print(f"Estraendo {desc}...")
    protein.get_descriptor(desc)
    filename = os.path.join("features_folder", f"{desc}.csv")
    protein.to_csv(filename, index=True, header=True)
    print(f"Salvato: {filename}")

# Bonus: unisci TUTTI in UN file master
print("Unendo...")
dfs = []
for desc in descriptors:
    df = pd.read_csv(os.path.join("features_folder", f"{desc}.csv"), index_col=0)
    dfs.append(df)
master = pd.concat(dfs, axis=1)
master.to_csv(os.path.join("features_folder", "ALL_FEATURES.csv"))
print("Master file: features_folder/ALL_FEATURES.csv ✅")

print("TUTTI i file sono in 'features_folder/'!")
