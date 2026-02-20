import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# -----------------------------------------------------------------------------
# CONFIGURAZIONE E COSTANTI
# -----------------------------------------------------------------------------
INPUT_FASTA = "../../sequences_analysis/analysis_optimized/5_consensus_data/unaligned.fasta"
OUTPUT_DIR = "./biochemical_analysis/biochemical_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_WT_NAME = "TonB_WT"

SEQ_WT_NAME = "TonB_WT"

# Scala di Idrofobicità di Kyte-Doolittle
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Scala di Carica (a pH fisiologico ~7.4)
CHARGE_SCALE = {
    'K': 1.0, 'R': 1.0, 'D': -1.0, 'E': -1.0, 'H': 0.1, 
    'A': 0, 'N': 0, 'C': 0, 'Q': 0, 'G': 0, 'I': 0, 'L': 0, 
    'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Domini strutturali di TonB per coerenza visiva
TONB_DOMAINS = [
    {"start": 1, "end": 32, "color": "gray", "alpha": 0.1, "label": "TM Anchor (Rigid)"},
    {"start": 65, "end": 105, "color": "orange", "alpha": 0.1, "label": "Proline Linker (Disordered?)"},
    {"start": 150, "end": 239, "color": "green", "alpha": 0.1, "label": "C-Term Barrel (Folded)"}
]

# -----------------------------------------------------------------------------
# FUNZIONI DI CALCOLO
# -----------------------------------------------------------------------------
def calculate_sliding_window(seq, scale, window=9):
    scores = [scale.get(aa, 0) for aa in seq]
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    start_idx = window // 2
    x_axis = np.arange(start_idx, start_idx + len(moving_avg))
    return x_axis, moving_avg

def calculate_sliding_flexibility(seq):
    analyzer = ProteinAnalysis(str(seq))
    try:
        flex = analyzer.flexibility()
        x_axis = np.arange(4, 4 + len(flex))
        return x_axis, flex
    except Exception:
        return [], []

def extract_global_features(seq):
    clean_seq = str(seq).replace("-", "").replace("X", "")
    if len(clean_seq) < 10: return None
        
    analyzer = ProteinAnalysis(clean_seq)
    return {
        "Length": len(clean_seq),
        "Isoelectric_Point": analyzer.isoelectric_point(),
        "Instability_Index": analyzer.instability_index(),
        "GRAVY": analyzer.gravy()
    }

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
if not os.path.exists(INPUT_FASTA):
    print(f"ERRORE: Non trovo il file {INPUT_FASTA}")
    exit()

print(f"Lettura sequenze da {INPUT_FASTA}...")
records = list(SeqIO.parse(INPUT_FASTA, "fasta"))

wt_record = next((r for r in records if r.id == SEQ_WT_NAME), None)
if not wt_record:
    print("ERRORE: Wild Type non trovato nel FASTA (controlla SEQ_WT_NAME).")
    exit()

wt_seq = str(wt_record.seq).replace("-", "")

hydro_data, flex_data, charge_data = [], [], []
global_features = []

print("Calcolo delle feature biochimiche locali e globali...")
for rec in records:
    clean_s = str(rec.seq).replace("-", "")
    
    # Globali
    feats = extract_global_features(clean_s)
    if feats:
        feats["Type"] = "Wild Type" if rec.id == SEQ_WT_NAME else "Generated (ESM-2)"
        feats["ID"] = rec.id
        global_features.append(feats)
    
    # Locali (filtro lunghezza)
    if abs(len(clean_s) - len(wt_seq)) < 20:
        tipo = feats["Type"]
        
        x_h, y_h = calculate_sliding_window(clean_s, KD_SCALE, window=9)
        if len(x_h) > 0: hydro_data.append({"Type": tipo, "x": x_h, "y": y_h})
            
        x_c, y_c = calculate_sliding_window(clean_s, CHARGE_SCALE, window=9)
        if len(x_c) > 0: charge_data.append({"Type": tipo, "x": x_c, "y": y_c})
            
        x_f, y_f = calculate_sliding_flexibility(clean_s)
        if len(x_f) > 0: flex_data.append({"Type": tipo, "x": x_f, "y": y_f})

df_global = pd.DataFrame(global_features)

# -----------------------------------------------------------------------------
# PLOT 1: PROFILI LOCALI (3 Grafici Separati)
# -----------------------------------------------------------------------------
print("Generazione Plot 1: 3 Local Profiles Separati...")
sns.set_style("whitegrid")

plot_configs = [
    {"data": hydro_data, "filename": "1a_Local_Hydrophobicity.png", "ylabel": "Hydrophobicity (Kyte-Doolittle)", "title": "Transmembrane Propensity (Window=9)"},
    {"data": flex_data, "filename": "1b_Local_Flexibility.png", "ylabel": "Flexibility Score (Vihinen)", "title": "Intrinsic Disorder & Flexibility (Window=9)"},
    {"data": charge_data, "filename": "1c_Local_Charge.png", "ylabel": "Net Charge per Residue", "title": "Electrostatic Distribution (Window=9)"}
]

for config in plot_configs:
    fig, ax = plt.subplots(figsize=(16, 5))
    dataset = config["data"]
    
    # Linee generate
    for d in dataset:
        if d["Type"] == "Generated (ESM-2)":
            ax.plot(d["x"], d["y"], color='#1f77b4', alpha=0.35, linewidth=1.5)
            
    # Linea WT
    wt_d = next((d for d in dataset if d["Type"] == "Wild Type"), None)
    if wt_d:
        ax.plot(wt_d["x"], wt_d["y"], color='#d62728', linewidth=3.0, label='TonB Wild Type', zorder=10)
        
    # Domini
    for dom in TONB_DOMAINS:
        ax.axvspan(dom["start"], dom["end"], color=dom["color"], alpha=dom["alpha"], label=dom["label"])
        
    ax.axhline(0 if "Flex" not in config["title"] else 1.0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel("Residue Position", fontsize=12)
    ax.set_ylabel(config["ylabel"], fontsize=12)
    ax.set_title(config["title"], fontsize=14, fontweight='bold', loc='left')
    ax.set_xlim(0, 240)
    
    # Legenda pulita
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{config['filename']}", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: METRICHE GLOBALI (Boxplot Classico)
# -----------------------------------------------------------------------------
print("Generazione Plot 2: Global Biochemical Boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = [
    ("Isoelectric_Point", "pI (Isoelectric Point)", "Il pI determina la carica a pH neutro"),
    ("Instability_Index", "Instability Index", "< 40 indica una proteina stabile in vivo"),
    ("GRAVY", "GRAVY (Global Hydrophobicity)", "Valori positivi = Idrofobica")
]

wt_row = df_global[df_global["Type"] == "Wild Type"].iloc[0]

for i, (col, title, desc) in enumerate(metrics):
    ax = axes[i]
    
    sns.boxplot(
        data=df_global[df_global["Type"] == "Generated (ESM-2)"], 
        y=col, color="white", ax=ax, width=0.4, showfliers=False,
        boxprops=dict(edgecolor='#1f77b4', linewidth=2),
        whiskerprops=dict(color='#1f77b4', linewidth=2),
        capprops=dict(color='#1f77b4', linewidth=2),
        medianprops=dict(color='#1f77b4', linewidth=2)
    )
    
    sns.stripplot(
        data=df_global[df_global["Type"] == "Generated (ESM-2)"], 
        y=col, color="#1f77b4", alpha=0.5, size=5, ax=ax, jitter=True
    )
    
    ax.scatter(0, wt_row[col], color='#d62728', marker='*', s=400, edgecolor='black', zorder=10, label="TonB WT")
    
    if col == "Instability_Index":
        ax.axhline(40, color='red', linestyle=':', label='Stability Threshold (40)')

    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_ylabel(desc, fontsize=11)
    ax.set_xticks([]) 
    ax.legend(loc="best")

plt.suptitle("Global Biochemical Properties: ESM-2 Optimized vs Wild Type", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_Global_Metrics_Boxplots.png", dpi=300)
plt.close()

print(f"Analisi conclusa! Trovi i grafici aggiornati in: {OUTPUT_DIR}")