
***

# 🧬 ESM Analyzer — TonB Protein Embedding \& Evolution Simulation

## ⚙️ Preliminary Setup

Before running any scripts, set up your environment with the following commands:

```bash
conda create -n bio python=3.11.13
pip install torch fair-esm biopython numpy scipy tqdm pandas joblib seaborn "fastcore<1.9,>=1.8.0" fastai==2.8.4
pip install -U scikit-learn==1.7.2
```

> ⚠️ **Note:**
> When running the Python scripts for the first time, the **ESM model** will be downloaded automatically (about 15 minutes).
> After that, it will be cached in the `.cache/` directory.

***

## 📂 Folder Structure

### `sequences/`

Contains text files with the amino acid sequences of **TonB** and other proteins.

***

### `layer_selection/`

This script:

- Generates **conservative** and **non-conservative** mutations.
- Computes the **ratio between cosine similarities** (TonB vs. mutants).
- Analyzes multiple ESM layers (**20 to 33**).

> 💡 **Tip:**
> - Avoid layers beyond 33 — they are highly **anisotropic**.
> - Avoid very early layers — they contain little biological meaning.
> - For reference, **layer 28** was used in the TonB analysis.

***

### `cosine_similarity/`

Creates **histograms** of cosine similarity for varying numbers of TonB mutations.

> Uses **mean pooling** as the embedding aggregation method.

***

### `pca/`

Performs **Principal Component Analyses (PCA)** on four datasets:

1. `uniref50` subsample
2. Random sequences
3. TonB mutations
4. Combined dataset (50k samples from each)

Datasets are stored in:
`ESM_analyzer/pca/datasets/`

#### 🧩 Datasets

- The **uniref50 subsample** must be downloaded manually from (due to GitHub file size limitation).
- Move the downloaded file to:

```
/pca/datasets/uniref50_subsample.fasta
```

> ⚠️ The path **must be exact**, so that `.gitignore` can exclude it correctly during future commits.

other stuff on the pca.....

***

### `segment_pooling/`

**Mean pooling** was discarded because it collapses embeddings into a **narrow cone** in embedding space.

Instead, this module implements **segment-based pooling**:

1. Embed residues individually.
2. Divide embeddings into **segments** (possibly by domains).
3. Apply **mean pooling** within segments.
4. **Normalize** segment embeddings.
5. Compute sequence similarity as the **mean cosine similarity** of corresponding segments.

#### 🧪 Files

- `test.py` — Checks environment setup and verifies segment pooling functionality.
- `segment_number_selection.py` — Evaluates distances between conservative and non-conservative mutations for various segment counts.

> 📈 **Tradeoff:**
> - Fewer segments → more averaging, less local detail.
> - More segments → higher resolution, more noise.

***

### `markov/` *(Core module)*

The **heart of the project**: a simulation of protein evolution in embedding space using **Markov Chains**.

#### Main Script: `chain_cycle.py`

- Operates in the **PCA-transformed embedding space** (640 components).
- **Cosine similarity** computed using segment-based pooling.
- Starts from a random **239–AA sequence**.


#### 🔁 Algorithm Details

- **Mutations per step:** `K_PROPOSALS = 8`
- **Insertion/deletion rate:** 1% each
- **Scoring model:** `BLOSUM62`, temperature = 1.7
- **Selection:** Choose randomly from **TOP_M = 4** top candidates
- **Acceptance rule:**
    - If Δ(cosine) > 0 → accept
    - If Δ(cosine) < 0 → accept with **Metropolis criterion** (β = 800)
- If similarity plateaus for **lookback = 80** steps → reduce β to 100 to escape local minima


#### 🧭 Simulation Parameters

- Default steps: **7,500**
- Convergence typically around steps **4,000–5,000**
- Saves sequences with **cosine similarity > 0.90**

Each run generates:

- A **graph** of the Markov chain (≈10 min)
- A **.txt file** with high-similarity sequences
- Stored under: `ESM_analyzer/markov/runs/`

> 🧠 If the folder lacks a `.txt` file, the simulation did not reach 0.9 similarity.

***

### ⚡ Performance Benchmarks

| Device | GPU RAM | Speed (100 steps, k=8, m=4) |
| :-- | :-- | :-- |
| RTX 3070 Laptop | 8 GB | ~40 s |
| RTX 3060 Desktop | 12 GB | ~48 s |
| Kaggle Notebook (2×T4 GPUs) | max 15 Gb per GPU (?) | ~100 s |


***

## 🧪 Project Status

**Analysis section:** *Under construction... MSA for allignment of the plateau region ......*

***


