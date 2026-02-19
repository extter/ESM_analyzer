import argparse
import os
import random
import math
import gc
import warnings
from typing import List

# Ottimizzazione memoria CUDA per evitare frammentazione
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from esm import pretrained
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import joblib
from Bio import SeqIO
from Bio.Align import substitution_matrices

# -----------------------------------------------------------------------------
# COSTANTI GLOBALI E CONFIGURAZIONE
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {DEVICE}")

DEFAULT_CONFIG = {
    'tonb_ref_seq': "MTLDLPRRFPWPTLLSVCIHGAVVAGLLYTSVHQVIELPAPAQPISVTMVTPADLEPPQAVQPPPEPVVEPEPEPEPIPEPPKEAPVVIEKPKPKPKPKPKPVKKVQEQPKRDVKPVESRPASPFENTAPARLTSSTATAATSKPVTSVASGPRALSRNQPQYPARAQALRIEGQVKVKFDVTPDGRVDNVQILSAKPANMFEREVKNAMRRWRYEPGKPGSGIVVNILFKINGTTEIQ",
    'seq_len_range': (150, 700),
    'num_layer': 28,
    'pca_components': 640,
    'pca_batch_size': 64,
    'random_seed': 42,
    
    # Path configurazione
    'fasta_path': './datasets/uniref50_subsample.fasta',
    'output_dir': './datasets',
    'joblibs_dir': './joblibs',
    
    # Parametri mutazione
    'T_blosum': 1.5,
    'p_mut': 0.8,
    'p_ins': 0.1,
    'p_del': 0.1,
    'max_mutations': 20
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
try:
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
except Exception:
    print("Attenzione: impossibile caricare la matrice BLOSUM62. Le mutazioni potrebbero fallire.")
    BLOSUM62 = {}

random.seed(DEFAULT_CONFIG['random_seed'])
np.random.seed(DEFAULT_CONFIG['random_seed'])

os.makedirs(DEFAULT_CONFIG['output_dir'], exist_ok=True)
os.makedirs(DEFAULT_CONFIG['joblibs_dir'], exist_ok=True)

# -----------------------------------------------------------------------------
# 1. GENERAZIONE E CARICAMENTO SEQUENZE
# -----------------------------------------------------------------------------

def generate_random_dataset(n_samples: int) -> List[str]:
    """Genera sequenze proteiche casuali nel range di lunghezza di TonB."""
    print(f"Generazione di {n_samples} sequenze casuali...")
    dataset = []
    min_len = 200 # Limiti fissi per la lunghezza standard di TonB
    max_len = 300
    for _ in tqdm(range(n_samples), desc="Generazione Random"):
        length = random.randint(min_len, max_len)
        dataset.append("".join(random.choice(AA_LIST) for _ in range(length)))
    return dataset


def mutate_residue(seq: str, T: float = DEFAULT_CONFIG['T_blosum']) -> str:
    """Esegue una mutazione puntiforme basata sulla matrice BLOSUM62."""
    seq_list = list(seq)
    idx = random.randrange(len(seq_list))
    original = seq_list[idx]
    scores, choices = [], []
    
    for aa in AA_LIST:
        if aa == original:
            continue
        key = (original, aa) if (original, aa) in BLOSUM62 else (aa, original)
        if key in BLOSUM62:
            scores.append(BLOSUM62[key])
            choices.append(aa)
            
    if not choices:
        return ''.join(seq_list)
        
    exps = [math.exp(s / T) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]
    seq_list[idx] = random.choices(choices, weights=probs)[0]
    return ''.join(seq_list)


def insert_residue(seq: str) -> str:
    """Inserisce un amminoacido casuale in una posizione casuale."""
    seq_list = list(seq)
    idx = random.randrange(len(seq_list) + 1)
    seq_list.insert(idx, random.choice(AA_LIST))
    return ''.join(seq_list)


def delete_residue(seq: str) -> str:
    """Elimina un amminoacido in una posizione casuale (se len > 1)."""
    if len(seq) <= 1:
        return seq
    seq_list = list(seq)
    del seq_list[random.randrange(len(seq_list))]
    return ''.join(seq_list)


def markov_mutation(seq: str) -> str:
    """Seleziona e applica stocasticamente un tipo di mutazione."""
    r = random.random()
    if r < DEFAULT_CONFIG['p_mut']:
        return mutate_residue(seq)
    elif r < DEFAULT_CONFIG['p_mut'] + DEFAULT_CONFIG['p_ins']:
        return insert_residue(seq)
    else:
        return delete_residue(seq)


def generate_tonb_mutations(n_samples: int) -> List[str]:
    """Genera un dataset di sequenze mutate partendo da TonB."""
    print(f"Generazione di {n_samples} sequenze mutate di TonB...")
    dataset = []
    for _ in tqdm(range(n_samples), desc="Mutazione TonB"):
        n_mut = random.randint(1, DEFAULT_CONFIG['max_mutations'])
        seq = DEFAULT_CONFIG['tonb_ref_seq']
        for _ in range(n_mut):
            seq = markov_mutation(seq)
        dataset.append(seq)
    return dataset


def load_uniref_subsample(fasta_path: str, target_n: int, length_range: tuple) -> List[str]:
    """Carica un campione casuale da un file FASTA filtrato per lunghezza."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File FASTA non trovato: {fasta_path}")
    
    print(f"Lettura sequenze UniRef da: {fasta_path}")
    valid_records = []
    min_len, max_len = length_range
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if min_len <= len(record.seq) <= max_len:
                valid_records.append(str(record.seq))
                if len(valid_records) >= target_n * 3:
                    break
                    
    if not valid_records:
        raise ValueError("Nessuna sequenza trovata nel range di lunghezza specificato.")
    return random.sample(valid_records, min(target_n, len(valid_records)))

# -----------------------------------------------------------------------------
# 2. INTEGRAZIONE MODELLO ESM-2
# -----------------------------------------------------------------------------

def initialize_esm_model():
    """Carica il modello ESM-2 e gestisce ambienti Multi-GPU."""
    print("\nCaricamento modello ESM-2...")
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    if torch.cuda.device_count() > 1:
        print(f"Uso {torch.cuda.device_count()} GPU con DataParallel")
        model = nn.DataParallel(model)
    else:
        print("Uso singola GPU")
        
    return model, batch_converter


def forward_robust(model: nn.Module, tokens: torch.Tensor, layer: int):
    """Esegue il forward pass aggirando i bug di DataParallel con batch=1."""
    if isinstance(model, nn.DataParallel) and tokens.shape[0] == 1:
        # Se usiamo DataParallel, forziamo la singola GPU per batch size 1
        device_0 = f"cuda:{model.device_ids[0]}" if hasattr(model, 'device_ids') else "cuda:0"
        single_model = model.module.to(device_0)
        return single_model(tokens.to(device_0), repr_layers=[layer], return_contacts=False)
    
    return model(tokens, repr_layers=[layer], return_contacts=False)


@torch.no_grad()
def get_residue_embeddings_batch(sequences: List[str], model: nn.Module, batch_converter: callable) -> List[np.ndarray]:
    """Estrae gli embedding per residuo da ESM-2, ignorando <cls> ed <eos>."""
    data = [("seq", s) for s in sequences]
    _, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)
    
    out = forward_robust(model, batch_tokens, DEFAULT_CONFIG['num_layer'])
    token_reps = out["representations"][DEFAULT_CONFIG['num_layer']]
    
    # Taglio [1:1+len] per rimuovere il token <cls> a pos 0 e <eos> alla fine
    return [token_reps[i, 1:1+len(s)].detach().cpu().numpy() for i, s in enumerate(batch_strs)]

# -----------------------------------------------------------------------------
# 3. ROUTINE DI FITTING PCA
# -----------------------------------------------------------------------------

def execute_incremental_pca(sequences: List[str], dataset_name: str, model: nn.Module, batch_converter: callable):
    """Esegue IncrementalPCA sulle sequenze fornite e salva il modello."""
    print(f"\n--- Inizio IncrementalPCA per {dataset_name} ({len(sequences)} sequenze) ---")
    
    ipca = IncrementalPCA(n_components=DEFAULT_CONFIG['pca_components'], batch_size=None)
    batch_size = DEFAULT_CONFIG['pca_batch_size']
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Fit IncrementalPCA"):
        torch.cuda.empty_cache()
        gc.collect()
        
        batch_seqs = sequences[i : i + batch_size]
        try:
            emb_list = get_residue_embeddings_batch(batch_seqs, model, batch_converter)
            X_batch = np.concatenate(emb_list, axis=0)
            ipca.partial_fit(X_batch)
            del X_batch, emb_list
        except Exception as e:
            print(f"Errore durante il batch {i}: {e}")

    print(f"Fit IncrementalPCA Completato! Varianza spiegata (somma): {ipca.explained_variance_ratio_.sum():.4f}")
    
    # Salva il modello e i metadati
    pca_filename = os.path.join(DEFAULT_CONFIG['joblibs_dir'], f"{dataset_name}_ipca_fitted.joblib")
    meta_filename = os.path.join(DEFAULT_CONFIG['joblibs_dir'], f"{dataset_name}_pca_metadata.joblib")

    joblib.dump(ipca, pca_filename)
    joblib.dump({
        'pca_components': DEFAULT_CONFIG['pca_components'],
        'n_sequences_used': len(sequences),
        'model_name': 'esm2_t33_650M_UR50D',
        'layer_used': DEFAULT_CONFIG['num_layer'],
        'dataset_type': dataset_name
    }, meta_filename)

    print(f"PCA salvata in {pca_filename}")

# -----------------------------------------------------------------------------
# ENTRY POINT PRINCIPALE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Incremental PCA sugli embedding di ESM-2.")
    parser.add_argument("--dataset", choices=["random", "tonb", "uniref", "combined"], required=True, 
                        help="Seleziona il tipo di dataset da generare e fittare.")
    parser.add_argument("--samples", type=int, default=100000, 
                        help="Numero di sequenze da processare (ignorato per 'combined' se forzato a bilanciato).")
    args = parser.parse_args()

    # Determina le sequenze in base al dataset richiesto
    sequences = []
    
    if args.dataset == "random":
        csv_path = os.path.join(DEFAULT_CONFIG['output_dir'], "Random_dataset.csv")
        if not os.path.exists(csv_path):
            sequences = generate_random_dataset(args.samples)
            pd.DataFrame(sequences, columns=["sequence"]).to_csv(csv_path, index=False, header=False)
        else:
            print(f"Caricamento dataset Random esistente da {csv_path}")
            sequences = pd.read_csv(csv_path, header=None, names=["sequence"])["sequence"].tolist()[:args.samples]

    elif args.dataset == "tonb":
        csv_path = os.path.join(DEFAULT_CONFIG['output_dir'], "TonB_mutations_dataset.csv")
        if not os.path.exists(csv_path):
            sequences = generate_tonb_mutations(args.samples)
            pd.DataFrame(sequences, columns=["sequence"]).to_csv(csv_path, index=False, header=False)
        else:
            print(f"Caricamento dataset TonB esistente da {csv_path}")
            sequences = pd.read_csv(csv_path, header=None, names=["sequence"])["sequence"].tolist()[:args.samples]

    elif args.dataset == "uniref":
        sequences = load_uniref_subsample(DEFAULT_CONFIG['fasta_path'], args.samples, DEFAULT_CONFIG['seq_len_range'])

    elif args.dataset == "combined":
        combined_csv = os.path.join(DEFAULT_CONFIG['output_dir'], "dataset_proteine_balanced_150k.csv")
        # Per combined forziamo blocchi bilanciati da 50k come nello script originale
        n_block = 50000
        if not os.path.exists(combined_csv):
            print("Generazione dataset bilanciato (Random, TonB, UniRef)...")
            
            u_seqs = load_uniref_subsample(DEFAULT_CONFIG['fasta_path'], n_block, DEFAULT_CONFIG['seq_len_range'])
            
            r_csv = os.path.join(DEFAULT_CONFIG['output_dir'], "Random_dataset.csv")
            if os.path.exists(r_csv):
                r_seqs = random.sample(pd.read_csv(r_csv, header=None)["sequence"].tolist(), min(n_block, len(pd.read_csv(r_csv))))
            else:
                r_seqs = generate_random_dataset(n_block)
                
            t_csv = os.path.join(DEFAULT_CONFIG['output_dir'], "TonB_mutations_dataset.csv")
            if os.path.exists(t_csv):
                t_seqs = random.sample(pd.read_csv(t_csv, header=None)["sequence"].tolist(), min(n_block, len(pd.read_csv(t_csv))))
            else:
                t_seqs = generate_tonb_mutations(n_block)
                
            sequences = u_seqs + r_seqs + t_seqs
            random.shuffle(sequences)
            pd.DataFrame(sequences, columns=["sequence"]).to_csv(combined_csv, index=False, header=False)
            del u_seqs, r_seqs, t_seqs
        else:
            print(f"Caricamento dataset combinato esistente da {combined_csv}")
            sequences = pd.read_csv(combined_csv, header=None, names=["sequence"])["sequence"].tolist()

    if not sequences:
        print("Errore: Nessuna sequenza generata o caricata. Uscita.")
        exit(1)

    # Carica Modello ed esegui PCA
    esm_model, batch_converter = initialize_esm_model()
    
    # Capitalizza il nome del dataset per il salvataggio dei file (es. Random, Tonb, Combined)
    execute_incremental_pca(sequences, args.dataset.capitalize(), esm_model, batch_converter)