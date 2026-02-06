import os
import re
import Bio
from Bio import pairwise2
from Bio.Align import substitution_matrices
import pandas as pd
from pathlib import Path

CONFIG = { 
    'input_path' : './runs',
    'threshold' : 0.98,
    'output_path' : './bestsimilarity_dataset'
}

def extract_high_cosine_sequences(config):
    """
    Estrae da tutti i file txt nelle sottocartelle le sequenze con cosine_to_tonb >= threshold
    e le salva in un unico file output.
    """
    input_folder = Path(config['input_path'])
    output_file = Path(config['output_path']) / 'best_sequences.txt'
    threshold = config['threshold']
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    high_cosine_seqs = []
    total_processed = 0
    
    
    # Scansiona ricorsivamente tutti i file txt
    txt_files = list(input_folder.rglob("*.txt"))
    print(f"Trovati {len(txt_files)} file txt")
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] Processing: {txt_file.name}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Errore lettura: {e}")
            continue
        
        pattern = r'>(step=\d+\s+cosine_to_tonb=([0-9.]+)\s+length=\d+)\s*([A-Z\n]+?)(?=>|$)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        file_high_count = 0
        for header_full, cosine_str, sequence in matches:
            total_processed += 1
            try:
                cosine = float(cosine_str)
                if cosine >= threshold:
                    # Pulisci sequenza
                    clean_seq = re.sub(r'[\n\s]+', '', sequence.strip())
                    if len(clean_seq) > 10:  # Sequenza valida
                        high_cosine_seqs.append(f">{header_full}\n{clean_seq}")
                        file_high_count += 1
            except ValueError:
                continue
        
        print(f"{len(matches)} totali | {file_high_count} alte cosine (>= {threshold})")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(high_cosine_seqs) + "\n")
    
    print(f"\n COMPLETATO!")
    print(f"Sequenze estratte: {len(high_cosine_seqs)}")
    print(f"Totale processate: {total_processed:,}")
    print(f"Media/file: {len(high_cosine_seqs)/len(txt_files):.1f}")
    
    return len(high_cosine_seqs)

def pairwise_alignment(best_sequences_file):
    """
    Allineamento pairwise FIXATO (no pairwise2.identity)
    """
    print("\n🔗 ALIGNMENT PAIRWISE (BLOSUM62)")
    
    # Carica sequenze
    with open(best_sequences_file, 'r') as f:
        lines = f.readlines()
    
    sequences = []
    headers = []
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines): break
        header = lines[i].strip()[1:]  # Rimuovi >
        seq = lines[i+1].strip()
        if len(seq) > 10 and seq.isalpha():
            headers.append(header[:30] + "..." if len(header)>30 else header)
            sequences.append(seq)
    
    n_seqs = len(sequences)
    print(f"📊 {n_seqs} sequenze valide")
    
    if n_seqs < 2:
        print("❌ Almeno 2 sequenze necessarie!")
        return None, []
    
    # BLOSUM62
    blosum62 = substitution_matrices.load("BLOSUM62")
    
    # Matrix + top alignments
    distance_matrix = [[0]*n_seqs for _ in range(n_seqs)]
    best_alignments = []
    
    print("📈 Pairwise...")
    for i in range(n_seqs):
        print(f"  Seq {i+1}/{n_seqs}: {headers[i]}")
        for j in range(n_seqs):
            if i == j:
                distance_matrix[i][j] = 100.0
                continue
            
            # Allineamento GLOBAL (migliore per proteine simili)
            alignments = pairwise2.align.globalds(
                sequences[i], sequences[j], 
                blosum62, -10, -0.5  # Gap più soft
            )
            best = alignments[0]
            
            # IDENTITY FIX: calcola manualmente
            aligned_seq1 = best.seqA
            aligned_seq2 = best.seqB
            matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
            total_positions = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' or b != '-')
            identity = (matches / total_positions * 100) if total_positions > 0 else 0
            
            distance_matrix[i][j] = identity
            
            # Top 10
            if len(best_alignments) < 10 or identity > 85:
                best_alignments.append({
                    'pair': f"{headers[i]} vs {headers[j]}",
                    'identity': f"{identity:.1f}%",
                    'len': len(sequences[i]),
                    'alignment': pairwise2.format_alignment(*best)[:200] + "..."  # Truncate
                })
    
    # Salva
    output_dir = Path(CONFIG['output_path'])
    
    # Matrix CSV
    import csv
    with open(output_dir / 'identity_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [f"S{i+1}" for i in range(n_seqs)])
        for i, row in enumerate(distance_matrix):
            writer.writerow([headers[i]] + [f"{x:.1f}" for x in row])
    
    # Top TXT
    best_alignments.sort(key=lambda x: float(x['identity'][:-1]), reverse=True)
    with open(output_dir / 'top_alignments.txt', 'w') as f:
        f.write(f"🏆 TOP {len(best_alignments)} PAIRWISE ALIGNMENTS\n\n")
        for i, align in enumerate(best_alignments[:10], 1):
            f.write(f"{i}. {align['pair']}\n")
            f.write(f"   Identità: {align['identity']} | Len: {align['len']}\n")
            f.write(f"   {align['alignment']}\n\n")
    
    print(f"✅ {output_dir / 'identity_matrix.csv'}")
    print(f"✅ {output_dir / 'top_alignments.txt'}")
    
    # Stats
    ids = [distance_matrix[i][j] for i in range(n_seqs) for j in range(i+1, n_seqs)]
    print(f"\n📊 STATS:")
    print(f"  Media:    {sum(ids)/len(ids):.1f}%")
    print(f"  Max:      {max(ids):.1f}%")
    print(f"  Coppie >90%: {sum(1 for x in ids if x>90)}")
    
    return distance_matrix, best_alignments

# =========================
# ESECUZIONE
# =========================
best_file = Path(CONFIG['output_path']) / 'best_sequences.txt'
if best_file.exists():
    print("📂 Uso best_sequences.txt esistente")
else:
    n_extracted = extract_high_cosine_sequences(CONFIG)
    if n_extracted == 0:
        print("❌ Nessuna sequenza estratta. Abbassa threshold!")
        exit()

# 2. Allineamento pairwise
df_dist, top_aligns = pairwise_alignment(best_file)

print("\n🎉 PIPELINE COMPLETA!")
print("✅ best_sequences.txt     (raw)")
print("✅ pairwise_identity_matrix.csv  (heatmap)")
print("✅ top_alignments.txt     (dettagli)")