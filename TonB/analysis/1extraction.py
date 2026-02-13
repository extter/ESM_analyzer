import os
import re
import random
from pathlib import Path
from collections import defaultdict

# ========================
# INPUT PARAM
# ========================

threshold = float(input("Inserisci la threshold (es. 0.97): ").strip())
MAX_PER_RUN = 100
random.seed(42)

root_dir = Path("../markov/runs/")

pattern_cosine = re.compile(r"cosine_to_tonb=([0-9.]+)")
pattern_datetime = re.compile(r"_(\d{8})_(\d{6})")
pattern_time = re.compile(r">time=([0-9\-: ]+)")


extracted_file = Path(f"./extracted/best_than_{threshold:.5f}.txt")
normalized_file = Path(f"./extracted/normalized_best_than_{threshold:.5f}.txt")

extracted_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\nThreshold: {threshold}")
print("---- ESTRAZIONE ----")

# ========================
# STEP 1 — EXTRACTION
# ========================

tot_written = 0

with open(extracted_file, "w") as fout:

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:

            if fname.endswith(".txt") and "sequences" in fname:

                in_path = Path(dirpath) / fname

                m_data = pattern_datetime.search(fname)

                if m_data:
                    d, t = m_data.groups()
                    datetime_run = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
                else:
                    datetime_run = "unknown"

                header = None
                seq_lines = []

                with open(in_path) as fin:

                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith(">"):

                            if header:
                                seq = "".join(seq_lines)
                                m = pattern_cosine.search(header)

                                if m and float(m.group(1)) > threshold:
                                    fout.write(f">time={datetime_run} {header[1:]}\n")
                                    fout.write(seq + "\n")
                                    tot_written += 1

                            header = line
                            seq_lines = []

                        else:
                            seq_lines.append(line)

                # ultima sequenza
                if header:
                    seq = "".join(seq_lines)
                    m = pattern_cosine.search(header)

                    if m and float(m.group(1)) > threshold:
                        fout.write(f">time={datetime_run} {header[1:]}\n")
                        fout.write(seq + "\n")
                        tot_written += 1

print(f"Sequenze estratte: {tot_written}")

# ========================
# STEP 2 — NORMALIZATION
# ========================

print("\n---- NORMALIZZAZIONE ----")

runs = defaultdict(list)

with open(extracted_file) as fin:

    header = None
    seq_lines = []

    for line in fin:
        line = line.strip()

        if line.startswith(">"):

            if header:
                seq = "".join(seq_lines)
                match = pattern_time.search(header)
                time = match.group(1) if match else "unknown"
                runs[time].append((header, seq))

            header = line
            seq_lines = []

        else:
            seq_lines.append(line)

    if header:
        seq = "".join(seq_lines)
        match = pattern_time.search(header)
        time = match.group(1) if match else "unknown"
        runs[time].append((header, seq))


tot_before = sum(len(v) for v in runs.values())
tot_after = 0

with open(normalized_file, "w") as fout:

    for run_time, sequences in runs.items():

        if len(sequences) > MAX_PER_RUN:
            sequences = random.sample(sequences, MAX_PER_RUN)

        for header, seq in sequences:
            fout.write(header + "\n")
            fout.write(seq + "\n")

        tot_after += len(sequences)

print(f"Run trovate: {len(runs)}")
print(f"Sequenze prima: {tot_before}")
print(f"Sequenze dopo : {tot_after}")

print("\nDONE")
print(f"Extracted   → {extracted_file}")
print(f"Normalized  → {normalized_file}")
