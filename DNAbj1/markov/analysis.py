from pathlib import Path

# VERY IMPORTANT: AFTER THIS FILE, RUN THE FOLLOWING LINE ON THE BASH:
# mafft --auto --thread -1 filtered_cosine_gt_095.fasta > aligned_cosine_gt_095.fasta


input_file = "./runs/run_con_file_della_MSA/sequences_over_0.9.txt" #mettere la sequenza dalla cartella RUNS
output_fasta = "./runs/run_con_file_della_MSA/filtered_cosine_gt_095.fasta" #aggiustare il path con la cartella giusta in RUNS

threshold = 0.965

with open(input_file) as f, open(output_fasta, "w") as out:
    header = None
    seq = None

    for line in f:
        line = line.strip()
        if line.startswith(">"):
            header = line
            # estrai cosine_to_tonb
            cosine = float(
                [x.split("=")[1] for x in header.split() if x.startswith("cosine_to_tonb=")][0]
            )
        else:
            seq = line
            if cosine > threshold:
                out.write(header + "\n")
                out.write(seq + "\n")

print("FASTA filtrato scritto in:", output_fasta)


