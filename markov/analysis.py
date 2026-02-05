from pathlib import Path

input_file = "sequences_over_0.9.txt"
output_fasta = "filtered_cosine_gt_095.fasta"

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


