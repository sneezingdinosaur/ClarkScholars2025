import pandas as pd

# Input and output file names
input_file = "input.csv"
output_file = "output.tsv"

# Read the CSV file
df = pd.read_csv(input_file)

# Save as TSV (tab-separated)
df.to_csv(output_file, sep="\t", index=False)

print(f"Converted {input_file} → {output_file}")
