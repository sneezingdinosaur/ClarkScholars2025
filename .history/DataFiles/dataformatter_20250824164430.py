# save as convert_to_diabetes.py
import csv
import sys

IN_VALUES = {
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
}

def convert(in_path, out_path):
    with open(in_path, newline="", encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        if "study_group" not in reader.fieldnames:
            raise ValueError("Expected a 'study_group' column in the TSV header.")

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            if row.get("study_group") in IN_VALUES:
                row["study_group"] = "diabetes"
            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_diabetes.py input.tsv output.tsv")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
