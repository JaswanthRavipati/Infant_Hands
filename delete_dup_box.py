import os

LABELS_DIR = "/Volumes/Seagate/CSCI_B657/data/dataset/labels/train"

for file in os.listdir(LABELS_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABELS_DIR, file)

    with open(path, "r") as f:
        lines = f.readlines()

    # remove duplicates
    unique_lines = list(set(lines))

    with open(path, "w") as f:
        f.writelines(unique_lines)

print("Duplicate boxes removed")