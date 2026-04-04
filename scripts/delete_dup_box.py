import os
import logging
from config import DATASET_LABELS_TRAIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

for file in os.listdir(DATASET_LABELS_TRAIN):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(DATASET_LABELS_TRAIN, file)

    with open(path, "r") as f:
        lines = f.readlines()

    unique_lines = list(dict.fromkeys(lines))  # order-preserving dedup

    with open(path, "w") as f:
        f.writelines(unique_lines)

log.info("Duplicate boxes removed")
