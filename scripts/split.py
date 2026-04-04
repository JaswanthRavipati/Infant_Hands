import os
import shutil
import logging
from config import (
    DATASET_IMAGES_TRAIN, DATASET_LABELS_TRAIN,
    DATASET_IMAGES_TRAIN_SPLIT, DATASET_IMAGES_VAL,
    DATASET_LABELS_TRAIN_SPLIT, DATASET_LABELS_VAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Day-stratified split: Day 3 of every subject goes to val, Days 1+2 go to train.
# This tests generalization across time while keeping all subjects represented in val.
VAL_DAY = "day3"

log.info("Val day: %s | Train days: day1, day2", VAL_DAY)

for p in [DATASET_IMAGES_TRAIN_SPLIT, DATASET_IMAGES_VAL,
          DATASET_LABELS_TRAIN_SPLIT, DATASET_LABELS_VAL]:
    os.makedirs(p, exist_ok=True)

images = [f for f in os.listdir(DATASET_IMAGES_TRAIN) if f.endswith(".jpg")]

# Filenames are: {subject}_day{N}_{frame}.jpg
val_files   = [f for f in images if f"_{VAL_DAY}_" in f]
train_files = [f for f in images if f"_{VAL_DAY}_" not in f]

log.info("Total: %d | Train: %d | Val: %d", len(images), len(train_files), len(val_files))


def copy_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        img_src = os.path.join(DATASET_IMAGES_TRAIN, file)
        lbl_src = os.path.join(DATASET_LABELS_TRAIN, file.replace(".jpg", ".txt"))

        shutil.copy(img_src, os.path.join(img_dest, file))

        label_dest = os.path.join(lbl_dest, file.replace(".jpg", ".txt"))

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, label_dest)
        else:
            with open(label_dest, "w"):
                pass


copy_files(train_files, DATASET_IMAGES_TRAIN_SPLIT, DATASET_LABELS_TRAIN_SPLIT)
copy_files(val_files, DATASET_IMAGES_VAL, DATASET_LABELS_VAL)

log.info("Split completed")
