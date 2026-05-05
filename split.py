import os
import random
import shutil

# -----------------------------
# PATHS
# -----------------------------

BASE_PATH = "/Volumes/Seagate/CSCI_B657/data/dataset"

IMAGES_PATH = os.path.join(BASE_PATH, "images/train")
LABELS_PATH = os.path.join(BASE_PATH, "labels/train")

TRAIN_IMG = os.path.join(BASE_PATH, "images/train_split")
VAL_IMG = os.path.join(BASE_PATH, "images/val")

TRAIN_LBL = os.path.join(BASE_PATH, "labels/train_split")
VAL_LBL = os.path.join(BASE_PATH, "labels/val")

# -----------------------------
# SETTINGS
# -----------------------------

SPLIT_RATIO = 0.8
SEED = 42

# -----------------------------
# CREATE FOLDERS
# -----------------------------

for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(p, exist_ok=True)

# -----------------------------
# GET IMAGES
# -----------------------------

images = [f for f in os.listdir(IMAGES_PATH) if f.endswith(".jpg")]

random.seed(SEED)
random.shuffle(images)

split_idx = int(SPLIT_RATIO * len(images))

train_files = images[:split_idx]
val_files = images[split_idx:]

print(f"Total: {len(images)}")
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

# -----------------------------
# COPY FUNCTION
# -----------------------------

def copy_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        img_src = os.path.join(IMAGES_PATH, file)
        lbl_src = os.path.join(LABELS_PATH, file.replace(".jpg", ".txt"))

        shutil.copy(img_src, os.path.join(img_dest, file))

        label_dest = os.path.join(lbl_dest, file.replace(".jpg", ".txt"))

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, label_dest)
        else:
            open(label_dest, "w").close()

# -----------------------------
# EXECUTE SPLIT
# -----------------------------

copy_files(train_files, TRAIN_IMG, TRAIN_LBL)
copy_files(val_files, VAL_IMG, VAL_LBL)

print("Split completed")