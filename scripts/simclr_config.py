import os

# -----------------------------
# ENVIRONMENT DETECTION
# -----------------------------

ON_HPC = os.path.exists("/N/slate/veravi")

# -----------------------------
# DATA PATHS
# -----------------------------

if ON_HPC:
    HAND_CROP_DIR       = "/N/slate/veravi/data/hand_crop"
    LABELED_DATA_DIR    = "/N/slate/veravi/data/labeled_crops"
    CHECKPOINT_DIR      = "/N/slate/veravi/simclr_checkpoints"
    FINETUNE_CKPT_DIR   = "/N/slate/veravi/finetune_checkpoints"
else:
    HAND_CROP_DIR       = "/Volumes/Seagate/CSCI_B657/data/hand_crop"
    LABELED_DATA_DIR    = "/Volumes/Seagate/CSCI_B657/data/labeled_crops"
    CHECKPOINT_DIR      = "/Volumes/Seagate/CSCI_B657/simclr_checkpoints"
    FINETUNE_CKPT_DIR   = "/Volumes/Seagate/CSCI_B657/finetune_checkpoints"

# -----------------------------
# CLASSES
# -----------------------------

CLASS_NAMES = ["baby", "adult", "other_kids"]
NUM_CLASSES  = len(CLASS_NAMES)

# -----------------------------
# SIMCLR PRE-TRAINING
# -----------------------------

SIMCLR_BACKBONE     = "resnet50"
SIMCLR_OUT_DIM      = 128        # projection head output dimension
SIMCLR_TEMPERATURE  = 0.07
SIMCLR_EPOCHS       = 200
SIMCLR_BATCH_SIZE   = 256
SIMCLR_LR           = 0.3        # linearly scaled: LR * batch_size / 256
SIMCLR_WEIGHT_DECAY = 1e-4
SIMCLR_NUM_WORKERS  = 8
SIMCLR_IMG_SIZE     = 224

# -----------------------------
# SIMCLR FINE-TUNING
# -----------------------------

FINETUNE_EPOCHS      = 100
FINETUNE_LR          = 0.01
FINETUNE_BATCH_SIZE  = 128
FINETUNE_NUM_WORKERS = 8
FINETUNE_VAL_SPLIT   = 0.2       # fraction of labeled data held out for val
FINETUNE_SEED        = 42
