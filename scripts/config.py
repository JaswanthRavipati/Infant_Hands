import os

# -----------------------------
# ENVIRONMENT DETECTION
# -----------------------------

ON_HPC = os.path.exists("/N/slate/veravi")

# -----------------------------
# DATA PATHS
# -----------------------------

if ON_HPC:
    DATASET_ROOT = "/N/slate/veravi/dataset"
    YOLO_WEIGHTS = "/N/u/veravi/BigRed200/infanthands/runs/detect/train6/weights/best.pt"
    DATA_YAML = "/N/slate/veravi/dataset/dataset.yaml"
    TEST_VIDEO_DIR = "/N/slate/veravi/test_videos"
    TEST_VIDEO_NEW_DIR = "/N/slate/veravi/data_freeplay"
    TEST_OUTPUT_DIR = "/N/slate/veravi/yolo_outputs"
    HAND_CROP_DIR = "/N/slate/veravi/data/hand_crop"
else:
    DATA_ROOT = "/Volumes/Seagate/CSCI_B657/data"
    DATASET_ROOT = "/Volumes/Seagate/CSCI_B657/data/dataset"
    CSV_FILE = "/Volumes/Seagate/CSCI_B657/csv/Spatial_Master_5_subjects.csv"
    YOLO_WEIGHTS = "/N/u/veravi/BigRed200/infanthands/runs/detect/train6/weights/best.pt"
    DATA_YAML = "/Volumes/Seagate/CSCI_B657/dataset/dataset.yaml"
    TEST_VIDEO_DIR = "/Volumes/Seagate/CSCI_B657/test_videos"
    TEST_OUTPUT_DIR = "/Volumes/Seagate/CSCI_B657/yolo_outputs"
    HAND_CROP_DIR = "/Volumes/Seagate/CSCI_B657/data/hand_crop"

DATASET_IMAGES_TRAIN = os.path.join(DATASET_ROOT, "images/train")
DATASET_LABELS_TRAIN = os.path.join(DATASET_ROOT, "labels/train")
DATASET_IMAGES_TRAIN_SPLIT = os.path.join(DATASET_ROOT, "images/train_split")
DATASET_IMAGES_VAL = os.path.join(DATASET_ROOT, "images/val")
DATASET_LABELS_TRAIN_SPLIT = os.path.join(DATASET_ROOT, "labels/train_split")
DATASET_LABELS_VAL = os.path.join(DATASET_ROOT, "labels/val")

# -----------------------------
# SUBJECTS & DAYS
# -----------------------------

SUBJECTS = [25131, 25138, 25176, 25190, 25602]
TRAIN_DAYS = [1, 2, 3]

# -----------------------------
# FRAME EXTRACTION
# -----------------------------

TARGET_FPS = 5

# -----------------------------
# MEDIAPIPE
# -----------------------------

MP_MAX_HANDS = 2
MP_MIN_DETECTION_CONF = 0.4

# Pre-trained weights bundled in repo root
YOLO_PRETRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8m.pt")

# -----------------------------
# YOLO TRAINING
# -----------------------------

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 128
DEVICE = 0
NUM_WORKERS = 8

# -----------------------------
# YOLO INFERENCE
# -----------------------------

CONF_THRESHOLD = 0.25
INFER_IMG_SIZE = 512

# -----------------------------
# HAND CROP
# -----------------------------

CROP_PADDING = 20
CROP_MIN_SIZE = 50
CROP_RESIZE = 224

# -----------------------------
# SPLIT
# -----------------------------

SPLIT_RATIO = 0.8
SPLIT_SEED = 42

# -----------------------------
# DINO CLASSIFICATION
# -----------------------------
DINO_THRESHOLD = 0.4
CROP_RESIZE_DINO=518