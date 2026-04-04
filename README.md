# Infant Hands

A computer vision pipeline for detecting and tracking infant hand-object interactions from video data. Built for CSCI B657 (Computer Vision).

---

## Overview

The pipeline extracts video frames, filters them using MediaPipe and CSV annotations, generates YOLO bounding box labels, allows manual correction via CVAT, then fine-tunes a YOLOv8 model to detect infant hands.

---

## Pipeline

### Part 1 — Automated (run `run_pipeline_part1.sh`)

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `hands.py` | Extract frames from MP4 videos at 5 FPS |
| 2 | `detect_hands.py` | Filter frames to only those containing hands (MediaPipe + CSV annotations) |
| 3 | `generate_hand_labels.py` | Generate YOLO bounding box labels from MediaPipe landmarks |
| 4 | `delete_dup_box.py` | Remove duplicate bounding boxes from label files |

### Manual Step — CVAT Annotation Review

After Part 1, review and correct the auto-generated labels in CVAT before training:

1. Import images from `dataset/images/train/`
2. Import labels from `dataset/labels/train/` (YOLO format)
3. Review bounding boxes — fix misses, add padding, correct bad detections
4. Export corrected labels back to `dataset/labels/train/` (YOLO format)

### Part 2 — Post-CVAT (run `run_pipeline_part2.sh`)

| Stage | Script | Description |
|-------|--------|-------------|
| 5 | `split.py` | Day-stratified train/val split (Day 3 → val, Days 1+2 → train) |
| 6 | `train_yolo.py` | Fine-tune YOLOv8m on labeled dataset (100 epochs, GPU) |

---

## Usage

```bash
# Part 1 — automated preprocessing (run from repo root)
bash bash/run_pipeline_part1.sh

# Resume from a specific stage if needed
bash bash/run_pipeline_part1.sh --from 3

# ... manual CVAT review ...

# Part 2 — post-CVAT training (local)
bash bash/run_pipeline_part2.sh

# Part 2 — post-CVAT training (HPC cluster)
sbatch slurm/run_pipeline_part2_slurm.sh
```

**Inference on test videos** (run separately after training):
```bash
# Local
python scripts/test_yolo.py

# HPC
sbatch slurm/run_test.sh
```

---

## Project Structure

```
Infant_hands/
├── scripts/                         # All Python source files
│   ├── config.py                    # All paths and hyperparameters (single source of truth)
│   ├── hands.py                     # Stage 1: frame extraction
│   ├── detect_hands.py              # Stage 2: hand detection filter
│   ├── generate_hand_labels.py      # Stage 3: YOLO label generation
│   ├── delete_dup_box.py            # Stage 4: deduplication
│   ├── split.py                     # Stage 5: train/val split
│   ├── train_yolo.py                # Stage 6: YOLO training
│   ├── test_yolo.py                 # Inference on test videos
│   └── hand_crop.py                 # Extract 224x224 hand crops for CNN (standalone)
├── bash/                            # Local bash pipeline runners
│   ├── run_pipeline_part1.sh        # Automated pipeline (Stages 1–4)
│   └── run_pipeline_part2.sh        # Post-CVAT pipeline (Stages 5–6)
├── slurm/                           # HPC SLURM job scripts
│   ├── run_pipeline_part2_slurm.sh  # Part 2 pipeline on HPC (Stages 5–6)
│   ├── run_train.sh                 # Training-only SLURM job
│   └── run_test.sh                  # Inference SLURM job
├── yolov8m.pt                       # Pre-trained YOLOv8 Medium weights
├── requirements.txt                 # Python dependencies
├── test_audit.md                    # Bug audit report
└── README.md
```

---

## Configuration

All paths and hyperparameters are in `config.py`. It auto-detects whether it is running locally or on the HPC cluster (BigRed200) and sets paths accordingly.

Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SUBJECTS` | 5 subjects | Subject IDs: 25131, 25138, 25176, 25190, 25602 |
| `TRAIN_DAYS` | 1, 2, 3 | Days of recording per subject |
| `TARGET_FPS` | 5 | Frame extraction rate |
| `MP_MIN_DETECTION_CONF` | 0.4 | MediaPipe hand detection threshold |
| `EPOCHS` | 100 | YOLO training epochs |
| `IMG_SIZE` | 640 | YOLO training image size |
| `BATCH_SIZE` | 128 | Training batch size (A100 GPU) |
| `CONF_THRESHOLD` | 0.25 | YOLO inference confidence threshold |
| `SPLIT_RATIO` | Day-stratified | Day 3 → val, Days 1+2 → train |

---

## Setup

```bash
pip install -r requirements.txt
```

For HPC (BigRed200), submit jobs via SLURM:
```bash
sbatch run_train.sh
sbatch run_test.sh
```

---

## Data

- **Source videos**: `{DATA_ROOT}/{subject}/MP4/{subject}c_day{N}_merged.mp4`
- **Annotations**: `Spatial_Master_5_subjects.csv` — onset/offset timestamps (ms) + `child_in_hand._` contact labels
- **Dataset**: `dataset/images/train/` and `dataset/labels/train/` (YOLO format)

---

## Known Issues & Audit

See `test_audit.md` for the full bug audit. Key resolved issues:

- Bounding box coordinates clamped to `[0, 1]` to prevent corrupt YOLO labels
- Zero-area bounding boxes skipped to prevent NaN loss during training
- `cap.isOpened()` guard added to video processing
- FPS=0 fallback added for corrupt videos
- Deduplication uses order-preserving `dict.fromkeys()` instead of `set()`
- `sorted(os.listdir())` used for reproducible processing order
- Data leakage fixed: train/val split is now day-stratified, not random
