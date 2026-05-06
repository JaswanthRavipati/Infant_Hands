# Infant Hands

End-to-end computer vision pipeline for **egocentric infant hand detection** and **in-hand object identification** from head-mounted camera footage. Built for CSCI B657 (Computer Vision, Spring 2026) in collaboration with the Department of Psychological & Brain Sciences, Indiana University Bloomington.

**Authors:** Dheeraj Karanam, Venkata Sai Jaswanth Ravipati, Sandra Kettidathil Chandy
**Advisors:** Linda B. Smith, Minju Kim

For the full technical write-up see [REPORT.md](REPORT.md); the conference-style paper is in [paper.tex](paper.tex) / [paper.pdf](paper.pdf).

---
## Team members:
Dheeraj Karanam(dhkara), Venkata Sai Jaswanth Ravipati(veravi), Sandra Kettidathil Chandy(saketti)

## Overview

The pipeline answers two questions about head-mounted infant video:

1. **Where are the hands?** A YOLOv8m detector fine-tuned on CVAT-corrected, MediaPipe-bootstrapped labels.
2. **What is in the hand?** DINOv2 prototype matching on a padded crop around each detected hand box (one-shot, no per-class training).

It is trained on 5 infants × 4 days of egocentric recordings, with day-stratified splits (Days 1–2 → train, Day 3 → val, Day 4 → held-out test).

### Headline results (held-out Day-4 footage)

| Metric | Hand detector |
|---|---|
| Precision | ≈ 0.99 |
| Recall | ≈ 0.99 |
| mAP@50 | ≈ 0.998 |
| mAP@50-95 | ≈ 0.89 |

Object-interaction analysis spans **24 toy classes** and surfaces distinct per-infant interaction patterns (e.g., 25176 → cups/balls, 25602 → blocks, 25138 → spoons).

---

## Final pipeline

```
video frame
   │
   ▼
YOLOv8m hand detector  ──►  hand box (green)
   │
   ▼
pad ~2× per dimension (+50% w, +60% h each side)
   │
   ▼
resize 518×518  ──►  DINOv2 ViT-B/14 embedding (frozen)
   │
   ▼
cosine vs. per-class prototypes (mean-pooled, L2-normalized)
   │
   ▼
argmax  +  threshold (DINO_THRESHOLD = 0.4)  ──►  object label (or `unknown`)
   │
   ▼
annotated frame + per-frame CSV: (frame, t_sec, hand_id, hand_conf, object, object_conf)
```

Driver: [`scripts/final.py`](scripts/final.py) → [`slurm/run_yolo_dino.sh`](slurm/run_yolo_dino.sh).

---

## Pipeline stages

### Part 1 — Automated preprocessing (`bash/run_pipeline_part1.sh`)

| # | Script | Description |
|---|--------|-------------|
| 1 | `scripts/hands.py` | Extract frames at 5 FPS from egocentric MP4s |
| 2 | `scripts/detect_hands.py` | Filter to hand-containing frames (MediaPipe + CSV `child_in_hand._` annotations) |
| 3 | `scripts/generate_hand_labels.py` | MediaPipe landmarks → YOLO bounding-box labels |
| 4 | `scripts/delete_dup_box.py` | Order-preserving deduplication of label files |

### Manual — CVAT review

Auto-labels are imported into **CVAT**: missed hands added, false positives removed, padding corrected. Corrected labels are exported back to `dataset/labels/train/`.

### Part 2 — Detector training (`bash/run_pipeline_part2.sh` or SLURM)

| # | Script | Description |
|---|--------|-------------|
| 5 | `scripts/split.py` | Day-stratified split (Day 3 → val, Days 1+2 → train) |
| 6 | `scripts/train_yolo.py` | Fine-tune YOLOv8m for 100 epochs (A100, batch 128, img 640) |

### Part 3 — Object identification

| Script | Purpose |
|---|---|
| `scripts/train_dino.py` | Build DINOv2 prototype bank from 30 reference frames per class → `prototypes.pth` |
| `scripts/final.py` | YOLO + padded crop + DINOv2 prototype matching on test videos |
| `scripts/graph.py` | Hand-count and interaction timeseries plots |

---

## Usage

```bash
# Part 1 — auto preprocessing (resumable)
bash bash/run_pipeline_part1.sh
bash bash/run_pipeline_part1.sh --from 3   # resume at stage 3

# ... manual CVAT review ...

# Part 2 — detector training
bash bash/run_pipeline_part2.sh             # local
sbatch slurm/run_pipeline_part2_slurm.sh    # HPC (BigRed200)

# YOLO inference only
python scripts/test_yolo.py                 # local
sbatch slurm/run_test.sh                    # HPC

# Final YOLO + DINO end-to-end
python scripts/train_dino.py                # build prototype bank
sbatch slurm/run_yolo_dino.sh               # 24h, 64 GB
```

---

## Project structure

```
Infant_hands/
├── scripts/                         # All Python source
│   ├── config.py                    # Paths & hyperparameters (auto-detects HPC vs local)
│   ├── hands.py                     # Stage 1: frame extraction (5 FPS)
│   ├── detect_hands.py              # Stage 2: MediaPipe hand-frame filter
│   ├── generate_hand_labels.py      # Stage 3: MediaPipe → YOLO labels
│   ├── delete_dup_box.py            # Stage 4: deduplication
│   ├── split.py                     # Stage 5: day-stratified split
│   ├── train_yolo.py                # Stage 6: YOLOv8m fine-tune
│   ├── test_yolo.py                 # YOLO inference on test videos
│   ├── train_dino.py                # DINOv2 prototype bank
│   ├── test_dino.py                 # YOLO + DINO classification (older variant)
│   ├── final.py                     # ⭐ Production: YOLO + dynamic-pad + DINOv2
│   ├── sam_dino.py                  # SAM2 + DINO ablation (failed approach, retained)
│   ├── simclr_pretrain.py           # SimCLR scaffolding for future baby/adult head
│   ├── simclr_finetune.py
│   ├── graph.py                     # Hand-count + interaction timeseries
│   ├── make_figures.py              # Figures for paper.tex
│   ├── hand_crop.py                 # Generic 224×224 hand crop extraction
│   └── parent_hand_crop.py          # Parent-hand crop extraction (CSV-driven)
├── bash/                            # Local pipeline runners
│   ├── run_pipeline_part1.sh        # Stages 1–4
│   ├── run_pipeline_part2.sh        # Stages 5–6
│   └── run_pipeline_part3.sh        # DINO prototypes + final inference
├── slurm/                           # HPC SLURM job scripts
│   ├── run_pipeline_part2_slurm.sh
│   ├── run_train.sh
│   ├── run_test.sh
│   ├── run_yolo_dino.sh             # Final pipeline (24h, 64 GB)
│   ├── run_sam_dino.sh              # SAM2 ablation (12h, 96 GB)
│   ├── run_simclr_pretrain.sh
│   ├── run_simclr_finetune.sh
│   └── run_graph.sh
├── figures/                         # Paper figures (training curves, confusion matrix, etc.)
├── yolov8m.pt                       # Pretrained YOLOv8m init weights
├── yolo26n.pt                       # Alt YOLO weights
├── requirements.txt
├── REPORT.md                        # Full technical report
├── PAPER.md / paper.tex / paper.pdf # Conference-style paper
├── references.bib
├── test_audit.md                    # Bug audit & fix log
└── README.md
```

---

## Configuration

All paths and hyperparameters live in [`scripts/config.py`](scripts/config.py). It auto-detects HPC (BigRed200, presence of `/N/slate/veravi`) vs. local.

| Parameter | Value | Description |
|---|---|---|
| `SUBJECTS` | 25131, 25138, 25176, 25190, 25602 | 5 infants |
| `TRAIN_DAYS` | 1, 2, 3 | Day 4 held out as test |
| `TARGET_FPS` | 5 | Frame extraction rate |
| `MP_MIN_DETECTION_CONF` | 0.4 | MediaPipe hand threshold |
| `EPOCHS` | 100 | YOLO training |
| `IMG_SIZE` | 640 | YOLO training image size |
| `INFER_IMG_SIZE` | 512 | YOLO inference image size |
| `BATCH_SIZE` | 128 | A100 GPU |
| `CONF_THRESHOLD` | 0.25 | YOLO inference |
| `DINO_THRESHOLD` | 0.4 | DINOv2 cosine cutoff (`unknown` below) |
| `CROP_MIN_SIZE` | 50 px | Reject tiny crops before DINO |
| `SPLIT_SEED` | 42 | Reproducibility |

DINO crop padding (`scripts/final.py:149-150`):

```python
pad_w = int(0.5 * w)   # +50% on each side  → ~2× width
pad_h = int(0.6 * h)   # +60% on each side  → ~2.2× height
```

---

## Models used

| Component | Model | Source |
|---|---|---|
| Hand detector | YOLOv8m (fine-tuned) | `ultralytics`, init `yolov8m.pt` |
| Auto-label seed | MediaPipe Hands | `mediapipe` |
| Object proposer (ablation) | SAM2 Hiera-Large | `sam2` + `SAM2AutomaticMaskGenerator` |
| Object embedder | DINOv2 ViT-B/14 (`vit_base_patch14_dinov2`) | `timm` (frozen, pretrained) |
| Prototype bank | `prototypes.pth` | dict of `class → mean-pooled L2-normalized embedding` |

---

## Data

- **Source videos:** `{DATA_ROOT}/{subject}/MP4/{subject}c_day{N}_merged.mp4`
- **Behavioral annotations:** `Spatial_Master_5_subjects.csv` — onset/offset (ms) + `child_in_hand._` and `parent_in_hand._` contact labels
- **Hand-detection dataset:** `dataset/images/{train,val}/`, `dataset/labels/{train,val}/` (YOLO format)
- **DINOv2 prototypes:** 30 reference frames per object class, embedded at 518×518, mean-pooled and L2-normalized into `prototypes.pth`. **24 toy classes** are tracked: Cars, Spoon, Ball, Blocks, Avocado, Lion, Eggplant, Pot, Boxes, Spatula, Pomegranate, Raccoon, Alligator, Oven mitt, Drill, Tambourine, Concrete mixer, Mouse, Other, etc. (`Bed` excluded at load time).

---

## Failed approaches (preserved as ablations)

Both negative results materially shaped the final pipeline. Full diagnostics are in [REPORT.md §5](REPORT.md).

### 1. Baby vs. adult hand discrimination by bounding-box geometry
Five variants tried (single-threshold area, per-frame ratio, aspect ratio, vertical position, skin-tone). **All failed.** Pixel size measures **distance from camera**, not anatomical identity, and the two distributions overlap. Aspect ratio is dominated by grip posture, position by scene composition, skin tone by auto-exposure. The size-thresholding code path is unmaintained in the final pipeline; a SimCLR-pretrained appearance classifier (scaffolding in `scripts/simclr_*.py`) is the planned fix.

### 2. SAM2 standalone as an object proposer
Five SAM2 parameters swept (`points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, `MAX_BOXES`, DINO threshold). **All failed.** SAM2's mask-quality ranking rewards smooth, large, well-defined regions — exactly what *background* (rugs, walls, table) looks like, and exactly what *held toys* (small, irregular, occluded) do not. No semantic prior toward graspable objects. Fixed by replacing SAM2 with the **YOLO hand box as a learned spatial prior** + dynamic padding crop. `scripts/sam_dino.py` retained as a reproducible ablation.

**Cross-cutting lesson:** off-the-shelf general-purpose tools (geometric heuristics; SAM2 mask quality) cannot replace **learned domain-specific priors** (anatomical appearance; hand-region location).

---

## Limitations

1. **DINO localizes to the hand region, not the object** — results read as *hand-region object associations*, not pixel-precise object localization.
2. **Baby vs. adult disambiguation is unsolved** — needs an appearance-based classifier, not geometry.
3. **Single global DINO threshold** — per-class calibration likely lifts recall on visually similar classes (cups vs. bowls).
4. **Two-handed frames double-count** — each detected hand contributes its own object label.
5. **SAM2 alone unusable** as a proposer in this domain; retained only for ablation.

---

## Setup

```bash
pip install -r requirements.txt
```

For HPC (BigRed200):

```bash
sbatch slurm/run_train.sh
sbatch slurm/run_test.sh
sbatch slurm/run_yolo_dino.sh
```

---

## Robustness fixes

See [`test_audit.md`](test_audit.md) for the full audit. Resolved silent failures:

- MediaPipe coordinates clamped to `[0, 1]` (prevents corrupt YOLO labels).
- Zero-area boxes skipped (prevents NaN training loss).
- `cap.isOpened()` guard on video reads.
- `fps = 0` fallback for corrupt videos.
- `dict.fromkeys()` deduplication preserves order.
- `sorted(os.listdir())` everywhere file order affects results.
- Train/val split is **day-stratified**, not random — eliminates the data leakage that random splitting introduces in this dataset.

---

## Reference

[1] Bambach, S., Crandall, D. J., Smith, L. B., Yu, C. (2018). *Toddler-Inspired Visual Object Learning.* NeurIPS 31.
