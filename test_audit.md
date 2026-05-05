# Test Audit — Infant_hands Pipeline

## CRITICAL (7 issues)

| ID | File | Lines | Issue | Fix |
|----|------|-------|-------|-----|
| C-1 | `generate_hand_labels.py` | 84–98 | MediaPipe `landmark.x/y` not clamped to `[0,1]`. Out-of-frame hands yield `-0.02` or `1.03`, corrupting YOLO labels. | Clamp `x_min, x_max, y_min, y_max` to `[0.0, 1.0]` before computing center/size. ✅ Fixed |
| C-2 | `generate_hand_labels.py` | 88–98 | If all 21 landmarks coincide, `box_width = box_height = 0` → invalid YOLO annotation → NaN loss during training. | Skip box if `box_width <= 0 or box_height <= 0`. ✅ Fixed |
| C-3 | `generate_hand_labels.py` | 62–74 | Every image is unconditionally copied to the dataset folder before the hand-detection check, injecting unintended negatives. | Only copy image after confirming a detection (or add explicit `INCLUDE_NEGATIVES` flag). |
| C-4 | `delete_dup_box.py` | 15 | `list(set(lines))` is broken: `"0 0.5 ...\n"` and `"0 0.5 ..."` treated as different. Also randomizes line order. | Use `dict.fromkeys()` and strip lines before deduplication. ✅ Fixed |
| C-5 | `hands.py` | 24, 35 | `frame_interval = int(fps / 5)` → `ZeroDivisionError` when FPS=0 (corrupted/non-standard video). | `frame_interval = max(1, int(fps / 5))`. ✅ Fixed |
| C-6 | `test_yolo.py` | 51, 54 | `fps=0` for unreadable videos → `cv2.VideoWriter` with fps=0 silently produces corrupt output. | Fallback: `fps = fps if fps > 0 else 25`. |
| C-7 | `test_yolo.py` | 47–54 | `cap.isOpened()` never checked. Corrupt/missing video silently produces a 0×0 output file. | Check `cap.isOpened()` and skip if false. |

---

## WARNING (11 issues)

| ID | File | Lines | Issue |
|----|------|-------|-------|
| W-1 | `detect_hands.py` | 35–36, 72 | Frame math assumes CSV timestamps are milliseconds — never verified. Wrong unit → all frame indices 1000× off, silent zero matches. |
| W-2 | `detect_hands.py` | 60 | Column `child_in_hand._` is fragile; minor header change causes silent empty filter and zero output. |
| W-3 | `split.py` | 38 | Empty source folder silently prints "Split completed" with 0 files — no error raised. |
| W-4 | `split.py` | 11–18 | Re-running overwrites split without warning; `dataset.yaml` may still point to original `train/` rather than `train_split/`. |
| W-5 | `hands.py` | 24 | `int()` truncation on FPS causes slight overshoot (29.97 fps → 5.994 fps), misaligning frame indices used by `detect_hands.py`. Use `round()`. |
| W-6 | `generate_hand_labels.py` | 46 | `os.listdir()` is non-deterministic; use `sorted()` for reproducibility. ✅ Fixed |
| W-7 | `hand_crop.py` | 83–94 | Size filter `< 50px` checked *after* 20px padding is added. A 15×15 noisy detection becomes 55×55 and passes. Check raw box dimensions before padding. |
| W-8 | `train_yolo.py` | 25 | `batch=128` hardcoded for A100; crashes on smaller GPUs. Consider `batch=-1` for auto-sizing. |
| W-9 | `run_test.sh` | 21 | `cd ~/infanthands` may fail if path is wrong on cluster; `set -euo pipefail` will abort job. |
| W-10 | `run_train.sh` | 9 | `logs/` directory may not exist when SLURM resolves relative `--output` path; log file is lost. |
| W-11 | `test_yolo.py` | 65–68 | `device=0` hardcoded in inference call; crashes on CPU-only machines even though CUDA check is printed. |

---

## INFO (8 issues)

| ID | File | Issue |
|----|------|-------|
| I-1 | `generate_hand_labels.py:74`, `split.py:67` | `open(path, "w").close()` file handle leak — use `with open(...) as f: pass`. |
| I-2 | `detect_hands.py` | No progress output for multi-hour loops. |
| I-3 | `hands.py` | Stale frames from previous runs persist in output folder if video was shorter. |
| I-4 | `hand_crop.py`, `test_yolo.py`, `train_yolo.py` | HPC-absolute paths not portable without `argparse`/env vars. |
| I-5 | `split.py` | `shutil.copy` doubles storage; symlinks would avoid duplication. |
| I-6 | `test_yolo.py:16` | Inference uses `imgsz=512` but training used `imgsz=640`; suboptimal mAP at inference. |
| I-7 | `generate_hand_labels.py:102` | `images_processed` counter placement is accidentally correct but misleading. |
| I-8 | `run_test.sh:7` | 2h time limit may be insufficient; no graceful shutdown handler if SLURM kills mid-video. |
