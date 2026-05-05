#!/usr/bin/env python3

import pandas as pd
import cv2
import os
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------

CSV_FILE = "/Volumes/Seagate/CSCI_B657/csv/Spatial_Master_5_subjects.csv"
ROOT = "/Volumes/Seagate/CSCI_B657/data"

SUBJECTS = [25131, 25138, 25176, 25190, 25602]
DAYS = [1, 2, 3]

FPS = 5

YOLO_WEIGHTS = "/Volumes/Seagate/CSCI_B657/data/best.pt"

OUTPUT_DIR = "/Volumes/Seagate/CSCI_B657/data/parent_hand_crops"

CONF_THRESHOLD = 0.25
PADDING = 20

# -----------------------------
# SETUP
# -----------------------------

print("=== GENERATING PARENT HAND CROPS ===", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLO...", flush=True)
model = YOLO(YOLO_WEIGHTS)
print("YOLO loaded!", flush=True)

# -----------------------------
# LOAD CSV
# -----------------------------

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

df = df[
    (df["subject"].isin(SUBJECTS)) &
    (df["day"].isin(DAYS))
]

print("Filtered rows:", len(df), flush=True)

# -----------------------------
# FILTER: parent_in_hand
# -----------------------------

df = df[df["parent_in_hand._"].notna()]

print("Rows with parent hand:", len(df), flush=True)

# -----------------------------
# TIME → FRAME
# -----------------------------

df["start_frame"] = (df["onset"] / 1000 * FPS).astype(int)
df["end_frame"] = (df["offset"] / 1000 * FPS).astype(int)

# -----------------------------
# MAIN LOOP
# -----------------------------

total_frames = 0
total_crops = 0

for _, row in df.iterrows():

    subject = int(row["subject"])
    day = int(row["day"])

    frames_folder = f"{ROOT}/{subject}/frames/day{day}"

    if not os.path.exists(frames_folder):
        continue

    for frame_id in range(row["start_frame"], row["end_frame"] + 1):

        filename = f"frame_{frame_id:04d}.jpg"
        frame_path = os.path.join(frames_folder, filename)

        if not os.path.exists(frame_path):
            continue

        img = cv2.imread(frame_path)
        if img is None:
            continue

        height, width = img.shape[:2]
        total_frames += 1

        if total_frames % 200 == 0:
            print(f"Frames: {total_frames}, Crops: {total_crops}", flush=True)

        # -------- YOLO DETECTION --------
        results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

        if results.boxes is None:
            continue

        for i, box in enumerate(results.boxes.xyxy):

            x1, y1, x2, y2 = map(int, box)

            # Padding
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(width, x2 + PADDING)
            y2 = min(height, y2 + PADDING)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Filter small detections
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            # Resize (CNN ready)
            crop = cv2.resize(crop, (224, 224))

            # Save
            crop_name = f"{subject}_day{day}_f{frame_id}_h{i}.jpg"
            save_path = os.path.join(OUTPUT_DIR, crop_name)

            cv2.imwrite(save_path, crop)
            total_crops += 1

# -----------------------------
# RESULTS
# -----------------------------

print("================================", flush=True)
print("Total frames processed:", total_frames, flush=True)
print("Total crops saved:", total_crops, flush=True)
print("Saved at:", OUTPUT_DIR, flush=True)