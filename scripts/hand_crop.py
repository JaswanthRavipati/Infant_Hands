#!/usr/bin/env python3

import os
import cv2
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------

ROOT = "/Volumes/Seagate/CSCI_B657/data"

SUBJECTS = [25131, 25138, 25176, 25190, 25602]
DAYS = [1, 2, 3]

YOLO_WEIGHTS = "/Volumes/Seagate/CSCI_B657/data/best.pt"

OUTPUT_DIR = "/Volumes/Seagate/CSCI_B657/data/hand_crops"

CONF_THRESHOLD = 0.25
PADDING = 20

# -----------------------------
# SETUP
# -----------------------------

print("=== STARTING MULTI-SUBJECT CROP GENERATION ===", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLO...", flush=True)
model = YOLO(YOLO_WEIGHTS)
print("YOLO loaded!", flush=True)

total_frames = 0
total_crops = 0

# -----------------------------
# MAIN LOOP
# -----------------------------

MAX_CROPS_PER_VIDEO = 200

for subject in SUBJECTS:
    for day in DAYS:

        frames_folder = f"{ROOT}/{subject}/frames/day{day}"

        if not os.path.exists(frames_folder):
            print(f"Skipping missing folder: {frames_folder}", flush=True)
            continue

        print(f"Processing Subject {subject} Day {day}", flush=True)

        frame_files = sorted(os.listdir(frames_folder))

        crops_per_video = 0   # 🔥 reset for each video

        for frame_file in frame_files:

            # 🔴 Stop if limit reached
            if crops_per_video >= MAX_CROPS_PER_VIDEO:
                print(f"Reached {MAX_CROPS_PER_VIDEO} crops → moving to next video", flush=True)
                break

            if not frame_file.endswith(".jpg"):
                continue

            frame_path = os.path.join(frames_folder, frame_file)

            img = cv2.imread(frame_path)
            if img is None:
                continue

            height, width = img.shape[:2]
            total_frames += 1

            if total_frames % 200 == 0:
                print(f"Frames: {total_frames}, Total Crops: {total_crops}", flush=True)

            # -------- YOLO DETECTION --------
            results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

            if results.boxes is None:
                continue

            for i, box in enumerate(results.boxes.xyxy):

                # 🔴 Stop inside detection loop also
                if crops_per_video >= MAX_CROPS_PER_VIDEO:
                    break

                x1, y1, x2, y2 = map(int, box)

                # Padding
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(width, x2 + PADDING)
                y2 = min(height, y2 + PADDING)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Filter small boxes
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                # Resize for CNN
                crop = cv2.resize(crop, (224, 224))

                # Save crop
                crop_name = f"{subject}_day{day}_{frame_file.replace('.jpg','')}_h{i}.jpg"
                save_path = os.path.join(OUTPUT_DIR, crop_name)

                cv2.imwrite(save_path, crop)

                total_crops += 1
                crops_per_video += 1
                
# -----------------------------
# RESULTS
# -----------------------------

print("================================", flush=True)
print("Total frames processed:", total_frames, flush=True)
print("Total crops saved:", total_crops, flush=True)
print("Saved at:", OUTPUT_DIR, flush=True)