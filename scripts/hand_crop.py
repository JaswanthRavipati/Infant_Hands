#!/usr/bin/env python3

import os
import cv2
import logging
from ultralytics import YOLO
from config import (
    DATA_ROOT, SUBJECTS, TRAIN_DAYS, YOLO_WEIGHTS,
    HAND_CROP_DIR, CONF_THRESHOLD, CROP_PADDING, CROP_MIN_SIZE, CROP_RESIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

log.info("=== STARTING MULTI-SUBJECT CROP GENERATION ===")

os.makedirs(HAND_CROP_DIR, exist_ok=True)

log.info("Loading YOLO...")
model = YOLO(YOLO_WEIGHTS)
log.info("YOLO loaded!")

total_frames = 0
total_crops = 0

for subject in SUBJECTS:
    for day in TRAIN_DAYS:

        frames_folder = f"{DATA_ROOT}/{subject}/frames/day{day}"

        if not os.path.exists(frames_folder):
            log.warning("Skipping missing folder: %s", frames_folder)
            continue

        log.info("Processing Subject %s Day %s", subject, day)

        frame_files = sorted(os.listdir(frames_folder))

        for frame_file in frame_files:

            if not frame_file.endswith(".jpg"):
                continue

            frame_path = os.path.join(frames_folder, frame_file)

            img = cv2.imread(frame_path)
            if img is None:
                continue

            height, width = img.shape[:2]
            total_frames += 1

            if total_frames % 200 == 0:
                log.info("Frames: %d, Crops: %d", total_frames, total_crops)

            results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

            if results.boxes is None:
                continue

            for i, box in enumerate(results.boxes.xyxy):

                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1 - CROP_PADDING)
                y1 = max(0, y1 - CROP_PADDING)
                x2 = min(width, x2 + CROP_PADDING)
                y2 = min(height, y2 + CROP_PADDING)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                if (x2 - x1) < CROP_MIN_SIZE or (y2 - y1) < CROP_MIN_SIZE:
                    continue

                crop = cv2.resize(crop, (CROP_RESIZE, CROP_RESIZE))

                crop_name = f"{subject}_day{day}_{frame_file.replace('.jpg','')}_h{i}.jpg"
                cv2.imwrite(os.path.join(HAND_CROP_DIR, crop_name), crop)
                total_crops += 1

log.info("Total frames processed: %d", total_frames)
log.info("Total crops saved: %d", total_crops)
log.info("Saved at: %s", HAND_CROP_DIR)
