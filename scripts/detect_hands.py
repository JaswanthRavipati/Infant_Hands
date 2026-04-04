import pandas as pd
import cv2
import mediapipe as mp
import os
import logging
from config import (
    CSV_FILE, DATA_ROOT, SUBJECTS, TRAIN_DAYS, TARGET_FPS,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -----------------------------
# LOAD CSV
# -----------------------------

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

log.info("Columns: %s", list(df.columns))

df = df[(df["subject"].isin(SUBJECTS)) & (df["day"].isin(TRAIN_DAYS))]

log.info("Rows after filtering: %d", len(df))

# -----------------------------
# CONVERT TIMESTAMPS → FRAMES
# -----------------------------

df["start_frame"] = (df["onset"] / 1000 * TARGET_FPS).astype(int)
df["end_frame"] = (df["offset"] / 1000 * TARGET_FPS).astype(int)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

mp_hands = mp.solutions.hands

frames_checked = 0
frames_with_hands = 0

# -----------------------------
# HAND DETECTION LOOP
# -----------------------------

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONF
) as hands:

    for _, row in df.iterrows():

        if pd.notna(row["child_in_hand._"]):

            subject = row["subject"]
            day = row["day"]

            frames_folder = f"{DATA_ROOT}/{subject}/frames/day{day}"
            output_folder = f"{DATA_ROOT}/{subject}/hand_frames/day{day}"

            if not os.path.exists(frames_folder):
                log.warning("Frames folder missing: %s", frames_folder)
                continue

            os.makedirs(output_folder, exist_ok=True)

            for frame_id in range(row["start_frame"], row["end_frame"] + 1):

                filename = f"frame_{frame_id:04d}.jpg"
                path = os.path.join(frames_folder, filename)

                if not os.path.exists(path):
                    continue

                img = cv2.imread(path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                frames_checked += 1

                if results.multi_hand_landmarks:
                    frames_with_hands += 1
                    cv2.imwrite(os.path.join(output_folder, filename), img)

# -----------------------------
# RESULTS
# -----------------------------

log.info("Frames checked: %d", frames_checked)
log.info("Frames with hands: %d", frames_with_hands)
log.info("Processing complete!")
