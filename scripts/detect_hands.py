import pandas as pd
import cv2
import mediapipe as mp
import os

# -----------------------------
# SETTINGS
# -----------------------------

CSV_FILE = "/Volumes/Seagate/CSCI_B657/csv/Spatial_Master_5_subjects.csv"
ROOT = "/Volumes/Seagate/CSCI_B657/data"

SUBJECTS = [25131, 25138, 25176, 25190, 25602]
TRAIN_DAYS = [1, 2, 3]

FPS = 5

# -----------------------------
# LOAD CSV
# -----------------------------

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

df = df[(df["subject"].isin(SUBJECTS)) & (df["day"].isin(TRAIN_DAYS))]

print("Rows after filtering:", len(df))

# -----------------------------
# CONVERT TIMESTAMPS → FRAMES
# -----------------------------

df["start_frame"] = (df["onset"] / 1000 * FPS).astype(int)
df["end_frame"] = (df["offset"] / 1000 * FPS).astype(int)

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
        max_num_hands=2,
        min_detection_confidence=0.4
) as hands:

    for _, row in df.iterrows():

        # only intervals where hand touches object
        if pd.notna(row["child_in_hand._"]):

            subject = row["subject"]
            day = row["day"]

            # dynamic frames folder
            frames_folder = f"{ROOT}/{subject}/frames/day{day}"

            # output folder per subject/day
            output_folder = f"{ROOT}/{subject}/hand_frames/day{day}"
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

print("Frames checked:", frames_checked)
print("Frames with hands:", frames_with_hands)
print("Processing complete!")