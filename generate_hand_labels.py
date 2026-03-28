import cv2
import mediapipe as mp
import os

# -----------------------------
# SETTINGS
# -----------------------------

ROOT = "/Volumes/Seagate/CSCI_B657/data"

SUBJECTS = [25131, 25138, 25176, 25190, 25602]
TRAIN_DAYS = [1, 2, 3]

IMAGE_OUTPUT = f"{ROOT}/dataset/images/train"
LABEL_OUTPUT = f"{ROOT}/dataset/labels/train"

os.makedirs(IMAGE_OUTPUT, exist_ok=True)
os.makedirs(LABEL_OUTPUT, exist_ok=True)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

mp_hands = mp.solutions.hands

images_processed = 0
boxes_created = 0

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
) as hands:

    for subject in SUBJECTS:
        for day in TRAIN_DAYS:

            input_folder = f"{ROOT}/{subject}/hand_frames/day{day}"

            if not os.path.exists(input_folder):
                print("Missing folder:", input_folder)
                continue

            print(f"Processing Subject {subject} Day {day}")

            for file in os.listdir(input_folder):

                if not file.endswith(".jpg"):
                    continue

                img_path = os.path.join(input_folder, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb)

                # 🔥 UNIQUE filename (important to avoid overwrite)
                new_name = f"{subject}_day{day}_{file}"

                # copy image
                cv2.imwrite(os.path.join(IMAGE_OUTPUT, new_name), img)

                label_path = os.path.join(
                    LABEL_OUTPUT,
                    new_name.replace(".jpg", ".txt")
                )

                if not results.multi_hand_landmarks:
                    open(label_path, "w").close()
                    continue

                with open(label_path, "w") as f:

                    for hand_landmarks in results.multi_hand_landmarks:

                        xs = []
                        ys = []

                        for lm in hand_landmarks.landmark:
                            xs.append(lm.x)
                            ys.append(lm.y)

                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        box_width = x_max - x_min
                        box_height = y_max - y_min

                        f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")

                        boxes_created += 1

                images_processed += 1

print("Images processed:", images_processed)
print("Bounding boxes created:", boxes_created)
print("Dataset ready for YOLO training")