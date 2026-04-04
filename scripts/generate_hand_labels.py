import cv2
import mediapipe as mp
import os
import logging
from config import (
    DATA_ROOT, SUBJECTS, TRAIN_DAYS,
    DATASET_IMAGES_TRAIN, DATASET_LABELS_TRAIN,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs(DATASET_IMAGES_TRAIN, exist_ok=True)
os.makedirs(DATASET_LABELS_TRAIN, exist_ok=True)

mp_hands = mp.solutions.hands

images_processed = 0
boxes_created = 0

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONF
) as hands:

    for subject in SUBJECTS:
        for day in TRAIN_DAYS:

            input_folder = f"{DATA_ROOT}/{subject}/hand_frames/day{day}"

            if not os.path.exists(input_folder):
                log.warning("Missing folder: %s", input_folder)
                continue

            log.info("Processing Subject %s Day %s", subject, day)

            for file in sorted(os.listdir(input_folder)):

                if not file.endswith(".jpg"):
                    continue

                img_path = os.path.join(input_folder, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb)

                new_name = f"{subject}_day{day}_{file}"

                cv2.imwrite(os.path.join(DATASET_IMAGES_TRAIN, new_name), img)

                label_path = os.path.join(
                    DATASET_LABELS_TRAIN,
                    new_name.replace(".jpg", ".txt")
                )

                if not results.multi_hand_landmarks:
                    open(label_path, "w").close()
                    continue

                with open(label_path, "w") as f:

                    for hand_landmarks in results.multi_hand_landmarks:

                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]

                        x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
                        y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        box_width = x_max - x_min
                        box_height = y_max - y_min

                        if box_width <= 0 or box_height <= 0:
                            continue

                        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                        boxes_created += 1

                images_processed += 1

log.info("Images processed: %d", images_processed)
log.info("Bounding boxes created: %d", boxes_created)
log.info("Dataset ready for YOLO training")
