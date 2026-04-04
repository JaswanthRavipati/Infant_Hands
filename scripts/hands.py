import cv2
import os
import logging
from config import DATA_ROOT, SUBJECTS, TRAIN_DAYS, TARGET_FPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

for subject in SUBJECTS:
    for day in TRAIN_DAYS:

        video_path = f"{DATA_ROOT}/{subject}/MP4/{subject}c_day{day}_merged.mp4"

        if not os.path.exists(video_path):
            log.warning("Video not found: %s", video_path)
            continue

        output_folder = f"{DATA_ROOT}/{subject}/frames/day{day}"
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            log.warning("Could not read FPS for %s, skipping", video_path)
            cap.release()
            continue

        frame_interval = max(1, int(fps / TARGET_FPS))

        count = 0
        saved = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if count % frame_interval == 0:
                cv2.imwrite(f"{output_folder}/frame_{saved:04d}.jpg", frame)
                saved += 1

            count += 1

        cap.release()

        log.info("Finished subject %s day %s | Frames saved: %d", subject, day, saved)

log.info("All videos processed.")
