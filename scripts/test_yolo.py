import cv2
import os
import logging
import torch
from ultralytics import YOLO
from config import YOLO_WEIGHTS as MODEL_PATH, TEST_VIDEO_DIR, TEST_OUTPUT_DIR, CONF_THRESHOLD, INFER_IMG_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

log.info("CUDA: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    log.info("GPU: %s", torch.cuda.get_device_name(0))

model = YOLO(MODEL_PATH)

videos = [v for v in os.listdir(TEST_VIDEO_DIR) if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

log.info("Videos found: %s", videos)

for video in videos:

    video_path = os.path.join(TEST_VIDEO_DIR, video)
    output_path = os.path.join(TEST_OUTPUT_DIR, f"output_{video}")

    log.info("Processing: %s", video)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log.warning("Could not open video: %s, skipping", video)
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, imgsz=INFER_IMG_SIZE, device=0)

        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1

        if frame_count % 500 == 0:
            log.info("%s: %d frames processed", video, frame_count)

    cap.release()
    out.release()

    log.info("Saved: %s", output_path)

log.info("ALL VIDEOS DONE!")
