from ultralytics import YOLO
import cv2
import os
import torch

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/N/u/veravi/BigRed200/infanthands/runs/detect/train5/weights/best.pt"
VIDEO_DIR = "/N/slate/veravi/test_videos"

OUTPUT_DIR = "/N/slate/veravi/yolo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF = 0.25
IMG_SIZE = 512   # safe

# -----------------------------
# SYSTEM INFO
# -----------------------------
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# GET ALL VIDEOS
# -----------------------------
videos = [v for v in os.listdir(VIDEO_DIR) if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

print("Videos found:", videos)

# -----------------------------
# PROCESS EACH VIDEO
# -----------------------------
for video in videos:

    video_path = os.path.join(VIDEO_DIR, video)
    output_path = os.path.join(OUTPUT_DIR, f"output_{video}")

    print(f"\n🚀 Processing: {video}")

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=CONF,
            imgsz=IMG_SIZE,
            device=0
        )

        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1

        if frame_count % 500 == 0:
            print(f"{video}: {frame_count} frames processed")

    cap.release()
    out.release()

    print(f"✅ Saved: {output_path}")

print("\n🎉 ALL VIDEOS DONE!")