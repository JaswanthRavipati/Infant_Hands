import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from config import YOLO_WEIGHTS as MODEL_PATH, TEST_VIDEO_DIR, TEST_OUTPUT_DIR, CONF_THRESHOLD, INFER_IMG_SIZE

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(TEST_VIDEO_DIR)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
timestamps = []
hand_counts = []

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Count detections (assuming class 0 = hand)
    count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            count += 1

    # Compute timestamp
    timestamp = frame_idx / fps

    # Store
    timestamps.append(timestamp)
    hand_counts.append(count)

    frame_idx += 1

cap.release()

# -----------------------------
# SAVE CSV
# -----------------------------
csv_path = os.path.join(TEST_OUTPUT_DIR, "hand_detection_data.csv")

df = pd.DataFrame({
    "frame": list(range(len(timestamps))),
    "timestamp_sec": timestamps,
    "hand_count": hand_counts
})

df.to_csv(csv_path, index=False)

print(f"CSV saved at: {csv_path}")

# -----------------------------
# PLOT GRAPH
# -----------------------------
plt.figure()
plt.plot(timestamps, hand_counts)

plt.xlabel("Time (seconds)")
plt.ylabel("Number of Hands")
plt.title("Hands Detected Over Time")

plt.grid()

# SAVE GRAPH (before show)
graph_path = os.path.join(TEST_OUTPUT_DIR, "hand_detection_graph.png")
plt.savefig(graph_path, dpi=300, bbox_inches='tight')

print(f"Graph saved at: {graph_path}")

plt.show()