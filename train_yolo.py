from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------

DATA_YAML = "/N/slate/veravi/dataset/dataset.yaml"

EPOCHS = 100

# -----------------------------
# LOAD MODEL
# -----------------------------

model = YOLO("yolov8m.pt")

# -----------------------------
# TRAIN (FULL CONTROL)
# -----------------------------

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=640,
    batch=128,        # 🔥 increase for A100
    device=0,        # GPU
    workers=8,
    val=True,        # validation enabled
    verbose=True
)

# -----------------------------
# FINAL METRICS
# -----------------------------

print("\n===== TRAINING COMPLETE =====")

metrics = results.results_dict

print("Final Metrics:")
print(f"  mAP50: {metrics.get('metrics/mAP50', 'NA')}")
print(f"  mAP50-95: {metrics.get('metrics/mAP50-95', 'NA')}")