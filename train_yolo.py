from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------

DATA_YAML = "/Volumes/Seagate/CSCI_B657/data/dataset/dataset.yaml"

EPOCHS = 100
VAL_INTERVAL = 10

# -----------------------------
# LOAD MODEL
# -----------------------------

model = YOLO("yolov8m.pt")

# -----------------------------
# TRAIN LOOP (manual control)
# -----------------------------

for epoch in range(1, EPOCHS + 1):

    print(f"\n===== Epoch {epoch} =====")

    # train for 1 epoch at a time
    results = model.train(
        data=DATA_YAML,
        epochs=1,
        imgsz=640,
        batch=16,
        device="mps",
        verbose=False
    )

    # -----------------------------
    # TRAIN METRICS
    # -----------------------------

    train_metrics = results.results_dict

    print("Train Loss:")
    print(f"  Box Loss: {train_metrics.get('train/box_loss', 'NA')}")
    print(f"  Cls Loss: {train_metrics.get('train/cls_loss', 'NA')}")

    # -----------------------------
    # VALIDATION EVERY 10 EPOCHS
    # -----------------------------

    if epoch % VAL_INTERVAL == 0:

        print("\n--- Validation ---")

        val_results = model.val(data=DATA_YAML, device="mps")

        metrics = val_results.results_dict

        print("Validation Metrics:")
        print(f"  mAP50: {metrics.get('metrics/mAP50', 'NA')}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95', 'NA')}")