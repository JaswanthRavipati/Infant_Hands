import logging
from ultralytics import YOLO
from config import DATA_YAML, EPOCHS, IMG_SIZE, BATCH_SIZE, DEVICE, NUM_WORKERS, YOLO_PRETRAINED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

model = YOLO(YOLO_PRETRAINED)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=NUM_WORKERS,
    val=True,
    verbose=True
)

log.info("===== TRAINING COMPLETE =====")

metrics = results.results_dict
log.info("mAP50: %s", metrics.get("metrics/mAP50", "NA"))
log.info("mAP50-95: %s", metrics.get("metrics/mAP50-95", "NA"))
