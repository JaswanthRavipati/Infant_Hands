import os
import cv2
import torch
import pandas as pd
import timm
from PIL import Image
import torchvision.transforms as T

from ultralytics import YOLO
from config import *

# =========================
# INIT
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

print("Loading DINO...")
dino = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
dino.eval().to(device)

# =========================
# LOAD PROTOTYPES
# =========================
print("Loading prototypes...")
prototypes_raw = torch.load("/N/slate/veravi/model/prototypes.pth", map_location="cpu")

prototypes = {}
for label, value in prototypes_raw.items():

    if label == "Bed":
        continue

    if isinstance(value, list):
        value = torch.stack(value).mean(dim=0)
    elif isinstance(value, torch.Tensor) and value.ndim > 1:
        value = value.mean(dim=0)

    value = torch.nn.functional.normalize(value.float(), dim=0)
    prototypes[label] = value

print("Classes:", list(prototypes.keys()))

# =========================
# TRANSFORM
# =========================
transform = T.Compose([
    T.Resize((CROP_RESIZE_DINO, CROP_RESIZE_DINO)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# =========================
# DINO FUNCTIONS
# =========================
def get_embedding(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = dino(img)

    emb = emb.squeeze().cpu()
    emb = torch.nn.functional.normalize(emb, dim=0)
    return emb


def classify(crop):
    emb = get_embedding(crop)

    best_label = "unknown"
    best_score = -1

    for label, proto in prototypes.items():
        score = torch.dot(emb, proto).item()

        if score > best_score:
            best_score = score
            best_label = label

    if best_score < DINO_THRESHOLD:
        return "unknown", best_score

    return best_label, best_score


# =========================
# PROCESS VIDEOS
# =========================
videos = [v for v in os.listdir(TEST_VIDEO_DIR) if v.lower().endswith((".mp4",".avi",".mov",".mkv"))]

print("Videos:", videos)

for video in videos:

    print("\nProcessing:", video)

    video_path = os.path.join(TEST_VIDEO_DIR, video)
    output_path = os.path.join(TEST_OUTPUT_DIR, f"output_{video}")
    csv_path = os.path.join(TEST_OUTPUT_DIR, f"{video}.csv")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    results = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo(frame, conf=CONF_THRESHOLD, imgsz=INFER_IMG_SIZE, device=0)[0]

        if detections.boxes is not None:

            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1

                hand_conf = float(confs[i])

                # =========================
                # DYNAMIC PADDING
                # =========================
                pad_w = int(0.5 * w)
                pad_h = int(0.6 * h)

                x1p = max(0, x1 - pad_w)
                y1p = max(0, y1 - pad_h)
                x2p = min(width, x2 + pad_w)
                y2p = min(height, y2 + pad_h)

                crop = frame[y1p:y2p, x1p:x2p]

                if crop.shape[0] < CROP_MIN_SIZE or crop.shape[1] < CROP_MIN_SIZE:
                    continue

                # =========================
                # DINO CLASSIFICATION
                # =========================
                obj_label, obj_conf = classify(crop)

                # SKIP UNKNOWN OBJECTS
                if obj_label == "unknown" or obj_label == "Bed":
                    obj_label = ""
                    obj_conf = 0.0

                # =========================
                # DRAW HAND BOX (always)
                # =========================
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    frame,
                    f"H:{hand_conf:.2f}",
                    (x1, max(y1-10,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )


                # =========================
                # DRAW OBJECT ONLY IF VALID
                # =========================
                if obj_label != "":
                    cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (255,0,0), 2)

                    cv2.putText(
                        frame,
                        f"{obj_label}:{obj_conf:.2f}",
                        (x1p, max(y1p-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2
                    )

                # =========================
                # SAVE CSV (ONE ROW PER HAND)
                # =========================
                results.append({
                    "frame": frame_id,
                    "time_sec": round(frame_id / fps, 2),
                    "hand_id": i,
                    "hand_conf": round(hand_conf, 3),
                    "object_label": obj_label,
                    "object_conf": round(obj_conf, 3)
                })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print("Saved video:", output_path)
    print("Saved CSV:", csv_path)

print("\nDONE ALL VIDEOS")