import os
import cv2
import torch
import timm
import pandas as pd
from PIL import Image
from torchvision import transforms

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =========================
# CONFIG
# =========================
print("🚀 Starting SAM + DINO Pipeline")

INPUT_FOLDER = "/N/slate/veravi/test_videos"
OUTPUT_FOLDER = "/N/slate/veravi/dino_outputs"
CSV_FOLDER = "/N/slate/veravi/dino_outputs/csv"

PROTOTYPES_PATH = "/N/slate/veravi/model/prototypes.pth"
SAM2_CHECKPOINT = "/N/slate/veravi/model/sam2_hiera_large.pt"

CONFIG_ROOT = "/N/u/veravi/BigRed200/.conda/envs/hand_env/lib/python3.11/site-packages/sam2/configs"
SAM2_CONFIG = "sam2_hiera_l"

THRESHOLD = 0.3
MAX_BOXES = 4
FRAME_SKIP = 1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# LOAD DINO
# =========================
print("Loading DINO...")
dino = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
dino.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# =========================
# LOAD PROTOTYPES (REMOVE BED)
# =========================
print("Loading prototypes...")
prototypes_raw = torch.load(PROTOTYPES_PATH, map_location="cpu")

prototypes = {}

for label, value in prototypes_raw.items():

    # ❌ REMOVE BED HERE
    if label == "Bed":
        print("🚫 Removing Bed class")
        continue

    if isinstance(value, list):
        value = torch.stack(value).mean(dim=0)
    elif isinstance(value, torch.Tensor) and value.ndim > 1:
        value = value.mean(dim=0)

    value = value.float()
    value = torch.nn.functional.normalize(value, dim=0)

    prototypes[label] = value

print("Classes after cleaning:", list(prototypes.keys()))

# =========================
# LOAD SAM2
# =========================
print("Loading SAM2...")
sam2_model = build_sam2(
    config_file=SAM2_CONFIG,
    ckpt_path=SAM2_CHECKPOINT,
    hydra_overrides_extra=[f"hydra.searchpath=[file://{CONFIG_ROOT}/sam2]"]
)

sam2_model.to(device)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.75,
)

# =========================
# DINO EMBEDDING
# =========================
def get_embedding(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = dino(img)

    emb = emb.squeeze().cpu()
    emb = torch.nn.functional.normalize(emb, dim=0)
    return emb

# =========================
# CLASSIFY
# =========================
def classify(crop):
    emb = get_embedding(crop)

    best_label = "unknown"
    best_score = -1

    for label, proto in prototypes.items():
        score = torch.dot(emb, proto).item()

        if score > best_score:
            best_score = score
            best_label = label

    if best_score < THRESHOLD:
        return "unknown", best_score

    return best_label, best_score

# =========================
# GET BOXES
# =========================
def get_boxes(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(frame_rgb)

    boxes = []
    for m in masks:
        x, y, w, h = m["bbox"]
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes[:MAX_BOXES]

# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path):

    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]

    out_video = os.path.join(OUTPUT_FOLDER, f"{stem}_output.mp4")
    out_csv = os.path.join(CSV_FOLDER, f"{stem}.csv")

    print(f"\nProcessing {name}")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(
        out_video,
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

        if frame_id % FRAME_SKIP != 0:
            out.write(frame)
            frame_id += 1
            continue

        print(f"{name} | frame {frame_id}")

        boxes = get_boxes(frame)

        for rid, (x, y, w, h) in enumerate(boxes, 1):

            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            label, score = classify(crop)

            # ❌ skip unknown and Bed (extra safety)
            if label == "unknown" or label == "Bed":
                continue

            # ✅ draw
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{label}:{score:.2f}",
                (x, max(y-10,20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            results.append({
                "video": name,
                "frame": frame_id,
                "region": rid,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "score": score
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"Saved video: {out_video}")
    print(f"Saved CSV: {out_csv}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    videos = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".mp4",".avi",".mov",".mkv"))
    ]

    print(f"Found {len(videos)} videos")

    for v in videos:
        process_video(v)

    print("\n🚀 DONE ALL VIDEOS")