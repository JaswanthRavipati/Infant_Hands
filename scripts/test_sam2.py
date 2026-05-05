import os
import cv2
import csv
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# =========================
# ENV DETECTION
# =========================
ON_HPC = os.path.exists("/N/slate/veravi")

if ON_HPC:
    VIDEO_DIR = "/N/slate/veravi/test_videos"
    OUTPUT_DIR = "/N/slate/veravi/regions_output"
    CSV_OUTPUT = "/N/slate/veravi/dino_results.csv"
    PROTOTYPE_FILE = "/N/slate/veravi/model/prototypes.pth"
    SAM_CHECKPOINT = "/N/slate/veravi/model/sam2_hiera_large.pt"
else:
    VIDEO_DIR = "/Volumes/Seagate/CSCI_B657/test_videos"
    OUTPUT_DIR = "/Volumes/Seagate/CSCI_B657/data/regions_output"
    CSV_OUTPUT = "/Volumes/Seagate/CSCI_B657/data/results.csv"
    PROTOTYPE_FILE = "/Volumes/Seagate/CSCI_B657/data/model/prototypes.pth"
    SAM_CHECKPOINT = "/Volumes/Seagate/CSCI_B657/data/model/sam2_hiera_large.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# =========================
# DINO MODEL
# =========================
model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# =========================
# LOAD PROTOTYPES
# =========================
prototypes = torch.load(PROTOTYPE_FILE, map_location=device)
print(f"Loaded classes: {list(prototypes.keys())}")

# =========================
# SAM2 SETUP (FINAL FIX)
# =========================
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2.build_sam import build_sam2

CONFIG_ROOT = "/N/u/veravi/BigRed200/.conda/envs/hand_env/lib/python3.11/site-packages/sam2/configs"

sam_model = build_sam2(
    config_file="sam2_hiera_l",
    ckpt_path=SAM_CHECKPOINT,
    hydra_overrides_extra=[
        f'hydra.searchpath=[file://{CONFIG_ROOT}/sam2]'
    ]
)

sam_model.to(device)
predictor = SAM2ImagePredictor(sam_model)

# =========================
# FUNCTIONS
# =========================
def get_embedding(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)
    feat = F.normalize(feat, dim=1)
    return feat.squeeze()

def similarity(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

def predict_label(emb):
    best_label, best_score = None, -1

    for label, emb_list in prototypes.items():
        for ref in emb_list:
            ref = ref.to(device)
            score = similarity(emb, ref)

            if score > best_score:
                best_score = score
                best_label = label

    if best_score < 0.25:
        best_label = "unknown"

    return best_label, best_score

# =========================
# MAIN
# =========================
def process_video(video_path, rows):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    frame_id = 0
    video_name = os.path.basename(video_path).split(".")[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Skip frames
        if frame_id % 2 != 0:
            continue

        print(f"Processing frame {frame_id}")

        frame_name = f"{video_name}_frame_{frame_id:05d}"
        frame_dir = os.path.join(OUTPUT_DIR, frame_name)
        os.makedirs(frame_dir, exist_ok=True)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        masks, scores, _ = predictor.predict(multimask_output=True)

        region_id = 0

        for mask in masks:
            mask = mask.astype(np.uint8)

            if mask.sum() < 1000:
                continue

            ys, xs = np.where(mask > 0)

            if len(ys) == 0 or len(xs) == 0:
                continue

            region_id += 1

            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()

            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                pil_img = Image.fromarray(crop)
            except:
                continue

            region_path = os.path.join(frame_dir, f"region_{region_id}.png")
            pil_img.save(region_path)

            emb = get_embedding(pil_img)
            label, score = predict_label(emb)

            rows.append([
                frame_name,
                region_id,
                label,
                round(score, 4)
            ])

            print(f"{frame_name} | R{region_id} → {label} ({score:.3f})")

    cap.release()

def run():
    rows = []

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    if len(videos) == 0:
        print("❌ No videos found")
        return

    for video in videos:
        print(f"\nProcessing: {video}")
        process_video(os.path.join(VIDEO_DIR, video), rows)

    rows.sort(key=lambda x: x[0])

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "region", "label", "confidence"])
        writer.writerows(rows)

    print(f"\n✅ CSV saved: {CSV_OUTPUT}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 Script started")
    run()