import os
import csv
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
REGIONS_DIR = "/Volumes/Seagate/CSCI_B657/data/regions_output"
PROTOTYPE_FILE = "/Volumes/Seagate/CSCI_B657/data/model/prototypes.pth"
CSV_OUTPUT = "/Volumes/Seagate/CSCI_B657/data/results.csv"

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# LOAD DINO MODEL
# =========================
model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
model.eval()
model.to(device)

# =========================
# TRANSFORM (IMPORTANT)
# =========================
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# =========================
# LOAD PROTOTYPES
# =========================
if not os.path.exists(PROTOTYPE_FILE):
    raise FileNotFoundError(f"❌ Missing: {PROTOTYPE_FILE}")

prototypes = torch.load(PROTOTYPE_FILE)
print(f"Loaded classes: {list(prototypes.keys())}")

# =========================
# FUNCTIONS
# =========================
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img)

    return features.squeeze()

def similarity(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

def predict(query_embedding):
    best_label = None
    best_score = -1

    for label, emb_list in prototypes.items():
        for emb in emb_list:
            emb = emb.to(device)
            score = similarity(query_embedding, emb)

            if score > best_score:
                best_score = score
                best_label = label

    return best_label, best_score

# =========================
# MAIN PIPELINE
# =========================
def run():
    rows = []

    if not os.path.exists(REGIONS_DIR):
        raise FileNotFoundError(f"❌ Folder not found: {REGIONS_DIR}")

    frame_folders = sorted(os.listdir(REGIONS_DIR))

    for frame_folder in frame_folders:
        frame_path = os.path.join(REGIONS_DIR, frame_folder)

        if not os.path.isdir(frame_path):
            continue

        # -------------------------
        # Extract frame number
        # -------------------------
        try:
            frame_number = int(frame_folder.split("_")[-1])
        except:
            print(f"⚠️ Skipping invalid folder: {frame_folder}")
            continue

        region_files = sorted(os.listdir(frame_path))

        for region_file in region_files:
            if not region_file.lower().endswith(".png"):
                continue

            region_path = os.path.join(frame_path, region_file)

            # -------------------------
            # Extract region number
            # -------------------------
            try:
                region_number = int(region_file.split("_")[-1].split(".")[0])
            except:
                print(f"⚠️ Skipping invalid file: {region_file}")
                continue

            try:
                emb = get_embedding(region_path)
                label, score = predict(emb)

                if score < 0.20:
                    label = "unknown"

                rows.append([
                    frame_number,
                    region_number,
                    label,
                    round(score, 4)
                ])

                print(f"Frame {frame_number} | Region {region_number} → {label} ({score:.3f})")

            except Exception as e:
                print(f"❌ Error: {region_file} → {e}")

    # -------------------------
    # SORT RESULTS
    # -------------------------
    rows.sort(key=lambda x: (x[0], x[1]))

    # -------------------------
    # SAVE CSV
    # -------------------------
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "region", "label", "confidence"])
        writer.writerows(rows)

    print(f"\n✅ CSV saved successfully: {CSV_OUTPUT}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()