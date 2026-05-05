import os
import torch
import timm
from PIL import Image
from torchvision import transforms

# =========================
# CONFIG
# =========================
DATASET_DIR = "/Volumes/Seagate/CSCI_B657/data/toys_selected"         # training images (class folders)
SAVE_PATH = "/Volumes/Seagate/CSCI_B657/data/model/prototypes.pth"    # output file

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
# TRANSFORM (518 REQUIRED)
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
# EMBEDDING FUNCTION
# =========================
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img)

    return features.squeeze().cpu()  # save on CPU

# =========================
# TRAIN (CREATE PROTOTYPES)
# =========================
def train():
    prototypes = {}

    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(label_path):
            continue

        embeddings = []

        print(f"\nProcessing class: {label}")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            try:
                emb = get_embedding(img_path)
                embeddings.append(emb)
                print(f"  Loaded: {img_name}")
            except Exception as e:
                print(f"  Skipped {img_name}: {e}")

        if embeddings:
            prototypes[label] = embeddings

    # SAVE
    torch.save(prototypes, SAVE_PATH)
    print(f"\n✅ Saved prototypes to {SAVE_PATH}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()