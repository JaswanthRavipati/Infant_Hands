import os
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =========================
# CONFIG
# =========================
INPUT_DIR = "/Volumes/Seagate/CSCI_B657/data/test_sam"                # folder of frames
OUTPUT_DIR = "/Volumes/Seagate/CSCI_B657/data/regions_output"      # output folder
MODEL_PATH = "/Volumes/Seagate/CSCI_B657/data/model/sam_vit_b_01ec64.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else  "cpu"
print(f"Using device: {device}")

# =========================
# LOAD SAM
# =========================
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
sam.to(device)
print("SAM loaded")

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=1000
)
print("Mask generator ready")
# =========================
# PROCESS FRAMES
# =========================
def process_frame(frame_path, frame_name):
    frame = cv2.imread(frame_path)
    print(f"Processing: {frame_name}")

    if frame is None:
        print(f"❌ Failed: {frame_path}")
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape

    masks = mask_generator.generate(image)

    # filter masks
    filtered = []
    for m in masks:
        area = m['area']

        if area < 3000:
            continue

        if area > 0.8 * H * W:
            continue

        filtered.append(m)

    # create folder per frame
    frame_output_dir = os.path.join(OUTPUT_DIR, frame_name)
    os.makedirs(frame_output_dir, exist_ok=True)

    count = 0

    for m in filtered:
        x, y, w, h = map(int, m['bbox'])
        crop = image[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        if w < 20 or h < 20:
            continue

        save_path = os.path.join(frame_output_dir, f"region_{count}.png")

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, crop_bgr)

        count += 1

    print(f"{frame_name} → {count} regions saved")


# =========================
# MAIN LOOP
# =========================
def run():
    frame_files = sorted(os.listdir(INPUT_DIR))

    for file in frame_files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        frame_path = os.path.join(INPUT_DIR, file)
        frame_name = os.path.splitext(file)[0]

        process_frame(frame_path, frame_name)


if __name__ == "__main__":
    run()