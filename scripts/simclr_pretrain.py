"""
SimCLR Pre-training on hand crop images.

Learns visual representations from unlabeled 224x224 hand crops using
contrastive learning. No class labels needed.

Output: checkpoint saved to CHECKPOINT_DIR/simclr_resnet50_best.pt
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from simclr_config import (
    HAND_CROP_DIR, CHECKPOINT_DIR,
    SIMCLR_BACKBONE, SIMCLR_OUT_DIM, SIMCLR_TEMPERATURE,
    SIMCLR_EPOCHS, SIMCLR_BATCH_SIZE, SIMCLR_LR,
    SIMCLR_WEIGHT_DECAY, SIMCLR_NUM_WORKERS, SIMCLR_IMG_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# -----------------------------
# AUGMENTATION
# -----------------------------

def simclr_augment(img_size):
    kernel_size = int(0.1 * img_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                   saturation=0.8, hue=0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# -----------------------------
# DATASET — returns two augmented views of the same image
# -----------------------------

class HandCropDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith(".jpg")
        ]
        self.transform = transform
        log.info("Dataset: %d images from %s", len(self.paths), root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)  # two views

# -----------------------------
# MODEL — ResNet-50 + projection head
# -----------------------------

class SimCLR(nn.Module):
    def __init__(self, backbone_name, out_dim):
        super().__init__()
        backbone = getattr(models, backbone_name)(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# -----------------------------
# NT-XENT LOSS
# -----------------------------

def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                             # [2B, D]
    sim = torch.mm(z, z.T) / temperature                       # [2B, 2B]

    # mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # positive pair for view i is at index i+B (and vice versa)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z.device)

    return F.cross_entropy(sim, labels)

# -----------------------------
# TRAINING LOOP
# -----------------------------

augment   = simclr_augment(SIMCLR_IMG_SIZE)
dataset   = HandCropDataset(HAND_CROP_DIR, augment)
loader    = DataLoader(dataset, batch_size=SIMCLR_BATCH_SIZE, shuffle=True,
                       num_workers=SIMCLR_NUM_WORKERS, pin_memory=True,
                       drop_last=True)

model     = SimCLR(SIMCLR_BACKBONE, SIMCLR_OUT_DIM).to(DEVICE)
lr        = SIMCLR_LR * SIMCLR_BATCH_SIZE / 256               # linear scaling
optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                             momentum=0.9, weight_decay=SIMCLR_WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=SIMCLR_EPOCHS
)

best_loss = float("inf")

for epoch in range(1, SIMCLR_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for (v1, v2) in loader:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        z1 = model(v1)
        z2 = model(v2)
        loss = nt_xent_loss(z1, z2, SIMCLR_TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(loader)

    log.info("Epoch [%d/%d]  loss=%.4f  lr=%.6f",
             epoch, SIMCLR_EPOCHS, avg_loss, scheduler.get_last_lr()[0])

    if avg_loss < best_loss:
        best_loss = avg_loss
        ckpt_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_best.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": best_loss,
        }, ckpt_path)
        log.info("Saved best checkpoint → %s", ckpt_path)

    # periodic checkpoint every 50 epochs
    if epoch % 50 == 0:
        periodic_path = os.path.join(CHECKPOINT_DIR, f"simclr_resnet50_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
        }, periodic_path)

log.info("Pre-training complete. Best loss: %.4f", best_loss)
log.info("Checkpoint: %s/simclr_resnet50_best.pt", CHECKPOINT_DIR)
