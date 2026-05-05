"""
SimCLR Fine-tuning (Linear Probe) for hand class classification.

Freezes the pre-trained ResNet-50 backbone and trains a linear classifier
on top to distinguish: baby / adult / other_kids.

Expected labeled data directory structure:
    LABELED_DATA_DIR/
        baby/
            img1.jpg ...
        adult/
            img1.jpg ...
        other_kids/
            img1.jpg ...

Output: checkpoint saved to FINETUNE_CKPT_DIR/finetune_best.pt
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets

from simclr_config import (
    LABELED_DATA_DIR, CHECKPOINT_DIR, FINETUNE_CKPT_DIR,
    CLASS_NAMES, NUM_CLASSES,
    SIMCLR_BACKBONE, SIMCLR_OUT_DIM,
    FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH_SIZE,
    FINETUNE_NUM_WORKERS, FINETUNE_VAL_SPLIT, FINETUNE_SEED,
    SIMCLR_IMG_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs(FINETUNE_CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# -----------------------------
# DATA
# -----------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(SIMCLR_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(LABELED_DATA_DIR, transform=transform)

# verify class ordering matches config
assert full_dataset.classes == sorted(CLASS_NAMES), (
    f"Folder classes {full_dataset.classes} don't match CLASS_NAMES {CLASS_NAMES}. "
    f"Ensure labeled folder names are: {CLASS_NAMES}"
)
log.info("Classes: %s", full_dataset.classes)
log.info("Total labeled samples: %d", len(full_dataset))

val_size   = int(len(full_dataset) * FINETUNE_VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(FINETUNE_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE,
                          shuffle=True, num_workers=FINETUNE_NUM_WORKERS,
                          pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE,
                          shuffle=False, num_workers=FINETUNE_NUM_WORKERS,
                          pin_memory=True)

log.info("Train: %d | Val: %d", train_size, val_size)

# -----------------------------
# MODEL — load SimCLR backbone, freeze it, attach linear head
# -----------------------------

backbone = getattr(models, SIMCLR_BACKBONE)(weights=None)
feat_dim = backbone.fc.in_features
backbone.fc = nn.Identity()

ckpt_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_best.pt")
ckpt = torch.load(ckpt_path, map_location=DEVICE)

# strip the projector weights — only backbone keys needed
backbone_state = {
    k.replace("backbone.", ""): v
    for k, v in ckpt["model_state_dict"].items()
    if k.startswith("backbone.")
}
backbone.load_state_dict(backbone_state)
log.info("Loaded SimCLR backbone from epoch %d (loss=%.4f)",
         ckpt["epoch"], ckpt["loss"])

# freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

# linear classifier head
model = nn.Sequential(
    backbone,
    nn.Linear(feat_dim, NUM_CLASSES)
).to(DEVICE)

# -----------------------------
# TRAINING
# -----------------------------

optimizer = torch.optim.Adam(
    model[-1].parameters(),   # only train the linear head
    lr=FINETUNE_LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=FINETUNE_EPOCHS
)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

for epoch in range(1, FINETUNE_EPOCHS + 1):

    # ---- train ----
    model.train()
    model[0].eval()  # keep backbone in eval (frozen BN stats)
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- val ----
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    scheduler.step()
    val_acc    = correct / total
    avg_loss   = train_loss / len(train_loader)

    log.info("Epoch [%d/%d]  loss=%.4f  val_acc=%.4f",
             epoch, FINETUNE_EPOCHS, avg_loss, val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(FINETUNE_CKPT_DIR, "finetune_best.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": best_val_acc,
            "classes": CLASS_NAMES,
        }, save_path)
        log.info("Saved best model → %s  (val_acc=%.4f)", save_path, best_val_acc)

log.info("Fine-tuning complete. Best val acc: %.4f", best_val_acc)
