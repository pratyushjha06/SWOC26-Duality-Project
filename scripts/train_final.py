"""
Train Final UNet Segmentation Model
Integration-ready script (no Colab-specific code)

Usage example:

python train_final_unet.py \
  --train_dir /content/drive/MyDrive/SWOC26_Training/data/Train \
  --val_dir   /content/drive/MyDrive/SWOC26_Training/data/Val \
  --save_dir  ./models/final_unet \
  --epochs    30 \
  --batch_size 4 \
  --lr 1e-4
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ============================================================
# Mask Conversion (10-class IDs)
# ============================================================

# Raw IDs from organizer PDF -> class indices 0..9
RAWID_TO_CLASS = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}

N_CLASSES = 10


def convert_mask(mask_pil: Image.Image) -> np.ndarray:
    """
    Convert raw uint16 mask with IDs like 100,200,...,10000
    into a uint8 array of class indices 0..9.
    """
    mask_raw = np.array(mask_pil, dtype=np.uint16)
    class_mask = np.zeros_like(mask_raw, dtype=np.uint8)

    for raw_id, cid in RAWID_TO_CLASS.items():
        class_mask[mask_raw == raw_id] = cid

    return class_mask  # (H, W), values 0..9


# ============================================================
# Dataset
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_transform=None):
        """
        data_dir: contains Color_Images/ and Segmentation/
        """
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir  = os.path.join(data_dir, "Segmentation")

        self.ids = [f for f in os.listdir(self.image_dir)
                    if f.lower().endswith(".png")]

        self.img_transform = img_transform

        print(f"Loaded {len(self.ids)} samples from {data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_path  = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir,  name)

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # --- convert raw mask to class indices 0..9 ---
        mask_arr = convert_mask(mask)  # (H,W), uint8

        # --- transforms ---
        if self.img_transform is not None:
            img = self.img_transform(img)  # C,H,W tensor

        # Resize mask with NEAREST to preserve class IDs
        mask_pil_resized = mask.resize((476, 266), resample=Image.NEAREST)
        mask_arr = convert_mask(mask_pil_resized)  # 0..9

        # To tensor (H,W) -> (H,W) long
        mask_tensor = torch.from_numpy(mask_arr).long()

        return img, mask_tensor


# ============================================================
# UNet Model
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ============================================================
# Metrics and Loss
# ============================================================

def compute_iou(pred, target, num_classes=N_CLASSES):
    """
    pred: (B,C,H,W) logits
    target: (B,H,W) long
    """
    pred_classes = torch.argmax(pred, dim=1)  # (B,H,W)
    pred_flat = pred_classes.view(-1)
    target_flat = target.view(-1)

    ious = []
    for c in range(num_classes):
        pred_c = pred_flat == c
        target_c = target_flat == c
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((inter / union).item())
    return float(np.nanmean(ious))


def dice_loss(logits, targets, num_classes=N_CLASSES, smooth=1e-6):
    """
    Multi-class Dice loss on logits + target indices.
    """
    probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
    # one-hot targets: (B,H,W) -> (B,C,H,W)
    targets_onehot = F.one_hot(targets, num_classes=num_classes)  # (B,H,W,C)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    union = torch.sum(probs + targets_onehot, dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def build_weighted_ce(device):
    """
    Build class weights from your measured train frequencies.
    """
    # train_freq from your class-distribution notebook
    train_freq = np.array([
        0.0353,  # Trees
        0.0593,  # Lush Bushes
        0.1887,  # Dry Grass
        0.0110,  # Dry Bushes
        0.0439,  # Ground Clutter
        0.0281,  # Flowers
        0.0008,  # Logs
        0.0120,  # Rocks
        0.2445,  # Landscape
        0.3764   # Sky
    ], dtype=np.float32)

    eps = 1e-6
    inv_freq = 1.0 / (train_freq + eps)
    class_weights = inv_freq / inv_freq.mean()

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights_tensor)


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, ce_weighted, device):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train", ncols=100):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        ce = ce_weighted(preds, masks)
        dl = dice_loss(preds, masks)
        loss = ce + dl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, ce_weighted, device):
    model.eval()
    total_loss = 0.0
    ious = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", ncols=100):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            ce = ce_weighted(preds, masks)
            dl = dice_loss(preds, masks)
            loss = ce + dl

            total_loss += loss.item()
            ious.append(compute_iou(preds, masks))

    avg_loss = total_loss / len(loader)
    mean_iou = float(np.nanmean(ious))
    return avg_loss, mean_iou


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir",   required=True)
    parser.add_argument("--save_dir",  default="./models/final_unet")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",        type=float, default=1e-4)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Image transforms: final Brightness/Contrast + normalization
    img_transform = transforms.Compose([
        transforms.Resize((266, 476), interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_set = MaskDataset(args.train_dir, img_transform)
    val_set   = MaskDataset(args.val_dir,   img_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = UNet(in_channels=3, num_classes=N_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_weighted = build_weighted_ce(device)

    best_val_iou = -1.0
    best_path = os.path.join(args.save_dir, "best_model_final.pth")
    final_path = os.path.join(args.save_dir, "final_model.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, ce_weighted, device)
        val_loss, val_iou = evaluate(model, val_loader, ce_weighted, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   IoU : {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print("✓ Saved new best model (by Val IoU)")

    # Save final weights as well
    torch.save(model.state_dict(), final_path)
    print("\nTraining complete.")
    print("Best model (by IoU):", best_path)
    print("Final model:", final_path)


if __name__ == "__main__":
    main()