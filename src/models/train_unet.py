from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/processed/swefil/preprocessed")

TRAIN_IMAGES = ROOT / "images" / "train"
TRAIN_MASKS = ROOT / "masks" / "train"

TEST_IMAGES = ROOT / "images" / "test"
TEST_MASKS = ROOT / "masks" / "test"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-3

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SWEFIL U-NET BASELINE")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class SWEFILDataset(Dataset):

    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment

        self.images = sorted(self.image_dir.glob("*.png"))

        if len(self.images) == 0:
            raise RuntimeError(f"No PNG images found in {self.image_dir}")

        self.pairs = []

        for image_path in self.images:

            image_id = image_path.stem

            # Example:
            # image = 20100604185114Th_256.png
            # mask  = 20100604185114Th_mask_256.png

            if not image_id.endswith("_512"):
              continue

            mask_name = image_id.replace("_512", "_mask_512") + ".png"

            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                self.pairs.append((image_path, mask_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching image-mask pairs found.\n"
                f"Images: {self.image_dir}\n"
                f"Masks: {self.mask_dir}"
            )

        print(
            f"Loaded {len(self.pairs)} image-mask pairs "
            f"from {self.image_dir}"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.float32) / 255.0

        # ----------------------------------------------------
        # Basic augmentation
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # ----------------------------------------------------
        # Convert to PyTorch format
        # ----------------------------------------------------

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Ensure binary mask
        mask = (mask > 0.5).float()

        return image, mask


# ============================================================
# U-NET BLOCKS
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck

        self.bottleneck = DoubleConv(256, 512)

        # Decoder

        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )

        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        # Output

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(pred, target, smooth=1e-6):

    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred.sum(dim=1)
        + target.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(pred, target, smooth=1e-6):

    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    union = (
        pred.sum(dim=1)
        + target.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


# ============================================================
# DICE LOSS
# ============================================================

def soft_dice_loss(logits, target, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(
        probabilities.size(0), -1
    )

    target = target.view(
        target.size(0), -1
    )

    intersection = (
        probabilities * target
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1)
        + target.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):

        bce_loss = self.bce(
            logits,
            target
        )

        dice_loss = soft_dice_loss(
            logits,
            target
        )

        return bce_loss + dice_loss


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = SWEFILDataset(
    TRAIN_IMAGES,
    TRAIN_MASKS,
    augment=True
)

test_dataset = SWEFILDataset(
    TEST_IMAGES,
    TEST_MASKS,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# MODEL
# ============================================================

model = UNet().to(DEVICE)

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_dice = -1.0

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    model.eval()

    test_dice = 0.0
    test_iou = 0.0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            test_dice += dice_score(
                probabilities,
                masks
            )

            test_iou += iou_score(
                probabilities,
                masks
            )

    test_dice /= len(test_loader)
    test_iou /= len(test_loader)

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Dice: {test_dice:.4f} "
        f"IoU: {test_iou:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if test_dice > best_dice:

        best_dice = test_dice

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice": test_dice,
                "iou": test_iou,
            },
            MODEL_DIR / "unet_best.pth"
        )

        print(
            f"  ✓ Best model saved "
            f"(Dice: {best_dice:.4f})"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best Test Dice: {best_dice:.4f}")
print(f"Model saved to: {MODEL_DIR / 'unet_best.pth'}")