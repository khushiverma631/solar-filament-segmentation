from pathlib import Path
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/processed/swefil/preprocessed")

TEST_IMAGES = ROOT / "images" / "test"
TEST_MASKS = ROOT / "masks" / "test"

MODEL_PATH = Path("models/unet_best.pth")

OUTPUT_DIR = Path("outputs/unet_predictions_512")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = 42
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# LOAD MODEL
# ============================================================

print("=" * 60)
print("U-NET VISUAL PREDICTION")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Saved Dice: {checkpoint['dice']:.4f}")
print(f"Saved IoU: {checkpoint['iou']:.4f}")


# ============================================================
# FIND TEST IMAGE-MASK PAIRS
# ============================================================

pairs = []

for image_path in sorted(TEST_IMAGES.glob("*.png")):

    image_id = image_path.stem

    if not image_id.endswith("_512"):
        continue

    mask_name = image_id.replace(
        "_512",
        "_mask_512"
    ) + ".png"

    mask_path = TEST_MASKS / mask_name

    if mask_path.exists():
        pairs.append(
            (image_path, mask_path)
        )


print(f"Test pairs found: {len(pairs)}")


if len(pairs) == 0:
    raise RuntimeError(
        "No test image-mask pairs found."
    )


# ============================================================
# SELECT SAMPLES
# ============================================================

random.seed(SEED)

if len(pairs) > NUM_SAMPLES:
    selected_pairs = random.sample(
        pairs,
        NUM_SAMPLES
    )
else:
    selected_pairs = pairs


# ============================================================
# DICE / IOU
# ============================================================

def calculate_dice(pred, target):

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    denominator = (
        pred.sum() +
        target.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
    ) / denominator


def calculate_iou(pred, target):

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    union = np.logical_or(
        pred,
        target
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# PREDICTION
# ============================================================

all_dice = []
all_iou = []

for index, (image_path, mask_path) in enumerate(
    selected_pairs,
    start=1
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")

    mask = Image.open(
        mask_path
    ).convert("L")

    image_np = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    mask_np = np.asarray(
        mask,
        dtype=np.float32
    ) / 255.0

    mask_binary = mask_np > 0.5

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image_np
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )

    probability_np = (
        probability.squeeze()
        .cpu()
        .numpy()
    )

    prediction_binary = (
        probability_np > 0.5
    )

    # --------------------------------------------------------
    # Metrics for this image
    # --------------------------------------------------------

    dice = calculate_dice(
        prediction_binary,
        mask_binary
    )

    iou = calculate_iou(
        prediction_binary,
        mask_binary
    )

    all_dice.append(dice)
    all_iou.append(iou)

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = np.stack(
        [
            image_np,
            image_np,
            image_np
        ],
        axis=-1
    )

    # Red where prediction says filament
    overlay[prediction_binary, 0] = 1.0
    overlay[prediction_binary, 1] *= 0.25
    overlay[prediction_binary, 2] *= 0.25

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        image_np,
        cmap="gray"
    )

    axes[0].set_title(
        "Original H-alpha"
    )

    axes[1].imshow(
        mask_binary,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        prediction_binary,
        cmap="gray"
    )

    axes[2].set_title(
        "U-Net Prediction"
    )

    axes[3].imshow(
        overlay
    )

    axes[3].set_title(
        "Prediction Overlay"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_path.stem} | Dice: {dice:.4f} | IoU: {iou:.4f}"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"{index:02d}_{image_path.stem}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[{index:02d}/{len(selected_pairs)}] "
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | "
        f"{image_path.name}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("VISUAL VALIDATION COMPLETE")
print("=" * 60)

print(
    f"Average Dice: "
    f"{np.mean(all_dice):.4f}"
)

print(
    f"Average IoU:  "
    f"{np.mean(all_iou):.4f}"
)

print(
    f"Results saved to: "
    f"{OUTPUT_DIR}"
)