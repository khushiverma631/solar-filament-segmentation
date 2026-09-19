from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

from unet import UNet


MODEL_PATH = Path("models/unet_best.pth")

TEST_IMAGES = Path(
    "data/processed/swefil/preprocessed/images/test"
)

TEST_MASKS = Path(
    "data/processed/swefil/preprocessed/masks/test"
)

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("SWEFIL U-NET INFERENCE")
print("=" * 60)

print(f"Device: {DEVICE}")


# Load model
model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best Dice: {checkpoint['dice']:.4f}")
print(f"Best IoU: {checkpoint['iou']:.4f}")
print("Model loaded successfully!")


# Find test images
images = sorted(
    TEST_IMAGES.glob("*_512.png")
)

print(f"Test images found: {len(images)}")


# Run inference
with torch.no_grad():

    for image_path in images[:5]:

        image = Image.open(
            image_path
        ).convert("L")

        image_array = (
            np.asarray(
                image,
                dtype=np.float32
            ) / 255.0
        )

        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0).unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        # Prediction
        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities > 0.5
        ).float()

        prediction_array = (
            prediction[0, 0]
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

        # Ground truth
        mask_name = (
            image_path.stem
            .replace("_512", "_mask_512")
            + ".png"
        )

        mask_path = TEST_MASKS / mask_name

        mask = Image.open(
            mask_path
        ).convert("L")

        mask_array = np.asarray(mask)

        # Save prediction
        prediction_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_prediction.png"
        )

        Image.fromarray(
            prediction_array
        ).save(prediction_path)

        # Visualization
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(
            image_array,
            cmap="gray"
        )
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(
            mask_array,
            cmap="gray"
        )
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(
            prediction_array,
            cmap="gray"
        )
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()

        comparison_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_comparison.png"
        )

        plt.savefig(
            comparison_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Processed: {image_path.name}"
        )


print()
print("=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(
    f"Results saved to: {OUTPUT_DIR}"
)