from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path(
    "data/processed/swefil/preprocessed/images/train"
)

MASK_DIR = Path(
    "data/processed/swefil/preprocessed/masks/train"
)

OUTPUT_DIR = Path(
    "outputs/preprocessed_validation"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 5


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(image, mask):

    image = image.convert("RGB")

    image_array = np.array(image)
    mask_array = np.array(mask)

    overlay = image_array.copy()

    # Mark filament pixels
    filament_pixels = mask_array > 0

    # Red overlay
    overlay[filament_pixels] = [255, 0, 0]

    # Blend original + mask
    blended = (
        0.65 * image_array +
        0.35 * overlay
    ).astype(np.uint8)

    return Image.fromarray(blended)


# ============================================================
# CREATE 3-PANEL IMAGE
# ============================================================

def create_visualization(image, mask, output_path):

    overlay = create_overlay(
        image,
        mask
    )

    # Create canvas
    canvas = Image.new(
        "RGB",
        (768, 256),
        "white"
    )

    canvas.paste(
        image.convert("RGB"),
        (0, 0)
    )

    canvas.paste(
        mask.convert("RGB"),
        (256, 0)
    )

    canvas.paste(
        overlay,
        (512, 0)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        "IMAGE",
        fill="white"
    )

    draw.text(
        (266, 10),
        "GROUND TRUTH MASK",
        fill="white"
    )

    draw.text(
        (522, 10),
        "OVERLAY",
        fill="white"
    )

    canvas.save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        IMAGE_DIR.glob("*_256.png")
    )[:NUM_SAMPLES]

    print("=" * 60)
    print("PREPROCESSED DATA VISUAL VALIDATION")
    print("=" * 60)

    print("Samples:", len(image_files))

    for image_path in image_files:

        image_id = image_path.stem.replace(
            "_256",
            ""
        )

        mask_path = (
            MASK_DIR /
            f"{image_id}_mask_256.png"
        )

        if not mask_path.exists():

            print(
                f"WARNING: Missing mask: "
                f"{mask_path.name}"
            )

            continue

        image = Image.open(
            image_path
        )

        mask = Image.open(
            mask_path
        )

        output_path = (
            OUTPUT_DIR /
            f"{image_id}_validation.png"
        )

        create_visualization(
            image,
            mask,
            output_path
        )

        print(
            f"Created: {output_path.name}"
        )

    print("\n" + "=" * 60)
    print("VALIDATION VISUALS CREATED")
    print("=" * 60)

    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()