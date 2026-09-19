from pathlib import Path
from PIL import Image
import numpy as np
import csv


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/processed/swefil/preprocessed/masks")
OUTPUT_DIR = Path("data/processed/swefil/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CALCULATE FILAMENT AREA
# ============================================================

def calculate_area(mask_path):

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    # Every non-zero pixel belongs to the filament
    filament_pixels = mask > 0

    area_pixels = np.sum(filament_pixels)

    return int(area_pixels)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    mask_dir = BASE_DIR / split

    masks = sorted(
        mask_dir.glob("*_mask_256.png")
    )

    results = []

    print()
    print("=" * 60)
    print(f"{split.upper()} FILAMENT AREA")
    print("=" * 60)

    print(f"Masks found: {len(masks)}")

    for mask_path in masks:

        area = calculate_area(mask_path)

        results.append({
            "filename": mask_path.name,
            "area_pixels": area
        })

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_results(split, results):

    output_path = (
        OUTPUT_DIR /
        f"{split}_filament_area.csv"
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "area_pixels"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SOLAR FILAMENT AREA EXTRACTION")
    print("=" * 60)

    for split in ["train", "test"]:

        results = process_split(split)

        save_results(
            split,
            results
        )

        if results:

            areas = [
                r["area_pixels"]
                for r in results
            ]

            print(
                f"Minimum area : {min(areas)} pixels"
            )

            print(
                f"Maximum area : {max(areas)} pixels"
            )

            print(
                f"Average area : {np.mean(areas):.2f} pixels"
            )

    print()
    print("=" * 60)
    print("AREA EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()