# Solar Filament Segmentation

A deep learning project for pixel-level segmentation of solar filaments in H-alpha solar images using a U-Net architecture.

## Overview

Solar filaments are elongated structures observed in the solar atmosphere. Detecting and segmenting these structures from solar images can help automate their identification and support further analysis of solar activity.

This project explores a U-Net based image segmentation pipeline that takes an H-alpha solar image as input and predicts a pixel-level mask corresponding to the solar filament region.

## Problem

Unlike image classification, where the goal is to determine whether a filament is present, segmentation requires the model to identify the location of the filament at the pixel level.

The main challenge is preserving thin and elongated filament structures while producing accurate segmentation masks.

## Approach

The current implementation follows a U-Net based segmentation pipeline:

```text
H-alpha Solar Image
        |
        v
Preprocessing
        |
        v
512 x 512 Input
        |
        v
     U-Net
        |
        v
Predicted Segmentation Mask
        |
        v
Evaluation
(Dice / IoU)
```

### U-Net

The model uses the encoder-decoder structure of U-Net.

* The encoder extracts progressively higher-level visual features.
* The decoder reconstructs the spatial representation.
* Skip connections transfer spatial information from the encoder to the decoder.
* The final output is a pixel-level segmentation mask.

## Dataset

This project uses the **SWEFIL solar filament dataset**, containing H-alpha solar observations and corresponding filament annotations.

The dataset is **not included in this repository** because of its size.

Dataset source:

**SWEFIL — Solar Filament Dataset**

Please refer to the original dataset/source for access, licensing, and usage conditions.

## Training

The current experiment was trained with:

| Parameter     | Value     |
| ------------- | --------- |
| Model         | U-Net     |
| Input size    | 512 × 512 |
| Batch size    | 2         |
| Epochs        | 30        |
| Learning rate | 0.001     |
| Best epoch    | 27        |

Training was performed locally using CPU resources.

## Results

The best recorded validation performance was:

| Metric     |  Score |
| ---------- | -----: |
| Dice Score | 0.6749 |
| IoU        | 0.5372 |

The results demonstrate that the model is able to learn the general structure of solar filaments and produce meaningful segmentation masks.

However, thin filament structures can be difficult to preserve at the current 512 × 512 resolution.

## Current Limitation

The major limitation of the current implementation is the reduced input resolution.

The original solar imagery can contain considerably more detail than the 512 × 512 images used for the current experiment. Downsampling can cause very thin filament structures to become difficult to distinguish or disappear during preprocessing.

Training directly with higher-resolution images, such as 2048 × 2048 inputs, could preserve more fine-grained filament information.

However, this would substantially increase computational and memory requirements.

## Future Improvements

Potential improvements include:

* Training with higher-resolution solar images.
* Improving preservation of thin filament structures.
* Experimenting with additional segmentation loss functions.
* Data augmentation for improved generalization.
* Comparing U-Net with other segmentation architectures.
* More extensive evaluation on unseen solar observations.
* Improving inference efficiency for higher-resolution images.

## Project Structure

```text
solar-filament-segmentation/
│
├── app.py
├── app_advanced.py
├── environment.yml
├── .gitignore
│
├── src/
│   ├── evaluation/
│   │   └── visualize_preprocessed.py
│   │
│   ├── features/
│   │   └── extract_area.py
│   │
│   └── models/
│       ├── predict.py
│       ├── predict_unet.py
│       ├── train_unet.py
│       └── unet.py
│
└── README.md
```

The dataset, trained model weights, generated outputs, cache files, and local IDE configuration are excluded from the repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/khushiverma631/solar-filament-segmentation.git
cd solar-filament-segmentation
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate solar-filament
```

## Running the Project

Training:

```bash
python src/models/train_unet.py
```

Prediction:

```bash
python src/models/predict_unet.py
```

The exact commands may depend on the dataset location and local configuration.

## Technologies

* Python
* PyTorch
* NumPy
* OpenCV
* Matplotlib
* scikit-learn
* Conda

## Project Status

Current implementation: **U-Net based solar filament segmentation**

The current version serves as a working baseline. Further improvements will focus primarily on higher-resolution processing and better preservation of thin filament structures.

## Author

**Khushi Verma**

GitHub: [khushiverma631](https://github.com/khushiverma631)
