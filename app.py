from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/unet_best.pth")

RAW_IMAGE_DIR = Path(
    "data/processed/swefil/images/test"
)

RAW_MASK_DIR = Path(
    "data/processed/swefil/masks/test"
)

TARGET_SIZE = (512, 512)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Solar Filament Segmentation",
    
    layout="wide"
)


# ============================================================
# DASHBOARD STYLING
# ============================================================

st.markdown("""
<style>
.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1f2937;
    margin-top: 25px;
    margin-bottom: 12px;
    padding: 10px 15px;
    border-left: 6px solid #2563EB;
    background: linear-gradient(90deg, #EFF6FF, transparent);
    border-radius: 5px;
}

.section-step {
    font-size: 0.78rem;
    font-weight: 700;
    color: #2563EB;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)


def section_heading(number, title, color="#2563EB"):
    st.markdown(
        f"""
        <div class="section-title"
             style="border-left-color:{color};
                    background:linear-gradient(90deg, {color}12, transparent);">
            <div class="section-step" style="color:{color};">
                STEP {number}
            </div>
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


st.title("Solar Filament Segmentation")
st.subheader("SWEFIL + U-Net Segmentation Prototype")

st.write(
    "Upload a SWEFIL test image and visualize the complete "
    "segmentation pipeline."
)

st.divider()


# ============================================================
# U-NET
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
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=2,
            stride=2
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
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


# ============================================================
# PREPROCESSING
# Same pipeline used in preprocess_dataset.py
# ============================================================

def preprocess_image(image):

    # 1. Grayscale
    image = image.convert("L")

    # 2. Resize to 512 × 512
    image = image.resize(
        TARGET_SIZE,
        Image.Resampling.BILINEAR
    )

    # 3. Normalize 0–255 → 0–1
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    return image, image_array


# ============================================================
# PREPROCESS GROUND TRUTH MASK
# Same pipeline used during dataset preparation
# ============================================================

def preprocess_mask(mask):

    mask = mask.convert("L")

    # Resize using NEAREST
    mask = mask.resize(
        TARGET_SIZE,
        Image.Resampling.NEAREST
    )

    mask_array = np.asarray(mask)

    # Binary mask
    mask_array = np.where(
        mask_array > 0,
        1.0,
        0.0
    ).astype(np.float32)

    return mask_array


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    pred_pixels = prediction.sum()
    target_pixels = target.sum()

    union = np.logical_or(
        prediction,
        target
    ).sum()

    dice = (
        2 * intersection
        / (pred_pixels + target_pixels)
        if (pred_pixels + target_pixels) > 0
        else 1.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    return dice, iou


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(image_array, prediction):

    # Convert grayscale to RGB
    overlay = np.stack(
        [
            image_array,
            image_array,
            image_array
        ],
        axis=-1
    )

    # Highlight predicted filament
    overlay[prediction, 0] = 1.0
    overlay[prediction, 1] *= 0.2
    overlay[prediction, 2] *= 0.2

    return overlay


# ============================================================
# IMAGE SELECTION
# ============================================================

section_heading(1, "Select SWEFIL Test Image", "#1E3A5F")

image_files = sorted(
    RAW_IMAGE_DIR.glob("*.jpg")
)

if len(image_files) == 0:

    st.error(
        f"No test images found in {RAW_IMAGE_DIR}"
    )

    st.stop()


selected_image = st.selectbox(
    "Choose a test image",
    image_files,
    format_func=lambda x: x.name
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.divider()

section_heading(2, "Original Image", "#2563EB")

original_image = Image.open(
    selected_image
).convert("L")

st.image(
    original_image,
    caption=selected_image.name,
    width=500
)


# ============================================================
# PREPROCESSING
# ============================================================

st.divider()

section_heading(3, "Preprocessing", "#0F766E")

st.write(
    """
    The same preprocessing used for the U-Net training data:
    
   **Grayscale → Resize to 512×512 → Normalize 0–1**
    """
)

if st.button(
    " Preprocess Image",
    use_container_width=True
):

    processed_image, processed_array = preprocess_image(
        original_image
    )

    st.success(
        "Preprocessing completed."
    )

    st.image(
        processed_image,
        caption="Preprocessed 512×512 Image",
        width=500
    )

    st.session_state["processed_array"] = processed_array


# ============================================================
# RUN U-NET
# ============================================================

if "processed_array" in st.session_state:

    st.divider()

    section_heading(4, "U-Net Segmentation", "#0F766E")

    if st.button(
        "Run U-Net",
        use_container_width=True
    ):

        processed_array = st.session_state[
            "processed_array"
        ]

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(
            processed_array
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(DEVICE)

        # Prediction
        with torch.no_grad():

            logits = model(tensor)

            probabilities = torch.sigmoid(
                logits
            )

        probability = (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        )

        # Binary prediction
        prediction = (
            probability > 0.5
        )

        st.session_state["prediction"] = prediction
        st.session_state["processed_array"] = processed_array

        st.success(
            "U-Net segmentation completed."
        )


# ============================================================
# RESULTS
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state[
        "prediction"
    ]

    processed_array = st.session_state[
        "processed_array"
    ]


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    st.divider()

    section_heading(5, "Predicted Filament Mask", "#7C3AED")

    st.image(
        prediction.astype(np.uint8) * 255,
        caption="U-Net Predicted Binary Mask",
        width=500
    )

    st.write(
        "White = predicted solar filament | "
        "Black = background"
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    st.divider()

    section_heading(6, "Prediction Overlay", "#7C3AED")

    overlay = create_overlay(
        processed_array,
        prediction
    )

    st.image(
        overlay,
        caption="Predicted Filament Overlay",
        width=500
    )


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    st.divider()

    section_heading(7, "Ground Truth Mask", "#7C3AED")

    image_id = selected_image.stem

    ground_truth_path = (
        RAW_MASK_DIR /
        f"{image_id}_mask.png"
    )

    if ground_truth_path.exists():

        ground_truth_image = Image.open(
            ground_truth_path
        )

        ground_truth = preprocess_mask(
            ground_truth_image
        )

        st.image(
            (ground_truth * 255).astype(np.uint8),
            caption="SWEFIL Ground Truth Mask",
            width=500
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        dice, iou = calculate_metrics(
            prediction,
            ground_truth
        )

        st.divider()

        section_heading(8, "Segmentation Evaluation", "#6D28D9")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Dice Score",
                f"{dice:.4f}"
            )

        with col2:

            st.metric(
                "IoU Score",
                f"{iou:.4f}"
            )


        st.success(
            "Prediction compared with the corresponding "
            "SWEFIL ground-truth mask."
        )


    else:

        st.warning(
            "Ground-truth mask was not found for this image. "
            "Dice and IoU cannot be calculated."
        )

def evaluate_all_test_images(model):
    results = []

    image_files = sorted(
        RAW_IMAGE_DIR.glob("*.jpg")
    )

    for image_path in image_files:

        mask_path = (
            RAW_MASK_DIR /
            f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            continue

        # Image
        image = Image.open(image_path).convert("L")

        _, image_array = preprocess_image(image)

        tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():

            logits = model(tensor)

            probability = torch.sigmoid(
                logits
            )

        prediction = (
            probability.squeeze().cpu().numpy() > 0.5
        )

        # Ground truth
        ground_truth_image = Image.open(
            mask_path
        )

        ground_truth = preprocess_mask(
            ground_truth_image
        )

        # Metrics
        dice, iou = calculate_metrics(
            prediction,
            ground_truth
        )

        results.append({
            "image": image_path.name,
            "dice": dice,
            "iou": iou
        })

    return results


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


# ============================================================
# AUTOMATIC MODEL INSIGHTS
# ============================================================

st.divider()

section_heading(9, "Automatic Model Insights", "#047857")

if st.button(
    "Analyze All 42 Test Images",
    use_container_width=True
):

    with st.spinner(
        "Evaluating all test images..."
    ):

        all_results = evaluate_all_test_images(model)

    if len(all_results) == 0:

        st.error(
            "No valid image-mask pairs were found."
        )

    else:

        dice_scores = np.array([
            r["dice"]
            for r in all_results
        ])

        iou_scores = np.array([
            r["iou"]
            for r in all_results
        ])

        # Statistics
        mean_dice = dice_scores.mean()
        median_dice = np.median(dice_scores)

        mean_iou = iou_scores.mean()
        median_iou = np.median(iou_scores)

        best_idx = np.argmax(dice_scores)
        worst_idx = np.argmin(dice_scores)

        best_result = all_results[best_idx]
        worst_result = all_results[worst_idx]

        # Performance groups
        strong = np.sum(dice_scores >= 0.75)

        good = np.sum(
            (dice_scores >= 0.60) &
            (dice_scores < 0.75)
        )

        moderate = np.sum(
            (dice_scores >= 0.40) &
            (dice_scores < 0.60)
        )

        weak = np.sum(
            dice_scores < 0.40
        )

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        st.subheader("Overall Test-Set Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Images Evaluated",
                len(all_results)
            )

        with col2:
            st.metric(
                "Mean Dice",
                f"{mean_dice:.4f}"
            )

        with col3:
            st.metric(
                "Mean IoU",
                f"{mean_iou:.4f}"
            )

        with col4:
            st.metric(
                "Median Dice",
                f"{median_dice:.4f}"
            )

        # ----------------------------------------------------
        # AUTOMATIC INTERPRETATION
        # ----------------------------------------------------

        st.subheader("Automatic Insights")

        if mean_dice >= 0.75:
            quality = "Strong"
            message = (
                "The model shows strong segmentation "
                "overlap across the test set."
            )

        elif mean_dice >= 0.60:
            quality = "Good"
            message = (
                "The model shows good segmentation "
                "overlap across the test set, with "
                "some room for improvement."
            )

        elif mean_dice >= 0.40:
            quality = "Moderate"
            message = (
                "The model achieves moderate segmentation "
                "overlap. Further model or preprocessing "
                "improvements may be beneficial."
            )

        else:
            quality = "Needs Improvement"
            message = (
                "The model currently shows limited "
                "segmentation overlap and requires "
                "further improvement."
            )

        st.info(
            f"**Overall segmentation quality: {quality}**\n\n"
            f"{message}"
        )

        # ----------------------------------------------------
        # BEST / WORST CASE
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f" **Best case**\n\n"
                f"{best_result['image']}\n\n"
                f"Dice: **{best_result['dice']:.4f}**\n\n"
                f"IoU: **{best_result['iou']:.4f}**"
            )

        with col2:

            st.warning(
                f" **Weakest case**\n\n"
                f"{worst_result['image']}\n\n"
                f"Dice: **{worst_result['dice']:.4f}**\n\n"
                f"IoU: **{worst_result['iou']:.4f}**"
            )

        # ----------------------------------------------------
        # PERFORMANCE DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Dice Score Distribution")

        st.write(
            f" Strong (≥ 0.75): **{strong} images**"
        )

        st.write(
            f" Good (0.60–0.75): **{good} images**"
        )

        st.write(
            f" Moderate (0.40–0.60): **{moderate} images**"
        )

        st.write(
            f" Weak (< 0.40): **{weak} images**"
        )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("Model Recommendation")

        if weak > len(all_results) * 0.25:

            st.warning(
                "A relatively large portion of the test set "
                "has low Dice scores. Consider improving "
                "preprocessing, augmentation, loss functions, "
                "or model architecture."
            )

        elif weak > 0:

            st.info(
                "The model performs well on many images, "
                "but the low-performing cases should be "
                "visually inspected for difficult filament "
                "structures or annotation differences."
            )

        else:

            st.success(
                "All evaluated images achieved Dice ≥ 0.40. "
                "The baseline shows consistent segmentation "
                "performance across the test set."
            )

        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        st.subheader("Test-Set Results")

        results_sorted = sorted(
            all_results,
            key=lambda x: x["dice"],
            reverse=True
        )

        st.dataframe(
            results_sorted,
            use_container_width=True
        )


st.caption(
    f"Model: U-Net | "
    f"Best training epoch: {checkpoint['epoch']} | "
    f"Best Dice: {checkpoint['dice']:.4f} | "
    f"Best IoU: {checkpoint['iou']:.4f} | "
    f"Device: {DEVICE}"
)