from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/processed/swefil/preprocessed")

TEST_IMAGES = ROOT / "images" / "test"
TEST_MASKS = ROOT / "masks" / "test"

MODEL_PATH = Path("models/unet_best.pth")

PREDICTION_DIR = Path("outputs/unet_predictions")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Filament Segmentation",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
        background-color: #fafafa;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# LOAD MODEL
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
# FIND TEST PAIRS
# ============================================================

def get_test_pairs():

    pairs = []

    for image_path in sorted(
        TEST_IMAGES.glob("*.png")
    ):

        image_id = image_path.stem

        if not image_id.endswith("_256"):
            continue

        mask_name = (
            image_id.replace(
                "_256",
                "_mask_256"
            )
            + ".png"
        )

        mask_path = TEST_MASKS / mask_name

        if mask_path.exists():
            pairs.append(
                (image_path, mask_path)
            )

    return pairs


pairs = get_test_pairs()


# ============================================================
# METRICS
# ============================================================

def dice_score(pred, target):

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    denominator = (
        pred.sum()
        + target.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2 * intersection
    ) / denominator


def iou_score(pred, target):

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

@st.cache_data
def predict_image(image_path_string):

    image_path = Path(image_path_string)

    image = Image.open(
        image_path
    ).convert("L")

    image_np = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        image_np
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    probability_np = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )

    prediction = (
        probability_np > 0.5
    )

    return image_np, probability_np, prediction


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("☀️ Solar Filament")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Prediction Explorer",
        "Test Gallery"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Device:** `{DEVICE}`"
)

st.sidebar.write(
    f"**Test Images:** `{len(pairs)}`"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '☀️ Solar Filament Segmentation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'U-Net based semantic segmentation of solar filaments '
    'in H-alpha images'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.header("Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Best Dice",
            f"{checkpoint['dice']:.4f}"
        )

    with col2:

        st.metric(
            "Best IoU",
            f"{checkpoint['iou']:.4f}"
        )

    with col3:

        st.metric(
            "Best Epoch",
            checkpoint["epoch"]
        )

    with col4:

        st.metric(
            "Test Images",
            len(pairs)
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.header("Project Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write("**Task**")
        st.info(
            "Binary Semantic Segmentation"
        )

        st.write("**Dataset**")
        st.info("SWEFIL")

        st.write("**Input**")
        st.info(
            "Grayscale H-alpha Solar Image"
        )

    with config_col2:

        st.write("**Model**")
        st.info("U-Net")

        st.write("**Input Size**")
        st.info("256 × 256")

        st.write("**Output**")
        st.info(
            "0 = Background | 1 = Filament"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SAMPLE RESULT
    # --------------------------------------------------------

    st.header("Sample Segmentation")

    if pairs:

        selected = st.selectbox(
            "Select test image",
            range(len(pairs)),
            format_func=lambda x:
                pairs[x][0].stem
        )

        image_path, mask_path = pairs[selected]

        image_np, probability, prediction = (
            predict_image(
                str(image_path)
            )
        )

        ground_truth = (
            np.asarray(
                Image.open(
                    mask_path
                ).convert("L")
            ) / 255.0
        ) > 0.5

        dice = dice_score(
            prediction,
            ground_truth
        )

        iou = iou_score(
            prediction,
            ground_truth
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.image(
                image_np,
                caption="Original H-alpha",
                use_container_width=True
            )

        with c2:
            st.image(
                ground_truth,
                caption="Ground Truth",
                use_container_width=True
            )

        with c3:
            st.image(
                prediction,
                caption="U-Net Prediction",
                use_container_width=True
            )

        with c4:

            overlay = np.stack(
                [
                    image_np,
                    image_np,
                    image_np
                ],
                axis=-1
            )

            overlay[prediction, 0] = 1.0
            overlay[prediction, 1] *= 0.25
            overlay[prediction, 2] *= 0.25

            st.image(
                overlay,
                caption="Prediction Overlay",
                use_container_width=True
            )

        st.success(
            f"Dice: {dice:.4f}   |   IoU: {iou:.4f}"
        )


# ============================================================
# PREDICTION EXPLORER
# ============================================================

elif page == "Prediction Explorer":

    st.header("🔬 Prediction Explorer")

    if not pairs:

        st.error(
            "No test image-mask pairs found."
        )

    else:

        selected = st.selectbox(
            "Choose a test image",
            range(len(pairs)),
            format_func=lambda x:
                pairs[x][0].stem
        )

        image_path, mask_path = pairs[selected]

        image_np, probability, prediction = (
            predict_image(
                str(image_path)
            )
        )

        ground_truth = (
            np.asarray(
                Image.open(
                    mask_path
                ).convert("L")
            ) / 255.0
        ) > 0.5

        dice = dice_score(
            prediction,
            ground_truth
        )

        iou = iou_score(
            prediction,
            ground_truth
        )

        st.subheader(
            image_path.stem
        )

        c1, c2 = st.columns(2)

        with c1:

            st.image(
                image_np,
                caption="Original H-alpha Image",
                use_container_width=True
            )

        with c2:

            st.image(
                ground_truth,
                caption="Ground Truth Mask",
                use_container_width=True
            )

        c3, c4 = st.columns(2)

        with c3:

            st.image(
                prediction,
                caption="U-Net Binary Prediction",
                use_container_width=True
            )

        with c4:

            overlay = np.stack(
                [
                    image_np,
                    image_np,
                    image_np
                ],
                axis=-1
            )

            overlay[prediction, 0] = 1.0
            overlay[prediction, 1] *= 0.25
            overlay[prediction, 2] *= 0.25

            st.image(
                overlay,
                caption="Prediction Overlay",
                use_container_width=True
            )

        st.markdown("---")

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Dice Score",
                f"{dice:.4f}"
            )

        with m2:
            st.metric(
                "IoU Score",
                f"{iou:.4f}"
            )


# ============================================================
# TEST GALLERY
# ============================================================

elif page == "Test Gallery":

    st.header("🖼️ Test Image Gallery")

    st.write(
        f"Showing all {len(pairs)} test images."
    )

    for start in range(
        0,
        len(pairs),
        3
    ):

        cols = st.columns(3)

        for col_index in range(3):

            item_index = start + col_index

            if item_index >= len(pairs):
                break

            image_path, mask_path = (
                pairs[item_index]
            )

            image_np, probability, prediction = (
                predict_image(
                    str(image_path)
                )
            )

            ground_truth = (
                np.asarray(
                    Image.open(
                        mask_path
                    ).convert("L")
                ) / 255.0
            ) > 0.5

            dice = dice_score(
                prediction,
                ground_truth
            )

            iou = iou_score(
                prediction,
                ground_truth
            )

            with cols[col_index]:

                st.image(
                    prediction,
                    caption=image_path.stem,
                    use_container_width=True
                )

                st.write(
                    f"Dice: **{dice:.4f}**"
                )

                st.write(
                    f"IoU: **{iou:.4f}**"
                )

                if dice >= 0.8:

                    st.success(
                        "Strong segmentation"
                    )

                elif dice >= 0.6:

                    st.warning(
                        "Moderate segmentation"
                    )

                else:

                    st.error(
                        "Difficult case"
                    )

        st.markdown("---")