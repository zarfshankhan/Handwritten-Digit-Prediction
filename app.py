import streamlit as st
import torch
import numpy as np
import pandas as pd

from PIL import Image, UnidentifiedImageError

from models import CNNModel, ViTModel
from preprocessing import prepare_digit_image

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Handwritten Digit Prediction",
    page_icon="🔢",
    layout="wide"
)

# CUSTOM PAGE DESIGN

st.title("🔢 Handwritten Digit Prediction")

st.write(
    "AI-powered handwritten digit recognition "
    "using CNN and Vision Transformer."
)

st.markdown("""
Upload a handwritten digit image (0-9).

The application prepares the image into an 8×8
format and predicts the digit using two trained models.
""")

st.divider()

# DEVICE

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.caption(f"Running on: {DEVICE}")

# LOAD TRAINED MODELS

@st.cache_resource
def load_models():

    # Create CNN
    cnn = CNNModel()

    # Create ViT
    vit = ViTModel(
        patch_size=2,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        num_classes=10
    )

    # Load CNN checkpoint
    cnn_checkpoint = torch.load(
        "models/cnn_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    # Load ViT checkpoint
    vit_checkpoint = torch.load(
        "models/vit_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    # Restore weights
    cnn.load_state_dict(
        cnn_checkpoint["model_state_dict"]
    )

    vit.load_state_dict(
        vit_checkpoint["model_state_dict"]
    )

    # Move models to device
    cnn.to(DEVICE)
    vit.to(DEVICE)

    # Evaluation mode
    cnn.eval()
    vit.eval()

    return cnn, vit

# UPLOAD IMAGE

uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload one clear handwritten digit."
)

# PROCESS IMAGE AND PREDICT

if uploaded_file is not None:

    try:

        # Read original image
        uploaded_file.seek(0)

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # Prepare input image
        uploaded_file.seek(0)

        processed_image, input_tensor = (
            prepare_digit_image(uploaded_file)
        )

        
        # DISPLAY ORIGINAL AND PROCESSED IMAGE

        st.header("Image Processing")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.subheader("Prepared 8×8 Image")

            enlarged_image = processed_image.resize(
                (320, 320),
                Image.Resampling.NEAREST
            )

            st.image(
                enlarged_image,
                use_container_width=True
            )

        # DISPLAY 8x8 PIXEL VALUES

        st.subheader("8×8 Pixel Values")

        pixel_values = np.asarray(
            processed_image,
            dtype=np.float32
        ) / 255.0

        pixel_dataframe = pd.DataFrame(
            np.round(pixel_values, 2),
            index=[f"Row {i}" for i in range(8)],
            columns=[f"Col {i}" for i in range(8)]
        )

        st.dataframe(
            pixel_dataframe,
            use_container_width=True
        )

        
        # DISPLAY INPUT TENSOR INFORMATION (UPDATED)
        st.subheader("Model Input Information")
        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Original Height", f"{original_image.height} Px")

        with info_col2:
            st.metric("Original Width", f"{original_image.width} Px")

        with info_col3:
            st.metric("Model Input", "8 × 8 Px")

        st.write(
            "**Input Tensor Shape:**",
            str(input_tensor.shape)
        )

        st.write(
            "**Input Data Type:**",
            str(input_tensor.dtype)
        )

        st.write(
            "**Minimum Pixel Value:**",
            f"{input_tensor.min().item():.4f}"
        )

        st.write(
            "**Maximum Pixel Value:**",
            f"{input_tensor.max().item():.4f}"
        )

        st.info(
            "CNN input: [1, 1, 8, 8] | "
            "ViT input patches: [1, 16, 4]"
        )

        st.divider()

        # LOAD TRAINED MODELS

        with st.spinner("Loading trained models..."):

            cnn_model, vit_model = load_models()

        input_tensor = input_tensor.to(DEVICE)

        # MAKE PREDICTIONS

        with torch.no_grad():

            cnn_output = cnn_model(input_tensor)
            vit_output = vit_model(input_tensor)

            cnn_probabilities = torch.softmax(cnn_output, dim=1)
            vit_probabilities = torch.softmax(vit_output, dim=1)

            cnn_prediction = cnn_output.argmax(dim=1).item()
            vit_prediction = vit_output.argmax(dim=1).item()

            cnn_confidence = (cnn_probabilities[0, cnn_prediction].item() * 100)
            vit_confidence = (vit_probabilities[0, vit_prediction].item() * 100)

        # --- PREDICTION VERIFICATION CHECK ---
        # Agar dono models ki confidence 45% se kam hai, to matlab image sahi nahi hai
        if cnn_confidence < 45.0 and vit_confidence < 45.0:
            st.error(
                "⚠️ The model is not confident about this prediction. "
                "The uploaded image might not be a clear digit (0-9), or the image is too noisy. "
                "Please try uploading a clearer 8x8 handwritten digit."
            )
        else:
            # DISPLAY PREDICTIONS
            st.header("Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧠 CNN Model")
                st.metric("Predicted Digit", cnn_prediction)
                st.progress(cnn_confidence / 100)
                st.write(f"Confidence: {cnn_confidence:.2f}%")

            with col2:
                st.subheader("🔍 Vision Transformer")
                st.metric("Predicted Digit", vit_prediction)
                st.progress(vit_confidence / 100)
                st.write(f"Confidence: {vit_confidence:.2f}%")

            st.divider()

            # COMPARE PREDICTIONS
            st.subheader("Model Comparison")
            if cnn_prediction == vit_prediction:
                st.success(f"Both models predict digit {cnn_prediction}.")
            else:
                st.warning(
                    f"CNN predicts {cnn_prediction}, "
                    f"while ViT predicts {vit_prediction}."
                )

            # CLASS PROBABILITIES
            st.subheader("Class Probability Comparison")
            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:
                st.write("CNN Probabilities")
                for digit in range(10):
                    probability = cnn_probabilities[0, digit].item()
                    st.write(f"Digit {digit}: {probability * 100:.2f}%")

            with probability_col2:
                st.write("ViT Probabilities")
                for digit in range(10):
                    probability = vit_probabilities[0, digit].item()
                    st.write(f"Digit {digit}: {probability * 100:.2f}%")

    except (
        ValueError,
        UnidentifiedImageError,
        OSError
    ) as error:

        st.error(str(error))

    except FileNotFoundError:

        st.error(
            "Trained models not found. "
            "Please run python train.py first."
        )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )


else:

    st.info(
        "Please upload a PNG or JPEG image to start digit prediction."
    )


# FOOTER

st.divider()

st.caption(
    "Handwritten Digit Prediction | CNN + ViT"
)