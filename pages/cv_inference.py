import streamlit as st
import torch
from pages.models import load_models
from torchvision import transforms
from pages.utils import Normalize, ToTensor, Resize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

# ✅ Set the page layout to wide (full width)
st.set_page_config(layout="wide")


@st.cache_resource
def get_models():
    return load_models()


MODELS = get_models()
# ✅ Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image, model_name):
    ratio = 0.125
    if model_name == 'SegFormer':
        ratio = 0.25
    transformer = transforms.Compose([
        Resize(ratio),
        Normalize(mean=[0.4663, 0.4657, 0.3188], std=[1, 1, 1]),
        ToTensor()
    ])
    return transformer(image)


def postprocess_output(input_img, output, model_name):
    if model_name == "SegFormer":
        output = output.logits
        probs = torch.sigmoid(output)
        output = F.interpolate(probs, size=input_img.shape[-2:], mode='bilinear', align_corners=True)
    output = output.squeeze().cpu().detach().numpy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0

    return output


def apply_model(image, model_name):
    if model_name in MODELS:
        model = MODELS[model_name].to(device)
        input_image = preprocess_image(image, model_name).to(device)
        with torch.no_grad():
            output = model(input_image)
        output_image = postprocess_output(input_image, output, model_name)
        output_image = cv2.resize(output_image, image.shape[::-1][-2:], interpolation=cv2.INTER_CUBIC)
        return output_image
    else:
        return image


def get_overlay_map(original_image, binary_map, alpha=0.4, color=(255, 0, 0)):
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 0] = binary_map * (color[0] / 255.0)  # Red channel
    colored_mask[:, :, 1] = binary_map * (color[1] / 255.0)  # Green channel
    colored_mask[:, :, 2] = binary_map * (color[2] / 255.0)  # Blue channel

    overlayed_image = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
    return overlayed_image


# Initialize session state variables
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = None
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None
if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = None

st.title("Computer Vision Model Inference Hub")
# Create an image upload area
uploaded_files = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_files is not None:
    st.session_state["uploaded_images"] = uploaded_files
    st.session_state["processed_images"] = None
# if uploaded_files is not None:
#     if st.button("Upload"):
#         st.session_state.uploaded_images = uploaded_files
#         st.session_state.processed_images = None
#         st.success("Images uploaded successfully!")

available_models = ["Please select a model"] + list(MODELS.keys())
selected_model_name = st.selectbox("Select a model", available_models)

if selected_model_name:
    st.session_state["selected_model"] = selected_model_name

# if st.button("Confirm Model"):
#     st.session_state.selected_model = selected_model_name
#     st.success(f"Model '{selected_model_name}' selected!")
if st.button("Run Model"):
    if st.session_state.get("uploaded_images") is None or uploaded_files is None:
        st.warning("⚠ No image uploaded. Please upload an image.")
        st.session_state["processed_images"] = None
    if st.session_state.get("selected_model") is None or st.session_state["selected_model"] not in MODELS:
        st.warning("⚠ No model selected. Please choose a model from the list.")
        st.session_state["processed_images"] = None
    if uploaded_files and st.session_state["selected_model"] in MODELS:
        with st.spinner("Processing image..."):
            image = np.array(Image.open(st.session_state["uploaded_images"]))
            processed_image = apply_model(image, st.session_state["selected_model"])
            st.session_state["processed_images"] = processed_image

        st.success("Processing complete!")

st.write("### Processed Image:")
if st.session_state.get("processed_images") is not None and st.session_state["uploaded_images"] is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(st.session_state["uploaded_images"])
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        binary_map = st.session_state["processed_images"] * 255
        binary_map = binary_map.astype(np.uint8)
        overlay_map = get_overlay_map(np.array(original_image), binary_map, alpha=0.4, color=(0,255,0))
        st.image(overlay_map, use_container_width=True)
        st.markdown(
            """
            <div style="display: flex; align-items: center; padding: 10px; border-radius: 5px; background-color: #f0f0f0;">
                <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0); margin-right: 10px;"></div>
                <span style="font-size: 16px;">Green Areas: Detected Phragmites</span>
            </div>
            """,
            unsafe_allow_html=True
        )
