import streamlit as st
import requests
from PIL import Image
import os
from io import BytesIO

# API configuration
API_URL = "http://127.0.0.1:8000/predict"
RESULTS_URL = "http://127.0.0.1:8000/results"

st.set_page_config(page_title="Aerial Object Detector", layout="wide")
st.title("🚁 Aerial Image Object Detection")
st.markdown("Upload an aerial image to detect objects")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("Detection Results")
        with st.spinner("Processing image..."):
            try:
                # Send to API
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result_filename = response.json()["result_path"]
                    image_url = f"{RESULTS_URL}/{result_filename}"

                    # Get image from backend
                    result_response = requests.get(image_url, stream=True)

                    if result_response.status_code == 200:
                        result_image = Image.open(BytesIO(result_response.content))
                        st.image(result_image, use_container_width=True)
                    else:
                        st.error("Failed to retrieve the result image from server.")
                else:
                    st.error(f"Error processing image: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
