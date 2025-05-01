import streamlit as st
from detect import run
import tempfile
import os
import shutil

st.set_page_config(page_title="Object Detection App", layout="centered")
st.title("🧠 AI Object Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save image temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    st.info("Running object detection...")

    run(weights="weights/best.pt", source=temp_path, conf_thres=0.25, save_txt=False)

    # Get latest run folder
    result_folder = sorted(os.listdir("runs/detect"))[-1]
    result_img_path = os.path.join("runs/detect", result_folder, uploaded_file.name)

    st.image(result_img_path, caption="Detected Objects", use_column_width=True)

    # Cleanup
    shutil.rmtree(temp_dir)
