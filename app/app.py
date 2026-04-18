import streamlit as st
import sys
from pathlib import Path
import os

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import config
from system import run_system

# ─────────────────────────────────────────────
# UI TITLE
# ─────────────────────────────────────────────
st.set_page_config(page_title="Microstructure Analyzer", layout="centered")

st.title("🔬 Microstructure Analysis System")

st.write("Upload an image and select type")

# ─────────────────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ─────────────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────────────
mode = st.selectbox("Select Image Type", ["UHCS", "NEU"])

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────
if st.button("Run Analysis"):

    if uploaded_file is None:
        st.warning("Please upload an image first")
    else:
        # Save uploaded image temporarily
        temp_path = PROJECT_ROOT / "temp.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(str(temp_path), caption="Uploaded Image", use_column_width=True)

        st.write("Running analysis...")

        # Run system
        result = run_system(str(temp_path), mode=mode)

        st.success("Done!")

        # ─────────────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────────────
        if mode == "UHCS":
            st.subheader("Classification Result")
            
            st.subheader("Top Predictions")

            for i, pred in enumerate(result["top_predictions"], 1):
                st.write(f"{i}. {pred['class']} → {pred['confidence']:.4f}")

            # Show GradCAM
            gradcam_path = os.path.join(config.GRADCAM_DIR, os.path.basename(temp_path))
            if os.path.exists(gradcam_path):
                st.image(gradcam_path, caption="Grad-CAM")

        elif mode == "NEU":
            st.subheader("Detection + Severity")

            for defect in result:
                st.write(f"Defect: {defect['defect']}")
                st.write(f"Boxes: {defect['boxes']}")
                st.write(f"Avg Confidence: {defect['avg_conf']:.4f}")
                st.write(f"Severity: {defect['severity']}")
                st.write("---")

            # Show detection image
            pred_path = os.path.join(config.PREDICTIONS_DIR, os.path.basename(temp_path))
            if os.path.exists(pred_path):
                st.image(pred_path, caption="Detection Result")

        # Clean temp file
        os.remove(temp_path)