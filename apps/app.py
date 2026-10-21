"""
AeroBOS: Streamlit Video Processing Web Application
Module: apps/app.py

Features:
- st.file_uploader for MP4/AVI/MOV video ingestion.
- 'Process Video' execution button.
- Invokes real OpenCV BOSCore and PlumeAnalyzer.
- Side-by-side st.video playback of the raw upload versus the processed output.
"""

from __future__ import annotations

import os
import sys
import tempfile
import streamlit as st

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.process_video import process_video

st.set_page_config(
    page_title="AeroBOS: Optical Gas Imaging",
    page_icon="💨",
    layout="wide",
)

st.title("💨 AeroBOS: Real-Time Optical Gas Imaging")
st.markdown(
    "**Background-Oriented Schlieren (BOS) Offline Video Pipeline**  \n"
    "*OpenCV 5 DIS Optical Flow + 2D Poisson FFT Inversion + Adaptive Plume Segmentation*"
)

# Sidebar Configuration
st.sidebar.header("⚙️ Processing Parameters")
sigma_thresh = st.sidebar.slider("Adaptive Sigma Threshold (σ)", min_value=1.0, max_value=5.0, value=2.5, step=0.25)
min_area = st.sidebar.slider("Minimum Plume Area (pixels)", min_value=50, max_value=1000, value=150, step=25)
ref_mode = st.sidebar.selectbox("Reference Mode", options=["static", "running"], index=0, help="Static = compare to baseline frame 0; Running = compare to previous frame")

# Video Ingestion
st.subheader("1. Video Ingestion")
uploaded_file = st.file_uploader("Upload an MP4 video file", type=["mp4", "mov", "avi"])

# Quick benchmark fallback if available
sample_video_path = "sample_test_video.mp4"
use_sample = False
if not uploaded_file and os.path.exists(sample_video_path):
    use_sample = st.checkbox(f"Use benchmark test video ({sample_video_path})", value=True)

# Process Button
process_btn = st.button("🚀 Process Video", type="primary")

if process_btn:
    input_file_path = None

    if uploaded_file is not None:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        input_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif use_sample and os.path.exists(sample_video_path):
        input_file_path = sample_video_path

    if input_file_path is None:
        st.warning("Please upload a video file (.mp4) or select the benchmark sample.")
    else:
        output_file_path = os.path.splitext(input_file_path)[0] + "_annotated.mp4"

        with st.spinner("Processing video frames through AeroBOS (DIS Flow + Poisson FFT)..."):
            try:
                process_video(
                    input_path=input_file_path,
                    output_path=output_file_path,
                    sigma_threshold=sigma_thresh,
                    min_area_px=min_area,
                    reference_mode=ref_mode,
                )
                st.success("✅ AeroBOS Video Processing Complete!")

                # Side-by-side video playback
                st.subheader("2. Comparative Video Playback")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📹 Raw Uploaded Video (Invisible Plume)")
                    st.video(input_file_path)

                with col2:
                    st.markdown("#### 🎯 AeroBOS Annotated Video (TURBO Schlieren HUD)")
                    st.video(output_file_path)

                # Download button for annotated output
                if os.path.exists(output_file_path):
                    with open(output_file_path, "rb") as vid_file:
                        st.download_button(
                            label="⬇️ Download Annotated Video",
                            data=vid_file,
                            file_name=os.path.basename(output_file_path),
                            mime="video/mp4",
                        )

            except Exception as e:
                st.error(f"Error during video processing: {e}")
