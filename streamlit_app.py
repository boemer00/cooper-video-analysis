import os
import sys
import logging
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.pipeline_assemblyai import analyze_with_assemblyai
from src.visualization.plotly_visualizer import create_timeline_plot, create_distribution_plot

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Cooper Video Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Sidebar: API key & options
st.sidebar.header("Cooper")
api_key = os.getenv("ASSEMBLYAI_API_KEY") or st.sidebar.text_input(
    "AssemblyAI API Key", type="password",
    help="Get your key at https://www.assemblyai.com/"
)

# Add API key verification display
if api_key:
    st.sidebar.success("✅ API Key found")
    # Log first/last 4 chars only for security
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    st.sidebar.info(f"API Key: {masked_key}")
else:
    st.sidebar.error("❌ API Key missing")

# Analysis options
st.sidebar.header("Analysis Options")
facial_sampling_rate = st.sidebar.slider(
    "Facial Analysis Sampling Rate (seconds)",
    min_value=1,
    max_value=5,
    value=1,
    help="Sample 1 frame every N seconds for facial emotion analysis. Higher values are faster but less precise."
)

debug = st.sidebar.checkbox("Enable Debug Mode")

# Main UI
st.title("Video Sentiment Analysis 📱")
st.markdown(
    "Analyze short-form video sentiment and emotions using AI."
)

# File upload within form
with st.form(key="upload_form", clear_on_submit=False):
    uploaded = st.file_uploader(
        "Select a video file:",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: mp4, mov, avi, mkv"
    )
    analyze_btn = st.form_submit_button("Analyze")

if not api_key:
    st.error("🔑 API key is required.")
    st.stop()

results = None  # initialize results

if uploaded and analyze_btn:
    # Save to temp file
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.read())
        video_path = tmp.name

        # Create output directory
        dirs = Path("./streamlit_output")
        dirs.mkdir(exist_ok=True)

        # Analysis
        with st.spinner("Analyzing..."):
            try:
                # Validate key
                if len(api_key) < 10:
                    raise ValueError("Invalid API key format.")

                # Show more detailed debugging info
                if debug:
                    st.info(f"Using API key: {masked_key}")
                    st.info(f"Analyzing video: {uploaded.name} ({uploaded.size/(1024**2):.2f} MB)")
                    st.info(f"Temporary file: {video_path}")
                    st.info(f"Output directory: {dirs}")
                    st.info(f"Facial sampling rate: {facial_sampling_rate} second(s)")

                results = analyze_with_assemblyai(
                    video_path, str(dirs), api_key=api_key,
                    facial_sampling_rate=facial_sampling_rate
                )
                st.success("✅ Analysis Complete!")

            except Exception as e:
                st.error(f"Error: {e}")
                logger.error("Analysis failed", exc_info=True)
                if debug:
                    st.exception(e)

    finally:
        # Clean up temp file if created
        try:
            if 'video_path' in locals():
                os.remove(video_path)
                if debug:
                    st.write(f"Removed temp file: {video_path}")
        except Exception as cleanup_e:
            logger.warning(f"Could not remove temp file: {cleanup_e}")

    # Display
    if results:
        # Create interactive Plotly visualizations
        st.subheader("Analysis Results")

        # Create and display Distribution Analysis plot
        distribution_fig = create_distribution_plot(results.timeline_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

        # Create and display Timeline Analysis plot
        timeline_fig = create_timeline_plot(results.timeline_data)
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Display the average scores in three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Text Sentiment Scores")
            for sentiment, score in results.text_scores.items():
                st.metric(
                    label=sentiment.capitalize(),
                    value=f"{score:.1%}"
                )

        with col2:
            st.subheader("Voice Emotion Scores")
            for emotion, score in results.audio_scores.items():
                st.metric(
                    label=emotion.capitalize(),
                    value=f"{score:.1%}"
                )

        with col3:
            st.subheader("Facial Emotion Scores")
            for emotion, score in results.facial_scores.items():
                st.metric(
                    label=emotion.capitalize(),
                    value=f"{score:.1%}"
                )

# Debug info
if debug:
    with st.expander("🔍 Debug Info"):
        st.write("Python:", sys.version)
        st.write("Working Dir:", os.getcwd())
        st.write("Environment Variables:", [k for k in os.environ.keys() if not k.startswith('_')])
        if uploaded:
            st.write({
                "Name": uploaded.name,
                "Size": f"{uploaded.size/(1024**2):.2f} MB",
                "Type": uploaded.type,
            })
