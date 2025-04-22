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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: Inject style overrides for light theme main area with dark sidebar
st.markdown("""
    <style>
        /* Light theme for main content */
        .main {background-color: white; color: #333333;}

        /* Dark theme for sidebar */
        .css-6qob1r {background-color: #0e1117; color: white;}
        .css-1d391kg {color: white;}
        .st-bq {background-color: #1e2530;}

        /* Make sure text inputs in sidebar remain visible */
        .css-1qrvfrg {background-color: #2c3e50; color: white; border-color: #4a5568;}

        /* Other styling adjustments */
        h1, h2, h3 {color: #333333;}
        .stAlert {border-color: #0e1117;}
    </style>
""", unsafe_allow_html=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Sidebar: API key & options
st.sidebar.header("🔧 Settings")
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

debug = st.sidebar.checkbox("Enable Debug Mode")

# Main UI
st.title("Cooper Video Analysis 🎬")
st.markdown(
    "Analyze video sentiment and emotions using AssemblyAI."
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
    st.error("🔑 AssemblyAI API key is required.")
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

                results = analyze_with_assemblyai(
                    video_path, str(dirs), api_key=api_key
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

        # Create and display Timeline Analysis plot
        timeline_fig = create_timeline_plot(results.timeline_data)
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Create and display Distribution Analysis plot
        distribution_fig = create_distribution_plot(results.timeline_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

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
