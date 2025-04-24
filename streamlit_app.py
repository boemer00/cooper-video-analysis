import os
import sys
import logging
import tempfile
from pathlib import Path
from PIL import Image
import base64

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.pipeline_assemblyai import analyze_with_assemblyai
from src.visualization.plotly_visualizer import create_distribution_plot, create_timeline_plots

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Cooper Video Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #24231E;
    }

    /* Target ALL possible text elements in the sidebar to be white */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] [class*="css"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }

    /* Custom header color for Analysis Options */
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* MAIN CONTENT STYLING */
    /* Ensure main content text is black */
    [data-testid="stAppViewContainer"] *:not([data-testid="stSidebar"] *) {
        color: black !important;
    }

    /* Make sure form text is also black */
    .main .stButton,
    .main .stTextInput,
    .main .stFileUploader {
        color: black !important;
    }

    /* Force text color for data display elements */
    .main .stDataFrame,
    .main .stTable,
    .main .element-container {
        color: black !important;
    }

    /* Target text elements directly */
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: black !important;
    }

    /* Custom title color */
    [data-testid="stAppViewContainer"] > section:first-of-type div[data-testid="stVerticalBlock"] > div:first-child h1 {
        color: #ff7557 !important;
        font-weight: bold !important;
    }

    /* Specific rule for our custom title */
    #custom-title h1 {
        color: #ff7557 !important;
    }

    /* Style the Analyze button */
    .stButton button,
    .stForm [data-testid="stFormSubmitButton"] button,
    button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background-color: #ff7557 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }

    /* Override specific button classes */
    button.st-emotion-cache-19rxjzo,
    button.st-emotion-cache-1gulkj5 {
        background-color: #ff7557 !important;
    }

    /* Style the file uploader box text - more specific selectors */
    [data-testid="stFileDropzone"] div,
    [data-testid="stFileDropzone"] p,
    [data-testid="stFileDropzone"] span,
    [data-testid="stFileDropzone"] small,
    [data-testid="stFileDropzone"] svg,
    [data-testid="stFileDropzone"] path,
    [data-testid="stFileDropzoneInstructions"] > div,
    [data-testid="stFileDropzoneInstructions"] > div > div,
    [data-testid="stFileDropzoneInstructions"] > div > div > p {
        color: white !important;
        fill: white !important;
    }

    /* Make sure the file drop zone has a dark background */
    .stFileUploader [data-testid="stFileDropzone"] {
        background-color: #2b2b2b !important;
    }

    /* Add custom slider styling */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"],
    [data-testid="stSidebar"] [data-testid="stSlider"] span[data-baseweb="thumb"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="thumb"] {
        background-color: #ff7557 !important;
        border-color: #ff7557 !important;
    }

    /* Target the value display (number above the slider) - transparent background, text color only */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] div,
    [data-testid="stSidebar"] div.st-emotion-cache-5rimss > div,
    [data-testid="stSidebar"] div.st-emotion-cache-5rimss,
    [data-testid="stSidebar"] div.st-emotion-cache-5rimss > div > p,
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
        background-color: transparent !important;
        color: #ff7557 !important;
        font-weight: bold !important;
    }

    /* Make min/max values (1 and 5) have transparent backgrounds */
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww,
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww p,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] + div p,
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div + div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider-container"] div {
        background-color: transparent !important;
        color: white !important;
    }

    /* Remove the slider background completely */
    [data-testid="stSidebar"] [data-testid="stSlider"],
    [data-testid="stSidebar"] [data-testid="stSlider"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div {
        background-color: transparent !important;
    }

    /* Fix slider track - the background part (unfilled) */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="progressbar"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Fix slider track - the filled part ONLY */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="progressbar"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div,
    [data-testid="stSidebar"] .st-emotion-cache-1g805mo,
    [data-testid="stSidebar"] div.st-emotion-cache-1qxepGa,
    [data-testid="stSidebar"] div.st-emotion-cache-1g6s2x0,
    [data-testid="stSidebar"] div.css-1g6s2x0,
    [data-testid="stSidebar"] div.stSlider > div > div > div:first-child,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div div div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBar"] div,
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTrack"] div {
        background-color: #ff7557 !important;
    }

    /* Extra overrides for Streamlit styling - more specific to ensure the rule is applied */
    [data-testid="stSidebar"] div.stSlider > div > div > div > div[style*="background-color"],
    [data-testid="stSidebar"] div.stSlider [style*="background-color"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] [style*="background-color: rgb"],
    [data-testid="stSidebar"] div[role="slider"] div[style*="background-color"] {
        background-color: #ff7557 !important;
    }

    /* Target all elements with inline background styling in the slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(0, 104, 201)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(49, 130, 206)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(47, 128, 237)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(79, 143, 247)"] {
        background-color: #ff7557 !important;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Function to load and display the logo
def display_logo():
    # Use the Cooper logo from static directory
    logo_path = os.path.join("static", "cooper_logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=200)
    else:
        st.sidebar.error("Logo file not found")

# Apply the styling
load_css()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Sidebar: Logo and API key
display_logo()  # Replace "Cooper" header with logo
api_key = os.getenv("ASSEMBLYAI_API_KEY") or st.sidebar.text_input(
    "AssemblyAI API Key", type="password",
    help="Get your key at https://www.assemblyai.com/"
)

# Remove API key verification display
if not api_key:
    st.sidebar.error("❌ API Key missing")
    st.error("🔑 API key is required.")
    st.stop()

# Analysis options
st.sidebar.markdown('<div style="color: #ff7557; font-size: 20px; font-weight: bold; margin-bottom: 10px;">Analysis Options</div>', unsafe_allow_html=True)

facial_sampling_rate = st.sidebar.slider(
    "Facial Analysis Sampling Rate (seconds)",
    min_value=1,
    max_value=5,
    value=1,
    help="Sample 1 frame every N seconds for facial emotion analysis. Higher values are faster but less precise."
)

debug = st.sidebar.checkbox("Enable Debug Mode")

# Main UI
st.markdown('<div id="custom-title"><h1 style="color: #ff7557 !important; font-weight: bold; font-size: 2.5rem;">Video Sentiment Analysis 📱</h1></div>', unsafe_allow_html=True)
st.markdown(
    "Analyze short-form video sentiment and emotions using AI."
)

# File upload within form
with st.form(key="upload_form", clear_on_submit=False):
    # Add custom CSS for the dropzone
    st.markdown(
        """
        <style>
        /* Style the file uploader text */
        .stFileUploader div[data-testid="stFileDropzone"] p,
        .stFileUploader div[data-testid="stFileDropzone"] small,
        .stFileUploader div[data-testid="stFileDropzone"] svg,
        .stFileUploader div[data-testid="stFileDropzone"] span,
        .stFileUploader div[data-testid="stFileDropzone"] div,
        .stFileUploader div[data-testid="stFileDropzone"] div p {
            color: white !important;
            fill: white !important;
        }

        /* Background color for the dropzone */
        .stFileUploader div[data-testid="stFileDropzone"] {
            background-color: #2b2b2b !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Select a video file:",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: mp4, mov, avi, mkv"
    )

    # Custom styled submit button with HTML/CSS
    st.markdown(
        f"""
        <style>
        div[data-testid="stFormSubmitButton"] > button {{
            background-color: #ff7557 !important;
            color: white !important;
            border: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    analyze_btn = st.form_submit_button(
        "Analyze",
        use_container_width=False
    )

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

                # Create masked key for debug display only
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"

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

        # Create and display Timeline Analysis plots
        text_fig, voice_fig, facial_fig = create_timeline_plots(results.timeline_data)

        # Display each timeline figure with its own legend
        st.plotly_chart(text_fig, use_container_width=True)
        st.plotly_chart(voice_fig, use_container_width=True)

        # Only display facial emotion if we have data
        if facial_fig:
            st.plotly_chart(facial_fig, use_container_width=True)

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
