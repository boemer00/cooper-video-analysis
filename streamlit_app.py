import os
import sys
import logging
import tempfile
from pathlib import Path
from PIL import Image
import base64
import time

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.pipeline_assemblyai import analyze_with_assemblyai
from plot_vizualizer import (
    create_distribution_plot,
    create_timeline_plots,
    create_emotion_trends_plot,
    create_emotional_contagion_heatmap,
    create_emotion_transition_network_plot,
)
# Import Shein comments analysis
from shein_analysis import get_shein_analysis_for_streamlit

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Cooper Video Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required NLTK resources (for TextBlob)
try:
    # Only import these when needed
    import nltk
    from textblob import TextBlob
    # Try a simple operation that requires the NLTK data
    TextBlob("Test").words
except Exception as e:
    if "MissingCorpusError" in str(e) or "resource" in str(e).lower():
        st.error("""
        ### Missing NLTK Resources

        Some required NLTK resources are missing. Please run the setup script:
        ```
        python setup_nltk.py
        ```

        Error details: {}
        """.format(str(e)))
        st.stop()
    else:
        # It's another kind of error, just log it
        logging.error(f"Error checking NLTK resources: {e}")

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
    .stFileUploader div[data-testid="stFileDropzone"] {
        background-color: white !important;
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
        background-color: #24231e !important;
        color: #24231e !important;
        font-weight: bold !important;
    }

    /* Make min/max values (1 and 5) have transparent backgrounds */
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww,
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww p,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] + div p,
    [data-testid="stSidebar"] [data-testid="stSlider"] div.st-emotion-cache-16j8nww div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div + div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider-container"] div {
        background-color: #24231e !important;
        color: white !important;
    }

    /* Remove the slider background completely */
    [data-testid="stSidebar"] [data-testid="stSlider"],
    [data-testid="stSidebar"] [data-testid="stSlider"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div {
        background-color: #24231e !important;
    }

    /* Fix slider track - the background part (unfilled) */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="progressbar"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background-color: #24231e !important;
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
        background-color: #24231e !important;
    }

    /* Extra overrides for Streamlit styling - more specific to ensure the rule is applied */
    [data-testid="stSidebar"] div.stSlider > div > div > div > div[style*="background-color"],
    [data-testid="stSidebar"] div.stSlider [style*="background-color"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] [style*="background-color: rgb"],
    [data-testid="stSidebar"] div[role="slider"] div[style*="background-color"] {
        background-color: #24231e !important;
    }

    /* Target all elements with inline background styling in the slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(0, 104, 201)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(49, 130, 206)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(47, 128, 237)"],
    [data-testid="stSidebar"] [data-testid="stSlider"] *[style*="background-color: rgb(79, 143, 247)"] {
        background-color: #ff7557 !important;
    }

    /* Target the slider track to change blue color */
    .stSlider [data-testid="stTrack"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Target the slider track's filled portion (blue by default) */
    .stSlider [data-testid="stTrack"] > div {
        background-color: #4B8BBE !important;
    }

    /* Target the slider track to change blue color - more specific selectors */
    [data-testid="stSidebar"] .stSlider [data-testid="stTrack"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Target the slider track's filled portion with more specific selectors */
    [data-testid="stSidebar"] .stSlider [data-testid="stTrack"] > div,
    [data-testid="stSidebar"] .stSlider div[role="progressbar"] > div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background-color: rgb"],
    [data-testid="stSidebar"] .stSlider div[style*="background"] {
        background-color: #4B8BBE !important;
    }

    /* Override all blue colors in the app */
    :root {
        --primary-color: #ff7557 !important;
    }

    /* Extremely aggressive selectors for the blue slider track */
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background-color"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="progressbar"] div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[class*="Track"] div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[class*="filled"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] div > div > div:first-child,
    [data-testid="stSidebar"] .stSlider [data-testid="stTrack"] div,
    [data-testid="stSidebar"] .stSlider div[style*="background-color: rgb"],
    [data-testid="stSidebar"] .st-emotion-cache-* div[style*="background-color"],
    div[style*="background-color: rgb(49, 51, 63)"],
    div[style*="background-color: rgb(0, 104, 201)"],
    div[style*="background-color: rgb(47, 128, 237)"] {
        background-color: #ff7557 !important;
    }

    /* Style the file uploader text */
    .stFileUploader div[data-testid="stFileDropzone"] p,
    .stFileUploader div[data-testid="stFileDropzone"] small,
    .stFileUploader div[data-testid="stFileDropzone"] svg,
    .stFileUploader div[data-testid="stFileDropzone"] span,
    .stFileUploader div[data-testid="stFileDropzone"] div,
    .stFileUploader div[data-testid="stFileDropzone"] div p {
        color: white !important;
        fill: #333333 !important;
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

# Check for Anthropic API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    st.sidebar.warning("⚠️ Anthropic API key not found. Shein analysis will be limited.")

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

# Add tabs for different types of analysis
tab1, tab2 = st.tabs(["Video Analysis", "Shein Comments Analysis"])

with tab1:
    st.markdown("Analyze short-form video sentiment and emotions using AI.")

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
                background-color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

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
                color: #333333 !important;
                fill: #333333 !important;
            }

            /* Background color for the dropzone */
            .stFileUploader div[data-testid="stFileDropzone"] {
                background-color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Fixed CSS for file uploader with ultra-specific selectors
        st.markdown(
            """
            <style>
            /* Super specific selectors for file uploader text */
            section[data-testid="stFileUploadDropzone"] div p,
            section[data-testid="stFileUploadDropzone"] div span,
            section[data-testid="stFileUploadDropzone"] small,
            div[data-testid="stFileDropzone"] p,
            div[data-testid="stFileDropzone"] span,
            div[data-testid="stFileDropzone"] small,
            [data-testid="stFileDropzoneInstructions"] p,
            [data-testid="stFileDropzoneInstructions"] span,
            [data-testid="stFileDropzoneInstructions"] small {
                color: #333333 !important;
                fill: #333333 !important;
            }

            /* Super specific selector for file uploader background */
            section[data-testid="stFileUploadDropzone"],
            [data-testid="stFileDropzone"] {
                background-color: white !important;
                border: 1px dashed #cccccc !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Select a video file:",
            type=["mp4"],
            help="Supported formats: mp4"
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

            # Display conversational analysis results
            if hasattr(results, 'conversational_insights') and results.conversational_insights:
                insights = results.conversational_insights

                st.markdown("#### Content Summary")
                st.markdown(f"**{insights.summary}**")

            # Create and display Distribution Analysis plot
            distribution_fig = create_distribution_plot(results.timeline_data)
            st.plotly_chart(distribution_fig, use_container_width=True)

            # --- Emotion Analysis Visualization ---

            # Emotion Transition Network
            st.subheader("Emotion Transition Network")
            transition_fig = create_emotion_transition_network_plot(
                comments=results.comments
            )
            st.plotly_chart(transition_fig, use_container_width=True)

with tab2:
    st.markdown("<h2 style='color: #ff7557; font-weight: bold;'>Shein Comments Emotional Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Analyze TikTok comments about Shein products to understand emotional reactions and sentiment.")

    # Add a brief explanation
    with st.expander("About this analysis"):
        st.markdown("""
        This analysis examines TikTok comments about Shein products (primarily swimwear) to:

        - Generate an AI-powered emotional summary of customer sentiments
        - Identify the top adjectives used in comments
        - Extract words related to emotions and feelings about the brand
        - Visualize the emotional distribution across comments

        The analysis uses Anthropic Claude for natural language understanding and NRCLex for emotion detection.
        """)

    # Create a cleaner button with center alignment
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("Run Shein Comments Analysis", type="primary", use_container_width=True)

    # If button is clicked, run the analysis
    if analyze_btn:
        # Show a progress bar
        progress_bar = st.progress(0)

        # Step 1: Extract comments
        with st.spinner("Extracting comments..."):
            progress_bar.progress(20)
            st.markdown("**Step 1/3:** Extracting comments from sample data...")
            time.sleep(0.5)  # Small delay for better UX

        # Step 2: Analyze comments
        with st.spinner("Analyzing with Claude AI..."):
            progress_bar.progress(50)
            st.markdown("**Step 2/3:** Generating emotional insights with Claude AI...")
            time.sleep(0.5)  # Small delay for better UX

        # Step 3: Create visualizations
        with st.spinner("Creating visualizations..."):
            progress_bar.progress(80)
            st.markdown("**Step 3/3:** Creating emotion visualizations...")

            # Get the analysis results
            analysis_results = get_shein_analysis_for_streamlit()
            progress_bar.progress(100)

        # Clear the progress indicators
        progress_placeholder = st.empty()

        # Display results
        if analysis_results["success"]:
            # Get comment count from analysis results
            comment_count = analysis_results["comment_count"]

            # Show success message with comment count
            st.success(f"Extracted {comment_count} comments.")

            # Show demo mode notice if applicable
            if analysis_results["using_demo"]:
                st.info("📝 Running in demo mode with sample data. For full functionality, add your Anthropic API key to the .env file.")

            # Add expandable section for sample comments
            with st.expander("View sample comments"):
                for i in range(min(10, comment_count)):
                    st.markdown(f"- Sample comment #{i+1}")
                if comment_count > 10:
                    st.markdown(f"*...and {comment_count - 10} more*")

            # Create a two-column layout for the emotional summary and distribution
            col1, col2 = st.columns(2)

            with col1:
                # Emotional Summary header - salmon color
                st.markdown("<h3 style='color: #ff7557;'>Emotional Summary</h3>", unsafe_allow_html=True)

                # Summary with bold label
                st.markdown("<b>Summary:</b> " + analysis_results["summary_fig"].layout.annotations[1].text.replace("<b>Summary:</b> ", ""), unsafe_allow_html=True)

                # Top Adjectives with bold label
                st.markdown("<b>Top Adjectives:</b> " + analysis_results["summary_fig"].layout.annotations[2].text.replace("<b>Top Adjectives:</b> ", ""), unsafe_allow_html=True)

                # Emotional Words with bold label
                st.markdown("<b>Emotional Words:</b> " + analysis_results["summary_fig"].layout.annotations[3].text.replace("<b>Emotional Words:</b> ", ""), unsafe_allow_html=True)

            with col2:
                # Emotion Distribution header - salmon color
                st.markdown("<h3 style='color: #ff7557;'>Emotion Distribution</h3>", unsafe_allow_html=True)

                # Display the emotion distribution plot
                st.plotly_chart(analysis_results["emotion_fig"], use_container_width=True)

            # Add a separation line
            st.markdown("---")

            # Add a download section
            st.subheader("Download Results")
            col1, col2 = st.columns(2)

            with col1:
                # Create a download button for a report (placeholder)
                st.download_button(
                    label="Download Summary (PDF)",
                    data=b"Placeholder for PDF report",  # This would be replaced with actual PDF data
                    file_name="shein_analysis_summary.pdf",
                    mime="application/pdf",
                    disabled=True  # Disabled for now since we don't have actual PDF generation
                )

            with col2:
                # Create a download button for raw data (placeholder)
                st.download_button(
                    label="Download Raw Data (CSV)",
                    data=b"Placeholder for CSV data",  # This would be replaced with actual CSV data
                    file_name="shein_analysis_data.csv",
                    mime="text/csv",
                    disabled=True  # Disabled for now since we don't have actual CSV generation
                )

        else:
            st.error(f"Analysis failed: {analysis_results['error']}")

            if not anthropic_api_key:
                st.warning("""
                No Anthropic API key found. To enable full analysis:
                1. Get an API key from https://www.anthropic.com/
                2. Add it to your .env file as ANTHROPIC_API_KEY=your_key_here
                """)
    else:
        # Show a placeholder with sample image when no analysis is running
        st.info("Click the button above to analyze Shein comments and generate insights.")

        # Placeholder for when no analysis is running
        placeholder_col1, placeholder_col2 = st.columns(2)
        with placeholder_col1:
            st.markdown("### Emotional Summary")
            st.markdown("Analysis will show sentiment summary, top adjectives, and emotional words about Shein.")

        with placeholder_col2:
            st.markdown("### Emotion Distribution")
            st.markdown("Analysis will show the distribution of emotions across all comments.")

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
