#!/usr/bin/env python3
"""
Cooper Video Analysis - Streamlit App

A web interface for the Cooper Video Analysis tool that analyzes videos for sentiment and emotion.
"""
import os
import sys
import tempfile
import logging
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import analysis functions
from src.pipeline import analyze
from src.pipeline_assemblyai import analyze_with_assemblyai

# Set page configuration
st.set_page_config(
    page_title="Cooper Video Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main entry point for the Streamlit application."""

    # Header
    st.title("Cooper Video Analysis")
    st.markdown("### Video Sentiment and Emotion Analysis Tool")

    # Version info
    st.sidebar.info("Cooper Video Analysis v1.0.0")

    # Sidebar for options
    st.sidebar.title("Analysis Options")

    # Analysis model selection
    analysis_model = st.sidebar.radio(
        "Select Analysis Model",
        ["Standard (Wav2Vec2)", "AssemblyAI"],
        index=0,
    )

    # Check for AssemblyAI API key if that model is selected
    api_key = None
    if analysis_model == "AssemblyAI":
        api_key = os.getenv("ASSEMBLYAI_API_KEY")

        if not api_key:
            api_key = st.sidebar.text_input(
                "AssemblyAI API Key",
                type="password",
                help="Enter your AssemblyAI API key. You can get one at https://www.assemblyai.com/"
            )

            if not api_key:
                st.sidebar.error("AssemblyAI API key is required for this model.")

    # Output directory selection
    output_name = st.sidebar.text_input(
        "Output Folder Name",
        value="streamlit_output",
        help="Name of the folder where analysis results will be saved"
    )

    # Show debug info toggle
    show_debug = st.sidebar.checkbox("Show Debug Information", value=False)

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }

        if show_debug:
            st.write("File Details:", file_details)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            # Write uploaded file to temporary file
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

            if show_debug:
                st.write(f"Temporary file created at: {video_path}")

        # Create output directory
        output_dir = Path(output_name)
        os.makedirs(output_dir, exist_ok=True)

        # Analysis button
        if st.button("Analyze Video"):
            # Show spinner during analysis
            with st.spinner("Analyzing video, please wait..."):
                try:
                    if analysis_model == "Standard (Wav2Vec2)":
                        # Use standard model
                        st.info("Using Standard (Wav2Vec2) model for analysis...")
                        logger.info("Starting analysis with Standard model")
                        results = analyze(video_path, str(output_dir))
                    else:
                        # Use AssemblyAI model
                        st.info("Using AssemblyAI model for analysis...")
                        logger.info("Starting analysis with AssemblyAI model")
                        results = analyze_with_assemblyai(
                            video_path,
                            str(output_dir),
                            api_key=api_key
                        )

                    # Log the results for debugging
                    if show_debug:
                        st.write("Raw Results:", results)
                        logger.info(f"Analysis completed with results: {results}")

                    # Success message
                    st.success("Analysis complete!")

                    # Display results in 2 columns
                    col1, col2 = st.columns(2)

                    # Text Sentiment Results
                    with col1:
                        st.subheader("Text Sentiment")
                        for sentiment, score in results.text_scores.items():
                            st.metric(label=sentiment.capitalize(), value=f"{score:.2%}")

                    # Audio Emotion Results
                    with col2:
                        st.subheader("Audio Emotion")
                        for emotion, score in results.audio_scores.items():
                            st.metric(label=emotion.capitalize(), value=f"{score:.2%}")

                    # Show plots
                    st.subheader("Analysis Plots")
                    timeline_path = os.path.join(output_dir, "timeline_plot.png")
                    dist_path = os.path.join(output_dir, "distribution_plot.png")

                    if os.path.exists(timeline_path) and os.path.exists(dist_path):
                        plot_col1, plot_col2 = st.columns(2)

                        with plot_col1:
                            st.image(timeline_path, caption="Timeline Analysis", use_column_width=True)

                        with plot_col2:
                            st.image(dist_path, caption="Distribution Analysis", use_column_width=True)
                    else:
                        st.warning("Plot files were not generated. Check logs for details.")
                        if show_debug:
                            st.write(f"Looking for timeline plot at: {timeline_path}")
                            st.write(f"Looking for distribution plot at: {dist_path}")

                    # Show download links
                    st.subheader("Download Results")
                    st.info(f"Results saved to {output_dir} folder")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}", exc_info=True)
                    if show_debug:
                        st.exception(e)

                # Clean up the temporary file
                try:
                    os.unlink(video_path)
                    if show_debug:
                        st.write(f"Temporary file {video_path} removed")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
    else:
        # Show sample image and instructions when no file is uploaded
        st.info("Please upload a video file to analyze.")
        try:
            # Try to load from local path first
            image_path = "static/assets/video_upload.png"
            if os.path.exists(image_path):
                st.image(image_path, width=400)
            else:
                # Fall back to remote URL if local file is not available
                st.image("https://img.freepik.com/free-vector/video-upload-concept-illustration_114360-4702.jpg", width=400)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            # No need to show this error to the user

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error("Application error", exc_info=True)
