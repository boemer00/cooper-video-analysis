#!/usr/bin/env python3
"""
Cooper Video Analysis - Streamlit App (Cloud Deployment Version)

A simplified web interface for video analysis using AssemblyAI API.
This version is optimized for Streamlit Cloud deployment.
"""
import os
import sys
import tempfile
import logging
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Configure logging to capture detailed info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the AssemblyAI-specific analysis function
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
    st.sidebar.info("Cooper Video Analysis v1.0.1")

    # Sidebar for options
    st.sidebar.title("Analysis Options")

    # Get AssemblyAI API key - prioritize environment variable for cloud deployment
    api_key = os.getenv("ASSEMBLYAI_API_KEY")

    if not api_key:
        api_key = st.sidebar.text_input(
            "AssemblyAI API Key",
            type="password",
            help="Enter your AssemblyAI API key. You can get one at https://www.assemblyai.com/"
        )

        if not api_key:
            st.error("AssemblyAI API key is required. Please enter it in the sidebar.")
            st.stop()

    # Show debug info toggle
    show_debug = st.sidebar.checkbox("Show Debug Information", value=False)

    if show_debug:
        st.sidebar.info("Debug mode enabled - detailed information will be shown")
        st.sidebar.info(f"Python version: {sys.version}")
        st.sidebar.info(f"Current directory: {os.getcwd()}")

    # File uploader
    st.write("### Upload Video File")
    st.write("Upload a video file to analyze its sentiment and emotion.")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }

        if show_debug:
            st.write("File Details:", file_details)

        # Create output directory in a location Streamlit can write to
        output_dir = Path("./streamlit_output")
        try:
            os.makedirs(output_dir, exist_ok=True)
            if show_debug:
                st.write(f"Output directory created at: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            if show_debug:
                st.error(f"Error creating output directory: {str(e)}")
            # Fall back to temp directory
            output_dir = Path(tempfile.mkdtemp())
            if show_debug:
                st.write(f"Using temporary directory instead: {output_dir}")

        # Create temporary file for the uploaded video
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                # Write uploaded file to temporary file
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name

                if show_debug:
                    st.write(f"Temporary file created at: {video_path}")
        except Exception as e:
            logger.error(f"Error creating temporary file: {str(e)}")
            st.error(f"Error processing uploaded file: {str(e)}")
            st.stop()

        # Analysis button
        if st.button("Analyze Video"):
            # Show spinner during analysis
            with st.spinner("Analyzing video with AssemblyAI, please wait..."):
                try:
                    logger.info("Starting analysis with AssemblyAI")

                    # Test API key validity
                    if not api_key or len(api_key) < 10:
                        st.error("Invalid AssemblyAI API key. Please check your API key and try again.")
                        logger.error("Invalid API key format")
                        st.stop()

                    # Run the analysis with detailed logging
                    logger.info(f"Running analysis on video: {video_path}")
                    results = analyze_with_assemblyai(
                        video_path,
                        str(output_dir),
                        api_key=api_key
                    )
                    logger.info("Analysis completed successfully")

                    # Log the results for debugging
                    if show_debug:
                        st.write("Raw Text Scores:", results.text_scores)
                        st.write("Raw Audio Scores:", results.audio_scores)
                        logger.info(f"Analysis completed with text scores: {results.text_scores}")
                        logger.info(f"Analysis completed with audio scores: {results.audio_scores}")

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
        # Show instructions when no file is uploaded
        st.info("Please upload a video file to analyze using AssemblyAI.")
        st.write("This tool will:")
        st.write("1. Extract audio from your video")
        st.write("2. Analyze it for sentiment and emotion using AssemblyAI")
        st.write("3. Generate visualizations of the results")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error("Application error", exc_info=True)
