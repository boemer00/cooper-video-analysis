"""
Streamlit app for Comments Emotional Analysis

Run with:
    streamlit run streamlit_emotional_summary.py
"""

import streamlit as st
import os
from emotional_summary import extract_comments, analyze_with_claude, perform_local_analysis, create_emotional_summary_plot
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Comments Emotional Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main .stButton button,
    .main .stForm [data-testid="stFormSubmitButton"] button {
        background-color: #ff7557 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }

    h1, h2, h3 {
        color: #ff7557 !important;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Apply the styling
load_css()

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("⚠️ Anthropic API key not found. Please add it to your .env file as ANTHROPIC_API_KEY")

# Title and description
st.markdown("Analyze comments to understand emotional reactions and sentiment.")

# Sidebar
st.sidebar.image("static/cooper_logo.png", width=200)
st.sidebar.markdown("## Analysis Options")

# Main content
with st.container():
    # Run analysis button
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Extracting comments..."):
            comments = extract_comments()

            if not comments:
                st.error("No comments found. Please check the sample_comments.js file.")
            else:
                st.success(f"Extracted {len(comments)} comments.")

                # Show a sample of comments
                with st.expander("View sample comments"):
                    for i, comment in enumerate(comments[:10]):
                        st.markdown(f"- {comment}")
                    if len(comments) > 10:
                        st.markdown(f"*...and {len(comments) - 10} more*")

                # Run local analysis first
                with st.spinner("Performing local analysis..."):
                    local_analysis = perform_local_analysis(comments)

                # Analyze with Claude
                with st.spinner("Generating emotional summary with Claude AI..."):
                    try:
                        claude_analysis = analyze_with_claude(comments)

                        # Create visualizations
                        summary_fig, emotion_fig = create_emotional_summary_plot(claude_analysis, local_analysis)

                        # Display results in two columns
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Emotional Summary")
                            st.markdown(f"**Summary:** {claude_analysis['summary']}")
                            st.markdown(f"**Top Adjectives:** {', '.join(claude_analysis['top_adjectives'])}")
                            st.markdown(f"**Emotional Words:** {', '.join(claude_analysis['emotional_words'])}")

                        with col2:
                            st.subheader("Emotion Distribution")
                            st.plotly_chart(emotion_fig, use_container_width=True)

                        # Full width for summary visual
                        st.subheader("Visual Summary")
                        st.plotly_chart(summary_fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.warning("Showing only local analysis results.")

                        # Display local analysis results
                        st.subheader("Local Emotion Analysis")
                        st.json(local_analysis)

# Footer
st.markdown("---")
st.markdown("Cooper Video Analysis | Shein Comments Emotion Analysis")
