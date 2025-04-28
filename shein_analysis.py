"""
Shein Comments Analysis Module

This module provides functions to analyze Shein comments for integration into the main Streamlit app.
"""

import os
import plotly.graph_objects as go
from emotional_summary import extract_comments, analyze_with_claude, perform_local_analysis, create_emotional_summary_plot

def generate_mock_claude_analysis():
    """
    Generate mock Claude analysis when API key is not available.
    This provides a demo mode with reasonable results.
    """
    return {
        "summary": "The comments show a mix of positive and negative sentiments towards Shein products, particularly their swimwear. Many users express satisfaction with the quality and fit of Shein bikinis, while others caution about potential issues like color bleeding and durability. There is also some concern about the company's ethical practices, such as the use of child labor.",
        "top_adjectives": ["good", "cute", "cheeky"],
        "emotional_words": ["love", "recommend", "struggle", "careful", "caution", "concern"]
    }

def run_shein_comments_analysis():
    """
    Run the analysis of Shein comments and return the plotly figures.

    Returns:
        tuple: (summary_fig, emotion_fig, comment_count) - Plotly figures for summary and emotion distribution,
               plus the count of analyzed comments, or (None, None, 0) if analysis fails
    """
    try:
        # Extract comments
        comments = extract_comments()
        if not comments:
            print("No comments found in sample_comments.js")
            return None, None, 0

        # Get the comment count
        comment_count = len(comments)
        print(f"Extracted {comment_count} comments")

        # Run local analysis
        local_analysis = perform_local_analysis(comments)

        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")

        # Use Claude or fallback to mock analysis
        if api_key:
            # Run Claude analysis with actual API
            claude_analysis = analyze_with_claude(comments)
            print("Using Anthropic Claude for analysis")
        else:
            # Use mock analysis for demo purposes
            claude_analysis = generate_mock_claude_analysis()
            print("Using mock analysis (demo mode)")

        # Create visualizations
        summary_fig, emotion_fig = create_emotional_summary_plot(claude_analysis, local_analysis)

        return summary_fig, emotion_fig, comment_count

    except Exception as e:
        print(f"Error in Shein comments analysis: {e}")
        return None, None, 0

def get_shein_analysis_for_streamlit():
    """
    Function designed to be called from streamlit_app.py to display Shein analysis.

    Returns:
        dict: Analysis results with keys:
            - success: Boolean indicating if analysis was successful
            - summary_fig: Plotly figure for emotional summary
            - emotion_fig: Plotly figure for emotion distribution
            - comment_count: Number of comments analyzed
            - error: Error message if analysis failed
            - using_demo: Boolean indicating if using demo mode
    """
    try:
        # Check if we're using demo mode
        using_demo = os.getenv("ANTHROPIC_API_KEY") is None

        summary_fig, emotion_fig, comment_count = run_shein_comments_analysis()

        if summary_fig is None or emotion_fig is None:
            return {
                "success": False,
                "summary_fig": None,
                "emotion_fig": None,
                "comment_count": 0,
                "error": "Analysis failed to generate results. Check logs for details.",
                "using_demo": using_demo
            }

        return {
            "success": True,
            "summary_fig": summary_fig,
            "emotion_fig": emotion_fig,
            "comment_count": comment_count,
            "error": None,
            "using_demo": using_demo
        }

    except Exception as e:
        return {
            "success": False,
            "summary_fig": None,
            "emotion_fig": None,
            "comment_count": 0,
            "error": str(e),
            "using_demo": os.getenv("ANTHROPIC_API_KEY") is None
        }
