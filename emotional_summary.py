"""
Emotional Summary Generator

This script analyzes TikTok comments from sample_comments.js to:
1. Generate an emotional summary using Anthropic Claude
2. Identify top adjectives and emotional words related to Shein
3. Create visualizations of the emotional content
"""

import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from dotenv import load_dotenv
from anthropic import Anthropic
import re
from collections import Counter
from nrclex import NRCLex
from typing import List, Dict, Tuple

# Load environment variables for API keys
load_dotenv()

def extract_comments() -> List[str]:
    """Extract comments from the sample_comments.js file."""
    # Path to the sample comments file
    sample_path = "src/visualization/sample_comments.js"

    try:
        with open(sample_path, 'r') as f:
            content = f.read()

        # Find the raw comments array in the file
        # This handles the JavaScript format in the file
        start_idx = content.find('const rawComments = [')
        end_idx = content.rfind('];')

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not locate comments array in file")

        # Extract just the JSON array part
        json_str = content[start_idx + len('const rawComments = '):end_idx + 1]

        # Parse JSON
        comments_data = json.loads(json_str)

        # Extract just the text content
        return [comment.get("text", "") for comment in comments_data if comment.get("text")]

    except Exception as e:
        print(f"Error extracting comments: {e}")
        return []

def analyze_with_claude(comments: List[str]) -> Dict:
    """
    Use Anthropic Claude to analyze the emotional content of comments.

    Returns a dictionary with:
    - summary: Overall emotional summary
    - top_adjectives: Top 3 adjectives used
    - emotional_words: Top words related to emotions about Shein
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = Anthropic(api_key=api_key)

    # Prepare the prompt for Claude
    all_comments = "\n".join([f"- {comment}" for comment in comments])
    prompt = f"""
    Here are comments from TikTok about Shein products (primarily swimwear):

    {all_comments}

    Please analyze these comments and provide:
    1. A concise emotional summary of how people feel about Shein products (3-4 sentences)
    2. The top 3 most frequently used adjectives in these comments
    3. The top words related to emotions/feelings about the Shein brand

    Format your response as a JSON object with the keys: "summary", "top_adjectives", and "emotional_words".
    """

    # Call Claude API
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        system="You analyze social media comments and extract emotional insights in JSON format.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the JSON from Claude's response
    try:
        # Find JSON in the response
        response_text = message.content[0].text
        # Look for JSON pattern
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find a JSON object directly
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

        analysis = json.loads(json_str)
        return analysis
    except Exception as e:
        print(f"Error parsing Claude response: {e}")
        print(f"Raw response: {message.content}")
        # Return a basic structure if parsing fails
        return {
            "summary": "Failed to parse the emotional analysis.",
            "top_adjectives": [],
            "emotional_words": []
        }

def perform_local_analysis(comments: List[str]) -> Dict:
    """
    Perform a local analysis of comments as a backup and for validation.
    Uses NRCLex for emotion extraction and basic NLP for adjective extraction.
    """
    # Combine all comments into one text for analysis
    all_text = " ".join(comments)

    # Use NRCLex for emotion analysis
    emotion_analysis = NRCLex(all_text)
    emotion_counts = emotion_analysis.affect_frequencies

    # Basic adjective extraction based on common patterns
    # This is a simplified approach without proper POS tagging
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Simple list of common adjectives that might appear in these comments
    common_adjectives = ["good", "great", "cute", "small", "medium", "large",
                         "nice", "bad", "beautiful", "pretty", "comfortable",
                         "cheap", "expensive", "high", "low", "quality"]

    adjective_counts = Counter([word for word in words if word in common_adjectives])
    top_adjectives = [adj for adj, _ in adjective_counts.most_common(3)]

    # Combine results
    return {
        "emotion_frequencies": emotion_counts,
        "top_adjectives": top_adjectives
    }

def create_emotional_summary_plot(claude_analysis: Dict, local_analysis: Dict) -> Tuple[go.Figure, go.Figure]:
    """
    Create two plots:
    1. A summary card with Claude's analysis (modified to use more direct text extraction)
    2. An emotion distribution chart

    The styling matches the visual layout shown in the example image.
    """
    # Create summary card with improved styling
    summary_fig = go.Figure()

    # Set title using annotation with improved styling
    summary_fig.add_annotation(
        text="Emotional Summary",
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            size=24,
            color="#ff7557"  # Salmon color from the image
        ),
        xanchor="center",
        yanchor="bottom"
    )

    # Add the summary text with better formatting
    summary_fig.add_annotation(
        text=f"<b>Summary:</b> {claude_analysis['summary']}",
        x=0.02,
        y=0.9,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            size=14,
            color="#333333"
        ),
        align="left",
        xanchor="left",
        yanchor="top"
    )

    # Add the top adjectives with better formatting
    summary_fig.add_annotation(
        text=f"<b>Top Adjectives:</b> {', '.join(claude_analysis['top_adjectives'])}",
        x=0.02,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            size=14,
            color="#333333"
        ),
        align="left",
        xanchor="left",
        yanchor="top"
    )

    # Add the emotional words with better formatting
    summary_fig.add_annotation(
        text=f"<b>Emotional Words:</b> {', '.join(claude_analysis['emotional_words'])}",
        x=0.02,
        y=0.3,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            size=14,
            color="#333333"
        ),
        align="left",
        xanchor="left",
        yanchor="top"
    )

    # Update layout for better appearance
    summary_fig.update_layout(
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1]
        )
    )

    # Create emotion distribution chart with improved styling
    emotions = list(local_analysis["emotion_frequencies"].keys())
    scores = list(local_analysis["emotion_frequencies"].values())

    # Sort by score
    sorted_data = sorted(zip(emotions, scores), key=lambda x: x[1], reverse=True)
    emotions = [item[0].capitalize() for item in sorted_data]
    scores = [item[1] for item in sorted_data]

    # Color map for emotions with pastel colors similar to image
    emotion_colors = {
        'Positive': '#2ecc71',    # Green
        'Trust': '#3498db',       # Blue
        'Anticipation': '#1abc9c', # Teal
        'Negative': '#e74c3c',    # Red
        'Joy': '#f1c40f',         # Yellow
        'Sadness': '#34495e',     # Dark blue
        'Surprise': '#9b59b6',    # Purple
        'Fear': '#f39c12',        # Orange
        'Anger': '#e67e22',       # Dark orange
        'Disgust': '#7f8c8d',     # Gray
    }

    # Get colors for the sorted emotions
    colors = [emotion_colors.get(emotion, '#95a5a6') for emotion in emotions]

    # Format percentages
    pct_texts = [f"{score*100:.1f}%" for score in scores]

    # Create the bar chart without title (title will be added by Streamlit)
    emotion_fig = go.Figure()

    # Add the bar chart
    emotion_fig.add_trace(
        go.Bar(
            x=emotions,
            y=scores,
            marker_color=colors,
            text=pct_texts,
            textposition='outside',
            hoverinfo='text',
            hovertext=[f"{emotion}: {pct}" for emotion, pct in zip(emotions, pct_texts)],
            width=0.6  # Thinner bars like in the image
        )
    )

    # Update layout for better appearance
    emotion_fig.update_layout(
        title=dict(
            text="Emotion Distribution in Comments",
            font=dict(
                family="Plus Jakarta Sans, sans-serif",
                size=16,
                color="#333333"
            ),
            y=0.9,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        xaxis=dict(
            title=None,
            tickangle=45,
            tickfont=dict(
                family="Plus Jakarta Sans, sans-serif",
                size=12,
                color="#333333"
            )
        ),
        yaxis=dict(
            title="Frequency",
            titlefont=dict(
                family="Plus Jakarta Sans, sans-serif",
                size=14,
                color="#333333"
            ),
            tickfont=dict(
                family="Plus Jakarta Sans, sans-serif",
                size=12,
                color="#333333"
            ),
            tickformat=".1%",
            range=[0, max(scores) * 1.2]  # Give some space for the text
        ),
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=20, l=50, r=20)
    )

    # Darker gridlines
    emotion_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return summary_fig, emotion_fig

def main():
    """Run the complete analysis and display results."""
    # Extract comments from the sample file
    print("Extracting comments...")
    comments = extract_comments()

    if not comments:
        print("No comments found. Please check the sample_comments.js file.")
        return

    print(f"Extracted {len(comments)} comments.")

    # Run local analysis first
    print("Performing local analysis...")
    local_analysis = perform_local_analysis(comments)

    try:
        # Analyze with Claude
        print("Analyzing with Claude...")
        claude_analysis = analyze_with_claude(comments)

        # Create visualizations
        print("Creating visualizations...")
        summary_fig, emotion_fig = create_emotional_summary_plot(claude_analysis, local_analysis)

        # Save the figures
        summary_fig.write_html("emotional_summary_card.html")
        emotion_fig.write_html("emotion_distribution.html")

        print("Analysis complete! Results saved to 'emotional_summary_card.html' and 'emotion_distribution.html'")

        # For streamlit integration, you can return the figures
        return summary_fig, emotion_fig

    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
