"""
Import visualization functions from src/visualization/plotly_visualizer.py
This file exists to maintain compatibility with imports in streamlit_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

# Import the existing functions
from src.visualization.plotly_visualizer import (
    create_emotion_trends_plot,
    create_emotional_contagion_heatmap,
    create_emotion_transition_network_plot,
)

@dataclass
class TimelineData:
    """Data class for storing timeline analysis results."""
    timestamps: List[float]
    text_sentiment: List[Dict[str, float]]
    audio_emotion: List[Dict[str, float]]
    transcript_segments: List[Dict]
    facial_emotion: Optional[List[Dict[str, float]]] = None
    facial_timestamps: Optional[List[float]] = None

def create_distribution_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create interactive distribution plots with Plotly showing average sentiment and emotion.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A Plotly figure with subplots for sentiment, voice emotion, and facial emotion distribution
    """
    # Determine if we have facial data to plot
    has_facial_data = timeline_data.facial_emotion is not None and len(timeline_data.facial_emotion) > 0

    # Create a figure with appropriate number of subplots
    if has_facial_data:
        fig = sp.make_subplots(
            rows=1, cols=3,
            subplot_titles=("Average Text Sentiment", "Average Voice Emotion", "Average Facial Emotion"),
            horizontal_spacing=0.05
        )
    else:
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average Text Sentiment", "Average Voice Emotion"),
            horizontal_spacing=0.1
        )

    # Calculate average sentiment scores
    positive_avg = np.mean([s["positive"] for s in timeline_data.text_sentiment])
    negative_avg = np.mean([s["negative"] for s in timeline_data.text_sentiment])

    # Create DataFrame for sentiment
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Score': [positive_avg, negative_avg]
    })

    # Add sentiment bar chart (left subplot)
    fig.add_trace(
        go.Bar(
            x=sentiment_df['Sentiment'],
            y=sentiment_df['Score'],
            text=sentiment_df['Score'].apply(lambda x: f"{x:.1%}"),
            textposition='outside',
            marker_color=['#2ecc71', '#e74c3c'],
            name='Text Sentiment',
            showlegend=False
        ),
        row=1, col=1
    )

    # Calculate average emotion scores
    if timeline_data.audio_emotion:
        emotion_categories = list(timeline_data.audio_emotion[0].keys())
        emotion_averages = {}

        for category in emotion_categories:
            scores = [emotion.get(category, 0.0) for emotion in timeline_data.audio_emotion]
            emotion_averages[category] = np.mean(scores)

        # Create DataFrame for emotions
        emotion_df = pd.DataFrame({
            'Emotion': [cat.capitalize() for cat in emotion_averages.keys()],
            'Score': list(emotion_averages.values())
        })

        # Color map for emotions
        emotion_colors = {
            'Happy': '#3498db',  # Blue
            'Sad': '#9b59b6',    # Purple
            'Angry': '#e74c3c',  # Red
            'Neutral': '#95a5a6', # Gray
            'Fear': '#f39c12',   # Orange
            'Disgust': '#27ae60', # Green
            'Surprise': '#8e44ad' # Purple
        }

        # Map colors to the categories in the DataFrame
        colors = [emotion_colors.get(emotion, '#2c3e50') for emotion in emotion_df['Emotion']]

        # Add emotion bar chart (middle subplot)
        fig.add_trace(
            go.Bar(
                x=emotion_df['Emotion'],
                y=emotion_df['Score'],
                text=emotion_df['Score'].apply(lambda x: f"{x:.1%}"),
                textposition='outside',
                marker_color=colors,
                name='Voice Emotion',
                showlegend=False
            ),
            row=1, col=2
        )

    # Calculate average facial emotion scores if available
    if has_facial_data:
        facial_categories = list(timeline_data.facial_emotion[0].keys())
        facial_averages = {}

        for category in facial_categories:
            scores = [emotion.get(category, 0.0) for emotion in timeline_data.facial_emotion]
            facial_averages[category] = np.mean(scores)

        # Create DataFrame for facial emotions
        facial_df = pd.DataFrame({
            'Emotion': [cat.capitalize() for cat in facial_averages.keys()],
            'Score': list(facial_averages.values())
        })

        # Color map for facial emotions - use different shades
        facial_colors = {
            'Angry': '#c0392b',   # Dark red
            'Disgust': '#16a085', # Teal
            'Fear': '#d35400',    # Dark orange
            'Happy': '#2980b9',   # Dark blue
            'Sad': '#8e44ad',     # Dark purple
            'Surprise': '#d98880', # Light red
            'Neutral': '#7f8c8d'   # Dark gray
        }

        # Map colors to the categories in the DataFrame
        colors = [facial_colors.get(emotion, '#34495e') for emotion in facial_df['Emotion']]

        # Add facial emotion bar chart (right subplot)
        fig.add_trace(
            go.Bar(
                x=facial_df['Emotion'],
                y=facial_df['Score'],
                text=facial_df['Score'].apply(lambda x: f"{x:.1%}"),
                textposition='outside',
                marker_color=colors,
                name='Facial Emotion',
                showlegend=False
            ),
            row=1, col=3
        )

    # Update layout
    fig.update_layout(
        height=400,
        template="plotly",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # Update y-axis titles - keep "Score" only on the first plot
    fig.update_yaxes(
        title_text="Score",
        range=[0, 1],
        title_font=dict(color="#333333"),
        col=1
    )

    # Remove "Score" from other plots
    if has_facial_data:
        fig.update_yaxes(title_text=None, range=[0, 1], col=2)
        fig.update_yaxes(title_text=None, range=[0, 1], col=3)
    else:
        fig.update_yaxes(title_text=None, range=[0, 1], col=2)

    # Add gridlines with light gray color
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return fig

def create_text_sentiment_plot(timeline_data: TimelineData) -> go.Figure:
    """Create a plot for text sentiment."""
    fig = go.Figure()

    # Get the data for text sentiment
    timestamps = timeline_data.timestamps
    positive_scores = [s["positive"] for s in timeline_data.text_sentiment]
    negative_scores = [s["negative"] for s in timeline_data.text_sentiment]

    # Add traces for text sentiment
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positive_scores,
            mode='lines',
            name='Positive',
            line=dict(color='#2ecc71', width=2, shape='spline', smoothing=0.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=negative_scores,
            mode='lines',
            name='Negative',
            line=dict(color='#e74c3c', width=2, shape='spline', smoothing=0.5)
        )
    )

    # Set the layout
    fig.update_layout(
        title="Text Sentiment Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Sentiment Score",
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly"
    )

    return fig

def create_voice_emotion_plot(timeline_data: TimelineData) -> go.Figure:
    """Create a plot for voice emotion."""
    fig = go.Figure()

    # Get emotion data
    if timeline_data.audio_emotion:
        emotion_categories = list(timeline_data.audio_emotion[0].keys())

        # Prepare emotion data for each category
        emotion_data = {category: [] for category in emotion_categories}

        # Extract emotion scores for each category
        for emotion_dict in timeline_data.audio_emotion:
            for category in emotion_categories:
                emotion_data[category].append(emotion_dict.get(category, 0.0))

        # Create audio timestamps that match the length of audio emotion data
        audio_timestamps = np.linspace(
            min(timeline_data.timestamps),
            max(timeline_data.timestamps),
            len(timeline_data.audio_emotion)
        )

        # Color map for emotions
        emotion_colors = {
            'happy': '#3498db',  # Blue
            'sad': '#9b59b6',    # Purple
            'angry': '#e74c3c',  # Red
            'neutral': '#95a5a6', # Gray
            'fear': '#f39c12',   # Orange
            'disgust': '#27ae60', # Green
            'surprise': '#8e44ad' # Purple
        }

        # Add traces for each emotion
        for category, scores in emotion_data.items():
            color = emotion_colors.get(category, '#2c3e50')

            fig.add_trace(
                go.Scatter(
                    x=audio_timestamps,
                    y=scores,
                    mode='lines',
                    name=f"{category.capitalize()}",
                    line=dict(color=color, width=2, shape='spline', smoothing=0.5)
                )
            )

    # Set the layout
    fig.update_layout(
        title="Voice Emotion Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion Score",
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly"
    )

    return fig

def create_facial_emotion_plot(timeline_data: TimelineData) -> Optional[go.Figure]:
    """Create a plot for facial emotion if data is available."""
    # Check if we have facial data
    if not timeline_data.facial_emotion or not timeline_data.facial_timestamps:
        return None

    fig = go.Figure()

    facial_timestamps = timeline_data.facial_timestamps
    facial_categories = list(timeline_data.facial_emotion[0].keys())

    # Prepare facial emotion data for each category
    facial_data = {category: [] for category in facial_categories}

    # Extract emotion scores for each category
    for emotion_dict in timeline_data.facial_emotion:
        for category in facial_categories:
            facial_data[category].append(emotion_dict.get(category, 0.0))

    # Color map for facial emotions
    facial_colors = {
        'angry': '#c0392b',   # Dark red
        'disgust': '#16a085', # Teal
        'fear': '#d35400',    # Dark orange
        'happy': '#2980b9',   # Dark blue
        'sad': '#8e44ad',     # Dark purple
        'surprise': '#d98880', # Light red
        'neutral': '#7f8c8d'   # Dark gray
    }

    # Add traces for each facial emotion
    for category, scores in facial_data.items():
        color = facial_colors.get(category, '#34495e')

        fig.add_trace(
            go.Scatter(
                x=facial_timestamps,
                y=scores,
                mode='lines',
                name=f"{category.capitalize()}",
                line=dict(color=color, width=2, shape='spline', smoothing=0.5)
            )
        )

    # Set the layout
    fig.update_layout(
        title="Facial Emotion Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion Score",
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly"
    )

    return fig

def create_timeline_plots(timeline_data: TimelineData) -> Tuple[go.Figure, go.Figure, Optional[go.Figure]]:
    """
    Create separate interactive timeline plots with Plotly showing sentiment and emotion over time.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A tuple of Plotly figures for text sentiment, voice emotion, and facial emotion (which may be None)
    """
    text_fig = create_text_sentiment_plot(timeline_data)
    voice_fig = create_voice_emotion_plot(timeline_data)
    facial_fig = create_facial_emotion_plot(timeline_data)

    return text_fig, voice_fig, facial_fig

# Re-export these functions to maintain existing imports
__all__ = [
    'create_distribution_plot',
    'create_timeline_plots',
    'create_emotion_trends_plot',
    'create_emotional_contagion_heatmap',
    'create_emotion_transition_network_plot',
]
