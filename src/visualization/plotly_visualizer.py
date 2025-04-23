"""Interactive visualizations using Plotly for sentiment and emotion analysis results."""
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .visualizer import TimelineData

def create_timeline_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create an interactive timeline plot with Plotly showing sentiment and emotion over time.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A Plotly figure with two subplots for sentiment and emotion
    """
    # Create a figure with 2 subplots vertically stacked
    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Text Sentiment Over Time", "Voice Emotion Over Time"),
        vertical_spacing=0.3
    )

    # Get the data for text sentiment
    timestamps = timeline_data.timestamps
    positive_scores = [s["positive"] for s in timeline_data.text_sentiment]
    negative_scores = [s["negative"] for s in timeline_data.text_sentiment]

    # Add traces for text sentiment (top subplot)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positive_scores,
            mode='lines',
            name='Positive',
            line=dict(color='#2ecc71', width=2, shape='spline', smoothing=0.5)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=negative_scores,
            mode='lines',
            name='Negative',
            line=dict(color='#e74c3c', width=2, shape='spline', smoothing=0.5)
        ),
        row=1, col=1
    )

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
            min(timestamps),
            max(timestamps),
            len(timeline_data.audio_emotion)
        )

        # Color map for emotions
        emotion_colors = {
            'happy': '#3498db',  # Blue
            'sad': '#9b59b6',    # Purple
            'angry': '#e74c3c',  # Red
            'neutral': '#95a5a6' # Gray
        }

        # Add traces for each emotion (bottom subplot)
        for category, scores in emotion_data.items():
            color = emotion_colors.get(category, '#2c3e50')

            fig.add_trace(
                go.Scatter(
                    x=audio_timestamps,
                    y=scores,
                    mode='lines',
                    name=category.capitalize(),
                    line=dict(color=color, width=2, shape='spline', smoothing=0.5)
                ),
                row=2, col=1
            )

    # Update layout for better appearance
    fig.update_layout(
        height=600,
        template="plotly",
        # Move legend below the first graph
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.55,  # Position below the first subplot
            xanchor="center",
            x=0.5,   # Center horizontally
            font=dict(size=10)
        ),
        margin=dict(l=40, r=40, t=100, b=40),
        hovermode="x unified",
    )

    # Update axis labels with darker colors for light background
    # Extend y-axis range slightly to ensure lines at value 1 are visible
    fig.update_yaxes(title_text="Sentiment Score", range=[0, 1.05], row=1, col=1,
                    title_font=dict(color="#333333"))
    fig.update_yaxes(title_text="Emotion Score", range=[0, 1.05], row=2, col=1,
                    title_font=dict(color="#333333"))
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1,
                    title_font=dict(color="#333333"))

    # Add gridlines with light gray color
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    # Update annotation positions for better title spacing
    for i in fig['layout']['annotations']:
        i['y'] = i['y'] + 0.05

    return fig

def create_distribution_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create interactive distribution plots with Plotly showing average sentiment and emotion.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A Plotly figure with two subplots for sentiment and emotion distribution
    """
    # Create a figure with 2 subplots side by side
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
            name='Text Sentiment'
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
            'Neutral': '#95a5a6' # Gray
        }

        # Map colors to the categories in the DataFrame
        colors = [emotion_colors.get(emotion, '#2c3e50') for emotion in emotion_df['Emotion']]

        # Add emotion bar chart (right subplot)
        fig.add_trace(
            go.Bar(
                x=emotion_df['Emotion'],
                y=emotion_df['Score'],
                text=emotion_df['Score'].apply(lambda x: f"{x:.1%}"),
                textposition='outside',
                marker_color=colors,
                name='Audio Emotion'
            ),
            row=1, col=2
        )

    # Update layout for better appearance
    fig.update_layout(
        height=400,
        template="plotly",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Update axis formatting with darker colors for light background
    fig.update_yaxes(title_text="Score", range=[0, 1], tickformat=".0%", row=1, col=1,
                    title_font=dict(color="#333333"))
    fig.update_yaxes(title_text="Score", range=[0, 1], tickformat=".0%", row=1, col=2,
                    title_font=dict(color="#333333"))

    # Add gridlines with light gray color
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return fig
