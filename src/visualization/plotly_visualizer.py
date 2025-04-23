"""Interactive visualizations using Plotly for sentiment and emotion analysis results."""
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .visualizer import TimelineData

def create_text_sentiment_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create a timeline plot for text sentiment.

    Args:
        timeline_data: The timeline data containing sentiment scores

    Returns:
        A Plotly figure with text sentiment timeline
    """
    # Get the data for text sentiment
    timestamps = timeline_data.timestamps
    positive_scores = [s["positive"] for s in timeline_data.text_sentiment]
    negative_scores = [s["negative"] for s in timeline_data.text_sentiment]

    # Create the figure
    fig = go.Figure()

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

    # Update layout
    fig.update_layout(
        title="Text Sentiment Over Time",
        title_x=0.5,  # Center the title
        height=300,
        template="plotly",
        margin=dict(l=40, r=150, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )

    # Update axes
    fig.update_yaxes(
        title_text="Sentiment Score",
        range=[0, 1.05],
        title_font=dict(color="#333333")
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return fig

def create_voice_emotion_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create a timeline plot for voice emotions.

    Args:
        timeline_data: The timeline data containing voice emotion scores

    Returns:
        A Plotly figure with voice emotion timeline
    """
    # Get the emotion data
    timestamps = timeline_data.timestamps

    # Create the figure
    fig = go.Figure()

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
                    name=category.capitalize(),
                    line=dict(color=color, width=2, shape='spline', smoothing=0.5)
                )
            )

    # Update layout
    fig.update_layout(
        title="Voice Emotion Over Time",
        title_x=0.5,  # Center the title
        height=300,
        template="plotly",
        margin=dict(l=40, r=150, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )

    # Update axes
    fig.update_yaxes(
        title_text="Voice Emotion Score",
        range=[0, 1.05],
        title_font=dict(color="#333333")
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return fig

def create_facial_emotion_plot(timeline_data: TimelineData) -> Optional[go.Figure]:
    """
    Create a timeline plot for facial emotions.

    Args:
        timeline_data: The timeline data containing facial emotion scores

    Returns:
        A Plotly figure with facial emotion timeline or None if no facial data
    """
    # Check if we have facial data
    has_facial_data = timeline_data.facial_emotion is not None and len(timeline_data.facial_emotion) > 0

    if not has_facial_data:
        return None

    # Create the figure
    fig = go.Figure()

    # Get facial emotion data
    facial_timestamps = timeline_data.facial_timestamps
    facial_categories = list(timeline_data.facial_emotion[0].keys())

    # Prepare facial emotion data for each category
    facial_data = {category: [] for category in facial_categories}

    # Extract emotion scores for each category
    for emotion_dict in timeline_data.facial_emotion:
        for category in facial_categories:
            facial_data[category].append(emotion_dict.get(category, 0.0))

    # Color map for facial emotions - use different shades
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
                name=category.capitalize(),
                line=dict(color=color, width=2, shape='spline', smoothing=0.5)
            )
        )

    # Update layout
    fig.update_layout(
        title="Facial Emotion Over Time",
        title_x=0.5,  # Center the title
        height=300,
        template="plotly",
        margin=dict(l=40, r=150, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )

    # Update axes
    fig.update_yaxes(
        title_text="Facial Emotion Score",
        range=[0, 1.05],
        title_font=dict(color="#333333")
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        title_font=dict(color="#333333")
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

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

# Keeping the original function for backward compatibility
def create_timeline_plot(timeline_data: TimelineData) -> go.Figure:
    """
    Create an interactive timeline plot with Plotly showing sentiment and emotion over time.

    DEPRECATED: Use create_timeline_plots instead for better legend positioning.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A Plotly figure with subplots for sentiment, voice emotion, and facial emotion
    """
    # Determine if we have facial data to plot
    has_facial_data = timeline_data.facial_emotion is not None and len(timeline_data.facial_emotion) > 0

    # Create a figure with appropriate subplots (2 or 3 depending on data)
    if has_facial_data:
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Text Sentiment Over Time", "Voice Emotion Over Time", "Facial Emotion Over Time"),
            vertical_spacing=0.15,
            row_heights=[0.33, 0.33, 0.33]
        )
    else:
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
            line=dict(color='#2ecc71', width=2, shape='spline', smoothing=0.5),
            legendgroup="text",
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=negative_scores,
            mode='lines',
            name='Negative',
            line=dict(color='#e74c3c', width=2, shape='spline', smoothing=0.5),
            legendgroup="text",
            showlegend=True
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
            'neutral': '#95a5a6', # Gray
            'fear': '#f39c12',   # Orange
            'disgust': '#27ae60', # Green
            'surprise': '#8e44ad' # Purple
        }

        # Add traces for each emotion (middle subplot)
        for category, scores in emotion_data.items():
            color = emotion_colors.get(category, '#2c3e50')

            fig.add_trace(
                go.Scatter(
                    x=audio_timestamps,
                    y=scores,
                    mode='lines',
                    name=f"Voice {category.capitalize()}",
                    line=dict(color=color, width=2, shape='spline', smoothing=0.5),
                    legendgroup="voice",
                    showlegend=True
                ),
                row=2, col=1
            )

    # Get facial emotion data if available
    if has_facial_data:
        facial_timestamps = timeline_data.facial_timestamps
        facial_categories = list(timeline_data.facial_emotion[0].keys())

        # Prepare facial emotion data for each category
        facial_data = {category: [] for category in facial_categories}

        # Extract emotion scores for each category
        for emotion_dict in timeline_data.facial_emotion:
            for category in facial_categories:
                facial_data[category].append(emotion_dict.get(category, 0.0))

        # Color map for facial emotions - use different shades
        facial_colors = {
            'angry': '#c0392b',   # Dark red
            'disgust': '#16a085', # Teal
            'fear': '#d35400',    # Dark orange
            'happy': '#2980b9',   # Dark blue
            'sad': '#8e44ad',     # Dark purple
            'surprise': '#d98880', # Light red
            'neutral': '#7f8c8d'   # Dark gray
        }

        # Add traces for each facial emotion (bottom subplot)
        for category, scores in facial_data.items():
            color = facial_colors.get(category, '#34495e')

            fig.add_trace(
                go.Scatter(
                    x=facial_timestamps,
                    y=scores,
                    mode='lines',
                    name=f"Face {category.capitalize()}",
                    line=dict(color=color, width=2, shape='spline', smoothing=0.5),
                    legendgroup="face",
                    showlegend=True
                ),
                row=3, col=1
            )

    # Create separate legends for each subplot by using multiple annotation groups
    if has_facial_data:
        # Update the layout to include three separate legends
        fig.update_layout(
            legend_tracegroupgap=180,  # Add more space between legend groups
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                itemsizing="constant",
                font=dict(size=10),
                tracegroupgap=5
            )
        )

        # Create separate legend annotations for each group
        # Position text sentiment legend at the top
        for trace in fig.data:
            if trace.legendgroup == "text":
                trace.update(legendgroup="1_text")
            elif trace.legendgroup == "voice":
                trace.update(legendgroup="2_voice")
            elif trace.legendgroup == "face":
                trace.update(legendgroup="3_face")

        # Use updatemenus to control legend position (hidden from user)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="",
                            method="relayout",
                            args=[{"legend.y": 0.99, "legend.tracegroupgap": 180}],
                            visible=False
                        )
                    ],
                    visible=False
                )
            ]
        )
    else:
        # For 2-subplot case
        fig.update_layout(
            legend_tracegroupgap=120,  # Add more space between legend groups
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                itemsizing="constant",
                font=dict(size=10),
                tracegroupgap=5
            )
        )

        # Create separate legend groups for 2-subplot case
        for trace in fig.data:
            if trace.legendgroup == "text":
                trace.update(legendgroup="1_text")
            elif trace.legendgroup == "voice":
                trace.update(legendgroup="2_voice")

    # Update layout for better appearance
    fig.update_layout(
        height=800 if has_facial_data else 600,
        template="plotly",
        margin=dict(l=40, r=140, b=40, t=100),  # Increased right margin for legends
        hovermode="x unified",
    )

    # Update axis labels with darker colors for light background
    # Extend y-axis range slightly to ensure lines at value 1 are visible
    fig.update_yaxes(title_text="Sentiment Score", range=[0, 1.05], row=1, col=1,
                    title_font=dict(color="#333333"))
    fig.update_yaxes(title_text="Voice Emotion Score", range=[0, 1.05], row=2, col=1,
                    title_font=dict(color="#333333"))

    if has_facial_data:
        fig.update_yaxes(title_text="Facial Emotion Score", range=[0, 1.05], row=3, col=1,
                        title_font=dict(color="#333333"))
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1,
                        title_font=dict(color="#333333"))
    else:
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
