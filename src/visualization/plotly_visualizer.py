"""Interactive visualizations using Plotly for advanced emotion analytics."""
import json
import subprocess
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from src.visualization.visualizer import Visualizer
import plotly.subplots as sp
from typing import Tuple, Optional


def _prepare_comments_js(raw_comments: list) -> list:
    """
    Invoke the prepareComments.js Node script from the root directory to process comments.

    This function calls the script located at /prepareComments.js (root directory),
    NOT the sample data file in the visualization folder.

    Args:
        raw_comments: List of comment dictionaries with 'create_time' and 'text' fields

    Returns:
        List of processed comments with normalized timestamps

    Raises:
        RuntimeError: If the Node.js script fails for any reason
    """
    proc = subprocess.run(
        ['node', 'prepareComments.js'],  # Uses script from root directory
        input=json.dumps(raw_comments),
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"prepareComments.js failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def create_emotion_trends_plot(
    comments: list,
    freq: str = 'H',
    rolling_window: int = 3
) -> go.Figure:
    # first, normalize & timestamp-convert via your JS preparer
    comments = _prepare_comments_js(comments)

    # Check if we have enough data
    if len(comments) < 3:  # Need at least 3 comments for meaningful trends
        # Create an empty figure with an annotation
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for emotion trend analysis.<br>Upload a video with more dialogue.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title='Emotion Trends Over Time',
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig

    try:
        vis = Visualizer()
        trends = vis.compute_emotion_trends(
            comments,
            freq=freq,
            rolling_window=rolling_window
        )

        # Check if we have valid trends data
        if trends.empty or trends.isnull().all().all():
            # No valid trends found
            fig = go.Figure()
            fig.add_annotation(
                text="No emotion trends could be detected.<br>Try uploading a video with more emotional variation.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Emotion Trends Over Time',
                template='plotly_white',
                height=400,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            return fig

        fig = go.Figure()
        for emotion in trends.columns:
            fig.add_trace(
                go.Scatter(
                    x=trends.index,
                    y=trends[emotion],
                    mode='lines',
                    name=emotion.capitalize(),
                    hovertemplate='%{y:.2f} on %{x}'
                )
            )
        fig.update_layout(
            title='Emotion Trends Over Time',
            xaxis_title='Time',
            yaxis_title='Comment Count (rolling avg)',
            template='plotly_white',
            legend_title='Emotion'
        )
        return fig
    except Exception as e:
        # Create a figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not generate emotion trends plot.<br>Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='red')
        )
        fig.update_layout(
            title='Emotion Trends Over Time',
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig


def create_emotional_contagion_heatmap(
    comments: list,
    freq: str = 'H',
    maxlag: int = 3
) -> go.Figure:
    comments = _prepare_comments_js(comments)

    vis = Visualizer()

    # Check if we have enough data
    if len(comments) < 5:  # Need at least 5 comments for meaningful analysis
        # Create an empty figure with an annotation
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for emotional contagion analysis.<br>Upload a video with more dialogue.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title='Emotional Contagion Heatmap',
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig

    try:
        pvals = vis.compute_emotional_contagion(
            comments,
            freq=freq,
            maxlag=maxlag
        )
        # Check if we have valid p-values
        if pvals.isnull().all().all():
            # No valid p-values found
            fig = go.Figure()
            fig.add_annotation(
                text="No significant emotional patterns detected.<br>Try adjusting the frequency or uploading a different video.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Emotional Contagion Heatmap',
                template='plotly_white',
                height=400,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            return fig

        # Transform p-values to -log10
        mat = -np.log10(pvals.astype(float))

        fig = go.Figure(
            data=go.Heatmap(
                z=mat.values,
                x=[e.capitalize() for e in mat.columns],
                y=[e.capitalize() for e in mat.index],
                colorscale='Viridis',
                colorbar=dict(title='-log10(p-value)')
            )
        )
        fig.update_layout(
            title='Emotional Contagion Heatmap',
            xaxis_nticks=len(mat.columns),
            yaxis_nticks=len(mat.index),
            template='plotly_white'
        )
        return fig
    except Exception as e:
        # Create a figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not generate emotional contagion heatmap.<br>Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='red')
        )
        fig.update_layout(
            title='Emotional Contagion Heatmap',
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig


def create_emotion_transition_network_plot(
    comments: list
) -> go.Figure:
    comments = _prepare_comments_js(comments)

    # Check if we have enough data
    if len(comments) < 5:  # Need at least 5 comments for meaningful analysis
        # Create an empty figure with an annotation
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for emotion transition analysis.<br>Upload a video with more dialogue.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title='Emotion Transition Network',
            template='plotly_white',
            height=500,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig

    try:
        vis = Visualizer()
        trans = vis.compute_emotion_transitions(comments)

        # Check if we have valid transitions
        if trans.sum().sum() == 0:
            # No valid transitions found
            fig = go.Figure()
            fig.add_annotation(
                text="No emotion transitions detected.<br>Try uploading a video with more varied emotional content.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Emotion Transition Network',
                template='plotly_white',
                height=500,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            return fig

        # Build NetworkX graph
        G = nx.DiGraph()
        for src in trans.index:
            for tgt in trans.columns:
                weight = trans.loc[src, tgt]
                if weight > 0:
                    G.add_edge(src.capitalize(), tgt.capitalize(), weight=weight)

        # Check if we have any edges
        if len(G.edges()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No transitions between emotions detected.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Emotion Transition Network',
                template='plotly_white',
                height=500,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            return fig

        pos = nx.circular_layout(G)
        edge_x, edge_y = [], []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='#ffcc00'),
            text=list(G.nodes()),
            textposition='middle center',
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Emotion Transition Network',
            showlegend=False,
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )

        # annotate edge weights
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=f"{data['weight']:.2f}",
                showarrow=False,
                font=dict(size=8, color='#000')
            )

        return fig
    except Exception as e:
        # Create a figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not generate emotion transition network.<br>Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='red')
        )
        fig.update_layout(
            title='Emotion Transition Network',
            template='plotly_white',
            height=500,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        return fig


def create_timeline_plot(timeline_data):
    """
    Create an interactive timeline plot with Plotly showing sentiment and emotion over time.

    This is a backward compatibility function that returns the combined figure.
    For separate plots, use create_timeline_plots() instead.

    Args:
        timeline_data: The timeline data containing sentiment and emotion scores

    Returns:
        A single Plotly figure with subplots for sentiment and emotions
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

    # Update layout
    fig.update_layout(
        height=800 if has_facial_data else 600,
        template="plotly",
        showlegend=True,
        margin=dict(l=40, r=120, t=80, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15
        )
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1,
                     title_font=dict(color="#333333"))
    fig.update_yaxes(title_text="Emotion Score", row=2, col=1,
                     title_font=dict(color="#333333"))

    if has_facial_data:
        fig.update_yaxes(title_text="Emotion Score", row=3, col=1,
                         title_font=dict(color="#333333"))
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1,
                         title_font=dict(color="#333333"))
    else:
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1,
                         title_font=dict(color="#333333"))

    # Add gridlines with light gray color
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    return fig


def create_distribution_plot(timeline_data):
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
