"""Module for visualizing sentiment and emotion analysis results, extended with advanced emotion analytics."""
import io
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from nrclex import NRCLex
from statsmodels.tsa.stattools import grangercausalitytests

@dataclass
class TimelineData:
    """Data class for storing timeline analysis results."""
    timestamps: List[float]
    text_sentiment: List[Dict[str, float]]
    audio_emotion: List[Dict[str, float]]
    transcript_segments: List[Dict]
    facial_emotion: Optional[List[Dict[str, float]]] = None
    facial_timestamps: Optional[List[float]] = None

class Visualizer:
    """Creates visualizations and computes advanced emotion analytics."""

    def fuse_results(
        self,
        text_sentiment_results: List[Tuple[float, Dict[str, float]]],
        audio_emotion_results: List[Tuple[float, Dict[str, float]]],
        transcript_segments: List[Dict],
        facial_emotion_results: Optional[List[Tuple[float, Dict[str, float]]]] = None
    ) -> TimelineData:
        """
        Fuse text sentiment and audio emotion results into a unified timeline.

        Args:
            text_sentiment_results: List of (timestamp, sentiment) tuples
            audio_emotion_results: List of (timestamp, emotion) tuples
            transcript_segments: List of transcript segments with timestamps
            facial_emotion_results: List of (timestamp, emotion) tuples from facial analysis

        Returns:
            TimelineData: Fused timeline data
        """
        # Extract timestamps and results from both analyses
        text_timestamps = [t for t, _ in text_sentiment_results]
        text_sentiments = [s for _, s in text_sentiment_results]

        audio_timestamps = [t for t, _ in audio_emotion_results]
        audio_emotions = [e for _, e in audio_emotion_results]

        # Extract facial emotion data if available
        facial_timestamps = None
        facial_emotions = None
        if facial_emotion_results:
            facial_timestamps = [t for t, _ in facial_emotion_results]
            facial_emotions = [e for _, e in facial_emotion_results]

        # Create a unified timeline
        return TimelineData(
            timestamps=text_timestamps,  # Using text timestamps as reference
            text_sentiment=text_sentiments,
            audio_emotion=audio_emotions,
            transcript_segments=transcript_segments,
            facial_emotion=facial_emotions,
            facial_timestamps=facial_timestamps
        )

    def _classify_emotion(self, text: str) -> str:
        """Classify text into one of the NRC emotion categories."""
        emo = NRCLex(text)
        scores = emo.raw_emotion_scores  # dict: emotion -> count
        if not scores:
            return 'neutral'
        # Return the emotion with highest count
        return max(scores, key=scores.get)

    def compute_comment_emotions(
        self,
        comments: List[Dict]
    ) -> pd.DataFrame:
        """
        Compute emotion label for each comment using NRC Lexicon.
        Returns a DataFrame with columns ['create_time', 'emotion'].
        """
        df = pd.DataFrame(comments)
        # Assume create_time is UNIX timestamp in seconds
        df['create_time'] = pd.to_datetime(df['create_time'], unit='s')
        df['emotion'] = df['text'].apply(self._classify_emotion)
        return df[['create_time', 'emotion']]

    def compute_emotion_trends(
        self,
        comments: List[Dict],
        freq: str = 'H',
        rolling_window: int = 3
    ) -> pd.DataFrame:
        """
        Build an emotion trend DataFrame indexed by time bins (e.g., hourly) with
        columns for each emotion showing rolling average comment counts.
        """
        df = self.compute_comment_emotions(comments)
        df.set_index('create_time', inplace=True)
        # Count comments of each emotion per time bin
        counts = df.groupby([pd.Grouper(freq=freq), 'emotion']).size().unstack(fill_value=0)
        # Rolling average smooths the curve
        trends = counts.rolling(window=rolling_window, min_periods=1).mean()
        return trends

    def plot_emotion_trends(
        self,
        trends: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot emotion trends over time as a multi-line chart.
        Returns base64 string if output_path is None.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        for emotion in trends.columns:
            ax.plot(trends.index, trends[emotion], label=emotion)
        ax.set_ylabel('Comment Count (rolling avg)')
        ax.set_xlabel('Time')
        ax.set_title('Emotion Trends Over Time')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def compute_emotional_contagion(
        self,
        comments: List[Dict],
        freq: str = 'H',
        maxlag: int = 3
    ) -> pd.DataFrame:
        """
        Measure pairwise Granger causality p-values between emotion time-series.
        Returns a DataFrame of p-values indexed and columned by emotion.
        """
        trends = self.compute_emotion_trends(comments, freq=freq, rolling_window=1)
        emotions = trends.columns.tolist()
        pvals = pd.DataFrame(index=emotions, columns=emotions, dtype=float)
        for src in emotions:
            for tgt in emotions:
                data = trends[[tgt, src]].dropna()
                # Only test if enough data points
                if len(data) > maxlag:
                    result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    # extract smallest p-value across lags
                    p = min(res[0]['ssr_ftest'][1] for lag, res in result.items())
                    pvals.loc[src, tgt] = p
                else:
                    pvals.loc[src, tgt] = np.nan
        return pvals

    def plot_emotional_contagion(
        self,
        pvals: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot a heatmap of -log10(p-values) for emotional contagion.
        Returns base64 string if output_path is None.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        mat = -np.log10(pvals.astype(float))
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('-log10(p-value)')
        ax.set_title('Emotional Contagion Heatmap')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def compute_emotion_transitions(
        self,
        comments: List[Dict]
    ) -> pd.DataFrame:
        """
        Build a transition probability matrix between consecutive comment emotions.
        Returns a DataFrame with emotions as index and columns.
        """
        df = self.compute_comment_emotions(comments)
        df.sort_values('create_time', inplace=True)
        # Count transitions
        transitions = {}
        prev = None
        for emo in df['emotion']:
            if prev is not None:
                transitions[(prev, emo)] = transitions.get((prev, emo), 0) + 1
            prev = emo
        emotions = sorted(df['emotion'].unique())
        mat = pd.DataFrame(0, index=emotions, columns=emotions, dtype=float)
        for (i, j), count in transitions.items():
            mat.loc[i, j] = count
        # Normalize rows
        mat = mat.div(mat.sum(axis=1).replace({0: 1}), axis=0)
        return mat

    def plot_emotion_transition_network(
        self,
        trans_mat: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot a directed graph of emotion transitions with edge widths proportional to transition probability.
        Returns base64 string if output_path is None.
        """
        G = nx.DiGraph()
        for src in trans_mat.index:
            for tgt in trans_mat.columns:
                weight = trans_mat.loc[src, tgt]
                if weight > 0:
                    G.add_edge(src, tgt, weight=weight)
        pos = nx.circular_layout(G)
        weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=1500,
            font_size=10,
            width=weights,
            arrowstyle='-|>',
            arrowsize=12,
            ax=ax
        )
        # Edge labels
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        ax.set_title('Emotion Transition Network')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def plot_sentiment_vs_emotion(
        self,
        timeline_data: TimelineData,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot sentiment vs emotion over time.

        Args:
            timeline_data: Fused timeline data
            output_path: Path to save the plot image (if None, returns base64 encoded image)

        Returns:
            Optional[str]: Base64 encoded image if output_path is None
        """
        # Determine if we have facial data to plot
        has_facial_data = timeline_data.facial_emotion is not None and len(timeline_data.facial_emotion) > 0

        if has_facial_data:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot text sentiment
        timestamps = timeline_data.timestamps
        positive_scores = [s["positive"] for s in timeline_data.text_sentiment]
        negative_scores = [s["negative"] for s in timeline_data.text_sentiment]

        ax1.plot(timestamps, positive_scores, 'g-', label='Positive')
        ax1.plot(timestamps, negative_scores, 'r-', label='Negative')
        ax1.set_ylabel('Text Sentiment Score')
        ax1.set_title('Text Sentiment Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot audio emotion
        # Assuming we have emotion categories in the results
        # Get the first emotion result to extract categories
        if timeline_data.audio_emotion:
            emotion_categories = list(timeline_data.audio_emotion[0].keys())

            # Prepare emotion data
            emotion_data = {category: [] for category in emotion_categories}

            # Extract emotion scores for each category
            for emotion_dict in timeline_data.audio_emotion:
                for category in emotion_categories:
                    emotion_data[category].append(emotion_dict.get(category, 0.0))

            # Check if we have audio data with different length than text data
            # If so, we need to interpolate or resample to match the timestamps
            audio_timestamps = np.linspace(
                min(timestamps),
                max(timestamps),
                len(timeline_data.audio_emotion)
            )

            # Plot each emotion category with its appropriate timestamps
            for category, scores in emotion_data.items():
                ax2.plot(audio_timestamps, scores, label=category)

            ax2.set_ylabel('Audio Emotion Score')
            ax2.set_title('Voice Emotion Over Time')
            ax2.legend()
            ax2.grid(True)

        # Plot facial emotion if available
        if has_facial_data:
            facial_timestamps = timeline_data.facial_timestamps
            facial_categories = list(timeline_data.facial_emotion[0].keys())

            # Prepare emotion data
            facial_data = {category: [] for category in facial_categories}

            # Extract emotion scores for each category
            for emotion_dict in timeline_data.facial_emotion:
                for category in facial_categories:
                    facial_data[category].append(emotion_dict.get(category, 0.0))

            # Plot each facial emotion category
            for category, scores in facial_data.items():
                ax3.plot(facial_timestamps, scores, label=category)

            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Facial Emotion Score')
            ax3.set_title('Facial Emotion Over Time')
            ax3.legend()
            ax3.grid(True)
        else:
            ax2.set_xlabel('Time (seconds)')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            return None
        else:
            # Return base64 encoded image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str

    def plot_emotion_distribution(
        self,
        timeline_data: TimelineData,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot the distribution of emotions across the video.

        Args:
            timeline_data: Fused timeline data
            output_path: Path to save the plot image (if None, returns base64 encoded image)

        Returns:
            Optional[str]: Base64 encoded image if output_path is None
        """
        # Determine if we have facial data to plot
        has_facial_data = timeline_data.facial_emotion is not None and len(timeline_data.facial_emotion) > 0

        # Calculate number of charts needed
        num_charts = 2  # Text sentiment and audio emotion
        if has_facial_data:
            num_charts += 1  # Add facial emotion

        # Set up the figure size based on number of charts
        fig_width = 6 * num_charts
        fig, axs = plt.subplots(1, num_charts, figsize=(fig_width, 6))

        # Plot sentiment distribution
        positive_avg = np.mean([s["positive"] for s in timeline_data.text_sentiment])
        negative_avg = np.mean([s["negative"] for s in timeline_data.text_sentiment])

        sentiment_labels = ['Positive', 'Negative']
        sentiment_values = [positive_avg, negative_avg]

        axs[0].bar(sentiment_labels, sentiment_values, color=['green', 'red'])
        axs[0].set_ylabel('Average Score')
        axs[0].set_title('Average Text Sentiment')
        axs[0].set_ylim(0, 1)

        # Plot audio emotion distribution
        if timeline_data.audio_emotion:
            # Get emotion categories
            emotion_categories = list(timeline_data.audio_emotion[0].keys())

            # Calculate average score for each emotion
            emotion_averages = {}
            for category in emotion_categories:
                scores = [emotion.get(category, 0.0) for emotion in timeline_data.audio_emotion]
                emotion_averages[category] = np.mean(scores)

            categories = list(emotion_averages.keys())
            values = list(emotion_averages.values())

            # Choose a colormap based on number of categories
            colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

            axs[1].bar(categories, values, color=colors)
            axs[1].set_ylabel('Average Score')
            axs[1].set_title('Average Voice Emotion')
            axs[1].set_ylim(0, 1)
            plt.setp(axs[1].get_xticklabels(), rotation=45)

        # Plot facial emotion distribution if available
        if has_facial_data:
            # Get facial emotion categories
            facial_categories = list(timeline_data.facial_emotion[0].keys())

            # Calculate average score for each facial emotion
            facial_averages = {}
            for category in facial_categories:
                scores = [emotion.get(category, 0.0) for emotion in timeline_data.facial_emotion]
                facial_averages[category] = np.mean(scores)

            categories = list(facial_averages.keys())
            values = list(facial_averages.values())

            # Choose a colormap based on number of categories
            colors = plt.cm.plasma(np.linspace(0, 1, len(categories)))

            axs[2].bar(categories, values, color=colors)
            axs[2].set_ylabel('Average Score')
            axs[2].set_title('Average Facial Emotion')
            axs[2].set_ylim(0, 1)
            plt.setp(axs[2].get_xticklabels(), rotation=45)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            return None
        else:
            # Return base64 encoded image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
