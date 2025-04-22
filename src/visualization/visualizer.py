"""Module for visualizing sentiment and emotion analysis results."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import io
import base64
from dataclasses import dataclass

@dataclass
class TimelineData:
    """Data class for storing timeline analysis results."""
    timestamps: List[float]
    text_sentiment: List[Dict[str, float]]
    audio_emotion: List[Dict[str, float]]
    transcript_segments: List[Dict]

class Visualizer:
    """Creates visualizations for sentiment and emotion analysis results."""

    def fuse_results(
        self,
        text_sentiment_results: List[Tuple[float, Dict[str, float]]],
        audio_emotion_results: List[Tuple[float, Dict[str, float]]],
        transcript_segments: List[Dict]
    ) -> TimelineData:
        """
        Fuse text sentiment and audio emotion results into a unified timeline.

        Args:
            text_sentiment_results: List of (timestamp, sentiment) tuples
            audio_emotion_results: List of (timestamp, emotion) tuples
            transcript_segments: List of transcript segments with timestamps

        Returns:
            TimelineData: Fused timeline data
        """
        # Extract timestamps and results from both analyses
        text_timestamps = [t for t, _ in text_sentiment_results]
        text_sentiments = [s for _, s in text_sentiment_results]

        audio_timestamps = [t for t, _ in audio_emotion_results]
        audio_emotions = [e for _, e in audio_emotion_results]

        # Create a unified timeline
        return TimelineData(
            timestamps=text_timestamps,  # Using text timestamps as reference
            text_sentiment=text_sentiments,
            audio_emotion=audio_emotions,
            transcript_segments=transcript_segments
        )

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

            # Plot each emotion category
            for category, scores in emotion_data.items():
                ax2.plot(timestamps[:len(scores)], scores, label=category)

            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Audio Emotion Score')
            ax2.set_title('Voice Emotion Over Time')
            ax2.legend()
            ax2.grid(True)

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
        if not timeline_data.audio_emotion:
            return None

        # Get emotion categories
        emotion_categories = list(timeline_data.audio_emotion[0].keys())

        # Calculate average score for each emotion
        emotion_averages = {}
        for category in emotion_categories:
            scores = [emotion.get(category, 0.0) for emotion in timeline_data.audio_emotion]
            emotion_averages[category] = np.mean(scores)

        # Calculate average sentiment
        positive_avg = np.mean([s["positive"] for s in timeline_data.text_sentiment])
        negative_avg = np.mean([s["negative"] for s in timeline_data.text_sentiment])

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot sentiment distribution
        sentiment_labels = ['Positive', 'Negative']
        sentiment_values = [positive_avg, negative_avg]

        ax1.bar(sentiment_labels, sentiment_values, color=['green', 'red'])
        ax1.set_ylabel('Average Score')
        ax1.set_title('Average Text Sentiment')
        ax1.set_ylim(0, 1)

        # Plot emotion distribution
        categories = list(emotion_averages.keys())
        values = list(emotion_averages.values())

        # Choose a colormap based on number of categories
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

        ax2.bar(categories, values, color=colors)
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Voice Emotion')
        ax2.set_ylim(0, 1)
        plt.xticks(rotation=45)

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
