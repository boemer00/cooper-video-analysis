"""Main pipeline for video sentiment and emotion analysis."""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
from dataclasses import dataclass

from .preprocessing import extract_audio, Transcriber, AudioEmotionAnalyzer
from .inference import TextSentimentAnalyzer
from .visualization import Visualizer, TimelineData

@dataclass
class AnalysisResults:
    """Container for the results of the video analysis."""
    text_scores: Dict
    audio_scores: Dict
    timeline_data: TimelineData
    timeline_plot_bytes: Optional[str] = None
    dist_plot_bytes: Optional[str] = None

class Pipeline:
    """Main pipeline for video sentiment and emotion analysis."""

    def __init__(
        self,
        whisper_model: str = "base",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        emotion_model: str = "r-f/wav2vec-english-speech-emotion-recognition"
    ):
        """
        Initialize the video analysis pipeline.

        Args:
            whisper_model: Model name for Whisper transcription
            sentiment_model: Model name for text sentiment analysis
            emotion_model: Model name for audio emotion analysis
        """
        self.transcriber = Transcriber(model_name=whisper_model)
        self.text_analyzer = TextSentimentAnalyzer(model_name=sentiment_model)
        self.audio_analyzer = AudioEmotionAnalyzer(model_name=emotion_model)
        self.visualizer = Visualizer()

    def analyze(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        generate_plots: bool = True,
        save_plots: bool = False
    ) -> AnalysisResults:
        """
        Analyze a video for sentiment and emotion.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save outputs (if None, uses a temp directory)
            generate_plots: Whether to generate visualization plots
            save_plots: Whether to save plots to disk

        Returns:
            AnalysisResults: Results of the analysis
        """
        # Create output directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio(video_path, output_dir)

        # Transcribe audio to text
        print("Transcribing audio to text...")
        full_text, segments = self.transcriber.transcribe_audio(str(audio_path))

        # Analyze text sentiment
        print("Analyzing text sentiment...")
        text_sentiment_results = self.text_analyzer.analyze_segments(segments)

        # Analyze audio emotion
        print("Analyzing audio emotion...")
        audio_emotion_results = self.audio_analyzer.analyze_audio_emotion(str(audio_path))

        # Fuse results
        print("Fusing results...")
        timeline_data = self.visualizer.fuse_results(
            text_sentiment_results,
            audio_emotion_results,
            segments
        )

        # Calculate average scores
        avg_positive = sum(s["positive"] for _, s in text_sentiment_results) / len(text_sentiment_results)
        avg_negative = sum(s["negative"] for _, s in text_sentiment_results) / len(text_sentiment_results)

        # For audio, we need to get emotion categories first
        emotion_categories = list(audio_emotion_results[0][1].keys())
        avg_emotions = {}
        for category in emotion_categories:
            avg_emotions[category] = sum(e[1].get(category, 0.0) for e in audio_emotion_results) / len(audio_emotion_results)

        # Prepare the result object
        results = AnalysisResults(
            text_scores={"positive": avg_positive, "negative": avg_negative},
            audio_scores=avg_emotions,
            timeline_data=timeline_data
        )

        # Generate plots if requested
        if generate_plots:
            print("Generating plots...")
            # Timeline plot
            if save_plots:
                timeline_path = os.path.join(output_dir, "timeline_plot.png")
                self.visualizer.plot_sentiment_vs_emotion(timeline_data, timeline_path)
                results.timeline_plot_bytes = None
            else:
                results.timeline_plot_bytes = self.visualizer.plot_sentiment_vs_emotion(timeline_data)

            # Distribution plot
            if save_plots:
                dist_path = os.path.join(output_dir, "distribution_plot.png")
                self.visualizer.plot_emotion_distribution(timeline_data, dist_path)
                results.dist_plot_bytes = None
            else:
                results.dist_plot_bytes = self.visualizer.plot_emotion_distribution(timeline_data)

        return results


def analyze(video_path: str, output_dir: Optional[str] = None) -> AnalysisResults:
    """
    Analyze a video file for sentiment and emotion.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save results (if None, uses a temp directory)

    Returns:
        AnalysisResults: Results of the analysis
    """
    pipeline = Pipeline()
    return pipeline.analyze(video_path, output_dir, save_plots=True)
