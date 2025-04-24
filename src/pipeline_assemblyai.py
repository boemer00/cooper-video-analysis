"""Pipeline using AssemblyAI for video sentiment and emotion analysis."""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
from dataclasses import dataclass

from .preprocessing import extract_audio, AssemblyAIAnalyzer
from .visualization import Visualizer, TimelineData
from .inference import FacialEmotionAnalyzer
from .conversational_analysis import ConversationalAnalyzer, ConversationalInsights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResults:
    """Container for the results of the video analysis."""
    text_scores: Dict
    audio_scores: Dict
    facial_scores: Dict
    timeline_data: TimelineData
    conversational_insights: Optional[ConversationalInsights] = None
    timeline_plot_bytes: Optional[str] = None
    dist_plot_bytes: Optional[str] = None

class AssemblyAIPipeline:
    """Pipeline using AssemblyAI for video sentiment and emotion analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the AssemblyAI pipeline.

        Args:
            api_key: AssemblyAI API key (optional, will use env var if not provided)
        """
        logger.info("Initializing AssemblyAI pipeline")
        self.assemblyai_analyzer = AssemblyAIAnalyzer(api_key=api_key)
        self.facial_analyzer = FacialEmotionAnalyzer()
        self.visualizer = Visualizer()
        self.conversational_analyzer = ConversationalAnalyzer()

    def analyze(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        generate_plots: bool = True,
        save_plots: bool = False,
        facial_sampling_rate: int = 1
    ) -> AnalysisResults:
        """
        Analyze a video for sentiment and emotion using AssemblyAI.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save outputs (if None, uses a temp directory)
            generate_plots: Whether to generate visualization plots
            save_plots: Whether to save plots to disk
            facial_sampling_rate: Sample 1 frame every N seconds for facial analysis

        Returns:
            AnalysisResults: Results of the analysis
        """
        # Create output directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Analysis output directory: {output_dir}")

        # Extract audio from video
        logger.info("Extracting audio from video...")
        audio_path = extract_audio(video_path, output_dir)
        logger.info(f"Audio extracted to: {audio_path}")

        # Use AssemblyAI for both transcription and emotion analysis
        logger.info("Analyzing with AssemblyAI...")
        try:
            full_text, segments, text_sentiment_results, audio_emotion_results = (
                self.assemblyai_analyzer.analyze_audio_combined(str(audio_path))
            )
            logger.info(f"AssemblyAI analysis complete: {len(segments)} segments processed")
        except Exception as e:
            logger.error(f"AssemblyAI analysis failed: {str(e)}")
            # Provide minimal fallback results if analysis fails
            full_text = "Analysis failed"
            segments = [{"text": "Analysis failed", "start": 0, "end": 0}]
            text_sentiment_results = [(0, {"positive": 0.5, "negative": 0.5})]
            audio_emotion_results = [(0, {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0})]
            logger.warning("Using fallback neutral results due to analysis failure")

        # Analyze facial emotions
        logger.info("Analyzing facial emotions from video frames...")
        try:
            facial_emotion_results = self.facial_analyzer.analyze_video(
                video_path,
                sampling_rate=facial_sampling_rate
            )
            logger.info(f"Facial emotion analysis complete: {len(facial_emotion_results)} frames processed")
        except Exception as e:
            logger.error(f"Facial emotion analysis failed: {str(e)}")
            # Provide minimal fallback results if analysis fails
            facial_emotion_results = [(0, {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0})]
            logger.warning("Using fallback neutral results due to facial analysis failure")

        # Fuse results
        logger.info("Fusing results...")
        timeline_data = self.visualizer.fuse_results(
            text_sentiment_results,
            audio_emotion_results,
            segments,
            facial_emotion_results
        )

        # Calculate average scores for text sentiment
        if text_sentiment_results:
            avg_positive = sum(s["positive"] for _, s in text_sentiment_results) / len(text_sentiment_results)
            avg_negative = sum(s["negative"] for _, s in text_sentiment_results) / len(text_sentiment_results)
        else:
            # Fallback if no text sentiment results
            avg_positive = 0.5
            avg_negative = 0.5

        # Calculate average scores for audio emotion
        if audio_emotion_results:
            # Get emotion categories from first result
            emotion_categories = list(audio_emotion_results[0][1].keys())
            avg_emotions = {}
            for category in emotion_categories:
                avg_emotions[category] = sum(
                    e[1].get(category, 0.0) for e in audio_emotion_results
                ) / len(audio_emotion_results)
        else:
            # Fallback if no audio emotion results
            avg_emotions = {"neutral": 1.0}

        # Calculate average scores for facial emotion
        if facial_emotion_results:
            # Get emotion categories from first result
            facial_categories = list(facial_emotion_results[0][1].keys())
            avg_facial_emotions = {}
            for category in facial_categories:
                avg_facial_emotions[category] = sum(
                    e[1].get(category, 0.0) for e in facial_emotion_results
                ) / len(facial_emotion_results)
        else:
            # Fallback if no facial emotion results
            avg_facial_emotions = {"neutral": 1.0}

        logger.info(f"Text sentiment scores: positive={avg_positive:.2f}, negative={avg_negative:.2f}")
        logger.info(f"Audio emotion scores: {', '.join([f'{k}={v:.2f}' for k, v in avg_emotions.items()])}")
        logger.info(f"Facial emotion scores: {', '.join([f'{k}={v:.2f}' for k, v in avg_facial_emotions.items()])}")

        # Add conversational analysis
        logger.info("Performing conversational flow analysis...")
        try:
            conversational_insights = self.conversational_analyzer.analyze_transcript(full_text, segments)
            logger.info("Conversational analysis complete")
        except Exception as e:
            logger.error(f"Conversational analysis failed: {str(e)}")
            # Provide minimal fallback results if analysis fails
            conversational_insights = ConversationalInsights(
                summary="Analysis failed.",
                sarcasm_detected=[],
                slang_terms=[],
                top_adjectives=[],
                transcript_snippet=""
            )
            logger.warning("Using fallback results due to conversational analysis failure")

        # Prepare the result object
        results = AnalysisResults(
            text_scores={"positive": avg_positive, "negative": avg_negative},
            audio_scores=avg_emotions,
            facial_scores=avg_facial_emotions,
            timeline_data=timeline_data,
            conversational_insights=conversational_insights
        )

        # Generate plots if requested
        if generate_plots:
            logger.info("Generating plots...")
            # Timeline plot
            if save_plots:
                timeline_path = os.path.join(output_dir, "timeline_plot.png")
                self.visualizer.plot_sentiment_vs_emotion(timeline_data, timeline_path)
                results.timeline_plot_bytes = None
                logger.info(f"Timeline plot saved to: {timeline_path}")
            else:
                results.timeline_plot_bytes = self.visualizer.plot_sentiment_vs_emotion(timeline_data)

            # Distribution plot
            if save_plots:
                dist_path = os.path.join(output_dir, "distribution_plot.png")
                self.visualizer.plot_emotion_distribution(timeline_data, dist_path)
                results.dist_plot_bytes = None
                logger.info(f"Distribution plot saved to: {dist_path}")
            else:
                results.dist_plot_bytes = self.visualizer.plot_emotion_distribution(timeline_data)

        return results


def analyze_with_assemblyai(
    video_path: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    facial_sampling_rate: int = 1
) -> AnalysisResults:
    """
    Analyze a video file using AssemblyAI.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save results (if None, uses a temp directory)
        api_key: AssemblyAI API key (optional)
        facial_sampling_rate: Sample 1 frame every N seconds for facial analysis

    Returns:
        AnalysisResults: Results of the analysis
    """
    logger.info(f"Starting analysis of video: {video_path}")
    pipeline = AssemblyAIPipeline(api_key=api_key)
    return pipeline.analyze(
        video_path,
        output_dir,
        save_plots=True,
        facial_sampling_rate=facial_sampling_rate
    )
