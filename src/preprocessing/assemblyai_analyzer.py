"""Module for analyzing audio using AssemblyAI."""
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
import assemblyai as aai
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class AssemblyAIAnalyzer:
    """Analyzes audio files using AssemblyAI API for transcription and emotion."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AssemblyAI analyzer.

        Args:
            api_key (str, optional): AssemblyAI API key. If None, will look for
                                    ASSEMBLYAI_API_KEY environment variable.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key not provided. Either pass it as an argument or "
                "set the ASSEMBLYAI_API_KEY environment variable in .env file."
            )

        # Validate API key format
        if len(self.api_key) < 10:
            raise ValueError("Invalid AssemblyAI API key format")

        # Initialize AssemblyAI client
        try:
            aai.settings.api_key = self.api_key
            logger.info("AssemblyAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AssemblyAI client: {str(e)}")
            raise RuntimeError(f"AssemblyAI client initialization failed: {str(e)}")

    def analyze_audio(self, audio_path: str) -> Tuple[str, List[Dict], List[Tuple[float, Dict[str, float]]]]:
        """
        Process audio file with AssemblyAI for transcription and emotion analysis.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Tuple containing:
                - Full text transcript (str)
                - Transcript segments with metadata (List[Dict])
                - Emotion results with timestamps (List[Tuple[float, Dict[str, float]]])
        """
        logger.info(f"Uploading audio to AssemblyAI: {audio_path}")

        # Verify that the audio file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Create the transcriber with minimal configuration
            transcriber = aai.Transcriber()

            # Log file size
            file_size = os.path.getsize(audio_path) / (1024*1024)  # In MB
            logger.info(f"Audio file size: {file_size:.2f} MB")

            # Use the default configuration and just enable sentiment analysis
            transcript = transcriber.transcribe(audio_path)

            if not transcript:
                logger.error("Transcription failed: No transcript returned")
                raise RuntimeError("Transcription failed: No transcript returned")

            logger.info("Transcription complete!")

            # Extract full transcript text
            full_text = transcript.text

            if not full_text:
                logger.warning("Transcription returned empty text")

            logger.info(f"Transcript (first 100 chars): {full_text[:100] if full_text else 'Empty'}")

            # Create simple segments based on words with timestamps
            segments = []
            current_segment = {"text": "", "start": 0, "end": 0, "speaker": "unknown", "confidence": 0}

            if hasattr(transcript, 'words') and transcript.words:
                logger.info(f"Found {len(transcript.words)} words with timestamps")
                for word in transcript.words:
                    if not current_segment["text"]:
                        current_segment["start"] = word.start / 1000  # ms to seconds

                    current_segment["text"] += f" {word.text}"
                    current_segment["end"] = word.end / 1000  # ms to seconds

                    # End segment at punctuation
                    if any(p in word.text for p in ['.', '!', '?']):
                        segments.append(current_segment)
                        current_segment = {"text": "", "start": 0, "end": 0, "speaker": "unknown", "confidence": 0}

                # Add any remaining segment
                if current_segment["text"]:
                    segments.append(current_segment)
            else:
                logger.warning("No words with timestamps found in transcript")

            # If no words were found, create a single segment with the full text
            if not segments and full_text:
                logger.info("Creating single segment with full text")
                segments = [{"text": full_text, "start": 0, "end": 0, "speaker": "unknown", "confidence": 0}]

            # Log segment count
            logger.info(f"Created {len(segments)} text segments")

            # Create emotion analysis based on positive/negative sentiment
            emotion_results = []

            for segment in segments:
                # Calculate midpoint timestamp
                timestamp = (segment["start"] + segment["end"]) / 2

                # Analyze text sentiment with a simple rule-based approach
                text = segment["text"].lower()

                # Create default emotion scores
                emotion_scores = {
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0,
                    "neutral": 1.0  # Default to neutral
                }

                # Simple keyword-based sentiment analysis
                positive_words = ["good", "great", "amazing", "excellent", "love", "wonderful", "happy", "awesome"]
                negative_words = ["bad", "terrible", "awful", "hate", "sad", "angry", "upset", "disappointing"]

                # Count positive and negative words
                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)

                # Calculate scores based on word counts
                if pos_count > 0 or neg_count > 0:
                    total = pos_count + neg_count
                    if pos_count > neg_count:
                        emotion_scores["happy"] = pos_count / total
                        emotion_scores["neutral"] = 1.0 - emotion_scores["happy"]
                    elif neg_count > pos_count:
                        # Split negative sentiment between sad and angry
                        emotion_scores["sad"] = (neg_count / total) * 0.6
                        emotion_scores["angry"] = (neg_count / total) * 0.4
                        emotion_scores["neutral"] = 1.0 - (emotion_scores["sad"] + emotion_scores["angry"])

                emotion_results.append((timestamp, emotion_scores))

            # Log emotion analysis results
            if all(er[1]["neutral"] == 1.0 for er in emotion_results):
                logger.warning("All segments classified as neutral - this might indicate an issue")
            else:
                logger.info("Emotion analysis complete with varied results")

            return full_text, segments, emotion_results

        except Exception as e:
            logger.error(f"AssemblyAI analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"AssemblyAI analysis failed: {str(e)}")

    def analyze_audio_combined(self, audio_path: str) -> Tuple[
        str,
        List[Dict],
        List[Tuple[float, Dict[str, float]]],
        List[Tuple[float, Dict[str, float]]]
    ]:
        """
        Analyze audio for both transcription/sentiment and separate emotion analysis.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Tuple containing:
                - Full text transcript (str)
                - Transcript segments with metadata (List[Dict])
                - Text sentiment results (List[Tuple[float, Dict[str, float]]])
                - Audio emotion results (List[Tuple[float, Dict[str, float]]])
        """
        try:
            # Get transcription and sentiment
            full_text, segments, emotion_results = self.analyze_audio(audio_path)

            # Text sentiment is the same as audio emotion for this implementation
            text_sentiment_results = []

            for timestamp, emotion_scores in emotion_results:
                # Convert emotion scores to sentiment scores
                sentiment_scores = {
                    "positive": emotion_scores["happy"],
                    "negative": emotion_scores["sad"] + emotion_scores["angry"]
                }

                # Normalize to ensure they sum to 1
                total = sentiment_scores["positive"] + sentiment_scores["negative"]
                if total > 0:
                    sentiment_scores["positive"] /= total
                    sentiment_scores["negative"] /= total
                else:
                    sentiment_scores["positive"] = 0.5
                    sentiment_scores["negative"] = 0.5

                text_sentiment_results.append((timestamp, sentiment_scores))

            # Log overall results
            logger.info(f"Analysis complete - Transcript length: {len(full_text)} chars, Segments: {len(segments)}")

            return full_text, segments, text_sentiment_results, emotion_results

        except Exception as e:
            logger.error(f"Combined audio analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Combined audio analysis failed: {str(e)}")
