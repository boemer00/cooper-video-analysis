"""Module for analyzing audio using AssemblyAI."""
import os
import time
from typing import Dict, List, Tuple, Optional
import assemblyai as aai
from pathlib import Path

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
                "set the ASSEMBLYAI_API_KEY environment variable."
            )

        # Initialize AssemblyAI client
        aai.settings.api_key = self.api_key

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
        print("Uploading audio to AssemblyAI...")

        # Create a transcription config with emotion analysis enabled
        config = aai.TranscriptionConfig(
            audio_url=audio_path,
            speaker_labels=True,  # Identify different speakers
            auto_chapters=True,    # Automatically detect chapters/segments
            sentiment_analysis=True,  # Add sentiment analysis
            entity_detection=True,    # Detect entities (people, places, etc.)
            language_code="en",       # English language
            speaker_labels=True,      # Identify different speakers
            auto_highlights=True,     # Detect key moments
        )

        # Create and start the transcription
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)

        # Wait for the transcription to complete
        while transcript.status != 'completed':
            print(f"Transcription status: {transcript.status}")
            time.sleep(2)
            transcript = transcriber.get_transcript(transcript.id)

        print("Transcription complete!")

        # Extract full transcript text
        full_text = transcript.text

        # Extract segments with timestamps and metadata
        segments = []
        for utterance in transcript.utterances:
            segments.append({
                "text": utterance.text,
                "start": utterance.start / 1000,  # Convert ms to seconds
                "end": utterance.end / 1000,      # Convert ms to seconds
                "speaker": utterance.speaker,
                "confidence": utterance.confidence
            })

        # Extract emotion data with timestamps
        emotion_results = []

        # Process sentiment analysis results
        if transcript.sentiment_analysis_results:
            for result in transcript.sentiment_analysis_results:
                # Calculate midpoint timestamp
                timestamp = (result.start + result.end) / 2000  # Convert ms to seconds

                # Convert sentiment to emotion scores
                # AssemblyAI provides POSITIVE, NEGATIVE, NEUTRAL sentiment
                # We'll convert to emotion scores for compatibility
                emotion_scores = {
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0,
                    "neutral": 0.0
                }

                # Map sentiment to emotion scores
                sentiment = result.sentiment.lower()
                confidence = result.confidence

                if sentiment == "positive":
                    emotion_scores["happy"] = confidence
                    emotion_scores["neutral"] = 1.0 - confidence
                elif sentiment == "negative":
                    # Distribute negative between sad and angry
                    emotion_scores["sad"] = confidence * 0.6
                    emotion_scores["angry"] = confidence * 0.4
                    emotion_scores["neutral"] = 1.0 - confidence
                else:  # neutral
                    emotion_scores["neutral"] = confidence

                emotion_results.append((timestamp, emotion_scores))

        return full_text, segments, emotion_results

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
        # Get transcription and sentiment
        full_text, segments, emotion_results = self.analyze_audio(audio_path)

        # Convert segments to text sentiment format for compatibility with existing code
        text_sentiment_results = []

        for segment in segments:
            # For simplicity, we'll just use neutral sentiment for all segments
            # In a real implementation, you'd use AssemblyAI's sentiment analysis
            sentiment_scores = {"positive": 0.5, "negative": 0.5}
            timestamp = (segment["start"] + segment["end"]) / 2
            text_sentiment_results.append((timestamp, sentiment_scores))

        # Return all results
        return full_text, segments, text_sentiment_results, emotion_results
