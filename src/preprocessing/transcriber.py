"""Module for transcribing audio to text using Whisper."""
import whisper
from pathlib import Path
from typing import Dict, Tuple, List

class Transcriber:
    """Class for transcribing audio files to text with timestamps."""

    def __init__(self, model_name: str = "base"):
        """
        Initialize the Whisper transcriber.

        Args:
            model_name (str): The Whisper model to use: "tiny", "base", "small", "medium", "large"
        """
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_path: str) -> Tuple[str, List[Dict]]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Tuple[str, List[Dict]]: Full text transcript and segments with timestamps
        """
        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)

        # Transcribe the audio
        result = self.model.transcribe(audio)

        full_text = result["text"]
        segments = result["segments"]

        # Extract segments with timestamps
        timestamped_segments = [
            {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            }
            for segment in segments
        ]

        return full_text, timestamped_segments
