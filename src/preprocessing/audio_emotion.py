"""Module for analyzing emotion in audio using wav2vec2."""
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Dict, List, Tuple
import numpy as np

class AudioEmotionAnalyzer:
    """Analyzes emotion in audio files using a wav2vec2-based model."""

    def __init__(self, model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        """
        Initialize the audio emotion analyzer.

        Args:
            model_name (str): The pre-trained model to use for emotion recognition
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.id2label = self.model.config.id2label

    def analyze_segment(self, audio_array: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Analyze the emotion in an audio segment.

        Args:
            audio_array (np.ndarray): Audio segment data
            sampling_rate (int): Audio sampling rate

        Returns:
            Dict[str, float]: Emotion scores for the segment
        """
        # Prepare the audio for the model
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).to(self.device)

        # Get the model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

        # Create dictionary of emotion scores
        emotion_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return emotion_scores

    def analyze_audio_emotion(self, audio_path: str, segment_duration: float = 3.0) -> List[Tuple[float, Dict[str, float]]]:
        """
        Analyze emotion in an audio file, splitting it into segments.

        Args:
            audio_path (str): Path to the audio file
            segment_duration (float): Duration of each segment in seconds

        Returns:
            List[Tuple[float, Dict[str, float]]]: List of (timestamp, emotion_scores) tuples
        """
        # Load the audio file
        audio_array, sampling_rate = sf.read(audio_path)

        # Calculate the number of samples per segment
        samples_per_segment = int(segment_duration * sampling_rate)

        # Split the audio into segments and analyze each
        results = []
        for i in range(0, len(audio_array), samples_per_segment):
            segment = audio_array[i:i + samples_per_segment]
            if len(segment) < sampling_rate:  # Skip segments shorter than 1 second
                continue

            # Calculate timestamp for the middle of the segment
            timestamp = (i + len(segment) // 2) / sampling_rate

            # Analyze the segment
            emotion_scores = self.analyze_segment(segment, sampling_rate)

            results.append((timestamp, emotion_scores))

        return results
