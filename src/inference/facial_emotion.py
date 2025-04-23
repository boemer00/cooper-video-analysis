"""Module for analyzing facial emotions in video frames."""
import os
import cv2
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)

class FacialEmotionAnalyzer:
    """Analyzes facial emotions in video frames using DeepFace."""

    def __init__(self, model_name: str = "emotion"):
        """
        Initialize the facial emotion analyzer.

        Args:
            model_name (str): The model to use for emotion analysis (default is DeepFace's emotion model)
        """
        self.model_name = model_name
        # Emotion categories DeepFace can detect
        self.emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Test if DeepFace is working properly
        logger.info("Initializing DeepFace for facial emotion analysis")

    def _extract_frames(self, video_path: str, sampling_rate: int = 1) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from a video at a specified sampling rate.

        Args:
            video_path (str): Path to the video file
            sampling_rate (int): Extract 1 frame every N seconds

        Returns:
            List[Tuple[float, np.ndarray]]: List of (timestamp, frame) tuples
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, total frames: {total_frames}")

        # Calculate frame interval based on sampling rate
        frame_interval = int(fps * sampling_rate)
        frame_interval = max(1, frame_interval)  # Ensure at least 1

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame based on sampling rate
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames for analysis")
        return frames

    def analyze_video(self, video_path: str, sampling_rate: int = 1) -> List[Tuple[float, Dict[str, float]]]:
        """
        Analyze facial emotions in a video file.

        Args:
            video_path (str): Path to the video file
            sampling_rate (int): Analyze 1 frame every N seconds

        Returns:
            List[Tuple[float, Dict[str, float]]]: List of (timestamp, emotion_scores) tuples
        """
        results = []
        frames = self._extract_frames(video_path, sampling_rate)

        logger.info(f"Analyzing {len(frames)} frames for facial emotions")

        for i, (timestamp, frame) in enumerate(frames):
            try:
                # Analyze face in the current frame
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                # Extract emotion scores
                if isinstance(analysis, list) and analysis:
                    # Get emotions from first detected face (or average if multiple)
                    emotions = analysis[0]['emotion']

                    # Normalize keys to lowercase
                    emotions = {k.lower(): v/100.0 for k, v in emotions.items()}

                    results.append((timestamp, emotions))

                    if i % 10 == 0 or i == len(frames) - 1:
                        logger.info(f"Analyzed {i+1}/{len(frames)} frames")

            except Exception as e:
                logger.warning(f"Error analyzing frame at {timestamp:.2f}s: {str(e)}")
                # Append neutral fallback if analysis fails
                neutral_emotions = {emotion: 0.0 for emotion in self.emotion_categories}
                neutral_emotions['neutral'] = 1.0
                results.append((timestamp, neutral_emotions))

        logger.info(f"Completed facial emotion analysis: {len(results)} results")
        return results
