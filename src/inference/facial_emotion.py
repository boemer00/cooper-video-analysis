"""Module for analyzing facial emotions in video frames."""
import os
import cv2
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FacialEmotionAnalyzer:
    """Analyzes facial emotions in video frames using OpenCV."""

    def __init__(self):
        """Initialize the facial emotion analyzer."""
        self.emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Path to OpenCV's Haar cascade for face detection
        opencv_dir = os.path.dirname(cv2.__file__)
        self.face_cascade_path = os.path.join(opencv_dir, 'data', 'haarcascade_frontalface_default.xml')

        if not os.path.exists(self.face_cascade_path):
            # Fallback to the common location
            self.face_cascade_path = '/usr/local/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml'
            if not os.path.exists(self.face_cascade_path):
                logger.warning("Could not find OpenCV's face cascade file. Using a simplified approach.")
                self.face_cascade_path = None

        if self.face_cascade_path:
            self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
            logger.info(f"Using face detection cascade: {self.face_cascade_path}")
        else:
            self.face_cascade = None
            logger.warning("Face detection is disabled.")

        logger.info("Initialized simplified facial emotion analyzer")

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

    def _detect_faces(self, frame):
        """Detect faces in a frame using OpenCV's Haar cascade."""
        if self.face_cascade is None:
            # Just use a center region of the frame if no face detection is available
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            size = min(width, height) // 3
            return [(center_x - size//2, center_y - size//2, size, size)]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            # Fallback to center region if no faces detected
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            size = min(width, height) // 3
            return [(center_x - size//2, center_y - size//2, size, size)]

        return faces

    def _simple_emotion_analysis(self, frame) -> Dict[str, float]:
        """
        A simplified approach to emotion analysis using color and brightness.
        This is a fallback when more sophisticated models aren't available.
        """
        # Detect faces
        faces = self._detect_faces(frame)

        # Initialize emotion dictionary with default values
        emotions = {
            "neutral": 0.3,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "surprise": 0.0
        }

        # If no faces detected, return neutral
        if len(faces) == 0:
            emotions["neutral"] = 1.0
            return emotions

        face_region = None
        # Use the largest face
        if len(faces) > 1:
            largest_area = 0
            largest_face = None
            for (x, y, w, h) in faces:
                if w * h > largest_area:
                    largest_area = w * h
                    largest_face = (x, y, w, h)
            face_region = largest_face
        else:
            face_region = faces[0]

        if face_region:
            x, y, w, h = face_region
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(width - x, w)
            h = min(height - y, h)

            # Extract face region
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                emotions["neutral"] = 1.0
                return emotions

            # Convert to HSV for better color analysis
            hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

            # Extract features
            brightness = np.mean(hsv_face[:,:,2]) / 255.0  # Value channel
            saturation = np.mean(hsv_face[:,:,1]) / 255.0  # Saturation channel

            # Very basic emotion heuristics based on brightness and saturation
            # High brightness often correlates with happiness/surprise
            # Low brightness often correlates with sadness/fear
            # High saturation often correlates with stronger emotions

            # Happy tends to be bright
            if brightness > 0.6:
                emotions["happy"] += 0.3
                emotions["neutral"] -= 0.1

            # Sad tends to be darker
            if brightness < 0.4:
                emotions["sad"] += 0.3
                emotions["neutral"] -= 0.1

            # Surprise tends to have high contrast
            if saturation > 0.6:
                emotions["surprise"] += 0.2
                emotions["neutral"] -= 0.1

            # Angry and disgust tend to have medium-high saturation
            if 0.4 < saturation < 0.6:
                emotions["angry"] += 0.2
                emotions["disgust"] += 0.1
                emotions["neutral"] -= 0.1

            # Fear tends to have low saturation
            if saturation < 0.3:
                emotions["fear"] += 0.2
                emotions["neutral"] -= 0.1

            # Ensure all values are positive
            for emotion in emotions:
                emotions[emotion] = max(0.05, emotions[emotion])

            # Normalize to sum to 1
            total = sum(emotions.values())
            for emotion in emotions:
                emotions[emotion] /= total

            return emotions

        # Default neutral emotion if no faces detected
        emotions["neutral"] = 1.0
        return emotions

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
                emotions = self._simple_emotion_analysis(frame)
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
