"""Module for extracting audio from video files."""
import os
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_dir: str = None, sampling_rate: int = 16000) -> Path:
    """
    Extract the audio track from a video file.

    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save the extracted audio.
                                   If None, saves in the same directory as the video.
        sampling_rate (int, optional): Sampling rate for the extracted audio. Default is 16kHz,
                                     which is required for most speech emotion recognition models.

    Returns:
        Path: Path to the extracted audio file
    """
    try:
        # Convert to Path object
        video_path = Path(video_path)

        # Verify video file exists
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting audio from video: {video_path}")

        # Get video filename without extension
        video_filename = video_path.stem

        # Set output directory
        if output_dir is None:
            output_dir = video_path.parent
        else:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Audio will be saved to: {output_dir}")

        # Set output audio path - ensure it's unique
        audio_path = output_dir / f"{video_filename}.wav"

        # Create a unique filename if needed
        counter = 1
        while audio_path.exists():
            audio_path = output_dir / f"{video_filename}_{counter}.wav"
            counter += 1

        # Extract audio with a timeout (helpful for Streamlit)
        logger.info(f"Loading video: {video_path}")

        # Load the video file
        with VideoFileClip(str(video_path)) as video:
            # Check if video has audio
            if video.audio is None:
                logger.error(f"Video has no audio track: {video_path}")
                raise ValueError(f"Video has no audio track: {video_path}")

            logger.info(f"Video loaded, duration: {video.duration:.2f} seconds")
            audio = video.audio

            # Write audio with specific parameters:
            # - mono audio (1 channel) with ffmpeg_params=["-ac", "1"]
            # - sampling rate at 16kHz with fps=sampling_rate
            # - PCM 16-bit audio with codec='pcm_s16le'
            logger.info(f"Writing audio to {audio_path}")
            audio.write_audiofile(
                str(audio_path),
                codec='pcm_s16le',
                fps=sampling_rate,  # Set to 16kHz for emotion recognition models
                ffmpeg_params=["-ac", "1"],  # Mono audio (1 channel)
                verbose=False,  # Reduce console output
                logger=None  # Disable moviepy logging
            )

            logger.info(f"Audio extraction complete: {audio_path}")

        # Verify the audio file was created
        if not audio_path.exists():
            logger.error(f"Failed to create audio file: {audio_path}")
            raise FileNotFoundError(f"Failed to create audio file: {audio_path}")

        # Log the file size
        audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"Audio file size: {audio_size_mb:.2f} MB")

        return audio_path

    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error extracting audio from video: {str(e)}")
