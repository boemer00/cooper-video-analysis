"""Module for extracting audio from video files."""
import os
from pathlib import Path
from moviepy.editor import VideoFileClip

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
    # Get video filename without extension
    video_path = Path(video_path)
    video_filename = video_path.stem

    # Set output directory
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Set output audio path
    audio_path = output_dir / f"{video_filename}.wav"

    # Extract audio
    video = VideoFileClip(str(video_path))
    audio = video.audio

    # Write audio with specific parameters:
    # - mono audio (1 channel) with ffmpeg_params=["-ac", "1"]
    # - sampling rate at 16kHz with fps=sampling_rate
    # - PCM 16-bit audio with codec='pcm_s16le'
    audio.write_audiofile(
        str(audio_path),
        codec='pcm_s16le',
        fps=sampling_rate,  # Set to 16kHz for emotion recognition models
        ffmpeg_params=["-ac", "1"]  # Mono audio (1 channel)
    )

    # Close the video file
    video.close()

    return audio_path
