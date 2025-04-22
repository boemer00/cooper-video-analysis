#!/usr/bin/env python3
"""
Cooper Video Analysis - Video sentiment and emotion analysis tool.

This script analyzes a video file for sentiment and emotion.
"""
import argparse
import os
import sys
from pathlib import Path

from src.pipeline import analyze

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Analyze video for sentiment and emotion"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to analyze"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save the analysis results (default: ./output)"
    )

    args = parser.parse_args()

    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    try:
        print(f"Analyzing video: {video_path}")
        results = analyze(str(video_path), str(output_dir))

        # Print summary of results
        print("\nAnalysis complete!")
        print("\nText Sentiment:")
        for sentiment, score in results.text_scores.items():
            print(f"  {sentiment.capitalize()}: {score:.4f}")

        print("\nAudio Emotion:")
        for emotion, score in results.audio_scores.items():
            print(f"  {emotion.capitalize()}: {score:.4f}")

        print(f"\nPlots saved to {output_dir}")
        print(f"  - Timeline: {output_dir}/timeline_plot.png")
        print(f"  - Distribution: {output_dir}/distribution_plot.png")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
