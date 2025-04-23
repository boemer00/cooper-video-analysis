#!/usr/bin/env python3
"""
Cooper Video Analysis with AssemblyAI - Enhanced video sentiment and emotion analysis.

This script analyzes a video file for sentiment and emotion using AssemblyAI API.
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.pipeline_assemblyai import analyze_with_assemblyai

def main():
    """Main entry point for the AssemblyAI powered application."""
    parser = argparse.ArgumentParser(
        description="Analyze video for sentiment and emotion using AssemblyAI"
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
        default="output_assemblyai",
        help="Directory to save the analysis results (default: ./output_assemblyai)"
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        help="AssemblyAI API key (if not provided, uses ASSEMBLYAI_API_KEY from .env)"
    )
    parser.add_argument(
        "--facial-sampling-rate",
        "-f",
        type=int,
        default=1,
        help="Sample 1 frame every N seconds for facial emotion analysis (default: 1)"
    )

    args = parser.parse_args()

    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Check for API key in args or environment
    api_key = args.api_key or os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("Error: AssemblyAI API key not provided.")
        print("Either use --api-key option or set ASSEMBLYAI_API_KEY in .env file.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    try:
        print(f"Analyzing video with AssemblyAI: {video_path}")
        print(f"Facial sampling rate: {args.facial_sampling_rate} second(s)")
        results = analyze_with_assemblyai(
            str(video_path),
            str(output_dir),
            api_key=api_key,
            facial_sampling_rate=args.facial_sampling_rate
        )

        # Print summary of results
        print("\nAnalysis complete!")
        print("\nText Sentiment:")
        for sentiment, score in results.text_scores.items():
            print(f"  {sentiment.capitalize()}: {score:.4f}")

        print("\nAudio Emotion:")
        for emotion, score in results.audio_scores.items():
            print(f"  {emotion.capitalize()}: {score:.4f}")

        print("\nFacial Emotion:")
        for emotion, score in results.facial_scores.items():
            print(f"  {emotion.capitalize()}: {score:.4f}")

        print(f"\nPlots saved to {output_dir}")
        print(f"  - Timeline: {output_dir}/timeline_plot.png")
        print(f"  - Distribution: {output_dir}/distribution_plot.png")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
