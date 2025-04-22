# Cooper Video Analysis

A modular Python proof-of-concept application for analyzing sentiment and emotion in videos.

## Features

- Extracts audio tracks from TikTok-style videos
- Transcribes speech to text using OpenAI's Whisper model
- Analyzes transcript sentiment using DistilBERT
- Analyzes voice emotion using wav2vec2
- Produces comparative visualizations of text sentiment vs. voice emotion
- Provides a FastAPI serverless endpoint for Vercel deployment

## Project Structure

```
cooper-video-analysis/
├── api/
│   └── analyze.py        # FastAPI serverless endpoint
├── src/
│   ├── preprocessing/
│   │   ├── audio_extractor.py    # Extract audio from video
│   │   ├── transcriber.py        # Transcribe audio to text
│   │   └── audio_emotion.py      # Analyze audio for emotion
│   ├── inference/
│   │   └── text_sentiment.py     # Analyze text for sentiment
│   ├── visualization/
│   │   └── visualizer.py         # Create visualizations
│   └── pipeline.py               # Main pipeline
├── main.py                       # CLI entry point
├── vercel.json                   # Vercel configuration
└── requirements.txt              # Dependencies
```

## Installation

### Environment Setup

This project requires Python 3.12.9, which you can install using pyenv:

```bash
pyenv install 3.12.9
pyenv virtualenv 3.12.9 coop
pyenv activate coop
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python main.py path/to/video.mp4 --output-dir ./results
```

This will:
1. Extract audio from the video
2. Transcribe the audio to text
3. Analyze sentiment in the text
4. Analyze emotion in the audio
5. Generate comparison visualizations
6. Save the results to the specified output directory

### API

The application includes a FastAPI serverless function for deployment on Vercel:

```bash
# Local testing
uvicorn api.analyze:app --reload

# Deploy to Vercel
vercel
```

The API has two endpoints:
- `GET /`: Health check
- `POST /analyze`: Upload a video file for analysis

## API Response Format

```json
{
  "text_sentiment": {
    "positive": 0.65,
    "negative": 0.35
  },
  "voice_emotion": {
    "happy": 0.75,
    "sad": 0.05,
    "angry": 0.10,
    "neutral": 0.10
  },
  "plots": {
    "timeline": "base64-encoded PNG",
    "distribution": "base64-encoded PNG"
  }
}
```

## Models

- **Transcription**: Whisper (base) from OpenAI
- **Text Sentiment**: DistilBERT-based classifier from Hugging Face
- **Voice Emotion**: wav2vec2-based Speech Emotion Recognition model

## Development

### Running Tests

```bash
# TODO: Add tests
```

### Local Development

1. Clone the repository
2. Set up the environment as described above
3. Run the CLI script for local testing

### Vercel Deployment

For serverless API deployment:

```bash
vercel
```

## Limitations

- The serverless function has a 60-second timeout, which may not be sufficient for long videos
- Larger models may exceed memory limits in serverless environments
- Models are downloaded at runtime in the serverless function, which can cause cold start delays

## License

MIT
