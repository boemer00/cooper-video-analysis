# Cooper Video Analysis with AssemblyAI

This branch enhances the Cooper Video Analysis project with AssemblyAI integration for improved audio emotion detection and transcription.

## Features

- **Accurate Transcription**: Uses AssemblyAI's state-of-the-art model for accurate speech-to-text
- **Improved Emotion Detection**: Better emotion classification from speech
- **Speaker Identification**: Automatically identifies different speakers in the video
- **Entity Detection**: Recognizes entities mentioned in the speech (people, places, etc.)
- **Auto Chapters**: Automatically detects topic changes and segments the content

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API Key

You need an AssemblyAI API key to use this feature. You can get a free key with 5 hours of free processing per month at [AssemblyAI](https://www.assemblyai.com/).

Create a `.env` file in the project root:

```bash
# Copy the example env file
cp .env.example .env

# Edit the .env file with your API key
nano .env
```

Your `.env` file should look like:

```
ASSEMBLYAI_API_KEY=your_api_key_here
```

### 3. Run the Analysis

```bash
python main_assemblyai.py /path/to/your/video.mp4
```

This will:
1. Extract audio from the video
2. Upload it to AssemblyAI for processing
3. Analyze the transcript for sentiment
4. Map sentiment to emotion categories
5. Generate visualization plots

## Options

```
usage: main_assemblyai.py [-h] [--output-dir OUTPUT_DIR] [--api-key API_KEY] video_path

positional arguments:
  video_path            Path to the video file to analyze

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save the analysis results (default: ./output_assemblyai)
  --api-key API_KEY, -k API_KEY
                        AssemblyAI API key (if not provided, uses ASSEMBLYAI_API_KEY from .env)
```

## Results

The analysis produces:

1. Text sentiment scores (positive/negative)
2. Audio emotion scores (happy, sad, angry, neutral)
3. Timeline visualization plot
4. Distribution visualization plot

## Comparison with Local Models

The AssemblyAI integration provides several advantages over the local models:

1. **Higher Accuracy**: Commercial-grade models trained on massive datasets
2. **More Features**: Speaker identification, entity detection, chapter segmentation
3. **Better Differentiation**: Clearer distinction between emotional states
4. **Regular Updates**: AssemblyAI continuously improves its models

## Limitations

- Requires internet connectivity
- Processing large files may take longer (but is more accurate)
- Free tier is limited to 5 hours per month
