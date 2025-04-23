# Cooper Video Analysis - Streamlit App

A Streamlit-powered web application for analyzing sentiment and emotion in videos using Cooper Video Analysis with AssemblyAI integration.

## Features

- **Web Interface**: Easy-to-use Streamlit interface for video analysis
- **AssemblyAI Integration**:
  - **Accurate Transcription**: State-of-the-art model for speech-to-text
  - **Improved Emotion Detection**: Better classification from speech
  - **Speaker Identification**: Automatically identifies different speakers
  - **Entity Detection**: Recognizes people, places, and other entities
  - **Auto Chapters**: Automatically detects topic changes
- **Visual Results**: Interactive Plotly visualizations of sentiment and emotion scores
- **Debug Mode**: Toggle debug information for troubleshooting
- **Download Results**: Save analysis results for further use

## Project Structure

```
cooper-video-analysis/
├── api/
│   └── analyze.py        # FastAPI serverless endpoint
├── src/
│   ├── preprocessing/    # Audio extraction and processing
│   ├── inference/        # Sentiment analysis
│   ├── visualization/    # Visualization components
│   ├── pipeline.py       # Standard pipeline
│   └── pipeline_assemblyai.py  # AssemblyAI pipeline
├── streamlit_app.py      # Streamlit web interface
├── main.py               # CLI for standard pipeline
├── main_assemblyai.py    # CLI for AssemblyAI pipeline
└── requirements.txt      # Dependencies
```

## Local Setup

### Prerequisites

- Python 3.12.9
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cooper-video-analysis.git
cd cooper-video-analysis
```

2. Create a virtual environment (recommended):

```bash
# Using pyenv
pyenv install 3.12.9
pyenv virtualenv 3.12.9 coop
pyenv activate coop

# Or using standard venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your AssemblyAI API key:

```bash
# Copy the example env file if it exists
cp .env.example .env

# Or create a new .env file
echo "ASSEMBLYAI_API_KEY=your_api_key_here" > .env
```

You can get a free API key with 5 hours of free processing per month at [AssemblyAI](https://www.assemblyai.com/).

### Running the App Locally

```bash
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501

### Command Line Usage

For command line usage with AssemblyAI:

```bash
python main_assemblyai.py /path/to/your/video.mp4 --output-dir ./results
```

Options:
```
usage: main_assemblyai.py [-h] [--output-dir OUTPUT_DIR] [--api-key API_KEY] video_path

positional arguments:
  video_path            Path to the video file to analyze

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save results (default: ./output_assemblyai)
  --api-key API_KEY, -k API_KEY
                        AssemblyAI API key (if not in .env file)
```

## Deployment to Streamlit Cloud

### Option 1: Deploy from GitHub

1. Push your code to GitHub:

```bash
git add .
git commit -m "Streamlit app ready for deployment"
git push
```

2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Click "New app", select your repository, and enter:
   - Repository: `yourusername/cooper-video-analysis`
   - Branch: `main` (or your preferred branch)
   - Main file path: `streamlit_app.py`
   - If using a specialized requirements file: `requirements_streamlit.txt`

4. Add your AssemblyAI API key as a secret in the Streamlit Cloud settings:
   - Go to your app's settings
   - Scroll to "Secrets"
   - Add a new secret with the name `ASSEMBLYAI_API_KEY` and your API key as the value

## Using the App

1. Enter your AssemblyAI API key if not already configured
2. Upload a video file (supported formats: mp4, mov, avi, mkv)
3. Click "Analyze"
4. View the results with interactive visualizations:
   - Timeline analysis showing emotion and sentiment over time
   - Distribution analysis showing overall scores

## Limitations

- **File Size**: The app may struggle with very large video files
- **Processing Time**: Analysis can take time, especially with longer videos
- **AssemblyAI API**: Requires internet connectivity; free tier limited to 5 hours per month

## Troubleshooting

If you encounter issues:

1. Enable "Show Debug Information" in the sidebar
2. Check the logs for detailed error messages
3. Ensure your video file is in a supported format
4. Verify your AssemblyAI API key

## License

MIT

## Credits

- Built with [Streamlit](https://streamlit.io/)
- Powered by [AssemblyAI](https://www.assemblyai.com/)
