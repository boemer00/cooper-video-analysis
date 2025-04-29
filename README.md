# Cooper Video Analysis - Streamlit App

> **Demo Implementation of Cooper's Emotional Intelligence Platform**

A Streamlit-powered web application for analyzing sentiment and emotion in videos. This demo showcases core elements of Cooper's emotional intelligence capabilities, focusing on audiovisual content analysis as a foundation for our broader vision.

## Company Vision

**Cooper is building the emotional intelligence infrastructure for the AI economy — decoding human emotion across all media formats, and powering the next generation of marketing, customer experience, and entertainment. In a world where AI creates most content, emotional resonance becomes the new battleground — and Cooper is the infrastructure layer making it possible.**

Our technology analyzes multimodal content to extract emotional signals and patterns, providing actionable insights across industries. This demo application represents an initial implementation focused on video analysis, laying groundwork for our comprehensive emotional intelligence infrastructure.

## Future Infrastructure

This demo represents just the beginning. Cooper's future infrastructure will include:

1. **Emotional Intelligence APIs & SDKs**
   - Integration capabilities for AI apps, platforms, marketing tools, and entertainment products
   - Examples: AI video editors suggesting emotion-optimized edits, chatbot platforms with emotionally-tuned responses

2. **Emotional Data Graph & Repository**
   - Building the world's largest emotional intelligence database
   - Cross-referencing emotional reactions across demographics, cultures, and regions
   - Creating the essential reference layer for more human-like AI systems

3. **Real-Time Emotional Sentiment Feeds**
   - Live emotional heatmaps across social platforms, entertainment, and commerce
   - Enabling dynamic optimization for marketers, studios, and brands

4. **Emotional Content Generation Tools**
   - Pre-validation of content for emotional impact
   - Generation of emotionally-tuned scripts, ads, and social content

5. **Emotional Intelligence Standards and Certification**
   - Establishing industry benchmarks for emotional measurement
   - Providing certification ("Powered by Cooper Emotional Standards") for AI tools

## Current Demo Features

- **Multimodal Analysis**: Process video content to extract emotional intelligence from both audio and visual components
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

## Market Applications

This technology can be applied across multiple industries:

- **Marketing**: Optimize campaign effectiveness and measure emotional brand impact
- **Entertainment**: Improve audience engagement in movies, games, and streaming content
- **Customer Experience**: Gauge emotional responses to products, services, and support interactions
- **Content Creation**: Help creators understand and enhance emotional impact
- **Education**: Create more emotionally engaging learning experiences
- **AI Development**: Train more emotionally intelligent AI models and systems

## Limitations

- **File Size**: The app may struggle with very large video files
- **Processing Time**: Analysis can take time, especially with longer videos


## Development Roadmap

This demo represents our first step. Upcoming developments include:

- Enhanced multimodal analysis combining audio, visual, and textual signals
- Integration capabilities for third-party platforms via APIs
- Expanded emotional intelligence database across demographics
- Real-time processing capabilities for live content analysis
- Emotional content generation and pre-validation tools
- Standards development for emotional intelligence measurement


## License

2025 Cooper Ltd. All Rights Reserved.

This code is proprietary. No part of this repository may be
copied, distributed, or used in any form or by any means,
without the express prior written permission of Cooper Ltd.

## Setup and Installation

### Requirements
- Python 3.12 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd cooper-video-analysis
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK resources:
   ```bash
   python setup_nltk.py
   ```
   This step is **essential** as the application uses TextBlob and NRCLex which require specific NLTK corpora.


### Running the Application

Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

## Troubleshooting

If you encounter a `MissingCorpusError` or issues related to NLTK resources, run the setup script:
```bash
python setup_nltk.py
```

This will download all required NLTK resources for TextBlob and other NLP components.
