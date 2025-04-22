# Cooper Video Analysis - Streamlit App

A Streamlit-powered web application for analyzing sentiment and emotion in videos using Cooper Video Analysis.

## Features

- **Web Interface**: Easy-to-use interface for video analysis
- **Multiple Models**: Choose between local Wav2Vec2 model or AssemblyAI API
- **Visual Results**: View sentiment and emotion scores with clear visualizations
- **Debug Mode**: Toggle debug information for troubleshooting
- **Download Results**: Save analysis results for further use

## Screenshot

![Cooper Video Analysis App](static/assets/video_upload.png)

## Local Setup

### Prerequisites

- Python 3.12.9
- pip

### Installation

1. Clone the repository and switch to the streamlit branch:

```bash
git clone https://github.com/yourusername/cooper-video-analysis.git
cd cooper-video-analysis
git checkout streamlit
```

2. Create a virtual environment (recommended):

```bash
# Using pyenv
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

4. If using AssemblyAI, create a `.env` file with your API key:

```bash
echo "ASSEMBLYAI_API_KEY=your_api_key_here" > .env
```

### Running the App Locally

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## Deployment to Streamlit Cloud

### Option 1: Deploy from GitHub

1. Push your code to GitHub:

```bash
git add .
git commit -m "Streamlit app ready for deployment"
git push -u origin streamlit
```

2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Click "New app", select your repository, and enter:
   - Repository: `yourusername/cooper-video-analysis`
   - Branch: `streamlit`
   - Main file path: `app.py`

4. Add your AssemblyAI API key as a secret in the Streamlit Cloud settings.

### Option 2: Deploy with Streamlit CLI

If you have Streamlit CLI installed:

```bash
streamlit deploy
```

Follow the prompts to deploy your app.

## Using the App

1. Select the model you want to use (Standard or AssemblyAI)
2. If using AssemblyAI, enter your API key if not already configured
3. Enter an output folder name
4. Upload a video file
5. Click "Analyze Video"
6. View the results and download if needed

## Limitations

- **File Size**: The app may struggle with very large video files
- **Processing Time**: Analysis can take time, especially with longer videos
- **Model Accuracy**: Results depend on the quality of the audio in the video
- **AssemblyAI API**: Requires an API key and internet connection

## Troubleshooting

If you encounter issues:

1. Enable "Show Debug Information" in the sidebar
2. Check the logs for detailed error messages
3. Ensure your video file is in a supported format
4. Verify your AssemblyAI API key if using that model

## License

MIT

## Credits

- Image from [Freepik](https://www.freepik.com)
- Built with [Streamlit](https://streamlit.io/)
- Powered by [AssemblyAI](https://www.assemblyai.com/) and [Hugging Face](https://huggingface.co/)
