# Cooper Video Analysis - Streamlit Deployment Guide

This guide explains how to deploy the Cooper Video Analysis tool to Streamlit Cloud.

## Files for Deployment

The following files are specifically optimized for Streamlit Cloud deployment:

- `streamlit_app.py` - The main Streamlit application
- `requirements_streamlit.txt` - Minimal dependencies for cloud deployment

## Deployment Steps

### 1. Test Locally First

```bash
# Install the dependencies
pip install -r requirements_streamlit.txt

# Run the specialized Streamlit app
streamlit run streamlit_app.py
```

### 2. Create a GitHub Repository

Push your code to a GitHub repository if you haven't already.

### 3. Deploy to Streamlit Cloud

1. Visit [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Under "Repository," select your GitHub repository
5. Under "Branch," select the branch with your code
6. Under "Main file path," enter `streamlit_app.py`
7. Under "Requirements," enter `requirements_streamlit.txt`
8. Click "Deploy"

### 4. Add Secrets

Add your AssemblyAI API key as a secret in the Streamlit Cloud settings:

1. Go to your app's settings
2. Scroll to "Secrets"
3. Add a new secret with the name `ASSEMBLYAI_API_KEY` and your API key as the value

## Troubleshooting

If you encounter any issues with the deployment:

1. Enable "Show Debug Information" in the app
2. Check the Streamlit Cloud logs in the app settings
3. Test that your AssemblyAI API key works by running `test_assemblyai.py` locally
4. Make sure the API key is correctly set in the Streamlit Cloud secrets

## Important Notes

- This version uses a simplified implementation that relies solely on AssemblyAI
- The app creates files in a temporary directory on Streamlit Cloud
- Log files can help diagnose issues
