"""FastAPI serverless function for Vercel deployment."""
import os
import sys
import tempfile

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.pipeline import Pipeline

app = FastAPI(title="Cooper Video Analysis API")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Cooper Video Analysis API is running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze a video file for sentiment and emotion.

    Args:
        file: The video file to analyze

    Returns:
        JSON response with analysis results
    """
    # Check file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported formats: mp4, mov, avi, mkv"
        )

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # Initialize and run the pipeline
            pipeline = Pipeline(
                whisper_model="base",  # Using smaller models for serverless function
                sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
                emotion_model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            )

            # Analyze the video
            results = pipeline.analyze(
                video_path=temp_file_path,
                output_dir=None,
                generate_plots=True,
                save_plots=False
            )

            # Prepare response
            response = {
                "text_sentiment": results.text_scores,
                "voice_emotion": results.audio_scores,
                "plots": {
                    "timeline": results.timeline_plot_bytes,  # base64-encoded PNG
                    "distribution": results.dist_plot_bytes   # base64-encoded PNG
                }
            }

            return JSONResponse(content=response)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing video: {str(e)}"
            )
