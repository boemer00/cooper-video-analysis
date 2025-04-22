"""Module for analyzing sentiment in text."""
from transformers import pipeline
from typing import Dict, List, Tuple

class TextSentimentAnalyzer:
    """Analyzes sentiment in text using a transformer model."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the text sentiment analyzer.

        Args:
            model_name (str): The pre-trained model to use for sentiment analysis
        """
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in a piece of text.

        Args:
            text (str): The text to analyze

        Returns:
            Dict[str, float]: Sentiment scores (positive/negative)
        """
        result = self.sentiment_pipeline(text)[0]
        label = result["label"].lower()
        score = result["score"]

        # Create a standardized output format with both positive and negative scores
        if label == "positive":
            return {"positive": score, "negative": 1.0 - score}
        else:
            return {"positive": 1.0 - score, "negative": score}

    def analyze_segments(self, segments: List[Dict]) -> List[Tuple[float, Dict[str, float]]]:
        """
        Analyze sentiment in a list of text segments with timestamps.

        Args:
            segments (List[Dict]): List of segments with text and timestamps

        Returns:
            List[Tuple[float, Dict[str, float]]]: List of (timestamp, sentiment_scores) tuples
        """
        results = []

        for segment in segments:
            text = segment["text"]
            # Use the midpoint of the start and end as the timestamp
            timestamp = (segment["start"] + segment["end"]) / 2

            sentiment_scores = self.analyze_text_sentiment(text)

            results.append((timestamp, sentiment_scores))

        return results
