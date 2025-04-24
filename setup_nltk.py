"""
Setup script to download required NLTK resources.
Run this once before using the application to ensure all NLTK data is available.
"""
import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = [
        'punkt',
        'averaged_perceptron_tagger'
    ]

    for resource in resources:
        logger.info(f"Downloading NLTK resource: {resource}")
        try:
            nltk.download(resource)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {str(e)}")

    logger.info("NLTK resource setup complete")

if __name__ == "__main__":
    download_nltk_resources()
