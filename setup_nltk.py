"""
Setup script to download required NLTK resources.
Run this once before using the application to ensure all NLTK data is available.

This script downloads all required NLTK resources for:
1. Basic NLP tasks (punkt, averaged_perceptron_tagger)
2. TextBlob library (used by NRCLex for emotion analysis)

You must run this script before using the application:
    python setup_nltk.py
"""
import nltk
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download all required NLTK resources."""
    # Basic NLP resources
    basic_resources = [
        'punkt',
        'averaged_perceptron_tagger'
    ]

    # Resources needed by TextBlob (used by NRCLex)
    textblob_resources = [
        'brown',
        'wordnet',
        'conll2000',
        'movie_reviews'
    ]

    # Combine all resources
    all_resources = basic_resources + textblob_resources

    success = True
    for resource in all_resources:
        logger.info(f"Downloading NLTK resource: {resource}")
        try:
            nltk.download(resource)
            logger.info(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"✗ Failed to download {resource}: {str(e)}")
            success = False

    if success:
        logger.info("✅ NLTK resource setup complete - all downloads successful")
        return True
    else:
        logger.error("❌ Some NLTK resources failed to download")
        return False

if __name__ == "__main__":
    logger.info("Starting NLTK resource downloads")

    # Try to verify existing installations
    try:
        import textblob
        logger.info(f"TextBlob version: {textblob.__version__}")
    except ImportError:
        logger.error("TextBlob not installed. Please install it with: pip install textblob")
        sys.exit(1)

    # Download resources
    success = download_nltk_resources()

    # Show clear instructions based on result
    if success:
        print("\n" + "="*80)
        print("SETUP COMPLETE: All NLTK resources successfully downloaded")
        print("You can now run the application")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("SETUP INCOMPLETE: Some NLTK resources failed to download")
        print("Please check the log output above and try again")
        print("="*80 + "\n")
        sys.exit(1)
