# Shein Comments Emotional Analysis

This module analyzes TikTok comments about Shein products to generate an emotional summary and visualizations.

## Overview

The analysis uses:
- Anthropic's Claude API to generate an emotional summary and extract key insights
- NRCLex for local emotion analysis
- Plotly for visualization

## Setup

1. Ensure you have the required dependencies:
```bash
pip install anthropic plotly pandas nrclex nltk textblob streamlit python-dotenv
```

2. Create a `.env` file in the project root with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

3. Make sure NLTK dependencies are downloaded:
```bash
python setup_nltk.py
```

## Running the Analysis

### Standalone Analysis

Run the analysis directly:

```bash
python emotional_summary.py
```

This will:
- Extract comments from `src/visualization/sample_comments.js`
- Analyze them using Anthropic Claude
- Generate HTML files with the visualizations:
  - `emotional_summary_card.html`
  - `emotion_distribution.html`

### Streamlit App

Run the dedicated Streamlit app:

```bash
streamlit run streamlit_emotional_summary.py
```

This launches a web interface where you can:
- Run the analysis with a single click
- View the emotional summary
- See visualizations of the emotional content

## Integration with Main App

To integrate this analysis into the main Streamlit app:

1. Import the analysis function:
```python
from shein_analysis import get_shein_analysis_for_streamlit
```

2. Call the function where needed:
```python
results = get_shein_analysis_for_streamlit()
if results["success"]:
    st.plotly_chart(results["summary_fig"])
    st.plotly_chart(results["emotion_fig"])
else:
    st.error(results["error"])
```

## Files

- `emotional_summary.py`: Core analysis functions
- `streamlit_emotional_summary.py`: Standalone Streamlit app
- `shein_analysis.py`: Interface for integration with main app

## Troubleshooting

- Ensure your Anthropic API key is valid and has sufficient credits
- Check that `sample_comments.js` is accessible and properly formatted
- Verify all dependencies are installed
- If NRCLex errors occur, run `setup_nltk.py` to download required data
