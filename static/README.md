# Static Files

This directory can be used to store static files for the application.

## Logo Replacement Instructions

The current application uses a base64-encoded SVG placeholder for the logo. To replace it with your actual logo:

1. Prepare your logo image file (PNG format recommended)
2. Convert your image to base64 using an online converter or with Python:

```python
import base64
from pathlib import Path

# Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Example usage
image_path = "static/cooper_logo.png"  # Your logo file path
base64_string = image_to_base64(image_path)
print(f"data:image/png;base64,{base64_string}")
```

3. Open `streamlit_app.py` and replace the base64 string in the `display_logo()` function with your new base64 string.

4. Adjust the width and other styling as needed.

Alternatively, if you prefer to load the image from a file instead of using base64, you can modify the `display_logo()` function to use Streamlit's `st.image()` function instead.
