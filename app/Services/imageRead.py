import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO


def extract_text_from_image_url(image_url: str) -> str:
    """
    Extract text, formulas, and descriptions from an image using Gemini Vision API.
    Downloads the image from the given URL and sends it to Gemini for analysis.
    """

    # Download the image from URL
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    # Open as PIL Image
    image = Image.open(BytesIO(response.content))

    # Use Gemini model with vision capabilities
    model = genai.GenerativeModel("gemini-flash-latest")

    prompt = """Analyze this image carefully and extract ALL content visible in it.

Follow these instructions:
1. Extract ALL plain text exactly as it appears in the image.
2. If there are mathematical formulas or equations, convert them to LaTeX format.
3. If there are diagrams, charts, or figures, describe them clearly.
4. Provide a brief overall description of the image.

Return the result in this structured format:

EXTRACTED TEXT:
[all plain text found in the image, preserving original structure]

FORMULAS (LaTeX):
[any mathematical formulas found, converted to LaTeX. Write "None found" if no formulas exist]

DIAGRAMS/FIGURES:
[description of any visual elements. Write "None found" if no diagrams exist]

IMAGE DESCRIPTION:
[brief overall description of the image content]
"""

    gemini_response = model.generate_content([prompt, image])

    return gemini_response.text