import base64
import time
import requests
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def extract_text_from_image_url(image_url: str) -> str:
    """
    Extract text, formulas, and descriptions from an image using Gemini Vision API.
    Downloads the image from the given URL and sends it to Gemini for analysis.
    Includes retry logic for rate limiting.
    """

    # Download the image from URL with a standard User-Agent
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
    response = requests.get(image_url, headers=headers, timeout=30)
    response.raise_for_status()

    # Encode the image bytes to base64 for LangChain's message structure
    base64_image = base64.b64encode(response.content).decode("utf-8")

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

    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Initialize LangChain's ChatGoogleGenerativeAI component using gemini-2.0-flash
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

            # Create LangChain HumanMessage containing text and image_url dict with base64 data URL
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )

            # Invoke LLM with the message list
            gemini_response = llm.invoke([message])

            return gemini_response.content
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"Rate limit hit on image analysis (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    # Last attempt failed - return fallback
                    print("Rate limit exceeded on image analysis. Returning fallback.")
                    return generate_image_fallback()
            else:
                # Non-rate-limit error, raise it
                raise


def generate_image_fallback():
    """Fallback response when image extraction is rate limited."""
    return """EXTRACTED TEXT:
[Image processing temporarily unavailable due to rate limiting]

FORMULAS (LaTeX):
None extracted

DIAGRAMS/FIGURES:
Unable to analyze

IMAGE DESCRIPTION:
The image could not be processed at this time due to API rate limiting. Please try again later."""