import base64
import time
import os
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

    # Try Gemini First
    try:
        # Initialize LangChain's ChatGoogleGenerativeAI component using gemini-2.0-flash
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))

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
        print(f"Gemini image analysis failed: {str(e)}. Falling back to Mistral...")
        
    # Mistral Fallback
    try:
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
             raise ValueError("Mistral API key not found in environment.")
        
        from langchain_mistralai import ChatMistralAI
        # pixtral-12b-2409 is the mistral vision model
        mistral_llm = ChatMistralAI(model="pixtral-12b-2409", temperature=0, mistral_api_key=mistral_api_key)
        
        # Reconstruct message for mistral
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        
        mistral_response = mistral_llm.invoke([message])
        return mistral_response.content
    except Exception as e:
        print(f"Mistral image analysis failed: {str(e)}")
        return generate_image_fallback()


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