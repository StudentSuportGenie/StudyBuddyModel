import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GEMINI_API")


def configure_gemini():
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key not found. Set GEMINI_API in your .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)


configure_gemini()
