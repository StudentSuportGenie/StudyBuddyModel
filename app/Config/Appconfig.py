import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")

if not GOOGLE_API_KEY:
    print("Error: Google API key not found")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
