import os
from dotenv import load_dotenv
import google.generativeai as genai

# Get absolute path to the .env file (2 levels up from app/Config/Appconfig.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "..", "..", ".env")

load_dotenv(dotenv_path=env_path, override=True)
GOOGLE_API_KEY = os.getenv("GEMINI_API")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def configure_gemini():
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key not found. Set GEMINI_API in your .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)
    # Langchain's ChatGoogleGenerativeAI requires the GOOGLE_API_KEY env var
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    if MISTRAL_API_KEY:
        os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

configure_gemini()
