import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_transcript_processor.log'),
        logging.StreamHandler()
    ]
)

# Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not YOUTUBE_API_KEY:
    logging.error("YouTube API key not found. Set YOUTUBE_API_KEY environment variable.")
    raise ValueError("YouTube API key is required.")
if not OPENAI_API_KEY:
    logging.error("Open AI API key not found. Set OPENAI_API_KEY environment variable.")
    raise ValueError("Open AI API key is required.")

# Initialize OpenAI client with explicit configuration
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.openai.com/v1"
)

OUTPUT_DIR = 'output_data'
AUDIO_DIR = os.path.join(OUTPUT_DIR, 'audio_files')
CSV_FILENAME = 'youtube_transcripts.csv'
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_AUDIO_SIZE_MB = 25  # Whisper API file size limit
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Language name to ISO-639-1 code mapping
LANGUAGE_MAP = {
    'marathi': 'mr',
    'bengali': 'bn',
    'assamese': 'as',
    'tamil': 'ta',
    'hindi': 'hi',
    'kannada': 'kn',
    'odia': 'or',
    'telugu': 'te',
    'malayalam': 'ml',
    'gujarati': 'gu',
    'punjabi': 'pa',
    'english': 'en',
}

# Language name to full name mapping
LANGUAGE_NAME_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "or": "Odia",
    "as": "Assamese",
    "te": "Telugu",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


