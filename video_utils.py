import csv
import logging
import os
import time
import yt_dlp
from deep_translator import GoogleTranslator
from config import AUDIO_DIR, OUTPUT_DIR, CSV_FILENAME, MAX_RETRIES, RETRY_DELAY

def download_audio(video_id, url):
    try:
        url = url.strip().replace('//', '/').replace(':/', '://')
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(AUDIO_DIR, f'{video_id}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                output_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
                if os.path.exists(output_path):
                    logging.info(f"Audio downloaded successfully for video {video_id}")
                    return output_path
                else:
                    raise Exception("Audio file not found after download")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    except Exception as e:
        logging.error(f"Error downloading audio for video {video_id}: {str(e)}")
        return None

def detect_video_language(title, description):
    """
    Detect language from title and description using keywords and character sets.
    """
    language_keywords = {
        'bengali': 'bn',
        'odia': 'or',
        'tamil': 'ta',
        'marathi': 'mr',
        'hindi': 'hi',
        'kannada': 'kn',
        'assamese': 'as',
        'telugu': 'te',
        'malayalam': 'ml',
        'gujarati': 'gu',
        'punjabi': 'pa',
        'english': 'en'
    }
    
    kannada_chars = set('ಐಯೂಓಓಓಓಕ್ಘಘಚಜ್ಞಾಂತಧದಧನತಥದಧನಪಫಭಮಯರಲವಶ್ಚದಶ್ಯತಜ್ಞ')
    hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़')
    odia_chars = set('आईीयूऑएऑओऑॉक्घघघचझ्जन्तधदधनतथदधनपफभमयरलवश्षसहक्षत्रज्ञद्ध')
    text = f"{title} {description}".lower()
    
    # Check for language keywords in title/description
    for lang_name, lang_code in language_keywords.items():
        if lang_name in text:
            return lang_code
    
    # Fallback to character set detection
    kannada_count = sum(1 for char in text if char in kannada_chars)
    hindi_count = sum(1 for char in text if char in hindi_chars)
    odia_count = sum(1 for char in text if char in odia_chars)
    if kannada_count > hindi_count:
        return 'kn'
    elif hindi_count > kannada_count:
        return 'hi'
    elif odia_count > hindi_count:
        return 'or'

    # Additional heuristic for Marathi (common in Maharashtra, often linked to soybean farming)
    if 'soyabean' in text and 'maharashtra' in text or 'latur' in text:
        return 'mr'
    return 'en'

def translate_to_english(text):
    """Translate text to English using Google Translate."""
    try:
        if not text or text.strip() == "":
            return ""
            
        # Split text into chunks to handle long texts
        max_chunk_size = 1000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        for chunk in chunks:
            try:
                # Add delay between translations to avoid rate limiting
                time.sleep(1)
                # Translate chunk to English
                translation = GoogleTranslator(source='auto', target='en').translate(chunk)
                translated_chunks.append(translation)
            except Exception as e:
                logging.warning(f"Translation chunk failed: {str(e)}")
                translated_chunks.append(chunk)  # Keep original text if translation fails
                
        return ' '.join(translated_chunks)
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails
    
def check_video_processed(video_id):
    """Check if a video ID has already been processed in the CSV."""
    csv_file_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    if not os.path.exists(csv_file_path):
        return False
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Video ID'] == video_id:
                return True
    return False 