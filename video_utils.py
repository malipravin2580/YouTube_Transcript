import csv
import logging
import os
import time
import yt_dlp
from config import AUDIO_DIR, OUTPUT_DIR, CSV_FILENAME, MAX_RETRIES, RETRY_DELAY
from config import openai_client

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

def translate_to_english(text: str) -> str:
    print(text,":text")
    """
    Translate text to English using OpenAI's API with improved error handling and chunking.
    Returns the translated string, or the original text on error.
    """
    try:
        if not text or not text.strip():
            return ""
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            try:
                time.sleep(1)
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional translator. Translate the provided text to English with the following rules:\n"
                                "1. Translate the text exactly as provided, without adding or omitting any words or phrases.\n"
                                "2. Preserve the original meaning, tone, and context.\n"
                                "3. Maintain proper grammar, punctuation, and sentence structure in English.\n"
                                "4. Keep numbers, dates, names, and technical terms unchanged.\n"
                                "5. Ensure the translation is natural and fluent in English.\n"
                                "6. Do not add any explanatory notes, introductions, or additional text beyond the translation.\n"
                                "7. If the text is unclear, translate it as closely as possible to the original without guessing or adding context."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Translate this text to English, maintaining its original meaning and context:\n\n{chunk}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                translation = response.choices[0].message.content.strip()
                translated_chunks.append(translation)
                print(translation,":translation")
            except Exception as e:
                logging.error(f"Error translating chunk {i}: {str(e)}")
                translated_chunks.append(chunk)  # fallback to original chunk
        return ' '.join(translated_chunks)
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text  # fallback to original text

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