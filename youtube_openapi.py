import csv
import re
import logging
import os
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
import yt_dlp
from deep_translator import GoogleTranslator
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
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

openai_client = OpenAI(api_key=OPENAI_API_KEY)
OUTPUT_DIR = 'output_data'
AUDIO_DIR = os.path.join(OUTPUT_DIR, 'audio_files')
CSV_FILENAME = 'youtube_transcripts.csv'
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_AUDIO_SIZE_MB = 25  # Whisper API file size limit
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk

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

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def extract_video_id(url):
    try:
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        elif 'youtube.com' in url:
            if 'v=' in url:
                return parse_qs(urlparse(url).query)['v'][0]
            elif 'embed/' in url:
                return url.split('embed/')[1].split('?')[0]
        return None
    except Exception as e:
        logging.error(f"Error extracting video ID: {str(e)}")
        return None

def get_video_metadata(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        request = youtube.videos().list(
            part='snippet,contentDetails',
            id=video_id
        )
        response = request.execute()

        if not response['items']:
            logging.error("Video metadata not found.")
            return None, None, None, None

        item = response['items'][0]
        snippet = item['snippet']
        duration = item['contentDetails']['duration']
        return snippet['title'], snippet['description'], snippet['channelTitle'], duration
    except HttpError as e:
        if e.resp.status == 403:
            logging.error("YouTube Data API quota exceeded or API key is invalid.")
        elif e.resp.status == 404:
            logging.error("Video not found or is private.")
        else:
            logging.error(f"YouTube Data API error: {e}")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Unexpected error fetching video metadata: {e}")
        return None, None, None, None

def parse_duration(iso_duration):
    try:
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        logging.error(f"Error parsing duration {iso_duration}: {str(e)}")
        return 0

def clean_text_for_csv(text):
    """Clean text for CSV by removing newlines, extra spaces, and ensuring it's a string."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace (newlines, tabs) with a single space
    text = text.strip()
    return text

def clean_transcript_text(text):
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', text)
    text = re.sub(r'\b(um|uh|like|you know|basically|actually|literally)\b', '', text, flags=re.IGNORECASE)
    # Summarize repeated numbers
    text = re.sub(r'(\d+,\d+)(?:\s+\d+,\d+){3,}', r'\1...', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def transcribe_audio(audio_path, metadata_language):
    """
    Transcribe audio file using Open AI Whisper API, detecting the original language.
    
    Args:
        audio_path (str): Path to the audio file.
        metadata_language (str): Language detected from metadata to guide transcription.
    
    Returns:
        tuple: (transcribed_text, selected_language, duration in seconds)
    """
    try:
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return "", "en", 0

        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0

        # Check file size (Whisper API limit: 25 MB)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= MAX_AUDIO_SIZE_MB:
            # Detect language and transcribe
            with open(audio_path, 'rb') as audio_file:
                try:
                    # First pass: detect language
                    detection_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        prompt="Detect language only",
                        response_format="verbose_json"
                    )
                    detected_language_name = detection_response.language.lower()
                    logging.info(f"Whisper detected audio language: {detected_language_name}")

                    # Map to ISO-639-1 code
                    detected_language_code = LANGUAGE_MAP.get(detected_language_name, 'en')
                    logging.info(f"Detected language code: {detected_language_code}")

                    # Check if the language is supported by Whisper
                    supported_languages = {
                        'en',  # English
                        'hi',  # Hindi
                        'ta',  # Tamil
                        'te',  # Telugu
                        'ml',  # Malayalam
                        'mr',  # Marathi
                        'gu',  # Gujarati
                        'kn',  # Kannada
                        'pa',  # Punjabi
                        'bn',  # Bengali
                        'or',  # Odia
                        'as'   # Assamese
                    }
                    
                    if detected_language_code not in supported_languages:
                        logging.warning(f"Language {detected_language_code} not supported by Whisper, falling back to English")
                        detected_language_code = 'en'

                    # Reset file pointer and transcribe in detected language
                    audio_file.seek(0)
                    transcription_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=detected_language_code
                    )
                    transcribed_text = transcription_response.text
                    
                    # Verify transcription language
                    try:
                        lang = detect(transcribed_text)
                        if lang != detected_language_code:
                            logging.warning(f"Langdetect suggests {lang}, but using {detected_language_code} based on metadata/Whisper")
                    except:
                        pass
                    
                    logging.info(f"Transcribed audio {audio_path} in {detected_language_code}")
                    return transcribed_text, detected_language_code, duration_seconds
                    
                except Exception as e:
                    if "unsupported_language" in str(e):
                        logging.warning(f"Language not supported, falling back to English transcription")
                        audio_file.seek(0)
                        response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="en"
                        )
                        logging.info(f"Fallback: Transcribed audio {audio_path} in English")
                        return response.text, "en", duration_seconds
                    else:
                        raise
        else:
            # Split audio into chunks
            logging.info(f"Audio file {audio_path} ({file_size_mb:.2f} MB) exceeds {MAX_AUDIO_SIZE_MB} MB, splitting into chunks")
            chunks = []
            for i in range(0, duration_ms, CHUNK_DURATION_MS):
                chunk = audio[i:i + CHUNK_DURATION_MS]
                chunk_path = os.path.join(AUDIO_DIR, f"chunk_{i//1000}.mp3")
                chunk.export(chunk_path, format="mp3")
                chunks.append(chunk_path)

            # Transcribe first chunk to detect language
            with open(chunks[0], 'rb') as first_chunk_file:
                try:
                    detection_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=first_chunk_file,
                        prompt="Detect language only",
                        response_format="verbose_json"
                    )
                    detected_language_name = detection_response.language.lower()
                    logging.info(f"Detected audio language from first chunk: {detected_language_name}")

                    # Map to ISO-639-1 code
                    detected_language_code = LANGUAGE_MAP.get(detected_language_name, 'en')
                    
                    # Check if the language is supported
                    supported_languages = {
                        'en',  # English
                        'hi',  # Hindi
                        'ta',  # Tamil
                        'te',  # Telugu
                        'ml',  # Malayalam
                        'mr',  # Marathi
                        'gu',  # Gujarati
                        'kn',  # Kannada
                        'pa',  # Punjabi
                        'bn',  # Bengali
                        'or',  # Odia
                        'as'   # Assamese
                    }
                    
                    if detected_language_code not in supported_languages:
                        logging.warning(f"Language {detected_language_code} not supported by Whisper, falling back to English")
                        detected_language_code = 'en'
                except Exception as e:
                    logging.warning(f"Error detecting language from first chunk: {str(e)}")
                    detected_language_code = 'en'

            # Transcribe all chunks
            transcribed_texts = []
            for chunk_path in chunks:
                try:
                    with open(chunk_path, 'rb') as chunk_file:
                        response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=chunk_file,
                            language=detected_language_code
                        )
                    transcribed_texts.append(response.text)
                    os.remove(chunk_path)
                except Exception as e:
                    logging.warning(f"Error transcribing chunk {chunk_path}: {str(e)}")
                    os.remove(chunk_path)
                    continue

            if not transcribed_texts:
                logging.error(f"Failed to transcribe any chunks for {audio_path}")
                return "", "en", duration_seconds

            combined_text = ' '.join(transcribed_texts)
            logging.info(f"Transcribed {len(chunks)} chunks for {audio_path}")
            return combined_text, detected_language_code, duration_seconds

    except Exception as e:
        logging.error(f"Error transcribing audio {audio_path}: {str(e)}")
        # Fallback to English transcription
        try:
            with open(audio_path, 'rb') as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            logging.info(f"Fallback: Transcribed audio {audio_path} in English")
            return response.text, "en", duration_seconds
        except Exception as e2:
            logging.error(f"Fallback transcription failed: {str(e2)}")
            return "", "en", duration_seconds

def summarize_transcript(text, max_facts=5, max_length=200):
    try:
        if not text or text.strip() == "":
            logging.error("Empty transcript provided for summarization.")
            return ""

        prompt = f"""
        Summarize the following text into {max_facts} key facts, each no longer than {max_length} characters.
        Each fact should be concise, relevant, and written in clear English.
        Focus on key points related to agriculture, education, or other significant themes.
        Return the facts as a list, one fact per line.

        Text:
        {text[:4000]}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in summarizing and extracting key information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_facts * 100,
            temperature=0.3,
        )

        facts = response.choices[0].message.content.strip().split('\n')
        facts = [fact.strip() for fact in facts if fact.strip()]
        facts = [fact[:max_length-3] + "..." if len(fact) > max_length else fact for fact in facts]
        facts = facts[:max_facts]
        
        logging.info(f"Generated {len(facts)} facts using Open AI API.")
        return '\n'.join(facts)

    except Exception as e:
        logging.error(f"Error generating facts with Open AI: {str(e)}")
        logging.info("Falling back to heuristic-based summarization.")
        return summarize_transcript_fallback(text, max_facts, max_length)

def summarize_transcript_fallback(text, max_facts=5, max_length=200):
    try:
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        if not sentences:
            logging.error("No valid sentences found in transcript.")
            return ""

        key_phrases = {
            'en': [
                "important to note", "key point", "main idea", "this means",
                "according to", "research shows", "data indicates", "study found",
                "today we discuss", "in this program", "special guest", "we will cover",
                "crop insurance", "financial security", "natural disaster", "farmer benefit",
                "enrolment process", "claim amount", "agricultural scheme"
            ]
        }

        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            score = 0
            if any(phrase in sentence.lower() for phrase in key_phrases['en']):
                score += 3
            if re.search(r'\d+%|\d{4}|[\d,]+', sentence):
                score += 2
            if re.search(r'\b(crop|farmer|insurance|scheme|disaster|benefit|enrolment)\b', sentence.lower()):
                score += 2
            if 50 < len(sentence) < 250:
                score += 1
            if re.search(r'\b(um|uh|like|you know|basically|actually|literally)\b', sentence.lower()):
                score -= 1

            if score > 0:
                scored_sentences.append((sentence, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:max_facts]

        facts = []
        for sentence, _ in top_sentences:
            fact = re.sub(r'^(so|well|now|okay|right|you know|like|basically|actually,literally),?\s+', '', sentence, flags=re.IGNORECASE)
            if len(fact) > max_length:
                parts = re.split(r'[,;]', fact)
                fact = parts[0].strip() if parts else fact
            if len(fact) > max_length:
                fact = fact[:max_length-3] + "..."
            fact = re.sub(r'\s+', ' ', fact).strip()
            if fact and fact not in facts:
                facts.append(fact)

        facts_string = '\n'.join(facts)
        logging.info(f"Generated {len(facts)} facts from transcript (fallback).")
        return facts_string

    except Exception as e:
        logging.error(f"Error in fallback summarization: {str(e)}")
        return ""

def extract_theme(title, description, transcript):
    combined_text = f"{title} {description} {transcript}".lower()
    
    themes = {
        "Agricultural Education": [
            "agriculture", "farming", "crop", "irrigation", "farmer", "cultivation",
            "harvest", "soil", "seeds", "fertilizer", "pesticide", "organic",
            "sustainable", "agricultural show", "agricultural program",
            "crop insurance", "farmer welfare", "agricultural community",
            "agricultural knowledge"
        ],
        "UPSC Preparation": [
            "upsc", "civil service", "ias", "prelims", "mains", "exam",
            "preparation", "study", "current affairs", "general studies",
            "optional subject", "interview", "coaching", "mock test"
        ],
        "Environmental Studies": [
            "environment", "ecology", "climate", "conservation", "biodiversity",
            "sustainability", "pollution", "renewable energy", "climate change",
            "wildlife", "forest", "natural resources"
        ],
        "Programming": [
            "python", "django", "coding", "programming", "software", "development",
            "web", "application", "database", "algorithm", "debug", "code",
            "framework", "api"
        ],
        "General Education": [
            "learn", "course", "tutorial", "education", "teaching", "learning",
            "study", "knowledge", "skill", "training", "workshop", "seminar", "lecture"
        ]
    }
    
    stopwords = {
        'this', 'that', 'with', 'from', 'have', 'they', 'will', 'about', 'very',
        'video', 'welcome', 'please', 'subscribe', "channel", "like", "share",
        'today', 'going', 'know', 'think', 'just', 'make', 'want', 'need',
        'good', 'great', 'nice', 'well', 'much', 'many', 'lot', 'really',
        'actually', 'basically', 'literally'
    }
    
    theme = "General Education"
    max_matches = 0
    for theme_name, keywords in themes.items():
        matches = sum(1 for keyword in keywords if keyword in combined_text)
        if matches > max_matches:
            max_matches = matches
            theme = theme_name
    
    keyword_scores = {}
    words = re.findall(r'\b[\w]{4,}\b', combined_text, re.UNICODE)
    word_counts = Counter(word for word in words if word not in stopwords)
    
    for word, count in word_counts.items():
        score = count
        context_patterns = [
            r'\b(important|key|main|primary|essential|critical)\s+\w+\s+' + word,
            r'\b' + word + r'\s+(program|scheme|initiative|project|system)',
            r'\b(learn|understand|know|study)\s+about\s+' + word,
            r'\b' + word + r'\s+(development|improvement|enhancement|advancement)'
        ]
        for pattern in context_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 2
        if any(word in theme_keywords for theme_keywords in themes.values()):
            score += 3
        keyword_scores[word] = score
    
    top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = ', '.join(word for word, score in top_keywords[:15])
    
    for keyword in themes.get(theme, []):
        if keyword not in keywords and keyword in combined_text:
            keywords += f", {keyword}"
    
    return theme, keywords[:500]

def extract_keywords_from_text(text, max_keywords=15):
    try:
        if not text or text.strip() == "":
            logging.error("Empty text provided for keyword extraction.")
            return ""

        prompt = f"""
        Extract up to {max_keywords} important keywords from the following text.
        Focus on terms related to agriculture, education, or other significant themes.
        Return the keywords as a comma-separated list in English.

        Text:
        {text[:4000]}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in extracting relevant keywords in English."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3,
        )

        keywords = response.choices[0].message.content.strip()
        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        keywords = keywords[:max_keywords]
        
        agricultural_terms = {
            'crop', 'farmer', 'agriculture', 'farming', 'irrigation', 'harvest',
            'soil', 'fertilizer', 'pesticide', 'organic', 'sustainable',
            'insurance', 'scheme', 'subsidy', 'benefit', 'welfare'
        }
        educational_terms = {
            'learn', 'study', 'education', 'training', 'workshop', 'seminar',
            'lecture', 'course', 'program', 'tutorial', 'guide', 'instruction'
        }
        for term in agricultural_terms.union(educational_terms):
            if term in text.lower() and term not in keywords:
                keywords.append(term)

        keywords = list(dict.fromkeys(keywords))[:max_keywords]
        keyword_string = ', '.join(keywords)
        
        logging.info(f"Extracted {len(keywords)} keywords using Open AI API.")
        return keyword_string

    except Exception as e:
        logging.error(f"Error extracting keywords with Open AI: {str(e)}")
        logging.info("Falling back to heuristic-based keyword extraction.")
        return extract_keywords_from_text_fallback(text, max_keywords)

def extract_keywords_from_text_fallback(text, max_keywords=15):
    try:
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'they', 'will', 'about', 'very',
            'video', 'welcome', 'please', 'subscribe', "channel", "like", "share",
            'today', 'going', 'know', 'think', 'just', 'make', 'want', 'need',
            'good', 'great', 'nice', 'well', 'much', 'many', 'lot', 'really',
            'actually', 'basically', 'literally', 'the', 'and', 'or', 'but', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'
        }
        
        context_patterns = [
            r'\b(important|key|main|primary|essential|critical)\s+\w+\s+(\w+)',
            r'\b(\w+)\s+(program|scheme|initiative|project|system)',
            r'\b(learn|understand|know|study)\s+about\s+(\w+)',
            r'\b(\w+)\s+(development|improvement|enhancement|advancement)',
            r'\b(according to|research shows|study found|data indicates)\s+(\w+)',
            r'\b(significant|major|notable|important)\s+(\w+)',
            r'\b(\w+)\s+(benefit|advantage|impact|effect)'
        ]
        
        words = re.findall(r'\b[\w]{4,}\b', text.lower())
        word_scores = Counter()
        
        for word in words:
            if word not in stopwords:
                word_scores[word] += 1
                for pattern in context_patterns:
                    matches = re.finditer(pattern, text.lower())
                    for match in matches:
                        if word in match.groups():
                            word_scores[word] += 2
        
        top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in top_keywords[:max_keywords]]
        
        agricultural_terms = {
            'crop', 'farmer', 'agriculture', 'farming', 'irrigation', 'harvest',
            'soil', 'fertilizer', 'pesticide', 'organic', 'sustainable',
            'insurance', 'scheme', 'subsidy', 'benefit', 'welfare'
        }
        
        educational_terms = {
            'learn', 'study', 'education', 'training', 'workshop', 'seminar',
            'lecture', 'course', 'program', 'tutorial', 'guide', 'instruction'
        }
        
        for term in agricultural_terms.union(educational_terms):
            if term in text.lower() and term not in keywords:
                keywords.append(term)
        
        keywords = [k.strip() for k in keywords if k.strip()]
        keywords = list(dict.fromkeys(keywords))
        keyword_string = ', '.join(keywords[:max_keywords])
        
        logging.info(f"Extracted {len(keywords)} keywords from text (fallback)")
        return keyword_string
        
    except Exception as e:
        logging.error(f"Error in fallback keyword extraction: {str(e)}")
        return ""

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
        max_chunk_size = 5000
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

def process_youtube_video(url):
    url = url.strip().replace('//', '/').replace(':/', '://')
    video_id = extract_video_id(url)
    if not video_id:
        logging.error("Invalid YouTube URL.")
        return

    # Check for duplicates
    if check_video_processed(video_id):
        logging.info(f"Video {video_id} already processed, skipping.")
        return

    try:
        title, description, channel_name, duration = get_video_metadata(video_id)
        if not title:
            logging.error("Failed to fetch video metadata.")
            return

        detected_language = detect_video_language(title, description)
        logging.info(f"Detected video language from metadata: {detected_language}")

        audio_path = download_audio(video_id, url)
        if not audio_path:
            logging.error(f"Failed to download audio for video {video_id}")
            return

        try:
            transcribed_text, selected_language, video_duration = transcribe_audio(audio_path, detected_language)
            if not transcribed_text:
                logging.error(f"Failed to transcribe audio for video {video_id}")
                return

            cleaned_text = clean_transcript_text(transcribed_text)
            logging.info(f"Translating transcript to English: {cleaned_text[:100]}...")
            translated_text = translate_to_english(cleaned_text)
            logging.info(f"Translation completed: {translated_text[:100]}...")

            facts = summarize_transcript(translated_text)
            theme, _ = extract_theme(title, description, translated_text)
            keywords = extract_keywords_from_text(translated_text)

            csv_file_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
            file_exists = os.path.isfile(csv_file_path)

            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writerow([
                        'Timestamp', 'Video Link', 'Video ID', 'Video Title', 'Channel Name', 
                        'Description', 'Video Duration', 'Audio File Name', 'Original Language', 
                        'Original Transcription', 'Translated Version', 'Facts', 'Theme', 'Keywords'
                    ])
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                original_language_name = LANGUAGE_NAME_MAP.get(selected_language, selected_language)
                # Clean all fields for CSV
                row = [
                    timestamp,
                    url,
                    video_id,
                    clean_text_for_csv(title),
                    clean_text_for_csv(channel_name),
                    clean_text_for_csv(description),
                    video_duration,
                    audio_path,
                    original_language_name,
                    clean_text_for_csv(cleaned_text),
                    clean_text_for_csv(translated_text),
                    clean_text_for_csv(facts),
                    theme,
                    clean_text_for_csv(keywords)
                ]
                writer.writerow(row)

            logging.info(f"Transcript and metadata appended to {csv_file_path}")
            
        except Exception as e:
            logging.error(f"Error processing transcript: {e}")
            return
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())

def process_urls_from_csv(csv_file_path):
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_urls = 0
            processed_urls = 0
            
            for row in reader:
                total_urls += 1
            
            csvfile.seek(0)
            next(reader)
            
            for row in reader:
                url = row['URL'].strip()
                if not url:
                    continue
                    
                logging.info(f"Processing URL {processed_urls + 1} of {total_urls}: {url}")
                try:
                    process_youtube_video(url)
                    processed_urls += 1
                    time.sleep(RETRY_DELAY)
                except Exception as e:
                    logging.error(f"Error processing URL {url}: {str(e)}")
                    continue
                    
            logging.info(f"Completed processing {processed_urls} out of {total_urls} URLs")
            
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")

def main():
    print("YouTube Transcript Processor")
    print("----------------------------")
    
    url_csv_path = 'url_path.csv'
    
    if not os.path.exists(url_csv_path):
        print(f"Error: {url_csv_path} file not found!")
        return
        
    print(f"Reading URLs from {url_csv_path}")
    process_urls_from_csv(url_csv_path)
    print("Processing completed!")

if __name__ == "__main__":
    main()