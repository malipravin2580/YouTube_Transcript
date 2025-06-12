import csv
import re
import logging
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import YOUTUBE_API_KEY, OUTPUT_DIR, CSV_FILENAME, LANGUAGE_NAME_MAP
from clean_text_from_csv import clean_text_for_csv, clean_transcript_text
from video_utils import download_audio, detect_video_language, check_video_processed, translate_to_english
from transcribe_audio import transcribe_audio, summarize_transcript, extract_theme, extract_keywords_from_text

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


def process_youtube_video(url):
    url = url.strip().replace('//', '/').replace(':/', '://')
    video_id = extract_video_id(url)
    if not video_id:
        logging.error("Invalid YouTube URL.")
        return

    # Check for duplicates
    if check_video_processed(video_id):
        logging.info(f"Video {video_id} already processed, skipping.")
        return f"Video {video_id} already processed"

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
            return csv_file_path
            
        except Exception as e:
            logging.error(f"Error processing transcript: {e}")
            return
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())