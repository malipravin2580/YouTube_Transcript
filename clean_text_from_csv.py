import re
import logging

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
