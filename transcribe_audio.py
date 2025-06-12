import re
import logging
import os
from collections import Counter
from pydub import AudioSegment
from langdetect import detect
from config import openai_client, MAX_AUDIO_SIZE_MB, CHUNK_DURATION_MS, LANGUAGE_MAP, AUDIO_DIR

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

                    # Reset file pointer and try transcription with detected language
                    audio_file.seek(0)
                    try:
                        transcription_response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language=detected_language_code
                        )
                        transcribed_text = transcription_response.text
                    except Exception as e:
                        if "unsupported_language" in str(e) or "400" in str(e):
                            logging.warning(f"Error with {detected_language_code} transcription, trying without language specification")
                            audio_file.seek(0)
                            transcription_response = openai_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                            transcribed_text = transcription_response.text
                        else:
                            raise
                    
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
                    if "unsupported_language" in str(e) or "400" in str(e):
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
                        try:
                            response = openai_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=chunk_file,
                                language=detected_language_code
                            )
                        except Exception as e:
                            if "unsupported_language" in str(e) or "400" in str(e):
                                logging.warning(f"Error with {detected_language_code} transcription for chunk, trying without language specification")
                                chunk_file.seek(0)
                                response = openai_client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=chunk_file
                                )
                            else:
                                raise
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
    """
    Extract the main theme and related keywords from the video content using GPT-4.
    The theme is dynamically determined based on the actual content rather than predefined categories.
    
    Args:
        title (str): Video title
        description (str): Video description
        transcript (str): Video transcript
    
    Returns:
        tuple: (main_theme, related_keywords)
    """
    try:
        combined_text = f"Title: {title}\nDescription: {description}\nTranscript: {transcript}"
        
        prompt = f"""
        Analyze the following video content and:
        1. Identify the main theme/topic (be specific and descriptive)
        2. Extract 15 most relevant keywords that represent the core concepts
        Focus on the actual content and context, not just surface-level categorization.
        
        Content:
        {combined_text[:4000]}
        
        Return the response in this format:
        THEME: [specific theme description]
        KEYWORDS: [comma-separated keywords]
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert content analyzer. Your task is to identify the main theme "
                        "and extract relevant keywords from video content. Be specific and descriptive "
                        "in identifying themes, focusing on the actual content rather than broad categories. "
                        "Consider the context, tone, and specific topics discussed."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        theme = ""
        keywords = ""
        
        for line in result.split('\n'):
            if line.startswith('THEME:'):
                theme = line.replace('THEME:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keywords = line.replace('KEYWORDS:', '').strip()
        
        if not theme or not keywords:
            logging.warning("Failed to extract theme or keywords, falling back to basic analysis")
            return extract_theme_fallback(title, description, transcript)
            
        logging.info(f"Extracted theme: {theme}")
        logging.info(f"Extracted keywords: {keywords}")
        
        return theme, keywords[:500]
        
    except Exception as e:
        logging.error(f"Error in theme extraction: {str(e)}")
        return extract_theme_fallback(title, description, transcript)

def extract_theme_fallback(title, description, transcript):
    """
    Fallback method for theme extraction using basic text analysis.
    """
    try:
        combined_text = f"{title} {description} {transcript}".lower()
        
        # Common context indicators
        context_indicators = {
            'agriculture': ['crop', 'farmer', 'farming', 'agriculture', 'harvest', 'soil', 'irrigation'],
            'education': ['learn', 'study', 'education', 'course', 'training', 'workshop'],
            'technology': ['software', 'programming', 'computer', 'digital', 'online', 'app'],
            'business': ['business', 'market', 'finance', 'investment', 'company', 'startup'],
            'health': ['health', 'medical', 'disease', 'treatment', 'doctor', 'hospital'],
            'environment': ['environment', 'climate', 'pollution', 'conservation', 'sustainable'],
            'social': ['community', 'society', 'people', 'social', 'development', 'welfare']
        }
        
        # Score each context based on keyword presence and context
        context_scores = {}
        for context, keywords in context_indicators.items():
            score = 0
            for keyword in keywords:
                # Basic keyword match
                if keyword in combined_text:
                    score += 1
                # Context-based scoring
                patterns = [
                    rf'\b(important|key|main|primary|essential|critical)\s+\w+\s+{keyword}',
                    rf'\b{keyword}\s+(program|scheme|initiative|project|system)',
                    rf'\b(learn|understand|know|study)\s+about\s+{keyword}',
                    rf'\b{keyword}\s+(development|improvement|enhancement|advancement)'
                ]
                for pattern in patterns:
                    if re.search(pattern, combined_text):
                        score += 2
            context_scores[context] = score
        
        # Get the highest scoring context
        main_context = max(context_scores.items(), key=lambda x: x[1])[0]
        
        # Extract relevant keywords
        words = re.findall(r'\b[\w]{4,}\b', combined_text)
        word_scores = Counter()
        
        for word in words:
            if word not in {'this', 'that', 'with', 'from', 'have', 'they', 'will', 'about', 'very',
                          'video', 'welcome', 'please', 'subscribe', 'channel', 'like', 'share'}:
                word_scores[word] += 1
                
                # Boost score for words related to main context
                if any(keyword in word for keyword in context_indicators[main_context]):
                    word_scores[word] += 2
        
        # Get top keywords
        top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = ', '.join(word for word, score in top_keywords[:15])
        
        # Create a descriptive theme
        theme = f"{main_context.title()} - {', '.join(keywords.split(', ')[:3])}"
        
        return theme, keywords[:500]
        
    except Exception as e:
        logging.error(f"Error in fallback theme extraction: {str(e)}")
        return "General Content", "content, video, information"

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