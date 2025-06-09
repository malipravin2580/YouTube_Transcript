# YouTube Transcript Processor

A Python application that processes YouTube videos to extract transcripts, translate them to English, and generate summaries, themes, and keywords.

## Project Structure

```
youtube_video/
├── config.py                 # Configuration settings and environment variables
├── youtube_transcript.py     # Main script for processing YouTube videos
├── process_video.py          # Core video processing functionality
├── transcribe_audio.py       # Audio transcription and analysis
├── video_utils.py           # Utility functions for video processing
├── clean_text_from_csv.py   # Text cleaning utilities
├── output_data/             # Directory for processed data
│   ├── audio_files/        # Downloaded audio files
│   └── youtube_transcripts.csv  # Processed transcript data
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```


## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- OpenAI API key
- YouTube Data API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube_video
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage
### Process a Single YouTube URL From api
```bash
python main.py 
```

```bash
python youtube_transcript.py --url "https://www.youtube.com/watch?v=your_video_id"
```

### Process a Single YouTube URL

```bash
python youtube_transcript.py --url "https://www.youtube.com/watch?v=your_video_id"
```

### Process Multiple URLs from a CSV File

```bash
python youtube_transcript.py --csv "path/to/your/urls.csv"
```

### Use Default CSV File

```bash
python youtube_transcript.py
```

The CSV file should have a column named "URL" containing YouTube URLs.

## Output

The processed data is saved in the following format:

- Audio files are saved in `output_data/audio_files/`
- Transcripts and metadata are saved in `output_data/youtube_transcripts.csv`

The CSV file contains the following columns:
- Timestamp
- Video Link
- Video ID
- Video Title
- Channel Name
- Description
- Video Duration
- Audio File Name
- Original Language
- Original Transcription
- Translated Version
- Facts
- Theme
- Keywords

