from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
from process_video import process_youtube_video
import os
import uvicorn

app = FastAPI(title="YouTube Transcript Processor")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_transcript_processor.log'),
        logging.StreamHandler()
    ]
)

# Create output directories
os.makedirs('output_data', exist_ok=True)
os.makedirs('output_data/audio_files', exist_ok=True)

class URLRequest(BaseModel):
    url: str

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Transcript Processor</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen">
        <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
            <h1 class="text-2xl font-bold text-center text-gray-800 mb-6">YouTube Transcript Processor</h1>
            <div class="space-y-4">
                <div>
                    <label for="youtube-url" class="block text-sm font-medium text-gray-700">YouTube URL</label>
                    <input
                        type="text"
                        id="youtube-url"
                        class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        placeholder="Enter YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)"
                    />
                </div>
                <button
                    onclick="processURL()"
                    class="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                    Process Video
                </button>
                <div id="status" class="text-sm text-gray-600 mt-4"></div>
            </div>
        </div>

        <script>
            async function processURL() {
                const urlInput = document.getElementById('youtube-url').value.trim();
                const statusDiv = document.getElementById('status');

                if (!urlInput) {
                    statusDiv.innerHTML = '<span class="text-red-500">Please enter a valid YouTube URL.</span>';
                    return;
                }

                statusDiv.innerHTML = '<span class="text-blue-500">Processing...</span>';

                try {
                    const response = await fetch('/process_url', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url: urlInput }),
                    });

                    const result = await response.json();

                    if (response.ok) {
                        statusDiv.innerHTML = `<span class="text-green-500">${result.message}</span>`;
                    } else {
                        statusDiv.innerHTML = `<span class="text-red-500">Error: ${result.detail}</span>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<span class="text-red-500">Failed to process URL: ${error.message}</span>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/process_url")
async def process_url(request: URLRequest):
    try:
        url = request.url.strip()
        
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
            
        logging.info(f"Processing URL: {url}")
        
        # Process the YouTube URL
        result = process_youtube_video(url)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to process video. Please check if the video is available and try again."
            )
        
        return {
            "message": "Video processed successfully! Check the output_data directory for results.",
            "status": "success"
        }
        
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error processing URL: {error_message}")
        
        if "403" in error_message:
            error_message = "Access to this video is restricted. Please try a different video or check if the video is publicly available."
        elif "404" in error_message:
            error_message = "Video not found. Please check the URL and try again."
        elif "private" in error_message.lower():
            error_message = "This video is private. Please use a public video URL."
            
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 