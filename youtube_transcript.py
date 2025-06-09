import csv
import logging
import os
import time
import argparse
from config import RETRY_DELAY
from process_video import process_youtube_video

def process_single_url(url):
    """Process a single YouTube URL."""
    try:
        url = url.strip()
        if not url:
            logging.error("Empty URL provided")
            return False
            
        logging.info(f"Processing single URL: {url}")
        process_youtube_video(url)
        return True
    except Exception as e:
        logging.error(f"Error processing URL {url}: {str(e)}")
        return False

def process_urls_from_csv(csv_file_path):
    """Process multiple URLs from a CSV file."""
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
            return processed_urls, total_urls
            
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
        return 0, 0
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description='YouTube Transcript Processor')
    parser.add_argument('--url', type=str, help='Single YouTube URL to process')
    parser.add_argument('--csv', type=str, help='Path to CSV file containing YouTube URLs')
    args = parser.parse_args()

    print("YouTube Transcript Processor")
    print("----------------------------")
    
    if args.url:
        # Process single URL
        print(f"Processing single URL: {args.url}")
        if process_single_url(args.url):
            print("URL processing completed successfully!")
        else:
            print("URL processing failed!")
    
    elif args.csv:
        # Process URLs from CSV
        if not os.path.exists(args.csv):
            print(f"Error: {args.csv} file not found!")
            return
            
        print(f"Reading URLs from {args.csv}")
        processed, total = process_urls_from_csv(args.csv)
        print(f"Processing completed! Processed {processed} out of {total} URLs")
    
    else:
        # Default behavior: use url_path.csv
        url_csv_path = 'url_path.csv'
        if not os.path.exists(url_csv_path):
            print(f"Error: {url_csv_path} file not found!")
            return
            
        print(f"Reading URLs from {url_csv_path}")
        processed, total = process_urls_from_csv(url_csv_path)
        print(f"Processing completed! Processed {processed} out of {total} URLs")

if __name__ == "__main__":
    main()