import yt_dlp
import os

def download_youtube_video(youtube_url, output_dir="downloads"):
    """
    Downloads a YouTube video to a specified directory.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_dir (str): Directory to save the downloaded video.
                          Defaults to "downloads" in the current working directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Attempting to download video from: {youtube_url}")

    # yt-dlp options for video-only download
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Prioritize mp4 for compatibility
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'), # Output file path template
        'noplaylist': True, # Only download single video if URL is part of a playlist
        'merge_output_format': 'mp4', # Ensure output is mp4 if video and audio streams are merged
        'keepvideo': True, # Keep the downloaded video file
        'paths': {'home': output_dir}, # Set download path
        'writethumbnail': False, # Don't download thumbnail
        'progress_hooks': [lambda d: print(d['status'], d.get('filename', '')) if d['status'] != 'downloading' else None], # Basic progress
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            # You can get the exact filepath here if you need it for further processing
            if 'filepath' in info_dict:
                video_filepath = info_dict['filepath']
            elif 'requested_downloads' in info_dict and info_dict['requested_downloads']:
                video_filepath = info_dict['requested_downloads'][0]['filepath']
            else:
                # Fallback to construct filename, though yt-dlp usually provides it
                video_title = info_dict.get('title', 'downloaded_video')
                video_ext = info_dict.get('ext', 'mp4')
                video_filepath = os.path.join(output_dir, f"{video_title}.{video_ext}")
                # Simple cleanup for filename in case of unsupported chars
                video_filepath = "".join([c for c in video_filepath if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()

        print(f"Video downloaded successfully to: {video_filepath}")

    except yt_dlp.DownloadError as e:
        print(f"Error downloading YouTube video: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example YouTube URL (Rick Astley!)
    # Replace with an actual YouTube video URL you want to download

    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_folder = os.path.join(script_dir, "downloads")

    download_youtube_video(youtube_link, output_dir=download_folder)

    print("\n--- Process Finished ---")