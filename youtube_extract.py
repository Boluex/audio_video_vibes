import yt_dlp
from moviepy.editor import VideoFileClip
import os

def download_youtube_video_and_extract_audio(youtube_url, output_dir="downloads", audio_format="mp3"):
    """
    Downloads a YouTube video and then extracts its audio.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_dir (str): Directory to save the downloaded video and extracted audio.
                          Defaults to "downloads" in the current working directory.
        audio_format (str): The format for the extracted audio (e.g., "mp3", "wav", "aac").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Step 1: Download the YouTube Video (using yt-dlp) ---
    print(f"Attempting to download video from: {youtube_url}")

    # yt-dlp options
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best', # Download best quality video and audio, then merge
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'), # Output file path template
        'noplaylist': True, # Only download single video if URL is part of a playlist
        'merge_output_format': 'mp4', # Ensure output is mp4 (for moviepy compatibility)
        'postprocessors': [{ # Ensure audio is extracted first, then downloaded
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192', # Example audio quality (kbps)
        }],
        'keepvideo': True, # Keep the downloaded video file after audio extraction
        'paths': {'home': output_dir}, # Set download path
        'writethumbnail': False, # Don't download thumbnail
    }

    video_filepath = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            # yt-dlp saves the video by title. We need to reconstruct the filename.
            # It might have a different extension if merge_output_format isn't respected or due to best format.
            # Best way is to get the actual filepath from info_dict if available.
            # info_dict provides 'filepath' or 'requested_downloads'
            if 'requested_downloads' in info_dict and info_dict['requested_downloads']:
                # For combined video/audio, it might be the first entry, or the merged one
                # yt-dlp usually reports the final merged file.
                video_filepath = info_dict['requested_downloads'][0]['filepath']
            elif 'filepath' in info_dict:
                video_filepath = info_dict['filepath']
            else:
                # Fallback: construct path based on title and original ext
                # This can be tricky if yt-dlp re-encodes or uses a different ext
                video_title = info_dict.get('title', 'downloaded_video')
                video_ext = info_dict.get('ext', 'mp4') # Default to mp4
                video_filepath = os.path.join(output_dir, f"{video_title}.{video_ext}")
                # Clean up filename for special characters, etc. if needed
                # For robust solution, consider using regex for title cleaning
                video_filepath = "".join([c for c in video_filepath if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()

        print(f"Video downloaded to: {video_filepath}")

    except yt_dlp.DownloadError as e:
        print(f"Error downloading YouTube video: {e}")
        return # Exit if download fails
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return

    if not video_filepath or not os.path.exists(video_filepath):
        print("Could not determine or find the downloaded video file. Skipping audio extraction.")
        return

    # --- Step 2: Extract Audio (using moviepy) ---
    output_audio_filename = f"{os.path.splitext(os.path.basename(video_filepath))[0]}.{audio_format}"
    output_audio_path = os.path.join(output_dir, output_audio_filename)

    try:
        print(f"Attempting to extract audio from '{video_filepath}' to '{output_audio_path}'")
        video_clip = VideoFileClip(video_filepath)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path)
        audio_clip.close()
        video_clip.close()
        print(f"Audio extracted successfully to '{output_audio_path}'")

    except Exception as e:
        print(f"An error occurred during audio extraction: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/shorts/wiRbGLk45b8" # Example YouTube URL (Rick Astley!)
    # Replace with an actual YouTube video URL you want to download and extract audio from

    # It's good practice to place your script in a dedicated directory
    # and run it from there. Let's assume your script is in
    # /home/deji/kube_proj/youtube_audio_extractor/
    # And you want downloads to go into /home/deji/kube_proj/youtube_audio_extractor/downloads

    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_folder = os.path.join(script_dir, "downloads")

    download_youtube_video_and_extract_audio(youtube_link, output_dir=download_folder, audio_format="mp3")

    print("\n--- Process Finished ---")