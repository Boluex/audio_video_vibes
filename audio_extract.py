from moviepy.editor import VideoFileClip
import yt_dlp
import os

# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
# from moviepy.video.io import VideoFileClip

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): The path to the input video file.
        output_audio_path (str): The path where the extracted audio will be saved.
                                 Common formats are .mp3, .wav, .aac.
    """
    try:
        # Load the video file
        video = VideoFileClip(video_path)

        # Extract the audio
        audio = video.audio

        # Write the audio to a file
        audio.write_audiofile(output_audio_path)

        # Close the video and audio objects
        audio.close()
        video.close()

        print(f"Audio extracted successfully from '{video_path}' to '{output_audio_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy video file for testing (you'll replace this with your actual video)
    # For demonstration, let's assume you have a video named 'my_video.mp4'
    # in the same directory as your Python script.
    # If not, you'll need to provide the full path to your video file.

    input_video_file = "/home/deji/kube_proj/audio_extract/when_we_pray_chant.mp4"  # Replace with your video file path
    output_audio_file = "when_we_pray_chant.mp3" # You can choose .wav, .aac, etc.

    # Before running, make sure 'my_video.mp4' exists or change the path.
    # For a quick test, you can download any short .mp4 video.

    extract_audio_from_video(input_video_file, output_audio_file)


















