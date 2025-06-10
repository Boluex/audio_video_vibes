import moviepy.editor as mp
from moviepy.video.fx.all import fadein, fadeout
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import textwrap

# --- Helper function for text overlay (using Pillow for advanced text rendering) ---
def create_text_clip_with_style(text, clip_size, text_style="Minimal", duration=None, fps=24):
    """
    Creates a text video clip with specified styling.

    Args:
        text (str): The text to display.
        clip_size (tuple): (width, height) of the clip.
        text_style (str): 'Meme Style', 'Minimal', 'Dynamic', 'Retro'.
        duration (float): Duration of the text clip in seconds. If None, it will be transparent.
        fps (int): Frames per second.

    Returns:
        moviepy.editor.ImageClip: An ImageClip with the rendered text.
    """
    width, height = clip_size
    
    # Default text properties
    font_path = None
    font_size = int(height * 0.1)
    text_color = "white"
    stroke_color = "black"
    stroke_width = 0
    bg_color = None # Transparent by default

    if text_style == "Meme Style":
        # Impact font is not standard; try a common bold system font or specify path
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" # Common bold font path on Ubuntu
        font_size = int(height * 0.15)
        text_color = "white"
        stroke_color = "black"
        stroke_width = int(font_size * 0.05)
        text_pos = ('center', 'center') # Position text in the middle
        if not os.path.exists(font_path):
            print(f"Warning: Impact-like font not found at {font_path}. Using default.")
            font_path = None # Fallback to MoviePy's default if not found

    elif text_style == "Minimal":
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Simple sans-serif
        font_size = int(height * 0.07)
        text_color = "white"
        stroke_width = 0
        text_pos = ('center', height * 0.8) # Bottom center
        if not os.path.exists(font_path):
            print(f"Warning: DejaVuSans font not found at {font_path}. Using default.")
            font_path = None

    elif text_style == "Dynamic":
        # Dynamic usually implies animation, which is harder with a static image clip.
        # We'll apply subtle animation later if needed. For now, focus on static look.
        font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" # Bold, modern
        font_size = int(height * 0.1)
        text_color = "#FFD700" # Gold color
        stroke_color = "black"
        stroke_width = int(font_size * 0.02)
        text_pos = ('center', 'center')
        if not os.path.exists(font_path):
            print(f"Warning: FreeSansBold font not found at {font_path}. Using default.")
            font_path = None

    elif text_style == "Retro":
        # Use a more blocky/serif font if available, or a system default that looks 'old'
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf" # Example serif font
        font_size = int(height * 0.08)
        text_color = "#FFA07A" # Light Salmon, vintage feel
        stroke_color = "#8B4513" # Saddle Brown
        stroke_width = int(font_size * 0.03)
        text_pos = ('center', height * 0.2) # Top center
        if not os.path.exists(font_path):
            print(f"Warning: DejaVuSerif font not found at {font_path}. Using default.")
            font_path = None
    else: # Default if style is not recognized
        font_path = None # MoviePy default
        font_size = int(height * 0.08)
        text_color = "white"
        stroke_color = "black"
        stroke_width = int(font_size * 0.03)
        text_pos = ('center', 'center')

    # Load font, with fallback
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Fallback for generic font if specified path doesn't exist or is None
            # Pillow's default is often 'DejaVuSans.ttf' or similar on Linux
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        print(f"Warning: Could not load specified font: {font_path}. Using a default font.")
        font = ImageFont.load_default() # Fallback for font loading issues


    # Wrap text to fit within width if it's too long
    # Estimate max line width (rough heuristic)
    max_line_width = int(width * 0.8 / font_size * 2) # Adjust factor as needed for font
    wrapped_text = textwrap.fill(text, width=max_line_width)

    # Create a transparent image for drawing text
    txt_img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)

    # Calculate text position (multi-line aware)
    # Get bounding box for the whole text (Pillow 10.0.0+ has getbbox)
    bbox = d.textbbox((0,0), wrapped_text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate x position
    if text_pos[0] == 'center':
        x = (width - text_width) / 2
    else:
        x = text_pos[0] # Assume numerical x if not 'center'

    # Calculate y position
    if text_pos[1] == 'center':
        y = (height - text_height) / 2
    else:
        y = text_pos[1] # Assume numerical y if not 'center'


    # Draw text with stroke (if stroke_width > 0)
    if stroke_width > 0:
        d.text((x, y), wrapped_text, font=font, fill=text_color,
               stroke_width=stroke_width, stroke_fill=stroke_color)
    else:
        d.text((x, y), wrapped_text, font=font, fill=text_color)

    # Convert PIL Image to MoviePy ImageClip
    # The duration is important here. If duration is None, it means the text is transparent
    # until it's added as an overlay on another clip. If duration is set, it's a static clip.
    text_clip = mp.ImageClip(np.array(txt_img), duration=duration)
    return text_clip.set_pos(text_pos if text_style != "Meme Style" else ("center", "center"))


def create_video_from_images_and_music(
    image_paths,
    audio_path,
    output_path="output_video.mp4",
    image_display_duration=3.0,
    transition_duration=1.0,
    music_segment_duration=20.0,
    fps=24,
    texts=None, # List of dictionaries: [{'text': 'Your text', 'start_time': 0, 'end_time': 5, 'style': 'Minimal'}, ...]
    video_size=(1280, 720) # Default video resolution
):
    """
    Creates a video from a list of images with cross-fade transitions and background music.
    Optionally adds text overlays.

    Args:
        image_paths (list): A list of paths to the image files.
        audio_path (str): The path to the audio file.
        output_path (str): The path to save the resulting video.
        image_display_duration (float): How long each image is displayed (excluding transition time).
        transition_duration (float): Duration of the cross-fade transition in seconds.
        music_segment_duration (float): Duration of the music to use in seconds.
        fps (int): Frames per second for the output video.
        texts (list): Optional list of text dictionaries for overlays.
        video_size (tuple): (width, height) for the output video.
    """
    all_clips = []
    current_time = 0

    # Ensure all images are resized to the target video_size and converted to ImageClip
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            # Resize image to fit video_size while maintaining aspect ratio
            img.thumbnail(video_size, Image.Resampling.LANCZOS) # Resize if larger
            # Create a new blank image with target_size and paste the resized image into center
            new_img = Image.new("RGB", video_size, (0, 0, 0)) # Black background
            paste_x = (video_size[0] - img.width) // 2
            paste_y = (video_size[1] - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))

            clip = mp.ImageClip(np.array(new_img), duration=image_display_duration)
            all_clips.append(clip)

            # Apply transitions if not the last image
            if i < len(image_paths) - 1:
                clip = clip.crossfadeout(transition_duration)
                next_clip = mp.ImageClip(np.array(Image.open(image_paths[i+1]).resize(video_size)), duration=image_display_duration).crossfadein(transition_duration)
                
                # Create a composite clip for the transition part
                transition_clip = mp.CompositeVideoClip([clip, next_clip]) \
                                  .subclip(image_display_duration - transition_duration, image_display_duration + transition_duration)
                all_clips.append(transition_clip)
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Skipping.")
            continue

    if not all_clips:
        print("Error: No valid images found to create video.")
        return

    final_video_clip = mp.concatenate_videoclips(all_clips, method="compose")
    total_video_duration = final_video_clip.duration

    # Add background music
    try:
        audio = mp.AudioFileClip(audio_path).subclip(0, min(music_segment_duration, total_video_duration))
        # Loop music if video is longer than the music segment duration
        if total_video_duration > music_segment_duration:
            num_loops = int(total_video_duration / audio.duration) + 1
            audio = mp.concatenate_audioclips([audio] * num_loops).subclip(0, total_video_duration)
        final_video_clip = final_video_clip.set_audio(audio)
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}. Continuing without audio.")
    except Exception as e:
        print(f"Error loading audio: {e}. Continuing without audio.")

    # Add text overlays if provided
    if texts:
        text_clips_overlays = []
        for text_data in texts:
            text_content = text_data.get('text', '')
            start_time = text_data.get('start_time', 0)
            end_time = text_data.get('end_time', total_video_duration)
            style = text_data.get('style', 'Minimal')

            text_clip = create_text_clip_with_style(text_content, video_size, style, duration=end_time - start_time, fps=fps)
            
            # Position the text dynamically for 'Meme Style' (top/bottom)
            if style == "Meme Style":
                # For meme style, usually text is at top and bottom.
                # Here, we'll try to put it at the center by default of the text clip.
                # The create_text_clip_with_style positions it, but you could adjust.
                text_clip = text_clip.set_pos(('center', 'center'))

            text_clip = text_clip.set_start(start_time).set_end(end_time)
            text_clips_overlays.append(text_clip)
        
        # Overlay all text clips onto the main video
        final_video_clip = mp.CompositeVideoClip([final_video_clip] + text_clips_overlays)


    # Write the final video
    try:
        print(f"Rendering video to: {output_path}. This might take a while...")
        final_video_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate="5000k")
        print(f"Video created successfully at: {output_path}")
    except Exception as e:
        print(f"Error writing video: {e}")
        # If there's an error during writing, it's often due to FFmpeg or codecs.
        print("Please ensure FFmpeg is correctly installed and accessible in your PATH.")
        print("You might also try different codecs or reduce bitrate if it's a memory issue.")

# --- Example Usage ---
if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "generated_videos")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- Create Dummy Files for Testing (if you don't have real ones yet) ---
    # Create 5 dummy images
    dummy_image_paths = []
    for i in range(1, 6):
        dummy_img_path = os.path.join(script_dir, f"dummy_image_{i}.png")
        if not os.path.exists(dummy_img_path):
            Image.new('RGB', (1920, 1080), color=(i*50 % 255, i*100 % 255, i*150 % 255)).save(dummy_img_path)
        dummy_image_paths.append(dummy_img_path)

    # Create a dummy audio file (replace with your actual music.mp3 for better results)
    dummy_audio_path = os.path.join(script_dir, "dummy_music.mp3")
    if not os.path.exists(dummy_audio_path):
        try:
            import wave
            import struct
            print(f"Creating a dummy audio file: {dummy_audio_path}")
            frequency = 440.0 # Hz
            duration = 30 # seconds
            sample_rate = 44100 # Hz
            amplitude = 0.5
            
            with wave.open(dummy_audio_path, 'w') as f:
                f.setnchannels(1)
                f.setsampwidth(2) # 2 bytes per sample (16-bit)
                f.setframerate(sample_rate)
                
                frames = []
                for i in range(int(duration * sample_rate)):
                    value = int(32767 * amplitude * np.sin(2 * np.pi * frequency * i / sample_rate))
                    frames.append(struct.pack('<h', value)) # '<h' for signed short
                f.writeframes(b''.join(frames))
            print("Dummy audio file created.")
        except Exception as e:
            print(f"Could not create dummy audio file (requires numpy, wave, struct): {e}. Please provide a real MP3/WAV.")
            dummy_audio_path = None # Set to None if creation fails

    # --- Video Creation Parameters ---
    image_files = dummy_image_paths # Use your actual image paths here
    audio_file = dummy_audio_path  # Use your actual audio path here
    output_video_file = os.path.join(output_folder, "my_cool_video.mp4")

    # Define text overlays (optional)
    text_overlays = [
        {'text': 'Welcome to My Journey!', 'start_time': 0, 'end_time': 4, 'style': 'Meme Style'},
        {'text': 'Beautiful Scenery', 'start_time': 5, 'end_time': 9, 'style': 'Minimal'},
        {'text': 'The Adventure Continues!', 'start_time': 10, 'end_time': 14, 'style': 'Dynamic'},
        {'text': 'A Trip Down Memory Lane', 'start_time': 15, 'end_time': 19, 'style': 'Retro'},
        {'text': 'Thanks for Watching!', 'start_time': 20, 'end_time': 24, 'style': 'Meme Style'}
    ]

    if image_files and audio_file:
        create_video_from_images_and_music(
            image_paths=image_files,
            audio_path=audio_file,
            output_path=output_video_file,
            image_display_duration=4.0,       # Each image displayed for 4 seconds
            transition_duration=1.0,          # 1 second cross-fade transition
            music_segment_duration=25.0,      # Use first 25 seconds of music
            fps=24,
            texts=text_overlays,              # Pass the text overlays here
            video_size=(1280, 720)            # Output video resolution (adjust as needed)
        )
    else:
        print("Cannot create video: Missing image files or audio file.")

    print("\n--- Video Creation Process Finished ---")