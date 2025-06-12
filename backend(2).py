import os
import shutil
import uuid
from typing import List, Optional, Tuple
import math
import time # Import the time module for cleanup logic
from fastapi.middleware.cors import CORSMiddleware
# Import moviepy and its effects
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import numpy as np
import yt_dlp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, HttpUrl, Field
import textwrap
import traceback # For detailed error logging

# --- Configuration ---
UPLOAD_DIR = "uploaded_files"
OUTPUT_DIR = "processed_outputs"
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
MAX_IMAGES = 5 # Maximum number of images allowed for video creation

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Video & Audio Processing API",
    description="API for downloading videos, extracting audio, and creating videos from images and music.",
    version="1.2.6", # Version incremented
)

# --- Pydantic Models for API Endpoints ---
class FileOperationRequest(BaseModel):
    file_id: str

class ExtractAudioResponse(BaseModel):
    message: str
    audio_file_uuid: str
    audio_file_path: str

class DownloadYouTubeVideoRequest(BaseModel):
    youtube_url: HttpUrl

class DownloadYouTubeVideoResponse(BaseModel):
    message: str
    file_id: str
    video_file_path: str

class TextOverlay(BaseModel):
    text: str
    style: str = "Minimal"
    image_index: Optional[int] = Field(None, description="The 0-based index of the image for the text. If null, text appears globally.")
    position: str = Field("bottom", pattern="^(top|center|bottom)$", description="Vertical position of the text ('top', 'center', 'bottom').")

class CreateVideoFromImagesRequest(BaseModel):
    image_file_ids: List[str] = Field(..., max_length=MAX_IMAGES, description=f"List of IDs for uploaded images (max {MAX_IMAGES}).")
    audio_file_id: str
    output_filename: str = "output_video.mp4"
    image_display_duration: float = Field(3.0, gt=0)
    transition_duration: float = Field(1.0, ge=0)
    music_segment_start_time: float = Field(0.0, ge=0)
    audio_segment_duration_from_music: Optional[float] = Field(
        None, 
        gt=0,
        description="Desired duration of the audio segment to take from the music file (before looping/truncating). If None, uses audio from start_time to end of music file."
    )
    fps: int = Field(24, gt=0)
    texts: Optional[List[TextOverlay]] = None
    video_aspect_ratio: str = Field("16:9", pattern="^(16:9|9:16|1:1)$")
    enable_image_animations: bool = False

class CreateVideoFromImagesResponse(BaseModel):
    message: str
    video_file_uuid: str
    video_file_path: str
    
class UploadFileResponse(BaseModel):
    message: str
    file_id: str
    file_path: str
    filename: str

# --- Helper function to resolve File ID to a path ---
def resolve_file_id_to_path(file_id: str, directory: str = UPLOAD_DIR) -> str:
    """Finds a file in a directory based on its UUID prefix."""
    matching_files = [f for f in os.listdir(directory) if f.startswith(file_id)]
    if not matching_files:
        raise HTTPException(status_code=404, detail=f"File with ID '{file_id}' not found in {directory}.")
    return os.path.join(directory, matching_files[0])

# --- Helper function for text overlay ---
def create_text_clip_with_style(text, clip_size, text_style="Minimal", duration=None, fps=24, position="bottom"):
    width, height = clip_size
    font_path = None
    font_options = {
        "Meme Style": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf", "impact.ttf", "Impact"],
        "Minimal": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "arial.ttf", "Arial"],
        "Dynamic": ["/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/System/Library/Fonts/Supplemental/Arial-Bold.ttf", "arialbd.ttf", "Arial Bold"],
        "Retro": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf", "georgia.ttf", "Georgia"],
    }
    font_size_map = {"Meme Style": 0.15, "Minimal": 0.07, "Dynamic": 0.10, "Retro": 0.08}
    font_size = int(height * font_size_map.get(text_style, 0.08))
    font = None

    for f_name_or_path in font_options.get(text_style, font_options["Minimal"]):
        try:
            if os.path.exists(f_name_or_path):
                font = ImageFont.truetype(f_name_or_path, font_size)
                break
            else: 
                font = ImageFont.truetype(f_name_or_path, font_size)
                break
        except IOError:
            continue
    
    if font is None: 
        try:
            font = ImageFont.truetype("arial.ttf", font_size) 
        except IOError:
            font = ImageFont.load_default() 

    avg_char_width_approx = font_size * 0.5 
    max_chars_per_line = int((width * 0.9) / avg_char_width_approx) if avg_char_width_approx > 0 else 20
    wrapped_text = "\n".join(textwrap.wrap(text, width=max_chars_per_line, break_long_words=True, replace_whitespace=False))
    
    temp_img_for_text_measure = Image.new('RGBA', (1,1))
    temp_draw = ImageDraw.Draw(temp_img_for_text_measure)
    try:
        text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font, stroke_width=2)
    except AttributeError:
         text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)
    del temp_draw
    del temp_img_for_text_measure

    text_block_width = text_bbox[2] - text_bbox[0]
    text_block_height = text_bbox[3] - text_bbox[1]

    txt_img = Image.new('RGBA', clip_size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)
    
    draw_x = (width - text_block_width) / 2
    if position == "top": draw_y = height * 0.1
    elif position == "center": draw_y = (height - text_block_height) / 2
    else: draw_y = height * 0.9 - text_block_height
    draw_y = max(0, draw_y)

    d.text((draw_x, draw_y), wrapped_text, font=font, fill="white", stroke_width=2, stroke_fill="black", align="center")
    
    actual_duration = duration if duration and duration > 0 else None
    return mp.ImageClip(np.array(txt_img), transparent=True, duration=actual_duration).set_position('center')


# --- Core Logic Functions ---

def _extract_audio_from_video_logic(video_path: str, output_audio_path: str):
    try:
        with mp.VideoFileClip(video_path) as video:
            if video.audio is None:
                return False, "The provided video file does not contain an audio track."
            video.audio.write_audiofile(output_audio_path, logger=None)
        return True, f"Audio extracted successfully to '{output_audio_path}'"
    except Exception as e:
        return False, f"An error occurred during audio extraction: {e}"

def _download_youtube_video_logic(youtube_url: str, output_dir: str):
    file_id = str(uuid.uuid4())
    output_template = os.path.join(output_dir, f'{file_id}.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'merge_output_format': 'mp4',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            downloaded_file_path = os.path.join(output_dir, f"{file_id}.mp4")
            if not os.path.exists(downloaded_file_path):
                found_files = [f for f in os.listdir(output_dir) if f.startswith(file_id) and f.endswith(info_dict.get('ext', 'mp4'))]
                if found_files:
                    downloaded_file_path = os.path.join(output_dir, found_files[0])
                else:
                    raise FileNotFoundError(f"yt-dlp downloaded video, but expected file {file_id}.mp4 or similar not found.")
        return True, (file_id, downloaded_file_path)
    except Exception as e:
        return False, f"Error downloading YouTube video: {e}"

# ... (imports and other functions remain the same) ...

def _create_video_from_images_and_music_logic(
    image_paths: List[str], audio_path: str, output_path: str,
    image_display_duration: float, transition_duration: float,
    music_segment_start_time: float, 
    audio_segment_duration_from_music: Optional[float],
    fps: int, texts: Optional[List[TextOverlay]],
    video_size: Tuple[int, int], enable_image_animations: bool
):
    processed_image_clips = []
    video_w, video_h = video_size 
    
    animation_type = "up_down"
    if enable_image_animations and texts and len(texts) == 1 and texts[0].image_index is None:
        animation_type = "rotate"

    base_video = None
    final_video_clip = None
    text_clips_to_composite = []
    
    # Declare audio clips here for the finally block
    full_audio_clip_main = None # Renamed for clarity
    temp_audio_segment_intermediate = None
    audio_segment_final_for_video = None


    try:
        # --- IMAGE PROCESSING ---
        # (This part is assumed to be working correctly based on prior logs)
        # (Keeping it brief here for focus)
        for i, img_path in enumerate(image_paths):
            # ... (full image processing loop as in your last version) ...
            # Creates final_clip_for_this_image_slot
            # processed_image_clips.append(final_clip_for_this_image_slot)
            # Ensure img_pil is closed within this loop's try/finally
            pass # Placeholder for brevity, use your full image loop here

        # Re-paste your full image processing loop here
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i}: {img_path}")
            img_pil = None 
            try:
                img_pil = Image.open(img_path).convert("RGB")
                base_image_clip = mp.ImageClip(np.array(img_pil)).set_duration(image_display_duration)
                if not isinstance(base_image_clip, mp.VideoClip):
                    raise ValueError(f"ImageClip creation failed for {img_path}")
                # print(f"  Image {i} loaded. Duration: {base_image_clip.duration}, Size: {base_image_clip.size}")

                img_w_orig, img_h_orig = base_image_clip.w, base_image_clip.h
                img_aspect = img_w_orig / img_h_orig if img_h_orig > 0 else 1
                video_aspect = video_w / video_h if video_h > 0 else 1

                if img_aspect > video_aspect:
                    resized_clip = base_image_clip.resize(height=video_h)
                else:
                    resized_clip = base_image_clip.resize(width=video_w)
                
                cropped_clip = resized_clip.crop(x_center=resized_clip.w / 2, y_center=resized_clip.h / 2, width=video_w, height=video_h)
                cropped_clip = cropped_clip.set_duration(image_display_duration)
                # print(f"  Image {i} resized & cropped to target. Duration: {cropped_clip.duration}, Size: {cropped_clip.size}")

                current_image_canvas_clip = cropped_clip

                if enable_image_animations:
                    # print(f"  Image {i} applying animation: {animation_type}")
                    animated_content_clip = None

                    if animation_type == "up_down":
                        content_to_animate = cropped_clip.resize(1.1)
                        if not isinstance(content_to_animate, mp.VideoClip):
                            raise ValueError("Resize for up_down animation failed.")
                        
                        base_y_offset = (video_h - content_to_animate.h) / 2
                        pan_amplitude = (content_to_animate.h - video_h) / 2
                        current_image_effect_duration = cropped_clip.duration

                        def get_y_position(t, b_offset=base_y_offset, p_amp=pan_amplitude, p_period=current_image_effect_duration):
                            if p_period is None or p_period == 0: return b_offset
                            return b_offset + p_amp * math.sin((2 * math.pi * t) / p_period)
                        
                        animated_content_clip = content_to_animate.set_position(
                            lambda t, b=base_y_offset, pa=pan_amplitude, p=current_image_effect_duration: ('center', get_y_position(t, b, pa, p))
                        )
                        animated_content_clip = animated_content_clip.set_duration(current_image_effect_duration)

                    elif animation_type == "rotate":
                        animated_content_clip = cropped_clip.fx(vfx.rotate, lambda t: 5 * t, expand=False).resize(1.2)
                        animated_content_clip = animated_content_clip.set_duration(cropped_clip.duration)
                    
                    if animated_content_clip and isinstance(animated_content_clip, mp.VideoClip):
                        pil_transparent_img = Image.new("RGBA", (video_w, video_h), (0,0,0,0)) 
                        background_canvas = mp.ImageClip(np.array(pil_transparent_img), duration=cropped_clip.duration)
                        del pil_transparent_img 

                        current_image_canvas_clip = mp.CompositeVideoClip(
                            [background_canvas, animated_content_clip.set_position('center')], 
                            size=(video_w, video_h),
                            use_bgclip=True 
                        ).set_duration(cropped_clip.duration)
                        # print(f"  Image {i} animation applied and composited. Canvas Duration: {current_image_canvas_clip.duration}, Canvas Size: {current_image_canvas_clip.size}, Animated Content Size: {animated_content_clip.size if animated_content_clip else 'N/A'}")
                    # else:
                        # print(f"  Image {i} animation failed. Using static cropped clip.")
                        # current_image_canvas_clip = cropped_clip # already set
                
                final_clip_for_this_image_slot = current_image_canvas_clip
                if i < len(image_paths) - 1 and transition_duration > 0:
                    if final_clip_for_this_image_slot.duration is None or final_clip_for_this_image_slot.duration <= 0:
                        raise ValueError(f"Clip for image {i} has invalid duration before crossfade.")
                    
                    actual_transition = transition_duration
                    if transition_duration >= final_clip_for_this_image_slot.duration:
                        # print(f"Warning: transition_duration ({transition_duration}) >= clip.duration ({final_clip_for_this_image_slot.duration}) for image {i}. Adjusting transition.")
                        actual_transition = max(0.1, final_clip_for_this_image_slot.duration * 0.5)
                    
                    final_clip_for_this_image_slot = final_clip_for_this_image_slot.crossfadeout(actual_transition)
                    # print(f"  Image {i} crossfade applied. Final slot duration: {final_clip_for_this_image_slot.duration}, Size: {final_clip_for_this_image_slot.size}")
                
                if not isinstance(final_clip_for_this_image_slot, mp.VideoClip) or final_clip_for_this_image_slot.duration is None or final_clip_for_this_image_slot.duration <= 0:
                    raise ValueError(f"Processed clip for image {i} is invalid. Size: {final_clip_for_this_image_slot.size if final_clip_for_this_image_slot else 'None'}")
                
                processed_image_clips.append(final_clip_for_this_image_slot)
            
            finally:
                if img_pil and hasattr(img_pil, 'close'):
                    img_pil.close()
                    img_pil = None
        # End of re-pasted image loop

        if not processed_image_clips:
            return False, "No valid image clips could be created."
        print("All image clips processed.")

        base_video = mp.concatenate_videoclips(processed_image_clips, method="compose")
        if hasattr(base_video, 'size') and base_video.size != video_size:
            print(f"CRITICAL WARNING: base_video size after concatenation is {base_video.size}, "
                  f"but expected {video_size}.")

        if not isinstance(base_video, mp.VideoClip) or base_video.duration is None or base_video.duration <= 0:
            size_info = f"Size: {base_video.size if hasattr(base_video, 'size') else 'N/A'}"
            duration_info = f"Duration: {base_video.duration if hasattr(base_video, 'duration') else 'N/A'}"
            raise ValueError(f"Base video concatenation resulted in an invalid clip. {size_info}, {duration_info}")
        total_video_duration = base_video.duration
        print(f"Base video concatenated. Total duration: {total_video_duration}, Size: {base_video.size if hasattr(base_video, 'size') else 'N/A'}")

        if texts:
            # ... (text processing loop as in your last version) ...
            pass # Placeholder
        # Re-paste text loop
        if texts:
            for text_data in texts:
                text_start_time = 0
                text_duration = total_video_duration 
                if text_data.image_index is not None and 0 <= text_data.image_index < len(processed_image_clips):
                    current_start_time = 0
                    for k_idx in range(text_data.image_index):
                        clip_eff_duration = image_display_duration
                        if k_idx < len(image_paths) - 1 and transition_duration > 0:
                            actual_trans = min(transition_duration, image_display_duration * 0.5) 
                            clip_eff_duration -= actual_trans
                        current_start_time += clip_eff_duration
                    text_start_time = current_start_time
                    text_duration = image_display_duration 
                
                if text_duration <=0: continue

                txt_clip_candidate = create_text_clip_with_style(
                    text_data.text, video_size, text_data.style, text_duration, fps, text_data.position
                )
                txt_clip_candidate = txt_clip_candidate.set_start(text_start_time).set_duration(text_duration)
                
                if txt_clip_candidate.duration is not None and txt_clip_candidate.duration > 0:
                    if txt_clip_candidate.end > total_video_duration:
                        txt_clip_candidate = txt_clip_candidate.set_duration(max(0, total_video_duration - txt_clip_candidate.start))
                    if txt_clip_candidate.duration > 0:
                        text_clips_to_composite.append(txt_clip_candidate)
            print(f"Processed {len(text_clips_to_composite)} text clips.")
        # End of re-pasted text loop


        if text_clips_to_composite:
            if hasattr(base_video, 'size') and base_video.size != video_size:
                print(f"Warning: base_video size {base_video.size} != target {video_size} before text composite. Resizing.")
                base_video = base_video.resize(newsize=video_size)
            all_visual_layers = [base_video] + text_clips_to_composite
            final_video_clip = mp.CompositeVideoClip(all_visual_layers, size=video_size)
            final_video_clip = final_video_clip.set_duration(total_video_duration) 
            print(f"Text clips composited. Final clip size: {final_video_clip.size}") 
        else:
            final_video_clip = base_video 
            if hasattr(final_video_clip, 'size') and final_video_clip.size != video_size:
                print(f"Warning: Final clip (no text) size {final_video_clip.size} != target {video_size}. Forcing resize.")
                final_video_clip = final_video_clip.resize(newsize=video_size)
            
            if final_video_clip.duration != total_video_duration:
                print(f"Warning: Final clip duration {final_video_clip.duration} != total {total_video_duration}. Forcing.")
                final_video_clip = final_video_clip.set_duration(total_video_duration)
            print(f"No text clips. Using base video as final. Final clip size: {final_video_clip.size if hasattr(final_video_clip, 'size') else 'N/A'}") 
        
        if not isinstance(final_video_clip, mp.VideoClip) or final_video_clip.duration is None or final_video_clip.duration <= 0:
            size_info = f"Size: {final_video_clip.size if hasattr(final_video_clip, 'size') else 'N/A'}"
            raise ValueError(f"Final visual clip is invalid. {size_info}")
        print(f"Final visual clip ready. Duration: {final_video_clip.duration}, Size: {final_video_clip.size if hasattr(final_video_clip, 'size') else 'N/A'}")

        # --- AUDIO PROCESSING (REVISED) ---
        print(f"Attempting to load audio from: {audio_path}")
        if not os.path.exists(audio_path):
            return False, f"Audio file does not exist at path: {audio_path}"
        if not os.access(audio_path, os.R_OK):
            return False, f"Audio file is not readable at path: {audio_path}"
        if os.path.getsize(audio_path) == 0:
            return False, f"Audio file is empty at path: {audio_path}"

        try:
            full_audio_clip_main = mp.AudioFileClip(audio_path) # Not using 'with' here
            
            print(f"DEBUG AUDIO: full_audio_clip_main created. Duration: {full_audio_clip_main.duration if hasattr(full_audio_clip_main, 'duration') else 'N/A'}")
            if not hasattr(full_audio_clip_main, 'reader') or full_audio_clip_main.reader is None:
                print("CRITICAL DEBUG AUDIO: full_audio_clip_main.reader is None or missing!")
                # full_audio_clip_main.close() done in finally
                return False, "Failed to initialize audio reader for the main audio file."
            else:
                print(f"DEBUG AUDIO: full_audio_clip_main.reader type: {type(full_audio_clip_main.reader)}")
                try:
                    _ = full_audio_clip_main.get_frame(0.01)
                    print("DEBUG AUDIO: Successfully got a test frame from full_audio_clip_main.reader.")
                except Exception as e_get_frame:
                    print(f"CRITICAL DEBUG AUDIO: Failed to get test frame from full_audio_clip_main.reader: {e_get_frame}")
                    # full_audio_clip_main.close() done in finally
                    return False, f"Audio reader for main file failed on get_frame: {e_get_frame}"

            if music_segment_start_time >= full_audio_clip_main.duration:
                return False, "Music start time is beyond the audio's total duration."
            
            subclip_end_time = None
            if audio_segment_duration_from_music is not None:
                subclip_end_time = music_segment_start_time + audio_segment_duration_from_music
            
            temp_audio_segment_intermediate = full_audio_clip_main.subclip(music_segment_start_time, subclip_end_time)
            print(f"DEBUG AUDIO: temp_audio_segment_intermediate created. Duration: {temp_audio_segment_intermediate.duration}")
            
            if temp_audio_segment_intermediate.duration == 0:
                final_video_clip = final_video_clip.set_audio(None)
                audio_segment_final_for_video = None 
                # temp_audio_segment_intermediate.close() # Closed in finally if distinct and not None
            elif temp_audio_segment_intermediate.duration < total_video_duration:
                audio_segment_final_for_video = temp_audio_segment_intermediate.fx(vfx.audio_loop, duration=total_video_duration)
                print(f"DEBUG AUDIO: audio_segment_final_for_video (looped). Duration: {audio_segment_final_for_video.duration}")
            else: 
                audio_segment_final_for_video = temp_audio_segment_intermediate.subclip(0, total_video_duration)
                print(f"DEBUG AUDIO: audio_segment_final_for_video (subclipped). Duration: {audio_segment_final_for_video.duration}")

            if audio_segment_final_for_video:
                final_video_clip = final_video_clip.set_audio(audio_segment_final_for_video)
            # else audio already set to None if temp_audio_segment_intermediate.duration was 0

        except Exception as e_audio_block:
            print(f"ERROR during main audio processing block: {type(e_audio_block).__name__} - {e_audio_block}")
            traceback.print_exc()
            # Relevant clips will be closed in the main finally block
            return False, f"Failed during audio processing: {e_audio_block}"
        # --- END AUDIO PROCESSING ---

        print("Audio processed and attached.")
        
        if final_video_clip.duration is None or final_video_clip.duration <= 0 :
            return False, f"Error: Video has no duration ({final_video_clip.duration}) before writing."
        
        if final_video_clip.audio is None:
            print("FINAL CHECK: Final video clip has no audio track. This might be intended.")
        else:
            print(f"FINAL CHECK: Final video clip HAS audio. Duration: {final_video_clip.audio.duration}")
            if not hasattr(final_video_clip.audio, 'make_frame') or final_video_clip.audio.make_frame is None:
                 print("CRITICAL FINAL CHECK: final_video_clip.audio.make_frame is None or missing!")
                 return False, "Audio track on final video is invalid (no make_frame)."
            try:
                _ = final_video_clip.audio.get_frame(0.01)
                print("FINAL CHECK: Successfully got a test frame from final_video_clip.audio.")
            except Exception as e_final_audio_get_frame:
                print(f"CRITICAL FINAL CHECK: Failed to get test frame from final_video_clip.audio: {e_final_audio_get_frame}")
                traceback.print_exc()
                return False, f"Final audio track failed on get_frame: {e_final_audio_get_frame}"

        final_video_clip.write_videofile(
            output_path, fps=fps, codec='libx264', audio_codec='aac', threads=4, logger=None 
        )
        print("Video file written successfully.")

    except Exception as e:
        print(f"Error during video creation logic: {type(e).__name__} - {e}")
        traceback.print_exc() 
        return False, f"Internal error during video generation: {type(e).__name__} - {e}"
    finally:
        print("--- Entering main finally block for cleanup ---")
        if final_video_clip and hasattr(final_video_clip, 'close'): 
            print("Closing final_video_clip")
            final_video_clip.close()
        if base_video and hasattr(base_video, 'close'): 
            print("Closing base_video")
            base_video.close()
        for i_clip, clip in enumerate(processed_image_clips):
            if hasattr(clip, 'close'): 
                print(f"Closing processed_image_clip {i_clip}")
                clip.close()
        for i_clip, clip in enumerate(text_clips_to_composite):
            if hasattr(clip, 'close'): 
                print(f"Closing text_clip {i_clip}")
                clip.close()
        
        # Close audio clips in reverse order of dependency or if they are distinct objects
        if audio_segment_final_for_video and hasattr(audio_segment_final_for_video, 'close'):
            # If this is the same object as temp_audio_segment_intermediate, it's closed once.
            # If it's different (e.g., after a loop or subclip that creates a new obj), close it.
            print("Closing audio_segment_final_for_video")
            audio_segment_final_for_video.close()
        
        if temp_audio_segment_intermediate and hasattr(temp_audio_segment_intermediate, 'close') and \
           (audio_segment_final_for_video is None or temp_audio_segment_intermediate is not audio_segment_final_for_video):
            # Close temp only if it's a distinct intermediate object and hasn't been closed
            print("Closing temp_audio_segment_intermediate")
            temp_audio_segment_intermediate.close()
            
        if full_audio_clip_main and hasattr(full_audio_clip_main, 'close'):
            print("Closing full_audio_clip_main")
            full_audio_clip_main.close()
        print("--- Exiting main finally block ---")

    return True, output_path

# --- API Endpoints ---
# (Remain the same as your last full version)
# ... (copy and paste all your endpoint functions here: upload_file, extract_audio, etc.) ...
@app.post("/uploadfile/", response_model=UploadFileResponse, summary="Upload a file (video, image, or audio)")
async def upload_file(file: UploadFile = File(...)):
    if file.size is None or file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB or size unknown.")
    try:
        file_id = str(uuid.uuid4())
        original_filename = file.filename if file.filename else "unknown"
        file_extension = os.path.splitext(original_filename)[1]
        if not file_extension:
            content_type = file.content_type
            if content_type == "video/mp4": file_extension = ".mp4"
            elif content_type == "audio/mpeg": file_extension = ".mp3"
            elif content_type == "image/jpeg": file_extension = ".jpg"
            elif content_type == "image/png": file_extension = ".png"
            else: file_extension = ".bin"
            
        file_location = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return UploadFileResponse(message="File uploaded successfully", file_id=file_id, file_path=file_location, filename=original_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        if file:
            await file.close()

@app.post("/extract-audio/", response_model=ExtractAudioResponse, summary="Extract audio from an uploaded video")
async def extract_audio(request: FileOperationRequest):
    try:
        video_input_path = resolve_file_id_to_path(request.file_id, UPLOAD_DIR)
    except HTTPException as e:
        raise e

    output_audio_uuid = str(uuid.uuid4())
    output_audio_path = os.path.join(OUTPUT_DIR, f"{output_audio_uuid}.mp3")
    success, message_or_path = _extract_audio_from_video_logic(video_input_path, output_audio_path)

    if success:
        return ExtractAudioResponse(message="Audio extracted.", audio_file_uuid=output_audio_uuid, audio_file_path=output_audio_path)
    else:
        if os.path.exists(output_audio_path):
            try: os.remove(output_audio_path)
            except OSError: pass
        raise HTTPException(status_code=500, detail=message_or_path)

@app.post("/download-youtube-video/", response_model=DownloadYouTubeVideoResponse, summary="Download a YouTube video")
async def download_youtube_video(request: DownloadYouTubeVideoRequest):
    success, result = _download_youtube_video_logic(str(request.youtube_url), UPLOAD_DIR)
    if success:
        file_id, file_path = result
        return DownloadYouTubeVideoResponse(message="YouTube video downloaded successfully.", file_id=file_id, video_file_path=file_path)
    else:
        raise HTTPException(status_code=500, detail=result)

@app.post("/create-video-from-images/", response_model=CreateVideoFromImagesResponse, summary="Create video from images, music, and text")
async def create_video_from_images(request: CreateVideoFromImagesRequest):
    try:
        resolved_image_paths = [resolve_file_id_to_path(img_id, UPLOAD_DIR) for img_id in request.image_file_ids]
        
        resolved_audio_path = None
        try:
            resolved_audio_path = resolve_file_id_to_path(request.audio_file_id, UPLOAD_DIR)
        except HTTPException:
            try:
                resolved_audio_path = resolve_file_id_to_path(request.audio_file_id, OUTPUT_DIR)
            except HTTPException as e_out:
                raise HTTPException(status_code=404, detail=f"Audio file ID '{request.audio_file_id}' not found in uploads or outputs.") from e_out

    except HTTPException as e:
        raise e 

    aspect_ratios = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
    target_video_size = aspect_ratios.get(request.video_aspect_ratio)

    output_video_uuid = str(uuid.uuid4())
    output_video_path = os.path.join(OUTPUT_DIR, f"{output_video_uuid}.mp4")

    success, result_path_or_message = _create_video_from_images_and_music_logic(
        image_paths=resolved_image_paths, audio_path=resolved_audio_path, output_path=output_video_path,
        image_display_duration=request.image_display_duration, transition_duration=request.transition_duration,
        music_segment_start_time=request.music_segment_start_time,
        audio_segment_duration_from_music=request.audio_segment_duration_from_music,
        fps=request.fps, texts=request.texts, video_size=target_video_size,
        enable_image_animations=request.enable_image_animations
    )

    if success:
        return CreateVideoFromImagesResponse(
            message="Video created successfully.",
            video_file_uuid=output_video_uuid,
            video_file_path=result_path_or_message
        )
    else:
        if os.path.exists(output_video_path):
            try: os.remove(output_video_path)
            except OSError: pass
        raise HTTPException(status_code=500, detail=result_path_or_message)

@app.get("/download-result/{file_uuid}", summary="Download a processed file")
async def download_result(file_uuid: str):
    file_path = ""
    try:
        file_path = resolve_file_id_to_path(file_uuid, OUTPUT_DIR)
    except HTTPException:
        try:
            file_path = resolve_file_id_to_path(file_uuid, UPLOAD_DIR)
        except HTTPException as e_upload:
             raise HTTPException(status_code=404, detail=f"File with UUID '{file_uuid}' not found in outputs or uploads.") from e_upload
    
    base_filename = os.path.basename(file_path)
    return FileResponse(path=file_path, filename=base_filename, media_type="application/octet-stream")


@app.delete("/cleanup-old-files/", summary="Clean up old uploaded and processed files")
async def cleanup_old_files(age_in_hours: int = Form(1, ge=1)):
    cutoff = time.time() - (age_in_hours * 3600)
    files_deleted_count = 0
    deletion_errors = []
    
    for directory_to_clean in [UPLOAD_DIR, OUTPUT_DIR]:
        if not os.path.isdir(directory_to_clean): continue
        for filename in os.listdir(directory_to_clean):
            filepath = os.path.join(directory_to_clean, filename)
            try:
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    files_deleted_count += 1
            except Exception as e:
                deletion_errors.append(f"Error deleting {filepath}: {e}")
    
    return {
        "message": f"Cleanup complete. Deleted {files_deleted_count} old file(s).",
        "errors": deletion_errors
    }



# In your FastAPI main.py / backend(2).py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ... other imports ...

# Define allowed origins
origins = [
    "http://localhost:8081", # Your Vite frontend development URL
    "http://localhost:5173", # Another common Vite port
    "http://localhost:3000", # Common Create React App port
    # Add your production frontend URL here later, e.g., "https://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],    # Allows all methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],    # Allows all headers
)
