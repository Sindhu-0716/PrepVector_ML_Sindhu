import os
from gtts import gTTS # type: ignore
from moviepy.editor import ImageSequenceClip, AudioFileClip # type: ignore

# File paths
video_file = "sample_video.mp4"
audio_file = "sample_audio.mp3"
frame_dir = "frames"

# Ensure frames directory exists
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

# Create an audio narration using gTTS
text = "This is a sample video with an audio track for testing transcription."
tts = gTTS(text)
tts.save(audio_file)
print(f"✅ Audio file created: {audio_file}")

# Generate frames with moving text
from PIL import Image, ImageDraw, ImageFont # type: ignore

width, height = 1280, 720
num_frames = 30  # Number of frames for 3 seconds of video

for i in range(num_frames):
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Moving text
    text_position = (50 + i * 10, height // 2)
    draw.text(text_position, "Sample Video with Audio", fill=(255, 255, 255))
    
    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    img.save(frame_path)

print(f"✅ Frames created in {frame_dir}")

# Create video from images
clip = ImageSequenceClip(frame_dir, fps=10)
audio = AudioFileClip(audio_file)
clip = clip.set_audio(audio)

# Export the final video
clip.write_videofile(video_file, codec="libx264", audio_codec="aac", fps=10)
print(f"✅ Video file generated: {video_file}")
