import os
import whisper # type: ignore
import time
import subprocess

# Manually set ffmpeg path (update this to match your installation)
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\ffmpeg\\bin"

# Supported file formats
AUDIO_FORMATS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")
VIDEO_FORMATS = (".mp4", ".mov", ".avi", ".mkv")

class TranscriptProcessor:
    def __init__(self, model_size="large"):
        """Initialize Whisper model"""
        self.transcription_engine = whisper.load_model(model_size)

    def extract_audio(self, video_path):
        """Extract audio from a video file using ffmpeg"""
        audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
        print(f"🎥 Extracting audio from: {video_path}")

        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Audio extracted: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg Error: {e}")
            return None

    def get_transcripts(self, file_path):
        """Generate transcript from an audio or video file"""
        print(f"📂 Checking file path: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' not found.")
            return None, None

        # Convert video to audio if necessary
        if file_path.lower().endswith(VIDEO_FORMATS):
            file_path = self.extract_audio(file_path)
            if not file_path:
                return None, None  # If extraction fails

        print(f"✅ Processing file: {file_path}")
        try:
            start_time = time.time()
            result = self.transcription_engine.transcribe(file_path, fp16=False)
            end_time = time.time()
            time_taken = end_time - start_time
            return result["text"], time_taken
        except Exception as e:
            print(f"❌ Error Transcribing File: {e}")
            return None, None

if __name__ == "__main__":
    file_path = input("Enter the full path to the audio or video file: ").strip()
    
    processor = TranscriptProcessor(model_size="base")
    transcript, duration = processor.get_transcripts(file_path)

    if transcript:
        print("\n--- Transcript ---\n")
        print(transcript)
        print(f"\nTime taken: {duration:.2f} seconds")
    else:
        print("❌ Transcription failed.")
