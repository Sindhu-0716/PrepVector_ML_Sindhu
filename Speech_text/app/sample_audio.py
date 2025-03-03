from gtts import gTTS # type: ignore

text = "Hello! This is a sample audio file for testing transcription using Whisper."
audio = gTTS(text)
audio.save("sample_audio.mp3")

print("Sample audio saved as 'sample_audio.mp3'")
