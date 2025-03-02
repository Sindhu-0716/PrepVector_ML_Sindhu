import streamlit as st
import os
import subprocess
import platform
from transcripts import TranscriptProcessor  # Import transcription script

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Speech-to-Text Transcription",
    page_icon="🎤",
    layout="wide",
)

# ✅ Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Function to Transcribe File
def transcribe_file(file_path):
    processor = TranscriptProcessor(model_size="base")
    transcript, duration = processor.get_transcripts(file_path)

    if transcript:
        st.success("✅ Transcription Completed!")
        st.markdown("**Transcript:**")
        st.text_area("", transcript, height=200)
        st.write(f"🕒 **Time Taken:** {duration:.2f} seconds")
    else:
        st.error("❌ Transcription failed.")

# ✅ Streamlit UI
st.markdown("<h1 style='text-align: center; color:#faa356;'>Speech-to-Text Transcription App 🎙️</h1>", unsafe_allow_html=True)

# 🔹 **File Upload Section**
st.write("### 📤 Upload Your Audio or Video File")
uploaded_file = st.file_uploader("Upload an MP3, WAV, or MP4 file", type=["mp3", "wav", "mp4"])

if uploaded_file:
    # ✅ Save the uploaded file
    temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write(f"📁 File uploaded: {uploaded_file.name}")

    if st.button("Transcribe File"):
        transcribe_file(temp_file_path)

# 🔹 Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.write("💡 **Tip:** Use clear audio/video with minimal background noise for better transcription accuracy.")
