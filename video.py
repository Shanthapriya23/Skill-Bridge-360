import os
import torch
import moviepy.editor as mp
from faster_whisper import WhisperModel
from transformers import pipeline

# Constants
VIDEO_PATH = "downloaded_video.mp4"  # Change this to your actual file
OUTPUT_VIDEO = "summary_video.mp4"
TARGET_DURATION = 180  # 3 minutes
ASPECT_RATIO = (720, 1280)  # Lower resolution for faster processing
MAX_WORDS = 250  # Words per chunk

# Step 1: Extract Audio from Video
def extract_audio(video_path, audio_path="audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=16000)
    return audio_path

# Step 2: Transcribe Audio using Faster-Whisper (Tiny Model for Speed)
def transcribe_audio(audio_path):
    model = WhisperModel("tiny", compute_type="int8", device="cpu")  # Use tiny model for speed
    segments, _ = model.transcribe(audio_path)
    transcript = " ".join(segment.text for segment in segments)
    return transcript

# Step 3: Chunk Text Efficiently
def chunk_text(text, max_words=MAX_WORDS):
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

# Step 4: Summarize Transcript using Batch Processing
def summarize_text(text_chunks):
    summarizer = pipeline("summarization", model="t5-small", device=-1)  # CPU usage
    return " ".join(summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"] for chunk in text_chunks)

# Step 5: Extract Key Moments in Video
def get_key_moments(video_path, num_clips=5):
    video = mp.VideoFileClip(video_path)
    clip_length = TARGET_DURATION // num_clips  # Divide into equal-length clips
    return [(i * clip_length, min((i + 1) * clip_length, video.duration)) for i in range(num_clips)]

# Step 6: Create Summary Video (Lower Resolution for Speed)
def create_summary_video(video_path, timestamps, output_path):
    clips = [mp.VideoFileClip(video_path).subclip(start, end).resize(ASPECT_RATIO) for start, end in timestamps]
    final_video = mp.concatenate_videoclips(clips)
    final_video.write_videofile(output_path, codec="libx264", fps=24, preset="fast")

# Execution
audio_path = extract_audio(VIDEO_PATH)
transcript = transcribe_audio(audio_path)
text_chunks = chunk_text(transcript)
summary = summarize_text(text_chunks)
timestamps = get_key_moments(VIDEO_PATH)
create_summary_video(VIDEO_PATH, timestamps, OUTPUT_VIDEO)

print("✅ Summary video created successfully!")
