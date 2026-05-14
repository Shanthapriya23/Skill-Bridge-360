import os
import cv2
import numpy as np
import torch
import moviepy.editor as mp
from faster_whisper import WhisperModel
from transformers import pipeline
import warnings
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import numpy as np
import torch
import moviepy.editor as mp
from faster_whisper import WhisperModel
from transformers import pipeline
import warnings
import moviepy.editor as mp
from moviepy.video.fx.all import crop

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# Constants
VIDEO_PATH = "downloaded_video.mp4"  # Input video file
OUTPUT_VIDEO = "summary_video.mp4"   # Output summarized video
TARGET_DURATION = 180  # Max summary video length in seconds (3 minutes)
MAX_WORDS = 250  # Max words per chunk for summarization
AUDIO_PATH = "audio.wav"

# [... Keep the previous imports and constants ...]
# Step 1: Extract Audio from Video
def extract_audio(video_path, audio_path=AUDIO_PATH):
    if not os.path.exists(video_path):
        print(f"⚠️ Video file not found: {video_path}")
        exit()

    print("🎵 Extracting audio from video...")
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=16000)
        return audio_path
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        exit()

# Step 2: Transcribe Audio using Faster-Whisper
def transcribe_audio(audio_path):
    print("📝 Transcribing audio...")
    if not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        exit()

    try:
        model = WhisperModel("small", compute_type="int8", device="cpu")
        segments, _ = model.transcribe(audio_path)
        transcript = " ".join(segment.text for segment in segments)
        return transcript
    except Exception as e:
        print(f"❌ Error in transcription: {e}")
        exit()

# Step 3: Chunk Text
def chunk_text(text, max_words=MAX_WORDS):
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

# Step 4: Summarize Text with Dynamic Length Handling
def summarize_text(text_chunks):
    print("🔍 Summarizing transcript...")
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        summaries = [
            summarizer(chunk, max_length=min(100, len(chunk.split()) // 2 + 10), 
                       min_length=max(20, len(chunk.split()) // 4), do_sample=False)[0]["summary_text"]
            for chunk in text_chunks
        ]
        summary_text = " ".join(summaries)
        print(f"✅ Summary:\n{summary_text}\n")
        return summary_text
    except Exception as e:
        print(f"❌ Error in summarization: {e}")
        exit()

# Step 5: Scene Detection with Improved Selection
def detect_scene_changes(video_path, num_clips=5):
    print("🎬 Detecting scene changes...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video file.")
            exit()

        frame_diffs = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_frame = None
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                frame_diffs.append(np.sum(diff))
            prev_frame = gray

        cap.release()

        if not frame_diffs:
            print("⚠️ No significant frame changes detected.")
            return []

        threshold = np.percentile(frame_diffs, 90)
        scene_changes = [i / fps for i, diff in enumerate(frame_diffs) if diff > threshold]

        if len(scene_changes) < 2:
            print("⚠️ Not enough scene changes detected.")
            return []

        num_scenes = min(num_clips, len(scene_changes))
        selected_timestamps = np.linspace(0, len(scene_changes) - 1, num_scenes, dtype=int)
        key_moments = [scene_changes[i] for i in selected_timestamps]

        print(f"✅ Detected {len(key_moments)} key moments:", key_moments)
        return key_moments
    except Exception as e:
        print(f"❌ Error detecting scenes: {e}")
        exit()

# New function to align summary with key timestamps
def align_summary_with_scenes(transcript, summary, key_timestamps, video_duration):
    print("🔗 Aligning summary with scene timestamps...")
    try:
        # Use sentence transformer for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sentence tokenization
        transcript_sentences = re.split(r'(?<=[.!?])\s+', transcript)
        summary_sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        # Encode sentences
        transcript_embeddings = model.encode(transcript_sentences)
        summary_embeddings = model.encode(summary_sentences)
        
        # Find most relevant transcript sentences to summary
        relevant_timestamps = []
        for summary_emb in summary_embeddings:
            # Compute similarity with transcript sentences
            similarities = cosine_similarity([summary_emb], transcript_embeddings)[0]
            
            # Find the index of most similar sentence
            most_similar_idx = np.argmax(similarities)
            
            # Estimate timestamp for this sentence (proportional to its position in transcript)
            estimated_timestamp = (most_similar_idx / len(transcript_sentences)) * video_duration
            
            # Find the closest key timestamp
            closest_timestamp = min(key_timestamps, key=lambda x: abs(x - estimated_timestamp))
            
            if closest_timestamp not in relevant_timestamps:
                relevant_timestamps.append(closest_timestamp)
        
        # Ensure we have at least 2 timestamps
        if len(relevant_timestamps) < 2:
            relevant_timestamps = key_timestamps[:min(5, len(key_timestamps))]
        
        print(f"✅ Aligned {len(relevant_timestamps)} timestamps with summary")
        return sorted(relevant_timestamps)
    
    except Exception as e:
        print(f"❌ Error aligning summary with scenes: {e}")
        return key_timestamps  # Fallback to original timestamps

# Modify the create_summary_video function to use aligned timestamps
def create_summary_video(video_path, timestamps, output_path, target_duration=60):
    print("🎥 Creating summary reel with summary-aligned timestamps...")
    try:
        video = mp.VideoFileClip(video_path)
        original_width, original_height = video.size
        aspect_ratio = original_width / original_height
        target_width, target_height = 1080, 1920  # 9:16 format

        if len(timestamps) < 2:
            print("⚠️ Not enough timestamps for summarization.")
            return

        # Scale duration to fit within target_duration
        total_duration = sum(timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1))
        scale_factor = min(1.0, target_duration / total_duration) if total_duration > target_duration else 1.0

        clips = []
        for i in range(len(timestamps) - 1):
            start, end = timestamps[i], timestamps[i + 1]
            duration = (end - start) * scale_factor
            try:
                clip = video.subclip(start, min(start + duration, video.duration))

                # Fit video into 720x1280 without cropping
                if aspect_ratio > (target_width / target_height):  # Video is wider
                    clip = clip.resize(width=target_width)  # Fit width
                else:  # Video is taller
                    clip = clip.resize(height=target_height)  # Fit height

                # Add black background (padding) if necessary
                final_clip = mp.CompositeVideoClip([
                    mp.ColorClip((target_width, target_height), color=(0, 0, 0)).set_duration(clip.duration), 
                    clip.set_position("center")
                ])

                clips.append(final_clip)
            except Exception as e:
                print(f"⚠️ Error loading clip from {start} to {end}: {e}")

        if not clips:
            print("⚠️ No valid video clips found.")
            return

        final_video = mp.concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(output_path, codec="libx264", fps=24, preset="fast")
        print("✅ Summary video created successfully in 9:16 format!")
    except Exception as e:
        print(f"❌ Error in video creation: {e}")

# Modify the main execution block
if __name__ == "__main__":
    print("🚀 Starting the summarization process...")
    audio_path = extract_audio(VIDEO_PATH)

    transcript = transcribe_audio(audio_path)
    text_chunks = chunk_text(transcript)
    summary = summarize_text(text_chunks)

    # Get original scene changes
    key_timestamps = detect_scene_changes(VIDEO_PATH)

    if not key_timestamps or len(key_timestamps) < 2:
        print("⚠️ Not enough key timestamps found. Exiting...")
        exit()

    # Get video duration
    video = mp.VideoFileClip(VIDEO_PATH)
    video_duration = video.duration
    video.close()

    # Align summary with scene timestamps
    aligned_timestamps = align_summary_with_scenes(transcript, summary, key_timestamps, video_duration)

    # Create summary video with aligned timestamps
    create_summary_video(VIDEO_PATH, aligned_timestamps, OUTPUT_VIDEO)
    print("🎉 Process completed successfully!")