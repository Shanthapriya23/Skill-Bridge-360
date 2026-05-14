from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import yt_dlp
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import speech_recognition as sr
import os
import time
import torch
import pythoncom  # Required for COM objects in multi-threaded environments                                                                                                                     
from werkzeug.utils import secure_filename
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import io
from flask_cors import CORS
from urllib.parse import urlparse, parse_qs
import timestamp
import pdf
import re
from ppt import extract_transcript, create_phrased_summary, create_ppt
import viewer_ppt
import reel
import lecture
from times import create_video_clippings, create_subchapter_clippings, extract_video_id, process_videos, sort_chapters_by_start_time

# Fix Unicode issues for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
PPT_FOLDER = "static/ppt"
IMAGE_FOLDER = "output_images"
REEL_FOLDER = "static/reel"
LECTURE_FOLDER = "static/lecture"
PPT_IMAGES_FOLDER = "static/lecture/ppt_images"

OUTPUT_DIR = "output_clips"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PPT_FOLDER, exist_ok=True)
os.makedirs(REEL_FOLDER, exist_ok=True)
os.makedirs(LECTURE_FOLDER, exist_ok=True)
os.makedirs(PPT_IMAGES_FOLDER, exist_ok=True)

#Set folders in Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config["IMAGE_FOLDER"] = IMAGE_FOLDER
app.config["PPT_FOLDER"] = PPT_FOLDER
app.config["REEL_FOLDER"] = REEL_FOLDER
app.config["LECTURE_FOLDER"] = LECTURE_FOLDER

# ----------- RAG FUNCTIONS --------------
def initialize_rag(documents):
    """ Initialize FAISS index and language model for retrieval-augmented generation """
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embeddings = embedding_model.encode(documents)

    # Create FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    # Load GPT-2 model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad_token_id for the tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return embedding_model, index, tokenizer, model
def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    Supports standard YouTube URLs and shortened youtu.be links.
    """
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        video_id = query.get("v", [""])[0]
    elif parsed_url.hostname == "youtu.be":
        video_id = parsed_url.path.lstrip("/")
    else:
        video_id = ""
    # A basic check: YouTube video IDs are typically 11 characters long.
    if not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        video_id = ""
    return video_id
    
def convert_seconds_to_hms(seconds):
    """Converts seconds to a HH:MM:SS string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"

def rag_pipeline(query, documents, embedding_model, index, tokenizer, model):
    query_embedding = embedding_model.encode([query])
    _, I = index.search(np.array(query_embedding), 1)
    retrieved_doc = documents[I[0][0]] if I[0][0] < len(documents) else "No relevant context found."
    print(f"Retrieved Document: {retrieved_doc}")  # Debugging
    input_text = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
   
     # Tokenize the input and generate a response
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Set pad_token_id if not set already
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add attention_mask to inputs
    inputs['attention_mask'] = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))

    output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_length=300, 
    no_repeat_ngram_size=2, 
    temperature=0.7,  # Controls randomness (lower = deterministic, higher = diverse)
    top_p=0.9,  # Controls diversity (higher = more diverse outputs)
    early_stopping=True
    )
    
    # Decode the response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Decoded Output: {decoded_output}")  # Debugging

    # Extract only the "Answer" part of the decoded output
    answer_start = decoded_output.find("Answer:")
    if answer_start == -1:
        # If "Answer:" is not found, return the entire decoded output
        answer = decoded_output
    else:
        # Extract the text after "Answer:"
        answer_start += len("Answer:")
        answer = decoded_output[answer_start:].strip()

        # Remove any additional "Question:" or "Context:" parts
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        if "Context:" in answer:
            answer = answer.split("Context:")[0].strip()
    return answer

# ----------- VIDEO PROCESSING FUNCTIONS --------------
def download_youtube_video(youtube_url, output_path):
    """ Download video from YouTube """
    ydl_opts = {'format': 'best', 'outtmpl': output_path, 'nocheckcertificate': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def extract_audio(video_path, audio_path):
    """ Extract audio from a video file """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()

# ---------------------- OFFLINE TRANSCRIPTION USING POCKETSPHINX ----------------------
def transcribe_audio(audio_path):
    """Transcribe audio using offline Pocketsphinx."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # Record entire audio file
    return recognizer.recognize_sphinx(audio)  # Offline transcription

def translate_text(text, language):
    """ Translate text using Google Translator """
    return GoogleTranslator(source="auto", target=language).translate(text)

def generate_audio(text, output_path, language):
    """ Convert text to speech using gTTS """
    tts = gTTS(text=text, lang=language)
    tts.save(output_path)

def overlay_audio(video_path, audio_path, output_path):
    """ Overlay new audio onto the video """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    audio.close()

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/process_video", methods=["POST"])
def process_video():
    youtube_url = request.form.get("youtube_url")
    target_language = request.form.get("language")
    uploaded_file = request.files.get("video_file")

    video_path = None
    if youtube_url:
        video_path = os.path.join(UPLOAD_FOLDER, "downloaded_video.mp4")
        download_youtube_video(youtube_url, video_path)
    elif uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(video_path)
    else:
        return jsonify({"error": "No video provided"}), 400

    audio_path = os.path.join(UPLOAD_FOLDER, "extracted_audio.wav")
    translated_audio_path = os.path.join(PROCESSED_FOLDER, "translated_audio.mp3")
    final_video_path = os.path.join(PROCESSED_FOLDER, "dubbed_video.mp4")

    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path)  # Convert speech to text
    translated_text = translate_text(transcription, target_language)  # Translate transcript
    generate_audio(translated_text, translated_audio_path, target_language)  # Generate audio
    overlay_audio(video_path, translated_audio_path, final_video_path)  # Create dubbed video

    global rag_documents, rag_index, rag_embedding_model, rag_tokenizer, rag_model
    rag_documents = transcription.split(". ")  # Split transcript into sentences
    rag_embedding_model, rag_index, rag_tokenizer, rag_model = initialize_rag(rag_documents)

    return jsonify({
        "dubbed_video": final_video_path,
        "transcription": transcription,
        "translated_transcription": translated_text
    })

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Ensure global variables are accessible
    global rag_documents, rag_embedding_model, rag_index, rag_tokenizer, rag_model

    if not rag_documents:
        return jsonify({"error": "RAG pipeline not initialized. Please process a video first."}), 400

    # Call rag_pipeline with all required arguments
    answer = rag_pipeline(question, rag_documents, rag_embedding_model, rag_index, rag_tokenizer, rag_model)

    return jsonify({"answer": answer})

@app.route("/get_time_stamp", methods=["POST"])
def get_time_stamp():
    """Generate timestamps for a YouTube video or uploaded video, and optionally create clippings."""
    # For multipart form-data, JSON data may not be present.
    data = request.get_json(silent=True) or {}
    video_url = data.get("video_url", "").strip() if data else ""
    if not video_url:
        video_url = request.form.get("video_url", "").strip()

    video_id = extract_video_id(video_url) if video_url else None
    uploaded_file = request.files.get("video_file")
    create_clips = request.form.get("create_clips", "false").lower() == "true"
    print(f"Create Clips Flag: {create_clips}")  # Debugging: Print the flag value
    # If a YouTube URL is provided, process it.
    if video_url:
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL provided."}), 400
        try:
            # Process video using the video ID
            chapters = timestamp.process_video(video_id)
            
            # Sort chapters by start time
            chapters_sorted =timestamp.sort_chapters_by_start_time(chapters)
            
            # Prepare a textual answer with each chapter on a new line.
            answer_lines = []
            for chapter in chapters_sorted:
                # Add the chapter title and start time
                chapter_time = convert_seconds_to_hms(chapter['start_time'])  # Directly call the function
                answer_lines.append(f"{chapter_time} : {chapter['title']}")
                # Add subchapters if they exist
                for subchapter in chapter.get('subchapters', []):
                    subchapter_time = convert_seconds_to_hms(subchapter['start_time'])  # Directly call the function
                    answer_lines.append(f"  {subchapter_time} : {subchapter['title']}")
            
            # Create clippings if requested
            clippings = []

            if create_clips:
                video_path = f"https://www.youtube.com/watch?v={video_id}"
                
                # Create a sanitized copy of chapters for clipping generation only
                sanitized_chapters = []
                for chapter in chapters_sorted:
                    # Copy the chapter but sanitize the title
                    sanitized_chapter = chapter.copy()
                    sanitized_chapter['title'] = sanitize_filename(chapter['title'])
                    
                    # Handle subchapters if they exist
                    if 'subchapters' in chapter:
                        sanitized_chapter['subchapters'] = [
                            {**sub, 'title': sanitize_filename(sub['title'])} 
                            for sub in chapter['subchapters']
                        ]
                    
                    sanitized_chapters.append(sanitized_chapter)
                
                # Use the sanitized copy for creating clippings
                chapter_clippings = timestamp.create_video_clippings(video_path, sanitized_chapters, OUTPUT_DIR)
                
                # Create clippings for subchapters using sanitized copy
                subchapter_clippings = []
                for chapter in sanitized_chapters:
                    if chapter.get('subchapters'):
                        subchapter_clippings.extend(timestamp.create_subchapter_clippings(video_path, [chapter], OUTPUT_DIR))
                
                clippings = chapter_clippings + subchapter_clippings

            return jsonify({
                "answer": "\n".join(answer_lines),
                "video_type": "youtube", 
                "video_id": video_id,
                "chapters": chapters_sorted,  # Original unsanitized data
                "clippings": clippings if create_clips else None
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif uploaded_file:
        try:
            # Save the uploaded file to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Process the uploaded video file
            chapters = process_videos(file_path)
            
            # Sort chapters by start time
            chapters_sorted = sort_chapters_by_start_time(chapters)
            
            # Prepare a textual answer with each chapter on a new line.
            answer_lines = []
            for chapter in chapters_sorted:
                # Add the chapter title and start time
                chapter_time = convert_seconds_to_hms(chapter['start_time'])  # Directly call the function
                answer_lines.append(f"{chapter_time} : {chapter['title']}")
                # Add subchapters if they exist
                for subchapter in chapter.get('subchapters', []):
                    subchapter_time = convert_seconds_to_hms(subchapter['start_time'])  # Directly call the function
                    answer_lines.append(f"  {subchapter_time} : {subchapter['title']}")
            
            # Create clippings if requested
            clippings = []
            if create_clips:
                try:
                    # Create a deep copy of chapters_sorted to modify for clipping generation
                    import copy
                    clipping_chapters = copy.deepcopy(chapters_sorted)
                    
                    # Sanitize titles only in the copy used for clipping generation
                    for chapter in clipping_chapters:
                        chapter['title'] = sanitize_filename(chapter['title'])
                        for subchapter in chapter.get('subchapters', []):
                            subchapter['title'] = sanitize_filename(subchapter['title'])
                    
                    # Create clippings using the sanitized copy
                    chapter_clippings = create_video_clippings(file_path, clipping_chapters, OUTPUT_DIR)
                    
                    # Create clippings for subchapters using the sanitized copy
                    subchapter_clippings = []
                    for chapter in clipping_chapters:
                        if chapter.get('subchapters'):
                            subchapter_clippings.extend(create_subchapter_clippings(file_path, [chapter], OUTPUT_DIR))
                    
                    # Combine chapter and subchapter clippings
                    clippings = chapter_clippings + subchapter_clippings
                
                except Exception as e:
                    return jsonify({"error": f"Error creating clips: {str(e)}"}), 500

            return jsonify({
                "answer": "\n".join(answer_lines),
                "video_type": "uploaded",
                "video_url": f"/static/uploads/{uploaded_file.filename}",
                "chapters": chapters_sorted,  # Original unsanitized chapter data
                "clippings": clippings if create_clips else None
            })  
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # If neither video URL nor video file is provided, return an error.
    else:
        return jsonify({"error": "No video URL or video file provided."}), 400
    
@app.route("/clippings/output_clips/<filename>")
def get_clipping(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def extract_video_id(url):
    # Extract YouTube video ID from URL
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def convert_seconds_to_hms(seconds):
    # Convert seconds to HH:MM:SS format
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@app.route("/get_pdf", methods=["POST"])
def get_pdf():   
    if 'file' not in request.files:
        pdf.logging.warning("No file part")
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        pdf.logging.warning("No selected file")
        return redirect(request.url)
    
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        pdf.logging.error("Invalid file type")
        return "Only PDF files are allowed."

    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Process the PDF and store embeddings
    pdf.process_pdf(file_path)

    return redirect(url_for('ask_query') + f"?file_path={file_path}")

@app.route('/ask_query', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        file_path = request.form.get('file_path', '')
        
        print(f"[DEBUG] Received POST request")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] File Path: {file_path}")

        response = pdf.query_pdf(query)

        # Post-process the response to format it better
        formatted_response = format_response(response)

        print(f"[DEBUG] Generated Response: {formatted_response}")  # Debugging output in terminal

        return render_template('query_response.html', query=query, response=formatted_response)

    file_path = request.args.get('file_path', '')  # Handle GET request
    print(f"[DEBUG] Received GET request with file_path: {file_path}")

    return render_template('ask_query.html', file_path=file_path)

@app.route('/get_ppt', methods=['POST'])
def get_ppt():
    try:
        pythoncom.CoInitialize()
        video_file = request.files.get('ppt_video_file')
        youtube_link = request.form.get('ppt_youtube_url')
        
        video_path = os.path.join(PPT_FOLDER, 'input_video.mp4')
        ppt_path = os.path.join(PPT_FOLDER, 'output_slides.pptx')

        if video_file:
            video_file.save(video_path)
        elif youtube_link:
            download_youtube_video(youtube_link, video_path)
        else:
            return jsonify({"error": "No video uploaded or YouTube link provided."}), 400

        # Generate PPT
        transcript = extract_transcript(video_path)
        phrases = create_phrased_summary(transcript)
        #bullet_points = summarize_transcript(transcript)
        create_ppt(phrases, ppt_path)

        # Convert PPT to images
        slide_images = viewer_ppt.convert_ppt_to_images(ppt_path, IMAGE_FOLDER)
        slide_urls = [url_for('serve_image', filename=img) for img in slide_images]

        return jsonify({
            "ppt_url": url_for('static', filename='ppt/output_slides.pptx'),
            "slides": slide_urls
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        pythoncom.CoUninitialize()

@app.route("/get_summary_video", methods=["POST"])
def get_summary_video():
    try:
        pythoncom.CoInitialize()
        video_file = request.files.get("summary_video_file")
        youtube_url = request.form.get("summary_utube_url")
        gender = request.form.get("gender", "male").lower() 
        
        video_path = os.path.join(LECTURE_FOLDER, 'input_video.mp4')
        output_lecture_path = os.path.join(LECTURE_FOLDER, 'final_lecture.mp4')
        
        if video_file:
            video_file.save(video_path)
        elif youtube_url:
           video_path = lecture.download_youtube_video(youtube_url)
        else:
            return jsonify({"error": "No video uploaded or YouTube link provided."}), 400

        # Step 2: Extract Audio
        #audio_path = os.path.join(LECTURE_FOLDER, 'extracted_audio.wav')
        audio_path = lecture.extract_audio(video_path)
        if not audio_path:
            return jsonify({"error": "Failed to extract audio from video."}), 500

        # Step 3: Transcribe audio
        transcript = lecture.transcribe_audio(audio_path)
        if not transcript:
            return jsonify({"error": "Failed to transcribe audio."}), 500

        # Step 4:create phrased summary
        phrases = lecture.create_phrased_summary(transcript) 

        # Step 5: Create PPT from summary phrases
        ppt_path = lecture.create_ppt(phrases)
        if not ppt_path:
            return jsonify({"error": "Failed to create PPT from summary."}), 500

        # Step 6: Convert PPT slides to images
        lecture.convert_ppt_to_images(ppt_path)

        # Step 7: Generate speech from summary
        lecture.unzip_dataset()  # Ensure the voice dataset is available
        reference_audio = (
            lecture.get_reference_audio_female()
            if gender == "female"
            else lecture.get_reference_audio()
        )
        if not reference_audio:
            print("⚠️ No reference voice sample found!")
            return
        speech_path = os.path.join(LECTURE_FOLDER, "summary_speech.wav")
        lecture.generate_speech(phrases, speech_path,reference_audio)

        # Step 8: Create video from PPT images
        video_from_images = lecture.create_video_from_images(PPT_IMAGES_FOLDER, os.path.join(LECTURE_FOLDER, "summary_video.mp4"))
        if not video_from_images:
            return jsonify({"error": "Failed to create video from PPT images."}), 500

        # Step 9: Overlay speech on video
        final_video_path = lecture.overlay_audio_on_video(video_from_images, speech_path, output_lecture_path)
        if not final_video_path:
            return jsonify({"error": "Failed to overlay speech on video."}), 500
        
        # Return the final video URL
        final_video_url = f"/static/lecture/final_lecture.mp4?t={int(time.time())}"  # Cache busting
        return jsonify({"message": "Summary video generated successfully!", "video_url": final_video_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        pythoncom.CoUninitialize()

@app.route('/get_reel', methods=['POST'])
def get_reel():
    try:
        pythoncom.CoInitialize()
        video_file = request.files.get('reel_video_file')
        youtube_link = request.form.get('reel_utube_url')
        
        video_path = os.path.join(REEL_FOLDER, 'input_video.mp4')
        output_reel_path = os.path.join(REEL_FOLDER, 'final_summary_reel.mp4')

        if video_file:
            video_file.save(video_path)
        elif youtube_link:
            download_youtube_video(youtube_link, video_path)
        else:
            return jsonify({"error": "No video uploaded or YouTube link provided."}), 400

        # Get video duration
        video = VideoFileClip(video_path)
        video_duration = video.duration
        video.close()

        # Generate summarized reel
        audio_path =reel.extract_audio(video_path, os.path.join(REEL_FOLDER, "audio.wav"))
        transcript = reel.transcribe_audio(audio_path)
        text_chunks = reel.chunk_text(transcript)
        summary = reel.summarize_text(text_chunks)
        key_timestamps = reel.detect_scene_changes(video_path)
        aligned_timestamp_list = reel.align_summary_with_scenes(transcript, summary, key_timestamps, video_duration)

        if not key_timestamps or len(key_timestamps) < 2:
            return jsonify({"error": "Not enough key timestamps found to create a summary video."}), 400

        reel.create_summary_video(video_path, aligned_timestamp_list, output_reel_path)

        # Force refresh of static files by appending a timestamp
        reel_url = f"/static/reel/final_summary_reel.mp4?t={int(time.time())}"
        return jsonify({"message": "Reel generated successfully!", 
                        "reel_url": reel_url})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        pythoncom.CoUninitialize()

@app.route('/static/reel/<filename>')
def serve_reel(filename):
    return send_from_directory(REEL_FOLDER, filename, mimetype='video/mp4', as_attachment=False)

def format_response(response):
    # Remove asterisks
    response = response.replace("*", "")
    
    # Split the response into lines based on common delimiters
    lines = response.split(". ")
    
    # Join the lines with proper spacing and newlines
    formatted_response = "\n\n".join(lines)
    
    return formatted_response
def sanitize_filename(name):
    return re.sub(r'[?<>:"/\\|*]', '', name).replace(' ', '_')

@app.route("/static/processed/<filename>")
def serve_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/slides/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
