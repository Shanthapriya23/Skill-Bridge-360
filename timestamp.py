import os
import json
import re
import concurrent.futures
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yt_dlp
from faster_whisper import WhisperModel
import torch
import subprocess

# Constants
GROQ_API_KEY = ""
OPENAI_API_KEY = ""

price_token = {
    'gpt-4o': {'input': 5/1000000, 'output': 15/1000000},
    'gpt-4o-2024-08-06': {'input': 2.5/1000000, 'output': 10/1000000},
    'gpt-4o-mini-2024-07-18': {'input': 0.15/1000000, 'output': 0.6/1000000},
    'llama3-8b-8192': {'input': 0.05 / 1000000, 'output': 0.08 / 1000000},
    'llama3-70b-8192': {'input': 0.59 / 1000000, 'output': 0.79 / 1000000},
    'claude-3-5-sonnet-20240620': {'input': 3/1000000, 'output': 15/1000000},
    'claude-3-haiku-20240307': {'input': 0.25/1000000, 'output': 1.25/1000000},
}

system_prompt_transcript_to_paragraphs = """
You are a helpful assistant. Your task is to improve the user input's readability: add punctuation if needed and remove verbal tics, and structure the text in paragraphs separated with '\n\n'. Keep the wording as faithful as possible to the original text. Put your answer within <answer></answer> tags.
"""

system_prompt_paragraphs_to_toc = """
You are a helpful assistant. You are given a transcript of a course in JSON format as a list of paragraphs, each containing 'paragraph_number' and 'paragraph_text' keys. Your task is to group consecutive paragraphs in chapters for the course and identify meaningful chapter titles. Format your result in JSON, with a list of dictionaries for chapters, with 'start_paragraph_number':integer and 'title':string as key:value.
"""

system_prompt_paragraphs_to_subtoc = """
You are a helpful assistant. You are given a chapter from a course transcript in JSON format as a list of paragraphs, each containing 'paragraph_number' and 'paragraph_text' keys. Your task is to group consecutive paragraphs into sub-chapters and identify meaningful sub-chapter titles. Ensure the titles are concise, unique, and directly related to the content of the sub-chapter. Format your result in JSON, with a list of dictionaries for sub-chapters under the key 'subchapters', with 'start_paragraph_number':integer and 'title':string as key:value.
"""

# Helper Functions
def create_directory(video_base):
    """Create a directory for storing intermediate files."""
    data_dir = f"examples/{video_base}"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_transcript(video_id, languages=["en"]):
    """Fetch transcript from YouTube using YouTubeTranscriptApi."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    print(f"[DEBUG] Retrieved transcript with {len(transcript)} segments")
    return [{'start': s['start'], 'text': s['text']} for s in transcript]

def download_audio(video_id, video_url, DOWNLOAD_DIR="temp_download"):
    """Download audio from YouTube using yt-dlp."""
    os.makedirs(f"{DOWNLOAD_DIR}/{video_id}", exist_ok=True)
    audio_path = f"{DOWNLOAD_DIR}/{video_id}/{video_id}_audio.mp4"
    ydl_opts = {'format': 'bestaudio', 'outtmpl': audio_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[DEBUG] Downloading audio from: {video_url}")
        ydl.download([video_url])
    print(f"[DEBUG] Audio downloaded to: {audio_path}")
    return audio_path

def speech_to_text(whisper_model, audio_file, initial_prompt="Use punctuation, like this.", language="en"):
    """Convert audio to text using WhisperModel."""
    print(f"[DEBUG] Running speech-to-text on {audio_file} using WhisperModel...")
    segments, _ = whisper_model.transcribe(audio_file, initial_prompt=initial_prompt, language=language)
    segments = list(segments)  # Convert generator to list
    print(f"[DEBUG] Speech-to-text produced {len(segments)} segments")
    return [{"start": round(s.start, 2), "duration": round(s.end - s.start, 2), "text": s.text} for s in segments]

def get_transcript_as_text(transcript):
    """Convert transcript segments into a single text string."""
    return ' '.join(s['text'] for s in transcript)

def call_llm(client, model, system_prompt, prompt, temperature=0, seed=42, response_format=None, max_tokens=4000):
    """Call the LLM API with the given prompt."""
    print(f"[DEBUG] Calling LLM model {model} with prompt length {len(prompt)}")
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        seed=seed,
        response_format=response_format,
        max_tokens=max_tokens
    )
    nb_input_tokens = response.usage.prompt_tokens
    nb_output_tokens = response.usage.completion_tokens
    price = nb_input_tokens * price_token[model]['input'] + nb_output_tokens * price_token[model]['output']
    print(f"[DEBUG] LLM call returned input tokens: {nb_input_tokens}; output tokens: {nb_output_tokens}, price: {price}")
    return response.choices[0].message.content, nb_input_tokens, nb_output_tokens, price

def process_chunk(chunk, llm_client, llm_model):
    """Process a chunk of text using the LLM."""
    response_content, nb_input_tokens, nb_output_tokens, price = call_llm(
        llm_client, llm_model, system_prompt_transcript_to_paragraphs, chunk, temperature=0.2, seed=42
    )
    if "</answer>" not in response_content:
        response_content += "</answer>"
    pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    response_content_edited = pattern.findall(response_content)
    if response_content_edited:
        print(f"[DEBUG] Processed chunk returned {len(response_content_edited[0].splitlines())} paragraphs")
        return response_content_edited[0], nb_input_tokens, nb_output_tokens, price
    return None, nb_input_tokens, nb_output_tokens, price

def transcript_to_paragraphs(transcript, llm_client, llm_model, chunk_size=5000):
    """Convert transcript into structured paragraphs using LLM."""
    transcript_as_text = ' '.join(s['text'] for s in transcript)
    paragraphs = []
    last_paragraph = ""
    total_nb_input_tokens, total_nb_output_tokens, total_price = 0, 0, 0
    chunks = [transcript_as_text[i:i + chunk_size] for i in range(0, len(transcript_as_text), chunk_size)]
    print(f"[DEBUG] Split transcript text into {len(chunks)} chunks")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, last_paragraph + " " + chunk, llm_client, llm_model) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            response_content_edited, nb_input_tokens, nb_output_tokens, price = future.result()
            if response_content_edited:
                paragraphs_chunk = response_content_edited.strip().split('\n\n')
                paragraphs += paragraphs_chunk[:-1]
                last_paragraph = paragraphs_chunk[-1]
                total_nb_input_tokens += nb_input_tokens
                total_nb_output_tokens += nb_output_tokens
                total_price += price

    paragraphs += [last_paragraph]
    print(f"[DEBUG] Generated {len(paragraphs)} paragraphs from transcript")
    return [{'paragraph_number': i, 'paragraph_text': paragraph} for i, paragraph in enumerate(paragraphs)], total_nb_input_tokens, total_nb_output_tokens, total_price

def transform_text_segments(text_segments, num_words=50):
    """Transform text segments into chunks of a specified number of words."""
    transformed_segments = []
    for i in range(len(text_segments)):
        combined_text = " ".join(text_segments[i]['text'].split()[:num_words])
        number_words_collected = len(text_segments[i]['text'].split())
        current_index = i
        while number_words_collected < num_words and (current_index + 1) < len(text_segments):
            current_index += 1
            next_text = text_segments[current_index]['text']
            next_words = next_text.split()
            if number_words_collected + len(next_words) <= num_words:
                combined_text += ' ' + next_text
                number_words_collected += len(next_words)
            else:
                combined_text += ' ' + ' '.join(next_words[:num_words - number_words_collected])
                number_words_collected = num_words
        transformed_segments.append(combined_text)
    return transformed_segments

def add_timestamps_to_paragraphs(transcript, paragraphs, num_words=50):
    """Add timestamps to paragraphs based on transcript."""
    transcript_num_words = transform_text_segments(transcript, num_words=num_words)
    paragraphs_num_words = transform_text_segments([{"start": p['paragraph_number'], "text": p['paragraph_text']} for p in paragraphs], num_words=num_words)
    vectorizer = TfidfVectorizer().fit_transform(transcript_num_words + paragraphs_num_words)
    vectors = vectorizer.toarray()
    for i, paragraph in enumerate(paragraphs):
        paragraph_vector = vectors[len(transcript_num_words) + i]
        similarities = cosine_similarity(vectors[:len(transcript_num_words)], paragraph_vector.reshape(1, -1))
        best_match_index = int(np.argmax(similarities))
        paragraph['matched_index'] = best_match_index
        paragraph['matched_text'] = transcript[best_match_index]['text']
        paragraph['start_time'] = max(0, int(transcript[best_match_index]['start']) - 2)
    print(f"[DEBUG] Added timestamps to {len(paragraphs)} paragraphs")
    return paragraphs

def post_process_subchapters(subchapters):
    """Post-process sub-chapter titles to remove duplicates and ensure consistency."""
    seen_titles = set()
    for subchapter in subchapters:
        title = subchapter['title']
        if title in seen_titles:
            count = 1
            new_title = f"{title} (Part {count})"
            while new_title in seen_titles:
                count += 1
                new_title = f"{title} (Part {count})"
            subchapter['title'] = new_title
        seen_titles.add(subchapter['title'])
    return subchapters

def paragraphs_to_toc(paragraphs, llm_client, llm_model, chunk_size=100):
    """Generate table of contents from paragraphs."""
    chapters = []
    number_last_chapter = 0
    total_nb_input_tokens, total_nb_output_tokens, total_price = 0, 0, 0
    while number_last_chapter < len(paragraphs):
        chunk = paragraphs[number_last_chapter:(number_last_chapter + chunk_size)]
        chunk_json_dump = json.dumps([{'paragraph_number': p['paragraph_number'], 'paragraph_text': p['paragraph_text']} for p in chunk])
        content, nb_input_tokens, nb_output_tokens, price = call_llm(
            llm_client, llm_model, system_prompt_paragraphs_to_toc, chunk_json_dump, temperature=0, seed=42, response_format={"type": "json_object"}
        )
        total_nb_input_tokens += nb_input_tokens
        total_nb_output_tokens += nb_output_tokens
        total_price += price
        chapters_chunk = json.loads(content)['chapters']
        print(f"[DEBUG] LLM returned {len(chapters_chunk)} chapters for current chunk")
        if number_last_chapter == chapters_chunk[-1]['start_paragraph_number']:
            break
        chapters += chapters_chunk[:-1]
        number_last_chapter = chapters_chunk[-1]['start_paragraph_number']
        if number_last_chapter >= len(paragraphs) - 5:
            break
    chapters += [chapters_chunk[-1]]
    print(f"[DEBUG] Total chapters generated: {len(chapters)}")
    return chapters, total_nb_input_tokens, total_nb_output_tokens, total_price

def paragraphs_to_subtoc(paragraphs, llm_client, llm_model, chunk_size=100):
    """Generate sub-chapters from paragraphs."""
    subchapters = []
    number_last_subchapter = 0
    total_nb_input_tokens, total_nb_output_tokens, total_price = 0, 0, 0
    while number_last_subchapter < len(paragraphs):
        chunk = paragraphs[number_last_subchapter:(number_last_subchapter + chunk_size)]
        chunk_json_dump = json.dumps([{'paragraph_number': p['paragraph_number'], 'paragraph_text': p['paragraph_text']} for p in chunk])
        content, nb_input_tokens, nb_output_tokens, price = call_llm(
            llm_client, llm_model, system_prompt_paragraphs_to_subtoc, chunk_json_dump, temperature=0, seed=42, response_format={"type": "json_object"}
        )
        total_nb_input_tokens += nb_input_tokens
        total_nb_output_tokens += nb_output_tokens
        total_price += price

        try:
            response_json = json.loads(content)
            if 'subchapters' not in response_json:
                print(f"Warning: 'subchapters' key not found in LLM response. Response: {response_json}")
                subchapters_chunk = []
            else:
                subchapters_chunk = response_json['subchapters']
                subchapters_chunk = post_process_subchapters(subchapters_chunk)
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response: {e}. Response: {content}")
            subchapters_chunk = []

        if not subchapters_chunk:
            break

        if number_last_subchapter == subchapters_chunk[-1]['start_paragraph_number']:
            break
        subchapters += subchapters_chunk[:-1]
        number_last_subchapter = subchapters_chunk[-1]['start_paragraph_number']
        if number_last_subchapter >= len(paragraphs) - 5:
            break
    subchapters += [subchapters_chunk[-1]]
    return subchapters, total_nb_input_tokens, total_nb_output_tokens, total_price

def get_chapters_with_subchapters(paragraphs, table_of_content, llm_client, llm_model):
    """Generate chapters with subchapters."""
    chapters = []
    for i in range(len(table_of_content)):
        if i < len(table_of_content) - 1:
            chapter_paragraphs = paragraphs[table_of_content[i]['start_paragraph_number']:table_of_content[i + 1]['start_paragraph_number']]
        else:
            chapter_paragraphs = paragraphs[table_of_content[i]['start_paragraph_number']:]

        subchapters, _, _, _ = paragraphs_to_subtoc(chapter_paragraphs, llm_client, llm_model)

        chapter_start_time = paragraphs[table_of_content[i]['start_paragraph_number']]['start_time']
        chapter_end_time = paragraphs[table_of_content[i + 1]['start_paragraph_number']]['start_time'] if i < len(table_of_content) - 1 else paragraphs[-1]['start_time']

        filtered_subchapters = []
        for subchapter in subchapters:
            subchapter_start_time = paragraphs[subchapter['start_paragraph_number']]['start_time']
            if chapter_start_time <= subchapter_start_time < chapter_end_time:
                if subchapter_start_time == chapter_start_time:
                    subchapter_start_time += 1  # Add a 1-second offset
                subchapter['start_time'] = subchapter_start_time
                filtered_subchapters.append(subchapter)

        chapter = {
            'num_chapter': i,
            'title': table_of_content[i]['title'],
            'start_paragraph_number': table_of_content[i]['start_paragraph_number'],
            'end_paragraph_number': table_of_content[i + 1]['start_paragraph_number'] if i < len(table_of_content) - 1 else len(paragraphs),
            'start_time': chapter_start_time,
            'end_time': chapter_end_time,
            'subchapters': filtered_subchapters
        }
        chapter['paragraphs'] = [paragraphs[j]['paragraph_text'] for j in range(chapter['start_paragraph_number'], chapter['end_paragraph_number'])]
        chapter['paragraph_timestamps'] = [paragraphs[j]['start_time'] for j in range(chapter['start_paragraph_number'], chapter['end_paragraph_number'])]
        chapters.append(chapter)
    return chapters

def convert_seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"

def sort_chapters_by_start_time(chapters):
    """Sort chapters by their start_time in ascending order."""
    return sorted(chapters, key=lambda x: x['start_time'])

def process_video(video_id):
    """Process a YouTube video to generate chapters and subchapters."""
    DATA_DIR = create_directory(video_id)
    
    # Get transcript
    transcript_data = get_transcript(video_id)
    print("[DEBUG] Transcript data retrieved.")
    
    # Download audio
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    path_to_audio = download_audio(video_id, video_url, DATA_DIR)
    
    # Speech-to-text
    whisper_model = WhisperModel(
        "tiny",  # Use "base", "small", or "tiny" for a smaller model
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float32"
    )
    transcript = speech_to_text(whisper_model, path_to_audio)
    with open(f"{DATA_DIR}/{video_id}_transcript.json", "w") as f:
        json.dump(transcript, f, indent=4)
    print(f"[DEBUG] Transcript saved to {DATA_DIR}/{video_id}_transcript.json")
    
    # Process transcript to paragraphs
    transcript_as_text = get_transcript_as_text(transcript)
    llm_client_format_transcript = Groq(api_key=GROQ_API_KEY)
    llm_model_format_transcript = 'llama3-8b-8192'
    chunk_size_format_transcript = 1500
    
    paragraphs, nb_input_tokens, nb_output_tokens, price = transcript_to_paragraphs(
        transcript, llm_client_format_transcript, llm_model_format_transcript, chunk_size=chunk_size_format_transcript
    )
    with open(f"{DATA_DIR}/{video_id}_paragraphs.json", "w") as f:
        json.dump(paragraphs, f, indent=4)
    print(f"[DEBUG] Paragraphs saved to {DATA_DIR}/{video_id}_paragraphs.json")
    
    # Add timestamps
    paragraphs = add_timestamps_to_paragraphs(transcript, paragraphs, num_words=50)
    with open(f"{DATA_DIR}/{video_id}_paragraphs.json", "w") as f:
        json.dump(paragraphs, f, indent=4)
    print(f"[DEBUG] Updated paragraphs with timestamps saved to {DATA_DIR}/{video_id}_paragraphs.json")
    
    # Generate table of contents
    llm_client_get_toc = Groq(api_key=GROQ_API_KEY)
    llm_model_get_toc = 'llama3-8b-8192'
    chunk_size_toc = 30
    
    table_of_content, total_nb_input_tokens, total_nb_output_tokens, total_price = paragraphs_to_toc(
        paragraphs, llm_client_get_toc, llm_model_get_toc, chunk_size=chunk_size_toc
    )
    with open(f"{DATA_DIR}/{video_id}_toc.json", "w") as f:
        json.dump(table_of_content, f, indent=4)
    print(f"[DEBUG] Table of Contents saved to {DATA_DIR}/{video_id}_toc.json")
    
    # Generate chapters with subchapters
    chapters = get_chapters_with_subchapters(paragraphs, table_of_content, llm_client_get_toc, llm_model_get_toc)
    chapters = sort_chapters_by_start_time(chapters)
    print(f"[DEBUG] Generated chapters with subchapters: {chapters}")

    # Save the chapters to a JSON file
    chapters_file = os.path.join(DATA_DIR, f"{video_id}_chapters_with_subchapters.json")
    with open(chapters_file, "w") as f:
        json.dump(chapters, f, indent=4)
    print(f"[DEBUG] Saved chapters to: {chapters_file}")    
    
    # Print chapters and subchapters
    for chapter in chapters:
        print(f"{convert_seconds_to_hms(chapter['start_time'])} : {chapter['title']}")
        for subchapter in chapter['subchapters']:
            print(f"  {convert_seconds_to_hms(subchapter['start_time'])} : {subchapter['title']}")
    return chapters
    
def download_video(video_url, output_file):
    """
    Downloads a YouTube video using yt-dlp (Python library).
    Returns the output file path if successful, None on failure.
    """
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_file,  # Saves to the specified output path
        'nocheckcertificate': True,
        'quiet': True,           # Suppresses yt-dlp output (optional)
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Video downloaded: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
        
def create_video_clippings(video_url, chapters, output_dir):
    print(f"Starting create_video_clippings with video_url: {video_url}, output_dir: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    clippings = []

    # Step 1: Download the video
    video_file = os.path.join(output_dir, "downloaded_video.mp4")
    downloaded_file = download_video(video_url, video_file)
    if not downloaded_file:
        print("Failed to download the video. Exiting...")
        return clippings

    # Step 2: Extract clips from the downloaded video
    for i, chapter in enumerate(chapters):
        start_time = chapter['start_time']
        end_time = chapters[i + 1]['start_time'] if i < len(chapters) - 1 else None

        print(f"Processing chapter: {chapter['title']}, Start: {start_time}, End: {end_time}")

        if start_time < 0 or (end_time is not None and end_time < 0) or (end_time is not None and start_time >= end_time):
            print(f"Invalid timestamps for chapter '{chapter['title']}'. Skipping...")
            continue

        output_file = os.path.join(output_dir, f"{chapter['title'].replace(' ', '_')}.mp4").replace("\\", "/")
        ffmpeg_command = [
            'ffmpeg',
            '-ss', str(start_time),  # Start time
            '-i', downloaded_file,  # Input video file
            '-c', 'copy',           # Copy codec (no re-encoding)
        ]

        if end_time is not None:
            ffmpeg_command.extend(['-to', str(end_time)])  # End time

        ffmpeg_command.append(output_file)  # Output file

        try:
            subprocess.run(ffmpeg_command, check=True)
            clippings.append(output_file)
            print(f"Created clipping: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating subclip for chapter '{chapter['title']}': {e}. Skipping...")

    print(f"Finished create_video_clippings. Generated {len(clippings)} clippings.")
    return clippings

def create_subchapter_clippings(video_url, chapters, output_dir):
    print(f"Starting create_subchapter_clippings with video_url: {video_url}, output_dir: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    clippings = []

    # Step 1: Download the video
    video_file = os.path.join(output_dir, "downloaded_video.mp4")
    downloaded_file = download_video(video_url, video_file)
    if not downloaded_file:
        print("Failed to download the video. Exiting...")
        return clippings

    # Step 2: Extract subchapter clips from the downloaded video
    for chapter in chapters:
        chapter_title = chapter['title']
        chapter_start_time = chapter['start_time']
        chapter_end_time = chapter['end_time']
        print(f"Processing chapter: {chapter_title}, Start: {chapter_start_time}, End: {chapter_end_time}")

        for i in range(len(chapter['subchapters'])):
            subchapter = chapter['subchapters'][i]
            subchapter_title = subchapter['title']
            subchapter_start_time = subchapter['start_time']
            subchapter_end_time = chapter['subchapters'][i + 1]['start_time'] if i < len(chapter['subchapters']) - 1 else chapter_end_time

            print(f"  Processing subchapter: {subchapter_title}, Start: {subchapter_start_time}, End: {subchapter_end_time}")

            if subchapter_start_time < 0 or subchapter_end_time < 0 or subchapter_start_time >= subchapter_end_time:
                print(f"  Invalid timestamps for subchapter '{subchapter_title}'. Skipping...")
                continue

            output_file = os.path.join(output_dir, f"{chapter_title.replace(' ', '_')}_{subchapter_title.replace(' ', '_')}.mp4").replace("\\", "/")
            ffmpeg_command = [
                'ffmpeg',
                '-ss', str(subchapter_start_time),  # Start time
                '-to', str(subchapter_end_time),   # End time
                '-i', downloaded_file,           # Input video file
                '-c', 'copy',                     # Copy codec (no re-encoding)
                output_file                       # Output file
            ]

            try:
                subprocess.run(ffmpeg_command, check=True)
                clippings.append(output_file)
                print(f"  Created subchapter clipping: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"  Error creating subclip for subchapter '{subchapter_title}': {e}")

    print(f"Finished create_subchapter_clippings. Generated {len(clippings)} clippings.")
    return clippings

def extract_video_id(url):
    """
    Extract the YouTube video ID from a given URL.
    Supports various YouTube URL formats.
    """
    # Regular expression to match YouTube video IDs
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    
    # Search for the video ID in the URL
    match = re.search(regex, url)
    
    # Return the video ID if found, otherwise return None
    return match.group(1) if match else None

def main():
    video_id = 'HhBhWdoXSdg'  # Replace with your YouTube video ID
    process_video(video_id)

if __name__ == "__main__":
    main()