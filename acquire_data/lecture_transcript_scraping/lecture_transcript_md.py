import os
import re

import torch
# from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- CONFIGURATION ---
# VIDEO_FILE = "lecture.mp4"
# AUDIO_FILE = "temp_audio.mp3"
AUDIO_FILE = "C:\\Users\\k97ev\\Documents\\git_repos\\dat255-teaching-assistant\\data\\lecture_mp3\\lecture3-computer_vision_and_the_concepts_of_layers.mp3"
OUTPUT_MD = "lecture_notes.md"
# MODEL_SIZE = "medium" # 2GB-3GB VRAM usage

# Paste your raw topic list here (supports both formats)
RAW_TOPICS = """
Deep learning engineering 0:03
Shallow learning 10:12
Deep learning
11:39
The feed-forward neural network
15:57
Image classification
17:21
Enter the convolution operation 22:15
an operation that takes in two functions and ret 22:36
Discrete convolution
24:57
Convolution over images
27:15
The convolution kernel (or filter)
30:12
More filters
32:03
Kernels for image recognition
33:21
Decomposition into simple patters
36:00
Keras layers
41:36
A superpower for ML developers
41:51
Keras 3 API documentation
42:18
Keras
42:42
The Conv2D layer
45:27
"""

def timestamp_to_seconds(ts):
    """Converts M:SS or H:MM:SS to total seconds."""
    parts = list(map(int, ts.split(':')))
    if len(parts) == 2: # M:SS
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] # H:MM:SS

def parse_topics(raw_text):
    """Parses mixed-format topic lists into sorted (seconds, title) tuples."""
    lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]
    parsed = []
    
    # Regex to find timestamps like 0:03, 10:12, 1:05:30
    ts_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(ts_pattern, line)
        
        if match:
            # Format: "Topic Name 0:03"
            ts = match.group(1)
            title = line.replace(ts, "").strip()
            parsed.append((timestamp_to_seconds(ts), title))
        elif i + 1 < len(lines):
            # Format: "Topic Name" \n "0:03"
            next_line = lines[i+1]
            next_match = re.search(ts_pattern, next_line)
            if next_match:
                ts = next_match.group(1)
                parsed.append((timestamp_to_seconds(ts), line))
                i += 1 # Skip the next line as we've used it
        i += 1
    
    return sorted(parsed, key=lambda x: x[0])

def run_transcription():
    # 1. Extract Audio
    # if not os.path.exists(AUDIO_FILE):
    #     print("--- Extracting Audio ---")
    #     video = VideoFileClip(VIDEO_FILE)
    #     video.audio.write_audiofile(AUDIO_FILE, logger=None)
    #     video.close()

    # 2. Parse Topics
    # Uses the provided RAW_TOPICS string to create a list of (timestamp_seconds, topic_title) tuples. 
    # This will be used to insert section headers in the Markdown output at the appropriate timestamps.
    # It also gives the model a helpful prompt of expected topic titles to improve transcription accuracy for technical terms.
    topics = parse_topics(RAW_TOPICS)
    topic_titles = [t[1] for t in topics]
    
    # 3. Initialize Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use smallest model if device is CPU to avoid out-of-memory errors
    model_size = "medium" if device == "cuda" else "tiny"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"--- Loading Whisper {model_size} on {device.upper()} with {compute_type} ---")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Transcribe with the topics as a prompt for better spelling
    segments, _ = model.transcribe(
        AUDIO_FILE, 
        language="en", 
        initial_prompt=", ".join(topic_titles)
    )

    # 4. Generate Markdown
    print("--- Transcribing and Generating Markdown ---")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(f"# Lecture Notes: {AUDIO_FILE}\n\n")
        
        current_topic_idx = 0
        
        for segment in tqdm(segments, desc="Transcribing"):
            # Check if we have moved into a new topic section
            while (current_topic_idx < len(topics) and 
                   segment.start >= topics[current_topic_idx][0]):
                
                start_time, title = topics[current_topic_idx]
                f.write(f"\n## {title} ({int(start_time//60)}:{int(start_time%60):02d})\n\n")
                current_topic_idx += 1
            
            # Write the segment text
            f.write(f"{segment.text.strip()} ")
            f.flush() # Live update the file

    print(f"\n--- Done! Saved to {OUTPUT_MD} ---")

if __name__ == "__main__":
    run_transcription()