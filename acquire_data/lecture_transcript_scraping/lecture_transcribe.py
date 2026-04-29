import os
import re
import sys
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- CONFIGURATION ---
DEFAULT_OUTPUT_DIR = Path("../../data/lecture_transcript_raw")

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

def parse_topics_file(file_path):
    """Parses the topics file to extract title and topics list."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    title = lines[0].replace('Title:', '').strip() if lines and lines[0].startswith('Title:') else "Lecture Notes"
    raw_topics = '\n'.join(lines[1:])
    topics = parse_topics(raw_topics)
    topics.insert(0, (0, title))  # Add title as first topic at 0:00
    return title, topics

def format_duration(seconds):
    """Formats seconds as H:MM:SS or M:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def run_transcription(audio_file, output_md, topics, title):
    # 2. Parse Topics
    # Uses the provided topics to create a list of (timestamp_seconds, topic_title) tuples. 
    # This will be used to insert section headers in the Markdown output at the appropriate timestamps.
    # It also gives the model a helpful prompt of expected topic titles to improve transcription accuracy for technical terms.
    topic_titles = [t[1] for t in topics]
    
    # 3. Initialize Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use smallest model if device is CPU to avoid out-of-memory errors
    model_size = "medium" if device == "cuda" else "tiny"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"--- Loading Whisper {model_size} on {device.upper()} with {compute_type} ---")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Transcribe with the topics as a prompt for better spelling.
    # NOTE: Do NOT call list() on segments — it's a generator that transcribes lazily.
    # Materialising it upfront causes all processing to finish before the loop runs,
    # making the progress bar appear to jump instantly from 0% to 100%.
    segments, info = model.transcribe(
        audio_file,
        language="en",
        initial_prompt=", ".join(topic_titles),
        vad_filter=True,
    )
    total_duration = info.duration  # Available immediately without consuming the generator
    print(f"--- Audio duration: {format_duration(total_duration)} ---")

    # 4. Generate Markdown
    print("--- Transcribing and Generating Markdown ---")

    def flush_buffer(f, buf):
        """Write buffered text (trimmed) followed by a newline. Returns empty buffer."""
        text = buf.strip()
        if text:
            f.write(text + "\n")
        return ""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")

        # topics[0] is the title itself (inserted by parse_topics_file) — skip it
        # since it's already written above as the # heading.
        current_topic_idx = 1 if topics and topics[0][1] == title else 0
        last_end = 0.0
        buffer = ""  # Accumulates segment text between section headers

        # Progress bar tracks audio seconds processed, so it reflects real transcription speed
        with tqdm(total=round(total_duration), desc="Transcribing", unit="sec", ncols=100) as pbar:
            for segment in segments:
                # Advance progress bar by the duration of this segment
                elapsed = segment.end - last_end
                pbar.update(round(elapsed))
                last_end = segment.end

                # Check if we have moved into a new topic section
                while (current_topic_idx < len(topics) and
                       segment.start >= topics[current_topic_idx][0]):

                    # Flush any buffered text before the new header.
                    # If buffer is empty (two headers in a row), nothing is written,
                    # which avoids a double blank line between adjacent headers.
                    buffer = flush_buffer(f, buffer)

                    start_time, topic_name = topics[current_topic_idx]
                    f.write(f"\n## {topic_name} ({int(start_time//60)}:{int(start_time%60):02d})\n\n")
                    current_topic_idx += 1

                buffer += segment.text.strip() + " "
                f.flush()  # Live update the file

        flush_buffer(f, buffer)  # Write remaining text after the last header

    print(f"\n--- Done! Saved to {output_md} ---")

def main():
    if len(sys.argv) < 3:
        print("Usage: python lecture_transcript_md.py <input.mp3> <topics.txt> [output.md]")
        sys.exit(1)

    mp3_file = Path(sys.argv[1])
    topics_file = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        output_md = Path(sys.argv[3])
    else:
        output_md = DEFAULT_OUTPUT_DIR / (mp3_file.stem + ".md")

    output_md.parent.mkdir(parents=True, exist_ok=True)

    try:
        title, topics = parse_topics_file(topics_file)
    except FileNotFoundError:
        print(f"Warning: Topics file {topics_file} not found. Proceeding without topics.")
        title = mp3_file.stem.replace('_', ' ').title()
        topics = [(0, title)]

    run_transcription(str(mp3_file), str(output_md), topics, title)

if __name__ == "__main__":
    main()