#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Convert all MP3 files from lecture_mp3 to MD transcripts in lecture_transcript_raw
echo "======================================="
echo "Generating transcripts for MP3 files"
echo "======================================="

for file in ../../data/lecture_mp3/*.mp3; do
    if [ -f "$file" ]; then
        base=$(basename "$file" .mp3)
        topics_file="../../data/lecture_content/${base}.txt"
        output_md="../../data/lecture_transcript_raw/${base}.md"
        echo "Transcribing: $file"
        if [ -f "$topics_file" ]; then
            echo "Topics file found: $topics_file"
        else
            echo "Warning: Topics file $topics_file not found. Proceeding without topics."
        fi
        python lecture_transcribe.py "$file" "$topics_file" "$output_md"
        
        if [ $? -eq 0 ]; then
            echo "✓ $base transcribed successfully"
        else
            echo "✗ Failed to transcribe $base"
        fi
        echo ""
    fi
done

echo "======================================="
echo "All transcriptions completed!"
echo "======================================="