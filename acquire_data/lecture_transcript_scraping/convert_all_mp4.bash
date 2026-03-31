#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Convert all MP4 files from lecture_mp4 to MP3s in lecture_mp3
echo "======================================="
echo "Converting MP4 files to MP3"
echo "======================================="

for file in ../../data/lecture_mp4/*.mp4; do
    if [ -f "$file" ]; then
        base=$(basename "$file" .mp4)
        echo "Converting: $file"
        python mp4_to_mp3.py "$file" "../../data/lecture_mp3/${base}.mp3"
        
        if [ $? -eq 0 ]; then
            echo "✓ $base converted successfully"
        else
            echo "✗ Failed to convert $base"
        fi
        echo ""
    fi
done

echo "======================================="
echo "All conversions completed!"
echo "======================================="