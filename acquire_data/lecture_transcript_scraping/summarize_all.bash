#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Summarize all refined transcripts from lecture_transcript_refined to lecture_transcript_summary
echo "======================================="
echo "Summarizing refined transcripts"
echo "======================================="

for file in ../../data/lecture_transcript_refined/*.md; do
    if [ -f "$file" ]; then
        base=$(basename "$file" .md)
        echo "Summarizing: $file"
        python lecture_transcript_summarize.py "$file"

        if [ $? -eq 0 ]; then
            echo "✓ $base summarized successfully"
        else
            echo "✗ Failed to summarize $base"
        fi
        echo ""
    fi
done

echo "======================================="
echo "All summarizations completed!"
echo "======================================="
