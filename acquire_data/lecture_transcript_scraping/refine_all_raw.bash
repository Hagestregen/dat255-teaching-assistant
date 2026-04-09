#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Refine all raw transcripts from lecture_transcript_raw to lecture_transcript
echo "======================================="
echo "Refining raw transcripts"
echo "======================================="

for file in ../../data/lecture_transcript_raw/*.md; do
    if [ -f "$file" ]; then
        base=$(basename "$file" .md)
        echo "Refining: $file"
        python lecture_transcript_refine.py "$file"
        
        if [ $? -eq 0 ]; then
            echo "✓ $base refined successfully"
        else
            echo "✗ Failed to refine $base"
        fi
        echo ""
    fi
done

echo "======================================="
echo "All refinements completed!"
echo "======================================="