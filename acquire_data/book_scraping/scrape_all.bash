#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Scrape all remaining chapters of "Deep Learning with Python" 3rd edition
# Reads commands from scrape_commands.txt and executes them one by one

echo "========================================"
echo "Starting to scrape Deep Learning with Python 3rd Edition"
echo "Chapters 3 to 20"
echo "========================================"

# Check if the commands file exists
if [ ! -f "scrape_commands.txt" ]; then
    echo "Error: scrape_commands.txt not found!"
    exit 1
fi

# Count total commands
total=$(wc -l < scrape_commands.txt)
current=1

# Read and execute each line
while IFS= read -r command || [ -n "$command" ]; do
    # Skip empty lines and comments
    if [[ -z "$command" || "$command" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    echo "[$current/$total] Running: $command"
    echo "----------------------------------------"
    
    # Execute the command
    eval "$command"
    
    # Check if command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Chapter completed successfully"
    else
        echo "✗ Failed to scrape this chapter"
    fi
    
    echo ""
    ((current++))
    
    # Optional: small delay to be gentle on the server
    sleep 1
done < scrape_commands.txt

echo "========================================"
echo "All scraping tasks completed!"
echo "========================================"