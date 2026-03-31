import sys
import os
from moviepy import VideoFileClip

def extract_audio(input_mp4, output_mp3):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_mp3), exist_ok=True)
    
    # Load the video file
    video = VideoFileClip(input_mp4)
    
    # Extract and save the audio
    video.audio.write_audiofile(output_mp3, bitrate="192k")
    
    # Clean up
    video.close()
    print(f"Done! Audio saved to {output_mp3}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python mp4_to_mp3.py <input.mp4> <output.mp3>")
        sys.exit(1)

    input_mp4 = sys.argv[1]
    output_mp3 = sys.argv[2]

    print(f"Converting: {input_mp4} to {output_mp3}")
    extract_audio(input_mp4, output_mp3)

if __name__ == "__main__":
    main()