import sys
import os
import subprocess
from pathlib import Path

# Make script runnable from any working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_OUTPUT_DIR = Path("../../data/lecture_mp3")


def extract_audio_with_ffmpeg(input_mp4: Path, output_mp3: Path):
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_mp4),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_mp3)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (return code {result.returncode})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print(f"Done! Audio saved to {output_mp3} (via ffmpeg)")


def extract_audio_with_moviepy(input_mp4: Path, output_mp3: Path):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    video = VideoFileClip(str(input_mp4))
    video.audio.write_audiofile(str(output_mp3), bitrate="192k")
    video.close()
    print(f"Done! Audio saved to {output_mp3} (via moviepy)")


def extract_audio(input_mp4: Path, output_mp3: Path):
    try:
        extract_audio_with_ffmpeg(input_mp4, output_mp3)
    except Exception as e:
        print(f"ffmpeg extraction failed, falling back to MoviePy: {e}")
        extract_audio_with_moviepy(input_mp4, output_mp3)

def main():
    if len(sys.argv) == 1:
        print("Usage: python mp4_to_mp3.py <input.mp4> [output.mp3]")
        sys.exit(1)

    input_mp4 = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_mp3 = Path(sys.argv[2])
    else:
        output_mp3 = DEFAULT_OUTPUT_DIR / (input_mp4.stem + ".mp3")

    print(f"Converting: {input_mp4} to {output_mp3}")
    extract_audio(input_mp4, output_mp3)

if __name__ == "__main__":
    main()