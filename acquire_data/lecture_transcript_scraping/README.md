# Scraping and Transcribing Video Lectures

The auto-generated transcripts from Panopto were often in Norwegian even though most lectures were in English, and generally too inaccurate to use. This pipeline downloads the lecture video, converts it to audio, transcribes it with Whisper, and optionally refines the raw transcript with a small language model.

## Pipeline Overview

```
Panopto (browser)
      │  mp4
      ▼
mp4_to_mp3.py
      │  mp3
      ▼
lecture_transcript_md.py
      │  data/lecture_transcript_raw/*.md
      ▼
lecture_transcript_refine.py
      │  data/lecture_transcript/*.md
      ▼
 Clean notes ✓
```

---

## Step 0 — Download Lecture Video from Panopto

There is no simple download button on Panopto. Use the browser's network inspector instead:

1. Open a lecture on Panopto and start playing it.
2. Right-click anywhere on the page and choose **Inspect**.
3. Go to the **Network** tab and type `mp4` in the filter field at the top.
4. Refresh the page. A URL ending in `.mp4` should appear in the list.
5. Double-click the URL to open it in a new tab, then right-click the video and choose **Save as**.

> **Note:** On some pages the audio stream is served as `.m4a` (MPEG-4 Audio, no video). Searching for `m4a` instead of `mp4` in the network tab will find it, but this did not work on Panopto.

---

## Step 1 — Convert MP4 to MP3

### Dependency: FFmpeg

FFmpeg must be installed on your system for the conversion step.

**Windows** — open PowerShell as Administrator and run:

```powershell
winget install -e --id Gyan.FFmpeg
```

Alternatively use Chocolatey (`choco install ffmpeg`) or download the release build manually from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your PATH.

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install ffmpeg -y
```

### Running the conversion

Single file:

```bash
python mp4_to_mp3.py "../../data/lecture_mp4/lecture10-sequences_and_time_series.mp4"
```

Output is saved to `../../data/lecture_mp3/lecture10-sequences_and_time_series.mp3`.

Batch (all files in `data/lecture_mp4/`):

```bash
# Windows
convert_all_mp4.bat

# Linux / macOS
./convert_all_mp4.bash
```

---

## Step 2 — Transcribe MP3 to Markdown

`lecture_transcript_md.py` uses Whisper to transcribe an MP3 file and inserts section headings at the topic timestamps you provide.

### Python dependencies

```bash
pip install -r requirements.txt
```

PyTorch must be installed separately before the requirements file, as the correct variant depends on your hardware:

```bash
# NVIDIA GPU (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch
```

### Topics file format

Each lecture needs a `.txt` file in `data/lecture_content/` with this structure:

```
Title: My Lecture Title
Topic Name 0:03
Another Topic 10:12
Third Topic 1:05:30
```

Timestamps can be `M:SS` or `H:MM:SS`. The two-line format (topic name on one line, timestamp on the next) is also supported.

### Usage

Single file:

```bash
python lecture_transcript_md.py <input.mp3> <topics.txt> [output.md]
```

| Argument | Description |
|---|---|
| `<input.mp3>` | Path to the MP3 file |
| `<topics.txt>` | Path to the topics file |
| `[output.md]` | Optional output path — defaults to `../../data/lecture_transcript_raw/<name>.md` |

If the topics file is not found the script warns and continues, using a title derived from the filename.

Example:

```bash
python lecture_transcript_md.py \
  "../../data/lecture_mp3/lecture3-computer_vision_and_the_concepts_of_layers.mp3" \
  "../../data/lecture_content/lecture3-computer_vision_and_the_concepts_of_layers.txt"
```

Output: `../../data/lecture_transcript_raw/lecture3-computer_vision_and_the_concepts_of_layers.md`

Batch (all MP3s in `data/lecture_mp3/`, topics from `data/lecture_content/`):

```bash
# Windows
transcribe_all_mp3.bat

# Linux / macOS
./transcribe_all_mp3.bash
```

The batch scripts warn if a topics file is missing but still attempt transcription for every MP3.

---

## Step 3 — Refine the Raw Transcript

`lecture_transcript_refine.py` post-processes the raw Markdown output by passing each section through a small instruction-tuned language model. It removes verbal fillers ("um", "ah", "okay so", audio-check lines) and cleans up grammar, without summarising or losing any technical content.

The model is downloaded automatically from HuggingFace on the first run and cached in `~/.cache/huggingface/`. Subsequent runs use the cache with no re-download.

### Usage

```bash
python lecture_transcript_refine.py <input_raw.md> [model_id]
```

| Argument | Description |
|---|---|
| `<input_raw.md>` | Path to a raw transcript in `data/lecture_transcript_raw/` |
| `[model_id]` | HuggingFace model ID — defaults to `Qwen/Qwen2.5-0.5B-Instruct` |

The output is saved to the same relative path under `data/lecture_transcript/` (the `_raw` suffix is stripped from the folder name automatically).

Example:

```bash
python lecture_transcript_refine.py \
  "../../data/lecture_transcript_raw/lecture3-computer_vision_and_the_concepts_of_layers.md"
```

Output: `../../data/lecture_transcript/lecture3-computer_vision_and_the_concepts_of_layers.md`

### Choosing a model

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | Default. Fast on CPU, good enough for filler removal. |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | Noticeably better at following the "don't summarise" rule. |
| `microsoft/Phi-3-mini-4k-instruct` | ~7 GB | High quality, needs a GPU to be practical. |

Batch (all raw transcripts in `data/lecture_transcript_raw/`):

```bash
# Windows
refine_all_raw.bat

# Linux / macOS
./refine_all_raw.bash
```
