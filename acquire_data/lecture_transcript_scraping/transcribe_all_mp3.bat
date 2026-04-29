@echo off
setlocal enabledelayedexpansion

:: Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo Generating transcripts for MP3 files
echo ========================================

for %%f in (..\..\data\lecture_mp3\*.mp3) do (
    echo Transcribing: %%f
    set "base=%%~nf"
    if exist "..\..\data\lecture_content\!base!.txt" (
        echo Topics file found: ..\..\data\lecture_content\!base!.txt
    ) else (
        echo Warning: Topics file ..\..\data\lecture_content\!base!.txt not found. Proceeding without topics.
    )
    python lecture_transcribe.py "%%f" "..\..\data\lecture_content\!base!.txt" "..\..\data\lecture_transcript_raw\!base!.md"
    
    if !errorlevel! equ 0 (
        echo ✓ !base! transcribed successfully
    ) else (
        echo ✗ Failed to transcribe !base!
    )
    
    echo.
)

echo ========================================
echo All transcriptions completed!
echo ========================================

pause