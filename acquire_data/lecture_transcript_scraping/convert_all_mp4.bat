@echo off
setlocal enabledelayedexpansion

:: Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo Converting MP4 files to MP3
echo ========================================

for %%f in (..\..\data\lecture_mp4\*.mp4) do (
    echo Converting: %%f
    python mp4_to_mp3.py "%%f" "..\..\data\lecture_mp3\%%~nf.mp3"
    
    if !errorlevel! equ 0 (
        echo ✓ %%~nf converted successfully
    ) else (
        echo ✗ Failed to convert %%~nf
    )
    
    echo.
)

echo ========================================
echo All conversions completed!
echo ========================================

pause