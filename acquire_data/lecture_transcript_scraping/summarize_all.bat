@echo off
setlocal enabledelayedexpansion

:: Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo Summarizing refined transcripts
echo ========================================

for %%f in (..\..\data\lecture_transcript_refined\*.md) do (
    echo Summarizing: %%f
    set "base=%%~nf"
    python lecture_transcript_summarize.py "%%f"

    if !errorlevel! equ 0 (
        echo ✓ !base! summarized successfully
    ) else (
        echo ✗ Failed to summarize !base!
    )

    echo.
)

echo ========================================
echo All summarizations completed!
echo ========================================

pause
