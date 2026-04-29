@echo off
setlocal enabledelayedexpansion

:: Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo Refining raw transcripts
echo ========================================

for %%f in (..\..\data\lecture_transcript_raw\*.md) do (
    echo Refining: %%f
    set "base=%%~nf"
    python lecture_transcript_refine.py "%%f"
    
    if !errorlevel! equ 0 (
        echo ✓ !base! refined successfully
    ) else (
        echo ✗ Failed to refine !base!
    )
    
    echo.
)

echo ========================================
echo All refinements completed!
echo ========================================

pause