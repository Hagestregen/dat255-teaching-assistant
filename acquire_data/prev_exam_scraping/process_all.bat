@echo off
:: process_all.bat
:: Runs PDF extraction script using PyMuPDF on every .pdf file in data\previous_exam\
:: and writes JSON output to data\previous_exam_extracted_raw\
::
:: Usage (from repo root or from this script's folder):
::   acquire_data\prev_exam_scraping\process_all.bat

setlocal enabledelayedexpansion

:: Resolve paths relative to this script
set "SCRIPT_DIR=%~dp0"
:: Strip trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Go up two levels to repo root
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

set "EXAM_DIR=%REPO_ROOT%\data\previous_exam"
set "OUT_DIR=%REPO_ROOT%\data\previous_exam_extracted_raw"

set "SCRIPT_MU=%SCRIPT_DIR%\extract_pymupdf.py"
if not exist "%OUT_DIR%"  mkdir "%OUT_DIR%"

echo ========================================
echo Extracting exam questions from PDFs
echo   Input : %EXAM_DIR%
echo   Output: %OUT_DIR%
echo ========================================

set "total=0"
for %%F in ("%EXAM_DIR%\*.pdf") do set /a total+=1

if %total%==0 (
    echo No .pdf files found in %EXAM_DIR%
    pause
    exit /b 1
)

set "current=1"

for %%F in ("%EXAM_DIR%\*.pdf") do (
    set "pdf=%%F"
    set "stem=%%~nF"

    echo.
    echo [!current!/%total%] %%~nxF
    echo ----------------------------------------

    echo   [pymupdf]
    python "%SCRIPT_MU%" "%%F" "%OUT_DIR%\!stem!.json"
    if !errorlevel! equ 0 (
        echo   -^> %OUT_DIR%\!stem!.json
    ) else (
        echo   ERROR: pymupdf extraction failed
    )

    set /a current+=1
)

echo.
echo ========================================
echo Done. Files written to %OUT_DIR%
echo ========================================
pause
