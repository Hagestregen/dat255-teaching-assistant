@echo off
setlocal enabledelayedexpansion

:: Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo Starting to scrape Deep Learning with Python 3rd Edition
echo Chapters given in scrape_commands.txt
echo ========================================

if not exist scrape_commands.txt (
    echo Error: scrape_commands.txt not found!
    pause
    exit /b 1
)

:: Count total lines
for /f %%a in ('find /c /v "" ^< scrape_commands.txt') do set total=%%a

set current=1

for /f "usebackq delims=" %%c in ("scrape_commands.txt") do (
    set "command=%%c"
    
    :: Skip empty lines and comments
    if not "!command!"=="" (
        if "!command:~0,1!" NEQ "#" (
            echo [!current!/%total%] Running: !command!
            echo ----------------------------------------
            
            :: Execute the command
            call !command!
            
            if %errorlevel% equ 0 (
                echo ✓ Chapter completed successfully
            ) else (
                echo ✗ Failed to scrape this chapter
            )
            
            echo.
            set /a current+=1
            
            :: Optional delay (1 second)
            timeout /t 1 >nul
        )
    )
)

echo ========================================
echo All scraping tasks completed!
echo ========================================

pause