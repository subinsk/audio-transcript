@echo off
echo ===================================
echo Audio Transcriber and Summarizer
echo ===================================
echo.

REM Check if Python virtual environment is activated
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Show usage information
echo BATCH PROCESSING OPTIONS:
echo Default: 5-minute batches
echo Custom: Add --batch-duration SECONDS
echo Example: run_transcriber.bat --batch-duration 180 (for 3-minute batches)
echo.

REM Run the transcription and summarization
echo Starting batch transcription and summarization...
echo.

if "%1"=="" (
    echo Processing webcams.webm with default settings...
    D:\projects\audio-transcripter\.venv\Scripts\python.exe audio_transcriber_summarizer.py
) else (
    echo Processing with custom parameters: %*
    D:\projects\audio-transcripter\.venv\Scripts\python.exe audio_transcriber_summarizer.py %*
)

echo.
echo ===================================
echo Batch processing completed!
echo Check the organized output folders.
echo ===================================
pause
