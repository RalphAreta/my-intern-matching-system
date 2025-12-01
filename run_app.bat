@echo off
echo Starting Internship Recommendation System...
echo.

REM Check if dependencies are installed
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
    echo.
)

REM Run the application
python app\main.py

