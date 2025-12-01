@echo off
echo ========================================
echo Internship Recommendation System
echo EXE Build Script
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    echo.
)

REM Build the executable
echo Creating executable...
echo This may take a few minutes...
echo.

REM Use PyInstaller with proper path handling
pyinstaller --noconsole --onefile ^
    --name "InternshipRecommendationSystem" ^
    --add-data "datasets;datasets" ^
    --hidden-import=sklearn.metrics.pairwise ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=sklearn ^
    --hidden-import=mlxtend.frequent_patterns ^
    --hidden-import=joblib ^
    app\main.py

echo.
echo ========================================
if exist "dist\InternshipRecommendationSystem.exe" (
    echo Build SUCCESSFUL!
    echo.
    echo Executable location: dist\InternshipRecommendationSystem.exe
    echo.
    echo IMPORTANT: Copy the 'datasets' folder to the same directory
    echo            as the EXE file for the application to work.
) else (
    echo Build FAILED. Check the error messages above.
)
echo ========================================
pause

