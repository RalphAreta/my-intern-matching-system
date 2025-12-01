@echo off
echo ========================================
echo Push to GitHub
echo ========================================
echo.
echo Make sure you have:
echo 1. Created the repository on GitHub
echo 2. Updated the remote URL if needed
echo.
pause

echo.
echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ERROR: Push failed!
    echo.
    echo Possible reasons:
    echo - Repository doesn't exist on GitHub yet
    echo - Authentication required (use GitHub Desktop or configure credentials)
    echo - Remote URL is incorrect
    echo.
    echo To update remote URL:
    echo   git remote set-url origin https://github.com/YOUR_USERNAME/my-intern-matching-system.git
) else (
    echo.
    echo SUCCESS! Your code has been pushed to GitHub.
)

pause


