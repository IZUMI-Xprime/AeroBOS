@echo off
setlocal

echo ==============================================================================
echo                 AeroBOS Git Remote Push Utility
echo ==============================================================================

if "%~1"=="" (
    set /p REPO_URL="Enter remote Git repository URL (e.g. https://github.com/USER/aerobos.git): "
) else (
    set REPO_URL=%~1
)

if "%REPO_URL%"=="" (
    echo Error: No repository URL provided. Aborting.
    exit /b 1
)

echo.
echo [*] Checking git remote status...
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Updating existing 'origin' remote to: %REPO_URL%
    git remote set-url origin "%REPO_URL%"
) else (
    echo [*] Adding 'origin' remote: %REPO_URL%
    git remote add origin "%REPO_URL%"
)

echo [*] Current branch: main
git branch -M main

echo [*] Pushing codebase and 20-image benchmark artifacts to %REPO_URL%...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Successfully pushed AeroBOS repository to: %REPO_URL%
    echo [OK] All source code, 20-image dataset, and web portfolio are live!
) else (
    echo.
    echo [ERROR] Git push failed. Please check your credentials or access permissions.
    echo Tip: For GitHub, authenticate using 'git credential-manager' or a Personal Access Token (PAT).
)

endlocal
