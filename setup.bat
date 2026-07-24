@echo off
REM ---------------------------------------------------------------------------
REM  ARISE one-time setup (Windows). Installs Python dependencies.
REM ---------------------------------------------------------------------------
echo.
echo === ARISE setup: installing Python dependencies ===
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found on your PATH.
  echo Install Python 3.10+ from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo Dependency installation failed. See the messages above.
  pause
  exit /b 1
)
echo.
echo === Setup complete. Double-click run_demo.bat to see ARISE in action. ===
echo.
pause
