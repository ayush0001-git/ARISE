@echo off
REM ---------------------------------------------------------------------------
REM  ARISE web console: drag-and-drop frontend + Ask-ARISE assistant.
REM  Opens http://127.0.0.1:8770 in your browser.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
start "" http://127.0.0.1:8770
python -m arise.webapp
if errorlevel 1 (
  echo.
  echo The console failed to start. If this is the first run, run setup.bat first.
  pause
)
