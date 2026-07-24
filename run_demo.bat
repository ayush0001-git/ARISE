@echo off
REM ---------------------------------------------------------------------------
REM  ARISE demo: generate a synthetic observing night (with a hidden asteroid,
REM  transient and variable star) and run the full pipeline on it, then open the
REM  HTML report in your browser.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
echo.
echo === Running the ARISE end-to-end demo ===
echo (generating a synthetic night + reducing it + hunting for new sources)
echo.
python -m arise.cli demo --instrument dfot_2kx2k --frames 6 --open
if errorlevel 1 (
  echo.
  echo The demo failed. If this is the first run, run setup.bat first.
  pause
  exit /b 1
)
echo.
echo === Done. The report opened in your browser; data is under data\ ===
echo.
pause
