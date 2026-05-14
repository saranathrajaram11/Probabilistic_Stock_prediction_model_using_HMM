@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo The "py" launcher was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

echo Installing dependencies with: py -3.12 -m pip ...
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo If install failed on hmmlearn, use Python 3.12:  py install 3.12
  pause
  exit /b 1
)

echo.
echo Starting Streamlit. Open http://localhost:8501 in your browser.
echo.
py -3.12 -m streamlit run dashboard.py --server.port 8501
pause
