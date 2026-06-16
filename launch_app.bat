@echo off
REM Launch COMPASS (the LVMOF Synthesis Assistant) Streamlit app on this workstation.
REM
REM   1. Activate the project Python environment if you use one
REM      (uncomment one of the lines below).
REM   2. Run from the repository root: launch_app.bat
REM   3. Chemists open http://<this-machine>:8501 in their browser.
REM
REM Set LVMOF_ADMIN=1 before launching to expose the Admin page.
REM
REM Output is shown in the terminal AND written to logs\app_<timestamp>.log.

cd /d "%~dp0"

REM -- Optional: activate environment ---

call .venv\Scripts\activate.bat

REM -- Build timestamped log path ---
if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOGFILE=logs\app_%TS%.log
echo Logging to %LOGFILE%

REM -- Run Streamlit (tee to terminal + log file) ---
powershell -NoProfile -Command ^
    "& { streamlit run app\streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false *>&1 | Tee-Object -FilePath '%LOGFILE%' }"
