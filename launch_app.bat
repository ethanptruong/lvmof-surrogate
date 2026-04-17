@echo off
REM Launch the LVMOF Synthesis Assistant Streamlit app on this workstation.
REM
REM   1. Activate the project Python environment if you use one
REM      (uncomment one of the lines below).
REM   2. Run from the repository root: launch_app.bat
REM   3. Chemists open http://<this-machine>:8501 in their browser.
REM
REM Set LVMOF_ADMIN=1 before launching to expose the Admin page.

cd /d "%~dp0"

REM ── Optional: activate environment ────────────────────────────────────────

call .venv\Scripts\activate.bat

REM ── Run Streamlit ─────────────────────────────────────────────────────────
streamlit run app\streamlit_app.py ^
    --server.port 8501 ^
    --server.address 0.0.0.0 ^
    --browser.gatherUsageStats false
