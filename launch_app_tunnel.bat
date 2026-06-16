@echo off
REM Launch COMPASS (LVMOF Synthesis Assistant) + a Cloudflare Quick Tunnel so the
REM app is reachable from anywhere via a temporary https://*.trycloudflare.com URL.
REM
REM Usage:
REM   1. Double-click this file (or run from a terminal in the repo root).
REM   2. Two windows open: one runs Streamlit, the other runs cloudflared.
REM   3. The cloudflared window will print a line like
REM         https://<random-words>.trycloudflare.com
REM      Open that URL on any device (your lab laptop) to use the app.
REM   4. Close either window to stop the demo.
REM
REM Notes:
REM   - The URL is regenerated each time you launch. Start the tunnel before
REM     the demo and copy the link to your laptop.
REM   - Keep this PC awake and on the network for the duration of the demo.
REM   - Set LVMOF_ADMIN=1 before launching to expose the Admin page.

cd /d "%~dp0"

call .venv\Scripts\activate.bat

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set APP_LOG=logs\app_%TS%.log
set TUN_LOG=logs\tunnel_%TS%.log

echo App log:    %APP_LOG%
echo Tunnel log: %TUN_LOG%
echo.

REM -- Start Streamlit in a new window (bound to localhost is enough; the tunnel reaches it locally) --
start "COMPASS Streamlit" powershell -NoExit -NoProfile -Command ^
    "& { streamlit run app\streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false *>&1 | Tee-Object -FilePath '%APP_LOG%' }"

REM -- Give Streamlit a moment to bind the port before the tunnel attaches --
powershell -NoProfile -Command "Start-Sleep -Seconds 4"

REM -- Start the Cloudflare quick tunnel in a second window --
REM    Watch this window for a line like: https://xxxx.trycloudflare.com
start "Cloudflare Tunnel" powershell -NoExit -NoProfile -Command ^
    "& { & 'C:\Program Files (x86)\cloudflared\cloudflared.exe' tunnel --url http://localhost:8501 *>&1 | Tee-Object -FilePath '%TUN_LOG%' }"

echo.
echo Both processes launched. Copy the https://*.trycloudflare.com URL from the
echo "Cloudflare Tunnel" window and open it on your laptop.
echo.
pause
