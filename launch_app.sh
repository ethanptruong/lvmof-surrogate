#!/usr/bin/env bash
# Launch the LVMOF Synthesis Assistant Streamlit app on this workstation.
#
#   1. Activate the project Python environment if you use one
#      (uncomment one of the lines below).
#   2. Run from the repository root:  ./launch_app.sh
#   3. Chemists open http://<this-machine>:8501 in their browser.
#
# Export LVMOF_ADMIN=1 before launching to expose the Admin page.

set -euo pipefail
cd "$(dirname "$0")"

# ── Optional: activate environment ────────────────────────────────────────
# Windows (Git Bash) uses .venv/Scripts; Linux/macOS uses .venv/bin.
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
# conda activate lvmof

# ── Run Streamlit ─────────────────────────────────────────────────────────
exec streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
