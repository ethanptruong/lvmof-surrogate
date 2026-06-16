#!/usr/bin/env sh
# COMPASS container entrypoint.
#
# Relocates the app's MUTABLE state - the experiment Excel, model checkpoints,
# and generated docs - onto the mounted persistent volume so it survives
# restarts and redeploys.  Each directory is seeded from the image on FIRST
# boot only; after that the volume copy is the source of truth.
#
# This is why no Python paths had to change: the app still reads/writes
# ./data, ./checkpoints and ./docs as before - those are now symlinks to the
# persistent volume.
set -eu

VOL="${COMPASS_DATA_DIR:-/data}"

for d in data checkpoints docs; do
  # First boot: the volume has no copy yet -> seed it from the image (if the
  # image shipped that directory; checkpoints is intentionally not shipped).
  if [ ! -e "$VOL/$d" ]; then
    mkdir -p "$VOL/$d"
    if [ -d "/app/$d" ]; then
      cp -a "/app/$d/." "$VOL/$d/" 2>/dev/null || true
    fi
  fi
  # Point the in-image directory at the persistent copy.
  rm -rf "/app/$d"
  ln -s "$VOL/$d" "/app/$d"
done

# Evaluation figures/CSVs (ROC, SHAP, confusion matrices, ...) are written to
# the project root, which is the ephemeral image layer. Retrains persist them
# to $VOL/artifacts; overlay those onto the baked-in copies at every boot so
# the website always shows the latest model's plots.
if [ -d "$VOL/artifacts" ]; then
  cp -a "$VOL/artifacts/." /app/ 2>/dev/null || true
fi

exec streamlit run app/streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
