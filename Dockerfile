# COMPASS — LVMOF synthesis assistant (Streamlit + heavy ML stack)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache the ChemBERTa model download on the persistent volume so it is
    # fetched once and survives redeploys.
    HF_HOME=/data/.hf_cache

# System libraries the scientific stack needs:
#   build-essential/gcc/g++      -> compile dscribe, llvmlite (umap-learn), etc.
#   libgomp1                     -> lightgbm / xgboost OpenMP runtime
#   libxrender1/libxext6/libsm6  -> rdkit 2D drawing
#   git                          -> a couple of pip deps fetch from VCS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git \
        libgomp1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of torch first (transformers/ChemBERTa needs it).
# The default torch wheel bundles CUDA and is ~2 GB larger — we don't have a GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then the project deps. Streamlit, filelock (cross-process Excel lock) and
# pandas are runtime deps that aren't pinned in requirements.txt, so add them.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install streamlit filelock pandas

# App code + seed data (experiment Excel, COSMO solvent profiles, lab logo).
# .dockerignore controls exactly what is baked in.
COPY . .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
