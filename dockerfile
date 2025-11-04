# Base (build + runtime)
FROM python:3.11-slim AS base

# System deps (build tools + runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy just requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source (keep structure as in your repo)
# If your repo root has src/, tests/, runs/, etc. — copy everything
COPY . /app

# Make sure Python can find your packages
ENV PYTHONPATH=/app

# (We'll create entrypoint.sh below.)
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
'PIPELINE="${1:-infer}"' \
'shift || true' \
'case "$PIPELINE" in' \
'  infer) python -m src.training.infer "$@" ;;' \
'  train) python -m src.training.train "$@" ;;' \
'  grid|grid_search) python -m src.training.grid_search_train "$@" ;;' \
'  eval)  python -m src.training.eval "$@" ;;' \
'  *)     echo "Unknown pipeline: $PIPELINE" >&2; exit 2 ;;' \
'esac' \
> /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Default command = infer pipeline
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["infer"]
