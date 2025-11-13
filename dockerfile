# 1. Base image
FROM python:3.11-slim

# 2. Plotform
ARG TARGETPLATFORM 
# Options: linux/amd64, linux/arm64/v8, linux/arm/v7

# 2. Set working directory
WORKDIR /app

# 3. System deps (Needed for numpy and sklearn compatibility with certain platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# 4. Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# 5. Install the correct torch wheel per-arch (CPU only)
ARG TORCH_VER="2.3.1"
# Adjust TORCH_VER if you need a specific version.
RUN case "$TARGETPLATFORM" in \
      "linux/amd64")  pip install --no-cache-dir torch==${TORCH_VER} --index-url https://download.pytorch.org/whl/cpu ;; \
      "linux/arm64"*) pip install --no-cache-dir torch==${TORCH_VER} --index-url https://download.pytorch.org/whl/cpu ;; \
      "linux/arm/v7") pip install --no-cache-dir torch==${TORCH_VER} --index-url https://download.pytorch.org/whl/cpu ;; \
      *) echo "Unsupported TARGETPLATFORM: $TARGETPLATFORM" && exit 1 ;; \
    esac

# 4. Copy only what’s needed
COPY src ./src
COPY infer.py .

# 4. Environment variables
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# 5. Create output directory
RUN mkdir -p /app/runs

# 6. Default command: run inference
CMD ["python", "infer.py", \
     "--ckpt", "runs/ai4i_run_1/model.ckpt", \
     "--csv", "src/data/ai4i2020.csv", \
     "--out", "runs/ai4i_run_1/infer_metrics.json", \
     "--failures-csv", "runs/ai4i_run_1/flagged_failures.csv"]
