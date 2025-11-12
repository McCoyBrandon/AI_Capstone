# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy only what’s needed
COPY src ./src
COPY infer.py .

# 4. Environment variables
#ENV PYTHONUNBUFFERED=1
#ENV PYTHONDONTWRITEBYTECODE=1

# 5. Create output directory
RUN mkdir -p /app/runs

# 6. Default command: run inference
CMD ["python", "infer.py", \
     "--ckpt", "runs/ai4i_run_1/model.ckpt", \
     "--csv", "src/data/ai4i2020.csv", \
     "--out", "runs/ai4i_run_1/infer_metrics.json", \
     "--failures-csv", "runs/ai4i_run_1/flagged_failures.csv"]
