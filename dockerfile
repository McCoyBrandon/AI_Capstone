# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. System deps (for numpy / scipy / sklearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files 
COPY . .

# 6. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 7. Make sure runs directory exists inside container
RUN mkdir -p /app/runs

# 8. Default command: train with run_name=Test_run_1
# (You can override this in `docker run` if you want eval or infer instead.)
CMD ["python", "-m", "src.training.train", "--run_name", "Test_run_1"]
