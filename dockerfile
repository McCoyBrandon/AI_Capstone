# syntax=docker/dockerfile:1.6
FROM --platform=$BUILDPLATFORM python:3.11-slim AS base

# Minimal OS deps (add more if your requirements need them, e.g., gcc, libgomp1, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "infer.py"]