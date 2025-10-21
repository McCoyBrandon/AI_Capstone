#!/usr/bin/env bash
set -euo pipefail

# One-time: enable qemu binfmt for cross-arch build + run
docker run --rm --privileged tonistiigi/binfmt --install all

# Create & bootstrap buildx builder (idempotent)
docker buildx create --use --name multiarch-builder || true
docker buildx inspect --bootstrap

# Build & (optionally) push multi-arch image
IMAGE="${IMAGE:-yourname/edge-infer:latest}"

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "$IMAGE" \
  -f Dockerfile \
  --push .