#!/bin/bash
# Build the KIMS evaluation image (run from anywhere inside the checkout).
set -e
cd "$(dirname "$0")/.."
docker build -t kims:v3 -f docker/Dockerfile .
