#!/bin/bash
# Build the KIMS evaluation image (run from the package root: kims/)
set -e
cd "$(dirname "$0")/.."
docker build -t kims -f docker/Dockerfile .
