#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker build -t kims:v3 -f docker/Dockerfile .
