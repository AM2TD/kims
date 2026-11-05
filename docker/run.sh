#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p result
docker run --gpus all -it --rm \
    -u "$(id -u):$(id -g)" \
    -v "$(pwd)/data:/kims/data" \
    -v "$(pwd)/result:/kims/result" \
    "$@" kims:v3
