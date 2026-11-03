#!/bin/bash
# Start the container with the GPU; data/ and result/ of this checkout are mounted so prepared datasets are
# visible inside and run outputs land on the host, owned by the calling user. Extra arguments are passed to docker run.
set -e
cd "$(dirname "$0")/.."
mkdir -p result
docker run --gpus all -it --rm \
    -u "$(id -u):$(id -g)" \
    -v "$(pwd)/data:/kims/data" \
    -v "$(pwd)/result:/kims/result" \
    "$@" kims:v3
