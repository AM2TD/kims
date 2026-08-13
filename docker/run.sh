#!/bin/bash
# Start the container with GPU access; result/ is mounted back to the host.
set -e
cd "$(dirname "$0")/.."
mkdir -p result
docker run --gpus all -it --rm -v "$(pwd)/result:/ws/src/kims/result" kims
