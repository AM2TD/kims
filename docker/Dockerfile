# KIMS: reproducible evaluation container (estimator only, no ROS)
#
#   docker/build.sh                      # builds the image kims:v3 (needs the NVIDIA container toolkit)
#   docker/run.sh                        # opens a shell with the GPU, data/ and result/ of this checkout mounted
#   (inside)  scripts/reproduce.sh iului tables
#             scripts/reproduce.sh       # all 104 flights once data/miluv, data/ntu and data/util are prepared
#
# The image carries the IULUI data; the other datasets are mounted from the host (see README, "Data").

FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libceres-dev libeigen3-dev \
        python3 python3-numpy python3-pandas python3-scipy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /kims
COPY . /kims

# KIMS_PORTABLE drops -march=native so the image runs on any x86-64 host; the GPU
# code is PTX and is compiled for the host GPU by the driver at first launch.
ENV KIMS_PORTABLE=1
RUN cmake -S . -B build -DKIMS_NO_ROS=ON && cmake --build build -j"$(nproc)"

ENV KIMS_OMP_THREADS=8
CMD ["bash"]
