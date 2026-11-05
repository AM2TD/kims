FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libceres-dev libeigen3-dev \
        python3 python3-numpy python3-pandas python3-scipy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /kims
COPY . /kims

ENV KIMS_PORTABLE=1
RUN cmake -S . -B build -DKIMS_NO_ROS=ON && cmake --build build -j"$(nproc)"

ENV KIMS_OMP_THREADS=8
CMD ["bash"]
