# KIMS — reproducible evaluation container
#
#   docker build -t kims -f docker/Dockerfile .
#   docker run --gpus all -it --rm -v "$(pwd)/result:/ws/src/kims/result" kims
#   (inside)  python3 src/kims/scripts/run.py --method all --dataset all
#             python3 src/kims/scripts/plot.py
#
# Requires the NVIDIA container toolkit (--gpus all).

FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
SHELL ["/bin/bash", "-c"]

# ── ROS Noetic (ubuntu 20.04) ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg2 lsb-release ca-certificates && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros1.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-ros-base ros-noetic-rosbag ros-noetic-sensor-msgs \
        ros-noetic-geometry-msgs ros-noetic-nav-msgs ros-noetic-tf2-ros \
        ros-noetic-serial \
        build-essential git \
        libceres-dev libeigen3-dev \
        python3-numpy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

# ── workspace: nlink_parser (UWB message definitions) + kims ─────────────────
WORKDIR /ws/src
RUN git clone --depth 1 --recursive https://github.com/nooploop-dev/nlink_parser.git
COPY . /ws/src/kims

WORKDIR /ws
ENV KIMS_PORTABLE=1
RUN source /opt/ros/noetic/setup.bash && catkin_make -j"$(nproc)"

RUN echo "source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash" \
        >> /root/.bashrc
WORKDIR /ws
CMD ["bash"]
