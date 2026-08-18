# KIMS: Kinematic-Constrained MPPI Smoothing for UWB-IMU 6-DoF Pose Estimation

[**🌐 Project Page**](https://am2td.github.io/kims/) | [**📄 Paper**](https://am2td.github.io/kims/#paper) | [**🎬 Demo**](https://am2td.github.io/kims/#demo)

When UWB anchors sit at nearly the same height, the range geometry admits a **mirrored solution** on the other
side of the anchor plane, and once a filter or optimizer drifts into that reflected basin it never
comes back. **KIMS** recasts MPPI as a fixed-lag smoother that samples *IMU input compensations*
instead of poses: every candidate trajectory is generated through IMU kinematic propagation, so
rollouts stay in the basin of the current estimate — no anchor-plane model, no structural prior.

- 🛡️ **Flip-ambiguity robust** — tracks flights that repeatedly cross an elevated anchor plane
- ⚡ **Real-time on GPU** — 10,000 rollouts per UWB epoch, runs onboard (Jetson Orin NX)
- 📦 **Batteries included** — all three datasets ship in `data/`, nothing to download


---

## 🛰️ The IULUI Dataset

This repository ships **IULUI**, our UWB-IMU flight campaign built specifically to study
**anchor-plane degeneracy**. Four anchors sit at nearly the same height (~1.3 m) and the
vehicle repeatedly and *legitimately* crosses the anchor plane — exactly the regime where
known-side or height-bound priors reject valid states and local estimators lock onto the mirror.

| Sequence | Motion | What it stresses |
|---|---|---|
| `SquareBob` | square path with vertical bobbing | repeated plane crossings |
| `SquareTurn` | square path with yawing corners | orientation coupling through the two-tag lever arm |
| `CircleSpin` | 2 m/s circle while spinning in yaw | continuous yaw + crossings, the hardest case |

Each sequence provides raw UWB ranges (2 tags × 4 anchors, Nooploop LinkTrack), a IMU, and
motion-capture ground truth — as both a rosbag and pre-extracted CSV.

> [!NOTE]
> Sequences from **MILUV** and **NTU VIRAL** are included for evaluation as well.

---

## 🛠️ Installation

```bash
sudo apt install build-essential cmake libeigen3-dev libceres-dev \
                 python3-numpy python3-matplotlib
```

plus a **CUDA toolkit** and an NVIDIA GPU. **ROS Noetic is only needed for the online node.**

### 🐳 Docker

Everything above, prebuilt:

```bash
./docker/build.sh        # ~10 min once
./docker/run.sh          # opens a shell; result/ is mounted back to the host
```

> [!IMPORTANT]
> The container needs GPU access — `docker/run.sh` already passes `--gpus all`
> (requires `nvidia-container-toolkit` on the host).

---

## 🚀 Quick Start — Offline

The offline tools read CSV and write TUM-format trajectories. **No ROS required.** This is the
path that reproduces the paper.

### 1. Build

```bash
# standalone, anywhere:
cmake -S . -B build -DKIMS_NO_ROS=ON
cmake --build build -j8            # binaries in build/bin/

# or inside a catkin workspace (also builds the online node):
catkin_make --pkg kims && source devel/setup.bash
```

### 2. Run

```bash
# one sequence
python3 scripts/run.py --method kims --dataset iului --sequence CircleSpin

# one dataset, every sequence
python3 scripts/run.py --method kims --dataset iului

# the full paper sweep: 3 datasets × 3 sequences × 5 methods (~25 min)
python3 scripts/run.py --method all --dataset all
```

| flag | choices |
|---|---|
| `--method` | `kims` · `ekf_rts` · `ukf_rts` · `particle_smoothing` · `sw_pgo` · `all` |
| `--dataset` | `iului` · `miluv` · `ntu` · `all` |
| `--sequence` | a sequence name · `all` |

### 3. Evaluate

```bash
python3 scripts/plot.py                  # everything found under result/
python3 scripts/plot.py --dataset iului  # one dataset
```



### 4. Tweak

All estimator settings are compiled in (`include/presets.h`, one preset per dataset) so the
released binaries are exactly the published configuration — but every value can be overridden at
the command line without rebuilding:

```bash
python3 scripts/run.py --method kims --dataset ntu -- N:=3000 T:=10 gamma:=5.0
```

> [!NOTE]
> The compensation penalty of Eq. (7) in the paper is the quadratic form on MILUV
> (`ctrl_mode 0`, `Σu = diag(σ²/w)`) and the bounded per-axis Tukey form on IULUI and
> NTU VIRAL (`ctrl_mode 1`). The Tukey scales `sigma_imu_acc`/`sigma_imu_gyr` are
> 2.33/0.17 on IULUI and 0.33/0.06 on NTU VIRAL, with cutoff `ctrl_delta` 0.86 and 1.00.
> All of these live in `include/presets.h` and can be overridden as `sigma_imu_acc:=`,
> `ctrl_delta:=`, `ctrl_mode:=` at the command line.

---

## 📡 Quick Start — Online

`kims_online` is a real-time ROS node running the same fixed-lag smoother: subscribe to IMU + UWB,
publish `kims/pose`, `kims/odom`, and TF (`map → base_link`).

First, the UWB driver messages ([`nlink_parser`](https://github.com/nooploop-dev/nlink_parser))
must be in the same workspace:

```bash
cd <catkin_ws>/src
git clone --recursive https://github.com/nooploop-dev/nlink_parser.git   # --recursive matters!
cd .. && catkin_make && source devel/setup.bash
```

Then pick any of the three modes:

```bash
# ① replay a dataset sequence as live topics — works for all three datasets
roslaunch kims kims_online.launch dataset:=iului sequence:=CircleSpin
roslaunch kims kims_online.launch dataset:=miluv sequence:=default_1_random3_0
roslaunch kims kims_online.launch dataset:=ntu   sequence:=eee_01

# ② replay the IULUI rosbag through the Nooploop driver topic
roslaunch kims kims_online.launch bag:=$(rospack find kims)/data/iului/AC1_CircleSpin.bag

# ③ live sensors — remap to your topics, deployment geometry in config/online_<dataset>.yaml
roslaunch kims kims_online.launch dataset:=iului play:=false \
    imu_topic:=/mavros/imu/data uwb_linktrack_topic:=/nlink_linktrack_tagframe0
```

For your own vehicle, publish one `kims/UwbRange` message per tag-anchor range and the node needs
nothing else — anchors, tag lever arms and tag ids are plain rosparams.

> [!WARNING]
> `rate:=` speeds up replay, but only as far as the machine solves in real time. If solving falls
> behind the sensor stream the window goes stale and the estimate degrades — keep `rate:=1.0` for
> meaningful numbers.

---

## 📁 Repository Structure

```
kims/
├── include/            solver + IO headers · presets.h = every published setting
├── src/
│   ├── offline/        kims, ekf_rts, ukf_rts, particle_smoothing, sw_pgo
│   └── online/         kims_online ROS node
├── scripts/
│   ├── run.py          runs any method on any dataset
│   ├── plot.py         RMSE table + figures
│   ├── replay.py       streams a CSV sequence as ROS topics
│   └── datasets.py     paths, sequences, evaluation conventions
├── launch/             one file per method + kims_online.launch
├── config/             anchor survey & tag geometry for the online node
├── data/               IULUI · MILUV · NTU VIRAL (ready to run)
├── docker/             Dockerfile, build.sh, run.sh
└── result/             outputs, one directory per sequence
```

---

## ⚖️ License

MIT — see [LICENSE](LICENSE).
