# KIMS: Kinematics-Constrained Sampling Smoother for UWB–IMU Pose Estimation

KIMS is a fixed-lag smoother for UWB ranging (two-way ranging or TDOA) and an IMU. It does not sample poses.
It importance-samples **IMU input compensations** over a sliding window and rolls each sample out through the IMU
kinematics on the GPU, so every hypothesis is a kinematically consistent trajectory. The UWB likelihood of the
whole window gives each sample its weight.

This repository holds the estimator and the evaluation of the RA-L submission *KIMS: Kinematics-Constrained MPPI
Smoothing for UWB-IMU 6-DoF Pose Estimation* (under review, double-blind). Project page: https://am2td.github.io/kims

## Method (final configuration)

At every UWB epoch KIMS samples a 33-dimensional vector:

- a 9-dimensional offset of the window's start state (position, velocity, attitude), and
- IMU input corrections on all six channels, each a constant term plus three cosine basis functions (`K = 4`).

Each of the `N = 2048` samples is propagated through the IMU kinematics over the window and weighted by the range
likelihood. `N = 2048` is the operating point of the paper (Table II); the RMSE is flat from `N = 1024` upward, and the
reference runs behind the tables were made at `N = 8192`, which agrees with `N = 2048` within 1 mm of median RMSE.

- **Proposal.** The previous posterior shifted by one epoch (90 %), mixed with the prior (10 %).
- **Adaptive tempering.** It starts when the first-pass effective sample size (ESS) falls below `0.0244 N`.
  Each stage keeps the ESS at or above `4V` (132), refits a Gaussian and redraws. It stops at `beta = 1` with
  an ESS of at least `0.05 N`.
- **Between windows.** Only the information of the epoch that leaves the window is carried forward, as a
  Laplace arrival covariance. On MILUV, UTIL and IULUI the new window head is a Kalman update of the
  dead-reckoning prediction with that epoch (`head_filt:=1`). NTU VIRAL uses the smoothed head (`head_filt:=0`).
- **IMU bias at rest.** The static IMU bias from the first seconds at rest is removed (`still_bias:=1`).
- **Range likelihood.** The range noise model differs per dataset:

  | Dataset | Model |
  |---|---|
  | MILUV | LOS/NLOS Gaussian mixture fitted on a calibration flight |
  | NTU VIRAL | Tukey, σ 0.3 m, 5σ cut |
  | UTIL (TDOA) | Tukey, σ 0.224 m, 5σ cut (the devkit values) |
  | IULUI | Tukey, σ 0.11 m, 3σ cut |

- **Window length.** MILUV 3 s (229 epochs), NTU VIRAL 2 s (118), UTIL 2 s (100 epochs at 50 Hz), IULUI 2 s (200).
- **Computation.** The rollout runs in FP32 and accumulates the window cost in double precision
  (`rollout_fp32:=1`). The host side uses 8 OpenMP threads (`KIMS_OMP_THREADS=8`).
- **Matched LM.** `lm:=1` solves the same window objective with Levenberg–Marquardt and a Laplace approximation.
  It exists only for the solver comparison.

The output is the window head, a fixed lag of `T-1` epochs behind the newest measurement. The newest-epoch
(lag-0) state is written as well.

## Results

KIMS uses IMU and UWB only. The tables report the median 3D position RMSE in metres over each dataset's sequences,
from the fixed-lag output.

| Dataset | Sequences | KIMS | SFUISE | Dataset filter | KIMS lower than SFUISE / filter |
|---|---|---|---|---|---|
| MILUV | 13 | **0.135** | 0.704 (3 diverged) | 0.265, EKF + height sensor | 13/13, 13/13 |
| NTU VIRAL | 9 | **0.559** | 2.537 (2 diverged) | — | 6/9 |
| UTIL | 79 | 0.187 | **0.184** (5 diverged) | 0.217, ESKF | 33/79, 78/79 |
| IULUI | 3 | **0.120** | 0.235 | — | 2/3 |

These are the medians of Table I of the paper (MILUV, NTU VIRAL and IULUI KIMS values are means over ten seeds; the
UTIL table of the paper lists one median per constellation). Over the 104 sequences KIMS never diverges and exceeds
1 m of RMSE on two (`eee_03`, `nya_02`), SFUISE diverges on ten and exceeds 1 m on 16, and the dataset filters exceed
1 m on 3 of 92.

How the tables are scored:

- **Scoring span and alignment.** MILUV, UTIL and IULUI use the whole ground-truth span with no alignment
  (motion capture or Vicon in the anchor frame). NTU VIRAL is scored on the Leica prism point with a 4-DoF (yaw and
  translation) alignment, from 10 s after the start to 10 s before the end.
- **Divergence.** "Diverged" means an RMSE above 100 m. KIMS does not diverge on any of the 104 sequences.
- **SFUISE.** It runs with the authors' published configuration from the same start pose as KIMS.
- **MILUV filter.** It is the MILUV paper's own EKF, which adds a laser height sensor to IMU + UWB and starts from the
  ground-truth state (its Table 10). The table reports our run of the released devkit code, which reproduces the
  published per-axis values within 9 mm (the devkit reports a per-axis RMS; the paper's Table 10 value times √3 is
  0.228 m with VINS-Fusion odometry). Without the height sensor the devkit EKF reaches a median of 0.648 m.
- **UTIL filter.** It is our run of the released devkit ESKF (the UTIL paper's Table 4 filter). The run matches Table 4
  within 0.01 m on 45 of 79 flights and is 0.03-0.31 m lower on the 27 const4 trial 3-7 flights (median 0.217 against
  0.220 in Table 4).
- **Orientation.** MILUV, UTIL and IULUI have attitude ground truth (NTU VIRAL's Leica gives position only). The
  orientation RMSE is the RMS rotation angle to the motion-capture attitude. Medians: MILUV KIMS 6.9°, EKF + height 8.7°
  (KIMS lower on 12/13), SFUISE 13.1° (its three diverged runs ranked on top; 10.9° over the finite runs); UTIL KIMS
  3.1°, ESKF 3.3°, SFUISE 13.4°; IULUI KIMS 4.0°, SFUISE 14.3°. SFUISE reports attitude in its own gravity-levelled
  navigation frame; after removing one constant rotation per run for every method, the medians are MILUV 5.1° / 8.1° /
  5.7° and IULUI 2.3° / 6.2° (`result/final/orientation_rmse.md`).
- **MILUV anchors.** MILUV constellation 2 uses the corrected anchor order: anchors 2 and 3 are swapped in the
  devkit (issue #56), which the range residuals confirm.
- **Lag-0 output.** The median lag-0 RMSE is 0.212 m on MILUV, 0.456 m on NTU VIRAL and 0.204 m on UTIL.

Known limitation: with anchors close to a common plane, the range likelihood is symmetric about that plane, and
the estimate can follow the mirror image of the true path. NTU VIRAL's three anchors always lie in one plane, and
this is what happens on eee_03 (34 % of the scored time on the mirror side) and nya_02 (74 %), the two runs above 1 m.

Per-epoch solve time at `N = 8192`, run alone on an RTX 5070 Ti with an i7-14700K, FP32 rollout and 8 host threads:

| Dataset | Mean epoch interval | Mean | Median | 95th percentile |
|---|---|---|---|---|
| MILUV (3 flights) | 14.4 ms | 8.7 ms | 5.2 ms | 20.7 ms |
| NTU VIRAL (2 flights) | 16.5 ms | 3.8 ms | 3.8 ms | 3.9 ms |
| UTIL (2 flights) | 20.0 ms | 5.4 ms | 4.2 ms | 10.1 ms |
| IULUI (SquareTurn) | 10.0 ms | 5.3 ms | 5.3 ms | 5.4 ms |

On MILUV the tempered solves form the tail and arrive in bursts. On the Jetson Orin NX of the IULUI robot (MAXN, fixed
clocks, FP32, 8 host threads) the median solve time on `CoplanarSquareTurn` is 22.8 ms at `N = 8192` and 6.0-9.6 ms
from `N = 1280` to `2560`, below the 10 ms epoch interval, with the RMSE of `N = 8192` (paper, Fig. 6).

## Build

Requirements:

- an NVIDIA GPU with the CUDA toolkit;
- CMake 3.16 or newer, Eigen 3, OpenMP and Ceres (only the first-submission `sw_pgo` baselines link Ceres, but
  CMake needs it to configure);
- Python 3 with numpy, pandas, scipy and matplotlib.

The estimator does not need ROS.

```bash
cmake -S . -B build -DKIMS_NO_ROS=ON
cmake --build build -j8
```

This builds `build/bin/kims` and the first-submission baselines (`ekf_rts`, `particle_smoothing`, `sw_pgo`,
`sw_pgo_iului`).
The runner scripts call `${KIMS_BUILD:-build}/bin/kims`.
Tested on Ubuntu 20.04 with CUDA 12.8, CMake 3.16, GCC 9 and an RTX 5070 Ti. The CMake file pins a Release build with
`-O3 -march=native -ffp-contract=off`; set `KIMS_PORTABLE=1` in the environment to drop `-march=native`.

### Docker

With the NVIDIA container toolkit installed:

```bash
docker/build.sh        # image kims:v3 from nvidia/cuda:12.8.1-devel-ubuntu20.04, builds the binaries
docker/run.sh          # shell in the container with the GPU and this checkout's data/ and result/ mounted (as your user)
```

Inside the container the commands below work unchanged, e.g. `scripts/reproduce.sh iului tables`. The image carries
the IULUI data; the other datasets are prepared on the host as described next and reach the container through the
`data/` mount.

## Data

| Dataset | In the repository | Preparation |
|---|---|---|
| IULUI (ours) | `data/iului/<seq>/{imu,uwb_range,mocap}.csv`; anchors, tag offsets and start pose in `data/iului/<seq>.json` (`AC1_<seq>.json` for the AC1 flights) | ships |
| MILUV | anchor files `data/miluv/anchors_constellation*.txt`, `experiments.csv` | `scripts/prepare_data.sh miluv` fetches the 13 single-robot experiments (ifo001 CSVs) from figshare |
| NTU VIRAL | anchor files `data/ntu/anchors_*.txt`, start poses `data/ntu/init_poses.json` | download `eee_01` … `sbs_03` from the NTU VIRAL website, then `scripts/prepare_data.sh ntu <dir with the zips>` (needs ROS `rosbag`) |
| UTIL | — | unpack the UTIL release, then `scripts/prepare_data.sh util <UTIL root>` (all 79 flights) |

IULUI contains the AC1 flights `SquareTurn`, `CircleSpin` and `SquareBob` (four coplanar anchors at 1.30 m) and the
AC3 flight `AC3_125000` (the CircleSpin maneuver under anchors spread from 0.24 m to 2.20 m). The paper evaluates
`SquareTurn`, `CircleSpin` and `AC3_125000` under the names `CoplanarSquareTurn`, `CoplanarCircleSpin` and
`NoncoplanarCircleSpin`.

## Reproducing the tables

```bash
scripts/reproduce.sh                  # KIMS on all 104 sequences at N = 2048 (3 runs at a time), then the tables
scripts/reproduce.sh miluv tables     # one dataset only
```

The KIMS rows go to `result/runlogs/repro_rmse.txt`, and the per-sequence MILUV and NTU VIRAL values to
`result/runlogs/repro_table_v2.txt`. The SFUISE and devkit columns are filled in when their outputs exist
([baselines/README.md](baselines/README.md)). The reference values of the paper are in
[result/final/](result/final/).

Single runs, with the paper's settings (the runners default to `KIMS_OMP_THREADS=8`, the host threads of the final
configuration):

```bash
F="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=2048 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1 rollout_fp32:=1"

# MILUV
scripts/revision/hp_mix.sh my_random3_0 default_1_random3_0 $F T:=229

# IULUI
scripts/revision/iu_run.sh my_SquareTurn SquareTurn T:=200 N:=2048 head_filt:=1 still_bias:=1 group_same_ts:=1 rollout_fp32:=1

# UTIL
scripts/revision/util_run.sh my_c1t1 const1-trial1-tdoa2 N:=2048 rollout_fp32:=1

# NTU VIRAL: start pose and anchors from data/ntu/init_poses.json
scripts/revision/ntu_run.sh my_sbs_02 sbs_02 T:=118 N:=2048 init_x:=16.039 init_y:=-12.987 init_z:=0.2 init_yaw:=116.0 \
    anchor_file:=$PWD/data/ntu/anchors_sbs_02.txt still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5 rollout_fp32:=1
```

Other options:

- `seed:=<n>` sets another random seed, and `N:=<n>` another sample count (the runner scripts default to `N:=8192`,
  the reference configuration; `reproduce.sh` and the examples above pass `N:=2048`).
- `lm:=1` runs the matched LM.

Each run writes `result/<dataset>/diag/<label>_*.txt` and the log `result/runlogs/<label>.log`. Keep the number of
concurrent runs within the GPU memory (about 0.8 GB each). Runs sharing the GPU give bit-identical outputs. Another
host thread count changes the summation order of the weight sums, so keep 8 threads to reproduce the tables exactly
(1 and 8 threads agree to four decimals in RMSE).

| File | Content |
|---|---|
| `<label>_pose.txt` | rows `t x y z qx qy qz qw`; each row carries the newest epoch's time and the window-head state from `T-1` epochs earlier, so scoring shifts the pose by `T-1` rows |
| `<label>_now_pose.txt` | state at the newest epoch (lag 0) |
| `<label>_flush_pose.txt` | all states of the first and last windows, used for whole-span scoring |
| `<label>_timing.txt` | per-epoch processing time in ms |
| `<label>_ess.txt`, `<label>_bias.txt`, `<label>_head.txt` | effective sample size and tempering passes, bias and head-filter diagnostics per solve |

## Repository layout

| Path | Content |
|---|---|
| `src/offline/kims_node.cu`, `src/offline/kims_solver.cu` | estimator: data loading, window, sampler, tempering, matched LM |
| `include/` | solver and dynamics headers (FP64 and FP32 rollout), dataset presets, CSV loading |
| `src/offline/{ekf_rts,particle_smoothing,sw_pgo,sw_pgo_iului}.cpp` | first-submission baselines (not used in the revised tables) |
| `baselines/` | how to run SFUISE and the dataset devkit filters, with the patches used |
| `scripts/run.py`, `scripts/datasets.py` | per-dataset command-line composition used by the runners |
| `scripts/prepare_data.sh`, `scripts/fetch_miluv.py`, `scripts/ntu_extract_bag.py` | dataset preparation |
| `scripts/reproduce.sh` | full reproduction of the KIMS results |
| `scripts/revision/` | per-dataset runners, scoring, tables, figures and the experiments of the revision |
| `docker/` | Dockerfile and helper scripts (see Build) |
| `config/imu_noise_*.yaml` | reference IMU noise values (the runners pass their own `q_acc`/`q_gyr`) |
| `result/final/` | the paper's tables and figures |

## License

See [LICENSE](LICENSE).
