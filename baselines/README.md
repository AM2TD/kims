# Baselines

The revised paper compares KIMS with two kinds of IMU + UWB baselines on every dataset:

- **SFUISE** (Li, Cao and Hanebeck, RA-L 2023, <https://github.com/ASIG-X/SFUISE>): a continuous-time
  sliding-window spline smoother, run with the authors' published configurations.
- **The dataset's own filter**, where the dataset provides one. NTU VIRAL and IULUI have none.
  - MILUV: the table uses the MILUV paper's own EKF results with an extra sensor, IMU + UWB + laser height (Table 10),
    multiplied by √3 (see below). The figures and the orientation scores use our run of the devkit EKF with the height
    sensor, which reproduces Table 10 (the paper reports position only).
  - UTIL: the table uses the UTIL paper's ESKF (Table 4, per flight); the figures use our run of the devkit ESKF
    (below), which matches Table 4 on 45 of 79 flights and is lower on the const4 trial 3-7 flights.

All baselines start from the same initial pose as KIMS. Their outputs go to `result/<dataset>/sfuise_<sequence>_pose.txt`
(and `result/util/eskf_<flight>_pose.txt`) and are scored by `scripts/evaluate.py sfuise`, like the KIMS runs.
The other two baselines of the paper, SW-PGO and the bootstrap particle filter, are part of this repository
(`scripts/run.py swpgo|pf`).

The paths below are environment variables with defaults:

| Variable | Default | Meaning |
|---|---|---|
| `SFUISE_WS` | `~/util_raw/sfuise_ws` | catkin workspace with the patched SFUISE |
| `SFUISE_BAGS` | `~/util_raw/kims_bags` | bags and configurations made by `sfuise_make_bag.py` |
| `SFUISE_LOGS` | `~/util_raw/sfuise_logs` | ROS logs of the SFUISE runs |
| `UTIL_ROOT` | `~/util_raw` | unpacked UTIL release (`dataset/`), the UTIL devkit clone (`util-uwb-dataset/`) and `pylib/` with `pyquaternion` |
| `MILUV_DEVKIT` | `~/miluv_devkit` | MILUV devkit clone |

## SFUISE

1. Clone SFUISE into the `src/` folder of a ROS Noetic catkin workspace. Then apply
   [`sfuise/sfuise_kims.patch`](sfuise/sfuise_kims.patch) inside the `sfuise` package:
   `patch -p1 < sfuise_kims.patch`.

   The patch adds three things and leaves the default behaviour unchanged:
   - optional per-tag lever arms (`tag_ids`, `tag_offsets`), because MILUV, NTU VIRAL and IULUI carry two to four tags;
   - an optional initial navigation-to-UWB transform (`t_nav_uwb_init`, `q_nav_uwb_init`), so that SFUISE starts
     from the same pose as KIMS;
   - guards against empty buffers, which otherwise crash on sequences whose ground truth starts late.
2. Build in Release mode (`-O3 -DNDEBUG`). Run with single-threaded BLAS: a Debug build with threaded BLAS fell
   behind real time and dropped data.
3. **MILUV, NTU VIRAL and IULUI (ranges).** Write the bag and the two configurations (0.3 m and 1.5 m residual
   gate) of each sequence:
   ```bash
   python3 baselines/sfuise/sfuise_make_bag.py miluv default_1_random3_0
   ```
   The configuration is the published ISAS-Walk one: weights, IMU variances, 10 Hz knots, a 100-knot (10 s)
   window, 50 calibration windows and a 0.3 m gate. The script fills in the dataset's rates, tags and anchors.
   Then run a job list with one `<dataset> <sequence>` per line:
   ```bash
   baselines/sfuise/sfuise_toa_batch.sh jobs.txt                 # published 0.3 m gate -> result/<ds>/sfuise_<seq>_pose.txt
   VARIANT=_g15 baselines/sfuise/sfuise_toa_batch.sh jobs.txt    # 1.5 m gate (sensitivity check only)
   ```
   The runs go one at a time with 1x playback, so nothing else should run on the machine. A run that crashes
   at start-up is retried up to three times.
4. **UTIL (TDoA).** The published UTIL configuration (`config_test_util.yaml`) is used for each flight:
   ```bash
   python3 baselines/sfuise/util_filter_bag.py <flight>       # drops physically impossible TDoA values (same rule as for KIMS)
   baselines/sfuise/util_sfuise.sh <flight> <ros port>      # -> result/util/sfuisetag_<flight>_pose.txt
   python3 baselines/sfuise/util_sfuise_body.py               # SFUISE reports the tag; moved to the body origin -> sfuise_<flight>_pose.txt
   ```

## MILUV devkit EKF (with and without the height sensor)

1. Clone <https://github.com/decargroup/miluv> (tested at commit `399ad46`) to `$MILUV_DEVKIT`.
2. Apply [`miluv_devkit/miluv_devkit.patch`](miluv_devkit/miluv_devkit.patch) there with `git apply`.
   The patch makes two changes:
   - adds `from __future__ import annotations`, which Python 3.8 needs;
   - restores the order of anchors 2 and 3 in constellation 2 (devkit issue #56).
3. Copy [`miluv_devkit/run_ekf_dump.py`](miluv_devkit/run_ekf_dump.py) into the devkit root, and link
   `$MILUV_DEVKIT/data` to `data/miluv` of this repository.
4. Run each experiment from the devkit root:
   ```bash
   python3 run_ekf_dump.py <experiment> results/ekf_<experiment>_h.csv             # IMU + UWB + height (MILUV paper, Table 10; figures)
   python3 run_ekf_dump.py <experiment> results/ekf_<experiment>_nh.csv noheight   # IMU + UWB only (reference)
   ```

The filter starts from the ground-truth state, as in the devkit. The `pos_rmse` that the devkit prints (and the MILUV
paper reports) is a per-axis RMS, so it equals the 3D RMSE divided by √3. With the height sensor, our runs match the per-axis values of
Table 10 within 0.2 mm on 10 experiments and within 5.3 mm on `remoteControlHighPace_1`; the two constellation-2 flights,
which we run with the corrected anchor order, differ by 4.2 and 3.2 mm. The dump also holds the estimated and ground-truth attitude (quaternion columns) for the orientation scores.

## UTIL devkit ESKF

The ESKF is the unmodified `eskf_class.py` of the UTIL devkit
(<https://github.com/learnsyslab/util-uwb-dataset>), driven the same way as the devkit's `main.py`:

```bash
python3 baselines/util_devkit/util_eskf.py <flight>    # -> result/util/eskf_<flight>_pose.txt
```

## SFUISE + Tukey (Table III)

[`sfuise/sfuise_tukey.patch`](sfuise/sfuise_tukey.patch), applied on top of `sfuise_kims.patch`, adds an optional
Tukey loss on the range residuals in place of the residual gate. Add `robust_delta: <cut in m>` to the configuration
(the KIMS cut times the range sigma of Table II: MILUV 5 x 0.081, NTU VIRAL 5 x 0.3, UTIL 5 x 0.224, IULUI 3 x 0.11 m) and
set `if_reject_uwb: false`. Without `robust_delta` SFUISE
behaves as before.
