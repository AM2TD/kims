# Orientation RMSE (deg)

Rotation angle of R_gt^T R_est against motion capture or Vicon. NTU VIRAL has no attitude ground truth (Leica prism position only).
**raw** = no alignment (as for position). **aligned** = one constant world-frame rotation per run removed, for every method alike;
SFUISE needs it because it reports attitude in its own navigation frame, which is levelled by its estimated gravity vector, so its raw
attitude carries a constant 4-18 deg tilt (KIMS 0.1-1 deg). KIMS and SFUISE start at yaw 0; the MILUV filter (devkit EKF + height,
our run of the MILUV paper's Table 10 filter, which reports position only) starts from the ground-truth attitude. GT start yaw in
brackets: on remoteControlHighPace_2 (-103 deg) and SquareTurn (-60 deg) KIMS needs about 15 s to turn its heading, which
dominates its error there (from 20 s on: 5.8 and 3.3 deg). KIMS: committed lag T-1 output (the window flush has no attitude).

## MILUV (13)

| Sequence (GT start yaw) | KIMS raw | KIMS aligned | SFUISE raw | SFUISE aligned | EKF + height raw | EKF + height aligned |
|---|---|---|---|---|---|---|
| default_1_remoteControlHighPace_1 (+1) | 7.0 | 5.1 | diverged | diverged | 23.5 | 21.8 |
| default_1_random3_1 (+4) | 7.0 | 5.8 | 7.7 | 5.7 | 11.5 | 8.2 |
| default_1_random3_0 (-9) | 8.3 | 6.7 | 11.4 | 7.4 | 8.8 | 8.5 |
| default_1_circular3D_0 (-8) | 8.7 | 6.2 | 10.5 | 6.6 | 11.9 | 9.2 |
| default_1_remoteControlLowPace_0_v2 (+5) | 6.7 | 5.1 | 19.2 | 5.7 | 8.2 | 8.1 |
| default_1_remoteControlLowPace_0 (+3) | 6.9 | 5.9 | 20.1 | 12.7 | 10.3 | 9.8 |
| default_1_circular2D_0 (+1) | 6.0 | 4.2 | 26.2 | 25.1 | 8.7 | 6.3 |
| default_1_remoteControlHighPace_0 (-3) | 4.3 | 3.7 | diverged | diverged | 7.3 | 7.0 |
| obstacles_1_random3_0b (-32) | 5.5 | 5.1 | 8.5 | 5.1 | 7.7 | 6.5 |
| default_1_random3_2 (-52) | 7.3 | 6.3 | diverged | diverged | 8.3 | 7.6 |
| default_1_remoteControlHighPace_2 (-103) | 19.3 | 19.1 | 13.1 | 4.1 | 13.8 | 13.5 |
| noAprilTags_1_random3_2b (+5) | 6.2 | 4.5 | 10.5 | 5.6 | 8.5 | 8.1 |
| noAprilTags_1_remoteControlHighPace_2b (-1) | 6.9 | 4.9 | 7.5 | 5.7 | 8.3 | 7.5 |
| **median** | **6.9** | **5.1** | 10.9 | 5.7 | 8.7 | 8.1 |

- KIMS lower than SFUISE: raw 9/10, aligned 7/10 (3 diverged runs left out)
- KIMS lower than EKF + height: raw 12/13, aligned 12/13

## IULUI (3)

| Sequence (GT start yaw) | KIMS raw | KIMS aligned | SFUISE raw | SFUISE aligned |
|---|---|---|---|---|
| SquareTurn (-60) | 9.7 | 9.6 | 8.0 | 6.2 |
| CircleSpin (+4) | 4.0 | 2.3 | 14.3 | 8.1 |
| AC3_125000 (+0) | 3.4 | 2.3 | 18.8 | 3.6 |
| **median** | **4.0** | **2.3** | 14.3 | 6.2 |

- KIMS lower than SFUISE: raw 2/3, aligned 2/3

## UTIL (79, one UWB tag)

With one tag the ranges carry no direct heading information; tilt is fixed by gravity, heading only weakly through the IMU
(horizontal acceleration). The devkit ESKF keeps most of its start heading (its mean heading error follows the GT start yaw,
r = -0.79); KIMS removes the start offset (r = 0.01) but its heading wanders on 5 of 79 flights (36-102 deg RMS:
const2-trial3-tdoa3, const4-trial2-tdoa2-traj2, const4-trial2-tdoa3-traj2/traj3, const4-trial3-tdoa2-traj1) while its position
there stays at or below the group median. The UTIL paper reports position only; ESKF here = our run of the devkit ESKF.

| Group (flights) | KIMS raw | KIMS aligned | SFUISE raw | SFUISE aligned | ESKF raw | ESKF aligned |
|---|---|---|---|---|---|---|
| const1 tdoa2 (6) | 3.1 | 0.7 | 9.9 | 4.3 | 4.5 | 0.9 |
| const1 tdoa3 (6) | 3.3 | 0.6 | 16.9 | 3.4 | 3.0 | 1.2 |
| const2 tdoa2 (6) | 2.2 | 0.4 | 14.4 | 3.7 | 2.9 | 0.8 |
| const2 tdoa3 (6) | 2.3 | 0.5 | 13.4 | 6.8 | 3.5 | 0.9 |
| const3 tdoa2 (8) | 4.0 | 0.8 | 14.5 | 7.3 | 2.5 | 1.1 |
| const3 tdoa3 (8) | 3.4 | 0.6 | 14.9 | 4.1 | 3.0 | 1.0 |
| const4 tdoa2 (21) | 3.1 | 0.8 | 11.8 | 9.0 | 3.3 | 1.4 |
| const4 tdoa3 (18) | 3.1 | 0.8 | 13.6 | 6.3 | 3.4 | 1.3 |
| **all** (79) | 3.1 | 0.7 | 13.4 | 6.0 | 3.3 | 1.1 |

- KIMS lower than SFUISE: raw 67/74, aligned 69/74 (5 diverged runs left out)

- KIMS lower than ESKF (devkit): raw 48/79, aligned 68/79

## Summary over runs

Every value above is an RMSE over time: sqrt(mean of theta_k^2), theta_k = rotation angle of R_gt^T R_est
(the length of the rotation-error vector, so the per-axis RMSEs of that vector combine as for 3D position). Tables give the
median over runs, as the position tables do; mean and time-weighted pooled RMSE for comparison (diverged runs left out):

| Dataset | Method | median | mean | pooled |
|---|---|---|---|---|
| MILUV | KIMS | 6.9 | 7.7 | 8.1 |
| MILUV | SFUISE | 10.9 | 13.5 | 14.6 |
| MILUV | EKF + height | 8.7 | 10.5 | 11.1 |
| IULUI | KIMS | 4.0 | 5.7 | 6.3 |
| IULUI | SFUISE | 14.3 | 13.7 | 14.5 |
| UTIL | KIMS | 3.1 | 7.4 | 18.3 |
| UTIL | SFUISE | 13.4 | 17.8 | 26.0 |
| UTIL | ESKF (devkit) | 3.3 | 3.5 | 3.8 |

On UTIL the median hides the five KIMS heading-drift flights; the mean and pooled values show them.
