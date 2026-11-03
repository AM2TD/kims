# KIMS vs baselines: 3D position RMSE (m)

KIMS uses IMU + UWB only (final configuration, fixed-lag output, seed 1234). Lower is better; **bold** = best per row.

## Summary (median over sequences)

| Dataset (sequences) | KIMS | SFUISE | Dataset filter(s), as published | KIMS lower than |
|---|---|---|---|---|
| MILUV (13) | **0.135** | 0.704 (3 diverged) | EKF + height 0.258 | SFUISE 13/13; EKF + height 13/13 |
| NTU VIRAL (9) | **0.435** | 2.537 (2 diverged) | none | SFUISE 6/9 |
| UTIL (79) | 0.187 | **0.184** (5 diverged) | ESKF 0.220 | SFUISE 33/79; ESKF 78/79 |
| IULUI (3) | **0.120** | 0.235 | none | SFUISE 2/3 |

Diverged = RMSE above 100 m (counted in the medians). KIMS diverges on none of the 104 sequences.

## MILUV (raw mocap ground truth, whole span, no alignment)

| Experiment | KIMS | SFUISE | EKF + height (MILUV paper Table 10 x sqrt3) |
|---|---|---|---|
| default_1_remoteControlHighPace_1 | **0.205** | diverged | 1.328 |
| default_1_random3_1 | **0.231** | 0.488 | 0.255 |
| default_1_random3_0 | **0.165** | 0.704 | 0.217 |
| default_1_circular3D_0 | **0.093** | 0.123 | 0.303 |
| default_1_remoteControlLowPace_0_v2 | **0.086** | 0.114 | 0.219 |
| default_1_remoteControlLowPace_0 | **0.202** | 0.955 | 0.247 |
| default_1_circular2D_0 | **0.134** | 0.192 | 0.335 |
| default_1_remoteControlHighPace_0 | **0.100** | diverged | 1.359 |
| obstacles_1_random3_0b | **0.130** | 0.639 | 0.252 |
| default_1_random3_2 | **0.135** | diverged | 0.258 |
| default_1_remoteControlHighPace_2 | **0.244** | 0.763 | 1.409 |
| noAprilTags_1_random3_2b | **0.130** | 0.368 | 0.241 |
| noAprilTags_1_remoteControlHighPace_2b | **0.191** | 1.144 | 0.273 |
| **median** | **0.135** | 0.704 | 0.258 |

## NTU VIRAL (Leica prism, 4-DoF alignment, start+10 s to end-10 s)

| Sequence | KIMS | SFUISE |
|---|---|---|
| eee_01 | 0.204 | **0.126** |
| eee_02 | **0.367** | 2.537 |
| eee_03 | 1.845 | **1.144** |
| nya_01 | **0.791** | diverged |
| nya_02 | 3.084 | **0.395** |
| nya_03 | **0.903** | diverged |
| sbs_01 | **0.190** | 0.207 |
| sbs_02 | **0.409** | 3.109 |
| sbs_03 | **0.435** | 3.146 |
| **median** | **0.435** | 2.537 |

## UTIL (Vicon, whole span, no alignment): medians per constellation and TDOA mode

| Group (flights) | KIMS | SFUISE | ESKF (UTIL paper Table 4) |
|---|---|---|---|
| const1 tdoa2 (6) | 0.104 | **0.101** | 0.113 |
| const1 tdoa3 (6) | 0.104 | **0.099** | 0.140 |
| const2 tdoa2 (6) | 0.103 | **0.102** | 0.107 |
| const2 tdoa3 (6) | 0.092 | **0.088** | 0.117 |
| const3 tdoa2 (8) | 0.200 | **0.179** | 0.217 |
| const3 tdoa3 (8) | **0.179** | 0.183 | 0.229 |
| const4 tdoa2 (21) | 0.286 | **0.240** | 0.473 |
| const4 tdoa3 (18) | 0.250 | **0.218** | 0.501 |
| **all (79)** | 0.187 | **0.184** | 0.220 |

## IULUI (Qualisys, whole span, no alignment)

| Flight | KIMS | SFUISE |
|---|---|---|
| SquareTurn | 0.120 | **0.106** |
| CircleSpin | **0.118** | 0.235 |
| AC3_125000 | **0.128** | 0.300 |
| **median** | **0.120** | 0.235 |

## Published results on UTIL subsets (as reported; tdoa2 flights)

He & Chan, IEEE TVT 2026 (training-free RAF-ESKF / AF-ESKF; LAF-ESKF is trained on ground truth):

| Flight | He & Chan | KIMS |
|---|---|---|
| const4-trial3-tdoa2-traj1 | **0.29** RAF-ESKF | 0.292 |
| const4-trial3-tdoa2-traj2 | **0.294** RAF-ESKF | 0.299 |
| const4-trial3-tdoa2-traj3 | 0.308 RAF-ESKF | **0.307** |
| const4-trial5-tdoa2-traj1 | **0.303** RAF-ESKF | 0.380 |
| const4-trial5-tdoa2-traj2 | **0.246** RAF-ESKF | 0.262 |
| const4-trial5-tdoa2-traj3 | **0.243** RAF-ESKF | 0.286 |
| const2-trial1-tdoa2 | 0.0995 AF-ESKF (LAF-ESKF, trained: 0.0445) | **0.095** |
| const2-trial4-tdoa2 | 0.0918 AF-ESKF (LAF-ESKF, trained: 0.0473) | **0.088** |

Li et al., Measurement 2026 (STG-AKF, learned from ground-truth bias labels on the other const4 scenarios / trajectories; its ESKF baseline is the UTIL Table 4 ESKF):

| const4 tdoa2 flights | STG-AKF cross-scenario | STG-AKF cross-trajectory | KIMS | ESKF (UTIL paper Table 4) |
|---|---|---|---|---|
| scenarios 1-6 = trials 1-6 (18), mean | **0.184** | 0.230 | 0.318 | 0.480 |
| scenario 7 = trial 7 manual, held out (3), mean | 0.314 | - | **0.207** | 0.483 |

## Conditions

- KIMS: IMU + UWB; fixed-lag output (3 s MILUV, 2 s others); FP32 rollout, 8 host threads, N 8192.
- SFUISE: the authors' published configuration (0.3 m gate, 10 s window), same start pose as KIMS; our runs.
- MILUV filters: the MILUV paper's own EKF with the laser height sensor (its Table 10), started from the ground-truth state; the paper's per-axis RMS multiplied by sqrt(3). Constellation-2 values are as published (devkit anchor order); KIMS and SFUISE use the corrected order.
- UTIL filter: the UTIL paper's ESKF (Table 4, per flight, as published). The figures need per-epoch errors, which the paper does not give, so they use our run of the released devkit ESKF; it matches Table 4 within 0.01 m on 45/79 flights and is 0.03-0.31 m lower (better) on the 27 const4 trial 3-7 flights.
- References (not in the table): devkit EKF IMU+UWB without height (our run) median 0.648, KIMS lower 13/13 (MILUV); devkit ESKF, our run of the released code, median 0.217, KIMS lower 78/79 (used in the figures) (UTIL).
- Per-flight UTIL values: result/final/util_table.txt.
