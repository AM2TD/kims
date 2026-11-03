# Reference results of the revised paper

All values are 3D position RMSE in metres unless stated otherwise. KIMS uses the final configuration: FP32 rollout,
8 host threads, `N = 8192` and seed 1234. The main README describes the scoring.

Table I of the paper reports, on MILUV, NTU VIRAL and IULUI, the mean over ten seeds (`figs/fig_seeds_final.txt`),
which differs from the seed-1234 values below only on `sbs_02` (0.559 against 0.409) and `nya_03` (0.880 against
0.903); every other sequence agrees within 2 mm. `reproduce_n2048.txt` is the output of `scripts/reproduce.sh` at the
paper's operating point `N = 2048` next to the `N = 8192` reference and Table I.

| File | Content |
|---|---|
| `baselines_rmse.md` | the final baseline comparison: summary, per-sequence tables for every dataset, published UTIL subsets (He & Chan, STG-AKF) and the conditions |
| `final_rmse_final.txt` | the comparison table: KIMS (seed 1234, `N = 8192`), SFUISE and the dataset filters per sequence (MILUV, NTU VIRAL, IULUI) and per UTIL group, with medians, divergence counts and win counts. The filter columns here are the published values (MILUV Table 10 x sqrt(3), UTIL Table 4); Table I of the paper uses our runs of the released devkit filters (MILUV median 0.265, UTIL 0.217), listed in the `reference:` lines |
| `reproduce_n2048.txt` | `scripts/reproduce.sh` at `N = 2048` against the `N = 8192` reference and Table I |
| `orientation_rmse.md`, `orient_final.txt` | orientation RMSE (deg) on MILUV and IULUI for KIMS, SFUISE and the MILUV filter, raw and after removing one constant rotation per run (scripts/revision/orient_final.py) |
| `sfuise_window.txt` | SFUISE with its published 10 s window vs the same configuration with the window matched to KIMS (3 s MILUV, 2 s elsewhere; 33 sequences, all diverge), and with the matched window plus a 1.5 m gate on six flights (scripts/revision/sfuise_window_table.py) |
| `imu_dead_reckoning.txt` | IMU-only drift after 1-10 s from ground-truth starts every second (same IMU stream and bias correction as KIMS), per dataset and per sequence (scripts/revision/imu_dead_reckoning.py) |
| `anchor_table.txt` | per sequence: anchors, anchor height spread, tags, measurement type, UWB events per second, duration (scripts/revision/anchor_table.py) |
| `lm_ablation.txt` | sampling (KIMS) vs the matched LM (same objective, mode + Laplace) on all 104 sequences: per sequence (MILUV, NTU VIRAL, IULUI) and per dataset (scripts/revision/lm_ablation_table.py) |
| `final_table_v2.txt` | MILUV and NTU VIRAL per sequence, including the matched LM and the first-submission baselines (SW-PGO, PS; NTU: fixed-lag ESKF) |
| `util_table.txt` | all 79 UTIL flights: 3D, xy and z for KIMS, the matched LM, SFUISE (body origin) and our devkit ESKF run (used in the figures), and the UTIL paper's Table 4 ESKF (3D, as published) |
| `timing_table.txt` | per-update processing time of every method, each run alone on the machine |
| `lag0_latency.txt` | accuracy of the newest-epoch (lag-0) output and a FIFO replay of the measured solve times |
| `figs/final_xyz_<dataset>_<page>.png` | x, y and z over time for KIMS, SFUISE and the dataset filter against the ground truth |
| `figs/fig_traj3d.*` | 3D overlay of ground truth, KIMS, SFUISE, the UWB anchors and the best-fit anchor plane, one flight per dataset (scripts/revision/fig_traj3d_final.py) |
| `figs/fig_strip_final.png`, `figs/fig_landscape_final.png` | sampling diagnostics on the IULUI SquareTurn plane crossing with the final configuration: per-epoch posterior of the window-start z for KIMS and the Laplace cloud of the matched LM (fig_strip.py), and the (dp_z, dv_z) cost landscape with the posterior samples and LM iterates at epoch 4720 (fig_landscape.py) |
| `figs/fig_traj.*` | x, y and z over time for one representative sequence per dataset (the paper's trajectory figure) |
| `figs/fig_boxplot_final.*` | per-epoch error distributions per dataset |
| `figs/fig_zcross_final.*` | altitude on the coplanar-anchor flights, with the mirror image of the ground truth |
| `figs/fig_nsweep_final.*` | accuracy (33 sequences) and per-epoch solve time (one flight per dataset, run alone) against the sample count N |
| `figs/fig_seeds_final.*` | 3D RMSE over seeds 1-10 and the table seed on the 33 sequences, and the time on the mirror side on the coplanar flights |

MILUV constellation 2 (`default_1_random3_2`, `default_1_remoteControlHighPace_2`) is scored with anchors 2 and 3 in
the corrected order (devkit issue #56) for KIMS, SFUISE, the devkit EKF and the matched LM. The SW-PGO and PS entries
of these two rows in `final_table_v2.txt` (first-submission baselines) were not re-run with that order.
