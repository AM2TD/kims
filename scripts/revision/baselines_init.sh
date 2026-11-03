#!/bin/bash
# In-house baselines with the start pose KIMS uses: NTU = rest-phase fit position
# and heading (init_x/y/z, init_yaw) with the tilt from the first 3 s of IMU (align_still:=3); MILUV = origin, heading 0,
# tilt from the first 3 s. Same estimator settings as before (EKF-RTS defaults, SW-PGO window 15, particle smoother defaults).
# Outputs result/baselines/{ekfi,swpgoi,psi}_<seq>_pose.txt; logs result/runlogs/<label>.log. usage: baselines_init.sh [parallel]
cd "$(dirname "$0")/../.."; P=${1:-6}
J=result/runlogs/baselines_init_jobs.txt; : > $J
ntu_init() { python3 - "$1" <<'PY'
import sys, json
s = sys.argv[1]
if s.startswith("eee"): print(json.load(open("result/runlogs/ntu_init_args.json"))[s])
else: print([x for x in open(f"result/runlogs/ntu_new_{s}.log") if " init:" in x][0].split("init:")[1].strip())
PY
}
cmd() {   # dataset method seq label extra
  A=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('$1','$2','$3','result/baselines/$4_$3_pose.txt')))")
  echo "${KIMS_BUILD:-build}/bin/$2 $A $5 > result/runlogs/$4_$3.log 2>&1" >> $J
}
for s in eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do
  I="$(ntu_init $s) align_still:=3"; [ "${s%%_*}" = eee ] && AF="anchor_file:=$PWD/data/ntu/anchors_eee.txt" || AF="anchor_file:=$PWD/data/ntu/anchors_$s.txt"
  cmd ntu ekf_rts $s ekfi "$I $AF"; cmd ntu sw_pgo $s swpgoi "window_size:=15 $I $AF"; cmd ntu particle_smoothing $s psi "$I $AF"
done
for s in $(grep '^miluv' result/runlogs/final_table.txt | awk '{print $2}'); do
  cmd miluv sw_pgo $s swpgoi "window_size:=15 align_still:=3"; cmd miluv particle_smoothing $s psi "align_still:=3"
done
mkdir -p result/baselines; wc -l < $J
OMP_NUM_THREADS=1 xargs -P $P -I{} bash -c "{}" < $J
echo BASELINES_INIT_DONE
