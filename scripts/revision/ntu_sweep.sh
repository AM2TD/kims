#!/bin/bash
# NTU window sweep of the configuration that holds on NTU: still-phase IMU bias
# subtraction, range std 0.3 with the 1.5 m Tukey cut, head filter off; per-sequence initial position from
# result/runlogs/ntu_init_args.json (heading is observable at rest from the tag array, init_yaw is irrelevant). 12 runs.
cd "$(dirname "$0")/../.."
R=scripts/revision
A="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
for T in 89 118 177 236; do for s in eee_01 eee_02 eee_03; do
  I=$(python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$s'])")
  $R/ntu_run.sh w${T}_$s $s T:=$T $I $A > /dev/null 2>&1 &
done; done
wait
echo NTU_SWEEP_DONE
