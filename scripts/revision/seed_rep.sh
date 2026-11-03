#!/bin/bash
# Seed repetition of the final configuration on the 8 main sequences.
# Reuses seed-1234 runs (sb_*, w118_*) and existing seed 1/2 runs where the configuration is identical.
cd "$(dirname "$0")/../.."
R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
NT="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
init() { python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])"; }
# wave 1
for s in 3 4; do $R/hp_mix.sh s${s}_hp default_1_remoteControlHighPace_1 $A T:=229 seed:=$s > /dev/null 2>&1 & done
for s in 1 2 3 4; do $R/hp_mix.sh s${s}_r3 default_1_random3_1 $A T:=229 seed:=$s > /dev/null 2>&1 & done
for s in 1 2 3 4; do $R/iu_run.sh s${s}_SquareBob SquareBob T:=200 head_filt:=1 still_bias:=1 seed:=$s > /dev/null 2>&1 & done
for s in 1 2 3 4; do $R/iu_run.sh s${s}_CircleSpin CircleSpin T:=200 head_filt:=1 still_bias:=1 seed:=$s > /dev/null 2>&1 & done
$R/iu_run.sh s3_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 seed:=3 > /dev/null 2>&1 &
wait
# wave 2
$R/iu_run.sh s4_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 seed:=4 > /dev/null 2>&1 &
for s in 3 4; do $R/ntu_run.sh s${s}_eee_01 eee_01 T:=118 $(init eee_01) $NT seed:=$s > /dev/null 2>&1 & done
for s in 1 2 3 4; do $R/ntu_run.sh s${s}_eee_02 eee_02 T:=118 $(init eee_02) $NT seed:=$s > /dev/null 2>&1 & done
for s in 1 2 3 4; do $R/ntu_run.sh s${s}_eee_03 eee_03 T:=118 $(init eee_03) $NT seed:=$s > /dev/null 2>&1 & done
wait
echo SEED_REP_DONE
