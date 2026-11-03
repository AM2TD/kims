#!/bin/bash
cd "$(dirname "$0")/../.."; R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1 lm:=1"
S="default_3_random3_1 default_3_zigzag_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
n=0; for s in $S; do $R/hp_mix.sh lm_$s $s $A T:=229 > /dev/null 2>&1 & n=$((n+1)); [ $((n%7)) = 0 ] && wait; done; wait; echo LM_MILUV_DONE
