#!/bin/bash
# Train/test check of the MILUV range mixture: the default mixture was fitted on the three constellation-0 flights
# random3_0, circular3D_0, LowPace_0_v2, which are also evaluated. Rerun those three with a mixture fitted on the
# other ten evaluated MILUV sequences (labels cv_<seq>), final configuration otherwise.
cd "$(dirname "$0")/../.."; R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
MIX="$1"
for s in default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2; do
  $R/hp_mix.sh cv_$s $s $A T:=229 $MIX > /dev/null 2>&1 &
done
wait; echo MIX_HOLDOUT_DONE
