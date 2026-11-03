#!/bin/bash
# Overnight batch (2026-10-07/08), run in order:
#  P1  NTU sbs_02 / nya_03: seeds at N 8192 and the initial-heading check (original still-fit heading vs heading 0)
#  P2  tempering trigger 0.0244 -> 0.0161 (= 4V/N, V 33): standalone timing pairs, then accuracy on all 25 sequences
#  P3  LM seed repeats on MILUV random3_0 / obstacles_0b (+ sampling seeds), NTU LM rerun with the original initial poses
# A phase whose run diverges (leaves a 60 m box) is stopped as a whole (feedback-stop-on-divergence) and the next phase starts.
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1
R=scripts/revision; L=result/runlogs; LOG=$L/overnight.log
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
NT="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
A_ANC="anchor_file:=$PWD/data/ntu/anchors"
ntu_init() { case $1 in
  eee_*)  python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])";;
  nya_01) echo "init_x:=6.468 init_y:=-2.934 init_z:=0.2 init_yaw:=195.0 ${A_ANC}_nya_01.txt";;
  nya_02) echo "init_x:=7.302 init_y:=-2.005 init_z:=0.2 init_yaw:=78.0 ${A_ANC}_nya_02.txt";;
  nya_03) echo "init_x:=6.805 init_y:=-3.815 init_z:=0.2 init_yaw:=294.0 ${A_ANC}_nya_03.txt";;
  sbs_01) echo "init_x:=19.312 init_y:=-9.862 init_z:=0.2 init_yaw:=90.0 ${A_ANC}_sbs_01.txt";;
  sbs_02) echo "init_x:=16.039 init_y:=-12.987 init_z:=0.2 init_yaw:=116.0 ${A_ANC}_sbs_02.txt";;
  sbs_03) echo "init_x:=13.874 init_y:=-8.847 init_z:=0.2 init_yaw:=111.0 ${A_ANC}_sbs_03.txt";; esac; }
ntu_init_noyaw() { ntu_init $1 | sed 's/ init_yaw:=[^ ]*//'; }
mil() { echo "$R/hp_mix.sh $1 $2 $A T:=229 $3 > /dev/null 2>&1"; }
iul() { echo "$R/iu_run.sh $1 $2 T:=200 head_filt:=1 still_bias:=1 $3 > /dev/null 2>&1"; }
ntu() { echo "$R/ntu_run.sh $1 $2 T:=118 $3 $NT $4 > /dev/null 2>&1"; }
nrun() { pgrep -fc '[b]uild_fp32/bin/'; }
STOP=0
check_div() { # $1 label file
  for lab in $(cat $1); do f=$(ls result/*/diag/${lab}_pose.txt 2>/dev/null | head -1); [ -n "$f" ] || continue
    if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 3600) bad=1 } END { exit !bad }'; then
      echo "$(date +%T) DIVERGED $lab -> stopping phase" >> $LOG; STOP=1
      for l2 in $(cat $1); do pkill -f "diag/${l2}_pose[.]txt"; done; return; fi
  done; }
run_phase() { # $1 job file (one command per line, label = 2nd field), $2 max parallel
  local lf=$1.labels; awk '{print $2}' $1 > $lf; STOP=0
  while read -r job; do
    [ $STOP = 1 ] && break
    while [ "$(nrun)" -ge $2 ]; do sleep 15; check_div $lf; [ $STOP = 1 ] && break; done
    [ $STOP = 1 ] && break
    bash -c "$job" & sleep 3
  done < $1
  while [ "$(nrun)" -gt 0 ]; do sleep 20; check_div $lf; done; wait; }
standalone() { # one run at a time, GPU otherwise idle
  while [ "$(nrun)" -gt 0 ]; do sleep 10; done; bash -c "$1"; }
J=$L/overnight_jobs; mkdir -p $J
echo "$(date +%T) START" > $LOG

# ---------------- P1 ----------------
: > $J/p1.txt
for sd in 1 2 3 4; do ntu k8s${sd}_s02 sbs_02 "$(ntu_init sbs_02)" "seed:=$sd" >> $J/p1.txt; done
ntu k8ny_s02 sbs_02 "$(ntu_init_noyaw sbs_02)" "" >> $J/p1.txt
ntu k8ny_y03 nya_03 "$(ntu_init_noyaw nya_03)" "" >> $J/p1.txt
for sd in 1 2; do ntu k8s${sd}_y03 nya_03 "$(ntu_init nya_03)" "seed:=$sd" >> $J/p1.txt; done
ntu k4y_s02 sbs_02 "$(ntu_init sbs_02)" "N:=4096" >> $J/p1.txt
ntu k2y_s02 sbs_02 "$(ntu_init sbs_02)" "N:=2048" >> $J/p1.txt
ntu lmy_s02 sbs_02 "$(ntu_init sbs_02)" "lm:=1" >> $J/p1.txt
for sd in 1 2; do ntu lmys${sd}_s02 sbs_02 "$(ntu_init sbs_02)" "lm:=1 seed:=$sd" >> $J/p1.txt; done
run_phase $J/p1.txt 15
python3 $R/overnight_tables.py p1 > $L/overnight_p1_table.txt 2>&1
echo "$(date +%T) P1_DONE" >> $LOG

# ---------------- P2 ----------------
TQ="ess_min:=0.0161"
standalone "$(mil tqs_hp default_1_remoteControlHighPace_1 "$TQ")"
standalone "$(mil tqs_r31 default_1_random3_1 "$TQ")"
standalone "$(mil tms_r31 default_1_random3_1 "")"
standalone "$(iul tqs_SquareTurn SquareTurn "$TQ")"
standalone "$(ntu tqs_e01 eee_01 "$(ntu_init eee_01)" "$TQ")"
standalone "$(ntu tms_e01 eee_01 "$(ntu_init eee_01)" "")"
python3 $R/overnight_tables.py p2t > $L/overnight_p2_timing.txt 2>&1
echo "$(date +%T) P2_TIMING_DONE" >> $LOG
: > $J/p2.txt
for s in default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b; do mil tq_$s $s "$TQ" >> $J/p2.txt; done
for s in SquareTurn CircleSpin SquareBob; do iul tq_$s $s "$TQ" >> $J/p2.txt; done
for s in eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do ntu tq_$s $s "$(ntu_init $s)" "$TQ" >> $J/p2.txt; done
run_phase $J/p2.txt 15
python3 $R/overnight_tables.py p2a > $L/overnight_p2_accuracy.txt 2>&1
echo "$(date +%T) P2_DONE" >> $LOG

# ---------------- P3 ----------------
: > $J/p3.txt
for e in default_1_random3_0:r30 obstacles_1_random3_0b:obs; do s=${e%%:*}; sh=${e##*:}
  mil lmk_$sh $s "lm:=1" >> $J/p3.txt
  for sd in 1 2; do mil lmks${sd}_$sh $s "lm:=1 seed:=$sd" >> $J/p3.txt; mil ks${sd}_$sh $s "seed:=$sd" >> $J/p3.txt; done
done
for s in nya_01 nya_02 nya_03 sbs_01 sbs_03; do ntu lmy_$s $s "$(ntu_init $s)" "lm:=1" >> $J/p3.txt; done
run_phase $J/p3.txt 15
python3 $R/overnight_tables.py p3 > $L/overnight_p3_table.txt 2>&1
echo "$(date +%T) P3_DONE" >> $LOG
