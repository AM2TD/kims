#!/usr/bin/env python3
"""Tables for scripts/revision/overnight.sh. usage: overnight_tables.py p1|p2t|p2a|p3
Scoring as final_table.py: MILUV/IULUI whole GT span (+ window flush), NTU 4-DoF aligned prism, window gt+10..end-10."""
import sys, os, numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/revision/final_table.py').read().split('MIL = [')[0])
LAG = {'miluv': 228, 'iului': 199, 'ntu': 117}
EPOCH = {'miluv': 13.0, 'iului': 10.0, 'ntu': 17.0}
def f_of(ds, lab): return f'result/{ds}/diag/{lab}_pose.txt'
def rm(ds, seq, lab):
    f = f_of(ds, lab)
    if not os.path.exists(f): return np.nan
    if not os.path.exists(f.replace('_pose.txt', '_timing.txt')): return np.nan   # unfinished
    return sc(ds, seq, f, LAG[ds])
def bias(ds, lab):
    f = f'result/{ds}/diag/{lab}_bias.txt'
    if not os.path.exists(f): return None
    B = np.array([[float(x) for x in l.split()[:16]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 16])
    return B[20:] if len(B) > 40 else None
def fmt(v, w=7): return ('%*.3f' % (w, v)) if np.isfinite(v) else ' ' * (w - 1) + '-'
def sd(vals):
    v = [x for x in vals if np.isfinite(x)]; return np.std(v, ddof=1) if len(v) > 1 else np.nan
ph = sys.argv[1]
if ph == 'p1':
    print("P1. NTU sbs_02 / nya_03 at N 8192: seeds, initial heading (still-fit heading vs 0), N 4096/2048 with the still-fit heading, LM with the still-fit heading.")
    print("3D rmse [m], 4-DoF aligned prism. 'yaw' = still-fit initial heading (as the original n_* runs); 'no-yaw' = heading 0 (as the earlier LM, floor and N-sweep runs).")
    P1ROWS = [('sbs_02 sampling N8192 yaw, seed 1234 (n_sbs_02)', 'sbs_02', ['n_sbs_02']),
            ('sbs_02 sampling N8192 yaw, seeds 1-4', 'sbs_02', [f'k8s{s}_s02' for s in (1, 2, 3, 4)]),
            ('sbs_02 sampling N8192 no-yaw', 'sbs_02', ['k8ny_s02']),
            ('sbs_02 sampling N8192 no-yaw + floor (fl_sbs_02)', 'sbs_02', ['fl_sbs_02']),
            ('sbs_02 sampling N4096 yaw', 'sbs_02', ['k4y_s02']), ('sbs_02 sampling N4096 no-yaw (sweep)', 'sbs_02', ['n4096_s02']),
            ('sbs_02 sampling N2048 yaw', 'sbs_02', ['k2y_s02']), ('sbs_02 sampling N2048 no-yaw (sweep, seeds 1234/1/2)', 'sbs_02', ['n2048_s02', 'n2048s1_s02', 'n2048s2_s02']),
            ('sbs_02 LM yaw, seeds 1234/1/2', 'sbs_02', ['lmy_s02', 'lmys1_s02', 'lmys2_s02']), ('sbs_02 LM no-yaw (lm_sbs_02)', 'sbs_02', ['lm_sbs_02']),
            ('nya_03 sampling N8192 yaw, seeds 1234/1/2', 'nya_03', ['n_nya_03', 'k8s1_y03', 'k8s2_y03']), ('nya_03 sampling N8192 no-yaw', 'nya_03', ['k8ny_y03']),
            ('nya_03 sampling N4096 / 2048 no-yaw (sweep)', 'nya_03', ['n4096_y03', 'n2048_y03']), ('nya_03 LM no-yaw (lm_nya_03)', 'nya_03', ['lm_nya_03'])]
    for name, seq, labs in P1ROWS:
        v = [rm('ntu', seq, l) for l in labs]
        print(f"{name:58s} | " + ' '.join(fmt(x) for x in v) + (f"   (sd {sd(v):.3f})" if len(v) > 1 and np.isfinite(sd(v)) else ''))
elif ph == 'p2t':
    print("P2a. Standalone timing: tempering trigger ess_min 0.0244 (ESS < 200 at N 8192) vs 0.0161 (ESS < 132 = 4V). One run on the GPU, first 20 solves excluded.")
    print(f"{'sequence':22s} {'trigger':>7s} | {'mean':>6s} {'median':>6s} {'p95':>6s} {'>epoch':>7s} | {'temper%':>7s} {'passes':>6s} | {'rmse':>6s}")
    for ds, seq, old, new in [('miluv', 'default_1_remoteControlHighPace_1', 'tm_hp', 'tqs_hp'), ('miluv', 'default_1_random3_1', 'tms_r31', 'tqs_r31'),
                              ('iului', 'SquareTurn', 'tm_SquareTurn', 'tqs_SquareTurn'), ('ntu', 'eee_01', 'tms_e01', 'tqs_e01')]:
        for trig, lab in (('0.0244', old), ('0.0161', new)):
            tf = f'result/{ds}/diag/{lab}_timing.txt'; B = bias(ds, lab)
            if not os.path.exists(tf) or B is None: print(f"{seq[:22]:22s} {trig:>7s} | missing"); continue
            t = np.loadtxt(tf)[20:]; n = min(len(t), len(B)); p = B[:n, 15]
            print(f"{seq[:22]:22s} {trig:>7s} | {t.mean():6.1f} {np.median(t):6.1f} {np.percentile(t,95):6.1f} {np.mean(t>EPOCH[ds])*100:6.0f}% | {np.mean(p>1)*100:6.1f}% {p.mean():6.2f} | {fmt(rm(ds, seq, lab), 6)}")
elif ph == 'p2a':
    print("P2b. Accuracy with tempering trigger 0.0161 vs the final configuration (0.0244), all 25 sequences (runs concurrent, so no timing here).")
    print(f"{'sequence':40s} | {'0.0244':>7s} {'0.0161':>7s} {'diff':>7s} | {'temper% old/new':>16s} {'passes old/new':>15s}")
    MILS = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
    cases = [('miluv', s, {'default_1_remoteControlHighPace_1': 'fx_hp', 'default_1_random3_0': 'fx_r30'}.get(s, 'fx_' + s)) for s in MILS]
    cases += [('iului', s, 'fx_' + s) for s in ['SquareTurn', 'CircleSpin', 'SquareBob']]
    cases += [('ntu', s, ('w118_' if s.startswith('eee') else 'n_') + s) for s in ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']]
    D = []
    for ds, seq, old in cases:
        a = rm(ds, seq, old); b = rm(ds, seq, 'tq_' + seq); Bo = bias(ds, old); Bn = bias(ds, 'tq_' + seq)
        tp = lambda B: (np.mean(B[:, 15] > 1) * 100, B[:, 15].mean()) if B is not None else (np.nan, np.nan)
        (to, po), (tn, pn) = tp(Bo), tp(Bn)
        if np.isfinite(a) and np.isfinite(b): D.append(b - a)
        print(f"{seq[:40]:40s} | {fmt(a)} {fmt(b)} {fmt(b - a) if np.isfinite(a) and np.isfinite(b) else '      -'} | {to:6.1f} / {tn:6.1f}   {po:5.2f} / {pn:5.2f}")
    if D: print(f"mean diff {np.mean(D):+.4f} m, max |diff| {np.max(np.abs(D)):.3f} m over {len(D)} sequences")
elif ph == 'p3':
    print("P3. LM vs sampling seed repeats (MILUV random3_0, obstacles_0b; N 8192, whole span with flush) and NTU LM rerun with the original initial poses.")
    for ds, seq, sh, samp in [('miluv', 'default_1_random3_0', 'r30', 'fx_r30'), ('miluv', 'obstacles_1_random3_0b', 'obs', 'fx_obstacles_1_random3_0b')]:
        s = [rm(ds, seq, l) for l in (samp, f'ks1_{sh}', f'ks2_{sh}')]; m = [rm(ds, seq, l) for l in (f'lmk_{sh}', f'lmks1_{sh}', f'lmks2_{sh}')]
        old = rm(ds, seq, 'lm_' + seq)
        print(f"{seq:34s} sampling seeds 1234/1/2: {' '.join(fmt(x) for x in s)} | LM seeds 1234/1/2: {' '.join(fmt(x) for x in m)} | earlier LM run {fmt(old)}")
    print("NTU nya/sbs: sampling (n_*, still-fit heading) vs LM with the same initial pose (lmy_*) vs the earlier LM (lm_*, heading 0)")
    for seq in ['nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']:
        lmy = 'lmy_s02' if seq == 'sbs_02' else 'lmy_' + seq
        print(f"  {seq}: sampling {fmt(rm('ntu', seq, 'n_' + seq))} | LM matched {fmt(rm('ntu', seq, lmy))} | LM earlier {fmt(rm('ntu', seq, 'lm_' + seq))}")
