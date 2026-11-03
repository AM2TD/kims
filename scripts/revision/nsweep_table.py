#!/usr/bin/env python3
"""Sample-count sweep (N = 256 .. 8192) of the final configuration: 3D rmse, z, wrong-side fraction, ESS and passes.
MILUV / IULUI: whole GT span with the window flush; NTU: 4-DoF aligned (prism), window gt+10 .. end-10. Same scorer as
final_table.py. N 8192 rows are the final-table runs."""
import sys, os, csv, numpy as np
sys.path.insert(0, 'scripts'); import eval_flip as ef
exec(open('scripts/revision/final_table.py').read().split('MIL = [')[0])
SEQ = [('miluv', 'default_1_remoteControlHighPace_1', 'hp1', 'fx_hp', 228), ('miluv', 'default_1_random3_1', 'r31', 'fx_default_1_random3_1', 228),
       ('miluv', 'default_1_random3_0', 'r30', 'fx_r30', 228), ('miluv', 'obstacles_1_random3_0b', 'obs', 'fx_obstacles_1_random3_0b', 228),
       ('miluv', 'default_1_remoteControlHighPace_0', 'hp0', 'fx_default_1_remoteControlHighPace_0', 228),
       ('iului', 'SquareTurn', 'st', 'fx_SquareTurn', 199), ('iului', 'CircleSpin', 'cs', 'fx_CircleSpin', 199),
       ('ntu', 'eee_01', 'e01', 'w118_eee_01', 117), ('ntu', 'sbs_02', 's02', 'n_sbs_02', 117), ('ntu', 'nya_03', 'y03', 'n_nya_03', 117)]
NS = [8192, 4096, 2048, 1024, 512, 256]
div = set(l.split()[1] for l in open('result/runlogs/nsweep_diverged.txt') if l.startswith('DIVERGED')) if os.path.exists('result/runlogs/nsweep_diverged.txt') else set()
stop = set(l.split()[1] for l in open('result/runlogs/nsweep_stopped.txt') if l.startswith('STOPPED')) if os.path.exists('result/runlogs/nsweep_stopped.txt') else set()
def diag(ds, lab): return f'result/{ds}/diag/{lab}'
def ess_pass(ds, lab):
    f = diag(ds, lab) + '_bias.txt'
    if not os.path.exists(f): return np.nan, np.nan
    B = np.array([[float(x) for x in l.split()[:16]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 16])
    return (np.median(B[20:, 13]), B[20:, 15].mean()) if len(B) > 40 else (np.nan, np.nan)
def cell(ds, seq, lab, lag):
    if lab in div: return 'DIV'
    if lab in stop: return 'COLLAPSE'
    f = diag(ds, lab) + '_pose.txt'
    if not os.path.exists(f): return None
    v = sc(ds, seq, f, lag)
    return v if np.isfinite(v) else None
print("N sweep, final configuration (only N changes). 3D rmse [m]; 'DIV' = run stopped after leaving a 60 m box; 'COLLAPSE' = stopped because the first-pass ESS stayed at 1 with 21 passes per solve. Seeds: s1/s2 = seed 1/2 (default 1234).")
print(f"{'sequence':6s} | " + ' '.join(f'{n:>7d}' for n in NS) + " | seed sd (N 2048 / 1024 / 512)")
summary = {n: [] for n in NS}
for ds, seq, sh, ref, lag in SEQ:
    row = []
    for n in NS:
        lab = ref if n == 8192 else f'n{n}_{sh}'
        v = cell(ds, seq, lab, lag); row.append(v)
        if isinstance(v, float): summary[n].append(v)
    sds = []
    for n in (2048, 1024, 512):
        vals = [cell(ds, seq, f'n{n}_{sh}', lag)] + [cell(ds, seq, f'n{n}s{s}_{sh}', lag) for s in (1, 2)]
        vals = [v for v in vals if isinstance(v, float)]
        sds.append(f'{np.std(vals, ddof=1):.3f}({len(vals)})' if len(vals) >= 2 else '-')
    print(f"{sh:6s} | " + ' '.join(('%7.3f' % v) if isinstance(v, float) else f'{(v or "-"):>7s}' for v in row) + " | " + ' / '.join(sds))
print(f"{'mean':6s} | " + ' '.join(('%7.3f' % np.mean(summary[n])) if summary[n] else f'{"-":>7s}' for n in NS) + "   (over sequences that finished)")
print(f"{'n ok':6s} | " + ' '.join(f'{len(summary[n]):>7d}' for n in NS))
print()
print("median ESS of the first pass / mean passes per solve")
for ds, seq, sh, ref, lag in SEQ:
    print(f"{sh:6s} | " + ' '.join('%5.0f/%4.2f' % ess_pass(ds, ref if n == 8192 else f'n{n}_{sh}') for n in NS))
