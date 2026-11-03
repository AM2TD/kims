#!/usr/bin/env python3
"""Shared loader for the plane-crossing analysis (sampling vs matched LM). Every analysis must use this so that time
alignment, frames and run pairs are identical.

load(ds, seq) -> dict with
  t        : (n,) times [s since GT start] of the estimate samples inside the GT span
  gt       : (n,3) GT body position at t, in the UWB/anchor frame
  samp, lm : (n,3) body position of the sampling run and of the matched-LM run at t (KIMS lag T applied exactly)
  c, n     : anchor-plane point and unit normal (normal oriented up)
  h_gt, h_samp, h_lm : signed height of the body above the anchor plane
  coplanar : True for MILUV constellation 1, IULUI and NTU (all anchors at one height); False otherwise
  diag     : sampling run per solve: t_solve (newest epoch of the window), ess1 (first-pass ESS of 8192), ess_final, passes
Frames: MILUV/IULUI GT is mocap in the anchor frame. NTU GT is the Leica prism in its own frame: it is mapped into the UWB
frame with a yaw+translation (4-DoF) fit to the sampling run's prism track, then converted to the body with the sampling
run's attitude (T_Body_Prism). The same GT is used for both methods.
"""
import os, sys, csv, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
import eval_flip as ef
from datasets import miluv_constellation
T_BP = np.array([-0.293656, -0.012288, -0.273095])
LAG = {'miluv': 228, 'iului': 199, 'ntu': 117}
MILUV = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0',
         'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0',
         'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2',
         'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
IULUI = ['SquareTurn', 'CircleSpin', 'SquareBob']
NTU = ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']
def labels(ds, seq):
    if ds == 'miluv':
        s = {'default_1_remoteControlHighPace_1': 'fx_hp', 'default_1_random3_0': 'fx_r30'}.get(seq, 'fx_' + seq)
        m = {'default_1_remoteControlHighPace_1': 'lm_hp', 'default_1_random3_1': 'lm_r3'}.get(seq, 'lm_' + seq)
        return s, m
    if ds == 'iului': return 'fx_' + seq, 'lm_' + seq
    if seq.startswith('eee'): return 'w118_' + seq, 'lm_' + seq
    return 'n_' + seq, ('lmy_s02' if seq == 'sbs_02' else 'lmy_' + seq)   # LM with the same initial pose as the sampling run
def all_cases():
    return [('miluv', s) for s in MILUV] + [('iului', s) for s in IULUI] + [('ntu', s) for s in NTU]
def _rows(f): return np.array([[float(x) for x in l.split()[:8]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])
def _R(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def _est(ds, lab):
    P = _rows(os.path.join(PKG, f'result/{ds}/diag/{lab}_pose.txt')); k = LAG[ds]
    return P[:-k, 0], P[k:, 1:4], P[k:, 4:8]      # time of the head, head position, head attitude (KIMS exact lag)
def load(ds, seq):
    ls, lm = labels(ds, seq)
    for l in (ls, lm):   # a run is complete only once its timing file exists
        if not os.path.exists(os.path.join(PKG, f'result/{ds}/diag/{l}_timing.txt')): raise FileNotFoundError(f'{ds}/{l} (missing or unfinished)')
    ts, xs, qs = _est(ds, ls); tl, xl, _ = _est(ds, lm)
    c, n = ef.anchor_plane(ds, seq) if not (ds == 'ntu') else (None, None)
    if ds == 'ntu':
        anc = 'data/ntu/anchors_eee.txt' if seq.startswith('eee') else f'data/ntu/anchors_{seq}.txt'
        A = np.loadtxt(os.path.join(PKG, anc)); c = A.mean(0); _, _, Vt = np.linalg.svd(A - c); n = Vt[-1]; n = n if n[2] > 0 else -n
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(os.path.join(PKG, f'data/ntu/{seq}/leica.csv'))))[1:]])
        gt_t, gp = L[:, 0], L[:, 1:4]
        prism = xs + np.einsum('nij,j->ni', np.array([_R(q) for q in qs]), T_BP)
        k = (ts > gt_t[0] + 10) & (ts < gt_t[-1] - 10)
        G = np.stack([np.interp(ts[k], gt_t, gp[:, j]) for j in range(3)], 1); E = prism[k]
        gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
        yaw = np.arctan2(np.sum(g2[:, 0]*e2[:, 1] - g2[:, 1]*e2[:, 0]), np.sum(g2[:, 0]*e2[:, 0] + g2[:, 1]*e2[:, 1]))
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]); tr = ec - Rz @ gc
        gp = (Rz @ gp.T).T + tr                                           # Leica prism in the UWB frame
        tt = ts[(ts >= gt_t[0]) & (ts <= gt_t[-1])]
        Rs = np.array([_R(q) for q in qs]); idx = np.searchsorted(ts, tt).clip(0, len(ts) - 1)
        gt = np.stack([np.interp(tt, gt_t, gp[:, j]) for j in range(3)], 1) - np.einsum('nij,j->ni', Rs[idx], T_BP)   # body = prism - R T_BP
        coplanar = True; t0 = gt_t[0]
    else:
        gt_t, gpos = ef.load_gt(ds, seq)
        tt = ts[(ts >= gt_t[0]) & (ts <= gt_t[-1])]
        gt = np.stack([np.interp(tt, gt_t, gpos[:, j]) for j in range(3)], 1)
        coplanar = (ds == 'iului') or (miluv_constellation(seq) == '1'); t0 = gt_t[0]
    samp = np.stack([np.interp(tt, ts, xs[:, j]) for j in range(3)], 1)
    lmx = np.stack([np.interp(tt, tl, xl[:, j]) for j in range(3)], 1)
    h = lambda X: (X - c) @ n
    # sampling-run diagnostics per solve (time = newest epoch of the window, i.e. when the window first sees an event)
    B = np.array([[float(x) for x in l.split()[:16]] for l in open(os.path.join(PKG, f'result/{ds}/diag/{ls}_bias.txt')) if not l.startswith('#') and len(l.split()) >= 16])
    diag = dict(t_solve=B[:, 0] - t0, ess1=B[:, 13], ess_final=B[:, 14], passes=B[:, 15])
    return dict(t=tt - t0, gt=gt, samp=samp, lm=lmx, c=c, n=n, h_gt=h(gt), h_samp=h(samp), h_lm=h(lmx), coplanar=coplanar, labels=(ls, lm), diag=diag)
if __name__ == '__main__':
    for ds, seq in all_cases():
        try:
            d = load(ds, seq); e = lambda X: np.sqrt(np.mean(np.sum((X - d['gt'])**2, 1)))
            print(f"{ds:6s} {seq:40s} n {len(d['t']):6d} coplanar {d['coplanar']!s:5s} rmse samp {e(d['samp']):.3f} lm {e(d['lm']):.3f} | GT h range {d['h_gt'].min():+.2f}..{d['h_gt'].max():+.2f}")
        except FileNotFoundError as ex: print(f"{ds:6s} {seq:40s} missing {ex}")
