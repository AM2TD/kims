"""usage: python3 scripts/revision/lm_mirror_split.py > result/runlogs/lm_mirror_split.txt (from the package root).
Where does the matched-LM vs KIMS gap come from? Per time sample, GT side of the anchor plane and whether each estimate is
nearer the true or the mirrored GT. Classes: near plane (|h_gt| <= 0.25 m), both on the true branch, either on the mirror branch."""
import sys, os, csv, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es, eval_flip as ef
os.chdir(es.PKG)
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("lm_lab = {"); j = src.index("\n", src.index("lm_lab.update")); exec(src[i:j + 1])
lm_lab.update({"SquareTurn": "lmf_SquareTurn", "CircleSpin": "lmf_CircleSpin"})
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def est(f, lag, prism=False):
    P = es.rows(f); th = P[:-lag, 0]; X = P[lag:, 1:4]
    if prism: X = np.array([X[k] + es.Rof(P[lag + k, 4:8]) @ T_BP for k in range(len(X))])
    return th, X
def fit4(lt, lp, th, E):
    k = (th > lt[0] + 10) & (th < lt[-1] - 10); G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = E[k]
    if not np.all(np.isfinite(E)) or np.abs(E).max() > 1e3: return np.inf, None
    gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    return np.sqrt(np.mean(np.sum((E - ((Rz @ G.T).T + t)) ** 2, 1))), (Rz @ lp.T).T + t
rows = []
def analyse(ds, s, tk, Xk, tl, Xl, gt_t, gp, c, n, lo, hi):
    t = tk[(tk > lo) & (tk < hi)]
    g = np.stack([np.interp(t, gt_t, gp[:, j]) for j in range(3)], 1); h = (g - c) @ n; gm = g - 2 * h[:, None] * n[None, :]
    xk = np.stack([np.interp(t, tk, Xk[:, j]) for j in range(3)], 1); xl = np.stack([np.interp(t, tl, Xl[:, j]) for j in range(3)], 1)
    ek, el = np.linalg.norm(xk - g, axis=1), np.linalg.norm(xl - g, axis=1)
    mk = np.linalg.norm(xk - gm, axis=1) < ek; ml = np.linalg.norm(xl - gm, axis=1) < el
    near = np.abs(h) <= 0.25; both = ~near & ~mk & ~ml; mir = ~near & (mk | ml)
    r = lambda e, m: np.sqrt(np.mean(e[m] ** 2)) if m.any() else np.nan
    gap = np.mean(el ** 2) - np.mean(ek ** 2)
    contrib = lambda m: (np.sum(el[m] ** 2) - np.sum(ek[m] ** 2)) / len(t) / gap if abs(gap) > 1e-9 else np.nan
    rows.append((ds, s, r(ek, np.ones_like(near)), r(el, np.ones_like(near)), 100 * np.mean(~near & mk), 100 * np.mean(~near & ml),
                 100 * np.mean(both), r(ek, both), r(el, both), contrib(both), contrib(mir), contrib(near)))
for s in ["default_1_remoteControlHighPace_1", "default_1_random3_1", "default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2",
          "default_1_remoteControlLowPace_0", "default_1_circular2D_0", "default_1_remoteControlHighPace_0", "obstacles_1_random3_0b", "default_1_random3_2",
          "default_1_remoteControlHighPace_2", "noAprilTags_1_random3_2b", "noAprilTags_1_remoteControlHighPace_2b"]:
    tk, Xk = est(f"result/miluv/diag/f32a_{s}_pose.txt", 228); tl, Xl = est(f"result/miluv/diag/{lm_lab[s]}_pose.txt", 228)
    gt_t, gp = ef.load_gt("miluv", s); c, n = ef.anchor_plane("miluv", s)
    analyse("MILUV", s.replace("default_1_", "").replace("remoteControl", "rc"), tk, Xk, tl, Xl, gt_t, gp, c, n, gt_t[0], gt_t[-1])
for s in ["SquareTurn", "CircleSpin"]:   # IULUI AC1, anchors coplanar at 1.30 m; final IULUI runs (lag 199)
    tk, Xk = est(f"result/iului/diag/f32g_{s}_pose.txt", 199); tl, Xl = est(f"result/iului/diag/{lm_lab[s]}_pose.txt", 199)
    gt_t, gp = ef.load_gt("iului", s); c, n = ef.anchor_plane("iului", s)
    analyse("IULUI", s, tk, Xk, tl, Xl, gt_t, gp, c, n, gt_t[0], gt_t[-1])
for s in ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]:
    tk, Xk = est(f"result/ntu/diag/f32a_{s}_pose.txt", 117, True); tl, Xl = est(f"result/ntu/diag/{lm_lab[s]}_pose.txt", 117, True)
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"data/ntu/{s}/leica.csv")))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
    if s.startswith("eee"): gt_t, gp = ef.load_gt("ntu", s)
    else:
        cand = [fit4(lt, lp, tk, Xk), fit4(lt, lp, tl, Xl)]
        fe = "result/baselines/ekfl_%s_pose.txt" % s
        if os.path.exists(fe): te, Xe = est(fe, 0, True) if False else (lambda P: (P[:, 0], np.array([P[k, 1:4] + es.Rof(P[k, 4:8]) @ T_BP for k in range(len(P))])))(es.rows(fe)); cand.append(fit4(lt, lp, te, Xe))
        gt_t, gp = lt, min(cand, key=lambda x: x[0])[1]
    analyse("NTU", s, tk, Xk, tl, Xl, gt_t, gp, np.array([0, 0, 1.5]), np.array([0, 0, 1.0]), lt[0] + 10, lt[-1] - 10)
print(f"{'':6s} {'sequence':26s} | RMSE KIMS    LM | mirror% KIMS  LM | both-true: share  KIMS    LM | share of the LM-KIMS MSE gap from: both-true mirror near-plane")
for ds, s, rk, rl, mk, ml, sb, bk, bl, cb, cm, cn in rows:
    print(f"{ds:6s} {s:26s} |     {rk:5.3f} {rl:5.3f} |      {mk:4.1f} {ml:4.1f} |          {sb:4.0f}% {bk:5.3f} {bl:5.3f} |                     {cb:+6.2f} {cm:+6.2f} {cn:+6.2f}")
