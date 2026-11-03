"""Range-error model from the constellation-0 calibration flights (same vehicle and UWB hardware as the AC1 test flights):
two-component Gaussian mixture of (measured range - ground-truth range), fitted by EM. Component 1 = line-of-sight core,
component 2 = NLOS / multipath delay (positive)."""
import csv, json, numpy as np
E = []
for seq in ["default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2"]:
    R = list(csv.DictReader(open(f"data/miluv/{seq}/ifo001/uwb_range.csv")))
    r = np.array([float(x["range"]) for x in R]); g = np.array([float(x["gt_range"]) for x in R])
    e = r - g; E.append(e[np.isfinite(e) & (g > 0.3)])
e = np.concatenate(E)
eps, m1, s1, m2, s2 = 0.2, 0.0, 0.08, 0.5, 0.3
for it in range(500):
    p1 = (1 - eps) * np.exp(-0.5 * ((e - m1) / s1) ** 2) / s1
    p2 = eps * np.exp(-0.5 * ((e - m2) / s2) ** 2) / s2
    w2 = p2 / (p1 + p2); w1 = 1 - w2
    eps = w2.mean(); m1 = np.sum(w1 * e) / w1.sum(); m2 = np.sum(w2 * e) / w2.sum()
    s1 = np.sqrt(np.sum(w1 * (e - m1) ** 2) / w1.sum()); s2 = np.sqrt(np.sum(w2 * (e - m2) ** 2) / w2.sum())
ll = np.mean(np.log((1 - eps) * np.exp(-0.5 * ((e - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi)) + eps * np.exp(-0.5 * ((e - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))))
out = dict(n=int(len(e)), eps=float(eps), mu1=float(m1), s1=float(s1), mu2=float(m2), s2=float(s2), mean_loglik=float(ll))
print(out)
# check against the test sequences (not used in the fit)
for seq in ["default_1_random3_1", "default_1_remoteControlHighPace_1"]:
    R = list(csv.DictReader(open(f"data/miluv/{seq}/ifo001/uwb_range.csv")))
    r = np.array([float(x["range"]) for x in R]); g = np.array([float(x["gt_range"]) for x in R]); t = r - g; t = t[np.isfinite(t) & (g > 0.3)]
    llt = np.mean(np.log((1 - eps) * np.exp(-0.5 * ((t - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi)) + eps * np.exp(-0.5 * ((t - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))))
    print(f"{seq}: mean log-likelihood under the calibration model {llt:.3f} (calibration {ll:.3f})")
json.dump(out, open("result/runlogs/range_mix_c0.json", "w"))
