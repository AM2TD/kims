#!/usr/bin/env python3
"""Floor-prior (z >= 0) ablation on NTU VIRAL for KIMS, the published configuration, EKF-RTS and SW-PGO.
4-DoF aligned 3D rmse [m]; KIMS lag 117, published lag 18, baselines lag 0."""
import sys, os, numpy as np
sys.path.insert(0, 'scripts'); exec(open('scripts/revision/final_table.py').read().split('MIL = [')[0])
NT = ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']
cols = [('KIMS', lambda s: (f'result/ntu/diag/{"w118_" if s.startswith("eee") else "n_"}{s}_pose.txt', 117)), ('KIMS+floor', lambda s: (f'result/ntu/diag/fl_{s}_pose.txt', 117)),
        ('pubKIMS', lambda s: (f'result/ntu/diag/pub_{s}_pose.txt', 18)), ('pubKIMS+fl', lambda s: (f'result/ntu/diag/pubfl_{s}_pose.txt', 18)),
        ('EKF-RTS', lambda s: (f'result/baselines/ekfrts_{s}_pose.txt', 0)), ('EKF-RTS+fl', lambda s: (f'result/baselines/ekfrtsfl_{s}_pose.txt', 0)),
        ('SW-PGO', lambda s: (f'result/baselines/swpgo_{s}_pose.txt', 0)), ('SW-PGO+fl', lambda s: (f'result/baselines/swpgofl_{s}_pose.txt', 0))]
print("NTU floor prior ablation (floor_penalty 50 = one-sided quadratic below z = 0, same weight convention in all four): 3D rmse, 4-DoF aligned, prism.")
print(f"{'sequence':8s} | " + ' '.join(f'{c[0]:>11s}' for c in cols))
M = []
for s in NT:
    v = [sc('ntu', s, *f(s)) for _, f in cols]; M.append(v)
    print(f"{s:8s} | " + ' '.join(('%11.3f' % x) if np.isfinite(x) else f'{"-":>11s}' for x in v))
M = np.array(M, dtype=float)
print(f"{'mean':8s} | " + ' '.join(('%11.3f' % np.nanmean(M[:, j])) if np.isfinite(M[:, j]).any() else f'{"-":>11s}' for j in range(M.shape[1])))
print(f"{'median':8s} | " + ' '.join(('%11.3f' % np.nanmedian(M[:, j])) if np.isfinite(M[:, j]).any() else f'{"-":>11s}' for j in range(M.shape[1])))
