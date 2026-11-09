#!/usr/bin/env python3
"""python3 scripts/run.py <kims|swpgo|pf> <miluv|ntu|util|iului> <sequence> [--label L] [--dry] [--args-only] [key:=value ...]"""
import argparse, json, os, subprocess, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCES = {
    "miluv": ["default_1_remoteControlHighPace_1", "default_1_random3_1", "default_1_random3_0", "default_1_circular3D_0",
              "default_1_remoteControlLowPace_0_v2", "default_1_remoteControlLowPace_0", "default_1_circular2D_0",
              "default_1_remoteControlHighPace_0", "obstacles_1_random3_0b", "default_1_random3_2",
              "default_1_remoteControlHighPace_2", "noAprilTags_1_random3_2b", "noAprilTags_1_remoteControlHighPace_2b"],
    "ntu": ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"],
    "util": None,
    "iului": ["SquareTurn", "CircleSpin", "AC3_125000"],
}
WINDOW = {"miluv": 229, "ntu": 118, "util": 100, "iului": 200}

MILUV_TAGS = [(10, (0.13189, -0.17245, -0.05249)), (11, (-0.17542, 0.15712, -0.05307))]
NTU_TAGS = [(2000, (0.00, -0.45, 0.00)), (2001, (0.00, 0.45, 0.00)), (2010, (-0.60, 0.45, 0.00)), (2011, (-0.60, -0.45, 0.00))]
UTIL_TAG = (-0.01245, 0.00127, 0.0908)

def sequences(dataset):
    if SEQUENCES[dataset] is not None: return SEQUENCES[dataset]
    d = os.path.join(PKG, "data/util")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []

def miluv_constellation(seq):
    import csv
    f = os.path.join(PKG, "data/miluv/experiments.csv")
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            if r["experiment"] == seq: return r["anchor_constellation"]
    return seq.split("_")[3]

def iului_meta(seq):
    f = os.path.join(PKG, "data/iului", f"AC1_{seq}.json")
    return json.load(open(f if os.path.exists(f) else os.path.join(PKG, "data/iului", f"{seq}.json")))

def data_args(dataset, seq, out):
    a = {"dataset": dataset, "sequence": seq, "pkg_dir": PKG, "output_file": out}
    if dataset == "miluv":
        a.update(data_dir=f"{PKG}/data/miluv/{seq}/ifo001", anchor_file=f"{PKG}/data/miluv/anchors_constellation{miluv_constellation(seq)}.txt")
        for t, (tid, off) in enumerate(MILUV_TAGS):
            a[f"tag{t}_id"] = tid
            a.update({f"tag{t}_offset_{ax}": v for ax, v in zip("xyz", off)})
    elif dataset == "ntu":
        p = json.load(open(f"{PKG}/data/ntu/init_poses.json"))[seq]
        a.update(data_dir=f"{PKG}/data/ntu/{seq}", anchor_file=f"{PKG}/{p['anchor_file']}")
        for t, (tid, off) in enumerate(NTU_TAGS):
            a[f"tag{t}_id"] = tid
            a.update({f"tag{t}_offset_{ax}": v for ax, v in zip("xyz", off)})
        a.update({f"anchor{i}_id": aid for i, aid in enumerate((100, 101, 102))})
        a.update(init_x=p["init_x"], init_y=p["init_y"], init_z=p["init_z"], init_yaw=p["init_yaw"])
    elif dataset == "util":
        d = f"{PKG}/data/util/{seq}"
        a.update(data_dir=d, anchor_file=f"{d}/anchors_A.txt", anchor_b_file=f"{d}/anchors_B.txt", tag0_id=0,
                 num_anchors=json.load(open(f"{d}/meta.json"))["num_slots"], init_x=1.25, init_y=0.0, init_z=0.07, init_yaw=0,
                 group_same_ts=1)
        a.update({f"tag0_offset_{ax}": v for ax, v in zip("xyz", UTIL_TAG)})
    else:
        m = iului_meta(seq)
        a.update(data_dir=f"{PKG}/data/iului/{seq}", num_anchors=4, tag0_id=8, tag1_id=9,
                 init_x=m["init"][0], init_y=m["init"][1], init_z=m["init"][2])
        for i in range(4):
            a.update({f"anchor{i}_{ax}": m["anchors"][i][k] for k, ax in enumerate("xyz")})
        for t in (0, 1):
            a.update({f"tag{t}_offset_{ax}": m[f"tag{t}_offset"][k] for k, ax in enumerate("xyz")})
    return a

BASE = {
    "miluv": (229, 0.081, 5, 0.0229, 0.0113, 0.0577, 0.00577),
    "ntu":   (118, 0.3, 5, 0.014, 0.00061, 0.0707, 0.00707),
    "util":  (100, 0.224, 5, 0.179, 0.0089, 0.0707, 0.00707),
    "iului": (200, 0.11, 3, 0.040, 0.0013, 0.0707, 0.00707),
}

def baseline_args(method, dataset, seq, label):
    T, sig, cut, qa, qg, bwa, bwg = BASE[dataset]
    a = dict(q_acc=qa, q_gyr=qg, dc_acc=0.1, dc_gyr=0.01, align_still=1.0 if dataset == "util" else 3, still_bias=1)
    a["fresh_only" if dataset in ("miluv", "ntu") else "group_same_ts"] = 1
    if dataset == "ntu":
        f = json.load(open(f"{PKG}/data/ntu/init_fit.json"))[seq]
        a.update(init_x=f["init_x"], init_y=f["init_y"], init_yaw=f["init_yaw"])
    if dataset == "iului":
        m = iului_meta(seq); af = f"{PKG}/result/iului/{label}_{seq}_anchors.txt"
        os.makedirs(os.path.dirname(af), exist_ok=True)
        open(af, "w").write("# x y z (from the meta json)\n" + "".join("%f %f %f\n" % tuple(p) for p in m["anchors"]))
        a["anchor_file"] = af
    if method == "swpgo":
        a.update(window_size=T, std_uwb=sig, robust_tukey=1, robust_delta=cut, out_index=1, sparse=1, threads=1,
                 carry_cov=1, prior_pos=0.2, prior_vel=0.05, prior_rot=0.01, max_dt=5,
                 ftol=1e-12, gtol=1e-16, ptol=1e-12, max_iter=60)
    else:
        a.update(num_particles=8192, smooth_lag=T, sigma_uwb=sig, tukey_delta=cut, ess_thresh=0.5, seed=1234, bw_acc=bwa, bw_gyr=bwg)
    return a

def command(method, dataset, seq, label, extra):
    out = f"{PKG}/result/{dataset}/{label}_{seq}_pose.txt"
    a = data_args(dataset, seq, out)
    if method == "kims":
        binary = "kims"
    else:
        a.update(baseline_args(method, dataset, seq, label))
        binary = {"swpgo": "sw_pgo", "pf": "pf"}[method]
    for kv in extra:
        k, v = kv.split(":=", 1); a[k] = v
    b = os.path.join(os.environ.get("KIMS_BUILD", os.path.join(PKG, "build")), "bin", binary)
    return [b] + [f"{k}:={v}" for k, v in a.items()], out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["kims", "swpgo", "pf"])
    ap.add_argument("dataset", choices=list(SEQUENCES))
    ap.add_argument("sequence")
    ap.add_argument("--label")
    ap.add_argument("--dry", action="store_true", help="print the command only")
    ap.add_argument("--args-only", action="store_true", help="print the key:=value settings only (for the online ROS node)")
    args, extra = ap.parse_known_args()
    bad = [e for e in extra if ":=" not in e]
    if bad: sys.exit(f"[run] unknown arguments {bad} (overrides are key:=value)")
    label = args.label or args.method
    cmd, out = command(args.method, args.dataset, args.sequence, label, extra)
    if args.args_only: print(" ".join(cmd[1:])); return
    if args.dry: print(" ".join(cmd)); return
    if not os.path.exists(cmd[0]): sys.exit(f"[run] {cmd[0]} not found: build first (README) or set KIMS_BUILD")
    os.makedirs(os.path.dirname(out), exist_ok=True); os.makedirs(f"{PKG}/result/logs", exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("KIMS_OMP_THREADS", "8"))
    log = f"{PKG}/result/logs/{label}_{args.sequence}.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PKG)
    print(f"{label}_{args.sequence} {'done' if r.returncode == 0 else f'FAILED ({r.returncode}), see ' + log}")
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
