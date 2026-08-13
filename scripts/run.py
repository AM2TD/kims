#!/usr/bin/env python3
"""
Examples
    python3 scripts/run.py --method kims --dataset iului --sequence SquareBob
    python3 scripts/run.py --method all  --dataset all                # full paper sweep
    roslaunch kims kims.launch dataset:=ntu sequence:=eee_01

Estimator settings are compiled in (include/presets.h) and selected by
dataset. Anything can still be overridden on the command line, for example
    python3 scripts/run.py --method kims --dataset ntu -- N:=3000
"""
import argparse, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CONFIG, DATASETS, METHODS

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_binary(method, dataset):
    if method == "sw_map" and dataset == "iului":
        name = "sw_map_iului"
    else:
        name = method
    cands = []
    if os.environ.get("KIMS_BIN_DIR"):
        cands.append(os.path.join(os.environ["KIMS_BIN_DIR"], name))
    d = PKG
    for _ in range(4):
        d = os.path.dirname(d)
        cands.append(os.path.join(d, "devel", "lib", "kims", name))  
        cands.append(os.path.join(d, "build", "kims", "bin", name)) 
    cands.append(os.path.join(PKG, "build", "bin", name))            
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit(f"[run] binary '{name}' not found — build first (see the README) or "
             f"set KIMS_BIN_DIR. Tried:\n  " + "\n  ".join(cands))


def compose(dataset, method, seq, out_file):
    ds = CONFIG[dataset]
    a = [f"dataset:={dataset}", f"sequence:={seq}"]
    if dataset == "iului":
        meta = json.load(open(os.path.join(PKG, ds["meta"].format(seq=seq))))
        a += [f"data_dir:={os.path.join(PKG, ds['data_dir'].format(seq=seq))}",
              f"output_file:={out_file}",
              f"num_anchors:={ds['num_anchors']}",
              f"tag0_id:={ds['tag_ids'][0]}", f"tag1_id:={ds['tag_ids'][1]}",
              f"init_x:={meta['init'][0]}", f"init_y:={meta['init'][1]}",
              f"init_z:={meta['init'][2]}"]
        for i in range(ds["num_anchors"]):
            for k, ax in enumerate("xyz"):
                a.append(f"anchor{i}_{ax}:={meta['anchors'][i][k]}")
        for k, ax in enumerate("xyz"):
            a.append(f"tag0_offset_{ax}:={meta['tag0_offset'][k]}")
            a.append(f"tag1_offset_{ax}:={meta['tag1_offset'][k]}")
    elif dataset == "miluv":
        a += [f"data_dir:={os.path.join(PKG, ds['data_dir'].format(seq=seq))}",
              f"anchor_file:={os.path.join(PKG, ds['anchor_file'])}",
              f"output_file:={out_file}",
              f"tag0_id:={ds['tag_ids'][0]}", f"tag1_id:={ds['tag_ids'][1]}"]
        for t, off in enumerate(ds["tag_offsets"]):
            for k, ax in enumerate("xyz"):
                a.append(f"tag{t}_offset_{ax}:={off[k]}")
    else:
        a += [f"data_dir:={os.path.join(PKG, ds['data_dir'].format(seq=seq))}",
              f"anchor_file:={os.path.join(PKG, ds['anchor_file'])}",
              f"output_file:={out_file}"]
        for t, tid in enumerate(ds["tag_ids"]):
            a.append(f"tag{t}_id:={tid}")
            for k, ax in enumerate("xyz"):
                a.append(f"tag{t}_offset_{ax}:={ds['tag_offsets'][t][k]}")
        for i, aid in enumerate(ds["anchor_ids"]):
            a.append(f"anchor{i}_id:={aid}")
        a += [f"init_x:={ds['init'][0]}", f"init_y:={ds['init'][1]}",
              f"init_z:={ds['init'][2]}"]
    return a


def stage_gt(dataset, seq, res_dir):
    """Make sure the ground-truth file sits next to the pose outputs."""
    ds = CONFIG[dataset]
    if dataset == "miluv":
        src = os.path.join(PKG, ds["data_dir"].format(seq=seq), "mocap.csv")
        dst = os.path.join(res_dir, "mocap.csv")
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy(src, dst)
    elif dataset == "ntu":
        src = os.path.join(PKG, ds["data_dir"].format(seq=seq), "leica.csv")
        dst = os.path.join(res_dir, "leica.csv")
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy(src, dst)
    else:
        src = os.path.join(PKG, ds["data_dir"].format(seq=seq), "mocap.csv")
        dst = os.path.join(res_dir, "mocap.csv")
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy(src, dst)


def run_one(dataset, method, seq, extra):
    res_dir = os.path.join(PKG, "result", dataset, seq)
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"{method}_pose.txt")
    binary = find_binary(method, dataset)
    argv = [binary] + compose(dataset, method, seq, out) + extra
    print(f"[run] {dataset}/{seq} :: {method}  ({binary})", flush=True)
    t0 = time.time()
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        print(f"[run] WARNING: {method} on {dataset}/{seq} exited {r.returncode}")
    else:
        print(f"[run] {dataset}/{seq} :: {method}  done in {time.time() - t0:.1f} s",
              flush=True)
    stage_gt(dataset, seq, res_dir)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all", help="|".join(METHODS) + "|all")
    ap.add_argument("--dataset", default="all", help="|".join(DATASETS) + "|all")
    ap.add_argument("--sequence", default="all")
    args, rest = ap.parse_known_args()
    extra = [a for a in rest if ":=" in a]

    methods = METHODS if args.method == "all" else [args.method]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    if args.dataset not in DATASETS + ["all"]:
        sys.exit(f"[run] unknown dataset '{args.dataset}' — one of {DATASETS}|all")
    ran = 0
    for dataset in datasets:
        if args.sequence != "all" and args.sequence not in CONFIG[dataset]["sequences"]:
            if args.dataset == "all":
                continue
            sys.exit(f"[run] '{args.sequence}' is not a {dataset} sequence — one of "
                     f"{CONFIG[dataset]['sequences']}")
        seqs = (CONFIG[dataset]["sequences"] if args.sequence == "all"
                else [args.sequence])
        ran += len(seqs)
        for seq in seqs:
            for method in methods:
                run_one(dataset, method, seq, extra)
    if ran == 0:
        sys.exit(f"[run] no sequence matched '{args.sequence}' in {datasets}")
    print("[run] done — outputs in result/<dataset>/<sequence>/. "
          "Plot with: python3 scripts/plot.py")


if __name__ == "__main__":
    main()
