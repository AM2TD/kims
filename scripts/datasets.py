import os
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["iului", "miluv", "ntu"]
METHODS = ["kims", "kims2", "kims_mh", "kims_integrity", "ekf_rts", "ukf_rts", "particle_smoothing", "sw_pgo"]

IULUI = dict(
    sequences=["SquareBob", "SquareTurn", "CircleSpin"],
    bag="data/iului/AC1_{seq}.bag",        
    data_dir="data/iului/{seq}",
    imu_topic="/mavros/imu/data",
    meta="data/iului/AC1_{seq}.json",
    uwb_topic="/nlink_linktrack_tagframe0",
    gt_topic="/qualisys/gogogo/pose",
    num_anchors=4,
    tag_ids=[8, 9],
    lag_shift=-0.15,
    anchor_ids=[0, 1, 2, 3],
)

MILUV = dict(
    # Anchor constellation 1 (all six anchors at one height, 1.415-1.429 m): the coplanar-anchor setting that
    # matches IULUI AC1. Experiment names end with the constellation: <scenario>_<robots>_<trajectory>_<c>[_suffix].
    sequences=["default_1_random3_1", "default_3_random3_1", "default_3_zigzag_1",
               "default_1_remoteControlHighPace_1"],
    # constellation 0 (anchor heights 1.59-1.90 m), used in the published KIMS evaluation
    sequences_c0=["default_1_random3_0", "default_1_circular3D_0",
                  "default_1_remoteControlLowPace_0_v2"],
    data_dir="data/miluv/{seq}/ifo001",
    anchor_file="data/miluv/anchors_constellation{c}.txt",
    tag_ids=[10, 11],
    tag_offsets=[[0.13189, -0.17245, -0.05249],
                 [-0.17542, 0.15712, -0.05307]],
    anchor_ids=[0, 1, 2, 3, 4, 5],
    # Ground-truth z shift applied at evaluation. The published evaluation used -0.0924 m on constellation 0; removed
    # 2026-10-13: the anchors and tag lever arms here are the devkit's own (config/uwb/anchors.yaml, tags.yaml), the
    # dataset's gt_range column is reproduced exactly by the raw mocap pose (shift 0.000 m, rms 0.00 cm, checked
    # 2026-09-30), and the devkit EKF is scored on the raw mocap, so every method is scored on the raw mocap pose.
    gt_z_bias={},
)

NTU = dict(
    sequences=["eee_01", "eee_02", "eee_03"],
    data_dir="data/ntu/{seq}",
    anchor_file="data/ntu/anchors_eee.txt",
    tag_ids=[2000, 2001, 2010, 2011],
    anchor_ids=[100, 101, 102],
    tag_offsets=[[0.00, -0.45, 0.00], [0.00, 0.45, 0.00],
                 [-0.60, 0.45, 0.00], [-0.60, -0.45, 0.00]],
    init=[0.0, 0.0, 0.0],
    align={
        "eee_01": dict(rx=-0.59, ry=1.07, rz=63.60,
                       tx=28.8022, ty=-2.9042, tz=0.4194),
        "eee_02": dict(rx=3.72, ry=-6.81, rz=64.94,
                       tx=28.5094, ty=-2.7819, tz=-0.1053),
        "eee_03": dict(rx=-1.20, ry=1.46, rz=63.26,
                       tx=28.6368, ty=-2.1135, tz=0.0791),
    },
)

CONFIG = {"iului": IULUI, "miluv": MILUV, "ntu": NTU}


def miluv_constellation(seq):
    """Anchor constellation of a MILUV experiment: from data/miluv/experiments.csv (devkit list; e.g. the experiment
    default_1_remoteControlHighPace_0 used constellation 0b), else the 4th field of the name."""
    import csv
    f = os.path.join(PKG, "data", "miluv", "experiments.csv")
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            if r["experiment"] == seq: return r["anchor_constellation"]
    return seq.split("_")[3]


def anchor_path(dataset, seq):
    """Anchor file (relative to the package) for datasets that use one; MILUV depends on the constellation."""
    ds = CONFIG[dataset]
    if dataset == "miluv":
        return ds["anchor_file"].format(c=miluv_constellation(seq))
    return ds["anchor_file"]


def gt_z_bias(dataset, seq):
    if dataset != "miluv": return 0.0
    return CONFIG["miluv"]["gt_z_bias"].get(miluv_constellation(seq), 0.0)
