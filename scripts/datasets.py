DATASETS = ["iului", "miluv", "ntu"]
METHODS = ["kims", "ekf_rts", "ukf_rts", "particle_smoothing", "sw_map"]

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
    sequences=["default_1_random3_0", "default_1_circular3D_0",
               "default_1_remoteControlLowPace_0_v2"],
    data_dir="data/miluv/{seq}/ifo001",
    anchor_file="data/miluv/anchors_constellation0.txt",
    tag_ids=[10, 11],
    tag_offsets=[[0.13189, -0.17245, -0.05249],
                 [-0.17542, 0.15712, -0.05307]],
    anchor_ids=[0, 1, 2, 3, 4, 5],
    gt_z_bias=-0.0924,
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
