#pragma once

#include <string>

namespace kims {

struct KimsPreset {
    int    N;
    int    T;
    double dt;
    double gamma;
    double sigma_acc[3];
    double sigma_gyr[3];
    double w_uwb;
    double w_acc;
    double w_gyr;
    int    ctrl_mode;        // 0 quadratic, 1 bounded Tukey
    double sigma_imu_acc;
    double sigma_imu_gyr;
    double uwb_delta;
    double ctrl_delta;
    double uwb_std;          // negative keeps the per-range std from the data
    double floor_penalty;
    int    imu_z_down;
};

struct EkfPreset { double std_acc, std_gyr, std_uwb; };
struct SwMapPreset { int window_size; };
struct PsPreset {
    int    num_particles;
    int    seed;
    double sigma_uwb, sigma_pos, sigma_vel, sigma_gyro, ess_thresh;
    int    smooth_lag;
};

inline KimsPreset kimsPreset(const std::string& dataset,
                             const std::string& sequence = "") {
    KimsPreset p{};
    if (dataset == "miluv") {
        p = KimsPreset{10000, 15, 0.01, 10.0,
                       {3.0, 3.0, 3.0}, {0.08, 0.08, 0.05},
                       1.0, 0.01, 0.01,
                       0, 1.0, 1.0, 3.0, 1.0, -1.0, 0.0, 0};
    } else if (dataset == "ntu") {
        p = KimsPreset{10000, 18, 0.015, 3.3573,
                       {37.1062, 37.1062, 37.4373}, {0.0863, 0.0863, 0.0863},
                       1.7201, 0.0344, 0.01,
                       1, 0.3332, 0.0642, 27.64, 0.9985, 0.1267, 0.0, 1};
        if (sequence == "eee_03") p.w_gyr = 0.3;
    } else {
        p = KimsPreset{10000, 15, 0.01, 27.428,
                       {1.2, 1.2, 9.015}, {0.0231, 0.0231, 0.0384},
                       1.899, 0.1412, 0.0275,
                       1, 2.328, 0.1684, 3.0, 0.861, 0.1101, 0.0, 0};
    }
    return p;
}

inline EkfPreset ekfPreset(const std::string& dataset) {
    if (dataset == "iului") return EkfPreset{0.5, 0.05, 0.2};
    return EkfPreset{0.3, 0.01, 0.1};
}

inline EkfPreset ukfPreset(const std::string& dataset) {
    if (dataset == "iului") return EkfPreset{0.5, 0.005, 0.2};
    return EkfPreset{0.3, 0.01, 0.1};
}

inline SwMapPreset swMapPreset(const std::string& dataset) {
    if (dataset == "iului") return SwMapPreset{20};
    return SwMapPreset{15};
}

inline PsPreset psPreset(const std::string& dataset) {
    if (dataset == "iului") return PsPreset{10000, 42, 0.3, 0.02, 0.05, 0.01, 0.7, 15};
    if (dataset == "ntu")   return PsPreset{10000, 42, 0.5, 0.01, 0.02, 0.01, 0.5, 15};
    return PsPreset{10000, 42, 0.3, 0.1, 0.1, 0.01, 0.5, 15};
}

}  // namespace kims
