#pragma once

#include <map>
#include <string>

namespace kims {

inline std::map<std::string, std::string> paperConfig(const std::string& dataset, const std::string& sequence = "") {
    std::map<std::string, std::string> c = {
        {"gamma", "1"}, {"w_uwb", "1"}, {"ctrl_mode", "0"}, {"floor_penalty", "0"}, {"K", "4"}, {"carry_bias", "0"},
        {"fresh_only", "1"}, {"dc_acc", "0.1"}, {"dc_gyr", "0.01"}, {"true_time", "1"}, {"head", "2"}, {"seq", "1"},
        {"rb", "0"}, {"N", "2048"}, {"ess_min", "0.0244"}, {"seq_done", "0.05"}, {"seq_newmass", "0"}, {"seed", "1234"},
        {"still_bias", "1"}, {"rollout_fp32", "1"}, {"align_still", "3"}, {"imu_z_down", "0"}};
    auto set = [&c](std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const auto& p : kv) c[p.first] = p.second;
    };
    if (dataset == "miluv") {
        set({{"T", "229"}, {"dt", "0.0131"}, {"uwb_model", "1"}, {"head_filt", "1"},
             {"q_acc_x", "0.0229"}, {"q_acc_y", "0.0229"}, {"q_acc_z", "0.0229"},
             {"q_gyr_x", "0.01135"}, {"q_gyr_y", "0.01154"}, {"q_gyr_z", "0.01082"},
             {"sigma_acc_x", "3.0"}, {"sigma_acc_y", "3.0"}, {"sigma_acc_z", "3.0"},
             {"sigma_gyr_x", "0.08"}, {"sigma_gyr_y", "0.08"}, {"sigma_gyr_z", "0.05"},
             {"w_acc", "0.01"}, {"w_gyr", "0.01"}, {"sigma_imu_acc", "1.0"}, {"sigma_imu_gyr", "1.0"},
             {"uwb_delta", "3.0"}, {"ctrl_delta", "1.0"}, {"uwb_std", "-1.0"}});
    } else if (dataset == "ntu") {
        set({{"T", "118"}, {"dt", "0.0169"}, {"q_acc", "0.014"}, {"q_gyr", "0.00061"}, {"uwb_std", "0.3"},
             {"uwb_delta", "5"}, {"head_filt", "0"},
             {"sigma_acc_x", "37.1062"}, {"sigma_acc_y", "37.1062"}, {"sigma_acc_z", "37.4373"},
             {"sigma_gyr_x", "0.0863"}, {"sigma_gyr_y", "0.0863"}, {"sigma_gyr_z", "0.0863"},
             {"w_acc", "0.0344"}, {"w_gyr", sequence == "eee_03" ? "0.3" : "0.01"},
             {"sigma_imu_acc", "0.3332"}, {"sigma_imu_gyr", "0.0642"}, {"ctrl_delta", "0.9985"}, {"imu_z_down", "1"}});
    } else {
        set({{"sigma_acc_x", "1.2"}, {"sigma_acc_y", "1.2"}, {"sigma_acc_z", "9.015"},
             {"sigma_gyr_x", "0.0231"}, {"sigma_gyr_y", "0.0231"}, {"sigma_gyr_z", "0.0384"},
             {"sigma_imu_acc", "2.328"}, {"sigma_imu_gyr", "0.1684"}, {"ctrl_delta", "0.861"}});
        if (dataset == "util")
            set({{"T", "100"}, {"align_still", "1.0"}, {"dt", "0.02"}, {"q_acc", "0.179"}, {"q_gyr", "0.0089"},
                 {"uwb_std", "0.224"}, {"uwb_delta", "5"}, {"head_filt", "1"}, {"w_acc", "0"}, {"w_gyr", "0"}});
        else
            set({{"T", "200"}, {"dt", "0.01"}, {"q_acc", "0.040"}, {"q_gyr", "0.0013"}, {"head_filt", "1"},
                 {"group_same_ts", "1"}, {"w_acc", "0.1412"}, {"w_gyr", "0.0275"}, {"uwb_delta", "3.0"},
                 {"uwb_std", "0.1101"}});
    }
    return c;
}

}
