#pragma once

#include "data.h"

#include <cstdio>
#include <cstdlib>

namespace kims {

struct Session {
    std::vector<ImuData>  imu;
    std::vector<UwbRange> uwb;
    std::vector<GtPose>   gt;

    std::vector<double> anchors;
    std::vector<double> tag_offsets;
    int num_anchors = 0;
    int num_tags    = 0;

    Eigen::Vector3d init_p = Eigen::Vector3d::Zero();

    int  numRanges() const { return num_tags * num_anchors; }
    bool hasOrientationGt() const { return !gt.empty() && gt.front().orientation.w() != 2.0; }
};

inline std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> m;
    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);
        const size_t pos = s.find(":=");
        if (pos != std::string::npos) m[s.substr(0, pos)] = s.substr(pos + 2);
    }
    return m;
}

template <typename T>
inline T getArg(const std::map<std::string, std::string>& a,
                const std::string& key, T fallback);

template <>
inline std::string getArg<std::string>(const std::map<std::string, std::string>& a,
                                       const std::string& key, std::string fallback) {
    auto it = a.find(key);
    return it == a.end() ? fallback : it->second;
}
template <>
inline double getArg<double>(const std::map<std::string, std::string>& a,
                             const std::string& key, double fallback) {
    auto it = a.find(key);
    return it == a.end() ? fallback : std::atof(it->second.c_str());
}
template <>
inline int getArg<int>(const std::map<std::string, std::string>& a,
                       const std::string& key, int fallback) {
    auto it = a.find(key);
    return it == a.end() ? fallback : std::atoi(it->second.c_str());
}

inline bool loadImuCsv(const std::string& path, std::vector<ImuData>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[kims] cannot open IMU csv: " << path << "\n"; return false; }
    std::string line;
    std::getline(ifs, line);
    const std::vector<std::string> needed = {
        "timestamp",
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"};
    auto idx = parseHeader(line, needed);
    for (const auto& n : needed)
        if (idx[n] < 0) { std::cerr << "[kims] IMU csv missing column: " << n << "\n"; return false; }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto c = splitCsvRow(line);
        if (static_cast<int>(c.size()) <= idx["linear_acceleration.z"]) continue;
        ImuData d;
        d.timestamp = std::stod(c[idx["timestamp"]]);
        d.gyr = Eigen::Vector3d(std::stod(c[idx["angular_velocity.x"]]),
                                std::stod(c[idx["angular_velocity.y"]]),
                                std::stod(c[idx["angular_velocity.z"]]));
        d.acc = Eigen::Vector3d(std::stod(c[idx["linear_acceleration.x"]]),
                                std::stod(c[idx["linear_acceleration.y"]]),
                                std::stod(c[idx["linear_acceleration.z"]]));
        out.push_back(d);
    }
    return !out.empty();
}

inline bool loadUwbCsv(const std::string& path,
                       const std::vector<int>& tag_ids,
                       const std::vector<int>& anchor_ids,
                       std::vector<UwbRange>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[kims] cannot open UWB csv: " << path << "\n"; return false; }
    std::string line;
    std::getline(ifs, line);
    auto idx = parseHeader(line, {"timestamp", "range", "from_id", "to_id", "std"});
    for (const char* n : {"timestamp", "range", "from_id", "to_id"})
        if (idx[std::string(n)] < 0) {
            std::cerr << "[kims] UWB csv missing column: " << n << "\n"; return false;
        }
    const bool has_std = idx["std"] >= 0;

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto c = splitCsvRow(line);
        if (c.size() < 4) continue;
        const int from_id = std::stoi(c[idx["from_id"]]);
        const int to_id   = std::stoi(c[idx["to_id"]]);
        const auto t_it = std::find(tag_ids.begin(),    tag_ids.end(),    from_id);
        const auto a_it = std::find(anchor_ids.begin(), anchor_ids.end(), to_id);
        if (t_it == tag_ids.end() || a_it == anchor_ids.end()) continue;

        UwbRange d;
        d.timestamp = std::stod(c[idx["timestamp"]]);
        d.range     = std::stod(c[idx["range"]]);
        d.tag_idx   = static_cast<int>(t_it - tag_ids.begin());
        d.anc_idx   = static_cast<int>(a_it - anchor_ids.begin());
        d.std_dev   = has_std ? std::stod(c[idx["std"]]) : 0.1;
        if (d.std_dev < 1e-4) d.std_dev = 1e-4;
        out.push_back(d);
    }
    return !out.empty();
}

inline bool loadMocapCsv(const std::string& path, std::vector<GtPose>& out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string line;
    std::getline(ifs, line);
    const std::vector<std::string> needed = {
        "timestamp", "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
        "pose.orientation.w"};
    auto idx = parseHeader(line, needed);
    for (const auto& n : needed) if (idx[n] < 0) return false;

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto c = splitCsvRow(line);
        if (static_cast<int>(c.size()) <= idx["pose.orientation.w"]) continue;
        GtPose g;
        g.timestamp = std::stod(c[idx["timestamp"]]);
        g.position  = Eigen::Vector3d(std::stod(c[idx["pose.position.x"]]),
                                      std::stod(c[idx["pose.position.y"]]),
                                      std::stod(c[idx["pose.position.z"]]));
        g.orientation = Eigen::Quaterniond(std::stod(c[idx["pose.orientation.w"]]),
                                           std::stod(c[idx["pose.orientation.x"]]),
                                           std::stod(c[idx["pose.orientation.y"]]),
                                           std::stod(c[idx["pose.orientation.z"]]));
        out.push_back(g);
    }
    return !out.empty();
}

inline bool loadLeicaCsv(const std::string& path, std::vector<GtPose>& out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string line;
    std::getline(ifs, line);
    auto idx = parseHeader(line, {"timestamp", "x", "y", "z"});
    for (const char* n : {"timestamp", "x", "y", "z"})
        if (idx[std::string(n)] < 0) return false;

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto c = splitCsvRow(line);
        if (c.size() < 4) continue;
        GtPose g;
        g.timestamp = std::stod(c[idx["timestamp"]]);
        g.position  = Eigen::Vector3d(std::stod(c[idx["x"]]), std::stod(c[idx["y"]]),
                                      std::stod(c[idx["z"]]));
        g.orientation = Eigen::Quaterniond(2.0, 0.0, 0.0, 0.0);
        out.push_back(g);
    }
    return !out.empty();
}

inline bool loadAnchorFile(const std::string& path, std::vector<double>& out, int expected) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[kims] cannot open anchor file: " << path << "\n"; return false; }
    out.clear();
    std::string line;
    while (std::getline(ifs, line) && static_cast<int>(out.size()) < 3 * expected) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double x, y, z;
        if (ss >> x >> y >> z) { out.push_back(x); out.push_back(y); out.push_back(z); }
    }
    return static_cast<int>(out.size()) == 3 * expected;
}


inline bool loadSession(const std::map<std::string, std::string>& a, Session& s) {
    const std::string dataset = getArg<std::string>(a, "dataset", "iului");

    if (dataset == "iului") {
        s.num_tags    = 2;
        s.num_anchors = getArg<int>(a, "num_anchors", 4);
        s.anchors.assign(3 * s.num_anchors, 0.0);
        for (int i = 0; i < s.num_anchors; ++i) {
            const std::string p = "anchor" + std::to_string(i) + "_";
            s.anchors[3 * i + 0] = getArg<double>(a, p + "x", 0.0);
            s.anchors[3 * i + 1] = getArg<double>(a, p + "y", 0.0);
            s.anchors[3 * i + 2] = getArg<double>(a, p + "z", 0.0);
        }
        const std::string dir = getArg<std::string>(a, "data_dir", "");
        if (dir.empty()) {
            std::cerr << "[kims] data_dir is required\n"; return false;
        }
        std::vector<int> tag_ids = {getArg<int>(a, "tag0_id", 8),
                                    getArg<int>(a, "tag1_id", 9)};
        std::vector<int> anchor_ids;
        for (int i = 0; i < s.num_anchors; ++i) anchor_ids.push_back(i);
        if (!loadImuCsv(dir + "/imu.csv", s.imu)) return false;
        if (!loadUwbCsv(dir + "/uwb_range.csv", tag_ids, anchor_ids, s.uwb)) return false;
        loadMocapCsv(dir + "/mocap.csv", s.gt);
    } else if (dataset == "miluv" || dataset == "ntu") {
        const std::string dir = getArg<std::string>(a, "data_dir", "");
        const std::string anchor_file = getArg<std::string>(a, "anchor_file", "");
        if (dir.empty() || anchor_file.empty()) {
            std::cerr << "[kims] data_dir and anchor_file are required\n"; return false;
        }
        std::vector<int> tag_ids, anchor_ids;
        std::string imu_csv, uwb_csv;

        if (dataset == "miluv") {
            s.num_tags = 2;
            s.num_anchors = 6;
            for (int i = 0; i < 6; ++i) anchor_ids.push_back(i);
            tag_ids = {getArg<int>(a, "tag0_id", 10), getArg<int>(a, "tag1_id", 11)};
            imu_csv = dir + "/imu_px4.csv";
            uwb_csv = dir + "/uwb_range.csv";
        } else {
            s.num_tags = 4;
            s.num_anchors = 3;
            for (int i = 0; i < 4; ++i)
                tag_ids.push_back(getArg<int>(a, "tag" + std::to_string(i) + "_id", 2000 + i));
            for (int i = 0; i < 3; ++i)
                anchor_ids.push_back(getArg<int>(a, "anchor" + std::to_string(i) + "_id", 100 + i));
            imu_csv = dir + "/imu.csv";
            uwb_csv = dir + "/uwb_range.csv";
        }
        if (!loadAnchorFile(anchor_file, s.anchors, s.num_anchors)) return false;
        if (!loadImuCsv(imu_csv, s.imu)) return false;
        if (!loadUwbCsv(uwb_csv, tag_ids, anchor_ids, s.uwb)) return false;
        if (dataset == "miluv") loadMocapCsv(dir + "/mocap.csv", s.gt);
        else                    loadLeicaCsv(dir + "/leica.csv", s.gt);
    } else {
        std::cerr << "[kims] unknown dataset: " << dataset << "\n";
        return false;
    }

    s.tag_offsets.assign(3 * s.num_tags, 0.0);
    for (int t = 0; t < s.num_tags; ++t) {
        const std::string p = "tag" + std::to_string(t) + "_offset_";
        s.tag_offsets[3 * t + 0] = getArg<double>(a, p + "x", 0.0);
        s.tag_offsets[3 * t + 1] = getArg<double>(a, p + "y", 0.0);
        s.tag_offsets[3 * t + 2] = getArg<double>(a, p + "z", 0.0);
    }
    s.init_p = Eigen::Vector3d(getArg<double>(a, "init_x", 0.0),
                               getArg<double>(a, "init_y", 0.0),
                               getArg<double>(a, "init_z", 0.0));

    std::sort(s.imu.begin(), s.imu.end(),
              [](const ImuData& x, const ImuData& y) { return x.timestamp < y.timestamp; });
    std::sort(s.uwb.begin(), s.uwb.end(),
              [](const UwbRange& x, const UwbRange& y) { return x.timestamp < y.timestamp; });

    std::printf("[kims] %s: %zu IMU, %zu UWB events, %d tags x %d anchors\n",
                dataset.c_str(), s.imu.size(), s.uwb.size(), s.num_tags, s.num_anchors);
    return true;
}

}
