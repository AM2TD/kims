#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <map>

struct ImuData {
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};

struct UwbRange {
    double timestamp;
    int    from_id;
    int    to_id;
    double range;
    double std_dev;
};

struct MocapData {
    double timestamp;
    Eigen::Vector3d    position;
    Eigen::Quaterniond orientation;
};

struct LeicaData {
    double timestamp;
    Eigen::Vector3d position;
};

class STATE {
public:
    Eigen::Vector3d p;
    Eigen::Matrix3d R;
    Eigen::Vector3d v;

    STATE() {}
    explicit STATE(const double* x) {
        p = Eigen::Vector3d(x[0], x[1], x[2]);
        v = Eigen::Vector3d(x[3], x[4], x[5]);
        R << x[6],  x[7],  x[8],
             x[9],  x[10], x[11],
             x[12], x[13], x[14];
    }
};

inline std::vector<std::string> splitCsvRow(const std::string& line) {
    std::vector<std::string> cols;
    std::string field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            cols.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    cols.push_back(field);
    return cols;
}

inline std::map<std::string, int>
parseHeader(const std::string& header_line,
            const std::vector<std::string>& names) {
    std::map<std::string, int> idx;
    auto cols = splitCsvRow(header_line);
    for (auto& tok : cols) {
        tok.erase(0, tok.find_first_not_of(" \t\r\n"));
        tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
    }
    for (const auto& n : names) {
        auto it = std::find(cols.begin(), cols.end(), n);
        idx[n]  = (it != cols.end()) ? static_cast<int>(it - cols.begin()) : -1;
    }
    return idx;
}

inline bool loadImu(const std::string& path, std::vector<ImuData>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[io] Cannot open IMU file: " << path << "\n"; return false; }

    std::string line;
    std::getline(ifs, line);

    const std::vector<std::string> needed = {
        "timestamp",
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"
    };
    auto idx = parseHeader(line, needed);

    for (const auto& n : needed) {
        if (idx[n] < 0) {
            std::cerr << "[io] IMU CSV missing column: " << n << "\n";
            return false;
        }
    }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = splitCsvRow(line);

        int max_idx = static_cast<int>(cols.size()) - 1;
        auto safe = [&](int i) -> double {
            return (i >= 0 && i <= max_idx) ? std::stod(cols[i]) : 0.0;
        };

        ImuData d;
        d.timestamp = safe(idx["timestamp"]);
        d.gyr.x()   = safe(idx["angular_velocity.x"]);
        d.gyr.y()   = safe(idx["angular_velocity.y"]);
        d.gyr.z()   = safe(idx["angular_velocity.z"]);
        d.acc.x()   = safe(idx["linear_acceleration.x"]);
        d.acc.y()   = safe(idx["linear_acceleration.y"]);
        d.acc.z()   = safe(idx["linear_acceleration.z"]);
        out.push_back(d);
    }
    std::cout << "[io] Loaded " << out.size() << " IMU samples from " << path << "\n";
    return !out.empty();
}

inline bool loadUwbRange(const std::string& path,
                         const std::vector<int>& tag_ids,
                         const std::vector<int>& anchor_ids,
                         std::vector<UwbRange>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[io] Cannot open UWB file: " << path << "\n"; return false; }

    std::string line;
    std::getline(ifs, line);

    const std::vector<std::string> needed = {"timestamp", "range", "from_id", "to_id", "std"};
    auto idx = parseHeader(line, needed);
    for (const auto& n : {"timestamp", "range", "from_id", "to_id"}) {
        if (idx[std::string(n)] < 0) {
            std::cerr << "[io] UWB CSV missing column: " << n << "\n";
            return false;
        }
    }
    const bool has_std = (idx["std"] >= 0);
    if (!has_std)
        std::cerr << "[io] UWB CSV: 'std' column not found — using default 0.1 m\n";

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = splitCsvRow(line);

        int max_idx = static_cast<int>(cols.size()) - 1;
        if (max_idx < 3) continue;

        int from_id = std::stoi(cols[idx["from_id"]]);
        int to_id   = std::stoi(cols[idx["to_id"]]);

        if (std::find(tag_ids.begin(), tag_ids.end(), from_id) == tag_ids.end()) continue;
        if (std::find(anchor_ids.begin(), anchor_ids.end(), to_id) == anchor_ids.end()) continue;

        UwbRange d;
        d.timestamp = std::stod(cols[idx["timestamp"]]);
        d.range     = std::stod(cols[idx["range"]]);
        d.from_id   = from_id;
        d.to_id     = to_id;
        d.std_dev   = has_std ? std::stod(cols[idx["std"]]) : 0.1;
        if (d.std_dev < 1e-4) d.std_dev = 1e-4;
        out.push_back(d);
    }
    std::cout << "[io] Loaded " << out.size() << " UWB ranges from " << path << "\n";
    return !out.empty();
}

inline bool loadMocap(const std::string& path, std::vector<MocapData>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[io] Cannot open mocap file: " << path << "\n"; return false; }

    std::string line;
    std::getline(ifs, line);

    const std::vector<std::string> needed = {
        "timestamp",
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y",
        "pose.orientation.z", "pose.orientation.w"
    };
    auto idx = parseHeader(line, needed);
    for (const auto& n : needed) {
        if (idx[n] < 0) {
            std::cerr << "[io] mocap CSV missing column: " << n << "\n";
            return false;
        }
    }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = splitCsvRow(line);

        int max_idx = static_cast<int>(cols.size()) - 1;
        auto safe = [&](int i) -> double {
            return (i >= 0 && i <= max_idx) ? std::stod(cols[i]) : 0.0;
        };

        MocapData d;
        d.timestamp          = safe(idx["timestamp"]);
        d.position.x()       = safe(idx["pose.position.x"]);
        d.position.y()       = safe(idx["pose.position.y"]);
        d.position.z()       = safe(idx["pose.position.z"]);
        d.orientation.x()    = safe(idx["pose.orientation.x"]);
        d.orientation.y()    = safe(idx["pose.orientation.y"]);
        d.orientation.z()    = safe(idx["pose.orientation.z"]);
        d.orientation.w()    = safe(idx["pose.orientation.w"]);
        out.push_back(d);
    }
    std::cout << "[io] Loaded " << out.size() << " mocap samples from " << path << "\n";
    return !out.empty();
}

inline bool loadLeica(const std::string& path, std::vector<LeicaData>& out) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[io] Cannot open leica file: " << path << "\n"; return false; }

    std::string line;
    std::getline(ifs, line);

    const std::vector<std::string> needed = {
        "timestamp", "pose.position.x", "pose.position.y", "pose.position.z"};
    auto idx = parseHeader(line, needed);
    for (const auto& n : needed) {
        if (idx[n] < 0) {
            std::cerr << "[io] leica CSV missing column: " << n << "\n";
            return false;
        }
    }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = splitCsvRow(line);
        int max_idx = static_cast<int>(cols.size()) - 1;
        auto safe = [&](int i) -> double {
            return (i >= 0 && i <= max_idx) ? std::stod(cols[i]) : 0.0;
        };
        LeicaData d;
        d.timestamp    = safe(idx["timestamp"]);
        d.position.x() = safe(idx["pose.position.x"]);
        d.position.y() = safe(idx["pose.position.y"]);
        d.position.z() = safe(idx["pose.position.z"]);
        out.push_back(d);
    }
    std::cout << "[io] Loaded " << out.size() << " leica samples from " << path << "\n";
    return !out.empty();
}

inline bool loadAnchors(const std::string& path,
                        double* anchor, int num_anchors = 6) {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[io] Cannot open anchor file: " << path << "\n"; return false; }

    std::string line;
    int count = 0;
    while (std::getline(ifs, line) && count < num_anchors) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ss >> anchor[3*count + 0]
           >> anchor[3*count + 1]
           >> anchor[3*count + 2];
        ++count;
    }
    if (count < num_anchors) {
        std::cerr << "[io] Anchor file only has " << count
                  << " entries, expected " << num_anchors << "\n";
        return false;
    }
    return true;
}

inline ImuData interpolateImu(const std::vector<ImuData>& data,
                               double ts, size_t& hint) {
    while (hint + 1 < data.size() && data[hint + 1].timestamp < ts) ++hint;

    if (hint + 1 >= data.size()) return data.back();

    const ImuData& a = data[hint];
    const ImuData& b = data[hint + 1];
    double dt = b.timestamp - a.timestamp;
    if (dt < 1e-9) return a;

    double w = (ts - a.timestamp) / dt;
    w = std::max(0.0, std::min(1.0, w));

    ImuData out;
    out.timestamp = ts;
    out.acc = a.acc * (1.0 - w) + b.acc * w;
    out.gyr = a.gyr * (1.0 - w) + b.gyr * w;
    return out;
}

// Most recent range per (tag, anchor) pair, flattened tag-major as
// tag_idx * num_anchors + anc_idx.
class LatestRanges {
public:
    LatestRanges(const std::vector<int>& tag_ids,
                 const std::vector<int>& anchor_ids)
        : tag_ids_(tag_ids), anchor_ids_(anchor_ids),
          num_anc_(static_cast<int>(anchor_ids.size())),
          total_(static_cast<int>(tag_ids.size() * anchor_ids.size())) {
        ranges_.assign(total_, -1.0);
        stds_.assign  (total_,  0.1);
    }

    void update(const UwbRange& m) {
        auto t_it = std::find(tag_ids_.begin(),    tag_ids_.end(),    m.from_id);
        auto a_it = std::find(anchor_ids_.begin(), anchor_ids_.end(), m.to_id);
        if (t_it == tag_ids_.end() || a_it == anchor_ids_.end()) return;

        int tag_idx = static_cast<int>(t_it - tag_ids_.begin());
        int anc_idx = static_cast<int>(a_it - anchor_ids_.begin());
        int flat    = tag_idx * num_anc_ + anc_idx;

        ranges_[flat] = m.range;
        stds_  [flat] = m.std_dev;
    }

    bool allValid() const {
        for (double r : ranges_) if (r < 0.0) return false;
        return true;
    }

    int validCount() const {
        int cnt = 0;
        for (double r : ranges_) if (r > 0.0) ++cnt;
        return cnt;
    }

    const double* data()     const { return ranges_.data(); }
    const double* stdsData() const { return stds_.data(); }

    void copyTo(double* dst) const {
        std::copy(ranges_.begin(), ranges_.end(), dst);
    }
    void copyStdsTo(double* dst) const {
        std::copy(stds_.begin(), stds_.end(), dst);
    }

private:
    std::vector<int>    tag_ids_, anchor_ids_;
    int                 num_anc_, total_;
    std::vector<double> ranges_, stds_;
};
