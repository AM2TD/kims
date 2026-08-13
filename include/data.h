#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace kims {

struct ImuData {
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};

struct UwbRange {
    double timestamp;
    int    tag_idx;
    int    anc_idx;
    double range;
    double std_dev;
};

struct GtPose {
    double timestamp;
    Eigen::Vector3d    position;
    Eigen::Quaterniond orientation;
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

inline ImuData interpolateImu(const std::vector<ImuData>& data,
                              double ts, size_t& hint) {
    if (data.empty()) return ImuData{ts, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    if (ts <= data.front().timestamp) return data.front();
    if (ts >= data.back().timestamp)  return data.back();

    while (hint + 1 < data.size() && data[hint + 1].timestamp < ts) ++hint;
    while (hint > 0 && data[hint].timestamp > ts) --hint;

    const ImuData& a = data[hint];
    const ImuData& b = data[std::min(hint + 1, data.size() - 1)];
    const double dt = b.timestamp - a.timestamp;
    if (dt <= 0.0) return a;

    const double w = (ts - a.timestamp) / dt;
    ImuData out;
    out.timestamp = ts;
    out.acc = a.acc + w * (b.acc - a.acc);
    out.gyr = a.gyr + w * (b.gyr - a.gyr);
    return out;
}

class LatestRanges {
public:
    LatestRanges(int num_tags, int num_anchors, double std_default = 0.1)
        : num_tags_(num_tags), num_anc_(num_anchors),
          ranges_(num_tags * num_anchors, -1.0),
          stds_  (num_tags * num_anchors, std_default) {}

    void update(const UwbRange& m) {
        if (m.tag_idx < 0 || m.tag_idx >= num_tags_) return;
        if (m.anc_idx < 0 || m.anc_idx >= num_anc_)  return;
        const int flat = m.tag_idx * num_anc_ + m.anc_idx;
        ranges_[flat] = m.range;
        stds_  [flat] = m.std_dev;
    }

    bool allValid() const {
        for (double r : ranges_) if (r < 0.0) return false;
        return true;
    }
    int validCount() const {
        int c = 0;
        for (double r : ranges_) if (r > 0.0) ++c;
        return c;
    }
    int size()   const { return static_cast<int>(ranges_.size()); }
    int numAnc() const { return num_anc_; }
    int numTags() const { return num_tags_; }

    const double* data()     const { return ranges_.data(); }
    const double* stdsData() const { return stds_.data(); }
    void copyTo(double* dst)     const { std::copy(ranges_.begin(), ranges_.end(), dst); }
    void copyStdsTo(double* dst) const { std::copy(stds_.begin(),   stds_.end(),   dst); }

private:
    int num_tags_, num_anc_;
    std::vector<double> ranges_, stds_;
};

inline std::vector<std::string> splitCsvRow(const std::string& line) {
    std::vector<std::string> cols;
    std::string field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"')                      in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { cols.push_back(field); field.clear(); }
        else                               field += c;
    }
    cols.push_back(field);
    return cols;
}

inline std::map<std::string, int> parseHeader(const std::string& header,
                                              const std::vector<std::string>& names) {
    const auto cols = splitCsvRow(header);
    std::map<std::string, int> idx;
    for (const auto& n : names) {
        idx[n] = -1;
        for (size_t i = 0; i < cols.size(); ++i)
            if (cols[i] == n) { idx[n] = static_cast<int>(i); break; }
    }
    return idx;
}

}
