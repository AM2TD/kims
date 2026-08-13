// @file node_util.h
// @brief Argument parsing, timing output and the skew operator, shared by the
//        baseline nodes. Estimator math stays in each node.

#pragma once

#include <Eigen/Core>

#include <fstream>
#include <map>
#include <string>

inline std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> m;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto pos = s.find(":=");
        if (pos != std::string::npos) m[s.substr(0, pos)] = s.substr(pos + 2);
    }
    return m;
}

template<typename T>
inline T getArg(const std::map<std::string, std::string>&, const std::string&, T def) {
    return def;
}
template<> inline std::string getArg(const std::map<std::string, std::string>& m,
                                     const std::string& k, std::string def) {
    auto it = m.find(k); return it != m.end() ? it->second : def;
}
template<> inline int getArg(const std::map<std::string, std::string>& m,
                             const std::string& k, int def) {
    auto it = m.find(k); return it != m.end() ? std::stoi(it->second) : def;
}
template<> inline double getArg(const std::map<std::string, std::string>& m,
                                const std::string& k, double def) {
    auto it = m.find(k); return it != m.end() ? std::stod(it->second) : def;
}

inline std::string timingPath(const std::string& output_file) {
    std::string p = output_file;
    auto pos = p.rfind("_pose.txt");
    if (pos != std::string::npos) p.replace(pos, 9, "_timing.txt");
    else p += ".timing.txt";
    return p;
}

inline void writeTimingFile(const std::string& output_file, double per_step, size_t n) {
    std::ofstream tf(timingPath(output_file));
    tf << std::fixed;
    tf.precision(6);
    for (size_t k = 0; k < n; ++k) tf << per_step << "\n";
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return m;
}
