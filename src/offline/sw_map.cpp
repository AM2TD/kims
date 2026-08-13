#include "csv_io.h"
#include "node_util.h"

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <deque>
#include <map>
#include <string>
#include <cmath>
#include <chrono>


template <int NT, int NA>
struct UWBRangingCost {
    static constexpr int NR = NT * NA;

    UWBRangingCost(const double* ranges,
                   const Eigen::Matrix<double, NA, 3>& anchors,
                   const std::vector<Eigen::Vector3d>& tag_off)
        : anchors_(anchors) {
        for (int i = 0; i < NR; ++i) ranges_[i] = ranges[i];
        for (int t = 0; t < NT; ++t) tag_off_[t] = tag_off[t];
    }

    template <typename T>
    bool operator()(const T* const position, const T* const rotation,
                    T* residuals) const {
        Eigen::Matrix<T, 3, 3> R;
        R << rotation[0], rotation[1], rotation[2],
             rotation[3], rotation[4], rotation[5],
             rotation[6], rotation[7], rotation[8];

        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);

        Eigen::Matrix<T, 3, 1> tags[NT];
        for (int t = 0; t < NT; ++t) tags[t] = p + R * tag_off_[t].template cast<T>();

        for (int ti = 0; ti < NT; ++ti) {
            for (int ai = 0; ai < NA; ++ai) {
                int ri = ti * NA + ai;
                Eigen::Matrix<T, 3, 1> anc = anchors_.row(ai).transpose().template cast<T>();
                residuals[ri] = (tags[ti] - anc).norm() - T(ranges_[ri]);
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(const double* ranges,
                                       const Eigen::Matrix<double, NA, 3>& anchors,
                                       const std::vector<Eigen::Vector3d>& tag_off) {
        return new ceres::AutoDiffCostFunction<UWBRangingCost<NT, NA>, NR, 3, 9>(
            new UWBRangingCost<NT, NA>(ranges, anchors, tag_off));
    }

private:
    double ranges_[NR];
    Eigen::Matrix<double, NA, 3> anchors_;
    Eigen::Vector3d tag_off_[NT];
};

struct IMUDynamicsCost {
    IMUDynamicsCost(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double dt)
        : acc_(acc), gyr_(gyr), dt_(dt), rot_w_(5.0) {}

    template <typename T>
    bool operator()(const T* const pos_i, const T* const vel_i, const T* const rot_i,
                    const T* const pos_j, const T* const vel_j, const T* const rot_j,
                    T* residuals) const {
        Eigen::Matrix<T, 3, 3> R_i;
        R_i << rot_i[0], rot_i[1], rot_i[2],
               rot_i[3], rot_i[4], rot_i[5],
               rot_i[6], rot_i[7], rot_i[8];

        Eigen::Matrix<T, 3, 1> a_world = R_i * acc_.cast<T>();
        a_world[2] -= T(9.81);

        Eigen::Matrix<T, 3, 1> vi(vel_i[0], vel_i[1], vel_i[2]);
        Eigen::Matrix<T, 3, 1> pi(pos_i[0], pos_i[1], pos_i[2]);

        Eigen::Matrix<T, 3, 1> pred_pj = pi + vi * T(dt_) + T(0.5) * a_world * T(dt_ * dt_);
        Eigen::Matrix<T, 3, 1> pred_vj = vi + a_world * T(dt_);

        residuals[0] = pos_j[0] - pred_pj[0];
        residuals[1] = pos_j[1] - pred_pj[1];
        residuals[2] = pos_j[2] - pred_pj[2];
        residuals[3] = vel_j[0] - pred_vj[0];
        residuals[4] = vel_j[1] - pred_vj[1];
        residuals[5] = vel_j[2] - pred_vj[2];

        Eigen::Matrix<T, 3, 1> omega = gyr_.cast<T>() * T(dt_);
        T angle = omega.norm();
        Eigen::Matrix<T, 3, 3> dR = Eigen::Matrix<T, 3, 3>::Identity();
        if (angle > T(1e-6)) {
            Eigen::Matrix<T, 3, 1> axis = omega / angle;
            T c = ceres::cos(angle), s = ceres::sin(angle);
            Eigen::Matrix<T, 3, 3> K;
            K << T(0), -axis(2),  axis(1),
                 axis(2), T(0), -axis(0),
                -axis(1),  axis(0), T(0);
            dR = Eigen::Matrix<T, 3, 3>::Identity() + s * K + (T(1) - c) * K * K;
        }
        Eigen::Matrix<T, 3, 3> pred_Rj = R_i * dR;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                residuals[6 + i*3 + j] = T(rot_w_) * (rot_j[i*3+j] - pred_Rj(i, j));

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& acc,
                                        const Eigen::Vector3d& gyr, double dt) {
        return new ceres::AutoDiffCostFunction<IMUDynamicsCost, 15, 3, 3, 9, 3, 3, 9>(
            new IMUDynamicsCost(acc, gyr, dt));
    }

private:
    Eigen::Vector3d acc_, gyr_;
    double dt_, rot_w_;
};

struct RotationOrthogonalityCost {
    template <typename T>
    bool operator()(const T* const rotation, T* residuals) const {
        Eigen::Matrix<T, 3, 3> R;
        R << rotation[0], rotation[1], rotation[2],
             rotation[3], rotation[4], rotation[5],
             rotation[6], rotation[7], rotation[8];
        Eigen::Matrix<T, 3, 3> I = R.transpose() * R;
        residuals[0] = I(0,0) - T(1); residuals[1] = I(1,1) - T(1); residuals[2] = I(2,2) - T(1);
        residuals[3] = I(0,1);        residuals[4] = I(0,2);        residuals[5] = I(1,2);
        return true;
    }
    static ceres::CostFunction* Create() {
        return new ceres::AutoDiffCostFunction<RotationOrthogonalityCost, 6, 9>(
            new RotationOrthogonalityCost());
    }
};

struct WindowNode {
    double timestamp;
    double pos[3];
    double vel[3];
    double rot[9];
    double ranges[12];
    ImuData imu;
    bool has_uwb;
};

static void rotToArray(const Eigen::Matrix3d& R, double* arr) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            arr[i*3+j] = R(i, j);
}

template <int NT, int NA>
static int runSwMap(const std::vector<ImuData>& imu_data,
                    const std::vector<UwbRange>& uwb_raw,
                    const std::vector<int>& tag_ids,
                    const std::vector<int>& anchor_ids,
                    const Eigen::Matrix<double, NA, 3>& anchors,
                    const std::vector<Eigen::Vector3d>& tag_off,
                    int win_size, const Eigen::Vector3d& init_p,
                    bool z_down, const std::string& output_file) {
    const double init_x = init_p.x(), init_y = init_p.y(), init_z = init_p.z();
    std::ofstream ofs(output_file);
    if (!ofs) { std::cerr << "[SW-MAP] Cannot open output: " << output_file << "\n"; return 1; }
    ofs << "# time x y z qx qy qz qw\n";
    ofs << std::fixed; ofs.precision(9);

    LatestRanges latest(tag_ids, anchor_ids);
    std::deque<WindowNode> window;
    size_t imu_hint = 0;
    int solve_count = 0;
    std::vector<double> solve_times_vec;

    WindowNode first_node;
    first_node.timestamp = imu_data.empty() ? 0.0 : imu_data[0].timestamp;
    first_node.pos[0] = init_x; first_node.pos[1] = init_y; first_node.pos[2] = init_z;
    first_node.vel[0] = first_node.vel[1] = first_node.vel[2] = 0.0;
    rotToArray(z_down ? Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix()
                      : Eigen::Matrix3d::Identity(),
               first_node.rot);
    std::fill(first_node.ranges, first_node.ranges + 12, -1.0);
    first_node.has_uwb = false;
    window.push_back(first_node);

    for (const auto& uwb : uwb_raw) {
        latest.update(uwb);
        if (!latest.allValid()) continue;

        WindowNode node;
        node.timestamp = uwb.timestamp;
        node.has_uwb   = true;
        latest.copyTo(node.ranges);

        const WindowNode& prev = window.back();
        for (int k = 0; k < 3; ++k) node.pos[k] = prev.pos[k];
        for (int k = 0; k < 3; ++k) node.vel[k] = prev.vel[k];
        for (int k = 0; k < 9; ++k) node.rot[k] = prev.rot[k];
        node.imu = interpolateImu(imu_data, node.timestamp, imu_hint);

        window.push_back(node);
        if ((int)window.size() < win_size + 1) continue;

        auto t0s = std::chrono::high_resolution_clock::now();
        ceres::Problem problem;

        for (int w = 0; w < (int)window.size(); ++w) {
            WindowNode& nd = window[w];

            if (nd.has_uwb) {
                problem.AddResidualBlock(
                    UWBRangingCost<NT, NA>::Create(nd.ranges, anchors, tag_off),
                    nullptr, nd.pos, nd.rot);
            }
            problem.AddResidualBlock(RotationOrthogonalityCost::Create(), nullptr, nd.rot);

            if (w + 1 < (int)window.size()) {
                double dt = window[w+1].timestamp - nd.timestamp;
                if (dt > 0.0 && dt < 1.0) {
                    problem.AddResidualBlock(
                        IMUDynamicsCost::Create(nd.imu.acc, nd.imu.gyr, dt),
                        nullptr,
                        nd.pos, nd.vel, nd.rot,
                        window[w+1].pos, window[w+1].vel, window[w+1].rot);
                }
            }

            if (w == 0) {
                problem.AddParameterBlock(nd.pos, 3);
                problem.AddParameterBlock(nd.vel, 3);
                problem.AddParameterBlock(nd.rot, 9);

                problem.SetParameterBlockConstant(nd.pos);
                problem.SetParameterBlockConstant(nd.vel);
                problem.SetParameterBlockConstant(nd.rot);
            }
        }

        ceres::Solver::Options opts;
        opts.linear_solver_type           = ceres::DENSE_SCHUR;
        opts.max_num_iterations           = 20;
        opts.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        solve_times_vec.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0s).count());

        int out_idx = static_cast<int>(window.size()) - 2;
        WindowNode& out = window[out_idx];
        Eigen::Matrix3d R_out;
        R_out << out.rot[0], out.rot[1], out.rot[2],
                 out.rot[3], out.rot[4], out.rot[5],
                 out.rot[6], out.rot[7], out.rot[8];
        Eigen::Quaterniond q(R_out);
        q.normalize();
        ofs << out.timestamp << " "
            << out.pos[0] << " " << out.pos[1] << " " << out.pos[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        ++solve_count;
        window.pop_front();
    }

    ofs.close();

    {
        size_t N = solve_times_vec.size();
        double sum = 0.0, sum2 = 0.0, max_ms = 0.0;
        for (double ms : solve_times_vec) { sum += ms; sum2 += ms*ms; if (ms > max_ms) max_ms = ms; }
        double mean_ms = N > 0 ? sum / N : 0.0;
        double std_ms  = N > 1 ? std::sqrt((sum2 - sum*sum/N) / (N-1)) : 0.0;
        std::string timing_file = output_file;
        auto tpos = timing_file.rfind("_pose.txt");
        if (tpos != std::string::npos) timing_file.replace(tpos, 9, "_timing.txt");
        else timing_file += ".timing.txt";
        std::ofstream tf(timing_file);
        tf << std::fixed; tf.precision(6);
        for (double ms : solve_times_vec) tf << ms << "\n";
        printf("[SW-MAP] Timing: mean=%.3f ms  std=%.3f ms  max=%.3f ms  N=%zu  saved to %s\n",
               mean_ms, std_ms, max_ms, N, timing_file.c_str());
    }

    std::cout << "[SW-MAP] Done. Solves=" << solve_count
              << "  output=" << output_file << "\n";
    return 0;
}

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);
    const std::string dataset = getArg<std::string>(args, "dataset", std::string("ntu"));
    const bool is_ntu = (dataset != "miluv");

    std::string data_dir    = getArg<std::string>(args, "data_dir", std::string(""));
    std::string anchor_file = getArg<std::string>(args, "anchor_file", std::string(""));
    std::string output_file = getArg<std::string>(args, "output_file", std::string("sw_map_pose.txt"));
    std::string imu_file    = getArg<std::string>(args, "imu_file",
                                  data_dir + (is_ntu ? "/imu.csv" : "/imu_px4.csv"));
    std::string uwb_file    = getArg<std::string>(args, "uwb_file", data_dir + "/uwb_range.csv");

    const int NT = is_ntu ? 4 : 2;
    const int NA = is_ntu ? 3 : 6;

    std::vector<int> tag_ids, anchor_ids;
    std::vector<Eigen::Vector3d> tag_off;
    for (int t = 0; t < NT; ++t) {
        const std::string k = "tag" + std::to_string(t);
        tag_ids.push_back(getArg<int>(args, k + "_id", is_ntu ? 2000 : 10));
        tag_off.push_back(Eigen::Vector3d(getArg<double>(args, k + "_offset_x", 0.0),
                                          getArg<double>(args, k + "_offset_y", 0.0),
                                          getArg<double>(args, k + "_offset_z", 0.0)));
    }
    if (is_ntu) for (int a = 0; a < NA; ++a)
        anchor_ids.push_back(getArg<int>(args, "anchor" + std::to_string(a) + "_id", 100 + a));
    else        for (int a = 0; a < NA; ++a) anchor_ids.push_back(a);

    int    win_size = getArg<int>   (args, "window_size", is_ntu ? 15 : 15);
    double init_x   = getArg<double>(args, "init_x", 0.0);
    double init_y   = getArg<double>(args, "init_y", 0.0);
    double init_z   = getArg<double>(args, "init_z", 0.0);

    std::vector<ImuData>  imu_data;
    std::vector<UwbRange> uwb_raw;
    if (!loadImu(imu_file, imu_data)) return 1;
    if (!loadUwbRange(uwb_file, tag_ids, anchor_ids, uwb_raw)) return 1;

    std::vector<double> h_anchor(3 * NA);
    if (!loadAnchors(anchor_file, h_anchor.data(), NA)) return 1;

    if (is_ntu) {
        Eigen::Matrix<double, 3, 3> anchors;
        for (int i = 0; i < 3; ++i)
            anchors.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
        return runSwMap<4, 3>(imu_data, uwb_raw, tag_ids, anchor_ids, anchors, tag_off,
                              win_size, Eigen::Vector3d(init_x, init_y, init_z),
                              true, output_file);
    }
    Eigen::Matrix<double, 6, 3> anchors;
    for (int i = 0; i < 6; ++i)
        anchors.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
    return runSwMap<2, 6>(imu_data, uwb_raw, tag_ids, anchor_ids, anchors, tag_off,
                          win_size, Eigen::Vector3d(init_x, init_y, init_z),
                          false, output_file);
}
