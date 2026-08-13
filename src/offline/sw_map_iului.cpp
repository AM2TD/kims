#include "csv_io.h"
#include "node_util.h"

static std::vector<int> anchor_idx_all(int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = i;
    return v;
}

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


static Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi) {
    double angle = phi.norm();
    if (angle < 1e-7) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d ax = phi / angle;
    double c = std::cos(angle), s = std::sin(angle);
    Eigen::Matrix3d K;
    K <<      0, -ax(2),  ax(1),
          ax(2),      0, -ax(0),
         -ax(1),  ax(0),      0;
    return Eigen::Matrix3d::Identity() + s * K + (1.0 - c) * K * K;
}

struct ImuPreint {
    Eigen::Matrix3d delta_R;
    Eigen::Vector3d delta_v;
    Eigen::Vector3d delta_p;
    double          dt;

    ImuPreint() : delta_R(Eigen::Matrix3d::Identity()),
                  delta_v(Eigen::Vector3d::Zero()),
                  delta_p(Eigen::Vector3d::Zero()), dt(0.0) {}

    void integrate(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double step) {
        delta_p += delta_v * step + 0.5 * (delta_R * acc) * step * step;
        delta_v += delta_R * acc * step;
        delta_R  = delta_R * expSO3(gyr * step);
        dt      += step;
    }
};

static ImuPreint preintegrate(const std::vector<ImuData>& imu,
                               double t0, double t1, size_t& hint) {
    ImuPreint pre;
    if (t1 <= t0 || imu.empty()) return pre;

    while (hint + 1 < imu.size() && imu[hint + 1].timestamp <= t0)
        ++hint;

    double t_cur = t0;
    for (size_t k = hint; k < imu.size() && t_cur < t1; ++k) {
        double t_next = (k + 1 < imu.size())
                        ? std::min(imu[k + 1].timestamp, t1) : t1;
        double step = t_next - t_cur;
        if (step > 1e-9 && step < 0.5)
            pre.integrate(imu[k].acc, imu[k].gyr, step);
        t_cur = t_next;
        if (t_next < t1) hint = k;
    }
    return pre;
}

struct UWBRangingCost {
    UWBRangingCost(const double* ranges,
                   const Eigen::MatrixXd& anchors,
                   const Eigen::Vector3d& tag0_off,
                   const Eigen::Vector3d& tag1_off,
                   int num_anc)
        : anchors_(anchors), tag0_off_(tag0_off), tag1_off_(tag1_off),
          num_anc_(num_anc) {
        ranges_.assign(ranges, ranges + 2 * num_anc);
    }

    template <typename T>
    bool operator()(const T* const* parameters, T* residuals) const {
        const T* position = parameters[0];
        const T* quat     = parameters[1];

        Eigen::Quaternion<T> q(quat[3], quat[0], quat[1], quat[2]);
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();

        Eigen::Matrix<T, 3, 1> p(position[0], position[1], position[2]);
        Eigen::Matrix<T, 3, 1> t0 = p + R * tag0_off_.cast<T>();
        Eigen::Matrix<T, 3, 1> t1 = p + R * tag1_off_.cast<T>();

        for (int i = 0; i < num_anc_; ++i) {
            Eigen::Matrix<T, 3, 1> anc = anchors_.row(i).transpose().cast<T>();
            residuals[i]            = (ranges_[i] > 0.0)
                ? (t0 - anc).norm() - T(ranges_[i]) : T(0);
            residuals[i + num_anc_] = (ranges_[i + num_anc_] > 0.0)
                ? (t1 - anc).norm() - T(ranges_[i + num_anc_]) : T(0);
        }
        return true;
    }

    static ceres::CostFunction* Create(const double* ranges,
                                       const Eigen::MatrixXd& anchors,
                                       const Eigen::Vector3d& tag0_off,
                                       const Eigen::Vector3d& tag1_off,
                                       int num_anc) {
        auto* cost = new ceres::DynamicAutoDiffCostFunction<UWBRangingCost>(
            new UWBRangingCost(ranges, anchors, tag0_off, tag1_off, num_anc));
        cost->AddParameterBlock(3);
        cost->AddParameterBlock(4);
        cost->SetNumResiduals(2 * num_anc);
        return cost;
    }

private:
    std::vector<double>  ranges_;
    Eigen::MatrixXd      anchors_;
    Eigen::Vector3d      tag0_off_, tag1_off_;
    int                  num_anc_;
};

struct IMUPreIntCost {
    ImuPreint pre_;
    double    rot_w_;

    IMUPreIntCost(const ImuPreint& pre, double rot_w = 6.0)
        : pre_(pre), rot_w_(rot_w) {}

    template <typename T>
    bool operator()(const T* const pos_i, const T* const vel_i, const T* const quat_i,
                    const T* const pos_j, const T* const vel_j, const T* const quat_j,
                    T* residuals) const {
        Eigen::Quaternion<T> qi(quat_i[3], quat_i[0], quat_i[1], quat_i[2]);
        Eigen::Matrix<T,3,3> Ri = qi.toRotationMatrix();

        Eigen::Matrix<T,3,1> pi(pos_i[0], pos_i[1], pos_i[2]);
        Eigen::Matrix<T,3,1> vi(vel_i[0], vel_i[1], vel_i[2]);
        Eigen::Matrix<T,3,1> pj(pos_j[0], pos_j[1], pos_j[2]);
        Eigen::Matrix<T,3,1> vj(vel_j[0], vel_j[1], vel_j[2]);

        T dt = T(pre_.dt);
        Eigen::Matrix<T,3,1> g(T(0), T(0), T(-9.81));

        Eigen::Matrix<T,3,1> r_p =
            Ri.transpose() * (pj - pi - vi * dt - T(0.5) * g * dt * dt)
            - pre_.delta_p.cast<T>();
        residuals[0] = r_p(0);
        residuals[1] = r_p(1);
        residuals[2] = r_p(2);

        Eigen::Matrix<T,3,1> r_v =
            Ri.transpose() * (vj - vi - g * dt) - pre_.delta_v.cast<T>();
        residuals[3] = r_v(0);
        residuals[4] = r_v(1);
        residuals[5] = r_v(2);

        Eigen::Quaternion<T> qj_q(quat_j[3], quat_j[0], quat_j[1], quat_j[2]);
        Eigen::Quaternion<T> delta_q(pre_.delta_R.cast<T>());
        Eigen::Quaternion<T> q_err = delta_q.conjugate() * qi.conjugate() * qj_q;

        residuals[6] = T(rot_w_) * T(2.0) * q_err.x();
        residuals[7] = T(rot_w_) * T(2.0) * q_err.y();
        residuals[8] = T(rot_w_) * T(2.0) * q_err.z();

        return true;
    }

    static ceres::CostFunction* Create(const ImuPreint& pre, double rot_w = 6.0) {
        return new ceres::AutoDiffCostFunction<IMUPreIntCost, 9, 3, 3, 4, 3, 3, 4>(
            new IMUPreIntCost(pre, rot_w));
    }
};

struct WindowNode {
    double timestamp;
    double pos[3];
    double vel[3];
    double quat[4];
    std::vector<double> ranges;
    ImuPreint           preint;
    bool has_uwb;
};

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);    std::string output_file = getArg<std::string>(args, "output_file", std::string("ceres_pose.txt"));
    int    num_anchors = getArg<int>   (args, "num_anchors",   4);
    int    tag0_id     = getArg<int>   (args, "tag0_id",       0);
    int    tag1_id     = getArg<int>   (args, "tag1_id",       1);
    double t0x         = getArg<double>(args, "tag0_offset_x",  0.0);
    double t0z         = getArg<double>(args, "tag0_offset_z",  0.0);
    double t1x         = getArg<double>(args, "tag1_offset_x",  0.0);
    double t1z         = getArg<double>(args, "tag1_offset_z",  0.0);
    double t0y         = getArg<double>(args, "tag0_offset_y",  0.1317);
    double t1y         = getArg<double>(args, "tag1_offset_y", -0.1227);

    double anc_raw[12];
    anc_raw[0]  = getArg<double>(args, "anchor0_x", -2.9449);
    anc_raw[1]  = getArg<double>(args, "anchor0_y", -3.0166);
    anc_raw[2]  = getArg<double>(args, "anchor0_z",  1.6859);
    anc_raw[3]  = getArg<double>(args, "anchor1_x", -3.0967);
    anc_raw[4]  = getArg<double>(args, "anchor1_y",  3.1997);
    anc_raw[5]  = getArg<double>(args, "anchor1_z",  1.6656);
    anc_raw[6]  = getArg<double>(args, "anchor2_x",  2.3721);
    anc_raw[7]  = getArg<double>(args, "anchor2_y",  2.9621);
    anc_raw[8]  = getArg<double>(args, "anchor2_z",  1.6817);
    anc_raw[9]  = getArg<double>(args, "anchor3_x",  2.4376);
    anc_raw[10] = getArg<double>(args, "anchor3_y", -2.9803);
    anc_raw[11] = getArg<double>(args, "anchor3_z",  1.6931);

    int    win_size = getArg<int>   (args, "window_size", 20);
    double init_x   = getArg<double>(args, "init_x", 0.0);
    double init_y   = getArg<double>(args, "init_y", 0.0);
    double init_z   = getArg<double>(args, "init_z", 0.0);

    Eigen::Vector3d tag0_off(t0x, t0y, t0z);
    Eigen::Vector3d tag1_off(t1x, t1y, t1z);

    Eigen::MatrixXd anchors(num_anchors, 3);
    for (int i = 0; i < num_anchors; ++i)
        anchors.row(i) << anc_raw[3*i], anc_raw[3*i+1], anc_raw[3*i+2];

    std::vector<ImuData>   imu_data;
    std::vector<UwbRange>  uwb_raw;
    std::string dir = getArg<std::string>(args, "data_dir", std::string(""));
    if (dir.empty()) { std::cerr << "[SW-MAP] data_dir required\n"; return 1; }
    if (!loadImu(dir + "/imu.csv", imu_data)) return 1;
    if (!loadUwbRange(dir + "/uwb_range.csv", {tag0_id, tag1_id},
                      anchor_idx_all(num_anchors), uwb_raw)) return 1;
    double t0 = std::min(imu_data.front().timestamp, uwb_raw.front().timestamp);
    for (auto& d : imu_data)   d.timestamp -= t0;
    for (auto& d : uwb_raw)    d.timestamp -= t0;

    std::ofstream ofs(output_file);
    if (!ofs) { std::cerr << "[Ceres] Cannot open output: " << output_file << "\n"; return 1; }
    ofs << "# time x y z qx qy qz qw\n";
    ofs << std::fixed;
    ofs.precision(9);

    const int num_rng = 2 * num_anchors;
    std::vector<int> anchor_idx(num_anchors);
    for (int i = 0; i < num_anchors; ++i) anchor_idx[i] = i;
    LatestRanges latest({tag0_id, tag1_id}, anchor_idx);
    std::deque<WindowNode> window;
    size_t imu_hint = 0;
    int solve_count = 0;
    std::vector<double> solve_times_vec;

    static const double kTukeyDelta = 3.0;

    {
        WindowNode first;
        first.timestamp = imu_data.empty() ? 0.0 : imu_data[0].timestamp;
        first.pos[0] = init_x; first.pos[1] = init_y; first.pos[2] = init_z;
        first.vel[0] = first.vel[1] = first.vel[2] = 0.0;
        first.quat[0] = 0.0; first.quat[1] = 0.0;
        first.quat[2] = 0.0; first.quat[3] = 1.0;
        first.ranges.assign(num_rng, -1.0);
        first.has_uwb = false;
        window.push_back(first);
    }

    struct UwbFrame { double ts; std::vector<UwbRange> entries; };
    std::vector<UwbFrame> uwb_frames;
    for (const auto& uwb : uwb_raw) {
        if (uwb_frames.empty() || uwb.timestamp != uwb_frames.back().ts)
            uwb_frames.push_back({uwb.timestamp, {}});
        uwb_frames.back().entries.push_back(uwb);
    }

    for (const auto& frame : uwb_frames) {
        for (const auto& uwb : frame.entries)
            latest.update(uwb);
        if (!latest.allValid()) continue;

        WindowNode node;
        node.timestamp = frame.ts;
        node.has_uwb   = true;
        node.ranges.resize(num_rng);
        latest.copyTo(node.ranges.data());

        const WindowNode& prev = window.back();

        node.preint = preintegrate(imu_data, prev.timestamp, node.timestamp, imu_hint);

        {
            const Eigen::Vector3d g_world(0.0, 0.0, -9.81);
            Eigen::Quaterniond q_prev(prev.quat[3], prev.quat[0],
                                      prev.quat[1], prev.quat[2]);
            Eigen::Matrix3d    R_prev = q_prev.toRotationMatrix();
            Eigen::Vector3d    p_prev(prev.pos[0], prev.pos[1], prev.pos[2]);
            Eigen::Vector3d    v_prev(prev.vel[0], prev.vel[1], prev.vel[2]);
            const double       dt_imu = node.preint.dt;

            Eigen::Vector3d p_new = p_prev
                                  + v_prev * dt_imu
                                  + 0.5 * g_world * dt_imu * dt_imu
                                  + R_prev * node.preint.delta_p;
            Eigen::Vector3d v_new = v_prev
                                  + g_world * dt_imu
                                  + R_prev * node.preint.delta_v;
            Eigen::Quaterniond q_new(R_prev * node.preint.delta_R);
            q_new.normalize();

            for (int k = 0; k < 3; ++k) node.pos[k] = p_new(k);
            for (int k = 0; k < 3; ++k) node.vel[k] = v_new(k);
            node.quat[0] = q_new.x(); node.quat[1] = q_new.y();
            node.quat[2] = q_new.z(); node.quat[3] = q_new.w();
        }

        window.push_back(node);

        if ((int)window.size() < win_size + 1) continue;

        auto t0s = std::chrono::high_resolution_clock::now();
        ceres::Problem problem;

        for (int w = 0; w < (int)window.size(); ++w) {
            WindowNode& nd = window[w];

            problem.AddParameterBlock(nd.pos, 3);
            problem.AddParameterBlock(nd.vel, 3);
            problem.AddParameterBlock(nd.quat, 4);

#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
            problem.SetManifold(nd.quat, new ceres::EigenQuaternionManifold());
#else
            problem.SetParameterization(nd.quat,
                new ceres::EigenQuaternionParameterization());
#endif

            if (nd.has_uwb) {
                problem.AddResidualBlock(
                    UWBRangingCost::Create(nd.ranges.data(), anchors,
                                           tag0_off, tag1_off, num_anchors),
                    new ceres::TukeyLoss(kTukeyDelta),
                    nd.pos, nd.quat);
            }

            if (w + 1 < (int)window.size()) {
                const ImuPreint& pre = window[w + 1].preint;
                if (pre.dt > 1e-9 && pre.dt < 1.0) {
                    problem.AddResidualBlock(
                        IMUPreIntCost::Create(pre),
                        nullptr,
                        nd.pos, nd.vel, nd.quat,
                        window[w+1].pos, window[w+1].vel, window[w+1].quat);
                }
            }

            if (w == 0) {
                problem.SetParameterBlockConstant(nd.pos);
                problem.SetParameterBlockConstant(nd.vel);
                problem.SetParameterBlockConstant(nd.quat);
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
        Eigen::Quaterniond q(out.quat[3], out.quat[0], out.quat[1], out.quat[2]);
        q.normalize();
        ofs << (out.timestamp + t0) << " "
            << out.pos[0] << " " << out.pos[1] << " " << out.pos[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        ++solve_count;

        window.pop_front();
    }

    ofs.close();

    {
        size_t Nv = solve_times_vec.size();
        double sum = 0.0, sum2 = 0.0, max_ms = 0.0;
        for (double ms : solve_times_vec) { sum += ms; sum2 += ms*ms; if (ms > max_ms) max_ms = ms; }
        double mean_ms = Nv > 0 ? sum / Nv : 0.0;
        double std_ms  = Nv > 1 ? std::sqrt((sum2 - sum*sum/Nv) / (Nv-1)) : 0.0;
        std::string timing_file = output_file;
        auto tpos = timing_file.rfind("_pose.txt");
        if (tpos != std::string::npos) timing_file.replace(tpos, 9, "_timing.txt");
        else timing_file += ".timing.txt";
        std::ofstream tf(timing_file);
        tf << std::fixed; tf.precision(6);
        for (double ms : solve_times_vec) tf << ms << "\n";
        printf("[Ceres] Timing: mean=%.3f ms  std=%.3f ms  max=%.3f ms  N=%zu\n",
               mean_ms, std_ms, max_ms, Nv);
    }

    std::cout << "[Ceres] Done. Solves=" << solve_count
              << "  output=" << output_file << "\n";
    return 0;
}
