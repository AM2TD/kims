#include "csv_io.h"
#include "node_util.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <deque>
#include <map>
#include <string>
#include <cmath>
#include <chrono>
static double g_robust_delta = 0.0;
static bool   g_robust_tukey = false;
static double g_sigma_uwb = 1.0;
static bool   g_fresh_only = false;
static bool   g_mask_missing = false;
static double g_w_pos = 1.0, g_w_vel = 1.0, g_w_rot = 5.0;
static double g_q_acc = 0.0, g_q_gyr = 0.0;
static bool   g_use_bias = false;
struct SO3Plus {
    template <typename T> bool operator()(const T* x, const T* d, T* y) const { return Plus(x, d, y); }
    template <typename T> bool Plus(const T* x, const T* d, T* y) const {
        T dR[9]; ceres::AngleAxisToRotationMatrix(d, ceres::RowMajorAdapter3x3(dR));
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            y[3 * i + j] = T(0); for (int k = 0; k < 3; ++k) y[3 * i + j] += x[3 * i + k] * dR[3 * k + j]; }
        return true; }
    template <typename T> bool Minus(const T* y, const T* x, T* d) const {
        T M[9];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            M[3 * i + j] = T(0); for (int k = 0; k < 3; ++k) M[3 * i + j] += x[3 * k + i] * y[3 * k + j]; }
        ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3(static_cast<const T*>(M)), d);
        return true; }
};
template <typename T> static void so3Log(const T* Ra, const T* Rb, T* d) { SO3Plus().Minus(Rb, Ra, d); }
static void setSO3(ceres::Problem& problem, double* rot) {
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem.AddParameterBlock(rot, 9, new ceres::AutoDiffManifold<SO3Plus, 9, 3>);
#else
    problem.AddParameterBlock(rot, 9, new ceres::AutoDiffLocalParameterization<SO3Plus, 9, 3>);
#endif
}
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
                if (g_mask_missing && ranges_[ri] <= 0.0) { residuals[ri] = T(0); continue; }
                residuals[ri] = ((tags[ti] - anc).norm() - T(ranges_[ri])) / T(g_sigma_uwb);
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
static std::vector<Eigen::Vector3d> g_anchorsB;
static bool g_tdoa = false;
struct SingleRangeCost {
    SingleRangeCost(double range, const Eigen::Vector3d& anchor, const Eigen::Vector3d& off, const Eigen::Vector3d* anchorB)
        : range_(range), anchor_(anchor), off_(off), tdoa_(anchorB != nullptr), anchorB_(anchorB ? *anchorB : Eigen::Vector3d::Zero()) {}
    template <typename T> bool operator()(const T* const position, const T* const rotation, T* residual) const {
        Eigen::Matrix<T, 3, 3> R;
        R << rotation[0], rotation[1], rotation[2], rotation[3], rotation[4], rotation[5], rotation[6], rotation[7], rotation[8];
        Eigen::Matrix<T, 3, 1> tag = Eigen::Matrix<T, 3, 1>(position[0], position[1], position[2]) + R * off_.template cast<T>();
        T d = (tag - anchor_.template cast<T>()).norm();
        if (tdoa_) d = (tag - anchorB_.template cast<T>()).norm() - d + T(100.0);
        residual[0] = (d - T(range_)) / T(g_sigma_uwb);
        return true; }
    static ceres::CostFunction* Create(double range, const Eigen::Vector3d& anchor, const Eigen::Vector3d& off, const Eigen::Vector3d* anchorB) {
        return new ceres::AutoDiffCostFunction<SingleRangeCost, 1, 3, 9>(new SingleRangeCost(range, anchor, off, anchorB)); }
    double range_; Eigen::Vector3d anchor_, off_; bool tdoa_; Eigen::Vector3d anchorB_;
};
struct IMUDynamicsCost {
    IMUDynamicsCost(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double dt)
        : acc_(acc), gyr_(gyr), dt_(dt), rot_w_(g_w_rot) {
        if (g_q_acc > 0.0) {
            w_vel_ = 1.0 / (g_q_acc * std::sqrt(dt)); w_pos_ = std::sqrt(3.0) / (g_q_acc * dt * std::sqrt(dt)); rot_w_ = 1.0 / (g_q_gyr * std::sqrt(dt));
        } else { w_vel_ = g_w_vel; w_pos_ = g_w_pos; }
    }
    template <typename T>
    bool operator()(const T* const pos_i, const T* const vel_i, const T* const rot_i,
                    const T* const pos_j, const T* const vel_j, const T* const rot_j,
                    const T* const bias, T* residuals) const {
        Eigen::Matrix<T, 3, 3> R_i;
        R_i << rot_i[0], rot_i[1], rot_i[2],
               rot_i[3], rot_i[4], rot_i[5],
               rot_i[6], rot_i[7], rot_i[8];
        Eigen::Matrix<T, 3, 1> acc_c = acc_.cast<T>() - Eigen::Matrix<T, 3, 1>(bias[0], bias[1], bias[2]);
        Eigen::Matrix<T, 3, 1> a_world = R_i * acc_c;
        a_world[2] -= T(9.81);
        Eigen::Matrix<T, 3, 1> vi(vel_i[0], vel_i[1], vel_i[2]);
        Eigen::Matrix<T, 3, 1> pi(pos_i[0], pos_i[1], pos_i[2]);
        Eigen::Matrix<T, 3, 1> pred_pj = pi + vi * T(dt_) + T(0.5) * a_world * T(dt_ * dt_);
        Eigen::Matrix<T, 3, 1> pred_vj = vi + a_world * T(dt_);
        residuals[0] = T(w_pos_) * (pos_j[0] - pred_pj[0]);
        residuals[1] = T(w_pos_) * (pos_j[1] - pred_pj[1]);
        residuals[2] = T(w_pos_) * (pos_j[2] - pred_pj[2]);
        residuals[3] = T(w_vel_) * (vel_j[0] - pred_vj[0]);
        residuals[4] = T(w_vel_) * (vel_j[1] - pred_vj[1]);
        residuals[5] = T(w_vel_) * (vel_j[2] - pred_vj[2]);
        Eigen::Matrix<T, 3, 1> omega = (gyr_.cast<T>() - Eigen::Matrix<T, 3, 1>(bias[3], bias[4], bias[5])) * T(dt_);
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
        T P[9];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) P[3 * i + j] = pred_Rj(i, j);
        T e[3]; so3Log(P, rot_j, e);
        for (int i = 0; i < 3; ++i) residuals[6 + i] = T(rot_w_) * e[i];
        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3d& acc,
                                        const Eigen::Vector3d& gyr, double dt) {
        return new ceres::AutoDiffCostFunction<IMUDynamicsCost, 9, 3, 3, 9, 3, 3, 9, 6>(
            new IMUDynamicsCost(acc, gyr, dt));
    }
private:
    Eigen::Vector3d acc_, gyr_;
    double dt_, rot_w_, w_pos_, w_vel_;
};
struct BiasPriorCost {
    BiasPriorCost(double sa, double sg) : sa_(sa), sg_(sg) {}
    template <typename T> bool operator()(const T* const b, T* r) const {
        for (int i = 0; i < 3; ++i) { r[i] = b[i] / T(sa_); r[3 + i] = b[3 + i] / T(sg_); } return true;
    }
    static ceres::CostFunction* Create(double sa, double sg) { return new ceres::AutoDiffCostFunction<BiasPriorCost, 6, 6>(new BiasPriorCost(sa, sg)); }
    double sa_, sg_;
};
struct FloorCost {
    explicit FloorCost(double w) : w_(w) {}
    template <typename T> bool operator()(const T* const position, T* residual) const {
        residual[0] = position[2] < T(0.0) ? T(w_) * position[2] : T(0.0); return true;
    }
    static ceres::CostFunction* Create(double w) { return new ceres::AutoDiffCostFunction<FloorCost, 1, 3>(new FloorCost(w)); }
    double w_;
};
static double g_floor_w = 0.0;
struct NodePriorCost {
    NodePriorCost(const double* p, const double* v, const double* r, double sp, double sv, double sr) : sp_(sp), sv_(sv), sr_(sr) {
        for (int i = 0; i < 3; ++i) { p_[i] = p[i]; v_[i] = v[i]; } for (int i = 0; i < 9; ++i) r_[i] = r[i]; }
    template <typename T> bool operator()(const T* const p, const T* const v, const T* const r, T* res) const {
        for (int i = 0; i < 3; ++i) { res[i] = (p[i] - T(p_[i])) / T(sp_); res[3 + i] = (v[i] - T(v_[i])) / T(sv_); }
        T rr[9]; for (int i = 0; i < 9; ++i) rr[i] = T(r_[i]);
        T e[3]; so3Log(rr, r, e); for (int i = 0; i < 3; ++i) res[6 + i] = e[i] / T(sr_);
        return true; }
    static ceres::CostFunction* Create(const double* p, const double* v, const double* r, double sp, double sv, double sr) {
        return new ceres::AutoDiffCostFunction<NodePriorCost, 9, 3, 3, 9>(new NodePriorCost(p, v, r, sp, sv, sr)); }
    double p_[3], v_[3], r_[9], sp_, sv_, sr_;
};
static double g_prior_pos = 0.0, g_prior_vel = 0.0, g_prior_rot = 0.0;
struct CarriedPriorCost {
    CarriedPriorCost(const Eigen::Matrix<double, 6, 1>& pv, const double* r, const Eigen::Matrix<double, 9, 9>& Lt) : pv_(pv), Lt_(Lt) {
        for (int i = 0; i < 9; ++i) r_[i] = r[i]; }
    template <typename T> bool operator()(const T* const p, const T* const v, const T* const r, T* res) const {
        Eigen::Matrix<T, 9, 1> d;
        for (int i = 0; i < 3; ++i) { d[i] = p[i] - T(pv_[i]); d[3 + i] = v[i] - T(pv_[3 + i]); }
        T rr[9]; for (int i = 0; i < 9; ++i) rr[i] = T(r_[i]);
        T e[3]; so3Log(rr, r, e); for (int i = 0; i < 3; ++i) d[6 + i] = e[i];
        Eigen::Matrix<T, 9, 1> out = Lt_.template cast<T>() * d;
        for (int i = 0; i < 9; ++i) res[i] = out[i];
        return true; }
    static ceres::CostFunction* Create(const Eigen::Matrix<double, 6, 1>& pv, const double* r, const Eigen::Matrix<double, 9, 9>& Lt) {
        return new ceres::AutoDiffCostFunction<CarriedPriorCost, 9, 3, 3, 9>(new CarriedPriorCost(pv, r, Lt)); }
    Eigen::Matrix<double, 6, 1> pv_; double r_[9]; Eigen::Matrix<double, 9, 9> Lt_;
};
static bool g_carry_cov = false;
static double g_max_dt = 1.0;
static double g_ftol = 1e-6, g_gtol = 1e-10, g_ptol = 1e-8;
static long g_iters = 0; static int g_term[8] = {0, 0, 0, 0, 0, 0, 0, 0};
struct WindowNode {
    double timestamp;
    double pos[3];
    double vel[3];
    double rot[9];
    double ranges[32];
    ImuData imu;
    bool has_uwb;
};
static void rotToArray(const Eigen::Matrix3d& R, double* arr) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            arr[i*3+j] = R(i, j);
}
static Eigen::Matrix3d g_R0 = Eigen::Matrix3d::Identity();
static bool g_R0_set = false;
template <int NT, int NA>
static int runSwPgo(const std::vector<ImuData>& imu_data,
                    const std::vector<UwbRange>& uwb_raw,
                    const std::vector<int>& tag_ids,
                    const std::vector<int>& anchor_ids,
                    const Eigen::Matrix<double, NA, 3>& anchors,
                    const std::vector<Eigen::Vector3d>& tag_off,
                    int win_size, const Eigen::Vector3d& init_p,
                    bool z_down, const std::string& output_file,
                    bool group_same_ts = false, int out_index = -2, bool sparse = false, int max_iter = 20,
                    double dc_acc = 0.0, double dc_gyr = 0.0, int threads = 1) {
    const double init_x = init_p.x(), init_y = init_p.y(), init_z = init_p.z();
    std::ofstream ofs(output_file);
    if (!ofs) { std::cerr << "[SW-PGO] Cannot open output: " << output_file << "\n"; return 1; }
    ofs << "# time x y z qx qy qz qw\n";
    ofs << std::fixed; ofs.precision(9);
    LatestRanges latest(tag_ids, anchor_ids);
    std::deque<WindowNode> window;
    size_t imu_hint = 0, imu_mi = 0;
    int solve_count = 0;
    std::vector<double> solve_times_vec;
    double bias[6] = {0, 0, 0, 0, 0, 0};
    bool have_carry = false; Eigen::Matrix<double, 6, 1> carry_pv; double carry_r[9]; Eigen::Matrix<double, 9, 9> carry_Lt;
    double last_out_ts = -1.0;
    bool first_solve = true;
    auto writeNode = [&](const WindowNode& out) {
        if (out.timestamp <= last_out_ts) return;
        Eigen::Matrix3d R_out;
        R_out << out.rot[0], out.rot[1], out.rot[2], out.rot[3], out.rot[4], out.rot[5], out.rot[6], out.rot[7], out.rot[8];
        Eigen::Quaterniond q(R_out); q.normalize();
        ofs << out.timestamp << " " << out.pos[0] << " " << out.pos[1] << " " << out.pos[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        last_out_ts = out.timestamp;
    };
    WindowNode first_node;
    first_node.timestamp = imu_data.empty() ? 0.0 : imu_data[0].timestamp;
    first_node.pos[0] = init_x; first_node.pos[1] = init_y; first_node.pos[2] = init_z;
    first_node.vel[0] = first_node.vel[1] = first_node.vel[2] = 0.0;
    rotToArray(g_R0_set ? g_R0 : (z_down ? Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix()
                                         : Eigen::Matrix3d::Identity()),
               first_node.rot);
    std::fill(first_node.ranges, first_node.ranges + 32, -1.0);
    first_node.has_uwb = false;
    window.push_back(first_node);
    const double t_first_imu = imu_data.empty() ? 0.0 : imu_data.front().timestamp;
    for (size_t ui = 0; ui < uwb_raw.size(); ++ui) {
        const auto& uwb = uwb_raw[ui];
        latest.update(uwb);
        WindowNode node;
        node.timestamp = uwb.timestamp;
        node.has_uwb   = true;
        std::fill(node.ranges, node.ranges + 32, -1.0);
        auto keepIdx = [&](const UwbRange& m) {
            const auto ti = std::find(tag_ids.begin(), tag_ids.end(), m.from_id);
            const auto ai = std::find(anchor_ids.begin(), anchor_ids.end(), m.to_id);
            return (ti != tag_ids.end() && ai != anchor_ids.end()) ? int(ti - tag_ids.begin()) * NA + int(ai - anchor_ids.begin()) : -1;
        };
        if (group_same_ts) {
            int k = keepIdx(uwb); if (k >= 0) node.ranges[k] = uwb.range;
            while (ui + 1 < uwb_raw.size() && uwb_raw[ui + 1].timestamp <= uwb.timestamp + 1e-9) {
                ++ui; latest.update(uwb_raw[ui]); k = keepIdx(uwb_raw[ui]); if (k >= 0) node.ranges[k] = uwb_raw[ui].range;
            }
        } else if (g_fresh_only) {
            const int k = keepIdx(uwb); if (k >= 0) node.ranges[k] = uwb.range;
        } else {
            if (!latest.allValid()) continue;
            latest.copyTo(node.ranges);
        }
        if (node.timestamp < t_first_imu || node.timestamp <= window.back().timestamp + 1e-6) continue;
        node.imu = interpolateImu(imu_data, node.timestamp, imu_hint);
        {
            WindowNode& pn = window.back();
            while (imu_mi < imu_data.size() && imu_data[imu_mi].timestamp <= pn.timestamp) ++imu_mi;
            Eigen::Vector3d sa = Eigen::Vector3d::Zero(), sg = Eigen::Vector3d::Zero(); int na = 0;
            for (size_t j = imu_mi; j < imu_data.size() && imu_data[j].timestamp <= node.timestamp; ++j) { sa += imu_data[j].acc; sg += imu_data[j].gyr; ++na; }
            if (na > 0) { pn.imu.acc = sa / na; pn.imu.gyr = sg / na; }
        }
        {
            const WindowNode& prev = window.back();
            Eigen::Matrix3d Rp; Rp << prev.rot[0], prev.rot[1], prev.rot[2], prev.rot[3], prev.rot[4], prev.rot[5], prev.rot[6], prev.rot[7], prev.rot[8];
            const double dt = node.timestamp - prev.timestamp;
            const Eigen::Vector3d ba(bias[0], bias[1], bias[2]), bg(bias[3], bias[4], bias[5]);
            const Eigen::Vector3d aw = Rp * (prev.imu.acc - ba) - Eigen::Vector3d(0, 0, 9.81), vp(prev.vel[0], prev.vel[1], prev.vel[2]);
            const Eigen::Vector3d w = (prev.imu.gyr - bg) * dt;
            const Eigen::Matrix3d Rn = Rp * (w.norm() > 1e-9 ? Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
            for (int k = 0; k < 3; ++k) { node.pos[k] = prev.pos[k] + vp[k] * dt + 0.5 * aw[k] * dt * dt; node.vel[k] = vp[k] + aw[k] * dt; }
            rotToArray(Rn, node.rot);
        }
        window.push_back(node);
        if ((int)window.size() < win_size + 1) continue;
        auto t0s = std::chrono::high_resolution_clock::now();
        ceres::Problem problem;
        for (int w = 0; w < (int)window.size(); ++w) {
            WindowNode& nd = window[w];
            auto makeLoss = [&]() -> ceres::LossFunction* {
                return g_robust_delta > 0.0 ? (g_robust_tukey ? static_cast<ceres::LossFunction*>(new ceres::TukeyLoss(g_robust_delta)) : static_cast<ceres::LossFunction*>(new ceres::CauchyLoss(g_robust_delta))) : nullptr; };
            if (nd.has_uwb && g_mask_missing) {
                for (int ti = 0; ti < NT; ++ti) for (int ai = 0; ai < NA; ++ai) {
                    const int ri = ti * NA + ai; if (nd.ranges[ri] <= 0.0) continue;
                    problem.AddResidualBlock(SingleRangeCost::Create(nd.ranges[ri], anchors.row(ai).transpose(), tag_off[ti], g_tdoa ? &g_anchorsB[ai] : nullptr), makeLoss(), nd.pos, nd.rot);
                }
            } else if (nd.has_uwb) {
                problem.AddResidualBlock(UWBRangingCost<NT, NA>::Create(nd.ranges, anchors, tag_off), makeLoss(), nd.pos, nd.rot);
            }
            setSO3(problem, nd.rot);
            if (g_floor_w > 0.0) problem.AddResidualBlock(FloorCost::Create(g_floor_w), nullptr, nd.pos);
            if (w + 1 < (int)window.size()) {
                double dt = window[w+1].timestamp - nd.timestamp;
                if (dt > 0.0 && dt < g_max_dt) {
                    problem.AddResidualBlock(
                        IMUDynamicsCost::Create(nd.imu.acc, nd.imu.gyr, dt),
                        nullptr,
                        nd.pos, nd.vel, nd.rot,
                        window[w+1].pos, window[w+1].vel, window[w+1].rot, bias);
                }
            }
            if (w == 0) {
                problem.AddParameterBlock(nd.pos, 3);
                problem.AddParameterBlock(nd.vel, 3);
                if (g_carry_cov && have_carry) {
                    problem.AddResidualBlock(CarriedPriorCost::Create(carry_pv, carry_r, carry_Lt), nullptr, nd.pos, nd.vel, nd.rot);
                } else if (g_prior_pos > 0.0) {
                    problem.AddResidualBlock(NodePriorCost::Create(nd.pos, nd.vel, nd.rot, g_prior_pos, g_prior_vel, g_prior_rot), nullptr, nd.pos, nd.vel, nd.rot);
                } else {
                    problem.SetParameterBlockConstant(nd.pos);
                    problem.SetParameterBlockConstant(nd.vel);
                    problem.SetParameterBlockConstant(nd.rot);
                }
            }
        }
        if (g_use_bias) problem.AddResidualBlock(BiasPriorCost::Create(dc_acc, dc_gyr), nullptr, bias);
        else { problem.AddParameterBlock(bias, 6); problem.SetParameterBlockConstant(bias); }
        ceres::Solver::Options opts;
        opts.linear_solver_type           = sparse ? ceres::SPARSE_NORMAL_CHOLESKY : ceres::DENSE_SCHUR;
        opts.max_num_iterations           = max_iter;
        opts.num_threads                  = threads;
        opts.minimizer_progress_to_stdout = false;
        opts.function_tolerance = g_ftol; opts.gradient_tolerance = g_gtol; opts.parameter_tolerance = g_ptol;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        g_iters += summary.iterations.size() - 1; g_term[std::min(int(summary.termination_type), 7)]++;
        if (g_carry_cov) {
            ceres::Covariance::Options co; co.algorithm_type = ceres::SPARSE_QR; co.null_space_rank = -1;
            ceres::Covariance cov(co);
            WindowNode& n1 = window[1];
            const double* b[3] = {n1.pos, n1.vel, n1.rot}; const int ofs3[3] = {0, 3, 6};
            std::vector<std::pair<const double*, const double*>> blocks;
            for (int i = 0; i < 3; ++i) for (int j = i; j < 3; ++j) blocks.push_back({b[i], b[j]});
            have_carry = false;
            if (cov.Compute(blocks, &problem)) {
                Eigen::Matrix<double, 9, 9> S = Eigen::Matrix<double, 9, 9>::Zero();
                for (int i = 0; i < 3; ++i) for (int j = i; j < 3; ++j) {
                    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> blk; cov.GetCovarianceBlockInTangentSpace(b[i], b[j], blk.data());
                    S.block<3, 3>(ofs3[i], ofs3[j]) = blk; S.block<3, 3>(ofs3[j], ofs3[i]) = blk.transpose();
                }
                const double dt1 = window[2].timestamp - n1.timestamp;
                const double qa = g_q_acc > 0.0 ? g_q_acc : 0.1, qg = g_q_gyr > 0.0 ? g_q_gyr : 0.01;
                for (int i = 0; i < 3; ++i) { S(i, i) += std::pow(qa * dt1 * std::sqrt(dt1), 2) / 3.0 + 1e-10; S(3 + i, 3 + i) += qa * qa * dt1 + 1e-10; S(6 + i, 6 + i) += qg * qg * dt1 + 1e-10; }
                Eigen::LDLT<Eigen::Matrix<double, 9, 9>> ldlt(S);
                if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                    const Eigen::Matrix<double, 9, 9> info = ldlt.solve(Eigen::Matrix<double, 9, 9>::Identity());
                    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(0.5 * (info + info.transpose()));
                    if (llt.info() == Eigen::Success) {
                        carry_Lt = llt.matrixL().transpose();
                        for (int i = 0; i < 3; ++i) { carry_pv[i] = n1.pos[i]; carry_pv[3 + i] = n1.vel[i]; } for (int i = 0; i < 9; ++i) carry_r[i] = n1.rot[i];
                        have_carry = true;
                    }
                }
            }
        }
        solve_times_vec.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0s).count());
        const int n = static_cast<int>(window.size());
        const int out_idx = out_index >= 0 ? std::min(out_index, n - 1) : n + out_index;
        if (first_solve && out_index >= 0) { for (int w = 1; w <= out_idx; ++w) writeNode(window[w]); first_solve = false; }
        writeNode(window[out_idx]);
        ++solve_count;
        window.pop_front();
    }
    if (out_index >= 0 && !first_solve) for (int w = 1; w < (int)window.size(); ++w) writeNode(window[w]);
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
        printf("[SW-PGO] Timing: mean=%.3f ms  std=%.3f ms  max=%.3f ms  N=%zu  saved to %s\n",
               mean_ms, std_ms, max_ms, N, timing_file.c_str());
    }
    std::printf("[SW-PGO] iterations per solve %.2f, termination convergence %d no_convergence %d failure %d user %d\n",
                solve_count > 0 ? double(g_iters) / solve_count : 0.0, g_term[0], g_term[1], g_term[2], g_term[3]);
    std::cout << "[SW-PGO] Done. Solves=" << solve_count
              << "  output=" << output_file << "\n";
    return 0;
}
int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);
    const std::string dataset = getArg<std::string>(args, "dataset", std::string("ntu"));
    const bool is_iului = (dataset == "iului");
    const bool is_util = (dataset == "util");
    const bool is_ntu = (dataset != "miluv" && !is_iului && !is_util);
    std::string data_dir    = getArg<std::string>(args, "data_dir", std::string(""));
    std::string anchor_file = getArg<std::string>(args, "anchor_file", std::string(""));
    std::string output_file = getArg<std::string>(args, "output_file", std::string("sw_pgo_pose.txt"));
    std::string imu_file    = getArg<std::string>(args, "imu_file",
                                  data_dir + ((is_ntu || is_iului || is_util) ? "/imu.csv" : "/imu_px4.csv"));
    std::string uwb_file    = getArg<std::string>(args, "uwb_file", data_dir + "/uwb_range.csv");
    const int NT = is_ntu ? 4 : (is_util ? 1 : 2);
    const int NA = is_ntu ? 3 : (is_iului ? 4 : (is_util ? getArg<int>(args, "num_anchors", 8) : 6));
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
    g_robust_delta  = getArg<double>(args, "robust_delta", 0.0);
    g_robust_tukey  = getArg<int>(args, "robust_tukey", 0) != 0;
    g_sigma_uwb     = getArg<double>(args, "std_uwb", 1.0);
    g_fresh_only    = getArg<int>(args, "fresh_only", 0) != 0;
    const bool group_same_ts = getArg<int>(args, "group_same_ts", 0) != 0;
    g_mask_missing  = g_fresh_only || group_same_ts;
    g_q_acc         = getArg<double>(args, "q_acc", 0.0);
    g_q_gyr         = getArg<double>(args, "q_gyr", 0.0);
    const double dc_acc = getArg<double>(args, "dc_acc", 0.0), dc_gyr = getArg<double>(args, "dc_gyr", 0.0);
    g_use_bias      = dc_acc > 0.0 && dc_gyr > 0.0;
    const int    out_index = getArg<int>(args, "out_index", -2);
    const bool   sparse    = getArg<int>(args, "sparse", 0) != 0;
    const int    max_iter  = getArg<int>(args, "max_iter", 20);
    const int    threads   = getArg<int>(args, "threads", 1);
    g_prior_pos = getArg<double>(args, "prior_pos", 0.0); g_prior_vel = getArg<double>(args, "prior_vel", 0.05); g_prior_rot = getArg<double>(args, "prior_rot", 0.01);
    g_carry_cov = getArg<int>(args, "carry_cov", 0) != 0;
    g_max_dt = getArg<double>(args, "max_dt", 1.0);
    g_ftol = getArg<double>(args, "ftol", 1e-6); g_gtol = getArg<double>(args, "gtol", 1e-10); g_ptol = getArg<double>(args, "ptol", 1e-8);
    g_floor_w       = std::sqrt(getArg<double>(args, "floor_penalty", 0.0));
    double init_x   = getArg<double>(args, "init_x", 0.0);
    double init_y   = getArg<double>(args, "init_y", 0.0);
    double init_z   = getArg<double>(args, "init_z", 0.0);
    std::vector<ImuData>  imu_data;
    std::vector<UwbRange> uwb_raw;
    if (!loadImu(imu_file, imu_data)) return 1;
    if (getArg<int>(args, "still_bias", 0) != 0 && !imu_data.empty()) {
        const double still = std::max(getArg<double>(args, "align_still", 3.0), 0.5);
        Eigen::Vector3d am = Eigen::Vector3d::Zero(), gm = Eigen::Vector3d::Zero(); int n = 0;
        for (const auto& d : imu_data) { if (d.timestamp > imu_data.front().timestamp + still) break; am += d.acc; gm += d.gyr; ++n; }
        if (n > 0) { am /= n; gm /= n; const Eigen::Vector3d ba = (am.norm() - 9.81) * am.normalized(); for (auto& d : imu_data) { d.gyr -= gm; d.acc -= ba; } }
    }
    if (args.count("align_still") || args.count("init_yaw")) {
        g_R0 = initialAttitude(imu_data, dataset == "ntu", getArg<double>(args, "align_still", 0.0), getArg<double>(args, "init_yaw", 0.0));
        g_R0_set = true;
    }
    if (!loadUwbRange(uwb_file, tag_ids, anchor_ids, uwb_raw)) return 1;
    std::vector<double> h_anchor(3 * NA);
    if (!loadAnchors(anchor_file, h_anchor.data(), NA)) return 1;
    if (is_util) {
        std::vector<double> hb(3 * NA);
        if (!loadAnchors(getArg<std::string>(args, "anchor_b_file", std::string("")), hb.data(), NA)) return 1;
        for (int i = 0; i < NA; ++i) g_anchorsB.push_back(Eigen::Vector3d(hb[3*i], hb[3*i+1], hb[3*i+2]));
        g_tdoa = true;
        if (NA == 8) {
            Eigen::Matrix<double, 8, 3> A; for (int i = 0; i < 8; ++i) A.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
            return runSwPgo<1, 8>(imu_data, uwb_raw, tag_ids, anchor_ids, A, tag_off, win_size, Eigen::Vector3d(init_x, init_y, init_z), false, output_file, group_same_ts, out_index, sparse, max_iter, dc_acc, dc_gyr, threads);
        } else if (NA == 28) {
            Eigen::Matrix<double, 28, 3> A; for (int i = 0; i < 28; ++i) A.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
            return runSwPgo<1, 28>(imu_data, uwb_raw, tag_ids, anchor_ids, A, tag_off, win_size, Eigen::Vector3d(init_x, init_y, init_z), false, output_file, group_same_ts, out_index, sparse, max_iter, dc_acc, dc_gyr, threads);
        } else { std::cerr << "[SW-PGO] util: num_anchors must be 8 or 28\n"; return 1; }
    }
    if (is_ntu) {
        Eigen::Matrix<double, 3, 3> anchors;
        for (int i = 0; i < 3; ++i)
            anchors.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
        return runSwPgo<4, 3>(imu_data, uwb_raw, tag_ids, anchor_ids, anchors, tag_off,
                              win_size, Eigen::Vector3d(init_x, init_y, init_z),
                              true, output_file, group_same_ts, out_index, sparse, max_iter, dc_acc, dc_gyr, threads);
    }
    if (is_iului) {
        Eigen::Matrix<double, 4, 3> anchors;
        for (int i = 0; i < 4; ++i)
            anchors.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
        return runSwPgo<2, 4>(imu_data, uwb_raw, tag_ids, anchor_ids, anchors, tag_off,
                              win_size, Eigen::Vector3d(init_x, init_y, init_z),
                              false, output_file, group_same_ts, out_index, sparse, max_iter, dc_acc, dc_gyr, threads);
    }
    Eigen::Matrix<double, 6, 3> anchors;
    for (int i = 0; i < 6; ++i)
        anchors.row(i) << h_anchor[3*i], h_anchor[3*i+1], h_anchor[3*i+2];
    return runSwPgo<2, 6>(imu_data, uwb_raw, tag_ids, anchor_ids, anchors, tag_off,
                          win_size, Eigen::Vector3d(init_x, init_y, init_z),
                          false, output_file, group_same_ts, out_index, sparse, max_iter, dc_acc, dc_gyr, threads);
}
