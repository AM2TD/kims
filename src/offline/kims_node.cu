#include "session.h"
#include "mppi_solver.cuh"
#include "presets.h"
#include "basis_opt.h"

#include <algorithm>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>

using namespace kims;



int main(int argc, char** argv) {
    const auto args = parseArgs(argc, argv);

    Session S;
    if (!loadSession(args, S)) return -1;

    const std::string output_file = getArg<std::string>(args, "output_file", "mppi_pose.txt");

    const KimsPreset P = kimsPreset(getArg<std::string>(args, "dataset", "iului"),
                                    getArg<std::string>(args, "sequence", ""));

    const int    N     = getArg<int>   (args, "N",     P.N);
    const int    T     = getArg<int>   (args, "T",     P.T);
    const double dt    = getArg<double>(args, "dt",    P.dt);
    const double gamma = getArg<double>(args, "gamma", P.gamma);

    double sigma[DIM_U] = {
        getArg<double>(args, "sigma_acc_x", P.sigma_acc[0]),
        getArg<double>(args, "sigma_acc_y", P.sigma_acc[1]),
        getArg<double>(args, "sigma_acc_z", P.sigma_acc[2]),
        getArg<double>(args, "sigma_gyr_x", P.sigma_gyr[0]),
        getArg<double>(args, "sigma_gyr_y", P.sigma_gyr[1]),
        getArg<double>(args, "sigma_gyr_z", P.sigma_gyr[2])};

    const double w_uwb = getArg<double>(args, "w_uwb", P.w_uwb);
    const double w_acc = getArg<double>(args, "w_acc", P.w_acc);
    const double w_gyr = getArg<double>(args, "w_gyr", P.w_gyr);

    const int    ctrl_mode    = getArg<int>   (args, "ctrl_mode",     P.ctrl_mode);
    const double sigma_imu_a  = getArg<double>(args, "sigma_imu_acc", P.sigma_imu_acc);
    const double sigma_imu_g  = getArg<double>(args, "sigma_imu_gyr", P.sigma_imu_gyr);
    const double uwb_delta    = getArg<double>(args, "uwb_delta",     P.uwb_delta);
    const double ctrl_delta   = getArg<double>(args, "ctrl_delta",    P.ctrl_delta);
    const double floor_pen    = getArg<double>(args, "floor_penalty", P.floor_penalty);

    const double uwb_std_fixed = getArg<double>(args, "uwb_std", P.uwb_std);

    // ---- basis mode (docs/revision/input_space_v2.md): K > 0 enables it ----------------------
    // perturbation = carried bias b + delta_b (random-walk prior) + sum_k c_k phi_k(t), sampled from
    // the IMU-noise prior (densities q_*), weighted by the UWB likelihood only.
    const int    K        = getArg<int>   (args, "K",        0);
    const int    iters    = getArg<int>   (args, "iters",    2);
    const double ess_min  = getArg<double>(args, "ess_min",  0.1);
    const double q_acc    = getArg<double>(args, "q_acc",    0.040);   // m/s^2/sqrt(Hz)
    const double q_gyr    = getArg<double>(args, "q_gyr",    0.0013);  // rad/s/sqrt(Hz)
    // per-axis densities (Allan variance per axis); default to the scalar value
    const double q_axis[6] = {getArg<double>(args, "q_acc_x", q_acc), getArg<double>(args, "q_acc_y", q_acc),
                              getArg<double>(args, "q_acc_z", q_acc), getArg<double>(args, "q_gyr_x", q_gyr),
                              getArg<double>(args, "q_gyr_y", q_gyr), getArg<double>(args, "q_gyr_z", q_gyr)};
    // fresh_only:=1 puts each range measurement into exactly one window epoch (the first one after it arrives);
    // 0 (published behaviour) repeats the latest value of every tag-anchor pair in every epoch, which counts a
    // measurement about (number of pairs / ranges per epoch) times when ranging is round-robin (MILUV: 12x).
    const int fresh_only = getArg<int>(args, "fresh_only", 0);
    // opt:=1 Gauss-Newton, 2 Levenberg-Marquardt on the same window objective (include/basis_opt.h); 0 sampling
    const int opt      = getArg<int>(args, "opt", 0);
    const int opt_iter = getArg<int>(args, "opt_iter", opt == 1 ? 10 : 30);
    // branches:=2 keeps the estimate and its reflection about the anchor plane (position and velocity normal
    // components negated, attitude kept) as two heads run by the same sampler; the output is the branch with the
    // larger accumulated newest-epoch predictive log-likelihood. A branch that falls behind by more than
    // reseed_nats is re-seeded as the reflection of the leader.
    const int    branches   = getArg<int>   (args, "branches",   1);
    const double reseed_nats = getArg<double>(args, "reseed_nats", std::log(1000.0));
    // crn:=1 evaluates both branches with the same random draws and takes the evidence from the first pass
    const int    crn        = getArg<int>   (args, "crn",        0);
    // head:=1 samples the head state (dp, dv, dtheta) from N(0, P) with P carried from the previous solve as the weighted
    // covariance of the samples' next states, floored at one IMU step of process noise
    // head:=2 keeps the sampling and the weighted-mean head but sets P by a sample-based Laplace arrival cost: predict with
    // F regressed from all samples plus one step of white IMU noise, update with the curvature of the leaving epoch's cost
    // fitted over all samples
    const int    head_on    = getArg<int>   (args, "head",       0);
    const double hp0_p = getArg<double>(args, "head_p0_pos", 0.2), hp0_v = getArg<double>(args, "head_p0_vel", 0.05),
                 hp0_a = getArg<double>(args, "head_p0_att", 0.01);
    // seq:=1 (with head:=2): proposal = previous solve's weighted posterior moved one epoch (defensive mixture with the
    // target prior), adaptive tempering when it collapses. hbias:=1 adds the IMU bias to the
    // head state (15 dims) with a random-walk model (q_ba, q_bg) and the same arrival cost; use dc_acc:=-1 dc_gyr:=-1 so
    // the window-constant input mode keeps only the white-noise density.
    // true_time:=1 integrates every window step over the true spacing of its epochs (MILUV ranging is irregular, 5-95 %
    // spacing 8.5-25 ms, so one dt for the whole window stretches it between about 2 and 5.7 s), evaluates the basis
    // on the true epoch times and feeds each step the mean IMU sample over its interval
    const int    true_time  = getArg<int>   (args, "true_time",  0);
    const int    seq        = getArg<int>   (args, "seq",        0);
    const int    hbias      = getArg<int>   (args, "hbias",      0);
    // rbias:=1: common range bias as a head coordinate (mppi_solver.cuh); prior sd sig_rb0, random walk q_rb (m/sqrt(s))
    const int    rbias      = getArg<int>   (args, "rbias",      0);
    const int    arr_fixed  = getArg<int>   (args, "arr_fixed",  0);
    const double sig_rb0    = getArg<double>(args, "sig_rb0",    0.05);
    const double q_rb       = getArg<double>(args, "q_rb",       0.005);
    double rb_head = 0.0;
    const int    rb         = getArg<int>   (args, "rb",         0);   // with seq:=1: integrate the head/bias analytically
    const double seq_alpha  = getArg<double>(args, "seq_alpha",  0.9);
    const double seq_infl   = getArg<double>(args, "seq_inflate", 1.2);
    const int    seq_stages = getArg<int>   (args, "seq_stages", 20);
    const double q_ba     = getArg<double>(args, "q_ba",     0.003);   // m/s^3/sqrt(Hz)   bias random walk
    const double q_bg     = getArg<double>(args, "q_bg",     0.0001);  // rad/s^2/sqrt(Hz)
    const double sig_ba0  = getArg<double>(args, "sig_ba0",  0.2);     // initial bias std
    const double sig_bg0  = getArg<double>(args, "sig_bg0",  0.01);
    const double var_floor_frac = getArg<double>(args, "bias_var_floor", 1e-4);  // floor of Sigma_b as a fraction of sig_*0^2
    // carry-over of the window's bias increment: only the fraction of the window that leaves it per solve
    // (dt / (T dt) = 1/T), like the head state that advances one step. bias_gain:=-1 selects 1/T.
    double bias_gain = getArg<double>(args, "bias_gain", -1.0);
    if (bias_gain < 0.0) bias_gain = 1.0 / T;
    // 0: keep Sigma_b at its initial width (the window posterior is not independent evidence per solve),
    // 1: replace Sigma_b by the window posterior variance (collapses: overlapping windows are re-counted)
    const int bias_shrink = getArg<int>(args, "bias_shrink", 0);
    // v3 split prior: the window-constant (DC, k = 0) mode gets its own std in physical units (m/s^2, rad/s),
    // e.g. the bias uncertainty from a static calibration, while k >= 1 modes keep the white-noise density.
    // dc_*:=-1 keeps the white-noise density for the DC mode too (v2). carry_bias:=0 disables the carried
    // bias state (delta_b prior collapsed to ~0, no carry-over).
    const double dc_acc     = getArg<double>(args, "dc_acc",     -1.0);
    const double dc_gyr     = getArg<double>(args, "dc_gyr",     -1.0);
    const int    carry_bias = getArg<int>   (args, "carry_bias", 1);
    // head-velocity increment delta v (input_space_v2.md section 14): sampled per window with prior std
    // sqrt(sigma_b^2 T^2 + q_acc^2 T) where sigma_b = dc_acc (or q_acc if dc_acc < 0); carried into the head at 1/T per solve
    const int    dv_on   = getArg<int>   (args, "dv",      0);
    double       dv_gain = getArg<double>(args, "dv_gain", -1.0);
    if (dv_gain < 0.0) dv_gain = 1.0 / T;
    // proposal sigma of the constant per-rollout offset, as a multiple of the cost sigma.
    // prop_scale:=-1 selects the value the cost implies for a constant offset, 1/sqrt(T):
    // then the proposal equals the IMU prior and the weights should carry the UWB term only (w_acc:=0 w_gyr:=0).
    double prop_scale = getArg<double>(args, "prop_scale", 1.0);
    if (prop_scale < 0.0) prop_scale = 1.0 / std::sqrt(static_cast<double>(T));

    const double gravity[3] = {0.0, 0.0, -9.81};

    MPPISolver solver(N, T, dt, gamma, S.anchors.data(), gravity, sigma,
                      S.tag_offsets.data(), S.num_anchors, S.num_tags,
                      static_cast<unsigned long>(getArg<int>(args, "seed", SEED)));
    solver.w_uwb_ = w_uwb;
    solver.w_acc_ = w_acc;
    solver.w_gyr_ = w_gyr;
    if (S.tdoa) {   // dataset util: every "anchor" is an anchor-pair slot measuring |tag - B| - |tag - A|
        if (S.num_anchors > 64 || static_cast<int>(S.anchorsB.size()) != 3 * S.num_anchors) {
            std::fprintf(stderr, "[kims] TDOA: %d slots (max 64) and %zu anchor-B values\n", S.num_anchors, S.anchorsB.size()); return -1;
        }
        if (rb) { std::fprintf(stderr, "[kims] TDOA is not implemented in the rb path (use rb:=0)\n"); return -1; }
        solver.setTdoa(S.anchorsB.data(), S.num_anchors);
        std::printf("[kims] TDOA: %d anchor-pair slots, measurement |tag - B| - |tag - A|\n", S.num_anchors);
    }
    {
        double sigma_prop[DIM_U];
        for (int k = 0; k < DIM_U; ++k) sigma_prop[k] = sigma[k] * prop_scale;
        mppi_set_proposal_sigma(sigma_prop);
    }
    mppi_set_ctrl_mode(ctrl_mode, sigma_imu_a, sigma_imu_g);
    {   // range likelihood: uwb_model:=1 uses the LOS/NLOS mixture fitted on the calibration flights
        const int uwb_model = getArg<int>(args, "uwb_model", 0);
        const double mix[6] = {getArg<double>(args, "mix_eps", 0.2671), getArg<double>(args, "mix_mu1", -0.0069),
                               getArg<double>(args, "mix_s1", 0.0694), getArg<double>(args, "mix_mu2", 0.4402),
                               getArg<double>(args, "mix_s2", 0.3252), getArg<double>(args, "mix_tau", 0.0724)};
        mppi_set_uwb_model(uwb_model, mix);
        if (uwb_model > 0) std::printf("[kims] range likelihood model %d: eps %.3f, LOS (%.4f, %.4f, tau %.4f), NLOS (%.3f, %.3f)\n",
                                        uwb_model, mix[0], mix[1], mix[2], mix[5], mix[3], mix[4]);
    }
    if (K > 0) {
        solver.w_acc_ = 0.0; solver.w_gyr_ = 0.0;      // proposal = prior, weights = likelihood only
        solver.dv_on_ = dv_on; solver.dv_gain_ = dv_on ? dv_gain : 0.0;
        solver.head_on_ = head_on;
        { double q6[6]; for (int k = 0; k < 6; ++k) q6[k] = q_axis[k]; solver.setHeadQ(q6); }
        solver.setHeadSeed(static_cast<unsigned long>(getArg<int>(args, "seed", SEED)) * 7919UL + 17UL);
        solver.setBasis(K, iters, ess_min);
        if (opt > 0 && dv_on) { std::fprintf(stderr, "[kims] opt with dv is not supported\n"); return -1; }
        solver.prior_in_weight_ = getArg<int>(args, "prior_w", 0);   // 1: charge the prior in the weights (MAP-like)
        std::printf("[kims] basis mode: K=%d (dim %d) iters=%d ess_min=%.2f q_acc=%.4f q_gyr=%.5f q_ba=%.4f q_bg=%.5f sig_b0=(%.3f, %.4f)\n",
                    K, solver.basisDim(), iters, ess_min, q_acc, q_gyr, q_ba, q_bg, sig_ba0, sig_bg0);
    }
    mppi_set_deltas(uwb_delta, ctrl_delta);
    mppi_set_floor_penalty(floor_pen);
    mppi_set_floor_hard(getArg<int>(args, "floor_hard", 0));

    const int num_rng = solver.numRanges();

    const double t0 = std::min(S.imu.front().timestamp, S.uwb.front().timestamp);
    for (auto& d : S.imu) d.timestamp -= t0;
    for (auto& d : S.uwb) d.timestamp -= t0;
    {   // per tag-anchor LOS range offsets measured on the calibration flights, subtracted here
        const std::string off_file = getArg<std::string>(args, "range_offsets", std::string(""));
        if (!off_file.empty()) {
            std::ifstream fo(off_file);
            if (!fo) { std::fprintf(stderr, "[kims] cannot open range_offsets %s\n", off_file.c_str()); return 1; }
            std::vector<double> off(S.num_tags * S.num_anchors, 0.0);
            std::string line; int n_off = 0;
            while (std::getline(fo, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream is(line); int ti, ai; double v;
                if (is >> ti >> ai >> v && ti >= 0 && ti < S.num_tags && ai >= 0 && ai < S.num_anchors) { off[ti * S.num_anchors + ai] = v; ++n_off; }
            }
            for (auto& d : S.uwb) if (d.tag_idx >= 0 && d.anc_idx >= 0) d.range -= off[d.tag_idx * S.num_anchors + d.anc_idx];
            std::printf("[kims] range offsets: %d pairs from %s\n", n_off, off_file.c_str());
        }
    }

    std::printf("[kims] IMU [%.3f, %.3f] s | UWB [%.3f, %.3f] s | N=%d T=%d dt=%.3f gamma=%.3f prop_scale=%.4f\n",
                S.imu.front().timestamp, S.imu.back().timestamp,
                S.uwb.front().timestamp, S.uwb.back().timestamp, N, T, dt, gamma, prop_scale);

    const bool z_down = getArg<int>(args, "imu_z_down", P.imu_z_down) != 0;
    double x0[DIM_X] = {};
    x0[0] = S.init_p.x(); x0[1] = S.init_p.y(); x0[2] = S.init_p.z();
    x0[6]  = 1.0;
    x0[10] = z_down ? -1.0 : 1.0;
    x0[14] = z_down ? -1.0 : 1.0;
    // Initial alignment from gravity (standard IMU initialisation, no tuning): average the accelerometer over
    // the first align_still seconds (vehicle at rest) and rotate the base attitude so that R0 * a_mean = +g up.
    // Yaw stays as in the base attitude. Removes the mounting tilt from what the DC-mode prior must cover.
    const double align_still = getArg<double>(args, "align_still", 0.0);
    // static-alignment statistics for the carried-bias head state (hbias:=1): gyro mean (= gyro bias at rest, Earth
    // rate neglected) and its standard error, accelerometer magnitude, body up-axis
    Eigen::Vector3d still_gm = Eigen::Vector3d::Zero(), still_gse = Eigen::Vector3d::Zero(), still_zb(0, 0, 1);
    double still_am_norm = 9.81, still_tilt_se = 1e-4, still_bz_se = 1e-3; bool still_ok = false;
    Eigen::Vector3d still_ba = Eigen::Vector3d::Zero();   // accelerometer bias at rest: (|a| - g) along the measured gravity direction
    if (align_still > 0.0) {
        Eigen::Vector3d am = Eigen::Vector3d::Zero(); int na = 0;
        Eigen::Vector3d gs = Eigen::Vector3d::Zero(), gs2 = Eigen::Vector3d::Zero(), as2 = Eigen::Vector3d::Zero();
        for (const auto& d : S.imu) {
            if (d.timestamp > S.imu.front().timestamp + align_still) break;
            am += d.acc; ++na; gs += d.gyr; gs2 += d.gyr.cwiseProduct(d.gyr); as2 += d.acc.cwiseProduct(d.acc);
        }
        if (na > 1) {
            still_gm = gs / na;
            const Eigen::Vector3d var = (gs2 / na - still_gm.cwiseProduct(still_gm)).cwiseMax(0.0) * na / (na - 1.0);
            still_gse = (var / na).cwiseSqrt();
            const Eigen::Vector3d amn = am / na;
            const Eigen::Vector3d avar = (as2 / na - amn.cwiseProduct(amn)).cwiseMax(0.0) * na / (na - 1.0);
            still_tilt_se = std::sqrt(avar.maxCoeff() / na) / 9.81;   // alignment tilt standard error
            still_bz_se = std::sqrt(avar.maxCoeff() / na);              // standard error of |a| - g (vertical acc bias)
        }
        if (na > 0) {
            am /= na;
            Eigen::Matrix3d Rb = Eigen::Matrix3d::Identity();
            if (z_down) { Rb(1,1) = -1.0; Rb(2,2) = -1.0; }
            const Eigen::Vector3d ub = (Rb * am).normalized();
            const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(ub, Eigen::Vector3d(0, 0, 1));
            Eigen::Matrix3d R0 = q.toRotationMatrix() * Rb;
            // optional initial yaw (deg, about world z) for diagnostics of the global-initialisation problem
            const double init_yaw = getArg<double>(args, "init_yaw", 0.0);
            if (init_yaw != 0.0) R0 = Eigen::AngleAxisd(init_yaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) x0[6 + 3*i + j] = R0(i, j);
            still_zb = R0.transpose() * Eigen::Vector3d(0, 0, 1); still_am_norm = am.norm(); still_ok = true;
            const Eigen::Vector3d resid = R0 * am - Eigen::Vector3d(0, 0, 9.81);
            std::printf("[kims] gravity alignment over %.1f s (%d samples): tilt %.2f deg, |a|-g = %+.3f m/s^2, residual after alignment %.3f m/s^2\n",
                        align_still, na, std::acos(std::min(1.0, ub.z())) * 180.0 / M_PI, am.norm() - 9.81, resid.norm());
            still_ba = (am.norm() - 9.81) * am.normalized();
        }
    }
    // still_bias:=1 (standard static IMU initialisation, no tuning): subtract the rest-phase biases from the whole IMU stream,
    // the gyro mean (Earth rate neglected) and the accelerometer magnitude error |a| - g along the measured gravity direction.
    // The window's DC-mode prior then only has to cover the residual and the in-run drift, not the turn-on bias
    // (NTU VN-100: |a| - g = -0.19 m/s^2 against a 0.1 m/s^2 DC prior).
    const int still_bias = getArg<int>(args, "still_bias", 0);
    if (still_bias && still_ok) {
        for (auto& d : S.imu) { d.acc -= still_ba; d.gyr -= still_gm; }
        std::printf("[kims] still_bias: subtracted acc bias [%.4f %.4f %.4f] m/s^2 and gyro bias [%.5f %.5f %.5f] rad/s from the IMU stream\n",
                    still_ba.x(), still_ba.y(), still_ba.z(), still_gm.x(), still_gm.y(), still_gm.z());
        still_gm.setZero(); still_am_norm = 9.81;   // the hbias prior (if on) sees an unbiased stream
    }

    std::vector<double> U0(T * DIM_U, 0.0), Uopt(T * DIM_U, 0.0);
    double xn[DIM_X] = {};

    double bias[6] = {0, 0, 0, 0, 0, 0};
    double Sb[6]   = {sig_ba0 * sig_ba0, sig_ba0 * sig_ba0, sig_ba0 * sig_ba0,
                      sig_bg0 * sig_bg0, sig_bg0 * sig_bg0, sig_bg0 * sig_bg0};
    const int Dth = solver.basisDim();
    std::vector<double> prior_std(std::max(Dth, 1)), th_mean(std::max(Dth, 1)), th_var(std::max(Dth, 1));
    std::ofstream bias_file;
    if (K > 0) {
        std::string bf = output_file;
        const auto bp = bf.rfind("_pose.txt");
        if (bp != std::string::npos) bf.replace(bp, 9, "_bias.txt"); else bf += ".bias.txt";
        bias_file.open(bf);
        bias_file << "# time ba_x ba_y ba_z bg_x bg_y bg_z sb_ax sb_ay sb_az sb_gx sb_gy sb_gz ess1 ess2 iters dv_x dv_y dv_z\n";
        bias_file << std::fixed; bias_file.precision(6);
    }

    std::ofstream pose_file(output_file);
    if (!pose_file.is_open()) {
        std::fprintf(stderr, "[kims] cannot open output: %s\n", output_file.c_str());
        return -1;
    }
    pose_file << std::fixed;
    pose_file.precision(9);
    pose_file << "# time x y z qx qy qz qw\n";
    std::ofstream flush_file;   // whole-window positions at the first and the last solve (offline trajectory)
    { std::string ff = output_file; const auto bp = ff.rfind("_pose.txt"); if (bp != std::string::npos) ff.replace(bp, 9, "_flush_pose.txt"); else ff += ".flush.txt";
      flush_file.open(ff); flush_file << std::fixed; flush_file.precision(9); flush_file << "# time x y z qx qy qz qw  (window flush: first solve = epochs 0..T-1, last solve = last T epochs)\n"; }
    bool first_solve_done = false;
    std::ofstream now_file;   // lag-0 estimate at the newest window epoch (seq path)
    { std::string nf = output_file; const auto bp = nf.rfind("_pose.txt"); if (bp != std::string::npos) nf.replace(bp, 9, "_now_pose.txt"); else nf += ".now.txt";
      now_file.open(nf); now_file << std::fixed; now_file.precision(9); now_file << "# time x y z qx qy qz qw  (state at the newest window epoch, no lag)\n"; }

    LatestRanges latest(S.num_tags, S.num_anchors);

    kims::BasisOpt bopt(S.anchors, S.num_anchors, S.tag_offsets, S.num_tags, gravity);
    std::vector<double> phi_host;
    if (K > 0) {
        const double Tsb = T * dt;                 // same table as MPPISolver::setBasis (dt at construction)
        phi_host.resize(K * T);
        for (int kk = 0; kk < K; ++kk)
            for (int t = 0; t < T; ++t)
                phi_host[kk*T + t] = (kk == 0) ? 1.0 / std::sqrt(Tsb) : std::sqrt(2.0 / Tsb) * std::cos(M_PI * kk * (t + 0.5) / T);
        bopt.setBasis(K, T, phi_host);
    }
    double measured_dt_last = -1.0;
    // anchor plane (least-squares, normal oriented upward)
    Eigen::Vector3d plane_c = Eigen::Vector3d::Zero(), plane_n(0, 0, 1);
    {
        Eigen::MatrixXd Am(S.num_anchors, 3);
        for (int a = 0; a < S.num_anchors; ++a) for (int k = 0; k < 3; ++k) Am(a, k) = S.anchors[3*a + k];
        plane_c = Am.colwise().mean();
        Eigen::MatrixXd Cm = Am.rowwise() - plane_c.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cm, Eigen::ComputeThinV);
        plane_n = svd.matrixV().col(2);
        if (plane_n.z() < 0) plane_n = -plane_n;
    }
    solver.setPlane(plane_c.data(), plane_n.data());
    solver.seq_refl_ = getArg<double>(args, "seq_refl", 0.0);   // reflected proposal component (mppi_solver.cuh)
    auto reflect = [&](const double* a, double* b) {
        for (int i = 0; i < DIM_X; ++i) b[i] = a[i];
        Eigen::Vector3d p(a[0], a[1], a[2]), v(a[3], a[4], a[5]);
        p -= 2.0 * (p - plane_c).dot(plane_n) * plane_n;
        v -= 2.0 * v.dot(plane_n) * plane_n;
        for (int k = 0; k < 3; ++k) { b[k] = p[k]; b[3 + k] = v[k]; }
    };
    double x0B[DIM_X], xnB[DIM_X];
    double scoreA = 0.0, scoreB = 0.0;
    long reseeds = 0, leader_switches = 0; int leader_prev = 0;
    std::ofstream branch_file;
    if (branches == 2) {
        std::string bf = output_file;
        const auto bp = bf.rfind("_pose.txt");
        if (bp != std::string::npos) bf.replace(bp, 9, "_branch.txt"); else bf += ".branch.txt";
        branch_file.open(bf);
        branch_file << "# time scoreA-scoreB leader hA hB logpredA logpredB reseed\n";
        branch_file << std::fixed; branch_file.precision(5);
        std::printf("[kims] two branches: plane point (%.3f, %.3f, %.3f) normal (%.4f, %.4f, %.4f), reseed at %.2f nats\n",
                    plane_c.x(), plane_c.y(), plane_c.z(), plane_n.x(), plane_n.y(), plane_n.z(), reseed_nats);
    }
    bool branchB_init = false;
    // head covariances (row-major 9x9) of branch A and B, and the one-step process-noise floor
    double PA[81] = {0}, PB[81] = {0}, Pfloor[9];
    for (int k = 0; k < 3; ++k) { PA[10*k] = hp0_p*hp0_p; PA[10*(k+3)] = hp0_v*hp0_v; PA[10*(k+6)] = hp0_a*hp0_a; }
    std::copy(PA, PA + 81, PB);
    auto floorP = [&](double* P) { for (int k = 0; k < 9; ++k) P[10*k] = std::max(P[10*k], Pfloor[k]); };
    auto reflectP = [&](const double* Pin, double* Pout) {        // J P J', J = blockdiag(H, H, I), H = I - 2 n n'
        Eigen::Matrix<double, 9, 9> J = Eigen::Matrix<double, 9, 9>::Identity(), P;
        const Eigen::Matrix3d H = Eigen::Matrix3d::Identity() - 2.0 * plane_n * plane_n.transpose();
        J.block<3,3>(0,0) = H; J.block<3,3>(3,3) = H;
        for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) P(r, c) = Pin[9*r + c];
        const Eigen::Matrix<double, 9, 9> Q = J * P * J.transpose();
        for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) Pout[9*r + c] = Q(r, c);
    };
    // sequential mode: per-branch arrival covariance (HB x HB), carried proposal and carried bias
    const int HB = hbias ? 15 : (rbias ? 10 : 9), VB = HB + 6 * K;
    std::vector<double> PHA(HB * HB, 0.0), PHB, muA, SA, muB, SB;
    int validA = 0, validB = 0;
    double biasA[6] = {0, 0, 0, 0, 0, 0}, biasB[6] = {0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 3; ++k) { PHA[(HB+1)*k] = hp0_p*hp0_p; PHA[(HB+1)*(k+3)] = hp0_v*hp0_v; PHA[(HB+1)*(k+6)] = hp0_a*hp0_a; }
    if (hbias) for (int k = 0; k < 3; ++k) { PHA[(HB+1)*(9+k)] = sig_ba0*sig_ba0; PHA[(HB+1)*(12+k)] = sig_bg0*sig_bg0; }
    if (rbias && !hbias) { PHA[(HB+1)*9] = sig_rb0 * sig_rb0; solver.rbias_on_ = 1; solver.rb_q_ = q_rb; }
    if (hbias && still_ok) {
        // static alignment: the gyro mean at rest is the gyro bias (U = IMU + b, so b_g = -mean),
        // known to its standard error; the vertical accelerometer bias is |a| - g along the body up-axis; the horizontal
        // accelerometer bias is not separable from tilt at rest, and the alignment fixes their sum: the residual world
        // acceleration R0 (g dtheta x z_b + db) must vanish, i.e. dtheta = -(z_b x db)/g. So the prior over (dtheta, db_a)
        // is the image of db_a ~ N(0, sig_ba0^2 I) under that map (plus the yaw prior), not two independent blocks.
        for (int k = 0; k < 3; ++k) biasA[3 + k] = -still_gm[k];
        const Eigen::Vector3d zb = still_zb.normalized();
        for (int k = 0; k < 3; ++k) biasA[k] = -(still_am_norm - 9.81) * zb[k];
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();                    // dtheta = M db, M = -[z_b]x / g
        M << 0.0, zb.z(), -zb.y(), -zb.z(), 0.0, zb.x(), zb.y(), -zb.x(), 0.0;
        M /= 9.81;
        // the vertical accelerometer bias is measured at rest (|a| - g) to its standard error; only the horizontal part
        // (inseparable from tilt) keeps the turn-on uncertainty sig_ba0
        const Eigen::Matrix3d Pbb = sig_ba0 * sig_ba0 * (Eigen::Matrix3d::Identity() - zb * zb.transpose())
                                  + still_bz_se * still_bz_se * zb * zb.transpose();
        Eigen::Matrix3d Ptt = M * Pbb * M.transpose();
        // yaw (about the body up-axis) is not set by the alignment: keep the head attitude prior along z_b
        Ptt += hp0_a * hp0_a * zb * zb.transpose();
        Ptt += still_tilt_se * still_tilt_se * Eigen::Matrix3d::Identity();   // alignment noise keeps P regular
        const Eigen::Matrix3d Ptb = M * Pbb;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
            PHA[HB*(6 + r) + 6 + c] = Ptt(r, c);
            PHA[HB*(6 + r) + 9 + c] = Ptb(r, c); PHA[HB*(9 + c) + 6 + r] = Ptb(r, c);
            PHA[HB*(9 + r) + 9 + c] = Pbb(r, c);
            PHA[HB*(12 + r) + 12 + c] = (r == c) ? std::max(still_gse[r] * still_gse[r], 1e-12) : 0.0;
        }
        std::printf("[kims] static bias init: gyro bias %.5f %.5f %.5f rad/s (se %.1e %.1e %.1e), acc bias z %.4f m/s^2, alignment tilt se %.1e rad, vertical acc bias se %.1e m/s^2, tilt-bias prior correlated\n",
                    biasA[3], biasA[4], biasA[5], still_gse[0], still_gse[1], still_gse[2], biasA[2], still_tilt_se, still_bz_se);
    }
    PHB = PHA;
    // reflect a carried proposal (mean and covariance over V = HB + 6K, head block first) through the anchor plane:
    // J = blockdiag(Hn, Hn, I, ...) on (dp, dv, dtheta, bias, inputs); only position and velocity change sign in the normal
    auto reflectProp = [&](std::vector<double>& mu, std::vector<double>& S) {
        if (static_cast<int>(mu.size()) != VB || static_cast<int>(S.size()) != VB * VB) return;
        Eigen::MatrixXd J = Eigen::MatrixXd::Identity(VB, VB);
        const Eigen::Matrix3d Hn = Eigen::Matrix3d::Identity() - 2.0 * plane_n * plane_n.transpose();
        J.block(0, 0, 3, 3) = Hn; J.block(3, 3, 3, 3) = Hn;
        Eigen::Map<Eigen::VectorXd> m(mu.data(), VB); Eigen::Map<Eigen::MatrixXd> Sm(S.data(), VB, VB);
        m = J * m; Sm = (J * Sm * J.transpose()).eval();
    };
    auto reflectPH = [&](const std::vector<double>& Pin) {        // J P J', J = blockdiag(Hn, Hn, I3, I6)
        Eigen::MatrixXd J = Eigen::MatrixXd::Identity(HB, HB), P(HB, HB);
        const Eigen::Matrix3d Hn = Eigen::Matrix3d::Identity() - 2.0 * plane_n * plane_n.transpose();
        J.block(0, 0, 3, 3) = Hn; J.block(3, 3, 3, 3) = Hn;
        for (int r = 0; r < HB; ++r) for (int c = 0; c < HB; ++c) P(r, c) = Pin[HB*r + c];
        const Eigen::MatrixXd Q = J * P * J.transpose();
        std::vector<double> out(HB * HB);
        for (int r = 0; r < HB; ++r) for (int c = 0; c < HB; ++c) out[HB*r + c] = Q(r, c);
        return out;
    };
    if (seq && (head_on == 0 || K <= 0 || dv_on)) { std::fprintf(stderr, "[kims] seq:=1 needs K > 0, head:=2 and dv:=0\n"); return 1; }
    if (seq) {
        const double qb6[6] = {q_ba, q_ba, q_ba, q_bg, q_bg, q_bg};
        solver.rb_on_ = rb;
        solver.rb_verify_ = getArg<int>(args, "rb_verify", 0);
        solver.bias_filt_ = getArg<int>(args, "bias_filt", 0);
        solver.bias_gain_ = bias_gain;                        // 1/T unless bias_gain:= is given
        { double hg = getArg<double>(args, "head_gain", 1.0); solver.head_gain_ = hg < 0.0 ? 1.0 / T : hg; }
        solver.head_filt_ = getArg<int>(args, "head_filt", 0);
        solver.lm_on_ = getArg<int>(args, "lm", 0);               // matched LM baseline (mppi_solver.cuh)
        solver.lm_iters_ = getArg<int>(args, "lm_iters", 8);
        solver.lm_h_ = getArg<double>(args, "lm_h", 0.02);
        solver.seq_dump_ = getArg<int>(args, "seq_dump", 0);
        solver.seq_dump_file_ = getArg<std::string>(args, "seq_dump_file", "seq_dump.txt");
        solver.seq_strip_from_ = getArg<int>(args, "seq_strip_from", 0); solver.seq_strip_to_ = getArg<int>(args, "seq_strip_to", 0); solver.seq_strip_step_ = std::max(1, getArg<int>(args, "seq_strip_step", 1));
        solver.setHeadQb(qb6); solver.seq_alpha_ = seq_alpha; solver.seq_inflate_ = seq_infl; solver.seq_max_stages_ = seq_stages;
        solver.seq_debug_ = getArg<int>(args, "seq_debug", 0);
        solver.rollout_fp32_ = getArg<int>(args, "rollout_fp32", 0);
        solver.seq_headfix_ = getArg<int>(args, "seq_headfix", 0);
        solver.seq_shrink_carry_ = getArg<int>(args, "seq_shrink", 1);
        solver.seq_newmass_ = getArg<int>(args, "seq_newmass", 1);
        solver.seq_done_ = getArg<double>(args, "seq_done", 0.25);
        solver.seq_ewma_ = getArg<int>(args, "seq_ewma", 0);
        solver.seq_shrink_k_ = getArg<double>(args, "seq_shrink_k", 1.0);
        std::printf("[kims] sequential proposal: head dim %d, alpha %.2f, inflate %.2f, max stages %d, q_ba %.4g q_bg %.4g\n",
                    HB, seq_alpha, seq_infl, seq_stages, q_ba, q_bg);
    }
    std::ofstream head_file;
    if (head_on) {
        std::string hf = output_file;
        const auto hp = hf.rfind("_pose.txt");
        if (hp != std::string::npos) hf.replace(hp, 9, "_head.txt"); else hf += ".head.txt";
        head_file.open(hf);
        head_file << "# time sqrtP_A(px py pz vx vy vz ax ay az) sqrtP_B(...) dmean_A(9, head correction) info_A(pxx pyy pzz, 1/m^2) fitR2_A\n";
        head_file << std::fixed; head_file.precision(6);
    }
    long opt_iters_sum = 0, opt_solves = 0, opt_fail = 0;

    std::vector<double> ranges_flat(T * num_rng, -1.0);
    std::vector<double> stds_flat  (T * num_rng,  0.1);
    std::vector<double> accgyr_flat(T * 6,        0.0);

    std::deque<std::vector<double>> rng_window, std_window;
    std::vector<char> fresh(num_rng, 0);
    std::deque<ImuData>             imu_window;
    std::deque<double>              ts_window;
    double head_ts = -1.0;
    size_t imu_mi = 0, imu_hint2 = 0;
    std::vector<double> step_dts(T, dt), phi_tt(K > 0 ? K * T : 0, 0.0);

    size_t uwb_idx = 0, imu_hint = 0;
    int    solve_count = 0;
    double prev_ts = -1.0;
    std::vector<double> solve_times;
    std::vector<double> ess_log;

    const auto wall_start = std::chrono::high_resolution_clock::now();

    // group_same_ts:=1 puts every event with the same timestamp into one epoch (dataset util: util_prepare.py bins the
    // TDOA stream to a fixed epoch rate and stamps each bin's events with one time); default 0 keeps one event per epoch
    const int group_ts = getArg<int>(args, "group_same_ts", 0);
    while (uwb_idx < S.uwb.size()) {
        latest.update(S.uwb[uwb_idx]);
        {
            const auto& m = S.uwb[uwb_idx];
            if (m.tag_idx >= 0 && m.anc_idx >= 0) fresh[m.tag_idx * S.num_anchors + m.anc_idx] = 1;
        }
        const double ts = S.uwb[uwb_idx].timestamp;
        ++uwb_idx;
        if (group_ts)
            while (uwb_idx < S.uwb.size() && S.uwb[uwb_idx].timestamp <= ts + 1e-9) {
                const auto& m2 = S.uwb[uwb_idx];
                latest.update(m2);
                if (m2.tag_idx >= 0 && m2.anc_idx >= 0) fresh[m2.tag_idx * S.num_anchors + m2.anc_idx] = 1;
                ++uwb_idx;
            }

        if (ts <= prev_ts + 1e-6) continue;

        ImuData imu = interpolateImu(S.imu, ts, imu_hint);
        if (true_time && prev_ts > 0.0) {
            // mean IMU sample over (prev_ts, ts]: the zero-order-hold input that integrates this interval
            while (imu_mi < S.imu.size() && S.imu[imu_mi].timestamp <= prev_ts) ++imu_mi;
            Eigen::Vector3d sa = Eigen::Vector3d::Zero(), sg = Eigen::Vector3d::Zero(); int na = 0;
            for (size_t j = imu_mi; j < S.imu.size() && S.imu[j].timestamp <= ts; ++j) { sa += S.imu[j].acc; sg += S.imu[j].gyr; ++na; }
            if (na > 0) { imu.acc = sa / na; imu.gyr = sg / na; }
            else imu = interpolateImu(S.imu, 0.5 * (prev_ts + ts), imu_hint2);
            imu.timestamp = ts;
        }

        std::vector<double> rng(num_rng), std_v(num_rng);
        latest.copyTo(rng.data());
        latest.copyStdsTo(std_v.data());
        if (fresh_only) {
            for (int i = 0; i < num_rng; ++i) if (!fresh[i]) rng[i] = -1.0;
            std::fill(fresh.begin(), fresh.end(), 0);
        }
        rng_window.push_back(rng);
        std_window.push_back(std_v);
        imu_window.push_back(imu);
        ts_window.push_back(ts);

        if (static_cast<int>(rng_window.size()) > T) {
            rng_window.pop_front();
            std_window.pop_front();
            imu_window.pop_front();
            head_ts = ts_window.front(); ts_window.pop_front();
        }
        if (static_cast<int>(rng_window.size()) < T || !latest.allValid()) {
            prev_ts = ts;
            continue;
        }

        for (int t = 0; t < T; ++t) {
            std::copy(rng_window[t].begin(), rng_window[t].end(),
                      ranges_flat.data() + t * num_rng);
            if (uwb_std_fixed > 0.0)
                for (int i = 0; i < num_rng; ++i) stds_flat[t * num_rng + i] = uwb_std_fixed;
            else
                std::copy(std_window[t].begin(), std_window[t].end(),
                          stds_flat.data() + t * num_rng);
            accgyr_flat[t*6+0] = imu_window[t].acc.x();
            accgyr_flat[t*6+1] = imu_window[t].acc.y();
            accgyr_flat[t*6+2] = imu_window[t].acc.z();
            accgyr_flat[t*6+3] = imu_window[t].gyr.x();
            accgyr_flat[t*6+4] = imu_window[t].gyr.y();
            accgyr_flat[t*6+5] = imu_window[t].gyr.z();
        }

        if (true_time) {
            // per-step spacing: step 0 runs from the head epoch to the first window epoch
            for (int t = 0; t < T; ++t) {
                double g = t == 0 ? (head_ts > 0.0 ? ts_window[0] - head_ts : ts_window[1] - ts_window[0])
                                  : ts_window[t] - ts_window[t - 1];
                step_dts[t] = std::min(std::max(g, 1e-4), 1.0);
            }
            solver.setStepDts(step_dts.data()); measured_dt_last = step_dts[0];
            if (K > 0) {                                      // basis on the true step midpoints over [0, Ts]
                double Tsum = 0.0; for (int t = 0; t < T; ++t) Tsum += step_dts[t];
                double tau = 0.0;
                for (int t = 0; t < T; ++t) {
                    const double mid = tau + 0.5 * step_dts[t]; tau += step_dts[t];
                    for (int k = 0; k < K; ++k)
                        phi_tt[k * T + t] = (k == 0) ? 1.0 / std::sqrt(Tsum) : std::sqrt(2.0 / Tsum) * std::cos(M_PI * k * mid / Tsum);
                }
                solver.setPhi(phi_tt.data());
                for (int k = 0; k < K * T; ++k) phi_host[k] = phi_tt[k];
            }
        } else if (T >= 2 && prev_ts > 0.0) {
            const double measured_dt = ts - prev_ts;
            if (measured_dt > 1e-4 && measured_dt < 1.0) { solver.setDt(measured_dt); measured_dt_last = measured_dt; }
        }

        for (int t = 0; t < T; ++t)
            for (int k = 0; k < 6; ++k)
                U0[t*6 + k] = accgyr_flat[t*6 + k] + (K > 0 ? bias[k] : 0.0);

        const auto t_solve = std::chrono::high_resolution_clock::now();
        if (K > 0) {
            double dmA[9] = {0}, infA[3] = {0}, r2A = -1.0;       // head:=2 diagnostics of branch A
            double seq_ess1 = 0.0, seq_ess2 = 0.0; int seq_passes = 0;   // seq:=1 diagnostics
            double Ts = T * dt;
            if (true_time) { Ts = 0.0; for (int t = 0; t < T; ++t) Ts += step_dts[t]; }
            for (int k = 0; k < 6; ++k) {
                const double qb = k < 3 ? q_ba : q_bg;
                prior_std[k] = std::sqrt(Sb[k] + qb * qb * Ts);
            }
            for (int kk = 0; kk < K; ++kk)
                for (int ch = 0; ch < 6; ++ch) {
                    double sd = q_axis[ch];
                    if (kk == 0) {
                        const double dc = ch < 3 ? dc_acc : dc_gyr;
                        if (dc > 0.0) sd = dc * std::sqrt(Ts);      // c_0 * phi_0 = c_0 / sqrt(Ts) has std dc
                    }
                    prior_std[6 + 6*kk + ch] = sd;
                }
            if (!carry_bias) for (int k = 0; k < 6; ++k) prior_std[k] = 1e-6;
            if (dv_on) {
                const double sb = dc_acc > 0.0 ? dc_acc : q_acc;
                const double sv = std::sqrt(sb * sb * Ts * Ts + q_acc * q_acc * Ts);
                for (int i = 0; i < 3; ++i) prior_std[6 + 6*K + i] = sv;
            }
            if (opt > 0) {
                std::vector<int> active;
                for (int d = (carry_bias ? 0 : 6); d < 6 + 6*K; ++d) active.push_back(d);
                const double cur_dt = measured_dt_last > 0.0 ? measured_dt_last : dt;
                std::vector<double> th;
                const auto r = bopt.solve(opt, opt_iter, gamma, uwb_delta > 0.0 ? uwb_delta : 3.0, x0, U0.data(), cur_dt,
                                          ranges_flat.data(), stds_flat.data(), prior_std, active, th);
                double u0[6];
                for (int ch = 0; ch < 6; ++ch) {
                    double v = U0[ch] + th[ch];
                    for (int kk = 0; kk < K; ++kk) v += th[6 + 6*kk + ch] * phi_host[kk*T + 0];
                    u0[ch] = v;
                }
                bopt.stepHead(x0, u0, cur_dt, xn);
                for (int d = 0; d < Dth; ++d) { th_mean[d] = th[d]; th_var[d] = 0.0; }
                opt_iters_sum += r.iters; opt_solves += 1; if (!r.ok) ++opt_fail;
            } else if (seq) {
            solver.setBasisPrior(prior_std.data());
            auto buildU0 = [&](const double* b, std::vector<double>& U) {
                for (int t = 0; t < T; ++t) for (int k = 0; k < 6; ++k) U[t*6 + k] = accgyr_flat[t*6 + k] + b[k];
            };
            std::vector<double> U0A(T * DIM_U), U0B(T * DIM_U);
            buildU0(biasA, U0A);
            solver.setHeadCovN(PHA.data(), HB); solver.setProposal(muA.data(), SA.data(), validA);
            solver.flush_ = (!rb && (!first_solve_done || uwb_idx >= S.uwb.size())) ? 1 : 0;
            if (rb) solver.solveRB(xn, x0, U0A.data(), ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
            else solver.solveSeq(xn, x0, U0A.data(), ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
            if (solver.flush_ && flush_file.is_open() && static_cast<int>(solver.flush_pos_.size()) == 3 * T) {
                for (int t = 0; t < T; ++t)
                    flush_file << (ts_window[t] + t0) << " " << solver.flush_pos_[3*t] << " " << solver.flush_pos_[3*t+1] << " " << solver.flush_pos_[3*t+2] << " 0 0 0 1\n";
            }
            first_solve_done = true; solver.flush_ = 0;
            if (!arr_fixed) PHA = solver.last_PH_;   // arr_fixed:=1 (ablation): arrival covariance kept at its initial value
            muA = solver.prop_mu_; SA = solver.prop_S_; validA = solver.prop_valid_;
            if (hbias) for (int k = 0; k < 6; ++k) biasA[k] += solver.last_db_[k];
            if (rbias) { rb_head += solver.last_drb_; solver.rb_head_ = rb_head; }
            for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) PA[9*r + c] = PHA[HB*r + c];
            r2A = solver.last_fit_r2_;
            for (int k = 0; k < 9; ++k) dmA[k] = solver.last_dmean_[k];
            for (int k = 0; k < 3; ++k) infA[k] = solver.last_A_[10*k];
            seq_ess1 = solver.last_ess1_; seq_ess2 = solver.last_ess2_; seq_passes = solver.last_iters_;
            const double essA1 = solver.last_ess1_; const int passA = solver.last_iters_;
            if (branches == 2) {
                const double lpA = solver.last_logpred_;
                if (!branchB_init) { reflect(x0, x0B); PHB = reflectPH(PHA); std::copy(biasA, biasA + 6, biasB); validB = 0; branchB_init = true;
                                     if (rb) { muB = muA; SB = SA; validB = validA; } else { muB = muA; SB = SA; reflectProp(muB, SB); validB = validA; } }
                buildU0(biasB, U0B);
                solver.setHeadCovN(PHB.data(), HB); solver.setProposal(muB.data(), SB.data(), validB);
                if (rb) solver.solveRB(xnB, x0B, U0B.data(), ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
                else solver.solveSeq(xnB, x0B, U0B.data(), ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
                PHB = solver.last_PH_; muB = solver.prop_mu_; SB = solver.prop_S_; validB = solver.prop_valid_;
                if (hbias) for (int k = 0; k < 6; ++k) biasB[k] += solver.last_db_[k];
                for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) PB[9*r + c] = PHB[HB*r + c];
                seq_passes += solver.last_iters_;
                const double essB1 = solver.last_ess1_; const int passB = solver.last_iters_;
                const double lpB = solver.last_logpred_;
                scoreA += lpA; scoreB += lpB;
                const double mx = std::max(scoreA, scoreB); scoreA -= mx; scoreB -= mx;
                int reseed = 0;
                if (scoreA - scoreB > reseed_nats) {
                    reflect(xn, xnB); PHB = reflectPH(PHA); std::copy(biasA, biasA + 6, biasB); validB = 0;
                    if (rb) { muB = muA; SB = SA; validB = validA; }   // the proposal holds only input coefficients: hand it over
                    else { muB = muA; SB = SA; reflectProp(muB, SB); validB = validA; }   // pure sampling: hand over the mirrored proposal
                    scoreB = scoreA - reseed_nats; reseed = 2; ++reseeds;
                } else if (scoreB - scoreA > reseed_nats) {
                    reflect(xnB, xn); PHA = reflectPH(PHB); std::copy(biasB, biasB + 6, biasA); validA = 0;
                    if (rb) { muA = muB; SA = SB; validA = validB; }
                    else { muA = muB; SA = SB; reflectProp(muA, SA); validA = validB; }
                    scoreA = scoreB - reseed_nats; reseed = 1; ++reseeds;
                }
                const int leader = scoreA >= scoreB ? 0 : 1;
                if (leader != leader_prev) ++leader_switches;
                leader_prev = leader;
                const Eigen::Vector3d pA(xn[0], xn[1], xn[2]), pB(xnB[0], xnB[1], xnB[2]);
                branch_file << (ts + t0) << " " << (scoreA - scoreB) << " " << leader << " "
                            << (pA - plane_c).dot(plane_n) << " " << (pB - plane_c).dot(plane_n) << " "
                            << lpA << " " << lpB << " " << reseed << " " << essA1 << " " << passA << " " << essB1 << " " << passB << "\n";
            }
            for (int k = 0; k < 6; ++k) { bias[k] = biasA[k]; if (hbias) Sb[k] = PHA[(HB+1)*(9+k)]; }
            } else {
            solver.setBasisPrior(prior_std.data());
            if (head_on) {
                const double h = measured_dt_last > 0.0 ? measured_dt_last : dt;
                for (int k = 0; k < 3; ++k) {
                    Pfloor[k]     = q_axis[k] * q_axis[k] * h * h * h / 3.0;
                    Pfloor[3 + k] = q_axis[k] * q_axis[k] * h;
                    Pfloor[6 + k] = q_axis[3 + k] * q_axis[3 + k] * h;
                }
                solver.setHeadCov(PA);
            }
            if (branches == 2 && crn) solver.saveRng();
            solver.solveBasis(Uopt.data(), xn, x0, U0.data(),
                              ranges_flat.data(), stds_flat.data(), accgyr_flat.data(),
                              th_mean.data(), th_var.data());
            if (head_on) { std::copy(solver.last_P1_, solver.last_P1_ + 81, PA); if (head_on == 1) floorP(PA); }
            r2A = solver.last_fit_r2_;
            for (int k = 0; k < 9; ++k) dmA[k] = solver.last_dmean_[k];
            for (int k = 0; k < 3; ++k) infA[k] = solver.last_A_[10*k];
            if (branches == 2) {
                const double lpA = crn ? solver.last_logpred1_ : solver.last_logpred_;
                if (crn) solver.restoreRng();
                if (!branchB_init) { reflect(x0, x0B); branchB_init = true; }
                if (head_on) solver.setHeadCov(PB);
                std::vector<double> UoptB(T * DIM_U), thmB(th_mean.size()), thvB(th_var.size());
                solver.solveBasis(UoptB.data(), xnB, x0B, U0.data(),
                                  ranges_flat.data(), stds_flat.data(), accgyr_flat.data(),
                                  thmB.data(), thvB.data());
                const double lpB = crn ? solver.last_logpred1_ : solver.last_logpred_;
                if (head_on) { std::copy(solver.last_P1_, solver.last_P1_ + 81, PB); if (head_on == 1) floorP(PB); }
                scoreA += lpA; scoreB += lpB;
                const double mx = std::max(scoreA, scoreB); scoreA -= mx; scoreB -= mx;
                int reseed = 0;
                if (scoreA - scoreB > reseed_nats) { reflect(xn, xnB); if (head_on) reflectP(PA, PB); scoreB = scoreA - reseed_nats; reseed = 2; ++reseeds; }
                else if (scoreB - scoreA > reseed_nats) { reflect(xnB, xn); if (head_on) reflectP(PB, PA); scoreA = scoreB - reseed_nats; reseed = 1; ++reseeds; }
                const int leader = scoreA >= scoreB ? 0 : 1;
                if (leader != leader_prev) ++leader_switches;
                leader_prev = leader;
                const Eigen::Vector3d pA(xn[0], xn[1], xn[2]), pB(xnB[0], xnB[1], xnB[2]);
                branch_file << (ts + t0) << " " << (scoreA - scoreB) << " " << leader << " "
                            << (pA - plane_c).dot(plane_n) << " " << (pB - plane_c).dot(plane_n) << " "
                            << lpA << " " << lpB << " " << reseed << "\n";
            }
            }
            for (int k = 0; k < 6 && !seq; ++k) {
                if (carry_bias) bias[k] += bias_gain * th_mean[k];
                if (bias_shrink) {
                    const double s0 = k < 3 ? sig_ba0 : sig_bg0;
                    Sb[k] = std::max(th_var[k], var_floor_frac * s0 * s0);
                }
            }
            if (head_on) {
                head_file << (ts + t0);
                for (int k = 0; k < 9; ++k) head_file << " " << std::sqrt(std::max(0.0, PA[10*k]));
                for (int k = 0; k < 9; ++k) head_file << " " << (branches == 2 ? std::sqrt(std::max(0.0, PB[10*k])) : 0.0);
                for (int k = 0; k < 9; ++k) head_file << " " << dmA[k];
                for (int k = 0; k < 3; ++k) head_file << " " << infA[k];
                head_file << " " << r2A << "\n";
            }
            bias_file << (ts + t0);
            for (int k = 0; k < 6; ++k) bias_file << " " << bias[k];
            for (int k = 0; k < 6; ++k) bias_file << " " << std::sqrt(Sb[k]);
            if (seq) bias_file << " " << seq_ess1 << " " << seq_ess2 << " " << seq_passes;
            else bias_file << " " << solver.last_ess1_ << " " << solver.last_ess2_ << " " << solver.last_iters_;
            for (int i = 0; i < 3; ++i) bias_file << " " << (dv_on ? th_mean[6 + 6*K + i] : 0.0);
            bias_file << " " << rb_head << "\n";
        } else {
            solver.solve(Uopt.data(), xn, x0, U0.data(),
                         ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
        }
        solve_times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_solve).count());
        ess_log.push_back(solver.last_ess_);

        std::copy(xn, xn + DIM_X, x0);
        double xout[DIM_X];
        std::copy(xn, xn + DIM_X, xout);
        if (branches == 2 && K > 0) {
            std::copy(xnB, xnB + DIM_X, x0B);
            if (leader_prev == 1) std::copy(xnB, xnB + DIM_X, xout);
        }

        const STATE state(xout);
        const Eigen::Quaterniond q(state.R);
        if (seq && !rb && now_file.is_open()) {
            const Eigen::Matrix3d RT = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&solver.last_xT_[6]);
            const Eigen::Quaterniond qT(RT);
            now_file << (ts + t0) << " " << solver.last_xT_[0] << " " << solver.last_xT_[1] << " " << solver.last_xT_[2] << " "
                     << qT.x() << " " << qT.y() << " " << qT.z() << " " << qT.w() << "\n";
        }
        pose_file << (ts + t0) << " "
                  << state.p.x() << " " << state.p.y() << " " << state.p.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

        ++solve_count;
        std::cout << "[kims] #" << std::setw(6) << solve_count
                  << "  t=" << std::fixed << std::setprecision(2) << std::setw(8) << ts
                  << "  pos=[" << std::setprecision(3) << std::setw(8) << state.p.x()
                  << std::setw(8) << state.p.y() << std::setw(8) << state.p.z()
                  << " ]  quat=[" << std::setprecision(4) << std::setw(9) << q.x()
                  << std::setw(9) << q.y() << std::setw(9) << q.z()
                  << std::setw(9) << q.w() << " ]\n";

        prev_ts = ts;
    }
    pose_file.close();

    const double total_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wall_start).count();

    double sum = 0.0, sum2 = 0.0, max_ms = 0.0;
    for (double ms : solve_times) {
        sum += ms; sum2 += ms * ms;
        if (ms > max_ms) max_ms = ms;
    }
    const size_t Nv = solve_times.size();
    const double mean_ms = Nv ? sum / Nv : 0.0;
    const double std_ms  = Nv > 1 ? std::sqrt((sum2 - sum*sum/Nv) / (Nv - 1)) : 0.0;

    if (rb && solver.prof_n_ > 0) {
        const double n = static_cast<double>(solver.prof_n_), np = std::max(solver.prof_[5], 1.0);
        std::printf("[kims] solveRB profile per solve: draw %.2f ms | GPU pass %.2f ms (%.2f per pass, %.2f passes) | host RB %.2f ms | verify %.2f ms | rest %.2f ms | total %.2f ms\n",
                    solver.prof_[0] / n, solver.prof_[1] / n, solver.prof_[1] / np, np / n, solver.prof_[2] / n, solver.prof_[3] / n,
                    (solver.prof_[4] - solver.prof_[0] - solver.prof_[1] - solver.prof_[2] - solver.prof_[3]) / n, solver.prof_[4] / n);
    }
    if (solver.dbg_n_ > 0) {
        const double n = static_cast<double>(solver.dbg_n_); const int V = (hbias ? 15 : 9) + 6 * K;
        std::printf("[seqdbg] %ld solves | whitened posterior sd: min %.3f max %.3f (1 = proposal), mean shift %.2f sd\n",
                    solver.dbg_n_, std::sqrt(std::max(solver.dbg_wmin_ / n, 0.0)), std::sqrt(solver.dbg_wmax_ / n), solver.dbg_wshift_ / n);
        const char* nm[] = {"px","py","pz","vx","vy","vz","thx","thy","thz","bax","bay","baz","bgx","bgy","bgz"};
        std::printf("[seqdbg] per dim |shift|/sd, sd ratio:");
        for (int k = 0; k < V; ++k) std::printf(" %s:%.2f/%.2f", k < (hbias ? 15 : 9) ? nm[k] : (k - (hbias ? 15 : 9)) % 6 < 3 ? "a" : "g", solver.dbg_shift_[k] / n, solver.dbg_ratio_[k] / n);
        std::printf("\n");
    }
    std::string timing_file = output_file;
    const auto tpos = timing_file.rfind("_pose.txt");
    if (tpos != std::string::npos) timing_file.replace(tpos, 9, "_timing.txt");
    else                           timing_file += ".timing.txt";
    std::ofstream tf(timing_file);
    tf << std::fixed; tf.precision(6);
    for (double ms : solve_times) tf << ms << "\n";

    std::printf("[kims] %d solves in %.1f s | solve time mean=%.3f std=%.3f max=%.3f ms\n",
                solve_count, total_s, mean_ms, std_ms, max_ms);

    {
        std::string ess_file = output_file;
        const auto epos = ess_file.rfind("_pose.txt");
        if (epos != std::string::npos) ess_file.replace(epos, 9, "_ess.txt");
        else                           ess_file += ".ess.txt";
        std::ofstream ef(ess_file);
        ef << std::fixed; ef.precision(3);
        for (double e : ess_log) ef << e << "\n";
        std::vector<double> sorted(ess_log);
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
        double mean = 0.0;
        for (double e : ess_log) mean += e;
        if (!ess_log.empty()) mean /= ess_log.size();
        std::printf("[kims] ESS of %d samples: mean=%.1f median=%.1f -> %s\n", N, mean, med, ess_file.c_str());
        if (branches == 2) std::printf("[kims] two branches: %ld re-seeds, %ld leader switches\n", reseeds, leader_switches);
        if (opt > 0) std::printf("[kims] %s: %ld solves, mean iterations %.2f, failures %ld\n", opt == 1 ? "Gauss-Newton" : "Levenberg-Marquardt",
                                 opt_solves, opt_solves ? double(opt_iters_sum) / opt_solves : 0.0, opt_fail);
    }
    return 0;
}
