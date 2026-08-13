#include "session.h"
#include "mppi_solver.cuh"
#include "presets.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
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

    const double gravity[3] = {0.0, 0.0, -9.81};

    MPPISolver solver(N, T, dt, gamma, S.anchors.data(), gravity, sigma,
                      S.tag_offsets.data(), S.num_anchors, S.num_tags);
    solver.w_uwb_ = w_uwb;
    solver.w_acc_ = w_acc;
    solver.w_gyr_ = w_gyr;
    mppi_set_ctrl_mode(ctrl_mode, sigma_imu_a, sigma_imu_g);
    mppi_set_deltas(uwb_delta, ctrl_delta);
    mppi_set_floor_penalty(floor_pen);

    const int num_rng = solver.numRanges();

    const double t0 = std::min(S.imu.front().timestamp, S.uwb.front().timestamp);
    for (auto& d : S.imu) d.timestamp -= t0;
    for (auto& d : S.uwb) d.timestamp -= t0;

    std::printf("[kims] IMU [%.3f, %.3f] s | UWB [%.3f, %.3f] s | N=%d T=%d dt=%.3f gamma=%.3f\n",
                S.imu.front().timestamp, S.imu.back().timestamp,
                S.uwb.front().timestamp, S.uwb.back().timestamp, N, T, dt, gamma);

    const bool z_down = getArg<int>(args, "imu_z_down", P.imu_z_down) != 0;
    double x0[DIM_X] = {};
    x0[0] = S.init_p.x(); x0[1] = S.init_p.y(); x0[2] = S.init_p.z();
    x0[6]  = 1.0;
    x0[10] = z_down ? -1.0 : 1.0;
    x0[14] = z_down ? -1.0 : 1.0;

    std::vector<double> U0(T * DIM_U, 0.0), Uopt(T * DIM_U, 0.0);
    double xn[DIM_X] = {};

    std::ofstream pose_file(output_file);
    if (!pose_file.is_open()) {
        std::fprintf(stderr, "[kims] cannot open output: %s\n", output_file.c_str());
        return -1;
    }
    pose_file << std::fixed;
    pose_file.precision(9);
    pose_file << "# time x y z qx qy qz qw\n";

    LatestRanges latest(S.num_tags, S.num_anchors);

    std::vector<double> ranges_flat(T * num_rng, -1.0);
    std::vector<double> stds_flat  (T * num_rng,  0.1);
    std::vector<double> accgyr_flat(T * 6,        0.0);

    std::deque<std::vector<double>> rng_window, std_window;
    std::deque<ImuData>             imu_window;

    size_t uwb_idx = 0, imu_hint = 0;
    int    solve_count = 0;
    double prev_ts = -1.0;
    std::vector<double> solve_times;

    const auto wall_start = std::chrono::high_resolution_clock::now();

    while (uwb_idx < S.uwb.size()) {
        latest.update(S.uwb[uwb_idx]);
        const double ts = S.uwb[uwb_idx].timestamp;
        ++uwb_idx;

        if (ts <= prev_ts + 1e-6) continue;

        const ImuData imu = interpolateImu(S.imu, ts, imu_hint);

        std::vector<double> rng(num_rng), std_v(num_rng);
        latest.copyTo(rng.data());
        latest.copyStdsTo(std_v.data());
        rng_window.push_back(rng);
        std_window.push_back(std_v);
        imu_window.push_back(imu);

        if (static_cast<int>(rng_window.size()) > T) {
            rng_window.pop_front();
            std_window.pop_front();
            imu_window.pop_front();
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

        if (T >= 2 && prev_ts > 0.0) {
            const double measured_dt = ts - prev_ts;
            if (measured_dt > 1e-4 && measured_dt < 1.0) solver.setDt(measured_dt);
        }

        for (int t = 0; t < T; ++t)
            std::copy(accgyr_flat.data() + t*6, accgyr_flat.data() + t*6 + 6,
                      U0.data() + t*6);

        const auto t_solve = std::chrono::high_resolution_clock::now();
        solver.solve(Uopt.data(), xn, x0, U0.data(),
                     ranges_flat.data(), stds_flat.data(), accgyr_flat.data());
        solve_times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_solve).count());

        std::copy(xn, xn + DIM_X, x0);

        const STATE state(xn);
        const Eigen::Quaterniond q(state.R);
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

    std::string timing_file = output_file;
    const auto tpos = timing_file.rfind("_pose.txt");
    if (tpos != std::string::npos) timing_file.replace(tpos, 9, "_timing.txt");
    else                           timing_file += ".timing.txt";
    std::ofstream tf(timing_file);
    tf << std::fixed; tf.precision(6);
    for (double ms : solve_times) tf << ms << "\n";

    std::printf("[kims] %d solves in %.1f s | solve time mean=%.3f std=%.3f max=%.3f ms\n",
                solve_count, total_s, mean_ms, std_ms, max_ms);
    return 0;
}
