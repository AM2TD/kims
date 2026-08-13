#include "csv_io.h"
#include "node_util.h"

#include <Eigen/Dense>
#include <fstream>
#include <deque>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <map>
#include <string>
#include <cmath>
#include <chrono>


struct Particle {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d    position;
    Eigen::Vector3d    velocity;
    Eigen::Quaterniond orientation;
    double weight;
};

static Eigen::Quaterniond expMap(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-10) return Eigen::Quaterniond::Identity();
    Eigen::Vector3d axis = omega / theta;
    double ht = theta * 0.5;
    return Eigen::Quaterniond(std::cos(ht),
                              axis.x() * std::sin(ht),
                              axis.y() * std::sin(ht),
                              axis.z() * std::sin(ht));
}

static void resample(std::vector<Particle, Eigen::aligned_allocator<Particle>>& parts,
                     std::mt19937& gen, std::vector<int>& ancestor) {
    const int N = static_cast<int>(parts.size());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    std::vector<Particle, Eigen::aligned_allocator<Particle>> new_parts;
    new_parts.reserve(N);
    ancestor.resize(N);

    double r = dist(gen) / N;
    double c = parts[0].weight;
    int i = 0;
    for (int m = 0; m < N; ++m) {
        double u = r + m * (1.0 / N);
        while (u > c && i < N - 1) { ++i; c += parts[i].weight; }
        new_parts.push_back(parts[i]);
        new_parts.back().weight = 1.0 / N;
        ancestor[m] = i;
    }
    parts = new_parts;
}

struct GenealogySnap {
    double ts;
    std::vector<Eigen::Vector3d> pos;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> quat;
    std::vector<int> parent;
};

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);
    const std::string dataset = getArg<std::string>(args, "dataset", std::string("iului"));
    const bool is_ntu   = (dataset == "ntu");
    const bool is_bag   = (dataset == "iului");
    const bool paired   = !is_ntu;              // two-tag sets gate anchors pairwise
    const double max_rng = is_ntu ? 100.0 : 50.0;

    std::string output_file = getArg<std::string>(args, "output_file", std::string("ps_pose.txt"));

    std::vector<ImuData>  imu_data;
    std::vector<UwbRange> uwb_raw;
    std::vector<Eigen::Vector3d> anchors, tag_off;
    std::vector<int> tag_ids, anchor_ids;
    double t0 = 0.0;

    if (is_bag) {
        int num_anchors = getArg<int>(args, "num_anchors", 4);
        int tag0_id = getArg<int>(args, "tag0_id", 0);
        int tag1_id = getArg<int>(args, "tag1_id", 1);
        tag_ids = {tag0_id, tag1_id};
        tag_off = {Eigen::Vector3d(getArg<double>(args, "tag0_offset_x", 0.0),
                                   getArg<double>(args, "tag0_offset_y", 0.1317),
                                   getArg<double>(args, "tag0_offset_z", 0.0)),
                   Eigen::Vector3d(getArg<double>(args, "tag1_offset_x", 0.0),
                                   getArg<double>(args, "tag1_offset_y", -0.1227),
                                   getArg<double>(args, "tag1_offset_z", 0.0))};
        const double defs[12] = {-2.9449, -3.0166, 1.6859, -3.0967, 3.1997, 1.6656,
                                  2.3721,  2.9621, 1.6817,  2.4376, -2.9803, 1.6931};
        for (int i = 0; i < num_anchors; ++i) {
            anchors.push_back(Eigen::Vector3d(
                getArg<double>(args, "anchor" + std::to_string(i) + "_x", defs[3 * i]),
                getArg<double>(args, "anchor" + std::to_string(i) + "_y", defs[3 * i + 1]),
                getArg<double>(args, "anchor" + std::to_string(i) + "_z", defs[3 * i + 2])));
            anchor_ids.push_back(i);
        }

        std::string dir = getArg<std::string>(args, "data_dir", std::string(""));
        if (dir.empty()) { std::cerr << "data_dir required\n"; return 1; }
        if (!loadImu(dir + "/imu.csv", imu_data)) return 1;
        if (!loadUwbRange(dir + "/uwb_range.csv", tag_ids, anchor_ids, uwb_raw)) return 1;

        t0 = std::min(imu_data.front().timestamp, uwb_raw.front().timestamp);
        for (auto& d : imu_data) d.timestamp -= t0;
        for (auto& d : uwb_raw)  d.timestamp -= t0;
    } else {
        std::string data_dir    = getArg<std::string>(args, "data_dir", std::string(""));
        std::string anchor_file = getArg<std::string>(args, "anchor_file", std::string(""));
        std::string imu_file    = getArg<std::string>(args, "imu_file",
                                      data_dir + (is_ntu ? "/imu.csv" : "/imu_px4.csv"));
        std::string uwb_file    = getArg<std::string>(args, "uwb_file", data_dir + "/uwb_range.csv");

        const int num_tags = is_ntu ? 4 : 2;
        const int num_anc  = is_ntu ? 3 : 6;
        for (int t = 0; t < num_tags; ++t) {
            const std::string k = "tag" + std::to_string(t);
            tag_ids.push_back(getArg<int>(args, k + "_id", is_ntu ? 2000 : 10));
            tag_off.push_back(Eigen::Vector3d(getArg<double>(args, k + "_offset_x", 0.0),
                                              getArg<double>(args, k + "_offset_y", 0.0),
                                              getArg<double>(args, k + "_offset_z", 0.0)));
        }
        if (is_ntu) for (int a = 0; a < num_anc; ++a)
            anchor_ids.push_back(getArg<int>(args, "anchor" + std::to_string(a) + "_id", 100 + a));
        else        for (int a = 0; a < num_anc; ++a) anchor_ids.push_back(a);

        if (!loadImu(imu_file, imu_data)) return 1;
        if (!loadUwbRange(uwb_file, tag_ids, anchor_ids, uwb_raw)) return 1;

        std::vector<double> h_anchor(3 * num_anc);
        if (!loadAnchors(anchor_file, h_anchor.data(), num_anc)) return 1;
        for (int a = 0; a < num_anc; ++a)
            anchors.push_back(Eigen::Vector3d(h_anchor[3 * a], h_anchor[3 * a + 1],
                                              h_anchor[3 * a + 2]));
    }

    const int num_tags = static_cast<int>(tag_ids.size());
    const int num_anc  = static_cast<int>(anchor_ids.size());

    int    N_part      = getArg<int>   (args, "num_particles", 1000);
    double sigma_uwb   = getArg<double>(args, "sigma_uwb",  is_ntu ? 0.5 : 0.3);
    double sigma_pos   = getArg<double>(args, "sigma_pos",  is_bag ? 0.02 : (is_ntu ? 0.01 : 0.1));
    double sigma_vel   = getArg<double>(args, "sigma_vel",  is_bag ? 0.05 : (is_ntu ? 0.02 : 0.1));
    double sigma_gyro  = getArg<double>(args, "sigma_gyro", 0.01);
    double ess_thresh  = getArg<double>(args, "ess_thresh", is_bag ? 0.7 : 0.5);
    int    smooth_lag  = getArg<int>   (args, "smooth_lag", 15);
    double init_x      = getArg<double>(args, "init_x", 0.0);
    double init_y      = getArg<double>(args, "init_y", 0.0);
    double init_z      = getArg<double>(args, "init_z", 0.0);

    int seed = getArg<int>(args, "seed", 42);
    std::mt19937 gen(seed);
    std::normal_distribution<> pos_dist(0.0, 0.5);
    std::normal_distribution<> vel_dist(0.0, 0.1);

    std::vector<Particle, Eigen::aligned_allocator<Particle>> particles(N_part);
    const Eigen::Quaterniond q_init =
        is_ntu ? Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()))
               : Eigen::Quaterniond::Identity();

    for (auto& p : particles) {
        p.position    = Eigen::Vector3d(init_x + pos_dist(gen),
                                        init_y + pos_dist(gen),
                                        init_z + pos_dist(gen) * 0.1);
        p.velocity    = Eigen::Vector3d(vel_dist(gen), vel_dist(gen), vel_dist(gen));
        p.orientation = q_init;
        p.weight      = 1.0 / N_part;
    }

    std::ofstream ofs(output_file);
    if (!ofs) { std::cerr << "[PS] Cannot open output: " << output_file << "\n"; return 1; }
    ofs << "# time x y z qx qy qz qw\n";
    ofs << std::fixed;
    ofs.precision(9);

    std::ofstream ofs_ess;
    if (is_ntu) {
        std::string ess_file = output_file;
        auto pe = ess_file.rfind("_pose.txt");
        if (pe != std::string::npos) ess_file.replace(pe, 9, "_ess.txt");
        else                         ess_file += ".ess.txt";
        ofs_ess.open(ess_file);
        if (!ofs_ess) { std::cerr << "[PS] Cannot open ESS output: " << ess_file << "\n"; return 1; }
        ofs_ess << "# time ess z_mean\n";
        ofs_ess << std::fixed;
        ofs_ess.precision(9);
    }

    LatestRanges latest(tag_ids, anchor_ids);
    size_t uwb_ptr    = 0;
    double prev_ts    = -1.0;
    int    solve_count = 0;
    int    uwb_trigger = 0;
    std::vector<double> solve_times_vec;
    std::deque<GenealogySnap> genealogy;

    std::normal_distribution<> n_pos(0.0,  sigma_pos);
    std::normal_distribution<> n_vel(0.0,  sigma_vel);
    std::normal_distribution<> n_gyro(0.0, sigma_gyro);
    const double uwb_denom = 2.0 * sigma_uwb * sigma_uwb;

    for (size_t imu_i = 0; imu_i < imu_data.size(); ++imu_i) {
        const ImuData& im = imu_data[imu_i];

        double dt = 0.01;
        if (prev_ts > 0.0) {
            dt = im.timestamp - prev_ts;
            if (dt <= 0.0 || dt > 0.1) dt = 0.01;
        }
        prev_ts = im.timestamp;

        for (auto& p : particles) {
            Eigen::Vector3d a_world = p.orientation * im.acc;
            Eigen::Vector3d a_corr  = a_world - Eigen::Vector3d(0, 0, 9.81);

            p.velocity    += a_corr * dt + Eigen::Vector3d(n_vel(gen), n_vel(gen), n_vel(gen));
            p.position    += p.velocity * dt + Eigen::Vector3d(n_pos(gen), n_pos(gen), n_pos(gen));
            Eigen::Vector3d gyr_noisy = im.gyr + Eigen::Vector3d(n_gyro(gen), n_gyro(gen), n_gyro(gen));
            p.orientation = (p.orientation * expMap(gyr_noisy * dt)).normalized();
        }

        while (uwb_ptr < uwb_raw.size() &&
               uwb_raw[uwb_ptr].timestamp <= im.timestamp) {
            latest.update(uwb_raw[uwb_ptr]);
            ++uwb_ptr;
            ++uwb_trigger;

            if (!latest.allValid()) continue;

            auto t0s = std::chrono::high_resolution_clock::now();

            const double* ranges = latest.data();
            for (auto& p : particles) {
                double total_err = 0.0;
                int valid = 0;
                if (paired) {
                    Eigen::Vector3d t0pos = p.position + p.orientation * tag_off[0];
                    Eigen::Vector3d t1pos = p.position + p.orientation * tag_off[1];
                    for (int a = 0; a < num_anc; ++a) {
                        double d0 = ranges[a];
                        double d1 = ranges[a + num_anc];
                        if (d0 < 0.1 || d0 > max_rng || d1 < 0.1 || d1 > max_rng) continue;
                        double e0 = (t0pos - anchors[a]).norm() - d0;
                        double e1 = (t1pos - anchors[a]).norm() - d1;
                        total_err += e0 * e0 + e1 * e1;
                        valid += 2;
                    }
                } else {
                    for (int ti = 0; ti < num_tags; ++ti) {
                        Eigen::Vector3d tag_w = p.position + p.orientation * tag_off[ti];
                        for (int ai = 0; ai < num_anc; ++ai) {
                            double d_meas = ranges[ti * num_anc + ai];
                            if (d_meas < 0.1 || d_meas > max_rng) continue;
                            double e = (tag_w - anchors[ai]).norm() - d_meas;
                            total_err += e * e;
                            ++valid;
                        }
                    }
                }
                if (valid > 0)
                    p.weight *= std::exp(-total_err / (uwb_denom * valid));
            }

            double wsum = 0.0;
            for (const auto& p : particles) wsum += p.weight;
            if (wsum > 0)
                for (auto& p : particles) p.weight /= wsum;

            double ess_val = 0.0;
            {
                double sum_w2 = 0.0;
                double mean_z = 0.0;
                for (const auto& p : particles) {
                    sum_w2 += p.weight * p.weight;
                    mean_z += p.position.z() * p.weight;
                }
                ess_val = (sum_w2 > 0.0) ? 1.0 / sum_w2 : 0.0;
                if (is_ntu)
                    ofs_ess << uwb_raw[uwb_ptr - 1].timestamp << " "
                            << ess_val << " " << mean_z << "\n";
            }

            std::vector<int> anc(N_part);
            for (int m = 0; m < N_part; ++m) anc[m] = m;
            if (ess_val < ess_thresh * N_part)
                resample(particles, gen, anc);

            GenealogySnap snap;
            snap.ts = uwb_raw[uwb_ptr - 1].timestamp + t0;
            snap.parent = anc;
            snap.pos.reserve(N_part);
            snap.quat.reserve(N_part);
            for (const auto& p : particles) {
                snap.pos.push_back(p.position);
                snap.quat.push_back(p.orientation);
            }
            genealogy.push_back(std::move(snap));

            if ((int)genealogy.size() > smooth_lag) {
                Eigen::Vector3d est_pos = Eigen::Vector3d::Zero();
                Eigen::Vector4d q_sum   = Eigen::Vector4d::Zero();
                for (int m = 0; m < N_part; ++m) {
                    int idx = m;
                    for (int k = (int)genealogy.size() - 1; k > 0; --k)
                        idx = genealogy[k].parent[idx];
                    double w = particles[m].weight;
                    est_pos += genealogy.front().pos[idx] * w;
                    const Eigen::Quaterniond& qq = genealogy.front().quat[idx];
                    q_sum += Eigen::Vector4d(qq.w(), qq.x(), qq.y(), qq.z()) * w;
                }
                q_sum.normalize();
                Eigen::Quaterniond est_q(q_sum(0), q_sum(1), q_sum(2), q_sum(3));
                ofs << genealogy.front().ts << " "
                    << est_pos.x() << " " << est_pos.y() << " " << est_pos.z() << " "
                    << est_q.x() << " " << est_q.y() << " " << est_q.z() << " " << est_q.w()
                    << "\n";
                genealogy.pop_front();
            }

            double step_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - t0s).count();
            solve_times_vec.push_back(step_ms);
            ++solve_count;
        }
    }

    while (!genealogy.empty()) {
        Eigen::Vector3d est_pos = Eigen::Vector3d::Zero();
        Eigen::Vector4d q_sum   = Eigen::Vector4d::Zero();
        for (int m = 0; m < N_part; ++m) {
            int idx = m;
            for (int k = (int)genealogy.size() - 1; k > 0; --k)
                idx = genealogy[k].parent[idx];
            double w = particles[m].weight;
            est_pos += genealogy.front().pos[idx] * w;
            const Eigen::Quaterniond& qq = genealogy.front().quat[idx];
            q_sum += Eigen::Vector4d(qq.w(), qq.x(), qq.y(), qq.z()) * w;
        }
        q_sum.normalize();
        Eigen::Quaterniond est_q(q_sum(0), q_sum(1), q_sum(2), q_sum(3));
        ofs << genealogy.front().ts << " "
            << est_pos.x() << " " << est_pos.y() << " " << est_pos.z() << " "
            << est_q.x() << " " << est_q.y() << " " << est_q.z() << " " << est_q.w()
            << "\n";
        genealogy.pop_front();
    }

    ofs.close();
    if (is_ntu) ofs_ess.close();

    {
        size_t N = solve_times_vec.size();
        double sum = 0.0, sum2 = 0.0, max_ms = 0.0;
        for (double ms : solve_times_vec) { sum += ms; sum2 += ms*ms; if (ms > max_ms) max_ms = ms; }
        double mean_ms = N > 0 ? sum / N : 0.0;
        double std_ms  = N > 1 ? std::sqrt((sum2 - sum*sum/N) / (N-1)) : 0.0;
        std::ofstream tf(timingPath(output_file));
        tf << std::fixed; tf.precision(6);
        for (double ms : solve_times_vec) tf << ms << "\n";
        printf("[PS] Timing: mean=%.3f ms  std=%.3f ms  max=%.3f ms  N=%zu\n",
               mean_ms, std_ms, max_ms, N);
    }

    std::cout << "[PS] Done. IMU steps=" << imu_data.size()
              << "  UWB events=" << uwb_ptr
              << "  pose outputs=" << solve_count
              << "  output=" << output_file << "\n";
    return 0;
}
