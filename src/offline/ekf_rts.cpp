#include "csv_io.h"
#include "node_util.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Mat15 = Eigen::Matrix<double, 15, 15>;

struct Dims12 {
    using VecM   = Eigen::Matrix<double, 12, 1>;
    using MatMM  = Eigen::Matrix<double, 12, 12>;
    using MatM15 = Eigen::Matrix<double, 12, 15>;
    using Mat15M = Eigen::Matrix<double, 15, 12>;
    static VecM   vec(int)          { return VecM::Zero(); }
    static MatMM  eye(int, double s){ return MatMM::Identity() * s; }
    static MatM15 zeroM15(int)      { return MatM15::Zero(); }
};

struct DimsDyn {
    using VecM   = Eigen::VectorXd;
    using MatMM  = Eigen::MatrixXd;
    using MatM15 = Eigen::MatrixXd;
    using Mat15M = Eigen::MatrixXd;
    static VecM   vec(int d)           { return VecM::Zero(d); }
    static MatMM  eye(int d, double s) { return MatMM::Identity(d, d) * s; }
    static MatM15 zeroM15(int d)       { return MatM15::Zero(d, 15); }
};

struct EKFState {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d a_b = Eigen::Vector3d::Zero();
    Eigen::Vector3d w_b = Eigen::Vector3d::Zero();
};

struct ForwardData {
    EKFState x_pred, x_filt;
    Mat15    P_pred, P_filt, D;
    double   timestamp;
};

template <class D>
class EKF {
public:
    using VecM   = typename D::VecM;
    using MatMM  = typename D::MatMM;
    using MatM15 = typename D::MatM15;
    using Mat15M = typename D::Mat15M;

    EKF(int num_tags, int num_anc)
        : num_tags_(num_tags), num_anc_(num_anc), dim_(num_tags * num_anc) {
        _g << 0.0, 0.0, -9.81;
        std_acc_ = 0.3;
        std_gyr_ = 0.01;
        covP_    = Mat15::Identity() * 0.01;
        covR_    = D::eye(dim_, 0.01);
        tag_off_.resize(num_tags, Eigen::Vector3d::Zero());
        anchors_.resize(num_anc, Eigen::Vector3d::Zero());
    }

    void setAnchors(const std::vector<Eigen::Vector3d>& a)   { anchors_ = a; }
    void setTagOffsets(const std::vector<Eigen::Vector3d>& t){ tag_off_ = t; }
    void setUwbVar(double s) { covR_ = D::eye(dim_, s * s); }
    void setStdAcc(double s) { std_acc_ = s; }
    void setStdGyr(double s) { std_gyr_ = s; }

    static Eigen::Matrix3d Exp(const Eigen::Vector3d& w) {
        double th = w.norm();
        if (th < 1e-9) return Eigen::Matrix3d::Identity() + skew(w);
        Eigen::Matrix3d K = skew(w / th);
        return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1 - std::cos(th)) * K * K;
    }
    static Eigen::Vector3d Log(const Eigen::Matrix3d& Rm) {
        double c  = std::max(-1.0, std::min(1.0, 0.5 * (Rm.trace() - 1.0)));
        double th = std::acos(c);
        if (th < 1e-8) return Eigen::Vector3d::Zero();
        Eigen::Matrix3d Omega = (th / (2.0 * std::sin(th))) * (Rm - Rm.transpose());
        Omega = 0.5 * (Omega - Omega.transpose());
        return Eigen::Vector3d(Omega(2, 1), Omega(0, 2), Omega(1, 0));
    }
    static Eigen::VectorXd phiInv(const EKFState& s, const EKFState& ref) {
        Eigen::VectorXd xi(15);
        xi.segment<3>(0)  = s.p   - ref.p;
        xi.segment<3>(3)  = Log(ref.R.transpose() * s.R);
        xi.segment<3>(6)  = s.v   - ref.v;
        xi.segment<3>(9)  = s.a_b - ref.a_b;
        xi.segment<3>(12) = s.w_b - ref.w_b;
        return xi;
    }
    static EKFState phi(const EKFState& ref, const Eigen::VectorXd& xi) {
        EKFState ns;
        ns.p   = ref.p   + xi.segment<3>(0);
        ns.R   = ref.R   * Exp(xi.segment<3>(3));
        ns.v   = ref.v   + xi.segment<3>(6);
        ns.a_b = ref.a_b + xi.segment<3>(9);
        ns.w_b = ref.w_b + xi.segment<3>(12);
        return ns;
    }

    EKFState state;

    void beginPeriod() {
        state_prev_ = state;
        P_prev_     = covP_;
        F_accum_    = Mat15::Identity();
        covQ_accum_ = Mat15::Zero();
    }

    void predictStep(const ImuData& imu, double dt) {
        Eigen::Vector3d ac = imu.acc - state.a_b;
        Eigen::Vector3d wc = imu.gyr - state.w_b;

        Mat15 Fc = Mat15::Zero();
        Fc.block<3, 3>(0, 6)  =  Eigen::Matrix3d::Identity();
        Fc.block<3, 3>(3, 3)  = -skew(wc);
        Fc.block<3, 3>(3, 12) = -Eigen::Matrix3d::Identity();
        Fc.block<3, 3>(6, 3)  = -state.R * skew(ac);
        Fc.block<3, 3>(6, 9)  = -state.R;

        Mat15 Fk = Mat15::Identity() + Fc * dt;
        F_accum_ = Fk * F_accum_;

        Mat15 Q = Mat15::Zero();
        Q.block<3, 3>(3, 3)   = Eigen::Matrix3d::Identity() * std_acc_ * std_acc_ * dt * dt;
        Q.block<3, 3>(6, 6)   = Eigen::Matrix3d::Identity() * std_acc_ * std_acc_ * dt * dt;
        Q.block<3, 3>(9, 9)   = Eigen::Matrix3d::Identity() * 1e-6;
        Q.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;
        covQ_accum_ += Q;

        EKFState ns;
        ns.R   = state.R * Exp(wc * dt);
        ns.v   = state.v + (state.R * ac + _g) * dt;
        ns.p   = state.p + state.v * dt + 0.5 * (state.R * ac + _g) * dt * dt;
        ns.a_b = state.a_b;
        ns.w_b = state.w_b;
        state  = ns;
    }

    void endPeriod(ForwardData& fd) {
        covP_ = F_accum_ * P_prev_ * F_accum_.transpose() + covQ_accum_;
        covP_ = 0.5 * (covP_ + covP_.transpose());
        Mat15 Dm = P_prev_ * F_accum_.transpose();
        fd.x_pred = state;
        fd.P_pred = covP_;
        fd.D      = Dm;
    }

    void uwbUpdate(const VecM& z, ForwardData& fd, double ts) {
        VecM   z_pred = D::vec(dim_);
        MatM15 H      = D::zeroM15(dim_);

        for (int t = 0; t < num_tags_; ++t) {
            Eigen::Vector3d tw = state.p + state.R * tag_off_[t];
            for (int a = 0; a < num_anc_; ++a) {
                const int row = t * num_anc_ + a;
                Eigen::Vector3d diff = tw - anchors_[a];
                double dist = diff.norm();
                z_pred(row) = dist;
                if (dist < 1e-6) continue;
                Eigen::RowVector3d uv = diff.transpose() / dist;
                H.template block<1, 3>(row, 0) =  uv;
                H.template block<1, 3>(row, 3) = -uv * state.R * skew(tag_off_[t]);
            }
        }

        MatMM  S = H * covP_ * H.transpose() + covR_;
        Mat15M K = covP_ * H.transpose() * S.inverse();

        Eigen::VectorXd dxi = K * (z - z_pred);
        state  = phi(state, dxi);
        covP_ -= K * H * covP_;
        covP_  = 0.5 * (covP_ + covP_.transpose());

        fd.x_filt    = state;
        fd.P_filt    = covP_;
        fd.timestamp = ts;
    }

    static std::vector<EKFState> rtsSmooth(const std::vector<ForwardData>& fwd) {
        const int N = static_cast<int>(fwd.size());
        if (N == 0) return {};
        std::vector<EKFState> xs(N);
        std::vector<Mat15>    Ps(N);
        xs[N - 1] = fwd[N - 1].x_filt;
        Ps[N - 1] = fwd[N - 1].P_filt;
        for (int k = N - 2; k >= 0; --k) {
            Mat15 G = fwd[k + 1].D * fwd[k + 1].P_pred.ldlt().solve(Mat15::Identity());
            Eigen::VectorXd delta = phiInv(xs[k + 1], fwd[k + 1].x_pred);
            xs[k] = phi(fwd[k].x_filt, G * delta);
            Ps[k] = fwd[k].P_filt + G * (Ps[k + 1] - fwd[k + 1].P_pred) * G.transpose();
            Ps[k] = 0.5 * (Ps[k] + Ps[k].transpose());
        }
        return xs;
    }

private:
    int             num_tags_, num_anc_, dim_;
    Eigen::Vector3d _g;
    double          std_acc_, std_gyr_;
    Mat15           covP_, covQ_accum_, P_prev_, F_accum_;
    MatMM           covR_;
    std::vector<Eigen::Vector3d> anchors_, tag_off_;
    EKFState        state_prev_;
};

template <class D>
static int runEkf(const std::map<std::string, std::string>& args,
                  const std::vector<ImuData>& imu_data,
                  const std::vector<UwbRange>& uwb_data,
                  const std::vector<int>& tag_ids,
                  const std::vector<int>& anchor_ids,
                  const std::vector<Eigen::Vector3d>& anchors,
                  const std::vector<Eigen::Vector3d>& tag_off,
                  double std_acc, double std_gyr, double std_uwb,
                  const Eigen::Vector3d& init_p, bool z_down, double t_offset,
                  const std::string& output_file) {
    const int num_tags = static_cast<int>(tag_ids.size());
    const int num_anc  = static_cast<int>(anchor_ids.size());
    const int dim      = num_tags * num_anc;

    EKF<D> ekf(num_tags, num_anc);
    ekf.setAnchors(anchors);
    ekf.setTagOffsets(tag_off);
    ekf.setUwbVar(std_uwb);
    ekf.setStdAcc(std_acc);
    ekf.setStdGyr(std_gyr);
    ekf.state.p = init_p;
    if (z_down)
        ekf.state.R = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();

    auto fwd_t0 = std::chrono::high_resolution_clock::now();
    std::vector<ForwardData> fwd;
    LatestRanges latest(tag_ids, anchor_ids);
    size_t imu_hint    = 0;
    double last_ts     = -1.0;
    bool   period_open = false;

    for (const auto& uwb : uwb_data) {
        if (last_ts < 0.0) {
            while (imu_hint + 1 < imu_data.size() &&
                   imu_data[imu_hint].timestamp < uwb.timestamp) ++imu_hint;
            last_ts = uwb.timestamp;
        } else {
            ekf.beginPeriod();
            period_open = true;

            double t = last_ts;
            while (imu_hint + 1 < imu_data.size() &&
                   imu_data[imu_hint + 1].timestamp <= uwb.timestamp) {
                ++imu_hint;
                double dt = imu_data[imu_hint].timestamp - t;
                if (dt > 0.0 && dt < 0.5) ekf.predictStep(imu_data[imu_hint], dt);
                t = imu_data[imu_hint].timestamp;
            }
            if (uwb.timestamp > t + 1e-9) {
                ImuData im = interpolateImu(imu_data, uwb.timestamp, imu_hint);
                double dt = uwb.timestamp - t;
                if (dt > 0.0 && dt < 0.5) ekf.predictStep(im, dt);
            }
            last_ts = uwb.timestamp;
        }

        latest.update(uwb);

        if (period_open) {
            ForwardData fd;
            ekf.endPeriod(fd);
            period_open = false;

            if (latest.allValid()) {
                typename D::VecM z = D::vec(dim);
                const double* r = latest.data();
                for (int i = 0; i < dim; ++i) z(i) = r[i];
                ekf.uwbUpdate(z, fd, uwb.timestamp);
                fwd.push_back(fd);
            }
        }
    }

    double fwd_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - fwd_t0).count();
    std::cout << "[EKF-RTS] Forward pass: " << fwd.size() << " updates\n";
    if (fwd.size() < 2) {
        std::cerr << "[EKF-RTS] Not enough updates for smoothing.\n"; return 1;
    }

    auto rts_t0 = std::chrono::high_resolution_clock::now();
    std::vector<EKFState> smoothed = EKF<D>::rtsSmooth(fwd);
    double rts_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - rts_t0).count();

    std::ofstream ofs(output_file);
    if (!ofs) { std::cerr << "[EKF-RTS] Cannot open output: " << output_file << "\n"; return 1; }
    ofs << "# time x y z qx qy qz qw\n";
    ofs << std::fixed;
    ofs.precision(9);
    for (size_t k = 0; k < smoothed.size(); ++k) {
        const EKFState& s = smoothed[k];
        Eigen::Quaterniond q(s.R);
        q.normalize();
        ofs << (fwd[k].timestamp + t_offset) << " "
            << s.p.x() << " " << s.p.y() << " " << s.p.z() << " "
            << q.x()   << " " << q.y()   << " " << q.z()   << " " << q.w() << "\n";
    }
    ofs.close();

    double per_step = fwd.size() > 0 ? (fwd_ms + rts_ms) / fwd.size() : 0.0;
    writeTimingFile(output_file, per_step, fwd.size());
    std::cout << "[EKF-RTS] Done. output=" << output_file << "\n";
    return 0;
}

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);
    const std::string dataset = getArg<std::string>(args, "dataset", std::string("iului"));
    const std::string output_file = getArg<std::string>(args, "output_file", std::string("ekf_pose.txt"));

    std::vector<ImuData>  imu_data;
    std::vector<UwbRange> uwb_data;
    std::vector<Eigen::Vector3d> anchors, tag_off;
    std::vector<int> tag_ids, anchor_ids;

    if (dataset == "iului") {
        int num_anchors = getArg<int>(args, "num_anchors", 4);
        int tag0_id = getArg<int>(args, "tag0_id", 0);
        int tag1_id = getArg<int>(args, "tag1_id", 1);
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
        tag_ids = {tag0_id, tag1_id};

        std::string dir = getArg<std::string>(args, "data_dir", std::string(""));
        if (dir.empty()) { std::cerr << "data_dir required\n"; return 1; }
        if (!loadImu(dir + "/imu.csv", imu_data)) return 1;
        if (!loadUwbRange(dir + "/uwb_range.csv", tag_ids, anchor_ids, uwb_data)) return 1;

        double t0 = std::min(imu_data.front().timestamp, uwb_data.front().timestamp);
        for (auto& d : imu_data) d.timestamp -= t0;
        for (auto& d : uwb_data) d.timestamp -= t0;

        return runEkf<DimsDyn>(args, imu_data, uwb_data, tag_ids, anchor_ids, anchors, tag_off,
                               getArg<double>(args, "std_acc", 0.5),
                               getArg<double>(args, "std_gyr", 0.05),
                               getArg<double>(args, "std_uwb", 0.2),
                               Eigen::Vector3d(getArg<double>(args, "init_x", 0.0),
                                               getArg<double>(args, "init_y", 0.0),
                                               getArg<double>(args, "init_z", 0.0)),
                               false, t0, output_file);
    }

    const bool is_ntu = (dataset == "ntu");
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
    if (!loadUwbRange(uwb_file, tag_ids, anchor_ids, uwb_data)) return 1;

    std::vector<double> h_anchor(3 * num_anc);
    if (!loadAnchors(anchor_file, h_anchor.data(), num_anc)) return 1;
    for (int a = 0; a < num_anc; ++a)
        anchors.push_back(Eigen::Vector3d(h_anchor[3 * a], h_anchor[3 * a + 1], h_anchor[3 * a + 2]));

    return runEkf<Dims12>(args, imu_data, uwb_data, tag_ids, anchor_ids, anchors, tag_off,
                          getArg<double>(args, "std_acc", 0.3),
                          getArg<double>(args, "std_gyr", 0.01),
                          getArg<double>(args, "std_uwb", 0.1),
                          Eigen::Vector3d(getArg<double>(args, "init_x", 0.0),
                                          getArg<double>(args, "init_y", 0.0),
                                          getArg<double>(args, "init_z", 0.0)),
                          is_ntu, 0.0, output_file);
}
