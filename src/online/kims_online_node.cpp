// @file kims_online_node.cpp
// @brief Real-time KIMS node. Subscribes to IMU and UWB, runs the same
//        fixed-lag smoother as the offline tool, and publishes the committed
//        pose. Feed it live sensors or `rosbag play`.

#include "dims.h"
#include "csv_io.h"
#include "mppi_solver.cuh"
#include "presets.h"

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nlink_parser/LinktrackTagframe0.h>
#include <kims/UwbRange.h>
#include <kims/ImuSample.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>




// Exactly the interpolation the offline solver uses (include/data.h), on the
// rolling buffer. Same order of operations, so the same doubles come out.
static ImuData interpImu(const std::deque<ImuData>& data, double ts, size_t& hint) {
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

class KimsOnline {
    // one UWB epoch waiting for the IMU to bracket it
    struct Epoch {
        double    ts;
        ros::Time stamp;
        std::vector<double> ranges, stds;
    };

public:
    KimsOnline(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        std::string dataset, sequence;
        pnh.param<std::string>("dataset", dataset, std::string("iului"));
        pnh.param<std::string>("sequence", sequence, std::string(""));
        const kims::KimsPreset P = kims::kimsPreset(dataset, sequence);

        pnh.param("N", N_, P.N);
        pnh.param("T", T_, P.T);
        pnh.param("num_anchors", num_anchors_, 4);
        pnh.param("frame_id", frame_id_, std::string("map"));
        pnh.param("child_frame_id", child_frame_id_, std::string("base_link"));
        pnh.param("publish_tf", publish_tf_, true);
        pnh.param("tag_ids", tag_ids_, std::vector<int>{8, 9});
        pnh.getParam("anchor_ids", anchor_ids_);
        num_tags_ = static_cast<int>(tag_ids_.size());

        std::vector<double> anchors, tag_offsets, init;
        pnh.getParam("anchors", anchors);
        pnh.getParam("tag_offsets", tag_offsets);
        pnh.param("init", init, std::vector<double>{0.0, 0.0, 0.0});
        if (!sequence.empty()) pnh.getParam("init_" + sequence, init);
        if (static_cast<int>(anchors.size()) != 3 * num_anchors_) {
            ROS_FATAL("[kims] 'anchors' needs %d values, got %zu",
                      3 * num_anchors_, anchors.size());
            ros::shutdown();
            return;
        }
        tag_offsets.resize(3 * num_tags_, 0.0);
        init.resize(3, 0.0);
        if (anchor_ids_.empty())
            for (int i = 0; i < num_anchors_; ++i) anchor_ids_.push_back(i);

        double sigma[DIM_U] = {P.sigma_acc[0], P.sigma_acc[1], P.sigma_acc[2],
                               P.sigma_gyr[0], P.sigma_gyr[1], P.sigma_gyr[2]};
        const double gravity[3] = {0.0, 0.0, -9.81};

        solver_.reset(new MPPISolver(N_, T_, P.dt, P.gamma, anchors.data(), gravity,
                                     sigma, tag_offsets.data(), num_anchors_, num_tags_));
        solver_->w_uwb_ = P.w_uwb;
        solver_->w_acc_ = P.w_acc;
        solver_->w_gyr_ = P.w_gyr;
        mppi_set_ctrl_mode(P.ctrl_mode, P.sigma_imu_acc, P.sigma_imu_gyr);
        mppi_set_deltas(P.uwb_delta, P.ctrl_delta);
        mppi_set_floor_penalty(P.floor_penalty);
        uwb_std_fixed_ = P.uwb_std;
        z_down_        = P.imu_z_down != 0;

        num_rng_ = num_tags_ * num_anchors_;
        ranges_.assign(num_rng_, -1.0);
        stds_.assign(num_rng_, 0.1);

        std::fill(x0_, x0_ + DIM_X, 0.0);
        x0_[0] = init[0]; x0_[1] = init[1]; x0_[2] = init[2];
        x0_[6]  = 1.0;                              // initial attitude
        x0_[10] = z_down_ ? -1.0 : 1.0;             // a z-down IMU starts rolled
        x0_[14] = z_down_ ? -1.0 : 1.0;

        Uopt_.assign(DIM_U * T_, 0.0);
        U0_.assign(DIM_U * T_, 0.0);
        rng_flat_.assign(num_rng_ * T_, 0.0);
        std_flat_.assign(num_rng_ * T_, 0.1);
        acc_flat_.assign(DIM_U * T_, 0.0);

        pub_pose_ = nh.advertise<geometry_msgs::PoseStamped>("kims/pose", 10);
        pub_odom_ = nh.advertise<nav_msgs::Odometry>("kims/odom", 10);
        sub_imu_  = nh.subscribe("imu", 2000, &KimsOnline::imuCb, this);
        sub_imus_ = nh.subscribe("imu_sample", 2000, &KimsOnline::imuSampleCb, this);
        sub_uwb_  = nh.subscribe("uwb", 2000, &KimsOnline::rangeCb, this);
        sub_lt_   = nh.subscribe("uwb_linktrack", 2000, &KimsOnline::uwbCb, this);

        ROS_INFO("[kims] online: %s preset, N=%d T=%d, %d tags x %d anchors",
                 dataset.c_str(), N_, T_, num_tags_, num_anchors_);
    }

private:
    // Replay input: carries the sensor time as a double, so nothing is lost.
    void imuSampleCb(const kims::ImuSample::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        ImuData d;
        d.timestamp = msg->stamp;
        d.acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                                msg->linear_acceleration.z);
        d.gyr = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y,
                                msg->angular_velocity.z);
        pushImu(d);
    }

    void imuCb(const sensor_msgs::Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        ImuData d;
        d.timestamp = msg->header.stamp.toSec();
        d.acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                                msg->linear_acceleration.z);
        d.gyr = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y,
                                msg->angular_velocity.z);
        pushImu(d);
    }

    void pushImu(ImuData d) {
        if (!have_t0_) { t0_ = d.timestamp; have_t0_ = true; }
        d.timestamp -= t0_;
        imu_.push_back(d);
        while (imu_.size() > 4000) { imu_.pop_front(); if (imu_hint_ > 0) --imu_hint_; }
        tryPendingBody();
    }

    // Driver-agnostic range: one (tag, anchor) pair per message.
    void rangeCb(const kims::UwbRange::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = std::find(tag_ids_.begin(), tag_ids_.end(), msg->from_id);
        auto ia = std::find(anchor_ids_.begin(), anchor_ids_.end(), msg->to_id);
        if (it == tag_ids_.end() || ia == anchor_ids_.end()) return;
        if (msg->range < 0.01 || msg->range > 200.0) return;

        const int flat = static_cast<int>(it - tag_ids_.begin()) * num_anchors_
                       + static_cast<int>(ia - anchor_ids_.begin());
        ranges_[flat] = msg->range;
        stds_[flat]   = uwb_std_fixed_ > 0.0 ? uwb_std_fixed_
                                             : (msg->std > 0.0 ? msg->std : 0.1);
        stepExact(msg->stamp > 0.0 ? msg->stamp : msg->header.stamp.toSec(),
                  msg->header.stamp);
    }

    // Nooploop LinkTrack driver: one message carries a tag's whole distance array.
    void uwbCb(const nlink_parser::LinktrackTagframe0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        const int tid = static_cast<int>(msg->id);
        auto it = std::find(tag_ids_.begin(), tag_ids_.end(), tid);
        if (it == tag_ids_.end()) return;
        const int tag_idx = static_cast<int>(it - tag_ids_.begin());

        for (int a = 0; a < num_anchors_ && a < 8; ++a) {
            const double d = static_cast<double>(msg->dis_arr[a]);
            if (d < 0.01 || d > 200.0) continue;
            ranges_[tag_idx * num_anchors_ + a] = d;
            stds_  [tag_idx * num_anchors_ + a] = uwb_std_fixed_ > 0.0 ? uwb_std_fixed_ : 0.1;
        }
        step(ros::Time::now());
    }

    // Queue the epoch with the ranges as they stand right now; it runs as soon
    // as the IMU brackets it. Every epoch is queued, none is overwritten.
    void step(const ros::Time& stamp_in) {
        const ros::Time stamp = stamp_in.isZero() ? ros::Time::now() : stamp_in;
        stepExact(stamp.toSec(), stamp);
    }

    void stepExact(double ts_abs, const ros::Time& stamp_in) {
        const ros::Time stamp = stamp_in.isZero() ? ros::Time::now() : stamp_in;
        if (!have_t0_) { t0_ = ts_abs; have_t0_ = true; }
        const double ts = ts_abs - t0_;
        if (ts <= queued_ts_ + 1e-6) return;
        queued_ts_ = ts;
        pending_.push_back(Epoch{ts, stamp, ranges_, stds_});
        while (pending_.size() > 200) pending_.pop_front();
        tryPendingBody();
    }

    // Waiting for one IMU sample past the epoch is what makes the online
    // estimate match the offline one: the offline interpolation brackets the
    // epoch, so it needs the sample that arrives just after it.
    void tryPendingBody() {
        while (!pending_.empty() && imu_.size() >= 2 &&
               imu_.back().timestamp >= pending_.front().ts) {
            const Epoch e = pending_.front();
            pending_.pop_front();
            solveEpoch(e);
        }
    }

    void solveEpoch(const Epoch& e) {
        const double    ts    = e.ts;
        const ros::Time stamp = e.stamp;

        const ImuData im = interpImu(imu_, ts, imu_hint_);
        rng_win_.push_back(e.ranges);
        std_win_.push_back(e.stds);
        imu_win_.push_back(im);
        if (static_cast<int>(rng_win_.size()) > T_) {
            rng_win_.pop_front(); std_win_.pop_front(); imu_win_.pop_front();
        }

        const bool complete = std::none_of(e.ranges.begin(), e.ranges.end(),
                                           [](double r) { return r < 0.0; });
        if (static_cast<int>(rng_win_.size()) < T_ || !complete) {
            prev_ts_ = ts;
            return;
        }

        for (int t = 0; t < T_; ++t) {
            std::copy(rng_win_[t].begin(), rng_win_[t].end(), rng_flat_.begin() + t * num_rng_);
            if (uwb_std_fixed_ > 0.0)
                std::fill(std_flat_.begin() + t * num_rng_,
                          std_flat_.begin() + (t + 1) * num_rng_, uwb_std_fixed_);
            else
                std::copy(std_win_[t].begin(), std_win_[t].end(),
                          std_flat_.begin() + t * num_rng_);
            acc_flat_[t * 6 + 0] = imu_win_[t].acc.x();
            acc_flat_[t * 6 + 1] = imu_win_[t].acc.y();
            acc_flat_[t * 6 + 2] = imu_win_[t].acc.z();
            acc_flat_[t * 6 + 3] = imu_win_[t].gyr.x();
            acc_flat_[t * 6 + 4] = imu_win_[t].gyr.y();
            acc_flat_[t * 6 + 5] = imu_win_[t].gyr.z();
        }
        if (prev_ts_ > 0.0) {
            const double measured_dt = ts - prev_ts_;
            if (measured_dt > 1e-4 && measured_dt < 1.0) solver_->setDt(measured_dt);
        }
        std::copy(acc_flat_.begin(), acc_flat_.end(), U0_.begin());

        double xn[DIM_X];
        const ros::WallTime t_solve = ros::WallTime::now();
        solver_->solve(Uopt_.data(), xn, x0_, U0_.data(),
                       rng_flat_.data(), std_flat_.data(), acc_flat_.data());
        const double solve_ms = (ros::WallTime::now() - t_solve).toSec() * 1e3;
        std::copy(xn, xn + DIM_X, x0_);
        prev_ts_ = ts;

        publish(xn, ros::Time(ts + t0_));

        ++solve_count_;
        const STATE st(xn);
        Eigen::Quaterniond qd(st.R); qd.normalize();
        std::cout << "[kims] #" << std::setw(6) << solve_count_
                  << "  t=" << std::fixed << std::setprecision(2) << std::setw(8) << (ts + t0_)
                  << "  pos=[" << std::setprecision(3) << std::setw(8) << st.p.x()
                  << std::setw(8) << st.p.y() << std::setw(8) << st.p.z()
                  << " ]  quat=[" << std::setprecision(4) << std::setw(9) << qd.x()
                  << std::setw(9) << qd.y() << std::setw(9) << qd.z()
                  << std::setw(9) << qd.w() << " ]  "
                  << std::setprecision(1) << solve_ms << " ms" << std::endl;
    }

    void publish(const double* xn, const ros::Time& stamp) {
        const STATE s(xn);
        Eigen::Quaterniond q(s.R);
        q.normalize();

        geometry_msgs::PoseStamped ps;
        ps.header.stamp    = stamp;
        ps.header.frame_id = frame_id_;
        ps.pose.position.x = s.p.x();
        ps.pose.position.y = s.p.y();
        ps.pose.position.z = s.p.z();
        ps.pose.orientation.x = q.x(); ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z(); ps.pose.orientation.w = q.w();
        pub_pose_.publish(ps);

        nav_msgs::Odometry od;
        od.header = ps.header;
        od.child_frame_id = child_frame_id_;
        od.pose.pose = ps.pose;
        od.twist.twist.linear.x = s.v.x();
        od.twist.twist.linear.y = s.v.y();
        od.twist.twist.linear.z = s.v.z();
        pub_odom_.publish(od);

        if (publish_tf_) {
            geometry_msgs::TransformStamped tf;
            tf.header = ps.header;
            tf.child_frame_id = child_frame_id_;
            tf.transform.translation.x = ps.pose.position.x;
            tf.transform.translation.y = ps.pose.position.y;
            tf.transform.translation.z = ps.pose.position.z;
            tf.transform.rotation = ps.pose.orientation;
            tf_.sendTransform(tf);
        }
    }

    std::unique_ptr<MPPISolver> solver_;
    std::deque<ImuData>              imu_;
    std::deque<std::vector<double>>  rng_win_, std_win_;
    std::deque<ImuData>              imu_win_;
    std::vector<double> ranges_, stds_, Uopt_, U0_, rng_flat_, std_flat_, acc_flat_;
    std::vector<int>    tag_ids_, anchor_ids_;
    double  x0_[DIM_X];
    size_t  imu_hint_   {0};
    std::deque<Epoch> pending_;
    double  t0_         {0.0};
    bool    have_t0_    {false};
    double  prev_ts_    {-1.0};
    double  queued_ts_  {-1.0};
    int     solve_count_{0};
    std::mutex mtx_;

    int    N_, T_, num_tags_, num_anchors_, num_rng_;
    double uwb_std_fixed_;
    bool   z_down_, publish_tf_;
    std::string frame_id_, child_frame_id_;

    ros::Publisher  pub_pose_, pub_odom_;
    ros::Subscriber sub_imu_, sub_imus_, sub_uwb_, sub_lt_;
    tf2_ros::TransformBroadcaster tf_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kims_online");
    ros::NodeHandle nh, pnh("~");
    KimsOnline node(nh, pnh);
    ros::spin();
    return 0;
}
