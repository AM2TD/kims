#include "stream.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <topic_tools/shape_shifter.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

namespace {

const char* kTagframeMd5 = "20cc09884b3e1aa830a1d8a71796a857";
const size_t kIdOffset = 1, kDisOffset = 50;

double toSec(const ros::Time& t) { return static_cast<double>(t.sec) + static_cast<double>(t.nsec) / 1e9; }

struct Event {
    bool imu = false;
    double t = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero(), gyr = Eigen::Vector3d::Zero();
    int tag_id = -1;
    float dis[8] = {0};
};

class RosSource : public kims::StreamSource {
public:
    RosSource(ros::NodeHandle& nh, ros::NodeHandle& pnh, const std::map<std::string, std::string>& args, int num_anchors)
        : num_anchors_(num_anchors) {
        tag_ids_ = {kims::getArg<int>(args, "tag0_id", 8), kims::getArg<int>(args, "tag1_id", 9)};
        pnh.param<std::string>("imu_topic", imu_topic_, "/mavros/imu/data");
        pnh.param<std::string>("uwb_topic", uwb_topic_, "/nlink_linktrack_tagframe0");
        pnh.param<std::string>("gt_topic", gt_topic_, "/qualisys/gogogo/pose");
        pnh.param<std::string>("frame_id", frame_id_, "world");
        pnh.param<std::string>("bag", bag_path_, "");
        pnh.param("rate", rate_, 1.0);
        pnh.param("uwb_epoch", uwb_epoch_, 0.01);
        pnh.param("idle_timeout", idle_timeout_, 0.0);
        pub_pose_ = pnh.advertise<geometry_msgs::PoseStamped>("pose", 50);
        pub_now_ = pnh.advertise<geometry_msgs::PoseStamped>("pose_now", 50);
        pub_path_ = pnh.advertise<nav_msgs::Path>("path", 2, true);
        pub_gt_path_ = pnh.advertise<nav_msgs::Path>("ground_truth_path", 2, true);
        path_.header.frame_id = gt_path_.header.frame_id = frame_id_;
        if (!bag_path_.empty()) {
            bag_.open(bag_path_, rosbag::bagmode::Read);
            view_.reset(new rosbag::View(bag_, rosbag::TopicQuery(std::vector<std::string>{imu_topic_, uwb_topic_, gt_topic_})));
            it_ = view_->begin();
            ROS_INFO("[kims_online] reading %s (%s, %s) at %s", bag_path_.c_str(), imu_topic_.c_str(), uwb_topic_.c_str(),
                     rate_ > 0.0 ? (std::to_string(rate_) + "x real time").c_str() : "full speed");
        } else {
            sub_imu_ = nh.subscribe(imu_topic_, 2000, &RosSource::imuCb, this, ros::TransportHints().tcpNoDelay());
            sub_uwb_ = nh.subscribe(uwb_topic_, 2000, &RosSource::uwbCb, this, ros::TransportHints().tcpNoDelay());
            if (!gt_topic_.empty()) sub_gt_ = nh.subscribe(gt_topic_, 100, &RosSource::gtCb, this);
            ROS_INFO("[kims_online] subscribed to %s and %s", imu_topic_.c_str(), uwb_topic_.c_str());
        }
    }

    bool pull(kims::Session& S) override { return bag_path_.empty() ? pullTopics(S) : pullBag(S); }

    void onEstimate(double t_head, const double* p_head, const double* q_head,
                    double t_now, const double* p_now, const double* q_now) override {
        if (t_first_ < 0.0) t_first_ = t_now;
        if (t_now - t_first_ > 10.0) {
            if (bag_path_.empty()) latency_.push_back(toSec(ros::Time::now()) - t_now);
            else if (rate_ > 0.0 && paced_)
                latency_.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0_).count() -
                                   (t_now - bag_t0_) / rate_);
        }
        const geometry_msgs::PoseStamped head = pose(t_head, p_head, q_head), now = pose(t_now, p_now, q_now);
        pub_pose_.publish(head);
        pub_now_.publish(now);
        path_.poses.push_back(head);
        path_.header.stamp = head.header.stamp;
        if (++n_est_ % 10 == 0) pub_path_.publish(path_);
    }

    void report() {
        if (latency_.empty()) return;
        std::vector<double> l = latency_;
        std::sort(l.begin(), l.end());
        std::printf("[kims_online] latency of %zu estimates after the first 10 s, from the epoch's data to its estimate: median %.1f ms, 95%% %.1f ms, max %.1f ms\n",
                    l.size(), 1e3 * l[l.size() / 2], 1e3 * l[std::min(l.size() - 1, l.size() * 95 / 100)], 1e3 * l.back());
    }

private:
    geometry_msgs::PoseStamped pose(double t, const double* p, const double* q) const {
        geometry_msgs::PoseStamped m;
        m.header.stamp = ros::Time(t);
        m.header.frame_id = frame_id_;
        m.pose.position.x = p[0]; m.pose.position.y = p[1]; m.pose.position.z = p[2];
        m.pose.orientation.x = q[0]; m.pose.orientation.y = q[1]; m.pose.orientation.z = q[2]; m.pose.orientation.w = q[3];
        return m;
    }

    void append(kims::Session& S, const Event& e) const {
        if (e.imu) { S.imu.push_back(kims::ImuData{e.t, e.acc, e.gyr}); return; }
        int tag = -1;
        for (size_t k = 0; k < tag_ids_.size(); ++k) if (tag_ids_[k] == e.tag_id) tag = static_cast<int>(k);
        if (tag < 0) return;
        for (int a = 0; a < num_anchors_ && a < 8; ++a) {
            if (!(e.dis[a] > 0.0f)) continue;
            kims::UwbRange r;
            r.timestamp = e.t; r.tag_idx = tag; r.anc_idx = a;
            r.range = static_cast<double>(e.dis[a]); r.std_dev = 0.1;
            S.uwb.push_back(r);
        }
    }
    static bool decode(const std::vector<uint8_t>& buf, Event& e) {
        if (buf.size() < kDisOffset + 32) return false;
        e.tag_id = buf[kIdOffset];
        std::memcpy(e.dis, buf.data() + kDisOffset, 32);
        return true;
    }

    bool pullBag(kims::Session& S) {
        while (it_ != view_->end() && ros::ok()) {
            const rosbag::MessageInstance& m = *it_;
            Event e;
            bool ok = false;
            if (m.getTopic() == gt_topic_) {
                const auto gt = m.instantiate<geometry_msgs::PoseStamped>();
                const double t_rec = m.getTime().toSec();
                ++it_;
                if (gt) { if (rate_ > 0.0) pace(t_rec); gtCb(gt); }
                continue;
            }
            if (m.getTopic() == imu_topic_) {
                if (auto msg = m.instantiate<sensor_msgs::Imu>()) {
                    e.imu = true; e.t = toSec(msg->header.stamp);
                    e.gyr = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
                    e.acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                    ok = true;
                }
            } else if (m.getMD5Sum() == kTagframeMd5) {
                std::vector<uint8_t> buf(m.size());
                ros::serialization::OStream os(buf.data(), static_cast<uint32_t>(buf.size()));
                m.write(os);
                e.t = toSec(m.getTime());
                ok = decode(buf, e);
            } else if (!warned_) {
                ROS_WARN("[kims_online] %s is %s, expected nlink_parser/LinktrackTagframe0", m.getTopic().c_str(), m.getDataType().c_str());
                warned_ = true;
            }
            const double t_rec = m.getTime().toSec();
            ++it_;
            if (!ok) continue;
            if (rate_ > 0.0) pace(t_rec);
            append(S, e);
            return true;
        }
        return false;
    }
    void pace(double t_msg) {
        const auto now = std::chrono::steady_clock::now();
        if (!paced_) { paced_ = true; bag_t0_ = t_msg; wall_t0_ = now; return; }
        const auto due = wall_t0_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double>((t_msg - bag_t0_) / rate_));
        if (due > now) std::this_thread::sleep_until(due);
    }

    void gtCb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        if (!gt_path_.poses.empty() && (msg->header.stamp - gt_path_.poses.back().header.stamp).toSec() < 0.1) return;
        geometry_msgs::PoseStamped g = *msg;
        g.header.frame_id = frame_id_;
        gt_path_.poses.push_back(g);
        gt_path_.header.stamp = g.header.stamp;
        pub_gt_path_.publish(gt_path_);
    }

    void imuCb(const sensor_msgs::Imu::ConstPtr& msg) {
        Event e; e.imu = true; e.t = toSec(msg->header.stamp);
        e.gyr = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        e.acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        push(e);
    }
    void uwbCb(const topic_tools::ShapeShifter::ConstPtr& msg) {
        if (msg->getMD5Sum() != kTagframeMd5) {
            if (!warned_) { ROS_WARN("[kims_online] %s is %s, expected nlink_parser/LinktrackTagframe0", uwb_topic_.c_str(), msg->getDataType().c_str()); warned_ = true; }
            return;
        }
        std::vector<uint8_t> buf(msg->size());
        ros::serialization::OStream os(buf.data(), static_cast<uint32_t>(buf.size()));
        msg->write(os);
        Event e;
        if (!decode(buf, e)) return;
        double t = toSec(ros::Time::now());
        if (uwb_epoch_ > 0.0) {
            if (!grid_set_) { grid_t0_ = t; grid_set_ = true; }
            t = grid_t0_ + std::round((t - grid_t0_) / uwb_epoch_) * uwb_epoch_;
        }
        e.t = t;
        push(e);
    }
    void push(const Event& e) {
        { std::lock_guard<std::mutex> lk(mu_); queue_.push_back(e); last_rx_ = std::chrono::steady_clock::now(); got_any_ = true; }
        cv_.notify_one();
    }
    bool pullTopics(kims::Session& S) {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            if (!queue_.empty()) {
                for (const auto& e : queue_) append(S, e);
                queue_.clear();
                return true;
            }
            if (!ros::ok()) return false;
            if (idle_timeout_ > 0.0 && got_any_ &&
                std::chrono::steady_clock::now() - last_rx_ > std::chrono::duration<double>(idle_timeout_)) {
                ROS_INFO("[kims_online] no data for %.1f s: end of stream", idle_timeout_);
                return false;
            }
            cv_.wait_for(lk, std::chrono::milliseconds(50));
        }
    }

    int num_anchors_;
    std::vector<int> tag_ids_;
    std::string imu_topic_, uwb_topic_, gt_topic_, frame_id_, bag_path_;
    double rate_ = 1.0, uwb_epoch_ = 0.01, idle_timeout_ = 0.0;
    ros::Publisher pub_pose_, pub_now_, pub_path_, pub_gt_path_;
    ros::Subscriber sub_imu_, sub_uwb_, sub_gt_;
    nav_msgs::Path path_, gt_path_;
    long n_est_ = 0;
    std::vector<double> latency_;
    double t_first_ = -1.0;
    bool warned_ = false;
    rosbag::Bag bag_;
    std::unique_ptr<rosbag::View> view_;
    rosbag::View::iterator it_;
    bool paced_ = false;
    double bag_t0_ = 0.0;
    std::chrono::steady_clock::time_point wall_t0_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Event> queue_;
    std::chrono::steady_clock::time_point last_rx_;
    bool got_any_ = false, grid_set_ = false;
    double grid_t0_ = 0.0;
};

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "kims_online");
    ros::NodeHandle nh, pnh("~");
    std::string arg_str;
    if (!pnh.getParam("args", arg_str)) {
        ROS_ERROR("[kims_online] ~args is required: the key:=value settings of scripts/run.py kims <dataset> <sequence> --args-only");
        return 1;
    }
    std::map<std::string, std::string> args;
    auto parse = [&args](const std::string& str) {
        std::istringstream is(str);
        std::string tok;
        while (is >> tok) {
            size_t p = tok.find(":="), n = 2;
            if (p == std::string::npos) { p = tok.find('='); n = 1; }
            if (p != std::string::npos && p > 0) args[tok.substr(0, p)] = tok.substr(p + n);
            else ROS_WARN("[kims_online] ignoring setting '%s' (expected key=value)", tok.c_str());
        }
    };
    parse(arg_str);
    std::string set_str;
    if (pnh.getParam("set", set_str)) parse(set_str);
    if (kims::getArg<std::string>(args, "dataset", "") != "iului")
        ROS_WARN("[kims_online] the UWB input is a Nooploop LinkTrack tag frame (IULUI); dataset is %s",
                 kims::getArg<std::string>(args, "dataset", "").c_str());

    kims::Session S;
    if (!kims::loadSession(args, S, false)) return 1;
    RosSource source(nh, pnh, args, S.num_anchors);
    ros::AsyncSpinner spinner(1);
    spinner.start();
    const int rc = kims::kimsRun(args, S, &source);
    spinner.stop();
    source.report();
    return rc;
}
