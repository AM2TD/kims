#pragma once
#include "dims.h"

#include <cuda_runtime.h>
#include <cmath>

__constant__ double d_tag_offset_[NUM_TAG_MAX * 3];
__constant__ int    d_num_anchors_;
__constant__ int    d_num_tags_;

__constant__ double d_sigma_[DIM_U];        // per-step sigma used in the IMU cost
__constant__ double d_sigma_prop_[DIM_U];   // sigma of the constant per-rollout offset drawn by noise_kernel
__constant__ double d_sigma_imu_[2];
__constant__ int    d_ctrl_mode_;
__constant__ double d_uwb_delta_;
__constant__ double d_ctrl_delta_;
__constant__ double d_floor_penalty_;
__constant__ int    d_floor_hard_;      // 1: a rollout step with z < 0 gets a prohibitive cost (sample rejected)
__constant__ int    d_uwb_model_;          // 0: Tukey on (dist - z)/std (published); 1: LOS/NLOS Gaussian mixture of z - dist
__constant__ double d_mix_[6];
// TDOA (dataset util): slot ai measures |tag - B_ai| - |tag - A_ai| with A_ai = anchor[ai]; values are stored + TDOA_SHIFT
#define TDOA_SLOTS_MAX 64
#define TDOA_SHIFT     100.0
__constant__ int    d_tdoa_;
__constant__ double d_anchorB_[TDOA_SLOTS_MAX * 3];             // model 1: eps mu1 s1 mu2 s2; model 2: eps mu1 s1 L; model 3: eps mu1 s1 mu2 s2 tau

__device__ void vector_to_skew(const double* a, double* A_hat) {
    A_hat[0] =    0.0;   A_hat[1] = -a[2];   A_hat[2] =  a[1];
    A_hat[3] =  a[2];    A_hat[4] =   0.0;   A_hat[5] = -a[0];
    A_hat[6] = -a[1];    A_hat[7] =  a[0];   A_hat[8] =   0.0;
}

__device__ void Exp_SO3(const double* omega, double* R) {
    constexpr double TOL = 1e-6;
    double angle = sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);

    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;

    if (angle < TOL) return;

    double a[3] = {omega[0]/angle, omega[1]/angle, omega[2]/angle};
    double c = cos(angle);
    double s = sin(angle);

    double A_hat[9];
    vector_to_skew(a, A_hat);

    double A_hat2[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            A_hat2[3*i+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                A_hat2[3*i+j] += A_hat[3*i+k] * A_hat[3*k+j];
        }
    for (int i = 0; i < 9; ++i)
        R[i] += s * A_hat[i] + (1.0 - c) * A_hat2[i];
}

__device__ void step_dynamics(const double* x, const double* u,
                               double dt, const double* g, double* x_next) {
    double accWorld[3];
    accWorld[0] = x[6]*u[0]  + x[7]*u[1]  + x[8]*u[2]  + g[0];
    accWorld[1] = x[9]*u[0]  + x[10]*u[1] + x[11]*u[2] + g[1];
    accWorld[2] = x[12]*u[0] + x[13]*u[1] + x[14]*u[2] + g[2];

    x_next[0] = x[0] + x[3]*dt + 0.5*accWorld[0]*dt*dt;
    x_next[1] = x[1] + x[4]*dt + 0.5*accWorld[1]*dt*dt;
    x_next[2] = x[2] + x[5]*dt + 0.5*accWorld[2]*dt*dt;
    x_next[3] = x[3] + accWorld[0]*dt;
    x_next[4] = x[4] + accWorld[1]*dt;
    x_next[5] = x[5] + accWorld[2]*dt;

    double omega[3] = {u[3]*dt, u[4]*dt, u[5]*dt};
    double E[9];
    Exp_SO3(omega, E);

    x_next[6]  = E[0]*x[6]  + E[3]*x[7]  + E[6]*x[8];
    x_next[7]  = E[1]*x[6]  + E[4]*x[7]  + E[7]*x[8];
    x_next[8]  = E[2]*x[6]  + E[5]*x[7]  + E[8]*x[8];
    x_next[9]  = E[0]*x[9]  + E[3]*x[10] + E[6]*x[11];
    x_next[10] = E[1]*x[9]  + E[4]*x[10] + E[7]*x[11];
    x_next[11] = E[2]*x[9]  + E[5]*x[10] + E[8]*x[11];
    x_next[12] = E[0]*x[12] + E[3]*x[13] + E[6]*x[14];
    x_next[13] = E[1]*x[12] + E[4]*x[13] + E[7]*x[14];
    x_next[14] = E[2]*x[12] + E[5]*x[13] + E[8]*x[14];
}

__device__ inline double loss(double r, double delta) {
    if (fabs(r) >= delta) return delta * delta / 6.0;
    double t = 1.0 - (r / delta) * (r / delta);
    return delta * delta / 6.0 * (1.0 - t * t * t);
}

// Negative log-likelihood of one range given the predicted distance (d = dist - z) and its std column, with the
// derivative with respect to d and a Gauss-Newton weight (d rho / d d ~ W * (d - d0)). Model 1 is the two-component
// mixture of the range error z - dist fitted on the calibration flights: LOS core N(mu1, s1^2), NLOS delay N(mu2, s2^2).
__device__ inline void range_nll(double d, double sd, double LD, double* rho, double* drho, double* W) {
    if (d_uwb_model_ == 3) {
        // LOS core = N(mu1, s1^2) convolved with an exponential multipath delay (mean tau) [exGaussian], NLOS N(mu2, s2^2)
        const double eps = d_mix_[0], m1 = d_mix_[1], s1 = d_mix_[2], m2 = d_mix_[3], s2 = d_mix_[4], tau = d_mix_[5];
        const double r = -d, lam = 1.0 / tau;
        const double zz = (r - m1 - lam * s1 * s1) / s1;
        // log Phi(zz) and the inverse Mills ratio phi/Phi, stable for very negative zz via erfcx
        double logPhi, mills;
        if (zz < 0.0) { const double ex = erfcx(-zz * 0.70710678118654752); logPhi = log(0.5 * ex) - 0.5 * zz * zz; mills = 0.79788456080286536 / ex; }
        else { const double Ph = normcdf(zz); logPhi = log(Ph); mills = exp(-0.5 * zz * zz) * 0.39894228040143268 / Ph; }
        const double l1 = log(1.0 - eps) + log(lam) + lam * (m1 + 0.5 * lam * s1 * s1 - r) + logPhi;
        const double a2 = (r - m2) / s2;
        const double l2 = log(eps) - log(s2) - 0.5 * a2 * a2 - 0.9189385332;
        const double m = l1 > l2 ? l1 : l2;
        const double L = m + log(exp(l1 - m) + exp(l2 - m));
        const double p1 = exp(l1 - L), p2 = exp(l2 - L);
        const double dl1 = -lam + mills / s1;                       // d log f1 / d r
        const double dl2 = -(r - m2) / (s2 * s2);
        *rho = -L;
        *drho = (p1 * dl1 + p2 * dl2);                              // d rho / d d = -d rho / d r = d L / d r
        *W = p1 * mills * (zz + mills) / (s1 * s1) + p2 / (s2 * s2);
    } else if (d_uwb_model_ == 2) {
        // LOS core N(mu1, s1^2) + NLOS delay uniform on [0, L] (smooth edges): NLOS ranges carry no position pull
        const double eps = d_mix_[0], m1 = d_mix_[1], s1 = d_mix_[2], Lu = d_mix_[3];
        const double r = -d;
        const double a1 = (r - m1) / s1;
        const double l1 = log(1.0 - eps) - log(s1) - 0.5 * a1 * a1 - 0.9189385332;
        const double z0 = r / 0.01, z1 = (Lu - r) / 0.05;          // log-sigmoid edges
        const double ls0 = z0 > 0 ? -log1p(exp(-z0)) : z0 - log1p(exp(z0));
        const double ls1 = z1 > 0 ? -log1p(exp(-z1)) : z1 - log1p(exp(z1));
        const double l2 = log(eps / Lu) + ls0 + ls1;
        const double m = l1 > l2 ? l1 : l2;
        const double L = m + log(exp(l1 - m) + exp(l2 - m));
        const double p1 = exp(l1 - L), p2 = exp(l2 - L);
        const double dl2 = (1.0 - exp(ls0)) / 0.01 - (1.0 - exp(ls1)) / 0.05;   // d l2 / d r
        *rho = (log(1.0 - eps) - log(s1) - 0.9189385332) - L;
        *drho = -(p1 * (r - m1) / (s1 * s1) - p2 * dl2);
        *W = p1 / (s1 * s1);
    } else if (d_uwb_model_ == 1) {
        const double eps = d_mix_[0], m1 = d_mix_[1], s1 = d_mix_[2], m2 = d_mix_[3], s2 = d_mix_[4];
        const double r = -d;                                     // range error z - dist
        const double a1 = (r - m1) / s1, a2 = (r - m2) / s2;
        const double l1 = log(1.0 - eps) - log(s1) - 0.5 * a1 * a1;
        const double l2 = log(eps) - log(s2) - 0.5 * a2 * a2;
        const double m = l1 > l2 ? l1 : l2;
        const double L = m + log(exp(l1 - m) + exp(l2 - m));
        const double p1 = exp(l1 - L), p2 = exp(l2 - L);
        *rho = (log(1.0 - eps) - log(s1)) - L;                   // >= 0 up to the second component's contribution
        *drho = -(p1 * (r - m1) / (s1 * s1) + p2 * (r - m2) / (s2 * s2));
        *W = p1 / (s1 * s1) + p2 / (s2 * s2);
    } else {
        const double e = d / sd;
        *rho = loss(e, LD);
        const double q = (e / LD) * (e / LD);
        const double w = fabs(e) < LD ? (1.0 - q) * (1.0 - q) : 0.0;
        *drho = w * e / sd;
        *W = w / (sd * sd);
    }
}

__device__ double uwb_step_cost(const double* x,
                                const double* anchor,
                                const double* ranges,
                                const double* stds,
                                double rbias = 0.0) {         // common range bias (measured = distance + rbias + noise)
    const double LOSS_DELTA = d_uwb_delta_ > 0.0 ? d_uwb_delta_ : 3.0;
    const int num_anc  = d_num_anchors_;
    const int num_tags = d_num_tags_;

    double tag_w[NUM_TAG_MAX][3];
    for (int ti = 0; ti < num_tags; ++ti) {
        const double* off = &d_tag_offset_[3 * ti];
        for (int i = 0; i < 3; ++i)
            tag_w[ti][i] = x[i] + x[6+3*i]*off[0] + x[7+3*i]*off[1] + x[8+3*i]*off[2];
    }

    double s = 0.0;

    if (d_floor_hard_ && x[2] < 0.0)
        s += 1.0e6;                                   // hard floor: weight exp(-1e6) = 0, i.e. the sample is rejected
    else if (d_floor_penalty_ > 0.0 && x[2] < 0.0)
        s += d_floor_penalty_ * x[2] * x[2];

    for (int ti = 0; ti < num_tags; ++ti) {
        for (int ai = 0; ai < num_anc; ++ai) {
            const int ri = ti * num_anc + ai;
            if (ranges[ri] <= 0.0) continue;
            const double dx = tag_w[ti][0] - anchor[3*ai + 0];
            const double dy = tag_w[ti][1] - anchor[3*ai + 1];
            const double dz = tag_w[ti][2] - anchor[3*ai + 2];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (d_tdoa_) {
                const double bx = tag_w[ti][0] - d_anchorB_[3*ai], by = tag_w[ti][1] - d_anchorB_[3*ai + 1], bz = tag_w[ti][2] - d_anchorB_[3*ai + 2];
                dist = sqrt(bx*bx + by*by + bz*bz) - dist + TDOA_SHIFT;
            }
            double rho, drho, W;
            range_nll(dist + rbias - ranges[ri], stds[ri], LOSS_DELTA, &rho, &drho, &W);
            s += rho;
        }
    }
    return s;
}

// Unsaturated chi-square of one epoch: sum ((dist - z)/sigma)^2 over valid ranges (no robust loss).
__device__ double uwb_step_chi2(const double* x, const double* anchor,
                                const double* ranges, const double* stds) {
    const int num_anc  = d_num_anchors_;
    const int num_tags = d_num_tags_;
    double s = 0.0;
    for (int ti = 0; ti < num_tags; ++ti) {
        const double* off = &d_tag_offset_[3 * ti];
        double tw[3];
        for (int i = 0; i < 3; ++i)
            tw[i] = x[i] + x[6+3*i]*off[0] + x[7+3*i]*off[1] + x[8+3*i]*off[2];
        for (int ai = 0; ai < num_anc; ++ai) {
            const int ri = ti * num_anc + ai;
            if (ranges[ri] <= 0.0) continue;
            const double dx = tw[0] - anchor[3*ai], dy = tw[1] - anchor[3*ai+1], dz = tw[2] - anchor[3*ai+2];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (d_tdoa_) {
                const double bx = tw[0] - d_anchorB_[3*ai], by = tw[1] - d_anchorB_[3*ai+1], bz = tw[2] - d_anchorB_[3*ai+2];
                dist = sqrt(bx*bx + by*by + bz*bz) - dist + TDOA_SHIFT;
            }
            const double e = (dist - ranges[ri]) / stds[ri];
            s += e * e;
        }
    }
    return s;
}

__device__ double imu_step_cost(const double* ctrl3, const double* imu3,
                                const double* sigma3) {
    double s = 0.0;
    for (int j = 0; j < 3; ++j) {
        double e = (ctrl3[j] - imu3[j]) / sigma3[j];
        s += e * e;
    }
    return s;
}

__device__ inline double imu_step_cost_tukey(const double* ctrl3,
                                             const double* imu3,
                                             double sigma) {
    double s = 0.0;
    const double cd = d_ctrl_delta_ > 0.0 ? d_ctrl_delta_ : 1.0;
    for (int j = 0; j < 3; ++j)
        s += loss((ctrl3[j] - imu3[j]) / sigma, cd);
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// Single-precision rollout path (node arg rollout_fp32:=1). The same equations as
// the double functions above with the state, the dynamics and the per-step costs in float; the window cost is accumulated
// in double by the caller (importance weights depend on differences of sums over T steps). FP64 runs at 1/64 of the FP32
// rate on this GPU.
__device__ inline float erfcx_f(float x) { return erfcxf(x); }
__device__ inline float normcdf_f(float x) { return normcdff(x); }

__device__ inline void Exp_SO3_f(const float* omega, float* R) {
    const float angle = sqrtf(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
    R[0] = 1.f; R[1] = 0.f; R[2] = 0.f; R[3] = 0.f; R[4] = 1.f; R[5] = 0.f; R[6] = 0.f; R[7] = 0.f; R[8] = 1.f;
    if (angle < 1e-6f) return;
    const float a[3] = {omega[0]/angle, omega[1]/angle, omega[2]/angle};
    const float c = cosf(angle), s = sinf(angle);
    const float A[9] = {0.f, -a[2], a[1], a[2], 0.f, -a[0], -a[1], a[0], 0.f};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float a2 = 0.f;
            for (int k = 0; k < 3; ++k) a2 += A[3*i+k] * A[3*k+j];
            R[3*i+j] += s * A[3*i+j] + (1.f - c) * a2;
        }
}

__device__ inline void step_dynamics_f(const float* x, const float* u, float dt, const float* g, float* x_next) {
    float aw[3];
    aw[0] = x[6]*u[0]  + x[7]*u[1]  + x[8]*u[2]  + g[0];
    aw[1] = x[9]*u[0]  + x[10]*u[1] + x[11]*u[2] + g[1];
    aw[2] = x[12]*u[0] + x[13]*u[1] + x[14]*u[2] + g[2];
    for (int k = 0; k < 3; ++k) { x_next[k] = x[k] + x[3+k]*dt + 0.5f*aw[k]*dt*dt; x_next[3+k] = x[3+k] + aw[k]*dt; }
    const float om[3] = {u[3]*dt, u[4]*dt, u[5]*dt};
    float E[9]; Exp_SO3_f(om, E);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            x_next[6 + 3*r + c] = E[c]*x[6 + 3*r] + E[3 + c]*x[7 + 3*r] + E[6 + c]*x[8 + 3*r];
}

__device__ inline float loss_f(float r, float delta) {
    if (fabsf(r) >= delta) return delta * delta / 6.f;
    const float t = 1.f - (r / delta) * (r / delta);
    return delta * delta / 6.f * (1.f - t * t * t);
}

__device__ inline float range_nll_f(float d, float sd, float LD) {      // rho only (the rollout uses the cost)
    if (d_uwb_model_ == 3) {
        const float eps = (float)d_mix_[0], m1 = (float)d_mix_[1], s1 = (float)d_mix_[2], m2 = (float)d_mix_[3], s2 = (float)d_mix_[4], tau = (float)d_mix_[5];
        const float r = -d, lam = 1.f / tau;
        const float zz = (r - m1 - lam * s1 * s1) / s1;
        const float logPhi = zz < 0.f ? logf(0.5f * erfcx_f(-zz * 0.70710678f)) - 0.5f * zz * zz : logf(normcdf_f(zz));
        const float l1 = logf(1.f - eps) + logf(lam) + lam * (m1 + 0.5f * lam * s1 * s1 - r) + logPhi;
        const float a2 = (r - m2) / s2;
        const float l2 = logf(eps) - logf(s2) - 0.5f * a2 * a2 - 0.9189385332f;
        const float m = l1 > l2 ? l1 : l2;
        return -(m + logf(expf(l1 - m) + expf(l2 - m)));
    } else if (d_uwb_model_ == 2) {
        const float eps = (float)d_mix_[0], m1 = (float)d_mix_[1], s1 = (float)d_mix_[2], Lu = (float)d_mix_[3];
        const float r = -d, a1 = (r - m1) / s1;
        const float l1 = logf(1.f - eps) - logf(s1) - 0.5f * a1 * a1 - 0.9189385332f;
        const float z0 = r / 0.01f, z1 = (Lu - r) / 0.05f;
        const float ls0 = z0 > 0.f ? -log1pf(expf(-z0)) : z0 - log1pf(expf(z0));
        const float ls1 = z1 > 0.f ? -log1pf(expf(-z1)) : z1 - log1pf(expf(z1));
        const float l2 = logf(eps / Lu) + ls0 + ls1;
        const float m = l1 > l2 ? l1 : l2;
        return (logf(1.f - eps) - logf(s1) - 0.9189385332f) - (m + logf(expf(l1 - m) + expf(l2 - m)));
    } else if (d_uwb_model_ == 1) {
        const float eps = (float)d_mix_[0], m1 = (float)d_mix_[1], s1 = (float)d_mix_[2], m2 = (float)d_mix_[3], s2 = (float)d_mix_[4];
        const float r = -d, a1 = (r - m1) / s1, a2 = (r - m2) / s2;
        const float l1 = logf(1.f - eps) - logf(s1) - 0.5f * a1 * a1;
        const float l2 = logf(eps) - logf(s2) - 0.5f * a2 * a2;
        const float m = l1 > l2 ? l1 : l2;
        return (logf(1.f - eps) - logf(s1)) - (m + logf(expf(l1 - m) + expf(l2 - m)));
    }
    return loss_f(d / sd, LD);
}

// tag positions of one state (float), shared by the cost and the chi-square
__device__ inline void tag_positions_f(const float* x, float tw[][3], int num_tags) {
    for (int ti = 0; ti < num_tags; ++ti) {
        const float o0 = (float)d_tag_offset_[3*ti], o1 = (float)d_tag_offset_[3*ti + 1], o2 = (float)d_tag_offset_[3*ti + 2];
        for (int i = 0; i < 3; ++i) tw[ti][i] = x[i] + x[6+3*i]*o0 + x[7+3*i]*o1 + x[8+3*i]*o2;
    }
}

// predicted range (or TDOA + shift) of tag ti to slot ai, float
__device__ inline float pred_range_f(const float* tw, const double* anc, int ai) {
    const float dx = tw[0] - (float)anc[3*ai], dy = tw[1] - (float)anc[3*ai + 1], dz = tw[2] - (float)anc[3*ai + 2];
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    if (d_tdoa_) {
        const float bx = tw[0] - (float)d_anchorB_[3*ai], by = tw[1] - (float)d_anchorB_[3*ai + 1], bz = tw[2] - (float)d_anchorB_[3*ai + 2];
        dist = sqrtf(bx*bx + by*by + bz*bz) - dist + (float)TDOA_SHIFT;
    }
    return dist;
}

__device__ inline float uwb_step_cost_f(const float* x, const double* anc, const double* ranges, const double* stds, float rbias) {
    const float LD = d_uwb_delta_ > 0.0 ? (float)d_uwb_delta_ : 3.f;
    const int num_anc = d_num_anchors_, num_tags = d_num_tags_;
    float tw[NUM_TAG_MAX][3]; tag_positions_f(x, tw, num_tags);
    float s = 0.f;
    if (d_floor_hard_ && x[2] < 0.f) s += 1.0e6f;
    else if (d_floor_penalty_ > 0.0 && x[2] < 0.f) s += (float)d_floor_penalty_ * x[2] * x[2];
    for (int ti = 0; ti < num_tags; ++ti)
        for (int ai = 0; ai < num_anc; ++ai) {
            const int ri = ti * num_anc + ai;
            if (ranges[ri] <= 0.0) continue;
            s += range_nll_f(pred_range_f(tw[ti], anc, ai) + rbias - (float)ranges[ri], (float)stds[ri], LD);
        }
    return s;
}

__device__ inline float uwb_step_chi2_f(const float* x, const double* anc, const double* ranges, const double* stds) {
    const int num_anc = d_num_anchors_, num_tags = d_num_tags_;
    float tw[NUM_TAG_MAX][3]; tag_positions_f(x, tw, num_tags);
    float s = 0.f;
    for (int ti = 0; ti < num_tags; ++ti)
        for (int ai = 0; ai < num_anc; ++ai) {
            const int ri = ti * num_anc + ai;
            if (ranges[ri] <= 0.0) continue;
            const float e = (pred_range_f(tw[ti], anc, ai) - (float)ranges[ri]) / (float)stds[ri];
            s += e * e;
        }
    return s;
}
