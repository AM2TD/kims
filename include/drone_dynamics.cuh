#pragma once
#include "dims.h"

#include <cuda_runtime.h>
#include <cmath>

__constant__ double d_tag_offset_[NUM_TAG_MAX * 3];
__constant__ int    d_num_anchors_;
__constant__ int    d_num_tags_;

__constant__ double d_sigma_[DIM_U];
__constant__ double d_sigma_imu_[2];
__constant__ int    d_ctrl_mode_;
__constant__ double d_uwb_delta_;
__constant__ double d_ctrl_delta_;
__constant__ double d_floor_penalty_;

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

__device__ double uwb_step_cost(const double* x,
                                const double* anchor,
                                const double* ranges,
                                const double* stds) {
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

    if (d_floor_penalty_ > 0.0 && x[2] < 0.0)
        s += d_floor_penalty_ * x[2] * x[2];

    for (int ti = 0; ti < num_tags; ++ti) {
        for (int ai = 0; ai < num_anc; ++ai) {
            const int ri = ti * num_anc + ai;
            if (ranges[ri] <= 0.0) continue;
            const double dx = tag_w[ti][0] - anchor[3*ai + 0];
            const double dy = tag_w[ti][1] - anchor[3*ai + 1];
            const double dz = tag_w[ti][2] - anchor[3*ai + 2];
            const double dist = sqrt(dx*dx + dy*dy + dz*dz);
            s += loss((dist - ranges[ri]) / stds[ri], LOSS_DELTA);
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
