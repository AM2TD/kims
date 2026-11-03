#include "mppi_solver.cuh"
#include "drone_dynamics.cuh"
#include <Eigen/Dense>

#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

// abort on any CUDA error instead of continuing with unrun kernels (an out-of-memory start otherwise yields zeros)
static void cuda_check(const char* where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "[mppi] CUDA error at %s: %s\n", where, cudaGetErrorString(e)); std::abort(); }
}


__global__ void init_rng(curandStatePhilox4_32_10_t* states,
                         unsigned long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, i * 1024ULL, &states[i]);
}

__global__ void noise_kernel(curandStatePhilox4_32_10_t* states,
                             double* noise, int batch, int T) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) return;

    curandStatePhilox4_32_10_t local = states[sample];
    double v[DIM_U];
    for (int j = 0; j < DIM_U; ++j)
        v[j] = curand_normal_double(&local) * d_sigma_prop_[j];
    states[sample] = local;

    for (int t = 0; t < T; ++t) {
        const int base = (sample * T + t) * DIM_U;
        for (int j = 0; j < DIM_U; ++j) noise[base + j] = v[j];
    }
}

__global__ void rollout_kernel(int N, int T, int num_rng, double dt,
                               const double* g,
                               const double* anchor,
                               const double* ranges,
                               const double* stds,
                               const double* accgyr,
                               const double* U0,
                               const double* noise,
                               const double* x0,
                               const double* dv,        // N x 3 head-velocity increments or nullptr
                               double gamma,
                               double w_uwb, double w_acc, double w_gyr,
                               double* Ui, double* cost,
                               double* cost_uwb, double* cost_acc, double* cost_gyr,
                               double* cost_uwb_last,
                               double* cost_uwb_newest,
                               const double* dx0,       // N x 9 head offsets (dp, dv, dtheta body) or nullptr
                               double* x1_out,          // N x 16: state after the first step and the first epoch's UWB cost, or nullptr
                               const double* dts,       // T per-step time steps (true epoch spacing) or nullptr (uniform dt)
                               const double* rbias,     // N common range biases (head coordinate) or nullptr
                               double* xT_out,          // N x DIM_X: state at the newest window epoch (lag-0 output), or nullptr
                               double* xall_out) {      // N x T x 3: position at every window epoch (flush output), or nullptr
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;
    const double rb_s = rbias ? rbias[sample] : 0.0;

    double x[DIM_X];
    for (int i = 0; i < DIM_X; ++i) x[i] = x0[i];
    if (dv != nullptr) for (int i = 0; i < 3; ++i) x[3 + i] += dv[sample * 3 + i];
    if (dx0 != nullptr) {
        const double* d = dx0 + sample * 9;
        for (int i = 0; i < 3; ++i) { x[i] += d[i]; x[3 + i] += d[3 + i]; }
        double E[9]; Exp_SO3(d + 6, E);                    // R <- R Exp(dtheta)
        double R[9]; for (int i = 0; i < 9; ++i) R[i] = x[6 + i];
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            x[6 + 3*r + c] = R[3*r]*E[c] + R[3*r + 1]*E[3 + c] + R[3*r + 2]*E[6 + c];
    }

    double c = 0.0, c_uwb = 0.0, c_acc = 0.0, c_gyr = 0.0;

    const double* noise_base = noise + sample * T * DIM_U;
    double*       ui_base    = Ui    + sample * T * DIM_U;

    for (int t = 0; t < T; ++t) {
        const int off_u = t * DIM_U;
        const double* n  = noise_base + off_u;
        const double* u0 = U0         + off_u;
        double*       ui = ui_base    + off_u;

        for (int k = 0; k < DIM_U; ++k)
            ui[k] = u0[k] + n[k];

        double x_next[DIM_X];
        step_dynamics(x, ui, dts ? dts[t] : dt, g, x_next);

        double cu = w_uwb * uwb_step_cost(x_next, anchor, &ranges[t * num_rng], &stds[t * num_rng], rb_s);
        if (xall_out) for (int k = 0; k < 3; ++k) xall_out[((size_t)sample * T + t) * 3 + k] = x_next[k];
        if (t == 0 && x1_out != nullptr) {            // state at the first window epoch and that epoch's UWB cost
            for (int k = 0; k < DIM_X; ++k) x1_out[sample * (DIM_X + 1) + k] = x_next[k];
            x1_out[sample * (DIM_X + 1) + DIM_X] = cu;
        }
        if (t == T - 1) {
            cost_uwb_last[sample] = uwb_step_chi2(x_next, anchor, &ranges[t * num_rng], &stds[t * num_rng]);
            if (cost_uwb_newest) cost_uwb_newest[sample] = cu;
            if (xT_out) for (int k = 0; k < DIM_X; ++k) xT_out[sample * DIM_X + k] = x_next[k];
        }

        double ca, cg;
        if (d_ctrl_mode_ == 1) {
            ca = w_acc * imu_step_cost_tukey(ui,     &accgyr[t * 6],     d_sigma_imu_[0]);
            cg = w_gyr * imu_step_cost_tukey(ui + 3, &accgyr[t * 6 + 3], d_sigma_imu_[1]);
        } else {
            ca = w_acc * imu_step_cost(ui,     &accgyr[t * 6],     &d_sigma_[0]);
            cg = w_gyr * imu_step_cost(ui + 3, &accgyr[t * 6 + 3], &d_sigma_[3]);
        }
        c += cu + ca + cg;
        c_uwb += cu;  c_acc += ca;  c_gyr += cg;

        for (int k = 0; k < DIM_X; ++k) x[k] = x_next[k];
    }

    cost[sample]     = c;
    cost_uwb[sample] = c_uwb;
    cost_acc[sample] = c_acc;
    cost_gyr[sample] = c_gyr;
}

// Single-precision twin of rollout_kernel (node arg rollout_fp32:=1): state,
// dynamics and per-step costs in float, window costs accumulated in double, inputs/outputs kept in double.
__global__ void rollout_kernel_f(int N, int T, int num_rng, double dt,
                                 const double* g, const double* anchor, const double* ranges, const double* stds,
                                 const double* accgyr, const double* U0, const double* noise, const double* x0,
                                 const double* dv, double gamma, double w_uwb, double w_acc, double w_gyr,
                                 double* Ui, double* cost, double* cost_uwb, double* cost_acc, double* cost_gyr,
                                 double* cost_uwb_last, double* cost_uwb_newest, const double* dx0, double* x1_out,
                                 const double* dts, const double* rbias, double* xT_out, double* xall_out) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;
    const float rb_s = rbias ? (float)rbias[sample] : 0.f;
    const double* anc = anchor;                          // read per use: TDOA slots exceed NUM_ANC_MAX
    const float gf[3] = {(float)g[0], (float)g[1], (float)g[2]};
    float x[DIM_X];
    for (int i = 0; i < DIM_X; ++i) x[i] = (float)x0[i];
    if (dv != nullptr) for (int i = 0; i < 3; ++i) x[3 + i] += (float)dv[sample * 3 + i];
    if (dx0 != nullptr) {
        const double* d = dx0 + sample * 9;
        for (int i = 0; i < 3; ++i) { x[i] += (float)d[i]; x[3 + i] += (float)d[3 + i]; }
        const float th[3] = {(float)d[6], (float)d[7], (float)d[8]};
        float E[9]; Exp_SO3_f(th, E);
        float R[9]; for (int i = 0; i < 9; ++i) R[i] = x[6 + i];
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            x[6 + 3*r + c] = R[3*r]*E[c] + R[3*r + 1]*E[3 + c] + R[3*r + 2]*E[6 + c];
    }
    double c = 0.0, c_uwb = 0.0, c_acc = 0.0, c_gyr = 0.0;
    const double* noise_base = noise + sample * T * DIM_U;
    double*       ui_base    = Ui    + sample * T * DIM_U;
    for (int t = 0; t < T; ++t) {
        const int off_u = t * DIM_U;
        double ud[DIM_U]; float uf[DIM_U];
        for (int k = 0; k < DIM_U; ++k) { ud[k] = U0[off_u + k] + noise_base[off_u + k]; ui_base[off_u + k] = ud[k]; uf[k] = (float)ud[k]; }
        float xn[DIM_X];
        step_dynamics_f(x, uf, (float)(dts ? dts[t] : dt), gf, xn);
        const double cu = w_uwb * (double)uwb_step_cost_f(xn, anc, &ranges[t * num_rng], &stds[t * num_rng], rb_s);
        if (xall_out) for (int k = 0; k < 3; ++k) xall_out[((size_t)sample * T + t) * 3 + k] = (double)xn[k];
        if (t == 0 && x1_out != nullptr) {
            for (int k = 0; k < DIM_X; ++k) x1_out[sample * (DIM_X + 1) + k] = (double)xn[k];
            x1_out[sample * (DIM_X + 1) + DIM_X] = cu;
        }
        if (t == T - 1) {
            cost_uwb_last[sample] = (double)uwb_step_chi2_f(xn, anc, &ranges[t * num_rng], &stds[t * num_rng]);
            if (cost_uwb_newest) cost_uwb_newest[sample] = cu;
            if (xT_out) for (int k = 0; k < DIM_X; ++k) xT_out[sample * DIM_X + k] = (double)xn[k];
        }
        double ca = 0.0, cg = 0.0;                       // IMU terms on the double inputs (cheap, w_acc/w_gyr often 0)
        if (w_acc != 0.0 || w_gyr != 0.0) {
            if (d_ctrl_mode_ == 1) {
                ca = w_acc * imu_step_cost_tukey(ud,     &accgyr[t * 6],     d_sigma_imu_[0]);
                cg = w_gyr * imu_step_cost_tukey(ud + 3, &accgyr[t * 6 + 3], d_sigma_imu_[1]);
            } else {
                ca = w_acc * imu_step_cost(ud,     &accgyr[t * 6],     &d_sigma_[0]);
                cg = w_gyr * imu_step_cost(ud + 3, &accgyr[t * 6 + 3], &d_sigma_[3]);
            }
        }
        c += cu + ca + cg; c_uwb += cu; c_acc += ca; c_gyr += cg;
        for (int k = 0; k < DIM_X; ++k) x[k] = xn[k];
    }
    cost[sample] = c; cost_uwb[sample] = c_uwb; cost_acc[sample] = c_acc; cost_gyr[sample] = c_gyr;
}

// ------------------------------------------- Rao-Blackwellised head/bias ----
// For one input sample the window cost is linearised in the head/bias offset delta = (dp, dv, dtheta_body, dba, dbg):
// positions move exactly linearly with dp and dv, attitude and bias enter to first order. Per sample the kernel returns
// the cost at delta = 0, its gradient and Gauss-Newton Hessian (Tukey IRLS weights), the same for the newest epoch
// alone (branch evidence) and the Hessian of the leaving epoch alone (arrival cost), the state after the first step
// and its sensitivity to delta.
// one thread per sample with a long sequential loop: small blocks spread the 2048 samples over all SMs
#define RB_BLOCK 32
#define RB_H   15
#define RB_NS  120
#define RB_OUT 557
#define RB_O_J0 0
#define RB_O_G  1
#define RB_O_H  16
#define RB_O_JL 136
#define RB_O_GL 137
#define RB_O_HL 152
#define RB_O_H0 272
#define RB_O_X1 392
#define RB_O_S1 407
#define RB_O_G0 542
#ifndef RB_DOUBLE
typedef float rb_t;      // sensitivities and accumulators in single precision (FP64 is 1/64 rate on this GPU)
#else
typedef double rb_t;
#endif
__device__ __host__ inline int rb_idx(int a, int b) { return a * RB_H - a * (a - 1) / 2 + (b - a); }   // a <= b

// single-precision range likelihood for the RB kernel (residual formed in double, then cast); models 0 and 1 natively,
// the others through the double-precision range_nll
__device__ inline void range_nll_f(double dd, double sdd, double LDd, float* rho, float* drho, float* W) {
    if (d_uwb_model_ == 1) {
        const float eps = (float)d_mix_[0], m1 = (float)d_mix_[1], s1 = (float)d_mix_[2], m2 = (float)d_mix_[3], s2 = (float)d_mix_[4];
        const float r = (float)(-dd);
        const float a1 = (r - m1) / s1, a2 = (r - m2) / s2;
        const float c1 = logf(1.f - eps) - logf(s1);
        const float l1 = c1 - 0.5f * a1 * a1;
        const float l2 = logf(eps) - logf(s2) - 0.5f * a2 * a2;
        const float m = fmaxf(l1, l2);
        const float L = m + logf(expf(l1 - m) + expf(l2 - m));
        const float p1 = expf(l1 - L), p2 = expf(l2 - L);
        *rho = c1 - L;
        *drho = -(p1 * (r - m1) / (s1 * s1) + p2 * (r - m2) / (s2 * s2));
        *W = p1 / (s1 * s1) + p2 / (s2 * s2);
    } else if (d_uwb_model_ == 0) {
        const float sd = (float)sdd, LD = (float)LDd, e = (float)dd / sd;
        const float q = (e / LD) * (e / LD);
        const float w = fabsf(e) < LD ? (1.f - q) * (1.f - q) : 0.f;
        *rho = fabsf(e) >= LD ? LD * LD / 6.f : LD * LD / 6.f * (1.f - (1.f - q) * (1.f - q) * (1.f - q));
        *drho = w * e / sd;
        *W = w / (sd * sd);
    } else {
        double r0, d0, w0;
        range_nll(dd, sdd, LDd, &r0, &d0, &w0);
        *rho = (float)r0; *drho = (float)d0; *W = (float)w0;
    }
}

// exponential map with the small-angle series (exact to double precision for |w| < 0.1 rad), no trig in the usual case
__device__ inline void exp_so3_fast(const double* w, double* E) {
    const double t2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    double A, B;
    if (t2 < 0.01) {
        A = 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0));
        B = 0.5 - t2 / 24.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0));
    } else {
        const double t = sqrt(t2); A = sin(t) / t; B = (1.0 - cos(t)) / t2;
    }
    const double K[9] = {0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        double k2 = K[3*i] * K[j] + K[3*i + 1] * K[3 + j] + K[3*i + 2] * K[6 + j];
        E[3*i + j] = (i == j ? 1.0 : 0.0) + A * K[3*i + j] + B * k2;
    }
}

template <int HD> __device__ __host__ constexpr int rb_idx_t(int a, int b) { return a * HD - a * (a - 1) / 2 + (b - a); }

// HD = 9 (head only) or 15 (head + IMU bias). Per-thread arrays are sized by HD so that they stay in registers; the
// newest-epoch and leaving-epoch parts are accumulated directly in the output (touched in one epoch only). Output layout
// keeps the RB_H = 15 packing; entries beyond HD are not written and not read by the host.
template <int HD>
__global__ void rb_rollout_kernel_t(int N, int T, int num_rng, double dt, const double* g,
                                    const double* anchor, const double* ranges, const double* stds,
                                    const double* U0, const double* noise, const double* x0, const double* dts,
                                    double w_uwb, double* out) {
    constexpr int NS = HD * (HD + 1) / 2;
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;
    double* o = out + (size_t)sample * RB_OUT;
    double J0 = 0.0, Jl = 0.0;
    rb_t G[HD], HH[NS];
#pragma unroll
    for (int k = 0; k < HD; ++k) G[k] = 0;
#pragma unroll
    for (int k = 0; k < NS; ++k) HH[k] = 0;
    for (int k = 0; k < RB_H; ++k) { o[RB_O_GL + k] = 0.0; o[RB_O_G0 + k] = 0.0; }
    for (int k = 0; k < RB_NS; ++k) { o[RB_O_HL + k] = 0.0; o[RB_O_H0 + k] = 0.0; }
    double x[DIM_X];
    for (int i = 0; i < DIM_X; ++i) x[i] = x0[i];
    rb_t Sp[3 * HD], Sv[3 * HD], Sf[3 * HD];
#pragma unroll
    for (int k = 0; k < 3 * HD; ++k) { Sp[k] = 0; Sv[k] = 0; Sf[k] = 0; }
    for (int i = 0; i < 3; ++i) { Sp[i * HD + i] = 1; Sv[i * HD + 3 + i] = 1; Sf[i * HD + 6 + i] = 1; }
    const double LD = d_uwb_delta_ > 0.0 ? d_uwb_delta_ : 3.0;
    const double* nb = noise + (size_t)sample * T * DIM_U;
    for (int t = 0; t < T; ++t) {
        double u[DIM_U];
        for (int k = 0; k < DIM_U; ++k) u[k] = U0[t * DIM_U + k] + nb[t * DIM_U + k];
        const double h = dts ? dts[t] : dt;
        const double* R = x + 6;
        // one exponential map serves the state step (as in step_dynamics) and the sensitivity propagation
        double om[3] = {u[3] * h, u[4] * h, u[5] * h}, Ed[9];
        exp_so3_fast(om, Ed);
        double xn[DIM_X];
        for (int r = 0; r < 3; ++r) {
            const double aw = R[3*r] * u[0] + R[3*r + 1] * u[1] + R[3*r + 2] * u[2] + g[r];
            xn[r] = x[r] + x[3 + r] * h + 0.5 * aw * h * h;
            xn[3 + r] = x[3 + r] + aw * h;
        }
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            xn[6 + 3*r + c] = R[3*r] * Ed[c] + R[3*r + 1] * Ed[3 + c] + R[3*r + 2] * Ed[6 + c];
        // sensitivities: acceleration perturbation -R [a]x phi + R dba, attitude phi <- E' phi + h dbg
        const double a0 = u[0], a1 = u[1], a2 = u[2];
        const double Ask[9] = {0.0, -a2, a1, a2, 0.0, -a0, -a1, a0, 0.0};
        rb_t M[9], E[9];
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            M[3*r + c] = (rb_t)(R[3*r] * Ask[c] + R[3*r + 1] * Ask[3 + c] + R[3*r + 2] * Ask[6 + c]);
        for (int q2 = 0; q2 < 9; ++q2) E[q2] = (rb_t)Ed[q2];
        const rb_t hf = (rb_t)h;
#pragma unroll
        for (int c = 0; c < HD; ++c) {
            rb_t da[3], fn[3];
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                da[r] = -(M[3*r] * Sf[c] + M[3*r + 1] * Sf[HD + c] + M[3*r + 2] * Sf[2*HD + c]);
                if (c >= 9 && c < 12) da[r] += (rb_t)R[3*r + (c - 9)];
                fn[r] = E[r] * Sf[c] + E[3 + r] * Sf[HD + c] + E[6 + r] * Sf[2*HD + c];
                if (c >= 12 && c - 12 == r) fn[r] += hf;
            }
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                Sp[r*HD + c] += Sv[r*HD + c] * hf + (rb_t)0.5 * da[r] * hf * hf;
                Sv[r*HD + c] += da[r] * hf;
                Sf[r*HD + c] = fn[r];
            }
        }
        const double* Rn = xn + 6;
        for (int ti = 0; ti < d_num_tags_; ++ti) {
            const double* off = &d_tag_offset_[3 * ti];
            double tw[3]; rb_t B[9];
            for (int i = 0; i < 3; ++i) tw[i] = xn[i] + Rn[3*i] * off[0] + Rn[3*i + 1] * off[1] + Rn[3*i + 2] * off[2];
            const double Osk[9] = {0.0, -off[2], off[1], off[2], 0.0, -off[0], -off[1], off[0], 0.0};
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                B[3*r + c] = (rb_t)(-(Rn[3*r] * Osk[c] + Rn[3*r + 1] * Osk[3 + c] + Rn[3*r + 2] * Osk[6 + c]));
            for (int ai = 0; ai < d_num_anchors_; ++ai) {
                const int ri = ti * d_num_anchors_ + ai;
                const double z = ranges[t * num_rng + ri];
                if (z <= 0.0) continue;
                const double sd = stds[t * num_rng + ri];
                const double d0 = tw[0] - anchor[3*ai], d1 = tw[1] - anchor[3*ai + 1], d2 = tw[2] - anchor[3*ai + 2];
                const double dist = sqrt(d0*d0 + d1*d1 + d2*d2);
                float rho0, drho, Wg;
                range_nll_f(dist - z, sd, LD, &rho0, &drho, &Wg);
                const rb_t un[3] = {(rb_t)(d0 / dist), (rb_t)(d1 / dist), (rb_t)(d2 / dist)};
                const rb_t we = (rb_t)(w_uwb * drho), ww = (rb_t)(w_uwb * Wg);
                rb_t J[HD];
#pragma unroll
                for (int c = 0; c < HD; ++c) {
                    rb_t v = 0;
#pragma unroll
                    for (int r = 0; r < 3; ++r)
                        v += un[r] * (Sp[r*HD + c] + B[3*r] * Sf[c] + B[3*r + 1] * Sf[HD + c] + B[3*r + 2] * Sf[2*HD + c]);
                    J[c] = v;
                }
                const double rho = w_uwb * (double)rho0;
                J0 += rho;
#pragma unroll
                for (int a = 0; a < HD; ++a) {
                    G[a] += we * J[a];
#pragma unroll
                    for (int b2 = a; b2 < HD; ++b2) HH[rb_idx_t<HD>(a, b2)] += ww * J[a] * J[b2];
                }
                if (t == T - 1) {
                    Jl += rho;
                    for (int a = 0; a < HD; ++a) {
                        o[RB_O_GL + a] += we * J[a];
                        for (int b2 = a; b2 < HD; ++b2) o[RB_O_HL + rb_idx(a, b2)] += ww * J[a] * J[b2];
                    }
                }
                if (t == 0) for (int a = 0; a < HD; ++a) {
                    o[RB_O_G0 + a] += we * J[a];
                    for (int b2 = a; b2 < HD; ++b2) o[RB_O_H0 + rb_idx(a, b2)] += ww * J[a] * J[b2];
                }
            }
        }
        if (t == 0) {
            for (int i = 0; i < DIM_X; ++i) o[RB_O_X1 + i] = xn[i];
            for (int r = 0; r < 3; ++r) for (int c = 0; c < HD; ++c) {
                o[RB_O_S1 + r*RB_H + c] = Sp[r*HD + c]; o[RB_O_S1 + 45 + r*RB_H + c] = Sv[r*HD + c]; o[RB_O_S1 + 90 + r*RB_H + c] = Sf[r*HD + c];
            }
        }
        for (int i = 0; i < DIM_X; ++i) x[i] = xn[i];
    }
    o[RB_O_J0] = J0; o[RB_O_JL] = Jl;
    for (int k = 0; k < HD; ++k) o[RB_O_G + k] = G[k];
    for (int a = 0; a < HD; ++a) for (int b2 = a; b2 < HD; ++b2) o[RB_O_H + rb_idx(a, b2)] = HH[rb_idx_t<HD>(a, b2)];
}

// proposal draw for the sequential sampler (section 13): x_i ~ alpha N(mu, Lg Lg') + (1 - alpha) prior, with the log
// prior and the log mixture density of the draw. par = [mu(V) | Lg(V x V row-major lower) | LH(H x H lower) | sc(C)].
__global__ void prop_draw_kernel(curandStatePhilox4_32_10_t* states, int N, int H, int C, int have_g, int headfix,
                                 double alpha, double logdet_g, double logdet0, const double* par, double* x_out, double* lp_out,
                                 int D, double* theta, double* dx0, double* logpq, int have_r, double rho, double logdet_r,
                                 double* rb_out, double rb0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int V = H + C;
    const double* mu = par; const double* Lg = par + 64; const double* LH = par + 64 + 64 * 64; const double* sc = LH + 15 * 15;
    const double* mur = sc + 64; const double* Lr = mur + 64;
    curandStatePhilox4_32_10_t st = states[i];
    double z[64], x[64];
    for (int k = 0; k < V; ++k) z[k] = curand_normal_double(&st);
    const double u = curand_uniform_double(&st);
    const double wg = have_r ? alpha * (1.0 - rho) : alpha, wr = have_r ? alpha * rho : 0.0;
    const int comp = !have_g ? 2 : (u < wg ? 0 : (u < wg + wr ? 1 : 2));   // 0: carried posterior, 1: its reflection, 2: prior
    states[i] = st;
    if (comp == 0) {
        for (int r = 0; r < V; ++r) { double v = mu[r]; for (int c = 0; c <= r; ++c) v += Lg[r * 64 + c] * z[c]; x[r] = v; }
    } else if (comp == 1) {
        for (int r = 0; r < V; ++r) { double v = mur[r]; for (int c = 0; c <= r; ++c) v += Lr[r * 64 + c] * z[c]; x[r] = v; }
    } else {
        for (int r = 0; r < H; ++r) { double v = 0.0; for (int c = 0; c <= r; ++c) v += LH[r * 15 + c] * z[c]; x[r] = v; }
        for (int j = 0; j < C; ++j) x[H + j] = sc[j] * z[H + j];
    }
    if (headfix) for (int k = 0; k < H; ++k) x[k] = 0.0;
    // log prior: forward substitution with LH on the head, diagonal on the inputs
    double q = 0.0, a[64];
    for (int r = 0; r < H; ++r) { double v = x[r]; for (int c = 0; c < r; ++c) v -= LH[r * 15 + c] * a[c]; a[r] = v / LH[r * 15 + r]; q += a[r] * a[r]; }
    for (int j = 0; j < C; ++j) { const double v = x[H + j] / sc[j]; q += v * v; }
    const double L2PI = 1.8378770664093453;
    const double lp0 = -0.5 * (q + logdet0 + V * L2PI);
    double lq = lp0;
    if (have_g) {
        double qg = 0.0;
        for (int r = 0; r < V; ++r) { double v = x[r] - mu[r]; for (int c = 0; c < r; ++c) v -= Lg[r * 64 + c] * a[c]; a[r] = v / Lg[r * 64 + r]; qg += a[r] * a[r]; }
        const double lg = -0.5 * (qg + logdet_g + V * L2PI);
        double la = log(wg) + lg, lb = log(1.0 - alpha) + lp0;
        double m = la > lb ? la : lb;
        double ssum = exp(la - m) + exp(lb - m);
        if (have_r) {
            double qr = 0.0;
            for (int r = 0; r < V; ++r) { double v = x[r] - mur[r]; for (int c = 0; c < r; ++c) v -= Lr[r * 64 + c] * a[c]; a[r] = v / Lr[r * 64 + r]; qr += a[r] * a[r]; }
            const double lr = log(wr) - 0.5 * (qr + logdet_r + V * L2PI);
            if (lr > m) { ssum = ssum * exp(m - lr) + 1.0; m = lr; } else ssum += exp(lr - m);
        }
        lq = m + log(ssum);
    }
    for (int k = 0; k < V; ++k) x_out[(size_t)i * V + k] = x[k];
    lp_out[2 * i] = lp0; lp_out[2 * i + 1] = lq;
    // rollout inputs: head offset (9), constant input offset = bias offset (6, zero when H == 9), basis coefficients (C)
    for (int k = 0; k < 9; ++k) dx0[9 * i + k] = x[k];
    for (int k = 0; k < 6; ++k) theta[(size_t)i * D + k] = (H == 15) ? x[9 + k] : 0.0;
    if (rb_out) rb_out[i] = rb0 + ((H == 10) ? x[9] : 0.0);
    for (int j = 0; j < C; ++j) theta[(size_t)i * D + 6 + j] = x[H + j];
    for (int k = 6 + C; k < D; ++k) theta[(size_t)i * D + k] = 0.0;
    logpq[i] = lp0 - lq;
}

__global__ void step_dyn_kernel(const double* x, const double* u,
                                double dt, const double* g, double* x_next) {
    step_dynamics(x, u, dt, g, x_next);
}

void mppi_set_deltas(double uwb_delta, double ctrl_delta) {
    cudaMemcpyToSymbol(d_uwb_delta_,  &uwb_delta,  sizeof(double));
    cudaMemcpyToSymbol(d_ctrl_delta_, &ctrl_delta, sizeof(double));
}

void mppi_set_uwb_model(int model, const double* mix5) {
    cudaMemcpyToSymbol(d_uwb_model_, &model, sizeof(int));
    if (mix5) cudaMemcpyToSymbol(d_mix_, mix5, 6 * sizeof(double));
}

void mppi_set_floor_penalty(double weight) {
    cudaMemcpyToSymbol(d_floor_penalty_, &weight, sizeof(double));
}
void mppi_set_floor_hard(int hard) {
    cudaMemcpyToSymbol(d_floor_hard_, &hard, sizeof(int));
}

void mppi_set_proposal_sigma(const double* h_sigma_prop) {
    cudaMemcpyToSymbol(d_sigma_prop_, h_sigma_prop, DIM_U * sizeof(double));
}

void mppi_set_ctrl_mode(int mode, double sia, double sig) {
    double s2[2] = {sia, sig};
    cudaMemcpyToSymbol(d_ctrl_mode_, &mode, sizeof(int));
    cudaMemcpyToSymbol(d_sigma_imu_, s2, 2 * sizeof(double));
}

MPPISolver::MPPISolver(int N, int T, double dt, double gamma,
                       const double* h_anchor,
                       const double* h_gravity,
                       const double* h_sigma,
                       const double* tag_offsets,
                       int num_anchors,
                       int num_tags,
                       unsigned long seed)
    : N_(N), T_(T), dt_(dt), gamma_(gamma),
      num_anchors_(num_anchors), num_tags_(num_tags) {

    cudaMemcpyToSymbol(d_sigma_,       h_sigma,     DIM_U * sizeof(double));
    cudaMemcpyToSymbol(d_sigma_prop_,  h_sigma,     DIM_U * sizeof(double));   // default: proposal = cost sigma
    cudaMemcpyToSymbol(d_tag_offset_,  tag_offsets, 3 * num_tags * sizeof(double));
    cudaMemcpyToSymbol(d_num_anchors_, &num_anchors, sizeof(int));
    cudaMemcpyToSymbol(d_num_tags_,    &num_tags,    sizeof(int));
    { const int off = 0; cudaMemcpyToSymbol(d_tdoa_, &off, sizeof(int)); }

    cudaMalloc(&d_anchor_,  num_anchors * 3 * sizeof(double));
    cudaMemcpy( d_anchor_,  h_anchor,   num_anchors * 3 * sizeof(double),
                cudaMemcpyHostToDevice);

    cudaMalloc(&d_gravity_, 3 * sizeof(double));
    cudaMemcpy( d_gravity_, h_gravity,  3 * sizeof(double),
                cudaMemcpyHostToDevice);

    cudaStreamCreate(&h2dS_);
    cudaStreamCreate(&d2hS_);
    cudaStreamCreate(&rngS_);
    cudaStreamCreate(&rollS_);
    cudaEventCreate(&rngDone_);

    cudaMalloc(&d_states_, CHUNK * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<(CHUNK + BLOCK_SIZE - 1) / BLOCK_SIZE,
               BLOCK_SIZE, 0, rngS_>>>(d_states_, seed, CHUNK);

    h_cost_.resize(N_);
    h_Ui_.resize(N_ * T_ * DIM_U);
    h_wi_.resize(N_);

    int num_rng  = num_tags_ * num_anchors_;
    len_U0_     = T_ * DIM_U;
    len_ranges_ = T_ * num_rng;
    len_accgyr_ = T_ * 6;

    cudaMalloc(&d_x0_,     DIM_X       * sizeof(double));
    cudaMalloc(&d_U0_,     len_U0_     * sizeof(double));
    cudaMalloc(&d_ranges_, len_ranges_ * sizeof(double));
    cudaMalloc(&d_stds_,   len_ranges_ * sizeof(double));
    cudaMalloc(&d_accgyr_, len_accgyr_ * sizeof(double));
    cudaMalloc(&d_noise_,  N_ * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_Ui_,     N_ * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_cost_,     N_ * sizeof(double));
    cudaMalloc(&d_cost_uwb_, N_ * sizeof(double));
    cudaMalloc(&d_cost_last_, N_ * sizeof(double));
    h_cost_last_.resize(N_);
    cudaMalloc(&d_cost_acc_, N_ * sizeof(double));
    cudaMalloc(&d_cost_gyr_, N_ * sizeof(double));
    cudaMalloc(&d_wi_,       N_ * sizeof(double));
    cudaMalloc(&d_Uopt_,   T_ * DIM_U  * sizeof(double));
    cudaMalloc(&d_xn_,     DIM_X       * sizeof(double));

}

MPPISolver::~MPPISolver() {
    cudaFree(d_states_);
    cudaFree(d_anchor_);
    cudaFree(d_gravity_);
    cudaStreamDestroy(h2dS_);
    cudaStreamDestroy(d2hS_);
    cudaStreamDestroy(rngS_);
    cudaStreamDestroy(rollS_);
    cudaEventDestroy(rngDone_);

    cudaFree(d_x0_);    cudaFree(d_U0_);
    cudaFree(d_ranges_); cudaFree(d_stds_); cudaFree(d_accgyr_);
    cudaFree(d_noise_);  cudaFree(d_Ui_);
    cudaFree(d_cost_);
    cudaFree(d_cost_uwb_); cudaFree(d_cost_acc_); cudaFree(d_cost_gyr_); cudaFree(d_cost_last_);
    cudaFree(d_wi_);
    cudaFree(d_Uopt_);   cudaFree(d_xn_);
    if (d_phi_) { cudaFree(d_phi_); cudaFree(d_theta_); cudaFree(d_logpq_); cudaFree(d_mu_); cudaFree(d_L_); cudaFree(d_prior_std_); }
    if (d_dv_) cudaFree(d_dv_);
    if (d_cost_newest_) cudaFree(d_cost_newest_);
    if (d_states_bak_) cudaFree(d_states_bak_);
    if (d_dx0_) { cudaFree(d_dx0_); cudaFree(d_x1_); }
}

void MPPISolver::setDt(double dt) { dt_ = dt; }
void MPPISolver::setTdoa(const double* anchorB, int n_slots) {
    const int on = 1;
    cudaMemcpyToSymbol(d_anchorB_, anchorB, 3 * n_slots * sizeof(double));
    cudaMemcpyToSymbol(d_tdoa_, &on, sizeof(int));
}

void MPPISolver::solve(double* h_Uopt, double* h_xn,
                       const double* h_x0, const double* h_U0,
                       const double* h_ranges, const double* h_stds,
                       const double* h_accgyr) {
    int num_rng = num_tags_ * num_anchors_;

    cudaMemcpyAsync(d_x0_,     h_x0,     DIM_X        * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_U0_,     h_U0,     len_U0_      * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_ranges_, h_ranges, len_ranges_  * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_stds_,   h_stds,   len_ranges_  * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_accgyr_, h_accgyr, len_accgyr_  * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaStreamSynchronize(h2dS_);

    int batch_start = 0;
    while (batch_start < N_) {
        int batch       = std::min(CHUNK, N_ - batch_start);
        int batch_elems = batch * T_ * DIM_U;

        noise_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       BLOCK_SIZE, 0, rngS_>>>(
            d_states_,
            d_noise_ + batch_start * T_ * DIM_U,
            batch, T_);
        cudaEventRecord(rngDone_, rngS_);
        cudaStreamWaitEvent(rollS_, rngDone_, 0);

        rollout_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         BLOCK_SIZE, 0, rollS_>>>(
            batch, T_, num_rng, dt_, d_gravity_,
            d_anchor_, d_ranges_, d_stds_, d_accgyr_,
            d_U0_,
            d_noise_ + batch_start * T_ * DIM_U,
            d_x0_, nullptr, gamma_,
            w_uwb_, w_acc_, w_gyr_,
            d_Ui_    + batch_start * T_ * DIM_U,
            d_cost_  + batch_start,
            d_cost_uwb_ + batch_start,
            d_cost_acc_ + batch_start,
            d_cost_gyr_ + batch_start,
            d_cost_last_ + batch_start, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

        batch_start += batch;
    }

    std::vector<double> h_cost_uwb(N_), h_cost_acc(N_), h_cost_gyr(N_);
    cudaMemcpyAsync(h_cost_.data(),    d_cost_,     N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_uwb.data(), d_cost_uwb_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_acc.data(), d_cost_acc_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_gyr.data(), d_cost_gyr_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_Ui_.data(),      d_Ui_,       N_ * T_ * DIM_U * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_last_.data(), d_cost_last_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaStreamSynchronize(rollS_);

    double min_cost = std::numeric_limits<double>::max();
    for (int i = 0; i < N_; ++i)
        if (std::isfinite(h_cost_[i]) && h_cost_[i] < min_cost)
            min_cost = h_cost_[i];
    if (!std::isfinite(min_cost)) min_cost = 0.0;

    double sum_w = 0.0;
    for (int i = 0; i < N_; ++i) {
        h_wi_[i] = std::isfinite(h_cost_[i])
                   ? std::exp(-gamma_ * (h_cost_[i] - min_cost)) : 0.0;
        sum_w += h_wi_[i];
    }

    if (sum_w < 1e-300) {
        std::fill(h_wi_.begin(), h_wi_.end(), 1.0 / N_);
    } else {
        for (double& w : h_wi_) w /= sum_w;
    }

    {
        double s2 = 0.0;
        for (double w : h_wi_) s2 += w * w;
        last_ess_ = s2 > 0.0 ? 1.0 / s2 : 0.0;
    }

    last_cost_uwb_ = last_cost_acc_ = last_cost_gyr_ = 0.0;
    for (int i = 0; i < N_; ++i) {
        last_cost_uwb_ += h_wi_[i] * h_cost_uwb[i];
        last_cost_acc_ += h_wi_[i] * h_cost_acc[i];
        last_cost_gyr_ += h_wi_[i] * h_cost_gyr[i];
    }

    std::fill(h_Uopt, h_Uopt + T_ * DIM_U, 0.0);
    for (int i = 0; i < N_; ++i)
        for (int t = 0; t < T_; ++t)
            for (int k = 0; k < DIM_U; ++k)
                h_Uopt[t * DIM_U + k] +=
                    h_wi_[i] * h_Ui_[(i * T_ + t) * DIM_U + k];

    cudaMemcpy(d_Uopt_, h_Uopt, T_ * DIM_U * sizeof(double),
               cudaMemcpyHostToDevice);
    if (h_xn != nullptr) {
        step_dyn_kernel<<<1, 1>>>(d_x0_, d_Uopt_, dt_, d_gravity_, d_xn_);
        cudaMemcpy(h_xn, d_xn_, DIM_X * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }
}


// ------------------------------------------------------------------ basis mode ----
#define BASIS_DMAX 64
// theta_i = mu + L z (use_L) or prior_std * z (prior proposal). logpq = log p(theta) - log q(theta) up to a constant.
__global__ void basis_kernel(curandStatePhilox4_32_10_t* states,
                             const double* mu, const double* L, const double* prior_std,
                             const double* phi, int use_L, int batch, int T, int K, int dv_on,
                             double prior_w,
                             double* theta, double* logpq, double* noise, double* dv_out) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    const int D = 6 + 6 * K + (dv_on ? 3 : 0);
    double z[BASIS_DMAX], th[BASIS_DMAX];
    if (use_L == 2) {                                   // coefficients drawn on the host (solveSeq); expand only
        for (int d = 0; d < D; ++d) th[d] = theta[sample * D + d];
        for (int t = 0; t < T; ++t) {
            const int base = (sample * T + t) * DIM_U;
            for (int ch = 0; ch < DIM_U; ++ch) {
                double n = th[ch];
                for (int k = 0; k < K; ++k) n += th[6 + 6 * k + ch] * phi[k * T + t];
                noise[base + ch] = n;
            }
        }
        return;
    }
    curandStatePhilox4_32_10_t local = states[sample];
    double z2 = 0.0;
    for (int d = 0; d < D; ++d) { z[d] = curand_normal_double(&local); z2 += z[d] * z[d]; }
    states[sample] = local;
    double lp = 0.0, lp_nodc = 0.0;
    for (int d = 0; d < D; ++d) {
        double v = 0.0;
        if (use_L) {
            v = mu[d];
            for (int e = 0; e <= d; ++e) v += L[d * D + e] * z[e];
        } else {
            v = prior_std[d] * z[d];
        }
        th[d] = v;
        const double r = v / prior_std[d];
        lp += r * r;
        if (!(d >= 6 && d < 9)) lp_nodc += r * r;      // everything except the accelerometer DC mode (k = 0, ch 0..2)
    }
    logpq[sample] = use_L ? (0.5 * z2 - 0.5 * lp) : 0.0;
    // optional prior term in the weight: prior_w = 1 charges all coefficients (MAP-like),
    // prior_w = 2 charges all but the accelerometer DC mode (persistent bias stays free, time-varying
    // compensation and gyro offsets are charged at the IMU spec)
    if (prior_w > 1.5)      logpq[sample] += -0.5 * lp_nodc;
    else if (prior_w > 0.5) logpq[sample] += -0.5 * lp;
    for (int d = 0; d < D; ++d) theta[sample * D + d] = th[d];
    if (dv_on) for (int i = 0; i < 3; ++i) dv_out[sample * 3 + i] = th[6 + 6 * K + i];
    for (int t = 0; t < T; ++t) {
        const int base = (sample * T + t) * DIM_U;
        for (int ch = 0; ch < DIM_U; ++ch) {
            double n = th[ch];
            for (int k = 0; k < K; ++k) n += th[6 + 6 * k + ch] * phi[k * T + t];
            noise[base + ch] = n;
        }
    }
}

void MPPISolver::setBasis(int K, int iters, double ess_min) {
    K_ = K; D_ = 6 + 6 * K + (dv_on_ ? 3 : 0); iters_ = iters; ess_min_ = ess_min;
    if (D_ > BASIS_DMAX) { std::fprintf(stderr, "[mppi] basis dim %d > %d\n", D_, BASIS_DMAX); std::abort(); }
    const double Ts = T_ * dt_;
    std::vector<double> phi(K_ * T_);
    for (int k = 0; k < K_; ++k)
        for (int t = 0; t < T_; ++t)
            phi[k * T_ + t] = (k == 0) ? 1.0 / std::sqrt(Ts)
                                       : std::sqrt(2.0 / Ts) * std::cos(M_PI * k * (t + 0.5) / T_);
    if (d_phi_) { cudaFree(d_phi_); cudaFree(d_theta_); cudaFree(d_logpq_); cudaFree(d_mu_); cudaFree(d_L_); cudaFree(d_prior_std_); }
    cudaMalloc(&d_phi_,       K_ * T_ * sizeof(double));
    cudaMalloc(&d_theta_,     N_ * D_ * sizeof(double));
    cudaMalloc(&d_logpq_,     N_ * sizeof(double));
    cudaMalloc(&d_mu_,        D_ * sizeof(double));
    cudaMalloc(&d_L_,         D_ * D_ * sizeof(double));
    cudaMalloc(&d_prior_std_, D_ * sizeof(double));
    if (d_dv_) cudaFree(d_dv_);
    cudaMalloc(&d_dv_, N_ * 3 * sizeof(double));
    if (d_cost_newest_) cudaFree(d_cost_newest_);
    cudaMalloc(&d_cost_newest_, N_ * sizeof(double));
    if (d_dx0_) { cudaFree(d_dx0_); cudaFree(d_x1_); }
    cudaMalloc(&d_dx0_, N_ * 9 * sizeof(double));
    cudaMalloc(&d_x1_,  N_ * (DIM_X + 1) * sizeof(double));
    if (!d_xT_) cudaMalloc(&d_xT_, (size_t)N_ * DIM_X * sizeof(double)); h_xT_.assign((size_t)N_ * DIM_X, 0.0);
    if (!d_xall_) cudaMalloc(&d_xall_, (size_t)N_ * T_ * 3 * sizeof(double));
    h_dx0_.assign(N_ * 9, 0.0); h_dz_.assign(N_ * 9, 0.0); h_x1_.assign(N_ * (DIM_X + 1), 0.0);
    h_cost_newest_.assign(N_, 0.0);
    cudaMemcpy(d_phi_, phi.data(), K_ * T_ * sizeof(double), cudaMemcpyHostToDevice);
    h_phi_ = phi;
    h_theta_.resize(N_ * D_); h_logpq_.resize(N_);
}

void MPPISolver::saveRng() {
    if (!d_states_bak_) cudaMalloc(&d_states_bak_, CHUNK * sizeof(curandStatePhilox4_32_10_t));
    cudaMemcpy(d_states_bak_, d_states_, CHUNK * sizeof(curandStatePhilox4_32_10_t), cudaMemcpyDeviceToDevice);
}

void MPPISolver::restoreRng() {
    if (d_states_bak_) cudaMemcpy(d_states_, d_states_bak_, CHUNK * sizeof(curandStatePhilox4_32_10_t), cudaMemcpyDeviceToDevice);
}

void MPPISolver::setHeadCov(const double* P9) {
    Eigen::Matrix<double, 9, 9> P; for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) P(r, c) = 0.5 * (P9[9*r + c] + P9[9*c + r]);
    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(P);
    Eigen::Matrix<double, 9, 9> L;
    if (llt.info() == Eigen::Success) L = llt.matrixL();
    else { L.setZero(); for (int k = 0; k < 9; ++k) L(k, k) = std::sqrt(std::max(P(k, k), 0.0)); }
    for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) head_L_[9*r + c] = L(r, c);
}

void MPPISolver::setStepDts(const double* h_dts) {
    if (!d_dts_) cudaMalloc(&d_dts_, T_ * sizeof(double));
    cudaMemcpy(d_dts_, h_dts, T_ * sizeof(double), cudaMemcpyHostToDevice);
    true_dt_ = 1; dt_ = h_dts[0];                        // dt_: the head -> first-epoch step (one-step Q of the arrival cost)
}

void MPPISolver::setPhi(const double* h_phi) {
    h_phi_prev_ = h_phi_;
    h_phi_.assign(h_phi, h_phi + K_ * T_);
    cudaMemcpy(d_phi_, h_phi, K_ * T_ * sizeof(double), cudaMemcpyHostToDevice);
}

void MPPISolver::setBasisPrior(const double* h_prior_std) {
    h_prior_std_.assign(h_prior_std, h_prior_std + D_);
    cudaMemcpy(d_prior_std_, h_prior_std, D_ * sizeof(double), cudaMemcpyHostToDevice);
}

// one sampling + rollout pass over all N samples; fills h_cost_, h_Ui_, h_theta_, h_logpq_
void MPPISolver::basisPass(int use_L, double* h_lw) {
    const int Np = (pass_n_ > 0 && pass_n_ < N_) ? pass_n_ : N_;
    int batch_start = 0;
    while (batch_start < Np) {
        int batch = std::min(CHUNK, Np - batch_start);
        basis_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(
            d_states_, d_mu_, d_L_, d_prior_std_, d_phi_, use_L, batch, T_, K_, dv_on_,
            static_cast<double>(prior_in_weight_),
            d_theta_ + batch_start * D_, d_logpq_ + batch_start,
            d_noise_ + batch_start * T_ * DIM_U, d_dv_ + batch_start * 3);
        cudaEventRecord(rngDone_, rngS_);
        cudaStreamWaitEvent(rollS_, rngDone_, 0);
        auto rk = rollout_fp32_ ? rollout_kernel_f : rollout_kernel;
        rk<<<(batch + RB_BLOCK - 1) / RB_BLOCK, RB_BLOCK, 0, rollS_>>>(
            batch, T_, num_tags_ * num_anchors_, dt_, d_gravity_,
            d_anchor_, d_ranges_, d_stds_, d_accgyr_, d_U0_,
            d_noise_ + batch_start * T_ * DIM_U, d_x0_, dv_on_ ? d_dv_ + batch_start * 3 : nullptr, gamma_,
            w_uwb_, w_acc_, w_gyr_,
            d_Ui_ + batch_start * T_ * DIM_U, d_cost_ + batch_start,
            d_cost_uwb_ + batch_start, d_cost_acc_ + batch_start, d_cost_gyr_ + batch_start,
            d_cost_last_ + batch_start, d_cost_newest_ + batch_start,
            head_on_ ? d_dx0_ + batch_start * 9 : nullptr, head_on_ ? d_x1_ + batch_start * (DIM_X + 1) : nullptr, true_dt_ ? d_dts_ : nullptr,
            (rbias_on_ && d_rbias_) ? d_rbias_ + batch_start : nullptr,
            head_on_ ? d_xT_ + (size_t)batch_start * DIM_X : nullptr,
            (flush_ && d_xall_) ? d_xall_ + (size_t)batch_start * T_ * 3 : nullptr);
        batch_start += batch;
    }
    cudaMemcpyAsync(h_cost_newest_.data(), d_cost_newest_, Np * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    if (head_on_) cudaMemcpyAsync(h_x1_.data(), d_x1_, Np * (DIM_X + 1) * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    if (head_on_ && d_xT_) cudaMemcpyAsync(h_xT_.data(), d_xT_, (size_t)Np * DIM_X * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    if (flush_ && d_xall_) { if (h_xall_.size() < (size_t)N_ * T_ * 3) h_xall_.assign((size_t)N_ * T_ * 3, 0.0); cudaMemcpyAsync(h_xall_.data(), d_xall_, (size_t)Np * T_ * 3 * sizeof(double), cudaMemcpyDeviceToHost, rollS_); }
    cudaMemcpyAsync(h_cost_.data(),  d_cost_,  Np * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    if (!skip_ui_) cudaMemcpyAsync(h_Ui_.data(), d_Ui_, Np * T_ * DIM_U * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_theta_.data(), d_theta_, Np * D_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_logpq_.data(), d_logpq_, Np * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_last_.data(), d_cost_last_, Np * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaStreamSynchronize(rollS_);
    cuda_check("basisPass");
    for (int i = 0; i < Np; ++i)
        h_lw[i] = std::isfinite(h_cost_[i]) ? -gamma_ * h_cost_[i] + h_logpq_[i]
                                            : -std::numeric_limits<double>::infinity();
}

static double normalise_weights(const std::vector<double>& lw, std::vector<double>& w) {
    const int N = static_cast<int>(lw.size());
    double mx = -std::numeric_limits<double>::infinity();
    for (double v : lw) if (v > mx) mx = v;
    // vectorised exp over contiguous chunks, one per OpenMP thread; partial sums added in chunk order (deterministic
    // for a given thread count; host time)
#ifdef _OPENMP
    const int nch = (N >= 4096) ? std::max(1, omp_get_max_threads()) : 1;
#else
    const int nch = 1;
#endif
    std::vector<double> ps(nch, 0.0), ps2(nch, 0.0);
#pragma omp parallel for schedule(static) if (nch > 1)
    for (int c = 0; c < nch; ++c) {
        const int a = (int)((long)N * c / nch), b = (int)((long)N * (c + 1) / nch);
        Eigen::Map<const Eigen::ArrayXd> l(lw.data() + a, b - a);
        Eigen::Map<Eigen::ArrayXd> wa(w.data() + a, b - a);
        wa = l.isFinite().select((l - mx).exp(), 0.0);
        ps[c] = wa.sum();
    }
    double sum = 0.0; for (int c = 0; c < nch; ++c) sum += ps[c];
    if (sum < 1e-300) { std::fill(w.begin(), w.end(), 1.0 / N); return static_cast<double>(N); }
#pragma omp parallel for schedule(static) if (nch > 1)
    for (int c = 0; c < nch; ++c) {
        const int a = (int)((long)N * c / nch), b = (int)((long)N * (c + 1) / nch);
        Eigen::Map<Eigen::ArrayXd> wa(w.data() + a, b - a);
        wa /= sum; ps2[c] = wa.square().sum();
    }
    double s2 = 0.0; for (int c = 0; c < nch; ++c) s2 += ps2[c];
    return 1.0 / s2;
}

void MPPISolver::solveBasis(double* h_Uopt, double* h_xn,
                            const double* h_x0, const double* h_U0,
                            const double* h_ranges, const double* h_stds, const double* h_accgyr,
                            double* h_theta_mean, double* h_theta_var) {
    cudaMemcpyAsync(d_x0_,     h_x0,     DIM_X       * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_U0_,     h_U0,     len_U0_     * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_ranges_, h_ranges, len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_stds_,   h_stds,   len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_accgyr_, h_accgyr, len_accgyr_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaStreamSynchronize(h2dS_);

    std::vector<double> lw(N_);
    if (head_on_) {
        // dx0_i = L z_i, z_i ~ N(0, I9): head-state samples from the carried covariance P = L L'
        for (int i = 0; i < N_; ++i) {
            double z[9]; for (int k = 0; k < 9; ++k) { z[k] = head_nrm_(head_rng_); h_dz_[9*i + k] = z[k]; }
            for (int r = 0; r < 9; ++r) { double v = 0.0; for (int k = 0; k <= r; ++k) v += head_L_[9*r + k] * z[k]; h_dx0_[9*i + r] = v; }
        }
        cudaMemcpy(d_dx0_, h_dx0_.data(), N_ * 9 * sizeof(double), cudaMemcpyHostToDevice);
    }
    basisPass(0, lw.data());
    {   // predictive log-likelihood of the newest epoch from the first (prior-proposal, exact IS) pass
        double m1 = -std::numeric_limits<double>::infinity(), m2 = m1;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { m1 = std::max(m1, lw[i]); m2 = std::max(m2, lw[i] + gamma_ * h_cost_newest_[i]); }
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { s1 += std::exp(lw[i] - m1); s2 += std::exp(lw[i] + gamma_ * h_cost_newest_[i] - m2); }
        last_logpred1_ = (m1 + std::log(s1)) - (m2 + std::log(s2));
    }
    double ess = normalise_weights(lw, h_wi_);
    last_ess1_ = ess; last_ess2_ = 0.0; last_iters_ = 1;

    if (iters_ >= 2 && ess < ess_min_ * N_) {
        // weighted mean / covariance of theta -> Gaussian proposal
        std::vector<double> mu(D_, 0.0), S(D_ * D_, 0.0), Lm(D_ * D_, 0.0);
        for (int i = 0; i < N_; ++i)
            for (int d = 0; d < D_; ++d) mu[d] += h_wi_[i] * h_theta_[i * D_ + d];
        for (int i = 0; i < N_; ++i) {
            if (h_wi_[i] == 0.0) continue;
            for (int d = 0; d < D_; ++d) {
                const double a = h_theta_[i * D_ + d] - mu[d];
                for (int e = 0; e <= d; ++e) S[d * D_ + e] += h_wi_[i] * a * (h_theta_[i * D_ + e] - mu[e]);
            }
        }
        // jitter: at least 1e-4 of the prior variance on the diagonal, so the proposal never collapses
        std::vector<double> ps(D_);
        cudaMemcpy(ps.data(), d_prior_std_, D_ * sizeof(double), cudaMemcpyDeviceToHost);
        // defensive floor: 10 % of the prior variance on the diagonal, so a degenerate first pass
        // cannot produce a proposal narrower than the prior's tails
        for (int d = 0; d < D_; ++d) S[d * D_ + d] += 0.1 * ps[d] * ps[d];
        // Cholesky (lower)
        bool ok = true;
        for (int d = 0; d < D_ && ok; ++d) {
            for (int e = 0; e <= d; ++e) {
                double acc = S[d * D_ + e];
                for (int k = 0; k < e; ++k) acc -= Lm[d * D_ + k] * Lm[e * D_ + k];
                if (d == e) { if (acc <= 0.0) { ok = false; break; } Lm[d * D_ + d] = std::sqrt(acc); }
                else          Lm[d * D_ + e] = acc / Lm[e * D_ + e];
            }
        }
        if (ok) {
            cudaMemcpy(d_mu_, mu.data(), D_ * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_L_,  Lm.data(), D_ * D_ * sizeof(double), cudaMemcpyHostToDevice);
            basisPass(1, lw.data());
            ess = normalise_weights(lw, h_wi_);
            last_ess2_ = ess; last_iters_ = 2;
        }
    }
    last_ess_ = ess;
    {
        // log p(z_newest | z_older in window, head) = LSE_i(lw_i) - LSE_i(lw_i + gamma * l_i),
        // where lw_i already contains -gamma * l_i (the newest epoch's robust UWB cost of sample i)
        double m1 = -std::numeric_limits<double>::infinity(), m2 = m1;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { m1 = std::max(m1, lw[i]); m2 = std::max(m2, lw[i] + gamma_ * h_cost_newest_[i]); }
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { s1 += std::exp(lw[i] - m1); s2 += std::exp(lw[i] + gamma_ * h_cost_newest_[i] - m2); }
        last_logpred_ = (m1 + std::log(s1)) - (m2 + std::log(s2));
    }

    last_cost_uwb_ = 0.0;
    for (int i = 0; i < N_; ++i) last_cost_uwb_ += h_wi_[i] * h_cost_[i];

    std::fill(h_Uopt, h_Uopt + T_ * DIM_U, 0.0);
    for (int i = 0; i < N_; ++i)
        for (int t = 0; t < T_; ++t)
            for (int k = 0; k < DIM_U; ++k)
                h_Uopt[t * DIM_U + k] += h_wi_[i] * h_Ui_[(i * T_ + t) * DIM_U + k];
    for (int d = 0; d < D_; ++d) { h_theta_mean[d] = 0.0; h_theta_var[d] = 0.0; }
    for (int i = 0; i < N_; ++i)
        for (int d = 0; d < D_; ++d) h_theta_mean[d] += h_wi_[i] * h_theta_[i * D_ + d];
    for (int i = 0; i < N_; ++i)
        for (int d = 0; d < D_; ++d) { const double a = h_theta_[i * D_ + d] - h_theta_mean[d]; h_theta_var[d] += h_wi_[i] * a * a; }

    if (head_on_ && h_xn != nullptr) {
        // head <- weighted mean of the samples' states one step later (the window's estimate of the new head)
        constexpr int XS = DIM_X + 1;
        int ib = 0; for (int i = 1; i < N_; ++i) if (h_wi_[i] > h_wi_[ib]) ib = i;
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rref(&h_x1_[XS * ib + 6]);
        typedef Eigen::Matrix<double, 9, 1> V9; typedef Eigen::Matrix<double, 9, 9> M9;
        std::vector<V9> y(N_);
        V9 m = V9::Zero(), dm = V9::Zero();
        for (int i = 0; i < N_; ++i) {
            const double* xi = &h_x1_[XS * i];
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Ri(xi + 6);
            const Eigen::AngleAxisd aa(Eigen::Matrix3d(Rref.transpose() * Ri));
            const Eigen::Vector3d r = aa.angle() * aa.axis();
            y[i] << xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], r.x(), r.y(), r.z();
            m += h_wi_[i] * y[i];
            for (int k = 0; k < 9; ++k) dm[k] += h_wi_[i] * h_dx0_[9*i + k];
        }
        for (int k = 0; k < 9; ++k) last_dmean_[k] = dm[k];
        for (int k = 0; k < 6; ++k) h_xn[k] = m[k];
        const Eigen::Matrix3d Rm = Rref * Eigen::AngleAxisd(m.tail<3>().norm(), m.tail<3>().norm() > 1e-12 ? Eigen::Vector3d(m.tail<3>().normalized()) : Eigen::Vector3d::UnitX()).toRotationMatrix();
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) h_xn[6 + 3*r + c] = Rm(r, c);

        if (head_on_ == 1) {
            // P <- weighted covariance of the next states
            M9 C = M9::Zero();
            for (int i = 0; i < N_; ++i) { const V9 d = y[i] - m; C += h_wi_[i] * d * d.transpose(); }
            for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) last_P1_[9*r + c] = C(r, c);
            return;
        }

        // head_on_ == 2: sample-based Laplace arrival cost, computed from ALL samples (unweighted).
        // (1) prediction to the first window epoch: y1_i = c + B z_i + e_i with dx0_i = L z_i, so F P F' = B B';
        //     P1- = B B' + Q with Q the white IMU noise of one step
        V9 yb = V9::Zero(), zb = V9::Zero();
        for (int i = 0; i < N_; ++i) { yb += y[i]; for (int k = 0; k < 9; ++k) zb[k] += h_dz_[9*i + k]; }
        yb /= N_; zb /= N_;
        M9 Syz = M9::Zero(), Szz = M9::Zero();
        for (int i = 0; i < N_; ++i) {
            V9 z; for (int k = 0; k < 9; ++k) z[k] = h_dz_[9*i + k] - zb[k];
            Syz += (y[i] - yb) * z.transpose(); Szz += z * z.transpose();
        }
        const M9 B = Szz.transpose().ldlt().solve(Syz.transpose()).transpose();      // B = Syz Szz^-1
        const double h = dt_;
        Eigen::Matrix3d Sa = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) Sa(k, k) = head_q_[k] * head_q_[k];
        Sa = Rref * Sa * Rref.transpose();                                           // body acc noise in world
        M9 Q = M9::Zero();
        Q.block<3,3>(0,0) = Sa * h * h * h / 3.0; Q.block<3,3>(0,3) = Sa * h * h / 2.0;
        Q.block<3,3>(3,0) = Sa * h * h / 2.0;     Q.block<3,3>(3,3) = Sa * h;
        for (int k = 0; k < 3; ++k) Q(6 + k, 6 + k) = head_q_[3 + k] * head_q_[3 + k] * h;
        const M9 Pm = B * B.transpose() + Q;
        // (2) update with the epoch that leaves the window: its cost l0 is an exact function of (p1, R1), so fit
        //     l0 = c + g'u + u'Hu/2 in the whitened coordinates u = L6^-1 (y6 - mean) of (p1, dtheta1) and map H back
        const int id6[6] = {0, 1, 2, 6, 7, 8};
        Eigen::Matrix<double, 6, 1> yb6; for (int a = 0; a < 6; ++a) yb6[a] = yb[id6[a]];
        Eigen::Matrix<double, 6, 6> S6 = Eigen::Matrix<double, 6, 6>::Zero();
        double lmin = 1e300, lmax = -1e300;
        for (int i = 0; i < N_; ++i) {
            Eigen::Matrix<double, 6, 1> d; for (int a = 0; a < 6; ++a) d[a] = y[i][id6[a]] - yb6[a];
            S6 += d * d.transpose();
            const double l = h_x1_[XS * i + DIM_X];
            if (std::isfinite(l)) { lmin = std::min(lmin, l); lmax = std::max(lmax, l); }
        }
        S6 /= (N_ - 1);
        M9 A9 = M9::Zero();
        last_fit_r2_ = -1.0;
        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt6(S6);
        if (lmax - lmin > 1e-12 && llt6.info() == Eigen::Success) {
            const Eigen::Matrix<double, 6, 6> L6 = llt6.matrixL();
            constexpr int NF = 1 + 6 + 21;
            Eigen::MatrixXd X(N_, NF); Eigen::VectorXd lv(N_);
            int n = 0;
            for (int i = 0; i < N_; ++i) {
                const double l = h_x1_[XS * i + DIM_X];
                if (!std::isfinite(l)) continue;
                Eigen::Matrix<double, 6, 1> d; for (int a = 0; a < 6; ++a) d[a] = y[i][id6[a]] - yb6[a];
                const Eigen::Matrix<double, 6, 1> u = L6.triangularView<Eigen::Lower>().solve(d);
                int c = 0; X(n, c++) = 1.0;
                for (int a = 0; a < 6; ++a) X(n, c++) = u[a];
                for (int a = 0; a < 6; ++a) for (int b = a; b < 6; ++b) X(n, c++) = u[a] * u[b];
                lv[n++] = l;
            }
            X.conservativeResize(n, NF); lv.conservativeResize(n);
            const Eigen::VectorXd cf = X.colPivHouseholderQr().solve(lv);
            const Eigen::VectorXd res = lv - X * cf;
            const double var_l = (lv.array() - lv.mean()).square().mean();
            last_fit_r2_ = var_l > 0.0 ? 1.0 - res.squaredNorm() / n / var_l : -1.0;
            Eigen::Matrix<double, 6, 6> Hu = Eigen::Matrix<double, 6, 6>::Zero();
            int c = 7;
            for (int a = 0; a < 6; ++a) for (int b = a; b < 6; ++b) { if (a == b) Hu(a, a) = 2.0 * cf[c]; else Hu(a, b) = Hu(b, a) = cf[c]; ++c; }
            const Eigen::Matrix<double, 6, 6> Li = L6.inverse();
            Eigen::Matrix<double, 6, 6> A6 = Li.transpose() * Hu * Li;
            A6 = 0.5 * (A6 + A6.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(A6);
            const Eigen::Matrix<double, 6, 1> ev = es.eigenvalues().cwiseMax(0.0);            // no information where curvature <= 0
            A6 = gamma_ * es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
            for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) A9(id6[a], id6[b]) = A6(a, b);
        }
        M9 P1 = (M9::Identity() + Pm * A9).partialPivLu().solve(Pm);                     // (Pm^-1 + A)^-1
        P1 = 0.5 * (P1 + P1.transpose());
        if (!P1.allFinite() || P1.ldlt().info() != Eigen::Success) P1 = Pm;
        for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) { last_P1_[9*r + c] = P1(r, c); last_A_[9*r + c] = A9(r, c); }
        return;
    }
    cudaMemcpy(d_Uopt_, h_Uopt, T_ * DIM_U * sizeof(double), cudaMemcpyHostToDevice);
    if (h_xn != nullptr) {
        if (dv_on_ && dv_gain_ != 0.0) {
            // carry: v0 += gain * E[delta v] (gain = 1/T, the fraction of the window's data that is new per solve)
            double x0c[DIM_X];
            for (int i = 0; i < DIM_X; ++i) x0c[i] = h_x0[i];
            for (int i = 0; i < 3; ++i) x0c[3 + i] += dv_gain_ * h_theta_mean[6 + 6 * K_ + i];
            cudaMemcpy(d_x0_, x0c, DIM_X * sizeof(double), cudaMemcpyHostToDevice);
        }
        step_dyn_kernel<<<1, 1>>>(d_x0_, d_Uopt_, dt_, d_gravity_, d_xn_);
        cudaMemcpy(h_xn, d_xn_, DIM_X * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

// ------------------------------------------------ sequential proposal ----
static double lse2(double a, double b) {
    const double m = std::max(a, b);
    if (!std::isfinite(m)) return m;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

void MPPISolver::solveSeq(double* h_xn, const double* h_x0, const double* h_U0,
                          const double* h_ranges, const double* h_stds, const double* h_accgyr) {
    typedef Eigen::MatrixXd MX; typedef Eigen::VectorXd VX;
    if (dv_on_ || !head_on_ || K_ <= 0) { std::fprintf(stderr, "[mppi] solveSeq needs basis mode, head sampling, no dv\n"); std::abort(); }
    cudaMemcpyAsync(d_x0_,     h_x0,     DIM_X       * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_U0_,     h_U0,     len_U0_     * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_ranges_, h_ranges, len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_stds_,   h_stds,   len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_accgyr_, h_accgyr, len_accgyr_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaStreamSynchronize(h2dS_);

    const int H = head_dim_, C = 6 * K_, V = H + C, D = D_;
    constexpr int XS = DIM_X + 1;
    const double LOG2PI = std::log(2.0 * M_PI);

    // ---- target prior: arrival N(0, P_H) on the head offset, N(0, diag(sd_c^2)) on the input coefficients
    MX PH(H, H);
    for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c) PH(r, c) = 0.5 * (headP_[H*r + c] + headP_[H*c + r]);
    MX LH = MX::Zero(H, H);
    { Eigen::LLT<MX> llt(PH); if (llt.info() == Eigen::Success) LH = llt.matrixL();
      else for (int k = 0; k < H; ++k) LH(k, k) = std::sqrt(std::max(PH(k, k), 1e-18)); }
    double logdet0 = 0.0;
    for (int k = 0; k < H; ++k) logdet0 += 2.0 * std::log(LH(k, k));
    VX sc(C);
    for (int j = 0; j < C; ++j) { sc[j] = h_prior_std_[6 + j]; logdet0 += 2.0 * std::log(sc[j]); }
    auto lprior = [&](const double* v) {
        const VX a = LH.triangularView<Eigen::Lower>().solve(Eigen::Map<const VX>(v, H));
        double q = a.squaredNorm();
        for (int j = 0; j < C; ++j) { const double r = v[H + j] / sc[j]; q += r * r; }
        return -0.5 * (q + logdet0 + V * LOG2PI);
    };
    struct Gauss { VX mu; MX L; double logdet; bool ok; };
    auto make_gauss = [&](const VX& mu, MX S) {
        Gauss g; g.mu = mu; g.ok = false; S = 0.5 * (S + S.transpose());
        for (int k = 0; k < V; ++k) S(k, k) += 1e-12 + 1e-9 * (k < H ? PH(k, k) : sc[k - H] * sc[k - H]);
        Eigen::LLT<MX> llt(S);
        if (llt.info() != Eigen::Success || !S.allFinite()) return g;
        g.L = llt.matrixL(); g.logdet = 0.0;
        for (int k = 0; k < V; ++k) g.logdet += 2.0 * std::log(g.L(k, k));
        g.ok = std::isfinite(g.logdet); return g;
    };
    auto lgauss = [&](const Gauss& g, const double* v) {
        const VX a = g.L.triangularView<Eigen::Lower>().solve(Eigen::Map<const VX>(v, V) - g.mu);
        return -0.5 * (a.squaredNorm() + g.logdet + V * LOG2PI);
    };
    std::vector<double> hv(N_ * V), lp0(N_), lq(N_), th(N_ * D, 0.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double la = std::log(seq_alpha_), lb = std::log(1.0 - seq_alpha_);
    typedef std::chrono::high_resolution_clock Clk;
    auto msec = [](Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const Clk::time_point tS0 = Clk::now();
    const int PARN = 64 + 64 * 64 + 225 + 64 + 64 + 64 * 64;
    auto draw = [&](const Gauss* g) {                        // g == nullptr: target prior only (GPU kernel)
        const Clk::time_point td = Clk::now();
        skip_ui_ = 1;
        if (!d_prop_par_) { cudaMalloc(&d_prop_par_, PARN * sizeof(double)); cudaMalloc(&d_prop_x_, (size_t)N_ * 64 * sizeof(double)); cudaMalloc(&d_prop_lp_, (size_t)N_ * 2 * sizeof(double)); }
        if (rbias_on_ && !d_rbias_) cudaMalloc(&d_rbias_, (size_t)N_ * sizeof(double));
        std::vector<double> par(PARN, 0.0);
        if (g) { for (int k = 0; k < V; ++k) par[k] = g->mu[k]; for (int r = 0; r < V; ++r) for (int c = 0; c <= r; ++c) par[64 + r * 64 + c] = g->L(r, c); }
        for (int r = 0; r < H; ++r) for (int c = 0; c <= r; ++c) par[64 + 64 * 64 + r * 15 + c] = LH(r, c);
        for (int j = 0; j < C; ++j) par[64 + 64 * 64 + 225 + j] = sc[j];
        // reflected component: the carried posterior mapped through the anchor plane (position and velocity normal parts)
        int have_r = 0; double logdet_r = 0.0;
        if (g && seq_refl_ > 0.0 && plane_set_ && !seq_headfix_) {
            const Eigen::Vector3d nrm(plane_n_[0], plane_n_[1], plane_n_[2]), cen(plane_c_[0], plane_c_[1], plane_c_[2]);
            const Eigen::Vector3d x0p(h_x0[0], h_x0[1], h_x0[2]), x0v(h_x0[3], h_x0[4], h_x0[5]);
            const Eigen::Matrix3d Hn = Eigen::Matrix3d::Identity() - 2.0 * nrm * nrm.transpose();
            MX J = MX::Identity(V, V); J.block(0, 0, 3, 3) = Hn; J.block(3, 3, 3, 3) = Hn;
            VX mur = J * g->mu;
            const double h0 = nrm.dot(x0p - cen), v0n = nrm.dot(x0v);
            for (int k = 0; k < 3; ++k) { mur[k] -= 2.0 * nrm[k] * h0; mur[3 + k] -= 2.0 * nrm[k] * v0n; }
            const MX Sg = g->L * g->L.transpose();
            const Gauss gr = make_gauss(mur, J * Sg * J.transpose());
            if (gr.ok) {
                have_r = 1; logdet_r = gr.logdet;
                const int o = 64 + 64 * 64 + 225 + 64;
                for (int k = 0; k < V; ++k) par[o + k] = gr.mu[k];
                for (int r = 0; r < V; ++r) for (int c = 0; c <= r; ++c) par[o + 64 + r * 64 + c] = gr.L(r, c);
            }
        }
        cudaMemcpy(d_prop_par_, par.data(), par.size() * sizeof(double), cudaMemcpyHostToDevice);
        prop_draw_kernel<<<(N_ + 127) / 128, 128>>>(d_states_, N_, H, C, g ? 1 : 0, seq_headfix_, seq_alpha_, g ? g->logdet : 0.0, logdet0,
                                                   d_prop_par_, d_prop_x_, d_prop_lp_, D, d_theta_, d_dx0_, d_logpq_, have_r, seq_refl_, logdet_r,
                                                   rbias_on_ ? d_rbias_ : nullptr, rb_head_);
        cudaMemcpy(hv.data(), d_prop_x_, (size_t)N_ * V * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> lp(2 * N_);
        cudaMemcpy(lp.data(), d_prop_lp_, (size_t)N_ * 2 * sizeof(double), cudaMemcpyDeviceToHost);
        cuda_check("prop_draw");
        for (int i = 0; i < N_; ++i) { lp0[i] = lp[2 * i]; lq[i] = lp[2 * i + 1]; }
        prof_[0] += msec(td, Clk::now());
    };
    auto run = [&](std::vector<double>& lw) {           // rollout inputs were written by the draw kernel
        for (int i = 0; i < N_; ++i) h_logpq_[i] = lp0[i] - lq[i];
        { const Clk::time_point tb = Clk::now(); basisPass(2, lw.data()); prof_[1] += msec(tb, Clk::now()); prof_[5] += 1; }
    };
    auto wfit = [&](const std::vector<double>& w, VX& mu, MX& S) {
        Eigen::Map<const MX> Xm(hv.data(), V, N_);
        Eigen::Map<const VX> wv(w.data(), N_);
        mu = Xm * wv;
        const MX D = Xm.colwise() - mu;
        S = (D.array().rowwise() * wv.transpose().array()).matrix() * D.transpose();
    };

    // ---- pass 1 from the carried proposal (or the prior); rebuild by tempered Gaussian refits if it collapses
    std::vector<double> lw(N_), tmp(N_), wt(N_);
    MX Sprior = MX::Zero(V, V);
    Sprior.block(0, 0, H, H) = PH;
    for (int j = 0; j < C; ++j) Sprior(H + j, H + j) = sc[j] * sc[j];
    Gauss g; bool have_g = false;
    MX Scarry_in; bool have_carry_cov = false;
    MX Scur = Sprior;                                        // covariance of the main component that drew the samples
    if (prop_valid_ && static_cast<int>(prop_mu_.size()) == V) {
        MX S(V, V); for (int r = 0; r < V; ++r) for (int c = 0; c < V; ++c) S(r, c) = prop_S_[V*r + c];
        VX mu = Eigen::Map<const VX>(prop_mu_.data(), V);
        // move the input coefficients one epoch: the previous window's function on its epochs 1..T-1 = the current
        // window's epochs 0..T-2, projected on the current basis by least squares (per channel, K x K)
        const std::vector<double>& ph0 = (static_cast<int>(h_phi_prev_.size()) == K_ * T_) ? h_phi_prev_ : h_phi_;
        MX Phi(T_ - 1, K_), Psi(T_ - 1, K_);
        for (int t = 0; t < T_ - 1; ++t) for (int k = 0; k < K_; ++k) { Phi(t, k) = h_phi_[k * T_ + t]; Psi(t, k) = ph0[k * T_ + t + 1]; }
        const MX Mk = (Phi.transpose() * Phi).ldlt().solve(Phi.transpose() * Psi);
        MX Tm = MX::Identity(V, V);
        for (int ch = 0; ch < 6; ++ch) for (int k = 0; k < K_; ++k) for (int j = 0; j < K_; ++j)
            Tm(H + 6*k + ch, H + 6*j + ch) = Mk(k, j);
        mu = Tm * mu; S = Tm * S * Tm.transpose();
        g = make_gauss(mu, S); have_g = g.ok;
        if (have_g) { Scur = S; Scarry_in = S; have_carry_cov = true; }
    }
    const Gauss g_in = g; const bool have_in = have_g;
    if (!have_in) ewma_n_ = 0;
    // ---- matched LM machinery (lm:=1, and the seq_dump landscape): negative log target on the same GPU rollouts
    const int nst = 1 + 2 * V + V * (V - 1);          // central-difference stencil: gradient, diagonal, off-diagonal
    MX Pinv = MX::Zero(V, V);
    Pinv.block(0, 0, H, H) = PH.inverse();
    for (int j = 0; j < C; ++j) Pinv(H + j, H + j) = 1.0 / (sc[j] * sc[j]);
    VX hstep(V);
    for (int k = 0; k < H; ++k) hstep[k] = lm_h_ * std::sqrt(std::max(PH(k, k), 1e-18));
    for (int j = 0; j < C; ++j) hstep[H + j] = lm_h_ * sc[j];
    std::vector<double> Jv_;
    auto eval = [&](const MX& X, int n, std::vector<double>& out) {   // negative log target at the columns of X
        const Clk::time_point td = Clk::now();
        std::vector<double> dx0(n * 9, 0.0), thv(n * D, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < 9; ++k) dx0[9 * i + k] = X(k, i);
            if (H == 15) for (int k = 0; k < 6; ++k) thv[i * D + k] = X(9 + k, i);
            for (int j = 0; j < C; ++j) thv[i * D + 6 + j] = X(H + j, i);
        }
        cudaMemcpy(d_dx0_, dx0.data(), n * 9 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta_, thv.data(), n * D * sizeof(double), cudaMemcpyHostToDevice);
        if (rbias_on_ && d_rbias_) { std::vector<double> rbv(n); for (int i = 0; i < n; ++i) rbv[i] = rb_head_ + (H == 10 ? X(9, i) : 0.0); cudaMemcpy(d_rbias_, rbv.data(), n * sizeof(double), cudaMemcpyHostToDevice); }
        skip_ui_ = 1;
        pass_n_ = n; basisPass(2, lw.data()); pass_n_ = 0;
        prof_[0] += msec(td, Clk::now()); prof_[5] += 1;
        out.resize(n);
        for (int i = 0; i < n; ++i) out[i] = std::isfinite(h_cost_[i]) ? gamma_ * h_cost_[i] - lprior(X.col(i).data())
                                                                    : std::numeric_limits<double>::infinity();
    };
    VX lm_gr(V); MX lm_Hm(V, V); double lm_J0 = std::numeric_limits<double>::infinity();
    auto stencil = [&](const VX& at) {                  // lm_J0, lm_gr, lm_Hm at the point
        MX X(V, nst); int c = 0;
        X.col(c++) = at;
        for (int i = 0; i < V; ++i) { X.col(c) = at; X(i, c) += hstep[i]; ++c; X.col(c) = at; X(i, c) -= hstep[i]; ++c; }
        for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j) {
            X.col(c) = at; X(i, c) += hstep[i]; X(j, c) += hstep[j]; ++c;
            X.col(c) = at; X(i, c) -= hstep[i]; X(j, c) -= hstep[j]; ++c;
        }
        eval(X, nst, Jv_);
        lm_J0 = Jv_[0];
        for (int i = 0; i < V; ++i) {
            const double fp = Jv_[1 + 2 * i], fm = Jv_[2 + 2 * i];
            lm_gr[i] = (fp - fm) / (2.0 * hstep[i]);
            lm_Hm(i, i) = (fp + fm - 2.0 * lm_J0) / (hstep[i] * hstep[i]);
        }
        c = 1 + 2 * V;
        for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j) {
            const double fp = Jv_[c], fm = Jv_[c + 1]; c += 2;
            const double hij = (fp + fm - 2.0 * lm_J0 - hstep[i] * hstep[i] * lm_Hm(i, i) - hstep[j] * hstep[j] * lm_Hm(j, j)) / (2.0 * hstep[i] * hstep[j]);
            lm_Hm(i, j) = lm_Hm(j, i) = hij;
        }
        if (!lm_Hm.allFinite() || !std::isfinite(lm_J0)) { lm_Hm = Pinv; lm_gr = Pinv * at; }
    };
    auto psd = [&](const MX& A) {                        // data part (A - prior precision) clamped PSD, prior precision kept:
        const MX Dm = 0.5 * (A + A.transpose()) - Pinv;  // the posterior precision is never below the prior's
        Eigen::SelfAdjointEigenSolver<MX> es(Dm);
        VX ev = es.eigenvalues();
        for (int k = 0; k < V; ++k) ev[k] = std::max(ev[k], 0.0);
        return MX(es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose() + Pinv);
    };
    // LM from a start point: the accepted iterates (start first) in path, the start gradient in grad0 (optional);
    // lm_J0 / lm_gr / lm_Hm are left at the last iterate
    auto lm_run = [&](const VX& start, std::vector<VX>& path, VX* grad0) {
        VX v = start; double lambda = 1e-2; path.clear(); path.push_back(v);
        int it = 0;
        stencil(v);
        if (grad0) *grad0 = lm_gr;
        for (; it < lm_iters_; ++it) {
            const MX Hp = psd(lm_Hm);
            constexpr int NT = 8;
            MX Xc(V, NT); std::vector<double> lam(NT);
            for (int t = 0; t < NT; ++t) {
                lam[t] = lambda * std::pow(4.0, t);
                const MX Hd = Hp + lam[t] * MX(Pinv.diagonal().asDiagonal());
                Xc.col(t) = v - Hd.ldlt().solve(lm_gr);
            }
            std::vector<double> Jc; eval(Xc, NT, Jc);
            int best = -1;
            for (int t = 0; t < NT; ++t) if (std::isfinite(Jc[t]) && Jc[t] < lm_J0 && (best < 0 || Jc[t] < Jc[best])) best = t;
            if (best < 0) { lambda *= std::pow(4.0, NT); if (lambda > 1e12) break; continue; }
            const double dec = lm_J0 - Jc[best];
            v = Xc.col(best); lambda = std::max(1e-4, lam[best] / 3.0);
            path.push_back(v);
            stencil(v);
            if (dec < 1e-3) break;
        }
        return it;
    };
    const double ess_done = seq_done_ * N_;                  // rebuild until this fraction of the samples is effective
    const double n_fit = 4.0 * V;                             // effective samples needed for a V-dim covariance
    int passes = 1;
    double ess = 0.0;
    if (lm_on_) {
        // ---- matched Levenberg-Marquardt (mppi_solver.cuh): mode of the same target, then the Laplace Gaussian
        // N(mode, Hess^-1) drawn once in place of the weighted samples
        if (nst > N_) { std::fprintf(stderr, "[mppi] lm: N %d < stencil %d\n", N_, nst); std::abort(); }
        std::vector<VX> path;
        const int it = lm_run(have_g ? g.mu : VX::Zero(V), path, nullptr);
        const VX v = path.back();
        last_lm_iters_ = it; last_lm_J_ = lm_J0;
        MX S = psd(lm_Hm).inverse(); S = 0.5 * (S + S.transpose());
        g = make_gauss(v, S); have_g = g.ok;
        if (have_g) Scur = S;
        const double a_save = seq_alpha_; seq_alpha_ = 1.0;   // draw the Laplace Gaussian only
        draw(have_g ? &g : nullptr); run(lw); seq_alpha_ = a_save;
        for (int i = 0; i < N_; ++i) h_wi_[i] = 1.0 / N_;
        ess = static_cast<double>(N_); last_ess1_ = ess; passes = it + 2;
        if (seq_debug_ && (prof_n_ % 500 == 0))
            std::fprintf(stderr, "[lm] iters %d J %.2f |g| %.2e\n", it, lm_J0, lm_gr.norm());
    } else {
    draw(have_g ? &g : nullptr); run(lw);
    ess = normalise_weights(lw, h_wi_);
    last_ess1_ = ess;
    if (seq_debug_ && have_g) {
        // first-pass diagnostics: share of weight on the defensive (prior) draws, spread of the log-weights, and the
        // spread of the two parts of the log-weight (likelihood vs prior/proposal ratio)
        double wdef = 0.0; std::vector<double> lwm, lll, lpq;
        for (int i = 0; i < N_; ++i) {
            const bool main = (lp0[i] - lq[i]) < -std::log(1.0 - seq_alpha_) - 1e-9 && std::isfinite(lw[i]);
            (void)main;
            lll.push_back(-gamma_ * h_cost_[i]); lpq.push_back(h_logpq_[i]);
        }
        auto sd = [](const std::vector<double>& v) { double m = 0, q = 0; int n = 0; for (double x : v) if (std::isfinite(x)) { m += x; ++n; } m /= n;
                                                     for (double x : v) if (std::isfinite(x)) q += (x - m) * (x - m); return std::sqrt(q / n); };
        int ib0 = 0; for (int i = 1; i < N_; ++i) if (h_wi_[i] > h_wi_[ib0]) ib0 = i;
        std::fprintf(stderr, "[seq1] ESS1 %.1f | sd(-gamma J) %.2f sd(log p/q) %.2f | best sample: -gJ %.1f (median %.1f) logpq %.2f | head z offset of best %.4f\n",
                     ess, sd(lll), sd(lpq), lll[ib0], [&]{ std::vector<double> c = lll; std::nth_element(c.begin(), c.begin() + N_/2, c.end()); return c[N_/2]; }(),
                     lpq[ib0], hv[V*ib0 + 2]);
        (void)wdef;
    }
    if (ess < ess_min_ * N_) {
        auto ess_at = [&](double b) {
            for (int i = 0; i < N_; ++i) tmp[i] = std::isfinite(h_cost_[i]) ? -gamma_ * b * h_cost_[i] + h_logpq_[i]
                                                                           : -std::numeric_limits<double>::infinity();
            return normalise_weights(tmp, wt);
        };
        // starting exponent: the largest beta whose ESS still supports a covariance fit (0 if none)
        double beta = 0.0;
        {
            double lo = 0.0, hi = 1.0;
            if (ess_at(1.0) >= n_fit) lo = 1.0;
            else for (int it = 0; it < 40; ++it) { const double mid = 0.5 * (lo + hi); if (ess_at(mid) >= n_fit) lo = mid; else hi = mid; }
            beta = lo;
        }
        for (int st = 0; st < seq_max_stages_; ++st) {
            // next exponent: ESS may halve per stage but must stay >= n_fit so that the refit is well posed
            const double ess_cur = ess_at(beta);
            const double target = std::max(0.5 * ess_cur, n_fit);
            double bnew = beta;
            if (ess_at(1.0) >= target) bnew = 1.0;
            else if (ess_cur >= target) {
                double lo = beta, hi = 1.0;
                for (int it = 0; it < 40; ++it) { const double mid = 0.5 * (lo + hi); if (ess_at(mid) >= target) lo = mid; else hi = mid; }
                bnew = lo;
            }
            const double n_eff = ess_at(bnew);                // wt <- normalised weights at bnew
            VX mu; MX S; wfit(wt, mu, S);
            // shrink the weighted covariance toward the covariance that drew the samples, with the weight of an
            // inverse-Wishart prior of V pseudo-observations: the sample covariance of n_eff draws underestimates
            // the small eigenvalues by about (1 - sqrt(V / n_eff))^2
            const double rho = n_eff / (n_eff + seq_shrink_k_ * V);
            const MX Snew = rho * seq_inflate_ * S + (1.0 - rho) * Scur;
            g = make_gauss(mu, Snew); have_g = g.ok;
            if (have_g) Scur = Snew;
            beta = bnew;
            draw(have_g ? &g : nullptr); run(lw); ++passes;
            ess = normalise_weights(lw, h_wi_);
            if (seq_debug_) std::fprintf(stderr, "[seq] stage %d beta %.3g fitESS %.0f rho %.2f -> ESS(beta) %.1f ESS(1) %.1f\n",
                                         st, beta, n_eff, rho, ess_at(beta), ess);
            if (beta >= 1.0 && ess >= ess_done) break;
        }
        ess = normalise_weights(lw, h_wi_);
    }
    }   // sampling branch
    const Eigen::MatrixXd Sused = Scur;
    last_ess2_ = ess; last_ess_ = ess; last_iters_ = passes; last_stages_ = passes - 1;
    if (seq_debug_ == 3 && have_in) {
        VX mu = VX::Zero(V); MX S = MX::Zero(V, V);
        for (int i = 0; i < N_; ++i) if (h_wi_[i] > 0.0) mu += h_wi_[i] * Eigen::Map<const VX>(&hv[V * i], V);
        for (int i = 0; i < N_; ++i) if (h_wi_[i] > 0.0) { const VX d = Eigen::Map<const VX>(&hv[V * i], V) - mu; S += h_wi_[i] * d * d.transpose(); }
        const MX Sin = g_in.L * g_in.L.transpose();
        for (int k = 0; k < V; ++k) { dbg_shift_[k] += std::abs(mu[k] - g_in.mu[k]) / std::sqrt(Sin(k, k)); dbg_ratio_[k] += std::sqrt(S(k, k) / Sin(k, k)); }
        const MX W = g_in.L.inverse();
        const MX Sw = W * S * W.transpose(); const VX dw = W * (mu - g_in.mu);
        Eigen::SelfAdjointEigenSolver<MX> es(Sw);
        dbg_wmin_ += es.eigenvalues().minCoeff(); dbg_wmax_ += es.eigenvalues().maxCoeff(); dbg_wshift_ += dw.norm(); dbg_n_ += 1;
        if (dbg_n_ % 300 == 0) {
            const double n = static_cast<double>(dbg_n_);
            std::fprintf(stderr, "[seqdbg] %ld solves | whitened posterior sd min %.3f max %.3f (1 = proposal) | mean shift %.2f sd | ESS1 %.0f\n",
                         dbg_n_, std::sqrt(std::max(dbg_wmin_ / n, 0.0)), std::sqrt(dbg_wmax_ / n), dbg_wshift_ / n, last_ess1_);
            std::fprintf(stderr, "[seqdbg] shift/sd:");
            for (int k = 0; k < V; ++k) std::fprintf(stderr, " %.2f", dbg_shift_[k] / n);
            std::fprintf(stderr, "\n[seqdbg] sd ratio:");
            for (int k = 0; k < V; ++k) std::fprintf(stderr, " %.2f", dbg_ratio_[k] / n);
            std::fprintf(stderr, "\n");
        }
    }
    {   // predictive log-likelihood of the newest epoch (branch evidence), from the final pass
        double m1 = -std::numeric_limits<double>::infinity(), m2 = m1;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { m1 = std::max(m1, lw[i]); m2 = std::max(m2, lw[i] + gamma_ * h_cost_newest_[i]); }
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { s1 += std::exp(lw[i] - m1); s2 += std::exp(lw[i] + gamma_ * h_cost_newest_[i] - m2); }
        last_logpred_ = (m1 + std::log(s1)) - (m2 + std::log(s2)); last_logpred1_ = last_logpred_;
    }
    last_cost_uwb_ = 0.0;
    for (int i = 0; i < N_; ++i) last_cost_uwb_ += h_wi_[i] * h_cost_[i];

    const Clk::time_point tH = Clk::now();
    // ---- new head = weighted mean of the states one step later; bias offset mean
    int ib = 0; for (int i = 1; i < N_; ++i) if (h_wi_[i] > h_wi_[ib]) ib = i;
    const Eigen::Matrix3d Rref = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&h_x1_[XS * ib + 6]);
    std::vector<Eigen::Matrix<double, 9, 1>> y(N_);
    Eigen::Matrix<double, 9, 1> m = Eigen::Matrix<double, 9, 1>::Zero();
    {
#ifdef _OPENMP
        const int nth = omp_get_max_threads();
#else
        const int nth = 1;
#endif
        std::vector<Eigen::Matrix<double, 9, 1>> part(nth, Eigen::Matrix<double, 9, 1>::Zero());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) {
            const double* xi = &h_x1_[XS * i];
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Ri(xi + 6);
            const Eigen::AngleAxisd aa(Eigen::Matrix3d(Rref.transpose() * Ri));
            const Eigen::Vector3d r = aa.angle() * aa.axis();
            y[i] << xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], r.x(), r.y(), r.z();
#ifdef _OPENMP
            part[omp_get_thread_num()] += h_wi_[i] * y[i];
#else
            part[0] += h_wi_[i] * y[i];
#endif
        }
        for (int t = 0; t < nth; ++t) m += part[t];
    }
    Eigen::Matrix<double, 9, 1> dm = Eigen::Matrix<double, 9, 1>::Zero();   // weighted mean of the sampled head offset
    for (int i = 0; i < N_; ++i) if (h_wi_[i] > 0.0) for (int k = 0; k < 9; ++k) dm[k] += h_wi_[i] * hv[V*i + k];
    // head carry: m is the window's smoothed estimate of the new head. Moving the head
    // by the whole smoothed shift every solve while the arrival covariance admits one epoch of information lets the head
    // hop freely wherever the window likelihood is ambiguous (near the anchor plane). Carry only the leaving fraction:
    // head = (smoothed) - (1 - g) * (mean head offset mapped to the new head), g = head_gain_ (1 keeps the smoothed head).
    {
        const double hs = dt_;
        for (int k = 0; k < 3; ++k) { m[k] -= (1.0 - head_gain_) * (dm[k] + hs * dm[3 + k]); m[3 + k] -= (1.0 - head_gain_) * dm[3 + k]; m[6 + k] -= (1.0 - head_gain_) * dm[6 + k]; }
    }
    for (int k = 0; k < 6; ++k) h_xn[k] = m[k];
    if (flush_) {   // flush output: weighted mean position at every window epoch (first / last solve, offline trajectory)
        flush_pos_.assign((size_t)T_ * 3, 0.0);
        for (int i = 0; i < N_; ++i) {
            if (h_wi_[i] <= 0.0) continue;
            const double* xi = &h_xall_[(size_t)i * T_ * 3];
            for (int t = 0; t < T_; ++t) for (int k = 0; k < 3; ++k) flush_pos_[t * 3 + k] += h_wi_[i] * xi[t * 3 + k];
        }
    }
    {   // lag-0 output: weighted mean of the samples' states at the newest window epoch (position, velocity, attitude)
        Eigen::Matrix<double, 9, 1> mT = Eigen::Matrix<double, 9, 1>::Zero();
        const Eigen::Matrix3d RrefT = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&h_xT_[DIM_X * ib + 6]);
#ifdef _OPENMP
        const int nth = omp_get_max_threads();
#else
        const int nth = 1;
#endif
        std::vector<Eigen::Matrix<double, 9, 1>> part(nth, Eigen::Matrix<double, 9, 1>::Zero());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) {
            if (h_wi_[i] <= 0.0) continue;
            const double* xi = &h_xT_[DIM_X * i];
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Ri(xi + 6);
            const Eigen::AngleAxisd aa(Eigen::Matrix3d(RrefT.transpose() * Ri)); const Eigen::Vector3d r = aa.angle() * aa.axis();
#ifdef _OPENMP
            Eigen::Matrix<double, 9, 1>& pm = part[omp_get_thread_num()];
#else
            Eigen::Matrix<double, 9, 1>& pm = part[0];
#endif
            for (int k = 0; k < 6; ++k) pm[k] += h_wi_[i] * xi[k];
            for (int k = 0; k < 3; ++k) pm[6 + k] += h_wi_[i] * r[k];
        }
        for (int t = 0; t < nth; ++t) mT += part[t];
        const Eigen::Vector3d rT = mT.tail<3>();
        const Eigen::Matrix3d RT = RrefT * (rT.norm() > 1e-12 ? Eigen::AngleAxisd(rT.norm(), rT.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
        for (int k = 0; k < 6; ++k) last_xT_[k] = mT[k];
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) last_xT_[6 + 3*r + c] = RT(r, c);
    }
    const Eigen::Vector3d mr = m.tail<3>();
    const Eigen::Matrix3d Rm = Rref * (mr.norm() > 1e-12 ? Eigen::AngleAxisd(mr.norm(), mr.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) h_xn[6 + 3*r + c] = Rm(r, c);
    prof_[2] += msec(tH, Clk::now()); const Clk::time_point tA = Clk::now();
    // bias carry: the window's posterior mean of the bias offset is a smoothed
    // estimate over a window that overlaps the next one by (T-1)/T; applying it in full every solve re-counts the same
    // data and the bias random-walks. Only the fraction that leaves the window per solve is carried: gain = 1/T.
    for (int k = 0; k < 6; ++k) last_db_[k] = 0.0;
    if (H == 15) { for (int i = 0; i < N_; ++i) for (int k = 0; k < 6; ++k) last_db_[k] += h_wi_[i] * hv[V*i + 9 + k];
                   for (int k = 0; k < 6; ++k) last_db_[k] *= bias_gain_; }
    last_drb_ = 0.0;
    if (H == 10) { for (int i = 0; i < N_; ++i) last_drb_ += h_wi_[i] * hv[V*i + 9]; last_drb_ *= bias_gain_; }
    for (int k = 0; k < 9; ++k) { last_dmean_[k] = 0.0; for (int i = 0; i < N_; ++i) last_dmean_[k] += h_wi_[i] * hv[V*i + k]; }

    // head coordinates of the new head: (p, v, log(Rm' R)) - m, bias offset - its mean
    auto ycoord = [&](int i, double* out) {
        const double* xi = &h_x1_[XS * i];
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Ri(xi + 6);
        const Eigen::AngleAxisd aa(Eigen::Matrix3d(Rm.transpose() * Ri));
        const Eigen::Vector3d r = aa.angle() * aa.axis();
        for (int k = 0; k < 6; ++k) out[k] = xi[k] - m[k];
        for (int k = 0; k < 3; ++k) out[6 + k] = r[k];
        if (H == 15) for (int k = 0; k < 6; ++k) out[9 + k] = hv[V*i + 9 + k] - last_db_[k];
        if (H == 10) out[9] = hv[V*i + 9] - last_drb_;
    };
    MX Y(N_, H);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i) { double o[16]; ycoord(i, o); for (int k = 0; k < H; ++k) Y(i, k) = o[k]; }

    // ---- one-step process noise Q_H (white IMU noise over h; bias random walk)
    const double h = dt_;
    MX QH = MX::Zero(H, H);
    {
        Eigen::Matrix3d Sa = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) Sa(k, k) = head_q_[k] * head_q_[k];
        Sa = Rm * Sa * Rm.transpose();
        QH.block(0, 0, 3, 3) = Sa * h * h * h / 3.0; QH.block(0, 3, 3, 3) = Sa * h * h / 2.0;
        QH.block(3, 0, 3, 3) = Sa * h * h / 2.0;     QH.block(3, 3, 3, 3) = Sa * h;
        for (int k = 0; k < 3; ++k) QH(6 + k, 6 + k) = head_q_[3 + k] * head_q_[3 + k] * h;
        if (H == 15) for (int k = 0; k < 6; ++k) QH(9 + k, 9 + k) = head_qb_[k] * head_qb_[k] * h;
        if (H == 10) QH(9, 9) = rb_q_ * rb_q_ * h;
    }

    prof_[6] += msec(tA, Clk::now()); const Clk::time_point tB = Clk::now();
    // ---- next proposal: weighted posterior moved one epoch (head -> new head, basis coefficients shifted)
    {
        MX U(N_, V);                                     // head part in new-head coordinates; inputs shifted next solve
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) {
            if (!(h_wi_[i] > 0.0)) { U.row(i).setZero(); continue; }      // zero-weight samples do not enter (as before)
            for (int k = 0; k < H; ++k) U(i, k) = Y(i, k);
            for (int j = 0; j < C; ++j) U(i, H + j) = hv[V*i + H + j];
        }
        Eigen::Map<const VX> wv(h_wi_.data(), N_);
        const VX mu = U.transpose() * wv;
        MX D = U.rowwise() - mu.transpose();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) D.row(i) *= (h_wi_[i] > 0.0 ? std::sqrt(h_wi_[i]) : 0.0);
        MX S = D.transpose() * D;                        // sum_i w_i (u_i - mu)(u_i - mu)' (host time)
        // shrink toward the covariance of the proposal used in this solve (V pseudo-observations), then add the
        // one-epoch change: one step of process noise on the head and the prior mass of the new epoch on the inputs
        if (seq_ewma_ && have_carry_cov) {
            // the window posterior's covariance changes on the scale of a window, while one solve's weighted covariance
            // (about a hundred effective samples in V dims) has its eigenvalues spread by (1 -+ sqrt(V/n))^2
            // (Marchenko-Pastur): average the estimates over the last T solves (EWMA, time constant T, plain average
            // while fewer than T solves have contributed); the mean is taken fresh every solve
            ewma_n_ += 1;
            const double a = std::max(2.0 / (T_ + 1.0), 1.0 / (ewma_n_ + 1.0));
            S = (1.0 - a) * Scarry_in + a * seq_inflate_ * S;
        } else if (seq_shrink_carry_) {
            MX Sh = Sused;                                   // proposal coordinates of this solve ~ the next ones
            const double rho = ess / (ess + seq_shrink_k_ * V);
            S = rho * seq_inflate_ * S + (1.0 - rho) * Sh;
        } else {
            S *= seq_inflate_;
        }
        S.block(0, 0, H, H) += QH;
        if (seq_newmass_) for (int j = 0; j < C; ++j) S(H + j, H + j) += sc[j] * sc[j] / T_;
        prop_mu_.assign(mu.data(), mu.data() + V);
        prop_S_.resize(V * V);
        for (int r = 0; r < V; ++r) for (int c = 0; c < V; ++c) prop_S_[V*r + c] = S(r, c);
        prop_valid_ = ess >= n_fit ? 1 : 0;
    }

    prof_[7] += msec(tB, Clk::now()); const Clk::time_point tC = Clk::now();
    // ---- arrival covariance for the new head (sample-based Laplace, all samples unweighted, section 12):
    // predict F from the regression of the new-head coordinates on the sampled vector, update with the leaving epoch
    MX P1;
    Eigen::Matrix<double, 9, 1> g0y = Eigen::Matrix<double, 9, 1>::Zero(); bool have_g0 = false;
    {
        const int st = std::max(1, N_ / 2048), Ns = (N_ + st - 1) / st;       // strided subset for the regressions
        MX X(Ns, V + 1), Ys(Ns, H);
        for (int j = 0, i = 0; i < N_; i += st, ++j) { X(j, 0) = 1.0; for (int k = 0; k < V; ++k) X(j, 1 + k) = hv[V*i + k]; Ys.row(j) = Y.row(i); }
        const MX Bc = (X.transpose() * X).ldlt().solve(X.transpose() * Ys);                 // (V + 1) x H
        const MX F = Bc.block(1, 0, H, H).transpose();                  // H x H: new head coords per head offset
        const MX Pm = F * PH * F.transpose() + QH;
        MX A = MX::Zero(H, H);
        const int id6[6] = {0, 1, 2, 6, 7, 8};
        Eigen::Matrix<double, 6, 1> yb6 = Eigen::Matrix<double, 6, 1>::Zero();
        for (int i = 0; i < N_; ++i) for (int a = 0; a < 6; ++a) yb6[a] += Y(i, id6[a]) / N_;
        Eigen::Matrix<double, 6, 6> S6 = Eigen::Matrix<double, 6, 6>::Zero();
        double lmin = 1e300, lmax = -1e300;
        for (int i = 0; i < N_; ++i) {
            Eigen::Matrix<double, 6, 1> d; for (int a = 0; a < 6; ++a) d[a] = Y(i, id6[a]) - yb6[a];
            S6 += d * d.transpose();
            const double l = h_x1_[XS * i + DIM_X];
            if (std::isfinite(l)) { lmin = std::min(lmin, l); lmax = std::max(lmax, l); }
        }
        S6 /= (N_ - 1);
        last_fit_r2_ = -1.0;
        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt6(S6);
        if (lmax - lmin > 1e-12 && llt6.info() == Eigen::Success) {
            const Eigen::Matrix<double, 6, 6> L6 = llt6.matrixL();
            constexpr int NF = 1 + 6 + 21;
            MX Xq(Ns, NF); VX lv(Ns); int n = 0;
            for (int i = 0; i < N_; i += st) {
                const double l = h_x1_[XS * i + DIM_X];
                if (!std::isfinite(l)) continue;
                Eigen::Matrix<double, 6, 1> d; for (int a = 0; a < 6; ++a) d[a] = Y(i, id6[a]) - yb6[a];
                const Eigen::Matrix<double, 6, 1> u = L6.triangularView<Eigen::Lower>().solve(d);
                int c = 0; Xq(n, c++) = 1.0;
                for (int a = 0; a < 6; ++a) Xq(n, c++) = u[a];
                for (int a = 0; a < 6; ++a) for (int b = a; b < 6; ++b) Xq(n, c++) = u[a] * u[b];
                lv[n++] = l;
            }
            Xq.conservativeResize(n, NF); lv.conservativeResize(n);
            const VX cf = Xq.colPivHouseholderQr().solve(lv);
            const VX res = lv - Xq * cf;
            const double var_l = (lv.array() - lv.mean()).square().mean();
            last_fit_r2_ = var_l > 0.0 ? 1.0 - res.squaredNorm() / n / var_l : -1.0;
            Eigen::Matrix<double, 6, 6> Hu = Eigen::Matrix<double, 6, 6>::Zero();
            int c = 7;
            for (int a = 0; a < 6; ++a) for (int b = a; b < 6; ++b) { if (a == b) Hu(a, a) = 2.0 * cf[c]; else Hu(a, b) = Hu(b, a) = cf[c]; ++c; }
            const Eigen::Matrix<double, 6, 6> Li = L6.inverse();
            { const Eigen::Matrix<double, 6, 1> g6 = gamma_ * Li.transpose() * cf.segment<6>(1);   // leaving-epoch cost gradient at the sample mean, y coordinates
              for (int a = 0; a < 6; ++a) g0y[id6[a]] = g6[a]; have_g0 = true; }
            Eigen::Matrix<double, 6, 6> A6 = Li.transpose() * Hu * Li;
            A6 = 0.5 * (A6 + A6.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(A6);
            const Eigen::Matrix<double, 6, 1> ev = es.eigenvalues().cwiseMax(0.0);
            A6 = gamma_ * es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
            for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) A(id6[a], id6[b]) = A6(a, b);
        }
        P1 = (MX::Identity(H, H) + Pm * A).partialPivLu().solve(Pm);
        if (head_filt_ && have_g0 && (H == 9 || H == 10)) {
            // filtered head: the smoothed mean m moved the head by the whole window's
            // information while the arrival prior admits only the leaving epoch. Start from the dead-reckoning prediction
            // y_pred = -(F dm) (mean head offset removed; coordinates relative to m) and apply that epoch's Kalman update
            // with the quadratic model of its cost: y_filt = y_pred - (Pm^-1 + A)^-1 (g0 + A y_pred).
            Eigen::Matrix<double, 9, 1> yp;
            const double hs = dt_;
            for (int k = 0; k < 3; ++k) { yp[k] = -(dm[k] + hs * dm[3 + k]); yp[3 + k] = -dm[3 + k]; yp[6 + k] = -dm[6 + k]; }
            const Eigen::Matrix<double, 9, 9> Pm9 = Pm.block(0, 0, 9, 9), A9 = A.block(0, 0, 9, 9);
            const Eigen::Matrix<double, 9, 9> Lam = Pm9.inverse() + A9;
            const Eigen::Matrix<double, 9, 1> yf = yp - Lam.ldlt().solve(g0y + A9 * yp);
            if (yf.allFinite()) {
                Eigen::Matrix<double, 9, 1> mf; for (int k = 0; k < 9; ++k) mf[k] = m[k] + yf[k];
                for (int k = 0; k < 6; ++k) h_xn[k] = mf[k];
                const Eigen::Vector3d mr2 = mf.tail<3>();
                const Eigen::Matrix3d Rf = Rref * (mr2.norm() > 1e-12 ? Eigen::AngleAxisd(mr2.norm(), mr2.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) h_xn[6 + 3*r + c] = Rf(r, c);
                for (int k = 0; k < 9; ++k) { prop_mu_[k] -= yf[k]; head_shift_[k] = yf[k]; }   // proposal coordinates are relative to the new head
            }
        }
        P1 = 0.5 * (P1 + P1.transpose());
        if (!P1.allFinite() || P1.ldlt().info() != Eigen::Success) P1 = Pm;
        for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) last_A_[9*r + c] = A(r, c);
    }
    last_PH_.resize(H * H);
    for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c) last_PH_[H*r + c] = P1(r, c);
    for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) last_P1_[9*r + c] = P1(r, c);
    if (seq_strip_to_ > 0 && prof_n_ + 1 >= seq_strip_from_ && prof_n_ + 1 <= seq_strip_to_ && ((prof_n_ + 1 - seq_strip_from_) % seq_strip_step_) == 0) {
        // per-solve strip: the posterior samples' vertical head offset and weight (every 8th sample),
        // the arrival head z and the new head z
        FILE* fo = std::fopen(seq_dump_file_.empty() ? "seq_strip.txt" : seq_dump_file_.c_str(), "a");
        if (fo) {
            std::fprintf(fo, "E %ld %.6f %.6f %.1f %.6f %.6f\n", prof_n_ + 1, h_x0[2], h_x0[5], ess, h_xn[2], h_xn[5]);
            for (int i = 0; i < N_; i += 8) std::fprintf(fo, "Z %.5g %.5g %.6g\n", hv[V*i+2], hv[V*i+5], h_wi_[i]);
            std::fclose(fo);
        }
    }
    if (seq_dump_ > 0 && prof_n_ + 1 == seq_dump_) {
        // ---- landscape dump: this solve's posterior samples, the negative log
        // target on a (dp_z, dv_z) grid through the carried start and through the posterior mean, and the LM path from the
        // same start. Buffers are clobbered; nothing after this solve reads them.
        FILE* fo = std::fopen(seq_dump_file_.empty() ? "seq_dump.txt" : seq_dump_file_.c_str(), "w");
        if (fo) {
            // sample cloud from this solve's carried proposal (first pass, defensive mixture), whichever branch ran
            draw(have_in ? &g_in : nullptr); run(lw);
            ess = normalise_weights(lw, h_wi_);
            VX pm = VX::Zero(V);
            for (int i = 0; i < N_; ++i) if (h_wi_[i] > 0.0) pm += h_wi_[i] * Eigen::Map<const VX>(&hv[V * i], V);
            const VX start = have_in ? g_in.mu : VX::Zero(V);
            std::fprintf(fo, "# V %d H %d N %d T %d dt %.6f x0z %.6f x0vz %.6f ess %.1f passes %d\n", V, H, N_, T_, dt_, h_x0[2], h_x0[5], ess, passes);
            std::fprintf(fo, "START"); for (int k = 0; k < V; ++k) std::fprintf(fo, " %.9g", start[k]); std::fprintf(fo, "\n");
            std::fprintf(fo, "PMEAN"); for (int k = 0; k < V; ++k) std::fprintf(fo, " %.9g", pm[k]); std::fprintf(fo, "\n");
            std::fprintf(fo, "SC"); for (int j = 0; j < C; ++j) std::fprintf(fo, " %.9g", sc[j]); std::fprintf(fo, "\n");
            for (int i = 0; i < N_; ++i)
                std::fprintf(fo, "S %.6g %.6g %.6g %.6g %.6g %.6g %.9g %.6g\n", hv[V*i+0], hv[V*i+1], hv[V*i+2], hv[V*i+3], hv[V*i+4], hv[V*i+5],
                             h_wi_[i] > 0.0 ? std::log(h_wi_[i]) : -1e300, h_cost_[i]);
            const int NG = 61; const double pz = 1.0, vz = 1.0;
            for (int which = 0; which < 2; ++which) {
                const VX& base = which == 0 ? start : pm;
                MX X(V, NG * NG); int c = 0;
                for (int a = 0; a < NG; ++a) for (int b = 0; b < NG; ++b) {
                    X.col(c) = base; X(2, c) = base[2] + (-pz + 2.0 * pz * a / (NG - 1)); X(5, c) = base[5] + (-vz + 2.0 * vz * b / (NG - 1)); ++c;
                }
                std::vector<double> Jg; eval(X, NG * NG, Jg);
                std::vector<double> Ju(NG * NG); for (int i = 0; i < NG * NG; ++i) Ju[i] = gamma_ * h_cost_[i];   // likelihood part (eval leaves h_cost_)
                c = 0;
                for (int a = 0; a < NG; ++a) for (int b = 0; b < NG; ++b) { std::fprintf(fo, "G%d %.6g %.6g %.9g %.9g\n", which, X(2, c), X(5, c), Jg[c], Ju[c]); ++c; }
            }
            std::vector<VX> path; VX g0(V);
            lm_run(start, path, &g0);
            for (size_t k = 0; k < path.size(); ++k) { std::fprintf(fo, "L"); for (int d = 0; d < V; ++d) std::fprintf(fo, " %.9g", path[k][d]); std::fprintf(fo, "\n"); }
            std::fprintf(fo, "GRAD"); for (int k = 0; k < V; ++k) std::fprintf(fo, " %.9g", g0[k]); std::fprintf(fo, "\n");
            std::fprintf(fo, "LMJ %.9g\n", lm_J0);
            {   // Laplace covariance at the LM mode (what the LM branch would carry) and the weighted sample covariance
                const MX Sl = psd(lm_Hm).inverse();
                std::fprintf(fo, "LMCOV"); for (int r = 0; r < V; ++r) for (int c2 = 0; c2 < V; ++c2) std::fprintf(fo, " %.9g", Sl(r, c2)); std::fprintf(fo, "\n");
                MX Ss = MX::Zero(V, V);
                for (int i = 0; i < N_; ++i) if (h_wi_[i] > 0.0) { const VX d = Eigen::Map<const VX>(&hv[V * i], V) - pm; Ss += h_wi_[i] * d * d.transpose(); }
                std::fprintf(fo, "SCOV"); for (int r = 0; r < V; ++r) for (int c2 = 0; c2 < V; ++c2) std::fprintf(fo, " %.9g", Ss(r, c2)); std::fprintf(fo, "\n");
                std::fprintf(fo, "PH"); for (int r = 0; r < H; ++r) for (int c2 = 0; c2 < H; ++c2) std::fprintf(fo, " %.9g", PH(r, c2)); std::fprintf(fo, "\n");
            }
            std::fclose(fo);
            std::fprintf(stderr, "[seq_dump] solve %ld written\n", prof_n_ + 1);
        }
    }
    prof_[8] += msec(tC, Clk::now()); prof_[3] += msec(tA, Clk::now()); prof_[4] += msec(tS0, Clk::now()); prof_n_ += 1;
    if (seq_debug_ && prof_n_ % 500 == 0) {
        const double n = prof_n_, np = std::max(prof_[5], 1.0);
        std::fprintf(stderr, "[seqprof] per solve: draw %.1f | GPU pass %.1f (%.1f per pass, %.2f passes) | head %.1f | Y coords %.1f | proposal fit %.1f | arrival %.1f | rest %.1f | total %.1f ms\n",
                     prof_[0] / n, prof_[1] / n, prof_[1] / np, np / n, prof_[2] / n, prof_[6] / n, prof_[7] / n, prof_[8] / n,
                     (prof_[4] - prof_[0] - prof_[1] - prof_[2] - prof_[3]) / n, prof_[4] / n);
    }
}

// ---------------------------------- Rao-Blackwellised head/bias with the sequential proposal (section 15) ----
void MPPISolver::solveRB(double* h_xn, const double* h_x0, const double* h_U0,
                         const double* h_ranges, const double* h_stds, const double* h_accgyr) {
    typedef Eigen::MatrixXd MX; typedef Eigen::VectorXd VX;
    if (dv_on_ || K_ <= 0) { std::fprintf(stderr, "[mppi] solveRB needs basis mode without dv\n"); std::abort(); }
    cudaMemcpyAsync(d_x0_,     h_x0,     DIM_X       * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_U0_,     h_U0,     len_U0_     * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_ranges_, h_ranges, len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_stds_,   h_stds,   len_ranges_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_accgyr_, h_accgyr, len_accgyr_ * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaStreamSynchronize(h2dS_);
    if (!d_rb_) { cudaMalloc(&d_rb_, (size_t)N_ * RB_OUT * sizeof(double)); h_rb_.assign((size_t)N_ * RB_OUT, 0.0); cuda_check("solveRB alloc"); }
    cuda_check("solveRB entry");

    const int H = head_dim_, C = 6 * K_, V = C, D = D_;
    const double LOG2PI = std::log(2.0 * M_PI);
    // arrival prior on the head/bias offset
    MX PH(H, H);
    for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c) PH(r, c) = 0.5 * (headP_[H*r + c] + headP_[H*c + r]);
    MX PHinv; double logdetPH = 0.0;
    { Eigen::LLT<MX> llt(PH);
      if (llt.info() != Eigen::Success) { for (int k = 0; k < H; ++k) PH(k, k) = std::max(PH(k, k), 1e-18); llt.compute(PH); }
      const MX L = llt.matrixL(); for (int k = 0; k < H; ++k) logdetPH += 2.0 * std::log(L(k, k));
      PHinv = llt.solve(MX::Identity(H, H)); }
    // input prior
    VX sc(C); double logdetC = 0.0;
    for (int j = 0; j < C; ++j) { sc[j] = h_prior_std_[6 + j]; logdetC += 2.0 * std::log(sc[j]); }
    auto lprior = [&](const double* v) {
        double q = 0.0; for (int j = 0; j < C; ++j) { const double r = v[j] / sc[j]; q += r * r; }
        return -0.5 * (q + logdetC + C * LOG2PI);
    };
    struct Gauss { VX mu; MX L; double logdet; bool ok; };
    auto make_gauss = [&](const VX& mu, MX S) {
        Gauss g; g.mu = mu; g.ok = false; S = 0.5 * (S + S.transpose());
        for (int k = 0; k < V; ++k) S(k, k) += 1e-12 + 1e-9 * sc[k] * sc[k];
        Eigen::LLT<MX> llt(S);
        if (llt.info() != Eigen::Success || !S.allFinite()) return g;
        g.L = llt.matrixL(); g.logdet = 0.0;
        for (int k = 0; k < V; ++k) g.logdet += 2.0 * std::log(g.L(k, k));
        g.ok = std::isfinite(g.logdet); return g;
    };
    auto lgauss = [&](const Gauss& g, const double* v) {
        const VX a = g.L.triangularView<Eigen::Lower>().solve(Eigen::Map<const VX>(v, V) - g.mu);
        return -0.5 * (a.squaredNorm() + g.logdet + V * LOG2PI);
    };
    std::vector<double> hv(N_ * V), lp0(N_), lq(N_), th(N_ * D, 0.0), lmm(N_), mu_all(N_ * H), lamL((size_t)N_ * H * H, 0.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double la = std::log(seq_alpha_), lb = std::log(1.0 - seq_alpha_);
    typedef std::chrono::high_resolution_clock Clk;
    auto msec = [](Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const Clk::time_point t_solve0 = Clk::now();
    auto draw = [&](const Gauss* g) {
        const Clk::time_point t0 = Clk::now();
        VX z(V);
        for (int i = 0; i < N_; ++i) {
            for (int k = 0; k < V; ++k) z[k] = head_nrm_(head_rng_);
            double* v = &hv[V * i];
            if (g && unif(head_rng_) < seq_alpha_) { const VX x = g->mu + g->L * z; for (int k = 0; k < V; ++k) v[k] = x[k]; }
            else for (int j = 0; j < C; ++j) v[j] = sc[j] * z[j];
            lp0[i] = lprior(v);
            lq[i] = g ? lse2(la + lgauss(*g, v), lb + lp0[i]) : lp0[i];
        }
        prof_[0] += msec(t0, Clk::now());
    };
    // one pass: expand the coefficients, roll out with sensitivities, integrate the head/bias offset analytically
    auto unpack = [&](const double* src, MX& M) { for (int a = 0; a < H; ++a) for (int b = a; b < H; ++b) M(a, b) = M(b, a) = src[rb_idx(a, b)]; };
    auto run = [&](std::vector<double>& lw) {
        const Clk::time_point t0 = Clk::now();
        std::fill(th.begin(), th.end(), 0.0);
        for (int i = 0; i < N_; ++i) for (int j = 0; j < C; ++j) th[D*i + 6 + j] = hv[V*i + j];
        cudaMemcpy(d_theta_, th.data(), N_ * D * sizeof(double), cudaMemcpyHostToDevice);
        int bs = 0;
        while (bs < N_) {
            const int batch = std::min(CHUNK, N_ - bs);
            basis_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(
                d_states_, d_mu_, d_L_, d_prior_std_, d_phi_, 2, batch, T_, K_, 0, 0.0,
                d_theta_ + bs * D_, d_logpq_ + bs, d_noise_ + (size_t)bs * T_ * DIM_U, d_dv_ + bs * 3);
            cudaEventRecord(rngDone_, rngS_);
            cudaStreamWaitEvent(rollS_, rngDone_, 0);
            if (H == 9)
                rb_rollout_kernel_t<9><<<(batch + RB_BLOCK - 1) / RB_BLOCK, RB_BLOCK, 0, rollS_>>>(
                    batch, T_, num_tags_ * num_anchors_, dt_, d_gravity_, d_anchor_, d_ranges_, d_stds_, d_U0_,
                    d_noise_ + (size_t)bs * T_ * DIM_U, d_x0_, true_dt_ ? d_dts_ : nullptr, w_uwb_, d_rb_ + (size_t)bs * RB_OUT);
            else
                rb_rollout_kernel_t<15><<<(batch + RB_BLOCK - 1) / RB_BLOCK, RB_BLOCK, 0, rollS_>>>(
                    batch, T_, num_tags_ * num_anchors_, dt_, d_gravity_, d_anchor_, d_ranges_, d_stds_, d_U0_,
                    d_noise_ + (size_t)bs * T_ * DIM_U, d_x0_, true_dt_ ? d_dts_ : nullptr, w_uwb_, d_rb_ + (size_t)bs * RB_OUT);
            bs += batch;
        }
        cudaMemcpyAsync(h_rb_.data(), d_rb_, (size_t)N_ * RB_OUT * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaStreamSynchronize(rollS_);
        cuda_check("solveRB pass");
        const Clk::time_point t1 = Clk::now();
        prof_[1] += msec(t0, t1); prof_[5] += 1.0;
        MX Hs(H, H), Hl(H, H);
        for (int i = 0; i < N_; ++i) {
            const double* o = &h_rb_[(size_t)i * RB_OUT];
            unpack(o + RB_O_H, Hs); unpack(o + RB_O_HL, Hl);
            const VX g = Eigen::Map<const VX>(o + RB_O_G, RB_H).head(H), gl = Eigen::Map<const VX>(o + RB_O_GL, RB_H).head(H);
            auto marg = [&](double J, const VX& gg, const MX& HH, VX* mu) {
                const MX Lam = PHinv + gamma_ * HH;
                Eigen::LLT<MX> llt(Lam);
                if (llt.info() != Eigen::Success || !std::isfinite(J)) return -std::numeric_limits<double>::infinity();
                const VX sol = llt.solve(gg);
                const MX L = llt.matrixL(); double ld = 0.0; for (int k = 0; k < H; ++k) ld += 2.0 * std::log(L(k, k));
                if (mu) { *mu = -gamma_ * sol; for (int r = 0; r < H; ++r) for (int c = 0; c <= r; ++c) lamL[(size_t)i * H * H + r * H + c] = L(r, c); }
                return -gamma_ * J + 0.5 * gamma_ * gamma_ * gg.dot(sol) - 0.5 * ld - 0.5 * logdetPH;
            };
            VX mu(H);
            const double lm = marg(o[RB_O_J0], g, Hs, &mu);
            lmm[i] = marg(o[RB_O_J0] - o[RB_O_JL], g - gl, Hs - Hl, nullptr);
            for (int k = 0; k < H; ++k) mu_all[H*i + k] = mu[k];
            h_cost_[i] = std::isfinite(lm) ? -lm / gamma_ : std::numeric_limits<double>::infinity();
            h_logpq_[i] = lp0[i] - lq[i];
            lw[i] = std::isfinite(lm) ? lm + h_logpq_[i] : -std::numeric_limits<double>::infinity();
        }
        prof_[2] += msec(t1, Clk::now());
    };
    auto wfit = [&](const std::vector<double>& w, VX& mu, MX& S) {
        mu = VX::Zero(V); S = MX::Zero(V, V);
        for (int i = 0; i < N_; ++i) if (w[i] > 0.0) mu += w[i] * Eigen::Map<const VX>(&hv[V * i], V);
        for (int i = 0; i < N_; ++i) if (w[i] > 0.0) { const VX d = Eigen::Map<const VX>(&hv[V * i], V) - mu; S += w[i] * d * d.transpose(); }
    };

    std::vector<double> lw(N_), tmp(N_), wt(N_);
    bool verified = false;
    std::vector<double> dh((size_t)N_ * H, 0.0), lhq(N_, 0.0);
    auto verify = [&](std::vector<double>& lw, int& passes) {
        // ---- Laplace as proposal only (section 17): draw one head/bias offset per input sample from N(mu_i, Lambda_i^-1),
        // roll it out exactly and weight by the exact cost; the estimate is then the weighted sample mean
        const Clk::time_point tv0 = Clk::now();
        std::vector<double> lqh(N_), lph(N_);
        std::fill(th.begin(), th.end(), 0.0);
        VX z(H), d(H);
        for (int i = 0; i < N_; ++i) {
            for (int k = 0; k < H; ++k) z[k] = head_nrm_(head_rng_);
            Eigen::Map<const MX> L(&lamL[(size_t)i * H * H], H, H);        // row-major lower factor stored as (r, c) -> r*H + c
            MX Lm = MX::Zero(H, H); for (int r = 0; r < H; ++r) for (int c = 0; c <= r; ++c) Lm(r, c) = lamL[(size_t)i * H * H + r * H + c];
            // delta = mu + L^-T z  has covariance L^-T L^-1 = Lambda^-1
            d = Lm.transpose().triangularView<Eigen::Upper>().solve(z);
            double ldL = 0.0; for (int k = 0; k < H; ++k) ldL += std::log(Lm(k, k));
            for (int k = 0; k < H; ++k) d[k] += mu_all[H*i + k];
            lqh[i] = -0.5 * z.squaredNorm() + ldL - 0.5 * H * LOG2PI;        // log N(delta; mu, Lambda^-1)
            const VX a = PH.llt().matrixL().solve(d);
            lph[i] = -0.5 * a.squaredNorm() - 0.5 * logdetPH - 0.5 * H * LOG2PI;
            for (int k = 0; k < H; ++k) dh[(size_t)i * H + k] = d[k];
            for (int k = 0; k < 9; ++k) h_dx0_[9*i + k] = d[k];
            if (H == 15) for (int k = 0; k < 6; ++k) th[D*i + k] = d[9 + k];
            for (int j = 0; j < C; ++j) th[D*i + 6 + j] = hv[V*i + j];
        }
        cudaMemcpy(d_theta_, th.data(), N_ * D * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dx0_, h_dx0_.data(), N_ * 9 * sizeof(double), cudaMemcpyHostToDevice);
        int bs = 0;
        while (bs < N_) {
            const int batch = std::min(CHUNK, N_ - bs);
            basis_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(
                d_states_, d_mu_, d_L_, d_prior_std_, d_phi_, 2, batch, T_, K_, 0, 0.0,
                d_theta_ + bs * D_, d_logpq_ + bs, d_noise_ + (size_t)bs * T_ * DIM_U, d_dv_ + bs * 3);
            cudaEventRecord(rngDone_, rngS_);
            cudaStreamWaitEvent(rollS_, rngDone_, 0);
            rollout_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rollS_>>>(
                batch, T_, num_tags_ * num_anchors_, dt_, d_gravity_, d_anchor_, d_ranges_, d_stds_, d_accgyr_, d_U0_,
                d_noise_ + (size_t)bs * T_ * DIM_U, d_x0_, nullptr, gamma_, w_uwb_, 0.0, 0.0,
                d_Ui_ + (size_t)bs * T_ * DIM_U, d_cost_ + bs, d_cost_uwb_ + bs, d_cost_acc_ + bs, d_cost_gyr_ + bs,
                d_cost_last_ + bs, d_cost_newest_ + bs, d_dx0_ + bs * 9, d_x1_ + (size_t)bs * (DIM_X + 1), true_dt_ ? d_dts_ : nullptr, nullptr, nullptr, nullptr);
            bs += batch;
        }
        cudaMemcpyAsync(h_cost_.data(), d_cost_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaMemcpyAsync(h_cost_newest_.data(), d_cost_newest_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaMemcpyAsync(h_x1_.data(), d_x1_, (size_t)N_ * (DIM_X + 1) * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaStreamSynchronize(rollS_);
        cuda_check("solveRB verify pass");
        if (seq_debug_) {
            // Laplace marginal vs exact weight on the same samples: how wrong is the quadratic model?
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; int n = 0;
            std::vector<double> lap(N_), ex(N_);
            for (int i = 0; i < N_; ++i) {
                if (!std::isfinite(lw[i]) || !std::isfinite(h_cost_[i])) continue;
                lap[n] = lw[i]; ex[n] = -gamma_ * h_cost_[i] + lph[i] - lqh[i] + h_logpq_[i]; ++n;
            }
            for (int i = 0; i < n; ++i) { sa += lap[i]; sb += ex[i]; }
            sa /= n; sb /= n;
            for (int i = 0; i < n; ++i) { saa += (lap[i]-sa)*(lap[i]-sa); sbb += (ex[i]-sb)*(ex[i]-sb); sab += (lap[i]-sa)*(ex[i]-sb); }
            int ibl = 0, ibe = 0; for (int i = 1; i < n; ++i) { if (lap[i] > lap[ibl]) ibl = i; if (ex[i] > ex[ibe]) ibe = i; }
            std::fprintf(stderr, "[verify] n %d | laplace sd %.2f, exact sd %.2f, corr %.3f | exact minus laplace: mean %+.1f | best-laplace sample: exact rank gap %.1f | best-exact sample: laplace gap %.1f\n",
                         n, std::sqrt(saa/n), std::sqrt(sbb/n), sab/std::sqrt(saa*sbb), sb - sa, ex[ibe] - ex[ibl], lap[ibl] - lap[ibe]);
        }
        for (int i = 0; i < N_; ++i) {
            lhq[i] = lph[i] - lqh[i];
            lw[i] = std::isfinite(h_cost_[i]) ? -gamma_ * h_cost_[i] + lhq[i] + h_logpq_[i] : -std::numeric_limits<double>::infinity();
        }
        verified = true; ++passes;
        prof_[3] += msec(tv0, Clk::now());
    };
    // ---- pass 1 from the carried proposal (shifted one epoch) or the prior; tempered refits if it collapses
    MX Sprior = MX::Zero(V, V); for (int j = 0; j < C; ++j) Sprior(j, j) = sc[j] * sc[j];
    Gauss g; bool have_g = false; MX Scur = Sprior;
    if (prop_valid_ && static_cast<int>(prop_mu_.size()) == V) {
        MX S(V, V); for (int r = 0; r < V; ++r) for (int c = 0; c < V; ++c) S(r, c) = prop_S_[V*r + c];
        VX mu = Eigen::Map<const VX>(prop_mu_.data(), V);
        const std::vector<double>& ph0 = (static_cast<int>(h_phi_prev_.size()) == K_ * T_) ? h_phi_prev_ : h_phi_;
        MX Phi(T_ - 1, K_), Psi(T_ - 1, K_);
        for (int t = 0; t < T_ - 1; ++t) for (int k = 0; k < K_; ++k) { Phi(t, k) = h_phi_[k * T_ + t]; Psi(t, k) = ph0[k * T_ + t + 1]; }
        const MX Mk = (Phi.transpose() * Phi).ldlt().solve(Phi.transpose() * Psi);
        MX Tm = MX::Zero(V, V);
        for (int ch = 0; ch < 6; ++ch) for (int k = 0; k < K_; ++k) for (int j = 0; j < K_; ++j) Tm(6*k + ch, 6*j + ch) = Mk(k, j);
        mu = Tm * mu; S = Tm * S * Tm.transpose();
        g = make_gauss(mu, S); have_g = g.ok; if (have_g) Scur = S;
    }
    draw(have_g ? &g : nullptr); run(lw);
    int passes = 1;
    double ess = normalise_weights(lw, h_wi_);
    last_ess1_ = ess;
    const double ess_done = seq_done_ * N_, n_fit = 4.0 * V;
    if (seq_debug_ == 2 && !rb_checked_) {
        // finite-difference check of the kernel's gradient / Gauss-Newton Hessian for sample 0: roll out the same input
        // coefficients with the head/bias offset +-eps along each axis through the ordinary rollout (dx0 + bias offset)
        rb_checked_ = 1;
        const int nb = 2 * RB_H + 1;
        std::vector<double> th2(nb * D, 0.0), dx2(nb * 9, 0.0);
        const double eps[RB_H] = {1e-3,1e-3,1e-3, 1e-3,1e-3,1e-3, 1e-4,1e-4,1e-4, 1e-3,1e-3,1e-3, 1e-5,1e-5,1e-5};
        for (int s2 = 0; s2 < nb; ++s2) {
            for (int j = 0; j < C; ++j) th2[D*s2 + 6 + j] = hv[j];
            if (s2 == 0) continue;
            const int k = (s2 - 1) / 2; const double sg = (s2 % 2) ? 1.0 : -1.0;
            if (k < 9) dx2[9*s2 + k] = sg * eps[k]; else th2[D*s2 + (k - 9)] = sg * eps[k];
        }
        cudaMemcpy(d_theta_, th2.data(), nb * D * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dx0_, dx2.data(), nb * 9 * sizeof(double), cudaMemcpyHostToDevice);
        basis_kernel<<<1, BLOCK_SIZE>>>(d_states_, d_mu_, d_L_, d_prior_std_, d_phi_, 2, nb, T_, K_, 0, 0.0, d_theta_, d_logpq_, d_noise_, d_dv_);
        rollout_kernel<<<1, BLOCK_SIZE>>>(nb, T_, num_tags_ * num_anchors_, dt_, d_gravity_, d_anchor_, d_ranges_, d_stds_, d_accgyr_, d_U0_,
            d_noise_, d_x0_, nullptr, gamma_, w_uwb_, 0.0, 0.0, d_Ui_, d_cost_, d_cost_uwb_, d_cost_acc_, d_cost_gyr_, d_cost_last_,
            nullptr, d_dx0_, nullptr, true_dt_ ? d_dts_ : nullptr, nullptr, nullptr, nullptr);
        std::vector<double> cst(nb);
        cudaMemcpy(cst.data(), d_cost_, nb * sizeof(double), cudaMemcpyDeviceToHost);
        const double* o = &h_rb_[0];
        std::fprintf(stderr, "[rbcheck] J0 kernel %.6f rollout %.6f\n", o[RB_O_J0], cst[0]);
        for (int k = 0; k < RB_H; ++k) {
            const double gp = (cst[1 + 2*k] - cst[2 + 2*k]) / (2.0 * eps[k]);
            const double hp = (cst[1 + 2*k] - 2.0 * cst[0] + cst[2 + 2*k]) / (eps[k] * eps[k]);
            std::fprintf(stderr, "[rbcheck] dim %2d  grad kernel %+.5e  fd %+.5e | Hkk kernel(GN) %.5e  fd %.5e\n",
                         k, o[RB_O_G + k], gp, o[RB_O_H + rb_idx(k, k)], hp);
        }
    }
    // with verification the tempering trigger is the exact ESS of the first pass, not the Laplace one
    std::vector<double> lw_lap = lw, w_lap = h_wi_; const double ess_lap = ess;
    bool need_temper = ess < ess_min_ * N_;
    if (rb_verify_) {
        verify(lw, passes);
        ess = normalise_weights(lw, h_wi_);
        last_ess1_ = ess;
        need_temper = ess < ess_min_ * N_;
        lw.swap(lw_lap); h_wi_.swap(w_lap); ess = ess_lap;   // Laplace weights drive the proposal fit; exact ones kept in lw_lap
        if (need_temper) verified = false;
    }
    if (need_temper) {
        auto ess_at = [&](double b) {
            for (int i = 0; i < N_; ++i) tmp[i] = std::isfinite(h_cost_[i]) ? -gamma_ * b * h_cost_[i] + h_logpq_[i]
                                                                           : -std::numeric_limits<double>::infinity();
            return normalise_weights(tmp, wt);
        };
        double beta = 0.0;
        { double lo = 0.0, hi = 1.0;
          if (ess_at(1.0) >= n_fit) lo = 1.0;
          else for (int it = 0; it < 40; ++it) { const double mid = 0.5 * (lo + hi); if (ess_at(mid) >= n_fit) lo = mid; else hi = mid; }
          beta = lo; }
        for (int st = 0; st < seq_max_stages_; ++st) {
            const double ess_cur = ess_at(beta), target = std::max(0.5 * ess_cur, n_fit);
            double bnew = beta;
            if (ess_at(1.0) >= target) bnew = 1.0;
            else if (ess_cur >= target) {
                double lo = beta, hi = 1.0;
                for (int it = 0; it < 40; ++it) { const double mid = 0.5 * (lo + hi); if (ess_at(mid) >= target) lo = mid; else hi = mid; }
                bnew = lo;
            }
            const double n_eff = ess_at(bnew);
            VX mu; MX S; wfit(wt, mu, S);
            const double rho = n_eff / (n_eff + seq_shrink_k_ * V);
            const MX Snew = rho * seq_inflate_ * S + (1.0 - rho) * Scur;
            g = make_gauss(mu, Snew); have_g = g.ok; if (have_g) Scur = Snew;
            beta = bnew;
            draw(have_g ? &g : nullptr); run(lw); ++passes;
            ess = normalise_weights(lw, h_wi_);
            if (seq_debug_) std::fprintf(stderr, "[rb] stage %d beta %.3g fitESS %.0f -> ESS(1) %.1f\n", st, beta, n_eff, ess);
            if (beta >= 1.0 && ess >= ess_done) break;
        }
        ess = normalise_weights(lw, h_wi_);
    }
    last_ess2_ = ess; last_ess_ = ess; last_iters_ = passes; last_stages_ = passes - 1;
    // the input proposal for the next solve is fitted from the Laplace weights (self-consistent with the Laplace first
    // pass); the estimate uses the exact weights of the verification pass (section 17)
    std::vector<double> w_prop = h_wi_;
    double ess_prop = ess;
    if (rb_verify_) {
        if (!verified) verify(lw, passes);          // after tempering; otherwise the first-pass verification stands
        else { lw.swap(lw_lap); h_wi_.swap(w_lap); } // restore the exact weights of the first-pass verification
        ess = normalise_weights(lw, h_wi_);
        last_ess_ = ess; last_iters_ = passes;
    }
    if (verified) {   // branch evidence from the exact newest-epoch cost
        double m1 = -std::numeric_limits<double>::infinity(), m2 = m1;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { m1 = std::max(m1, lw[i]); m2 = std::max(m2, lw[i] + gamma_ * h_cost_newest_[i]); }
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { s1 += std::exp(lw[i] - m1); s2 += std::exp(lw[i] + gamma_ * h_cost_newest_[i] - m2); }
        last_logpred_ = (m1 + std::log(s1)) - (m2 + std::log(s2)); last_logpred1_ = last_logpred_;
    } else {   // newest-epoch predictive log-likelihood (branch evidence): marginal with and without that epoch
        double m1 = -std::numeric_limits<double>::infinity(), m2 = m1;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { m1 = std::max(m1, lw[i]); m2 = std::max(m2, lmm[i] + h_logpq_[i]); }
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < N_; ++i) if (std::isfinite(lw[i])) { s1 += std::exp(lw[i] - m1); s2 += std::exp(lmm[i] + h_logpq_[i] - m2); }
        last_logpred_ = (m1 + std::log(s1)) - (m2 + std::log(s2)); last_logpred1_ = last_logpred_;
    }
    last_cost_uwb_ = 0.0;

    // ---- new head: weighted mean of (state after the first step, moved by its sensitivity times the offset mean)
    int ib = 0; for (int i = 1; i < N_; ++i) if (h_wi_[i] > h_wi_[ib]) ib = i;
    auto x1of = [&](int i, Eigen::Vector3d& p, Eigen::Vector3d& v, Eigen::Matrix3d& R) {
        if (verified && rb_verify_ != 2) {   // exact state after the first step with the drawn offset applied
            const double* x1 = &h_x1_[(size_t)i * (DIM_X + 1)];
            for (int k = 0; k < 3; ++k) { p[k] = x1[k]; v[k] = x1[3 + k]; }
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R(r, c) = x1[6 + 3*r + c];
            return;
        }
        const double* o = &h_rb_[(size_t)i * RB_OUT];
        const double* x1 = o + RB_O_X1; const double* S1 = o + RB_O_S1;
        Eigen::Vector3d dp = Eigen::Vector3d::Zero(), dv = Eigen::Vector3d::Zero(), df = Eigen::Vector3d::Zero();
        for (int r = 0; r < 3; ++r) for (int c = 0; c < H; ++c) {
            const double m = mu_all[H*i + c];
            dp[r] += S1[r*RB_H + c] * m; dv[r] += S1[45 + r*RB_H + c] * m; df[r] += S1[90 + r*RB_H + c] * m;
        }
        for (int k = 0; k < 3; ++k) { p[k] = x1[k] + dp[k]; v[k] = x1[3 + k] + dv[k]; }
        Eigen::Matrix3d R1; for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R1(r, c) = x1[6 + 3*r + c];
        R = R1 * (df.norm() > 1e-12 ? Eigen::AngleAxisd(df.norm(), df.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
    };
    Eigen::Vector3d pb, vb; Eigen::Matrix3d Rref; x1of(ib, pb, vb, Rref);
    Eigen::Matrix<double, 9, 1> m = Eigen::Matrix<double, 9, 1>::Zero();
    for (int i = 0; i < N_; ++i) {
        if (h_wi_[i] <= 0.0) continue;
        Eigen::Vector3d p, v; Eigen::Matrix3d R; x1of(i, p, v, R);
        const Eigen::AngleAxisd aa(Eigen::Matrix3d(Rref.transpose() * R));
        const Eigen::Vector3d r = aa.angle() * aa.axis();
        Eigen::Matrix<double, 9, 1> y; y << p, v, r;
        m += h_wi_[i] * y;
    }
    for (int k = 0; k < 6; ++k) h_xn[k] = m[k];
    const Eigen::Vector3d mr = m.tail<3>();
    const Eigen::Matrix3d Rm = Rref * (mr.norm() > 1e-12 ? Eigen::AngleAxisd(mr.norm(), mr.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity());
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) h_xn[6 + 3*r + c] = Rm(r, c);
    for (int k = 0; k < 6; ++k) last_db_[k] = 0.0;
    for (int k = 0; k < 9; ++k) last_dmean_[k] = 0.0;
    for (int i = 0; i < N_; ++i) {
        const double* dsrc = (verified && rb_verify_ != 2) ? &dh[(size_t)i * H] : &mu_all[H*i];
        if (H == 15 && !bias_filt_) for (int k = 0; k < 6; ++k) last_db_[k] += h_wi_[i] * dsrc[9 + k];
        for (int k = 0; k < 9; ++k) last_dmean_[k] += h_wi_[i] * dsrc[k];
    }
    if (H == 15 && bias_filt_) {
        // filtering update of the bias: only the epoch that leaves the window moves it, so the overlapping
        // window is not re-applied every solve: d_f = -gamma (P^-1 + gamma A0)^-1 g0, averaged over the input samples
        MX Hs0(H, H);
        for (int i = 0; i < N_; ++i) {
            if (h_wi_[i] <= 0.0) continue;
            const double* o = &h_rb_[(size_t)i * RB_OUT];
            unpack(o + RB_O_H0, Hs0);
            const VX g0 = Eigen::Map<const VX>(o + RB_O_G0, RB_H).head(H);
            const VX df = -gamma_ * (PHinv + gamma_ * Hs0).ldlt().solve(g0);
            for (int k = 0; k < 6; ++k) last_db_[k] += h_wi_[i] * df[9 + k];
        }
    }

    // ---- next proposal for the input coefficients: weighted posterior (shifted at the start of the next solve)
    {
        VX mu; MX S; wfit(w_prop, mu, S);
        if (seq_shrink_carry_) { const double rho = ess_prop / (ess_prop + seq_shrink_k_ * V); S = rho * seq_inflate_ * S + (1.0 - rho) * Scur; }
        else S *= seq_inflate_;
        if (seq_newmass_) for (int j = 0; j < C; ++j) S(j, j) += sc[j] * sc[j] / T_;
        prop_mu_.assign(mu.data(), mu.data() + V);
        prop_S_.resize(V * V);
        for (int r = 0; r < V; ++r) for (int c = 0; c < V; ++c) prop_S_[V*r + c] = S(r, c);
        prop_valid_ = ess_prop >= n_fit ? 1 : 0;
    }

    // ---- arrival covariance of the new head/bias: filtered update with the leaving epoch, then one step
    {
        MX A0 = MX::Zero(H, H), F = MX::Zero(H, H), Hs(H, H);
        for (int i = 0; i < N_; ++i) {
            if (h_wi_[i] <= 0.0) continue;
            const double* o = &h_rb_[(size_t)i * RB_OUT];
            unpack(o + RB_O_H0, Hs); A0 += h_wi_[i] * Hs;
            const double* S1 = o + RB_O_S1;
            for (int r = 0; r < 3; ++r) for (int c = 0; c < H; ++c) {
                F(r, c) += h_wi_[i] * S1[r*RB_H + c]; F(3 + r, c) += h_wi_[i] * S1[45 + r*RB_H + c]; F(6 + r, c) += h_wi_[i] * S1[90 + r*RB_H + c];
            }
        }
        if (H == 15) for (int k = 0; k < 6; ++k) F(9 + k, 9 + k) = 1.0;
        const double h = dt_;
        MX QH = MX::Zero(H, H);
        Eigen::Matrix3d Sa = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) Sa(k, k) = head_q_[k] * head_q_[k];
        Sa = Rm * Sa * Rm.transpose();
        QH.block(0, 0, 3, 3) = Sa * h * h * h / 3.0; QH.block(0, 3, 3, 3) = Sa * h * h / 2.0;
        QH.block(3, 0, 3, 3) = Sa * h * h / 2.0;     QH.block(3, 3, 3, 3) = Sa * h;
        for (int k = 0; k < 3; ++k) QH(6 + k, 6 + k) = head_q_[3 + k] * head_q_[3 + k] * h;
        if (H == 15) for (int k = 0; k < 6; ++k) QH(9 + k, 9 + k) = head_qb_[k] * head_qb_[k] * h;
        const MX P0 = (PHinv + gamma_ * A0).ldlt().solve(MX::Identity(H, H));
        MX P1 = F * P0 * F.transpose() + QH;
        P1 = 0.5 * (P1 + P1.transpose());
        if (!P1.allFinite() || P1.ldlt().info() != Eigen::Success) P1 = PH + QH;
        last_PH_.resize(H * H);
        for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c) last_PH_[H*r + c] = P1(r, c);
        for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) { last_P1_[9*r + c] = P1(r, c); last_A_[9*r + c] = gamma_ * A0(r, c); }
        last_fit_r2_ = -1.0;
    }
    prof_[4] += msec(t_solve0, Clk::now()); prof_n_ += 1;
}

// ---------------------------------------------------------- integrity score ----
// D = min over rollouts that explain the newest epoch (raw UWB loss <= tau_z) of the
// physical compensation cost sum_t sum_j ((U_t,j - imu_t,j)/sigma_u,j)^2.
// Returns +inf when no rollout explains the epoch. frac_ok receives the fraction of
// rollouts that explain it.
double MPPISolver::integrityScore(double tau_z, const double* accgyr, double sigma_u_acc,
                                  double sigma_u_gyr, double* frac_ok, double* min_chi2) const {
    double best = std::numeric_limits<double>::infinity();
    double mc = std::numeric_limits<double>::infinity();
    int n_ok = 0;
    for (int i = 0; i < N_; ++i) {
        if (std::isfinite(h_cost_last_[i]) && h_cost_last_[i] < mc) mc = h_cost_last_[i];
        if (!std::isfinite(h_cost_last_[i]) || h_cost_last_[i] > tau_z) continue;
        ++n_ok;
        double j = 0.0;
        const double* U = &h_Ui_[(size_t)i * T_ * DIM_U];
        for (int t = 0; t < T_; ++t)
            for (int k = 0; k < DIM_U; ++k) {
                const double d = (U[t*DIM_U + k] - accgyr[t*6 + k]) / (k < 3 ? sigma_u_acc : sigma_u_gyr);
                j += d * d;
            }
        if (j < best) best = j;
    }
    if (frac_ok) *frac_ok = double(n_ok) / N_;
    if (min_chi2) *min_chi2 = mc;
    return best;
}
