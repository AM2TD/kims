#include "mppi_solver.cuh"
#include "drone_dynamics.cuh"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

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
        v[j] = curand_normal_double(&local) * d_sigma_[j];
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
                               double gamma,
                               double w_uwb, double w_acc, double w_gyr,
                               double* Ui, double* cost,
                               double* cost_uwb, double* cost_acc, double* cost_gyr) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;

    double x[DIM_X];
    for (int i = 0; i < DIM_X; ++i) x[i] = x0[i];

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
        step_dynamics(x, ui, dt, g, x_next);

        double cu = w_uwb * uwb_step_cost(x_next, anchor,
                                          &ranges[t * num_rng], &stds[t * num_rng]);

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

__global__ void step_dyn_kernel(const double* x, const double* u,
                                double dt, const double* g, double* x_next) {
    step_dynamics(x, u, dt, g, x_next);
}

void mppi_set_deltas(double uwb_delta, double ctrl_delta) {
    cudaMemcpyToSymbol(d_uwb_delta_,  &uwb_delta,  sizeof(double));
    cudaMemcpyToSymbol(d_ctrl_delta_, &ctrl_delta, sizeof(double));
}

void mppi_set_floor_penalty(double weight) {
    cudaMemcpyToSymbol(d_floor_penalty_, &weight, sizeof(double));
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
    cudaMemcpyToSymbol(d_tag_offset_,  tag_offsets, 3 * num_tags * sizeof(double));
    cudaMemcpyToSymbol(d_num_anchors_, &num_anchors, sizeof(int));
    cudaMemcpyToSymbol(d_num_tags_,    &num_tags,    sizeof(int));

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
    cudaFree(d_cost_uwb_); cudaFree(d_cost_acc_); cudaFree(d_cost_gyr_);
    cudaFree(d_wi_);
    cudaFree(d_Uopt_);   cudaFree(d_xn_);
}

void MPPISolver::setDt(double dt) { dt_ = dt; }

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
            d_x0_, gamma_,
            w_uwb_, w_acc_, w_gyr_,
            d_Ui_    + batch_start * T_ * DIM_U,
            d_cost_  + batch_start,
            d_cost_uwb_ + batch_start,
            d_cost_acc_ + batch_start,
            d_cost_gyr_ + batch_start);

        batch_start += batch;
    }

    std::vector<double> h_cost_uwb(N_), h_cost_acc(N_), h_cost_gyr(N_);
    cudaMemcpyAsync(h_cost_.data(),    d_cost_,     N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_uwb.data(), d_cost_uwb_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_acc.data(), d_cost_acc_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_cost_gyr.data(), d_cost_gyr_, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_Ui_.data(),      d_Ui_,       N_ * T_ * DIM_U * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
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
