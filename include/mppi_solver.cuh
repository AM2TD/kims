#pragma once

#include "dims.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

#define BLOCK_SIZE   256
#define CHUNK        34816
#define SEED         1234

void mppi_set_ctrl_mode(int mode, double sia, double sig);
void mppi_set_deltas(double uwb_delta, double ctrl_delta);
void mppi_set_floor_penalty(double weight);

class MPPISolver {
public:

    MPPISolver(int N, int T, double dt, double gamma,
               const double* h_anchor,
               const double* h_gravity,
               const double* h_sigma,
               const double* tag_offsets,
               int num_anchors,
               int num_tags);
    ~MPPISolver();

    void setDt(double dt);
    int  numAnchors() const { return num_anchors_; }
    int  numTags()    const { return num_tags_; }
    int  numRanges()  const { return num_tags_ * num_anchors_; }

    void solve(double* h_Uopt,
               double* h_xn,
               const double* h_x0,
               const double* h_U0,
               const double* h_ranges,
               const double* h_stds,
               const double* h_accgyr);

    double w_uwb_         {0.2};
    double w_acc_         {0.001};
    double w_gyr_         {0.015};

    double last_cost_uwb_ {0.0};
    double last_cost_acc_ {0.0};
    double last_cost_gyr_ {0.0};

private:

    int    N_, T_, num_anchors_, num_tags_;
    double dt_, gamma_;

    curandStatePhilox4_32_10_t* d_states_;
    double* d_anchor_;
    double* d_gravity_;

    cudaStream_t h2dS_, d2hS_, rngS_, rollS_;
    cudaEvent_t  rngDone_;

    double* d_x0_;
    double* d_U0_;
    double* d_ranges_;
    double* d_stds_;
    double* d_accgyr_;
    double* d_noise_;
    double* d_Ui_;
    double* d_cost_;
    double* d_cost_uwb_;
    double* d_cost_acc_;
    double* d_cost_gyr_;
    double* d_wi_;
    double* d_Uopt_;
    double* d_xn_;

    int len_U0_, len_ranges_, len_accgyr_;

    std::vector<double> h_cost_, h_Ui_, h_wi_;
};
