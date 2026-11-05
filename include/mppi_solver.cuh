#pragma once

#include "dims.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <limits>
#include <string>

#define BLOCK_SIZE   256
#define CHUNK        34816
#define SEED         1234

void mppi_set_ctrl_mode(int mode, double sia, double sig);
void mppi_set_proposal_sigma(const double* h_sigma_prop);
void mppi_set_deltas(double uwb_delta, double ctrl_delta);
void mppi_set_uwb_model(int model, const double* mix5);
void mppi_set_floor_penalty(double weight);
void mppi_set_floor_hard(int hard);

class MPPISolver {
public:

    MPPISolver(int N, int T, double dt, double gamma,
               const double* h_anchor,
               const double* h_gravity,
               const double* h_sigma,
               const double* tag_offsets,
               int num_anchors,
               int num_tags,
               unsigned long seed = SEED);
    ~MPPISolver();

    void setDt(double dt);
    void setTdoa(const double* anchorB, int n_slots);
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

    double integrityScore(double tau_z, const double* accgyr, double sigma_u_acc,
                          double sigma_u_gyr, double* frac_ok, double* min_chi2) const;

    double last_ess_      {0.0};

    void setBasis(int K, int iters, double ess_min);
    void setBasisPrior(const double* h_prior_std);
    int  basisDim() const { return K_ > 0 ? 6 + 6 * K_ + (dv_on_ ? 3 : 0) : 0; }
    int    dv_on_ {0};
    double dv_gain_ {0.0};
    void solveBasis(double* h_Uopt, double* h_xn,
                    const double* h_x0, const double* h_U0,
                    const double* h_ranges, const double* h_stds, const double* h_accgyr,
                    double* h_theta_mean, double* h_theta_var);
    int    prior_in_weight_ {0};
    double last_ess1_ {0.0}, last_ess2_ {0.0};
    double last_logpred_ {0.0};
    double last_logpred1_ {0.0};
    int    head_on_ {0};
    void   setHeadCov(const double* P9);
    void   setHeadSeed(unsigned long sd) { head_rng_.seed(sd); }
    double last_P1_[81] = {0};
    void   setHeadQ(const double* q6) { for (int k = 0; k < 6; ++k) head_q_[k] = q6[k]; }
    double last_A_[81] = {0};
    double last_dmean_[9] = {0};
    double last_fit_r2_ {-1.0};
    int    head_dim_ {9};
    double seq_alpha_ {0.9};
    double seq_inflate_ {1.2};
    int    seq_max_stages_ {20};
    int    seq_debug_ {0};
    int    seq_headfix_ {0};
    double seq_refl_ {0.0};
    double plane_c_[3] = {0, 0, 0}, plane_n_[3] = {0, 0, 1}; int plane_set_ {0};
    void   setPlane(const double* c, const double* n) { for (int k = 0; k < 3; ++k) { plane_c_[k] = c[k]; plane_n_[k] = n[k]; } plane_set_ = 1; }
    int    skip_ui_ {0};
    double *d_prop_par_ {nullptr}, *d_prop_x_ {nullptr}, *d_prop_lp_ {nullptr};
    double dbg_shift_[64] = {0}, dbg_ratio_[64] = {0}, dbg_wmin_ {0}, dbg_wmax_ {0}, dbg_wshift_ {0}; long dbg_n_ {0};
    double seq_done_ {0.25};
    double seq_shrink_k_ {1.0};
    int    seq_ewma_ {0};
    long   ewma_n_ {0};
    int    seq_newmass_ {1};
    int    seq_shrink_carry_ {1};
    void   setHeadCovN(const double* P, int H) { head_dim_ = H; headP_.assign(P, P + H * H); }
    void   setStepDts(const double* h_dts);
    void   setPhi(const double* h_phi);
    int    true_dt_ {0};
    void   setHeadQb(const double* qb6) { for (int k = 0; k < 6; ++k) head_qb_[k] = qb6[k]; }
    void   setProposal(const double* mu, const double* S, int valid) {
        const int V = (rb_on_ ? 0 : head_dim_) + 6 * K_; prop_valid_ = valid;
        if (valid) { prop_mu_.assign(mu, mu + V); prop_S_.assign(S, S + V * V); }
    }
    int    rb_on_ {0};
    int    rb_verify_ {0};
    double prof_[10] = {0};
    long   prof_n_ {0};
    int    rb_checked_ {0};
    int    bias_filt_ {0};
    double bias_gain_ {1.0};
    double head_gain_ {1.0};
    int    rollout_fp32_ {0};
    int    pass_n_ {0};
    std::string seq_dump_file_;
    long   seq_strip_from_ {0}, seq_strip_to_ {0}, seq_strip_step_ {1};
    int    rbias_on_ {0};
    double rb_head_ {0.0}, rb_q_ {0.0}, last_drb_ {0.0};
    double* d_rbias_ {nullptr};
    double last_xT_[DIM_X] = {0};
    int    flush_ {0};
    std::vector<double> flush_pos_;
    int    head_filt_ {0};
    double head_shift_[9] = {0};
    void   solveRB(double* h_xn, const double* h_x0, const double* h_U0,
                   const double* h_ranges, const double* h_stds, const double* h_accgyr);
    void   solveSeq(double* h_xn, const double* h_x0, const double* h_U0,
                    const double* h_ranges, const double* h_stds, const double* h_accgyr);
    std::vector<double> headP_, prop_mu_, prop_S_, last_PH_;
    int    prop_valid_ {0};
    double last_db_[6] = {0};
    int    last_stages_ {0};
    void saveRng();
    void restoreRng();
    int    last_iters_ {0};
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

    std::vector<double> h_cost_, h_Ui_, h_wi_, h_cost_last_;
    double* d_cost_last_;

    int    K_ {0}, D_ {0}, iters_ {1};
    double* d_dv_ {nullptr};
    double* d_cost_newest_ {nullptr};
    curandStatePhilox4_32_10_t* d_states_bak_ {nullptr};
    double *d_dx0_ {nullptr}, *d_x1_ {nullptr};
    std::vector<double> h_dx0_, h_dz_, h_x1_;
    double* d_xT_ {nullptr};
    std::vector<double> h_xT_;
    double* d_xall_ {nullptr};
    std::vector<double> h_xall_;
    double head_q_[6] = {0};
    double head_qb_[6] = {0};
    std::vector<double> h_prior_std_, h_phi_, h_phi_prev_;
    double* d_dts_ {nullptr};
    double* d_rb_ {nullptr};
    std::vector<double> h_rb_;
    double head_L_[81] = {0};
    std::mt19937_64 head_rng_ {12345};
    std::normal_distribution<double> head_nrm_ {0.0, 1.0};
    std::vector<double> h_cost_newest_;
    double ess_min_ {0.1};
    double *d_phi_ {nullptr}, *d_theta_ {nullptr}, *d_logpq_ {nullptr};
    double *d_mu_ {nullptr}, *d_L_ {nullptr}, *d_prior_std_ {nullptr};
    std::vector<double> h_theta_, h_logpq_;
    void basisPass(int use_L, double* h_lw);
};
